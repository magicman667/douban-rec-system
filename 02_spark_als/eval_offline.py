"""Evaluate ALS or four Top-10 rankers on a chronological test cohort."""

import argparse
import math
from pathlib import Path

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession, Window, functions as F

from feature_engineering import HIGH_CONTROVERSY, write_csv


TOP_K = 10
MODELS = ("B1", "B2", "B3", "Ours")


def make_cohort(train, test, profiles, max_users):
    seen = train.select("user_id", "movie_id").distinct()
    positives = (
        test.where(F.col("rating") >= 4).select("user_id", "movie_id").distinct()
        .join(seen, ["user_id", "movie_id"], "left_anti")
    )
    cohort = positives.groupBy("user_id").agg(F.count("*").alias("relevant_count")).join(
        profiles.select("user_id", "tolerance_tau", "genre_pref_json"), "user_id"
    )
    if max_users is not None:
        cohort = cohort.orderBy("user_id").limit(max_users)
    history = seen.groupBy("user_id").agg(F.collect_set("movie_id").alias("seen_ids"))
    return cohort.join(history, "user_id"), positives


def read_lists(spark, path):
    if path.endswith((".json", ".jsonl")):
        return spark.read.schema("user_id INT, movie_ids ARRAY<INT>").json(path)
    return spark.read.parquet(path).select("user_id", "movie_ids")


def baseline_candidates(model, cohort, films, include_popular=True):
    popularity = films.where(F.col("rating_count") > 0).orderBy(
        F.desc("bayesian_rating"), F.desc("rating_count"), "movie_id"
    )
    # Overfetch by at least the history length so filtering watched items preserves top 10.
    batches = cohort.withColumn(
        "history_bucket", F.ceil(F.log2(F.size("seen_ids") + 1)).cast("int")
    ).persist()
    b1, b2 = [], []
    try:
        for row in batches.select("history_bucket").distinct().collect():
            bucket = row.history_bucket
            k = TOP_K + 2 ** bucket - 1
            users = batches.where(F.col("history_bucket") == bucket).select("user_id")
            if include_popular:
                popular_ids = [row.movie_id for row in popularity.select("movie_id").take(k)]
                b1.append(users.withColumn("movie_ids", F.lit(popular_ids).cast("array<int>")))
            from_als = model.recommendForUserSubset(users, k)
            b2.append(from_als.select(
                "user_id", F.transform("recommendations", lambda rec: rec.movie_id).alias("movie_ids"),
                F.map_from_entries(F.transform(
                    "recommendations", lambda rec: F.struct(rec.movie_id, rec.rating)
                )).alias("movie_scores"),
            ))
        if not b2:
            raise ValueError("No eligible evaluation users.")
        popular = b1[0] if b1 else None
        als = b2[0]
        for rows in b1[1:]:
            popular = popular.unionByName(rows)
        for rows in b2[1:]:
            als = als.unionByName(rows)
        return popular, als
    finally:
        batches.unpersist()


def ranked_lists(cohort, candidates, films, name):
    lists = cohort.select("user_id", "seen_ids").join(candidates, "user_id", "left")
    if lists.where(F.col("movie_ids").isNull()).head(1):
        raise ValueError(f"{name}: missing list for at least one evaluation user")
    if lists.groupBy("user_id").count().where(F.col("count") != 1).head(1):
        raise ValueError(f"{name}: expected exactly one ranking per evaluation user")
    items = F.filter(F.col("movie_ids"), lambda item: item.isNotNull() & ~F.array_contains(
        F.col("seen_ids"), item
    ))
    ranked = lists.select(
        "user_id", F.posexplode(F.slice(F.array_distinct(items), 1, TOP_K)).alias("position", "movie_id")
    )
    if ranked.join(films.select("movie_id"), "movie_id", "left_anti").head(1):
        raise ValueError(f"{name}: ranking contains a movie ID outside the published catalog")
    return ranked.withColumn("rank", F.col("position") + 1).drop("position")


def genre_error(cohort, recommendations):
    preferences = cohort.select(
        "user_id", F.from_json("genre_pref_json", "map<string,double>").alias("preference")
    ).where(F.size("preference") > 0)
    target = preferences.select("user_id", F.explode("preference").alias("genre", "target"))
    actual = (
        recommendations.select("user_id", F.explode("genres").alias("genre"))
        .groupBy("user_id", "genre").count()
        .withColumn("actual", F.col("count") / F.sum("count").over(Window.partitionBy("user_id")))
        .select("user_id", "genre", "actual").join(preferences.select("user_id"), "user_id")
    )
    distance = target.join(actual, ["user_id", "genre"], "full").fillna(
        0.0, ["target", "actual"]
    ).groupBy("user_id").agg((F.sum(F.abs(F.col("target") - F.col("actual"))) / 2).alias("error"))
    # A missing list is assigned maximum error rather than half a normalized distribution.
    return distance.join(actual.select("user_id").distinct().withColumn("nonempty", F.lit(True)),
                         "user_id", "left").agg(
        F.avg(F.when(F.col("nonempty"), F.col("error")).otherwise(1.0)).alias("mean")
    ).first()["mean"]


def score(cohort, positives, ranked, films, catalog_count):
    recommendations = ranked.join(films.select("movie_id", "controversy_score", "genres"), "movie_id")
    relevant = positives.withColumn("relevant", F.lit(1))
    matched = recommendations.join(relevant, ["user_id", "movie_id"], "left").fillna(0, ["relevant"])
    user_metrics = matched.groupBy("user_id").agg(
        F.sum("relevant").alias("hits"),
        F.sum(F.col("relevant") / F.log2(F.col("rank") + 1)).alias("dcg"),
        F.sum((F.col("controversy_score") >= HIGH_CONTROVERSY).cast("long")).alias("landmines"),
    )
    ideal = sum(F.when(F.col("relevant_count") >= rank, 1.0 / math.log2(rank + 1))
                .otherwise(0.0) for rank in range(1, TOP_K + 1))
    summary = cohort.join(user_metrics, "user_id", "left").fillna(0, ["hits", "dcg", "landmines"]).agg(
        F.avg(F.col("dcg") / ideal).alias("ndcg"),
        F.avg(F.col("hits") / F.col("relevant_count")).alias("recall"),
        F.avg(F.when(F.col("tolerance_tau") < 0.28, F.col("landmines") / TOP_K)).alias("landmine"),
        F.count(F.when(F.col("tolerance_tau") < 0.28, 1)).alias("conservative"),
    ).first()
    coverage = ranked.select("movie_id").distinct().count() / catalog_count
    return summary, coverage, genre_error(cohort, recommendations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", help="Snapshot directory containing train_ratings and test_ratings")
    parser.add_argument("--hive-db", default="douban_db")
    parser.add_argument("--artifacts", required=True, help="Directory holding the group 2 Parquet and ALS outputs")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--b3", help="Member 3 ordered recommendations (Parquet or JSON Lines)")
    parser.add_argument("--ours", help="Member 3 ordered recommendations (Parquet or JSON Lines)")
    parser.add_argument("--export-cohort", help="Write eligible user IDs to a new Parquet path and exit")
    parser.add_argument("--als-only", action="store_true", help="Evaluate only B2")
    parser.add_argument("--als-recommendations", help="Write ranked B2 rows to a new Parquet path")
    parser.add_argument("--max-users", type=int, help="Deterministic small-sample run for development")
    args = parser.parse_args()
    if not args.export_cohort:
        if not args.analysis_dir:
            parser.error("Provide analysis-dir for scoring.")
        if not args.als_only and not (args.b3 and args.ours):
            parser.error("Provide b3 and ours for four-model scoring, or use als-only.")
    if args.max_users is not None and args.max_users < 1:
        parser.error("max-users must be positive.")
    spark = SparkSession.builder.appName("Member 2 offline Top-10 evaluation").enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        root = args.artifacts.rstrip("/")
        if args.input_parquet:
            source = args.input_parquet.rstrip("/")
            train = spark.read.parquet(f"{source}/train_ratings")
            test = spark.read.parquet(f"{source}/test_ratings")
        else:
            train = spark.table(f"{args.hive_db}.train_ratings")
            test = spark.table(f"{args.hive_db}.test_ratings")
        profiles = spark.read.parquet(f"{root}/user_profiles.parquet")
        cohort, positives = make_cohort(train, test, profiles, args.max_users)
        cohort = cohort.persist()
        n_users = cohort.count()
        print(f"Evaluation cohort: {n_users:,} eligible users")
        if n_users == 0:
            raise ValueError("No eligible evaluation users.")
        if args.export_cohort:
            cohort.select("user_id").write.parquet(args.export_cohort)
            return
        films = spark.read.parquet(f"{root}/movie_features.parquet")
        catalog_count = films.count()
        model = ALSModel.load(f"{root}/als_model")
        b1, b2 = baseline_candidates(model, cohort, films, include_popular=not args.als_only)
        if args.als_only:
            rankings = [("B2", b2)]
        else:
            rankings = list(zip(MODELS, [b1, b2, read_lists(spark, args.b3), read_lists(spark, args.ours)]))
        ranking_rows, landmine_rows, genre_rows = [], [], []
        for name, candidates in rankings:
            candidates = candidates.persist()
            try:
                ranked = ranked_lists(cohort, candidates, films, name).persist()
                try:
                    summary, coverage, error = score(cohort, positives, ranked, films, catalog_count)
                    if name == "B2" and args.als_recommendations:
                        ranked.join(candidates.select("user_id", "movie_scores"), "user_id").select(
                            "user_id", "movie_id", "rank",
                            F.element_at("movie_scores", F.col("movie_id")).alias("als_score"),
                        ).write.parquet(args.als_recommendations)
                finally:
                    ranked.unpersist()
            finally:
                candidates.unpersist()
            ranking_rows.append((name, summary.ndcg, summary.recall, coverage))
            landmine_rows.append((name, summary.landmine, summary.conservative))
            genre_rows.append((name, error))
        for rows, schema, filename in [
            (ranking_rows, "model STRING, ndcg_10 DOUBLE, recall_10 DOUBLE, coverage_10 DOUBLE",
             "model_metrics.csv"),
            (landmine_rows, "model STRING, landmine_10 DOUBLE, conservative_user_count LONG",
             "landmine_metrics.csv"),
            (genre_rows, "model STRING, genre_err_10 DOUBLE", "genre_error_metrics.csv"),
        ]:
            prefix = "als_" if args.als_only else ""
            write_csv(spark.createDataFrame(rows, schema), args.analysis_dir / (prefix + filename), ["model"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

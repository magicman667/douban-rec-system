"""Compute first-stage TDC-Rec features from the published training snapshot."""

import argparse
import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pyspark.sql import SparkSession, Window, functions as F


STAR_COLUMNS = [f"rating_{star}" for star in range(1, 6)]
UNKNOWN = "未知"
PRIOR = 20
HIGH_CONTROVERSY = 0.5


def movie_features(movies, train):
    counts = train.groupBy("movie_id").agg(
        F.count("*").alias("rating_count"),
        F.sum("rating").alias("rating_sum"),
        F.sum(F.col("rating") * F.col("rating")).alias("rating_square_sum"),
        *[F.sum((F.col("rating") == star).cast("long")).alias(column)
          for star, column in enumerate(STAR_COLUMNS, 1)],
    )
    observed = movies.join(counts, "movie_id", "left").fillna(
        0, ["rating_count", "rating_sum", "rating_square_sum", *STAR_COLUMNS]
    )
    genre = observed.groupBy("main_genre").agg(
        F.sum("rating_count").alias("genre_n"),
        F.sum("rating_sum").alias("genre_sum"),
        F.sum("rating_square_sum").alias("genre_square_sum"),
    )
    global_mean = train.agg(F.avg("rating").alias("global_mean"))
    if global_mean.first().global_mean is None:
        raise ValueError("The training ratings table is empty.")
    result = observed.join(genre, "main_genre").crossJoin(F.broadcast(global_mean))
    n = F.col("rating_count")
    genre_n = F.col("genre_n")
    mean = F.when(genre_n > 0, F.col("genre_sum") / genre_n).otherwise(F.col("global_mean"))
    genre_std = F.sqrt(F.greatest(F.lit(0.0), F.col("genre_square_sum") / genre_n - mean * mean))
    bayes = (F.col("rating_sum") + PRIOR * mean) / (n + PRIOR)
    film_mean = F.col("rating_sum") / n
    std_norm = F.sqrt(F.greatest(F.lit(0.0), F.col("rating_square_sum") / n - film_mean * film_mean)) / 2
    probabilities = [F.col(column) / n for column in STAR_COLUMNS]
    polarization = 4 * (probabilities[0] + probabilities[1]) * (probabilities[3] + probabilities[4])
    entropy = -sum(F.when(p > 0, p * F.log(p)).otherwise(0.0) for p in probabilities) / F.log(F.lit(5.0))
    controversy = n / (n + PRIOR) * (0.40 * std_norm + 0.35 * polarization + 0.25 * entropy)
    result = (
        result.withColumn("bayesian_rating", bayes)
        .withColumn("z_genre_score", F.when(genre_std > 0, (bayes - mean) / genre_std).otherwise(0.0))
        .withColumn("controversy_score", F.when(n > 0, controversy).otherwise(0.0))
        .withColumn("hist_5bins", F.array(*STAR_COLUMNS))
        .withColumn(
            "archetype_label",
            F.when((n == 0) | (F.col("z_genre_score") < 0), "低分避雷片")
            .when(F.col("controversy_score") >= HIGH_CONTROVERSY, "高分两极争议")
            .when(n < PRIOR, "小众冷门宝藏")
            .otherwise("大众稳妥神作"),
        )
    )
    return result.select(
        "movie_id", "title", "cover_url", "year", "genres", "main_genre",
        "regions", "tags", "storyline", "rating_count", "bayesian_rating",
        "z_genre_score", "controversy_score", "archetype_label", "hist_5bins",
    )


def user_profiles(train, films):
    watched = train.join(films.select("movie_id", "genres", "controversy_score"), "movie_id")
    is_positive = F.col("rating") >= 4
    is_divisive = F.col("controversy_score") >= HIGH_CONTROVERSY
    profile = watched.groupBy("user_id").agg(
        F.count("*").alias("rating_count"),
        F.avg(F.when(is_positive, F.col("controversy_score"))).alias("positive_controversy"),
        F.avg("rating").alias("mean_rating"),
        F.avg(F.when(is_divisive, F.col("rating"))).alias("divisive_rating"),
    )
    # Positive rating premium is expressed as a fraction of the four-star rating span.
    premium = F.greatest(F.lit(0.0), (F.col("divisive_rating") - F.col("mean_rating")) / 4)
    tau = F.when(
        F.col("positive_controversy").isNotNull(),
        F.least(F.lit(1.0), F.col("positive_controversy") + premium),
    ).otherwise(0.39)
    preference = (
        watched.where(is_positive)
        .select("user_id", F.explode("genres").alias("genre"))
        .groupBy("user_id", "genre").count()
        .withColumn("ratio", F.col("count") / F.sum("count").over(Window.partitionBy("user_id")))
        .groupBy("user_id")
        .agg(F.to_json(F.map_from_entries(F.sort_array(
            F.collect_list(F.struct("genre", "ratio"))
        ))).alias("genre_pref_json"))
    )
    return (
        profile.withColumn("tolerance_tau", tau)
        .withColumn("user_type_label", F.when(tau < 0.28, "稳妥防雷派")
                    .when(tau <= 0.50, "均衡观影派").otherwise("猎奇探索派"))
        .join(preference, "user_id", "left").fillna("{}", ["genre_pref_json"])
        .select("user_id", "rating_count", "tolerance_tau", "user_type_label", "genre_pref_json")
    )


def summarize(grouped, keys):
    return grouped.groupBy(*keys).agg(
        F.count("*").alias("movie_count"),
        F.avg("bayesian_rating").alias("avg_bayesian_rating"),
        F.avg("controversy_score").alias("avg_controversy"),
        *[F.sum(F.col("hist_5bins")[i]).alias(name) for i, name in enumerate(STAR_COLUMNS)],
        F.min(F.struct("controversy_score", (-F.col("bayesian_rating")).alias("rating"),
                       "movie_id", "title")).getField("title").alias("top_consensus_movie"),
        F.max(F.struct("controversy_score", F.col("bayesian_rating").alias("rating"),
                       "movie_id", "title")).getField("title").alias("top_polarizing_movie"),
    )


def write_csv(frame, path, order):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                            dir=path.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            writer = csv.writer(temp)
            writer.writerow(frame.columns)
            writer.writerows(frame.orderBy(*order).toLocalIterator())
            temp.close()
            temp_path.chmod(0o640)
            os.replace(temp_path, path)
        finally:
            temp.close()
            temp_path.unlink(missing_ok=True)


def overview(aggregate, key, region_codes=None):
    total = sum(F.col(column) for column in STAR_COLUMNS)
    iso = F.lit("")
    if region_codes:
        entries = [F.lit(value) for pair in region_codes.items() for value in pair]
        iso = F.coalesce(F.element_at(F.create_map(*entries), F.col(key)), iso)
    return aggregate.select(
        F.col(key).alias("name"), iso.alias("iso_code"), "movie_count",
        "avg_bayesian_rating", "avg_controversy",
        F.to_json(F.array(*[
            F.when(total > 0, F.col(column) / total).otherwise(0.0)
            for column in STAR_COLUMNS
        ])).alias("hist_ratio_5bins"), "top_consensus_movie", "top_polarizing_movie",
    )


def write_statistics(films, profiles, directory, region_codes):
    by_genre = films.withColumn("genre", F.explode("genres")).persist()
    by_region = films.withColumn("region", F.explode("regions")).persist()
    genre = summarize(by_genre, ["genre"]).persist()
    try:
        write_csv(genre.select("genre", *STAR_COLUMNS),
                  directory / "genre_rating_distribution.csv", ["genre"])
        write_csv(genre.select("genre", "movie_count", "avg_bayesian_rating", "avg_controversy"),
                  directory / "genre_controversy.csv", ["genre"])
        write_csv(overview(genre, "genre"), directory / "genre_stats.csv", ["name"])
        write_csv(overview(summarize(by_region, ["region"]), "region", region_codes),
                  directory / "region_stats.csv", ["name"])
        pairs = (by_genre.withColumn("region", F.explode("regions"))
                 .groupBy("genre", "region")
                 .agg(F.count("*").alias("movie_count"),
                      F.avg("controversy_score").alias("avg_controversy")))
        write_csv(pairs, directory / "genre_region_stats.csv", ["genre", "region"])
        types = profiles.groupBy("user_type_label").agg(F.count("*").alias("user_count"))
        total = types.agg(F.sum("user_count").alias("total"))
        write_csv(types.crossJoin(total).select(
            "user_type_label", "user_count", (F.col("user_count") / F.col("total")).alias("ratio")
        ), directory / "user_type_distribution.csv", ["user_type_label"])
    finally:
        genre.unpersist()
        by_region.unpersist()
        by_genre.unpersist()


def normalize_labels(column):
    values = F.array_distinct(F.filter(F.transform(F.col(column), F.trim),
                                        lambda value: value.isNotNull() & (value != "")))
    return F.when(F.size(values) > 0, values).otherwise(F.array(F.lit(UNKNOWN)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", help="Published snapshot directory containing movies and train_ratings")
    parser.add_argument("--hive-db", default="douban_db")
    parser.add_argument("--output", required=True, help="New version directory for this run")
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--region-codes", type=Path, help="Optional CSV with region,iso_code columns")
    args = parser.parse_args()
    region_codes = {}
    if args.region_codes:
        with args.region_codes.open(encoding="utf-8-sig", newline="") as source:
            region_codes = {row["region"]: row["iso_code"] for row in csv.DictReader(source)}
    spark = SparkSession.builder.appName("Member 2 full training features").enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        if args.input_parquet:
            root = args.input_parquet.rstrip("/")
            movies = spark.read.parquet(f"{root}/movies")
            train = spark.read.parquet(f"{root}/train_ratings")
        else:
            movies = spark.table(f"{args.hive_db}.movies")
            train = spark.table(f"{args.hive_db}.train_ratings")
        movies = (
            movies.withColumn("genres", normalize_labels("genres"))
            .withColumn("regions", normalize_labels("regions"))
            .withColumn("main_genre", F.when(F.length(F.trim("main_genre")) > 0, F.trim("main_genre"))
                        .otherwise(F.col("genres")[0]))
        )
        output = args.output.rstrip("/")
        films_path = f"{output}/movie_features.parquet"
        profiles_path = f"{output}/user_profiles.parquet"
        movie_features(movies, train).write.parquet(films_path)
        films = spark.read.parquet(films_path)
        user_profiles(train, films).write.parquet(profiles_path)
        profiles = spark.read.parquet(profiles_path)
        write_statistics(films, profiles, args.analysis_dir, region_codes)
        print(f"Movie features: {films.count():,}; user profiles: {profiles.count():,}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

"""Distributed Spark YARN jobs for Stage-1 ALS and Stage-2 MLlib GBTClassifier."""

import argparse
import json
import sys
import time
import types
from pathlib import Path
from packaging.version import Version

if "distutils" not in sys.modules:
    _d = types.ModuleType("distutils")
    _dv = types.ModuleType("distutils.version")
    _dv.LooseVersion = Version
    _d.version = _dv
    sys.modules["distutils"] = _d
    sys.modules["distutils.version"] = _dv

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, functions as F


FEATURE_COLS = [
    "als_dot",
    "als_cos",
    "user_als_norm",
    "movie_als_norm",
    "bayesian_rating",
    "z_genre_score",
    "controversy_score",
    "log_rating_count",
    "log_douban_votes",
    "movie_year",
    "user_mean_rating",
    "user_pos_rate",
    "user_rating_std",
    "user_tolerance_tau",
    "user_log_history",
    "genre_affinity",
    "genre_max_affinity",
    "rating_minus_user_mean",
    "excess_risk",
    "risk_margin",
    "c_times_tau",
]


def run_s1_als(spark: SparkSession, hdfs_dir: str) -> None:
    t0 = time.time()
    train = spark.read.parquet(f"{hdfs_dir}/train_s1.parquet").select("user_id", "movie_id", "rating")
    model = ALS(
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        rank=50,
        maxIter=10,
        regParam=0.1,
        numUserBlocks=10,
        numItemBlocks=10,
        coldStartStrategy="drop",
        seed=42,
    ).fit(train)
    factors = (
        model.userFactors.withColumn("factor_type", F.lit("user"))
        .unionByName(model.itemFactors.withColumn("factor_type", F.lit("movie")))
        .select("factor_type", "id", "features")
    )
    out = f"{hdfs_dir}/s1_als_factors.parquet"
    factors.write.mode("overwrite").partitionBy("factor_type").parquet(out)
    dt = time.time() - t0
    print(f"[spark s1_als] yarn_app_id={spark.sparkContext.applicationId}, sec={dt:.2f}")


def run_gbt(spark: SparkSession, hdfs_dir: str, out_json: Path) -> None:
    app_id = spark.sparkContext.applicationId
    train_df = spark.read.parquet(f"{hdfs_dir}/stage2_train.parquet")
    test_df = spark.read.parquet(f"{hdfs_dir}/stage2_test.parquet")

    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features", handleInvalid="keep")
    tr_vec = assembler.transform(train_df).select(F.col("y").cast("double").alias("y"), "features").cache()
    te_vec = assembler.transform(test_df).select("user_id", "movie_id", F.col("y").cast("double").alias("y"), "features")

    tr_count = tr_vec.count()
    t0 = time.time()
    gbt = GBTClassifier(
        labelCol="y",
        featuresCol="features",
        maxIter=30,
        maxDepth=5,
        stepSize=0.1,
        subsamplingRate=0.8,
        minInstancesPerNode=64,
        seed=42,
    )
    model = gbt.fit(tr_vec)
    train_sec = time.time() - t0
    tr_vec.unpersist()

    preds = (
        model.transform(te_vec)
        .withColumn("prob", vector_to_array("probability")[1])
        .select("user_id", "movie_id", F.col("y").cast("int").alias("y"), F.col("prob").cast("double").alias("prob"), "rawPrediction")
        .cache()
    )

    roc_auc = float(BinaryClassificationEvaluator(labelCol="y", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(preds))
    pr_auc = float(BinaryClassificationEvaluator(labelCol="y", rawPredictionCol="rawPrediction", metricName="areaUnderPR").evaluate(preds))
    stats = preds.agg(
        F.count("*").alias("n"),
        F.avg(
            -(
                F.col("y") * F.log(F.greatest(F.lit(1e-7), F.col("prob")))
                + (1 - F.col("y")) * F.log(F.greatest(F.lit(1e-7), 1.0 - F.col("prob")))
            )
        ).alias("logloss"),
        F.avg(F.when((F.col("prob") >= 0.5) == (F.col("y") == 1), 1.0).otherwise(0.0)).alias("acc"),
    ).first()

    preds.select("user_id", "movie_id", "y", "prob").write.mode("overwrite").parquet(f"{hdfs_dir}/spark_gbt_preds.parquet")
    preds.unpersist()

    importances = {col: float(val) for col, val in zip(FEATURE_COLS, model.featureImportances.toArray())}
    payload = {
        "yarn_app_id": app_id,
        "train_rows": int(tr_count),
        "test_rows": int(stats["n"]),
        "train_sec": round(train_sec, 2),
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "logloss": round(float(stats["logloss"]), 6),
        "accuracy": round(float(stats["acc"]), 6),
        "feature_importances": importances,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[spark gbt]", json.dumps(payload, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["s1_als", "gbt"], required=True)
    parser.add_argument("--hdfs-dir", default="hdfs:///tmp/ml_ranker")
    parser.add_argument("--out-json", type=Path, default=Path("/handoff/outputs/full-v1/ml_ranker/spark_gbt_metrics.json"))
    args = parser.parse_args()

    spark = SparkSession.builder.appName(f"Douban TDC-Rec ML [{args.mode}]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        if args.mode == "s1_als":
            run_s1_als(spark, args.hdfs_dir.rstrip("/"))
        else:
            run_gbt(spark, args.hdfs_dir.rstrip("/"), args.out_json)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

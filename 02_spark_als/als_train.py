"""Train Spark ALS and export the user and movie factors for member 3."""

import argparse

from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, functions as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", help="Published snapshot directory containing train_ratings")
    parser.add_argument("--hive-db", default="douban_db")
    parser.add_argument("--output", required=True, help="Directory for the model and factor table")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--reg-param", type=float, default=0.1)
    parser.add_argument("--blocks", type=int, default=10)
    args = parser.parse_args()
    if args.max_iter < 1 or args.reg_param < 0 or args.blocks < 1:
        parser.error("max-iter and blocks must be positive and reg-param must be nonnegative")

    spark = SparkSession.builder.appName("Member 2 rank-50 ALS").enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        if args.input_parquet:
            train = spark.read.parquet(f"{args.input_parquet.rstrip('/')}/train_ratings")
        else:
            train = spark.table(f"{args.hive_db}.train_ratings")
        train = train.select("user_id", "movie_id", "rating")
        if not train.head(1):
            raise ValueError("The training ratings table is empty.")
        model = ALS(
            userCol="user_id", itemCol="movie_id", ratingCol="rating", rank=50,
            maxIter=args.max_iter, regParam=args.reg_param, numUserBlocks=args.blocks,
            numItemBlocks=args.blocks, coldStartStrategy="drop", seed=42,
        ).fit(train)
        output = args.output.rstrip("/")
        model.save(f"{output}/als_model")
        factors = model.userFactors.withColumn("factor_type", F.lit("user")).unionByName(
            model.itemFactors.withColumn("factor_type", F.lit("movie"))
        ).select("factor_type", "id", "features")
        factor_path = f"{output}/als_factors.parquet"
        factors.write.partitionBy("factor_type").parquet(factor_path)
        stats = spark.read.parquet(factor_path).groupBy("factor_type").agg(
            F.count("*").alias("rows"), F.min(F.size("features")).alias("min_dim"),
            F.max(F.size("features")).alias("max_dim"),
        )
        for row in stats.collect():
            if row.min_dim != 50 or row.max_dim != 50:
                raise ValueError(f"Unexpected vector dimensions for {row.factor_type}.")
            print(f"ALS {row.factor_type}: {row.rows:,} vectors, 50 dimensions")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

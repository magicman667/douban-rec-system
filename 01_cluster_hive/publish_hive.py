"""Write typed Parquet to HDFS for the Hive external tables."""
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Douban full Hive tables").getOrCreate()
source, output = sys.argv[1:]
schemas = {
    "movies": "movie_id INT,title STRING,cover_url STRING,year INT,genres ARRAY<STRING>,regions ARRAY<STRING>,tags STRING,storyline STRING,main_genre STRING",
    "train_ratings": "rating_id BIGINT,user_id INT,movie_id INT,rating INT,rating_time STRING",
    "test_ratings": "rating_id BIGINT,user_id INT,movie_id INT,rating INT,rating_time STRING",
}
try:
    for table, schema in schemas.items():
        frame = spark.read.schema(schema).option("mode", "FAILFAST").json(f"{source}/{table}.jsonl")
        if "rating_time" in frame.columns:
            frame = frame.withColumn("rating_time", col("rating_time").cast("timestamp"))
        frame.write.mode("errorifexists").parquet(f"{output}/{table}")
finally:
    spark.stop()

"""Run identical genre/tag counting in Hadoop Streaming or Spark."""
import sys

mode = sys.argv[1]
if mode == "map":
    for line in sys.stdin:
        genre, tag = line.rstrip("\n").split("\t", 1)
        print(f"{genre}\t{tag}\t1")
elif mode == "reduce":
    from itertools import groupby
    for key, rows in groupby(sys.stdin, lambda row: row.rsplit("\t", 1)[0]):
        print(f"{key}\t{sum(int(row.rsplit(chr(9), 1)[1]) for row in rows)}")
elif mode == "spark":
    from pyspark.sql import SparkSession, functions as F
    spark = SparkSession.builder.appName("Douban genre tag WordCount").getOrCreate()
    try:
        paired = spark.read.text(sys.argv[2]).select(F.split("value", "\t", 2).alias("parts"))
        counts = paired.select(F.col("parts")[0].alias("genre"), F.col("parts")[1].alias("tag")).groupBy("genre","tag").count()
        counts.select(F.concat_ws("\t", "genre", "tag", F.col("count").cast("string")).alias("value")).write.text(sys.argv[3])
    finally:
        spark.stop()
else:
    raise SystemExit(f"Unknown mode: {mode}")

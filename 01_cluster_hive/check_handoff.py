"""Verify a member can aggregate the published training snapshot with Spark."""
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession

directory = Path(sys.argv[1])
expected = json.loads((directory / "counts.json").read_text())["train"]
spark = SparkSession.builder.appName("Verify full training handoff").getOrCreate()
try:
    counts = {
        row.rating: row["count"]
        for row in spark.read.parquet(str(directory / "parquet/train_ratings")).groupBy("rating").count().collect()
    }
    if not set(counts).issubset(range(1, 6)) or sum(counts.values()) != expected:
        raise ValueError("Published training counts differ from the cleaning manifest.")
    print("Spark training ratings:", json.dumps(counts, sort_keys=True))
finally:
    spark.stop()


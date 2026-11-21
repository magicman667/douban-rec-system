"""Detect one-star bursts in JSON microbatches and upsert MongoDB alerts."""
import argparse
import time
from pyspark.sql import SparkSession, functions as F
from mongo_store import write_stream_alert

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("checkpoint")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    spark = SparkSession.builder.appName("Douban review alerts").getOrCreate()
    events = spark.readStream.schema("movie_id INT, rating INT, event_time TIMESTAMP").json(args.source)
    counts = (events.where("rating = 1 AND movie_id IS NOT NULL AND event_time IS NOT NULL")
              .withWatermark("event_time", "2 minutes")
              .groupBy(F.window("event_time", "5 minutes", "1 minute"), "movie_id").count())

    def publish(batch, batch_id):
        for row in batch.where(F.col("count") >= args.threshold).toLocalIterator():
            write_stream_alert(row.movie_id, f"{row.window.start} 至 {row.window.end} 收到 {row['count']} 条一星评分")
        print(f"microbatch={batch_id} written_at={time.time():.3f}", flush=True)

    stream = counts.writeStream.outputMode("update").option("checkpointLocation", args.checkpoint).foreachBatch(publish)
    if args.once:
        stream = stream.trigger(availableNow=True)
    try:
        stream.start().awaitTermination()
    finally:
        spark.stop()

if __name__ == "__main__":
    main()

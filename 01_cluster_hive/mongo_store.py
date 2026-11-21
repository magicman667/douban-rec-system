"""MongoDB interface shared by recommendation and web modules."""
import argparse
import json
import os
from functools import lru_cache

from pymongo import MongoClient, UpdateOne

@lru_cache(maxsize=1)
def collection():
    client = MongoClient(os.environ.get("DOUBAN_MONGO_URI", "mongodb://127.0.0.1:27017"),
                         serverSelectionTimeoutMS=5000)
    return client.douban_db.movie_reviews

def get_movie_reviews(movie_id):
    return collection().find_one({"movie_id": int(movie_id)}, {"_id": 0})

def save_ai_summary(movie_id, summary):
    collection().update_one({"movie_id": int(movie_id)}, {"$set": {"ai_summary": summary}}, upsert=True)

def write_stream_alert(movie_id, msg):
    collection().update_one({"movie_id": int(movie_id)},
                            {"$set": {"stream_alert": True, "recent_drop_msg": msg}}, upsert=True)

def get_active_alerts(limit=20):
    if limit <= 0:
        return []
    return list(collection().find({"stream_alert": True}, {"_id": 0}).sort("movie_id", 1).limit(limit))

def import_reviews(path):
    target = collection()
    target.create_index("movie_id", unique=True)
    pending = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pending.append(UpdateOne({"movie_id": row["movie_id"]}, {
                "$set": row, "$setOnInsert": {"ai_summary": "", "stream_alert": False, "recent_drop_msg": ""}
            }, upsert=True))
            if len(pending) >= 500:
                target.bulk_write(pending, ordered=False)
                pending.clear()
    if pending:
        target.bulk_write(pending, ordered=False)
    print(target.count_documents({"pos_comments": {"$exists": True}}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reviews")
    import_reviews(parser.parse_args().reviews)

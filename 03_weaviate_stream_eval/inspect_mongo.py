from __future__ import annotations

import json
import os

from pymongo import MongoClient


def main() -> None:
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"), serverSelectionTimeoutMS=5000)
    try:
        report = {}
        for database_name in client.list_database_names():
            if database_name in {"admin", "config", "local"}:
                continue
            database = client[database_name]
            report[database_name] = {}
            for collection in database.list_collection_names():
                sample = database[collection].find_one({}, {"_id": 0}) or {}
                report[database_name][collection] = {
                    "count": database[collection].estimated_document_count(),
                    "sample_movie_id": sample.get("movie_id"),
                    "sample_fields": sorted(sample.keys()),
                }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()

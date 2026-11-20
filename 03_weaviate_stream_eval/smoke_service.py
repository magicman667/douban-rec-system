from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds

import rec_service


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("als_factors", type=Path)
    parser.add_argument("user_profiles", type=Path)
    parser.add_argument("eval_users", type=Path)
    parser.add_argument("--detail-movie-id", type=int)
    args = parser.parse_args()
    first_batch = next(ds.dataset(args.eval_users, format="parquet").to_batches(columns=["user_id"], batch_size=1))
    user_id = int(first_batch.column("user_id")[0].as_py())
    weaviate_client, mongo_client = rec_service.configure_live(args.als_factors, args.user_profiles)
    try:
        recommendations = rec_service.get_user_recommendations(user_id, top_k=10)
        scenes = rec_service.search_by_scene("雨天独自看的温暖科幻片", avoid_controversy=True, top_k=5)
        detail_movie_id = args.detail_movie_id or recommendations["recommendations"][0]["movie_id"]
        detail = rec_service.get_controversy_detail_and_twins(detail_movie_id)
        report = {
            "user_id": user_id,
            "recommendation_count": recommendations["count"],
            "recommendation_ids": [item["movie_id"] for item in recommendations["recommendations"]],
            "scene_count": len(scenes),
            "scene_ids": [item["movie_id"] for item in scenes],
            "detail_movie_id": detail["movie"]["movie_id"],
            "safe_twin_count": len(detail["safe_twins"]),
            "mongo_positive_comments": len(detail["pos_comments"]),
            "mongo_negative_comments": len(detail["neg_comments"]),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["recommendation_count"] != 10 or report["scene_count"] != 5:
            raise SystemExit(1)
    finally:
        weaviate_client.close()
        mongo_client.close()


if __name__ == "__main__":
    main()

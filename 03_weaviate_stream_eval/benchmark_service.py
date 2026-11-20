from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pyarrow.dataset as ds

import rec_service


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "requests": len(values),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "max_ms": round(max(ordered), 3),
    }


def timed(callable_):
    started = time.perf_counter()
    result = callable_()
    return result, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("als_factors", type=Path)
    parser.add_argument("user_profiles", type=Path)
    parser.add_argument("eval_users", type=Path)
    parser.add_argument("--users", type=int, default=50)
    args = parser.parse_args()
    user_ids = sorted(int(value) for value in ds.dataset(args.eval_users, format="parquet").to_table(columns=["user_id"]).column("user_id").to_pylist())[: args.users]
    weaviate_client, mongo_client = rec_service.configure_live(args.als_factors, args.user_profiles)
    recommendation_times: list[float] = []
    scene_times: list[float] = []
    detail_times: list[float] = []
    movie_ids: list[int] = []
    scenes = ["雨天温暖科幻", "周末轻松喜剧", "低分歧悬疑", "家庭共同观看", "高分动画冒险"]
    try:
        rec_service.get_user_recommendations(user_ids[0], top_k=10)
        for user_id in user_ids:
            result, elapsed = timed(lambda uid=user_id: rec_service.get_user_recommendations(uid, top_k=10))
            recommendation_times.append(elapsed)
            if result["recommendations"]:
                movie_ids.append(result["recommendations"][0]["movie_id"])
        for index in range(20):
            _, elapsed = timed(lambda query=scenes[index % len(scenes)]: rec_service.search_by_scene(query, True, 10))
            scene_times.append(elapsed)
        for movie_id in movie_ids[:20]:
            _, elapsed = timed(lambda mid=movie_id: rec_service.get_controversy_detail_and_twins(mid))
            detail_times.append(elapsed)
        print(json.dumps({"recommendations": summarize(recommendation_times), "scene_search": summarize(scene_times), "detail_and_twins": summarize(detail_times)}, ensure_ascii=False, indent=2))
    finally:
        weaviate_client.close()
        mongo_client.close()


if __name__ == "__main__":
    main()

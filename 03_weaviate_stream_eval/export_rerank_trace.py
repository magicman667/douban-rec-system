"""Export genuine before/after TDC-Rec reranking traces from live Weaviate and member-2 profiles."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Candidate, Movie, UserProfile, fuse_rankings, rrf_score  # noqa: E402


MOVIE_FIELDS = (
    "movie_id title main_genre genres regions cover_url year tags storyline "
    "rating_count bayesian_rating z_genre_score controversy_score archetype_label"
)


def _graphql(url: str, query: str) -> dict:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/graphql",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if body.get("errors"):
        raise RuntimeError(f"Weaviate GraphQL error: {body['errors']}")
    return body["data"]


def fetch_als_candidates(url: str, collection: str, user_vec: list[float], limit: int = 200) -> list[tuple[Movie, float]]:
    vec_literal = "[" + ",".join(f"{float(v):.8f}" for v in user_vec) + "]"
    query = (
        "{ Get { "
        f'{collection}(nearVector: {{ vector: {vec_literal}, targetVectors: ["collaborative"] }}, limit: {limit}) '
        f"{{ {MOVIE_FIELDS} _additional {{ distance }} }} "
        "} }"
    )
    rows = _graphql(url, query)["Get"][collection] or []
    result: list[tuple[Movie, float]] = []
    for row in rows:
        dist = (row.get("_additional") or {}).get("distance")
        score = 1.0 - float(dist) if dist is not None else 0.0
        result.append((Movie.from_mapping(row), score))
    return result


def fetch_movies_by_ids(url: str, collection: str, movie_ids: list[int]) -> dict[int, Movie]:
    if not movie_ids:
        return {}
    operands = ", ".join(
        f'{{ path: ["movie_id"], operator: Equal, valueInt: {int(mid)} }}' for mid in movie_ids
    )
    query = (
        "{ Get { "
        f"{collection}(where: {{ operator: Or, operands: [{operands}] }}, limit: {len(movie_ids)}) "
        f"{{ {MOVIE_FIELDS} }} "
        "} }"
    )
    rows = _graphql(url, query)["Get"][collection] or []
    return {int(row["movie_id"]): Movie.from_mapping(row) for row in rows}


def load_b3_subset(b3_path: Path, target_users: set[int]) -> dict[int, list[int]]:
    found: dict[int, list[int]] = {}
    with b3_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            uid = int(item["user_id"])
            if uid in target_users:
                found[uid] = [int(mid) for mid in item["movie_ids"]]
                if len(found) == len(target_users):
                    break
    return found


def build_trace_for_user(
    profile: UserProfile,
    als: list[tuple[Movie, float]],
    vector: list[tuple[Movie, float]],
    top_k: int = 10,
) -> list[dict]:
    merged: dict[int, Candidate] = {}
    for rank, (movie, score) in enumerate(als, start=1):
        merged[movie.movie_id] = Candidate(movie=movie, als_rank=rank, als_score=float(score))
    for rank, (movie, score) in enumerate(vector, start=1):
        cand = merged.setdefault(movie.movie_id, Candidate(movie=movie))
        cand.vector_rank = rank
        cand.vector_score = float(score)

    before_list = sorted(
        merged.values(),
        key=lambda c: (rrf_score(c.als_rank, c.vector_rank), c.movie.rating_count, -c.movie.movie_id),
        reverse=True,
    )
    before_rank = {c.movie.movie_id: idx for idx, c in enumerate(before_list, start=1)}

    after_list = fuse_rankings(als, vector, profile, top_k=len(merged))
    after_rank = {c.movie.movie_id: idx for idx, c in enumerate(after_list, start=1)}

    focus_ids = [c.movie.movie_id for c in after_list[:top_k]]
    for c in before_list[:top_k]:
        if c.movie.movie_id not in focus_ids:
            focus_ids.append(c.movie.movie_id)

    rows = []
    for mid in focus_ids:
        cand = merged[mid]
        rows.append(
            {
                "user_id": profile.user_id,
                "movie_id": mid,
                "rank_before": before_rank[mid],
                "rank_after": after_rank.get(mid, len(after_list) + 1),
                "controversy_score": round(cand.movie.controversy_score, 6),
                "tolerance_tau": round(profile.tolerance_tau, 6),
            }
        )
    rows.sort(key=lambda r: (r["rank_after"], r["rank_before"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        default="/srv/douban-data/outputs/member2/full-v1-20261119-205432/artifacts",
    )
    parser.add_argument(
        "--b3",
        type=Path,
        default=Path("/srv/douban-data/outputs/member3/full-v1-20261120/eval/b3.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/03_eval/rerank_trace.csv"),
    )
    parser.add_argument("--per-archetype", type=int, default=5)
    args = parser.parse_args()

    weaviate_url = f"http://{os.getenv('WEAVIATE_HOST', '127.0.0.1')}:{os.getenv('WEAVIATE_HTTP_PORT', '18080')}"
    collection = os.getenv("WEAVIATE_COLLECTION", "DoubanMovie")

    spark = (
        SparkSession.builder.appName("Export TDC-Rec rerank trace")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        root = args.artifacts.rstrip("/")
        profiles_df = spark.read.parquet(f"{root}/user_profiles.parquet")
        eval_users_df = spark.read.parquet(f"{root}/eval_users")
        factors_df = spark.read.parquet(f"{root}/als_factors.parquet").where(F.col("factor_type") == "user")

        joined = (
            profiles_df.join(eval_users_df, "user_id")
            .join(factors_df.select(F.col("id").alias("user_id"), "features"), "user_id")
            .where(F.col("rating_count") >= 20)
        )
        selected_rows = []
        for label in ("稳妥防雷派", "均衡观影派", "猎奇探索派"):
            subset = (
                joined.where(F.col("user_type_label") == label)
                .orderBy(F.desc("rating_count"), "user_id")
                .limit(args.per_archetype)
                .collect()
            )
            selected_rows.extend(subset)
    finally:
        spark.stop()

    profiles = {int(r["user_id"]): UserProfile.from_mapping(r.asDict()) for r in selected_rows}
    user_vecs = {int(r["user_id"]): [float(x) for x in r["features"]] for r in selected_rows}
    b3_lists = load_b3_subset(args.b3, set(profiles.keys()))

    all_trace_rows: list[dict] = []
    for uid, profile in profiles.items():
        als = fetch_als_candidates(weaviate_url, collection, user_vecs[uid], limit=200)
        b3_ids = b3_lists.get(uid, [])
        movie_map = fetch_movies_by_ids(weaviate_url, collection, b3_ids)
        vector = [(movie_map[mid], 1.0 - idx * 0.01) for idx, mid in enumerate(b3_ids) if mid in movie_map]
        all_trace_rows.extend(build_trace_for_user(profile, als, vector, top_k=10))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["user_id", "movie_id", "rank_before", "rank_after", "controversy_score", "tolerance_tau"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_trace_rows)
    print(f"Wrote {len(all_trace_rows)} rerank trace rows for {len(profiles)} users to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path

import pyarrow.dataset as ds

import rec_service
from core import fuse_rankings


def completed_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export B3 and Ours together with resume support")
    parser.add_argument("als_factors", type=Path)
    parser.add_argument("user_profiles", type=Path)
    parser.add_argument("eval_users", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b3_path = args.output_dir / "b3.jsonl"
    ours_path = args.output_dir / "ours.jsonl"
    b3_done = completed_lines(b3_path)
    ours_done = completed_lines(ours_path)
    if b3_done != ours_done:
        raise RuntimeError(f"断点行数不一致: B3={b3_done}, Ours={ours_done}")
    table = ds.dataset(args.eval_users, format="parquet").to_table(columns=["user_id"])
    user_ids = sorted(int(value) for value in table.column("user_id").to_pylist())
    if args.limit:
        user_ids = user_ids[: args.limit]
    user_ids = user_ids[b3_done:]
    weaviate_client, mongo_client = rec_service.configure_live(args.als_factors, args.user_profiles)
    repo = rec_service._repo()
    started = time.time()

    def calculate(user_id: int) -> tuple[int, list[int], list[int]]:
        profile = repo.get_profile(user_id)
        if profile is None:
            return user_id, [], []
        semantic = repo.semantic_for_user(user_id, args.candidate_limit)
        als = repo.als_candidates(user_id, args.candidate_limit)
        b3_ids = [movie.movie_id for movie, _ in semantic[: args.top_k]]
        ours_ids = [item.movie.movie_id for item in fuse_rankings(als, semantic, profile, top_k=args.top_k)]
        return user_id, b3_ids, ours_ids

    try:
        with b3_path.open("a", encoding="utf-8", buffering=1) as b3_handle, ours_path.open("a", encoding="utf-8", buffering=1) as ours_handle:
            executor = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
            results = executor.map(calculate, user_ids) if executor else map(calculate, user_ids)
            for index, (user_id, b3_ids, ours_ids) in enumerate(results, start=b3_done + 1):
                b3_handle.write(json.dumps({"user_id": user_id, "movie_ids": b3_ids}, ensure_ascii=False) + "\n")
                ours_handle.write(json.dumps({"user_id": user_id, "movie_ids": ours_ids}, ensure_ascii=False) + "\n")
                if index % 100 == 0:
                    elapsed = time.time() - started
                    print(json.dumps({"completed": index, "elapsed_sec": round(elapsed, 2), "users_per_sec": round((index - b3_done) / elapsed, 3)}, ensure_ascii=False), flush=True)
            if executor:
                executor.shutdown()
    finally:
        weaviate_client.close()
        mongo_client.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds


def load_export(path: Path, valid_movies: set[int]) -> tuple[set[int], dict]:
    users: set[int] = set()
    rows = empty = invalid_movies = too_long = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            user_id = int(item["user_id"])
            movie_ids = [int(value) for value in item["movie_ids"]]
            rows += 1
            users.add(user_id)
            empty += int(not movie_ids)
            too_long += int(len(movie_ids) > 10)
            invalid_movies += sum(movie_id not in valid_movies for movie_id in movie_ids)
    return users, {"rows": rows, "unique_users": len(users), "empty_lists": empty, "too_long_lists": too_long, "invalid_movie_ids": invalid_movies}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_users", type=Path)
    parser.add_argument("movie_features", type=Path)
    parser.add_argument("b3", type=Path)
    parser.add_argument("ours", type=Path)
    args = parser.parse_args()
    expected_users = set(int(value) for value in ds.dataset(args.eval_users, format="parquet").to_table(columns=["user_id"]).column("user_id").to_pylist())
    valid_movies = set(int(value) for value in ds.dataset(args.movie_features, format="parquet").to_table(columns=["movie_id"]).column("movie_id").to_pylist())
    b3_users, b3_report = load_export(args.b3, valid_movies)
    ours_users, ours_report = load_export(args.ours, valid_movies)
    valid = b3_users == expected_users == ours_users and not any((b3_report["too_long_lists"], b3_report["invalid_movie_ids"], ours_report["too_long_lists"], ours_report["invalid_movie_ids"]))
    report = {"expected_users": len(expected_users), "b3": b3_report, "ours": ours_report, "valid": valid}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

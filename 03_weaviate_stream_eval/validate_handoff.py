from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds


REQUIRED_MOVIES = {
    "movie_id", "title", "cover_url", "year", "genres", "main_genre", "regions",
    "tags", "storyline", "rating_count", "bayesian_rating", "z_genre_score",
    "controversy_score", "archetype_label", "hist_5bins",
}
REQUIRED_PROFILES = {"user_id", "rating_count", "tolerance_tau", "user_type_label", "genre_pref_json"}
REQUIRED_FACTORS = {"factor_type", "id", "features"}


def inspect(name: str, path: Path, required: set[str]) -> tuple[ds.Dataset, dict]:
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    columns = set(dataset.schema.names)
    missing = sorted(required - columns)
    return dataset, {
        "path": str(path),
        "rows": dataset.count_rows(),
        "columns": dataset.schema.names,
        "missing_required_columns": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate member-2 artifacts for member 3")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    artifacts = args.root / "artifacts"
    movies, movie_report = inspect("movies", artifacts / "movie_features.parquet", REQUIRED_MOVIES)
    profiles, profile_report = inspect("profiles", artifacts / "user_profiles.parquet", REQUIRED_PROFILES)
    factors, factor_report = inspect("factors", artifacts / "als_factors.parquet", REQUIRED_FACTORS)
    eval_users, eval_report = inspect("eval_users", artifacts / "eval_users", {"user_id"})
    factor_counts: dict[str, int] = {}
    dims: set[int] = set()
    for batch in factors.to_batches(columns=["factor_type", "features"], batch_size=8192):
        factor_types = batch.column("factor_type").to_pylist()
        features = batch.column("features").to_pylist()
        for kind in factor_types:
            factor_counts[str(kind)] = factor_counts.get(str(kind), 0) + 1
        dims.update(len(value) for value in features if value is not None)
    report = {
        "movie_features": movie_report,
        "user_profiles": profile_report,
        "als_factors": factor_report,
        "eval_users": eval_report,
        "factor_counts": factor_counts,
        "factor_dimensions": sorted(dims),
        "valid": not any(item["missing_required_columns"] for item in [movie_report, profile_report, factor_report, eval_report]) and sorted(dims) == [50],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

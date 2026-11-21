from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow.dataset as ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    dataset = ds.dataset(args.dataset, format="parquet", exclude_invalid_files=True)
    row_count = dataset.count_rows()
    movie_ids: set[int] = set()
    dimensions: set[int] = set()
    min_norm = float("inf")
    max_norm = 0.0
    checked_vectors = 0
    for batch in dataset.to_batches(columns=["movie_id", "semantic_vector"], batch_size=4096):
        ids = batch.column("movie_id").to_pylist()
        vectors = batch.column("semantic_vector").to_pylist()
        movie_ids.update(int(value) for value in ids)
        for vector in vectors[: min(16, len(vectors))]:
            dimensions.add(len(vector))
            norm = math.sqrt(sum(float(value) ** 2 for value in vector))
            min_norm = min(min_norm, norm)
            max_norm = max(max_norm, norm)
            checked_vectors += 1
    report = {
        "rows": row_count,
        "unique_movie_ids": len(movie_ids),
        "dimensions": sorted(dimensions),
        "sampled_vectors": checked_vectors,
        "sample_norm_min": round(min_norm, 6),
        "sample_norm_max": round(max_norm, 6),
        "valid": row_count == len(movie_ids) and dimensions == {512} and 0.999 <= min_norm <= max_norm <= 1.001,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

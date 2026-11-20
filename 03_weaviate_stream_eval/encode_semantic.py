from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer


COLUMNS = ["movie_id", "title", "genres", "main_genre", "regions", "tags", "storyline"]


def text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item is not None)
    return str(value)


def semantic_text(row: dict) -> str:
    return " ".join(text_value(row.get(name)).strip() for name in COLUMNS[1:] if text_value(row.get(name)).strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable BGE encoding for all movies")
    parser.add_argument("movie_features", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rows-per-part", type=int, default=2048)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(args.movie_features, format="parquet")
    model = SentenceTransformer(args.model, device="cpu")
    started = time.time()
    total = completed = 0
    for part_index, batch in enumerate(dataset.to_batches(columns=COLUMNS, batch_size=args.rows_per_part)):
        part = args.output_dir / f"part-{part_index:05d}.parquet"
        total += batch.num_rows
        if part.exists():
            completed += batch.num_rows
            continue
        rows = batch.to_pylist()
        vectors = model.encode(
            [semantic_text(row) for row in rows],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector_array = pa.array(vectors.tolist(), type=pa.list_(pa.float32(), 512))
        table = pa.table({"movie_id": batch.column("movie_id"), "semantic_vector": vector_array})
        temporary = part.with_suffix(".tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(part)
        completed += batch.num_rows
        elapsed = time.time() - started
        print(json.dumps({"part": part.name, "completed": completed, "total_seen": total, "elapsed_sec": round(elapsed, 1)}, ensure_ascii=False), flush=True)
    manifest = {
        "model": args.model,
        "rows": completed,
        "dimensions": 512,
        "normalized": True,
        "parts": len(list(args.output_dir.glob("part-*.parquet"))),
        "elapsed_sec": round(time.time() - started, 3),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

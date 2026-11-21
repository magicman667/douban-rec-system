from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("movie_features", type=Path)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    columns = ["title", "genres", "regions", "tags", "storyline"]
    frame = pd.read_parquet(args.movie_features, columns=columns).head(args.limit)
    texts = frame.fillna("").astype(str).agg(" ".join, axis=1).tolist()
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")
    vectors = model.encode(texts, normalize_embeddings=True)
    report = {
        "shape": list(vectors.shape),
        "squared_norms": [round(float(np.dot(vector, vector)), 6) for vector in vectors],
        "titles": frame["title"].tolist(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if vectors.shape != (args.limit, 512):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

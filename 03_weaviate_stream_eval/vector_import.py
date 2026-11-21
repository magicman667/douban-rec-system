from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import Movie, validate_vector


def load_movies(path: Path) -> list[Movie]:
    import pandas as pd

    frame = pd.read_parquet(path)
    required = {"movie_id", "title", "main_genre", "controversy_score", "z_genre_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"movie_features 缺少字段: {sorted(missing)}")
    return [Movie.from_mapping(row) for row in frame.to_dict(orient="records")]


def load_movie_factors(path: Path) -> dict[int, list[float]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    required = {"factor_type", "id", "features"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"als_factors 缺少字段: {sorted(missing)}")
    rows = frame[frame["factor_type"] == "movie"]
    return {int(row["id"]): validate_vector(row["features"], 50, "ALS") for row in rows.to_dict(orient="records")}


def encode_semantics(movies: list[Movie], model_name: str, batch_size: int) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [movie.semantic_text() for movie in movies],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [validate_vector(vector, 512, "BGE") for vector in vectors]


def load_semantic_vectors(path: Path, movies: list[Movie]) -> list[list[float]]:
    import pyarrow.dataset as ds

    table = ds.dataset(path, format="parquet", exclude_invalid_files=True).to_table(columns=["movie_id", "semantic_vector"])
    mapping = {
        int(movie_id): validate_vector(vector, 512, "BGE")
        for movie_id, vector in zip(table.column("movie_id").to_pylist(), table.column("semantic_vector").to_pylist())
    }
    missing = [movie.movie_id for movie in movies if movie.movie_id not in mapping]
    if missing:
        raise ValueError(f"语义向量缺少 {len(missing)} 部电影，示例: {missing[:5]}")
    return [mapping[movie.movie_id] for movie in movies]


def import_weaviate(
    movies: list[Movie],
    semantic_vectors: list[list[float]],
    als_vectors: dict[int, list[float]],
    *,
    http_host: str,
    http_port: int,
    grpc_port: int,
    collection_name: str,
) -> dict:
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property

    client = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=False,
        grpc_host=http_host,
        grpc_port=grpc_port,
        grpc_secure=False,
    )
    try:
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
        collection = client.collections.create(
            name=collection_name,
            vector_config=[
                Configure.Vectors.self_provided(name="semantic"),
                Configure.Vectors.self_provided(name="collaborative"),
            ],
            properties=[
                Property(name="movie_id", data_type=DataType.INT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="main_genre", data_type=DataType.TEXT),
                Property(name="genres", data_type=DataType.TEXT_ARRAY),
                Property(name="regions", data_type=DataType.TEXT_ARRAY),
                Property(name="cover_url", data_type=DataType.TEXT),
                Property(name="year", data_type=DataType.INT),
                Property(name="bayesian_rating", data_type=DataType.NUMBER),
                Property(name="z_genre_score", data_type=DataType.NUMBER),
                Property(name="controversy_score", data_type=DataType.NUMBER),
                Property(name="archetype_label", data_type=DataType.TEXT),
                Property(name="tags", data_type=DataType.TEXT),
                Property(name="storyline", data_type=DataType.TEXT),
                Property(name="rating_count", data_type=DataType.INT),
                Property(name="hist_5bins", data_type=DataType.INT_ARRAY),
            ],
        )
        imported = collaborative = 0
        with collection.batch.fixed_size(batch_size=128) as batch:
            for movie, semantic in zip(movies, semantic_vectors):
                vectors = {"semantic": semantic}
                if movie.movie_id in als_vectors:
                    vectors["collaborative"] = als_vectors[movie.movie_id]
                    collaborative += 1
                props = movie.storage_dict()
                if props["year"] is None:
                    props.pop("year")
                batch.add_object(properties=props, vector=vectors)
                imported += 1
        failed = len(collection.batch.failed_objects)
        if failed:
            raise RuntimeError(f"Weaviate 批量导入失败 {failed} 条")
        return {"movies_imported": imported, "collaborative_vectors": collaborative, "semantic_vectors": imported}
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="将二号产物导入 Weaviate 双命名向量集合")
    parser.add_argument("--movie-features", type=Path, required=True)
    parser.add_argument("--als-factors", type=Path, required=True)
    parser.add_argument("--semantic-vectors", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--grpc-port", type=int, default=50051)
    parser.add_argument("--collection", default="DoubanMovie")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="仅用于小批量烟雾测试")
    args = parser.parse_args()
    movies = load_movies(args.movie_features)
    if args.limit:
        movies = movies[: args.limit]
    als = load_movie_factors(args.als_factors)
    semantic = load_semantic_vectors(args.semantic_vectors, movies) if args.semantic_vectors else encode_semantics(movies, args.model, args.batch_size)
    result = import_weaviate(movies, semantic, als, http_host=args.host, http_port=args.http_port, grpc_port=args.grpc_port, collection_name=args.collection)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


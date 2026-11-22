from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

from core import Movie, UserProfile, validate_vector


class WeaviateRepository:
    """Weaviate v4 adapter backed by member-2 profiles and user factors."""

    def __init__(self, client, collection_name: str, profiles, user_vectors, encoder: Callable[[str], list[float]]):
        self.client = client
        self.collection = client.collections.get(collection_name)
        self.profiles = profiles
        self.user_vectors = user_vectors
        self.encoder = encoder
        self._semantic_cache: dict[tuple[str, int, int], list[tuple[Movie, float]]] = {}
        self._semantic_cache_lock = threading.Lock()

    @staticmethod
    def _movie(obj) -> Movie:
        return Movie.from_mapping(obj.properties)

    @staticmethod
    def _score(obj) -> float:
        distance = getattr(obj.metadata, "distance", None)
        return 1.0 - float(distance) if distance is not None else 0.0

    def get_movie(self, movie_id: int) -> Movie | None:
        from weaviate.classes.query import Filter
        response = self.collection.query.fetch_objects(filters=Filter.by_property("movie_id").equal(int(movie_id)), limit=1)
        return self._movie(response.objects[0]) if response.objects else None

    def get_profile(self, user_id: int) -> UserProfile | None:
        return self.profiles.get(int(user_id))

    def _near(self, vector, target: str, limit: int, min_ratings: int = 0):
        from weaviate.classes.query import Filter, MetadataQuery
        filters = Filter.by_property("rating_count").greater_or_equal(int(min_ratings)) if min_ratings and int(min_ratings) > 0 else None
        response = self.collection.query.near_vector(
            near_vector=vector,
            target_vector=target,
            filters=filters,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        return [(self._movie(obj), self._score(obj)) for obj in response.objects]

    def als_candidates(self, user_id: int, limit: int, min_ratings: int = 0):
        vector = self.user_vectors.get(int(user_id))
        if not vector:
            return []
        return self._near(vector, "collaborative", limit, min_ratings=min_ratings) if min_ratings > 0 else self._near(vector, "collaborative", limit)

    def semantic_candidates(self, query_text: str, limit: int, min_ratings: int = 0):
        key = (query_text, limit, int(min_ratings))
        with self._semantic_cache_lock:
            cached = self._semantic_cache.get(key)
        if cached is not None:
            return cached
        vec = self.encoder(query_text)
        rows = self._near(vec, "semantic", limit, min_ratings=min_ratings) if min_ratings > 0 else self._near(vec, "semantic", limit)
        with self._semantic_cache_lock:
            if len(self._semantic_cache) >= 512:
                self._semantic_cache.pop(next(iter(self._semantic_cache)))
            self._semantic_cache[key] = rows
        return rows

    def semantic_for_user(self, user_id: int, limit: int, min_ratings: int = 0):
        profile = self.get_profile(user_id)
        if not profile:
            return []
        query = " ".join(key for key, _ in sorted(profile.genre_preferences.items(), key=lambda item: item[1], reverse=True)[:5])
        return self.semantic_candidates(query or "电影", limit, min_ratings=min_ratings)

    def low_risk_twins(self, movie: Movie, limit: int, min_ratings: int = 50):
        from weaviate.classes.query import Filter, MetadataQuery
        vec = self.encoder(movie.semantic_text())
        if min_ratings and int(min_ratings) > 0:
            filtered = (
                Filter.by_property("main_genre").equal(movie.main_genre)
                & Filter.by_property("movie_id").not_equal(movie.movie_id)
                & Filter.by_property("controversy_score").less_than(0.45)
                & Filter.by_property("rating_count").greater_or_equal(int(min_ratings))
            )
            response = self.collection.query.near_vector(near_vector=vec, target_vector="semantic", filters=filtered, limit=limit, return_metadata=MetadataQuery(distance=True))
            if len(response.objects) >= limit:
                return [(self._movie(obj), self._score(obj)) for obj in response.objects]
        fallback_filter = (
            Filter.by_property("main_genre").equal(movie.main_genre)
            & Filter.by_property("movie_id").not_equal(movie.movie_id)
            & Filter.by_property("controversy_score").less_than(0.50)
        )
        response = self.collection.query.near_vector(near_vector=vec, target_vector="semantic", filters=fallback_filter, limit=limit, return_metadata=MetadataQuery(distance=True))
        return [(self._movie(obj), self._score(obj)) for obj in response.objects]


class MongoReviewStore:
    def __init__(self, collection):
        self.collection = collection

    def get_reviews(self, movie_id: int) -> dict:
        return self.collection.find_one({"movie_id": int(movie_id)}, {"_id": 0}) or {}

    def save_summary(self, movie_id: int, summary: str) -> None:
        self.collection.update_one({"movie_id": int(movie_id)}, {"$set": {"ai_summary": summary}}, upsert=True)


def build_live_backend(factors_path: Path, user_profiles_path: Path):
    import pandas as pd
    import weaviate
    from pymongo import MongoClient
    from sentence_transformers import SentenceTransformer

    profiles_frame = pd.read_parquet(user_profiles_path)
    profiles = {int(row["user_id"]): UserProfile.from_mapping(row) for row in profiles_frame.to_dict(orient="records")}
    user_rows = pd.read_parquet(factors_path, filters=[("factor_type", "=", "user")])
    user_vectors = {int(row["id"]): validate_vector(row["features"], 50, "ALS") for row in user_rows.to_dict(orient="records")}
    model = SentenceTransformer(os.getenv("BGE_MODEL", "BAAI/bge-small-zh-v1.5"))

    def encode(text: str) -> list[float]:
        return validate_vector(model.encode(text, normalize_embeddings=True), 512, "BGE")

    host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
    client = weaviate.connect_to_custom(http_host=host, http_port=int(os.getenv("WEAVIATE_HTTP_PORT", "18080")), http_secure=False, grpc_host=host, grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50052")), grpc_secure=False)
    repository = WeaviateRepository(client, os.getenv("WEAVIATE_COLLECTION", "DoubanMovie"), profiles, user_vectors, encode)
    mongo = MongoClient(os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017"))
    review_store = MongoReviewStore(mongo[os.getenv("MONGO_DATABASE", "douban_db")][os.getenv("MONGO_COLLECTION", "movie_reviews")])
    return repository, review_store, client, mongo


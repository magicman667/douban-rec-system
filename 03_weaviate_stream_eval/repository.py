from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from core import Movie, UserProfile


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class RetrievalRepository(Protocol):
    def get_movie(self, movie_id: int) -> Movie | None: ...
    def get_profile(self, user_id: int) -> UserProfile | None: ...
    def als_candidates(self, user_id: int, limit: int) -> list[tuple[Movie, float]]: ...
    def semantic_candidates(self, query_text: str, limit: int) -> list[tuple[Movie, float]]: ...
    def semantic_for_user(self, user_id: int, limit: int) -> list[tuple[Movie, float]]: ...
    def low_risk_twins(self, movie: Movie, limit: int) -> list[tuple[Movie, float]]: ...


@dataclass
class MemoryRepository:
    movies: dict[int, Movie]
    profiles: dict[int, UserProfile]
    semantic_vectors: dict[int, list[float]]
    als_movie_vectors: dict[int, list[float]]
    als_user_vectors: dict[int, list[float]]
    seen: dict[int, set[int]]

    def get_movie(self, movie_id: int) -> Movie | None:
        return self.movies.get(int(movie_id))

    def get_profile(self, user_id: int) -> UserProfile | None:
        return self.profiles.get(int(user_id))

    def _rank(self, query: Sequence[float], vectors: dict[int, list[float]], limit: int, excluded: set[int] | None = None):
        excluded = excluded or set()
        rows = [(self.movies[mid], cosine(query, vector)) for mid, vector in vectors.items() if mid in self.movies and mid not in excluded]
        return sorted(rows, key=lambda row: (row[1], row[0].rating_count), reverse=True)[:limit]

    def als_candidates(self, user_id: int, limit: int) -> list[tuple[Movie, float]]:
        vector = self.als_user_vectors.get(int(user_id), [])
        return self._rank(vector, self.als_movie_vectors, limit, self.seen.get(int(user_id), set())) if vector else []

    def semantic_candidates(self, query_text: str, limit: int) -> list[tuple[Movie, float]]:
        terms = set(query_text.lower().split())
        rows = []
        for movie in self.movies.values():
            text = movie.semantic_text().lower()
            score = sum(1 for term in terms if term in text) / max(1, len(terms))
            rows.append((movie, float(score)))
        return sorted(rows, key=lambda row: (row[1], row[0].rating_count), reverse=True)[:limit]

    def semantic_for_user(self, user_id: int, limit: int) -> list[tuple[Movie, float]]:
        profile = self.get_profile(user_id)
        if not profile:
            return []
        query = " ".join(name for name, _ in sorted(profile.genre_preferences.items(), key=lambda item: item[1], reverse=True)[:3])
        return self.semantic_candidates(query, limit)

    def low_risk_twins(self, movie: Movie, limit: int) -> list[tuple[Movie, float]]:
        rows = []
        for other in self.movies.values():
            if other.movie_id == movie.movie_id or other.main_genre != movie.main_genre:
                continue
            shared = len(set(movie.genres) & set(other.genres))
            score = shared + max(0.0, 1.0 - other.controversy_score)
            rows.append((other, score))
        return sorted(rows, key=lambda row: (row[1], row[0].bayesian_rating), reverse=True)[:limit]


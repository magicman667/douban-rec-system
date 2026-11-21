from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def parse_genre_preferences(raw: object) -> dict[str, float]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        try:
            result[str(key)] = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return result


@dataclass(slots=True)
class Movie:
    movie_id: int
    title: str
    main_genre: str = "未知"
    genres: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    cover_url: str = ""
    year: int | None = None
    tags: str = ""
    storyline: str = ""
    rating_count: int = 0
    bayesian_rating: float = 0.0
    z_genre_score: float = 0.0
    controversy_score: float = 0.0
    archetype_label: str = ""
    hist_5bins: list[int] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "Movie":
        def list_value(name: str) -> list:
            value = row.get(name)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = []
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    value = parsed if isinstance(parsed, list) else [value]
                except json.JSONDecodeError:
                    value = [item.strip() for item in value.split("/") if item.strip()]
            return list(value)

        year = row.get("year")
        try:
            normalized_year = int(year) if year not in (None, "") and math.isfinite(float(year)) else None
        except (TypeError, ValueError):
            normalized_year = None
        return cls(
            movie_id=int(row["movie_id"]),
            title=str(row.get("title") or ""),
            main_genre=str(row.get("main_genre") or "未知"),
            genres=[str(v) for v in list_value("genres")],
            regions=[str(v) for v in list_value("regions")],
            cover_url=str(row.get("cover_url") or ""),
            year=normalized_year,
            tags=str(row.get("tags") or ""),
            storyline=str(row.get("storyline") or ""),
            rating_count=int(row.get("rating_count") or 0),
            bayesian_rating=float(row.get("bayesian_rating") or 0.0),
            z_genre_score=float(row.get("z_genre_score") or 0.0),
            controversy_score=min(1.0, max(0.0, float(row.get("controversy_score") or 0.0))),
            archetype_label=str(row.get("archetype_label") or ""),
            hist_5bins=[int(v) for v in list_value("hist_5bins")],
        )

    def semantic_text(self) -> str:
        parts = [self.title, self.main_genre, " ".join(self.genres), " ".join(self.regions), self.tags, self.storyline]
        return " ".join(part.strip() for part in parts if part and part.strip())

    def public_dict(self) -> dict:
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "cover_url": self.cover_url,
            "year": self.year,
            "genres": self.genres,
            "main_genre": self.main_genre,
            "regions": self.regions,
            "bayesian_rating": self.bayesian_rating,
            "z_genre_score": self.z_genre_score,
            "controversy_score": self.controversy_score,
            "archetype_label": self.archetype_label,
        }

    def storage_dict(self) -> dict:
        payload = self.public_dict()
        payload.update(
            {
                "tags": self.tags,
                "storyline": self.storyline,
                "rating_count": self.rating_count,
                "hist_5bins": self.hist_5bins,
            }
        )
        return payload


@dataclass(slots=True)
class UserProfile:
    user_id: int
    rating_count: int
    tolerance_tau: float
    user_type_label: str
    genre_preferences: dict[str, float]

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "UserProfile":
        return cls(
            user_id=int(row["user_id"]),
            rating_count=int(row.get("rating_count") or 0),
            tolerance_tau=min(1.0, max(0.0, float(row.get("tolerance_tau") or 0.0))),
            user_type_label=str(row.get("user_type_label") or ""),
            genre_preferences=parse_genre_preferences(row.get("genre_pref_json")),
        )


@dataclass(slots=True)
class Candidate:
    movie: Movie
    als_rank: int | None = None
    vector_rank: int | None = None
    als_score: float | None = None
    vector_score: float | None = None
    utility: float = 0.0
    rrf_score: float = 0.0


def rrf_score(als_rank: int | None, vector_rank: int | None, *, k: int = 60) -> float:
    terms = []
    if als_rank is not None:
        terms.append(0.5 / (k + als_rank))
    if vector_rank is not None:
        terms.append(0.5 / (k + vector_rank))
    return float(k + 1) * sum(terms)


def utility_score(candidate: Candidate, profile: UserProfile) -> float:
    movie = candidate.movie
    base = rrf_score(candidate.als_rank, candidate.vector_rank)
    preference = profile.genre_preferences.get(movie.main_genre, 0.0)
    excess_risk = max(0.0, movie.controversy_score - profile.tolerance_tau)
    exploration = 0.25 * movie.controversy_score if profile.tolerance_tau > 0.5 else 0.0
    candidate.rrf_score = base
    candidate.utility = (
        base
        + 0.20 * _sigmoid(movie.z_genre_score)
        + 0.15 * preference
        - 1.2 * excess_risk**1.5
        + exploration
    )
    return candidate.utility


def fuse_rankings(
    als: Sequence[tuple[Movie, float]],
    vector: Sequence[tuple[Movie, float]],
    profile: UserProfile,
    *,
    top_k: int = 10,
    max_per_genre: int = 4,
    genre_filter: str | None = None,
) -> list[Candidate]:
    merged: dict[int, Candidate] = {}
    for rank, (movie, score) in enumerate(als, start=1):
        merged[movie.movie_id] = Candidate(movie=movie, als_rank=rank, als_score=float(score))
    for rank, (movie, score) in enumerate(vector, start=1):
        item = merged.setdefault(movie.movie_id, Candidate(movie=movie))
        item.vector_rank = rank
        item.vector_score = float(score)
    candidates = [item for item in merged.values() if not genre_filter or genre_filter in item.movie.genres or item.movie.main_genre == genre_filter]
    candidates.sort(key=lambda item: (utility_score(item, profile), item.movie.rating_count, -item.movie.movie_id), reverse=True)
    selected: list[Candidate] = []
    genre_counts: dict[str, int] = {}
    for item in candidates:
        genre = item.movie.main_genre or "未知"
        if genre_counts.get(genre, 0) >= max_per_genre:
            continue
        selected.append(item)
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def candidate_to_dict(item: Candidate, rank: int) -> dict:
    payload = item.movie.public_dict()
    payload.update(
        {
            "rank": rank,
            "utility_score": round(item.utility, 8),
            "rrf_score": round(item.rrf_score, 8),
            "als_rank": item.als_rank,
            "vector_rank": item.vector_rank,
            "als_score": item.als_score,
            "vector_score": item.vector_score,
        }
    )
    return payload


def validate_vector(vector: Iterable[float], expected_dim: int, label: str) -> list[float]:
    values = [float(value) for value in vector]
    if len(values) != expected_dim:
        raise ValueError(f"{label} 向量维度应为 {expected_dim}，实际为 {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} 向量包含非有限数")
    return values


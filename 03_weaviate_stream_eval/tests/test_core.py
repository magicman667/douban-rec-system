import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))

from core import Candidate, Movie, UserProfile, fuse_rankings, rrf_score, utility_score, validate_vector


def movie(mid, genre="剧情", controversy=0.1, z=0.0):
    return Movie(mid, f"m{mid}", main_genre=genre, genres=[genre], controversy_score=controversy, z_genre_score=z, rating_count=100)


def profile(tau):
    return UserProfile(1, 10, tau, "x", {"剧情": 0.8, "科幻": 0.2})


def test_rrf_matches_frozen_formula():
    assert round(rrf_score(1, 1), 8) == 1.0
    assert rrf_score(1, None) == 0.5


def test_risk_penalty_changes_user_order():
    safe = Candidate(movie(1, controversy=0.1), als_rank=2, vector_rank=2)
    risky = Candidate(movie(2, controversy=0.8), als_rank=1, vector_rank=1)
    assert utility_score(safe, profile(0.2)) > utility_score(risky, profile(0.2))
    safe2 = Candidate(movie(1, controversy=0.1), als_rank=2, vector_rank=2)
    risky2 = Candidate(movie(2, controversy=0.8), als_rank=1, vector_rank=1)
    assert utility_score(risky2, profile(0.8)) > utility_score(safe2, profile(0.8))


def test_fusion_deduplicates_and_limits_genre():
    films = [movie(i, "剧情") for i in range(1, 7)] + [movie(7, "科幻")]
    rows = [(film, 1.0 / index) for index, film in enumerate(films, 1)]
    selected = fuse_rankings(rows, list(reversed(rows)), profile(0.4), top_k=6, max_per_genre=4)
    assert len({item.movie.movie_id for item in selected}) == len(selected)
    assert sum(item.movie.main_genre == "剧情" for item in selected) <= 4


def test_vector_dimension_validation():
    assert len(validate_vector([0.0] * 50, 50, "ALS")) == 50
    try:
        validate_vector([0.0] * 49, 50, "ALS")
    except ValueError as exc:
        assert "50" in str(exc)
    else:
        raise AssertionError("dimension error was not raised")


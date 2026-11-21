import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))

from core import Movie, UserProfile
from rec_service import configure, get_controversy_detail_and_twins, get_user_recommendations, search_by_scene
from repository import MemoryRepository


def build_repo():
    movies = {
        1: Movie(1, "星际孤航", "科幻", ["科幻"], controversy_score=0.2, rating_count=100),
        2: Movie(2, "争议宇宙", "科幻", ["科幻"], controversy_score=0.8, rating_count=200),
        3: Movie(3, "安静星空", "科幻", ["科幻"], controversy_score=0.1, rating_count=50),
        4: Movie(4, "家庭旅程", "剧情", ["剧情"], controversy_score=0.1, rating_count=80),
    }
    profiles = {9: UserProfile(9, 20, 0.2, "稳妥防雷派", {"科幻": 0.8, "剧情": 0.2})}
    return MemoryRepository(movies, profiles, {i: [float(i), 1.0] for i in movies}, {i: [float(i), 1.0] for i in movies}, {9: [1.0, 1.0]}, {9: set()})


def test_three_contract_functions():
    configure(build_repo())
    result = get_user_recommendations(9, top_k=3)
    assert result["user"]["user_id"] == 9
    assert len(result["recommendations"]) == 3
    search = search_by_scene("科幻", avoid_controversy=True, top_k=3)
    assert search and all(item["controversy_score"] < 0.5 for item in search)
    detail = get_controversy_detail_and_twins(2)
    assert detail["movie"]["movie_id"] == 2
    assert len(detail["safe_twins"]) == 2


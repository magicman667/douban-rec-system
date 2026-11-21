from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Protocol

from core import UserProfile, candidate_to_dict, fuse_rankings
from repository import RetrievalRepository


class ReviewStore(Protocol):
    def get_reviews(self, movie_id: int) -> dict: ...
    def save_summary(self, movie_id: int, summary: str) -> None: ...


_repository: RetrievalRepository | None = None
_reviews: ReviewStore | None = None


def configure(repository: RetrievalRepository, review_store: ReviewStore | None = None) -> None:
    global _repository, _reviews
    _repository = repository
    _reviews = review_store


def configure_live(als_factors_path: str | Path, user_profiles_path: str | Path):
    """Connect real Weaviate/MongoDB and return handles the caller must close."""
    from live_backend import build_live_backend
    repository, review_store, weaviate_client, mongo_client = build_live_backend(Path(als_factors_path), Path(user_profiles_path))
    configure(repository, review_store)
    return weaviate_client, mongo_client


def _repo() -> RetrievalRepository:
    if _repository is None:
        raise RuntimeError("推荐服务尚未配置：请先调用 configure(repository, review_store)")
    return _repository


def get_user_recommendations(user_id: int, genre_filter: str | None = None, top_k: int = 10) -> dict:
    repo = _repo()
    profile = repo.get_profile(int(user_id))
    if profile is None:
        raise KeyError(f"未找到用户画像: {user_id}")
    pool = max(100, top_k * 20)
    als = repo.als_candidates(int(user_id), pool)
    vector = repo.semantic_for_user(int(user_id), pool)
    ranked = fuse_rankings(als, vector, profile, top_k=top_k, genre_filter=genre_filter)
    return {
        "user": {
            "user_id": profile.user_id,
            "rating_count": profile.rating_count,
            "tolerance_tau": profile.tolerance_tau,
            "user_type_label": profile.user_type_label,
            "genre_preferences": profile.genre_preferences,
        },
        "recommendations": [candidate_to_dict(item, rank) for rank, item in enumerate(ranked, 1)],
        "count": len(ranked),
    }


def search_by_scene(query_text: str, avoid_controversy: bool = True, top_k: int = 10) -> list[dict]:
    if not query_text or not query_text.strip():
        raise ValueError("query_text 不能为空")
    rows = _repo().semantic_candidates(query_text.strip(), max(50, top_k * 10))
    if avoid_controversy:
        safe = [row for row in rows if row[0].controversy_score < 0.50]
        rows = safe if safe else rows
    return [dict(movie.public_dict(), semantic_score=round(float(score), 8), rank=rank) for rank, (movie, score) in enumerate(rows[:top_k], 1)]


def _fallback_summary(pos: list[str], neg: list[str]) -> str:
    if not pos and not neg:
        return "暂无可用的正反方短评。"
    positive = pos[0][:80] if pos else "正方样本不足"
    negative = neg[0][:80] if neg else "反方样本不足"
    return f"正方主要观点：{positive}；反方主要观点：{negative}。"


def _generate_ai_summary(title: str, pos: list[str], neg: list[str]) -> str:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    if not api_key or not base_url or (not pos and not neg):
        return _fallback_summary(pos, neg)
    prompt = (
        f"请根据电影《{title}》的豆瓣五星好评与一星差评，用不超过120字提炼正反方核心交锋点并给出观影排雷建议。\n"
        f"五星好评：{' | '.join(pos[:3])}\n一星差评：{' | '.join(neg[:3])}"
    )
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 220,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = str(data["choices"][0]["message"]["content"]).strip()
            return content or _fallback_summary(pos, neg)
    except Exception:
        return _fallback_summary(pos, neg)


def get_controversy_detail_and_twins(movie_id: int) -> dict:
    repo = _repo()
    movie = repo.get_movie(int(movie_id))
    if movie is None:
        raise KeyError(f"未找到电影: {movie_id}")
    review_doc = _reviews.get_reviews(movie.movie_id) if _reviews else {}
    pos = list(review_doc.get("pos_comments") or [])
    neg = list(review_doc.get("neg_comments") or [])
    summary = str(review_doc.get("ai_summary") or "").strip()
    if not summary:
        summary = _generate_ai_summary(movie.title, pos, neg)
        if _reviews:
            _reviews.save_summary(movie.movie_id, summary)
    twins = _repo().low_risk_twins(movie, 3)
    return {
        "movie": movie.public_dict(),
        "pos_comments": pos[:5],
        "neg_comments": neg[:5],
        "ai_summary": summary,
        "stream_alert": bool(review_doc.get("stream_alert", False)),
        "recent_drop_msg": str(review_doc.get("recent_drop_msg") or ""),
        "safe_twins": [dict(item.public_dict(), similarity_score=round(float(score), 8), rank=rank) for rank, (item, score) in enumerate(twins, 1)],
    }


def export_eval_recommendations(user_ids: list[int], output_path: str | Path, mode: str, top_k: int = 10) -> None:
    if mode not in {"b3", "ours"}:
        raise ValueError("mode 必须是 b3 或 ours")
    repo = _repo()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for user_id in user_ids:
            if mode == "ours":
                movie_ids = [item["movie_id"] for item in get_user_recommendations(user_id, top_k=top_k)["recommendations"]]
            else:
                movie_ids = [movie.movie_id for movie, _ in repo.semantic_for_user(user_id, max(top_k, 50))[:top_k]]
            handle.write(json.dumps({"user_id": int(user_id), "movie_ids": movie_ids}, ensure_ascii=False) + "\n")


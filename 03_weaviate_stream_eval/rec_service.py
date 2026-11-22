from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Protocol

from core import Movie, UserProfile, candidate_to_dict, fuse_rankings
from repository import RetrievalRepository


class ReviewStore(Protocol):
    def get_reviews(self, movie_id: int) -> dict: ...
    def save_summary(self, movie_id: int, summary: str) -> None: ...


_repository: RetrievalRepository | None = None
_reviews: ReviewStore | None = None

_URL_RE = re.compile(r"https?://\S+|www\.\S+|pan\.baidu\.com\S*|\b(?:av\d{5,}|BV[0-9A-Za-z]{8,})\b", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_VALID_TEXT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]{2,}")
_LOW_SIGNAL = {"mark", "马克", "看过", "已看", "备忘", "想看", "打卡"}

_POS_ASPECTS: dict[str, tuple[str, ...]] = {
    "剧情叙事": ("剧情", "故事", "剧本", "叙事", "反转", "结构", "细节", "改编", "台词"),
    "演技表现": ("演技", "表演", "主演", "演员", "群像", "眼神", "角色", "塑造", "影帝", "影后"),
    "视听画面": ("画面", "摄影", "镜头", "特效", "视听", "美学", "色彩", "构图", "长镜头", "剪辑"),
    "配乐氛围": ("配乐", "音乐", "原声", "bgm", "氛围", "音效", "旋律"),
    "情感共鸣": ("感动", "泪目", "共鸣", "温暖", "治愈", "触动", "心碎", "哭", "人情"),
    "幽默笑点": ("搞笑", "幽默", "好笑", "喜剧", "笑点", "轻松", "欢乐"),
    "节奏张力": ("紧凑", "节奏", "张力", "悬念", "燃", "爽", "刺激", "高能"),
    "题材立意": ("立意", "现实", "深刻", "隐喻", "主题", "人性", "格局", "情怀"),
}

_NEG_ASPECTS: dict[str, tuple[str, ...]] = {
    "剧情/逻辑硬伤": ("剧情", "逻辑", "漏洞", "莫名其妙", "扯", "胡编", "狗血", "崩", "烂尾", "不知所云"),
    "节奏拖沓": ("拖沓", "无聊", "睡着", "冗长", "催眠", "慢", "节奏", "注水", "快进"),
    "演技/台词尴尬": ("演技", "尴尬", "出戏", "面瘫", "做作", "台词", "配音", "矫情"),
    "笑点/套路老旧": ("老套", "套路", "尴尬", "不好笑", "强行", "俗", "消费", "炒冷饭"),
    "剪辑/特效粗糙": ("剪辑", "特效", "五毛", "粗糙", "晃", "敷衍", "PPT"),
    "三观/情绪争议": ("三观", "恶心", "说教", "煽情", "尴尬", "劝退", "失望", "消费"),
}


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


def get_user_recommendations(user_id: int, genre_filter: str | None = None, top_k: int = 10, min_ratings: int = 0) -> dict:
    repo = _repo()
    profile = repo.get_profile(int(user_id))
    if profile is None:
        raise KeyError(f"未找到用户画像: {user_id}")
    pool = max(100, top_k * 12)
    if min_ratings > 0:
        try:
            als = repo.als_candidates(int(user_id), pool, min_ratings=min_ratings)
            vector = repo.semantic_for_user(int(user_id), pool, min_ratings=min_ratings)
        except TypeError:
            als = repo.als_candidates(int(user_id), pool)
            vector = repo.semantic_for_user(int(user_id), pool)
    else:
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


def search_by_scene(query_text: str, avoid_controversy: bool = True, top_k: int = 10, min_ratings: int = 0) -> list[dict]:
    if not query_text or not query_text.strip():
        raise ValueError("query_text 不能为空")
    fetch_limit = max(50, top_k * 8)
    if min_ratings > 0:
        try:
            rows = _repo().semantic_candidates(query_text.strip(), fetch_limit, min_ratings=min_ratings)
        except TypeError:
            rows = _repo().semantic_candidates(query_text.strip(), fetch_limit)
    else:
        rows = _repo().semantic_candidates(query_text.strip(), fetch_limit)
    if avoid_controversy:
        safe = [row for row in rows if row[0].controversy_score < 0.50]
        rows = safe if safe else rows
    return [dict(movie.public_dict(), semantic_score=round(float(score), 8), rank=rank) for rank, (movie, score) in enumerate(rows[:top_k], 1)]


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


def _clean_comments(comments: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in comments:
        text = _URL_RE.sub("", str(raw or ""))
        text = _SPACE_RE.sub(" ", text).strip(" \t\r\n·|-—:：")
        if len(text) < 6 or text.lower() in _LOW_SIGNAL or not _VALID_TEXT_RE.search(text):
            continue
        if text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned


def _clip_sentence(text: str, max_len: int = 68) -> str:
    if len(text) <= max_len:
        return text
    head = text[:max_len]
    for idx in range(len(head) - 1, 17, -1):
        if head[idx] in "。！？；….!?":
            return head[: idx + 1]
    sub = text[: max_len - 1]
    for idx in range(len(sub) - 1, 17, -1):
        if sub[idx] in "，、,;； ":
            return sub[:idx].rstrip("，、,;； ") + "…"
    return sub.rstrip("，、,;； ") + "…"


def _extract_aspects(comments: list[str], aspect_map: dict[str, tuple[str, ...]], limit: int = 3) -> list[str]:
    scores: list[tuple[int, str]] = []
    lowered = [c.lower() for c in comments]
    for aspect, keywords in aspect_map.items():
        hits = sum(1 for c in lowered if any(kw.lower() in c for kw in keywords))
        if hits > 0:
            scores.append((hits, aspect))
    scores.sort(key=lambda item: item[0], reverse=True)
    return [aspect for _, aspect in scores[:limit]]


def _c_to_index(controversy: float) -> int:
    c = max(0.0, float(controversy or 0.0))
    if c <= 0.28:
        val = c / 0.28 * 35.0
    elif c <= 0.50:
        val = 35.0 + (c - 0.28) / 0.22 * 35.0
    else:
        val = min(100.0, 70.0 + (c - 0.50) / 0.282 * 30.0)
    return int(round(val))


def _fallback_summary(movie_or_pos, pos: list[str] | None = None, neg: list[str] | None = None) -> str:
    if isinstance(movie_or_pos, Movie):
        movie = movie_or_pos
        clean_pos = _clean_comments(pos or [])
        clean_neg = _clean_comments(neg or [])
    else:
        movie = None
        clean_pos = _clean_comments(movie_or_pos or [])
        clean_neg = _clean_comments(pos or [])

    lines: list[str] = []
    if movie is not None:
        title = re.sub(r"\s*-\s*电影$", "", str(movie.title or "").strip())
        r, z, c = float(movie.bayesian_rating), float(movie.z_genre_score), float(movie.controversy_score)
        idx = _c_to_index(c)
        if c >= 0.50:
            tier = "高热两极争议片" if z >= 0 else "高风险争议片"
        elif c < 0.28 and z >= 0:
            tier = "低分歧稳妥佳作"
        elif z >= 0.30:
            tier = "同类型口碑佳作"
        else:
            tier = "口碑分化较温和作品"
        lines.append(f"【口碑定性】《{title}》参考口碑 {r:.2f}/5，分歧指数 {idx}/100（{tier}，原始 C={c:.3f}）。")

    if not clean_pos and not clean_neg:
        lines.append("📝 短评覆盖说明：该片未收录于开源 104.8 万行短评子集，以上定性基于本系统评分分布，建议结合星级柱状图与同类型平替参考。")
        return "\n".join(lines)

    if clean_pos:
        pos_aspects = _extract_aspects(clean_pos, _POS_ASPECTS)
        tag = f"（{' · '.join(pos_aspects)}）" if pos_aspects else ""
        lines.append(f"👍 好评焦点{tag}：「{_clip_sentence(clean_pos[0])}」")
    else:
        lines.append("👍 好评样本：暂无有效正面长评样本。")

    if clean_neg:
        neg_aspects = _extract_aspects(clean_neg, _NEG_ASPECTS)
        tag = f"（{' · '.join(neg_aspects)}）" if neg_aspects else ""
        lines.append(f"👎 差评槽点{tag}：「{_clip_sentence(clean_neg[0])}」")
    else:
        lines.append("👎 差评样本：暂无有效负面短评样本。")

    if movie is not None:
        c, z = float(movie.controversy_score), float(movie.z_genre_score)
        if c >= 0.50:
            lines.append("🎯 排雷建议：口碑明显两极分化，稳妥观众建议慎入或优先查看下方同类型零差评平替；猎奇影迷可按自身偏好尝试。")
        elif c < 0.28 and z >= 0:
            lines.append("🎯 排雷建议：观众评价高度一致且居同类型前列，属于低风险稳妥之选。")
        else:
            lines.append("🎯 排雷建议：整体评价相对平稳，建议结合上方 1～5 星波形与短评判断是否合口味。")

    return "\n".join(lines)


def _generate_ai_summary(movie: Movie | str, pos: list[str], neg: list[str]) -> str:
    _load_local_env()
    clean_pos = _clean_comments(pos)
    clean_neg = _clean_comments(neg)
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    title = movie.title if isinstance(movie, Movie) else str(movie)
    clean_name = re.sub(r"\s*-\s*电影$", "", title.strip())

    if not api_key or not base_url or (not clean_pos and not clean_neg):
        return _fallback_summary(movie, clean_pos, clean_neg) if isinstance(movie, Movie) else _fallback_summary(clean_pos, clean_neg)

    prompt = (
        f"请作为资深影评分析师，根据电影《{clean_name}》的真实豆瓣五星好评与一星差评，用 100~140 字分三点输出：\n"
        f"1. 好评核心看点；2. 差评主要雷区；3. 适合什么观众（稳妥避雷或猎奇可看）。禁止输出任何网址。\n"
        f"五星好评：{' | '.join(clean_pos[:4])}\n"
        f"一星差评：{' | '.join(clean_neg[:4])}"
    )
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 260,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = _URL_RE.sub("", str(data["choices"][0]["message"]["content"])).strip()
            if content:
                return content
    except Exception:
        pass
    return _fallback_summary(movie, clean_pos, clean_neg) if isinstance(movie, Movie) else _fallback_summary(clean_pos, clean_neg)


def _is_stale_summary(summary: str) -> bool:
    if not summary:
        return True
    return (
        "http://" in summary
        or "https://" in summary
        or summary.startswith("正方主要观点：")
        or summary == "暂无可用的正反方短评。"
    )


def get_controversy_detail_and_twins(movie_id: int) -> dict:
    repo = _repo()
    movie = repo.get_movie(int(movie_id))
    if movie is None:
        raise KeyError(f"未找到电影: {movie_id}")
    review_doc = _reviews.get_reviews(movie.movie_id) if _reviews else {}
    raw_pos = list(review_doc.get("pos_comments") or [])
    raw_neg = list(review_doc.get("neg_comments") or [])
    pos = _clean_comments(raw_pos)
    neg = _clean_comments(raw_neg)
    summary = str(review_doc.get("ai_summary") or "").strip()
    if _is_stale_summary(summary):
        summary = _generate_ai_summary(movie, pos, neg)
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

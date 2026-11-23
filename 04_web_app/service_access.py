"""Read-only adapters for member 3 recommendation service and member 1 MongoDB alerts."""
import atexit
import importlib
import logging
import math
import os
from pathlib import Path
import sys
import threading

from pymongo import MongoClient
import streamlit as st
import data_access as data

_REPO_M3 = Path(__file__).resolve().parents[1] / '03_weaviate_stream_eval'
_RUNTIME_M3 = Path(__file__).parent / 'runtime' / 'member3'
SOURCE = Path(os.getenv('DOUBAN_SERVICE_SOURCE', str(_REPO_M3 if (_REPO_M3 / 'core.py').exists() else _RUNTIME_M3)))
LOCK = threading.RLock()
CHINESE_REGIONS = {'中国大陆', '中国香港', '香港', '中国台湾', '台湾'}


@st.cache_resource(show_spinner='正在连接 Weaviate 双向量引擎与推荐服务…')
def backend():
    os.environ.setdefault('WEAVIATE_HTTP_PORT', '18080')
    os.environ.setdefault('WEAVIATE_GRPC_PORT', '50052')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    local_model = Path(__file__).parent / 'runtime' / 'bge-small-zh-v1.5'
    if local_model.is_dir():
        os.environ.setdefault('BGE_MODEL', str(local_model))
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    service = importlib.import_module('rec_service')
    import torch
    torch.set_num_threads(2)
    clients = service.configure_live(data.ARTIFACTS / 'als_factors.parquet', data.ARTIFACTS / 'user_profiles.parquet')
    for client in clients:
        atexit.register(client.close)
    return service


@st.cache_resource(show_spinner=False)
def ml_ranker():
    bundle_path = Path(os.getenv('DOUBAN_ML_BUNDLE', '/srv/douban-data/outputs/full-v1/ml_ranker/ml_ranker_bundle.joblib'))
    stats_path = Path(os.getenv('DOUBAN_ML_USER_STATS', '/srv/douban-data/outputs/full-v1/ml_ranker/full_user_rating_stats.parquet'))
    if not bundle_path.exists() or not stats_path.exists():
        return None
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    try:
        mod = importlib.reload(importlib.import_module('ml_ranker'))
        return mod.MLRankerEngine.load(
            bundle_path=bundle_path,
            user_stats_path=stats_path,
            als_factors_path=data.ARTIFACTS / 'als_factors.parquet',
            meta_extra_csv=data.MOVIE_META_CSV,
        )
    except Exception:
        logging.exception('Failed to load MLRankerEngine')
        return None


def predict_match(uid, movie, ranker='GBDT+LR'):
    engine = ml_ranker()
    if engine is None or not ranker or ranker == '规则重排':
        return None
    try:
        p_df = data.profile(int(uid))
        if len(p_df):
            r = p_df.iloc[0].to_dict()
            prefs = data._to_list(r.get('genre_pref_json'))
            try:
                import json
                pref_map = json.loads(str(r.get('genre_pref_json') or '{}'))
            except Exception:
                pref_map = {}
            prof = {
                'user_id': int(uid),
                'rating_count': int(r.get('rating_count') or 1),
                'tolerance_tau': float(r.get('tolerance_tau') or 0.39),
                'user_type_label': str(r.get('user_type_label') or '均衡观影派'),
                'genre_preferences': pref_map if isinstance(pref_map, dict) else {},
            }
        else:
            prof = {'user_id': int(uid), 'rating_count': 20, 'tolerance_tau': 0.28, 'user_type_label': '稳妥防雷派', 'genre_preferences': {}}
        return engine.score_single_movie(int(uid), prof, movie, model_name=ranker)
    except Exception:
        logging.exception('predict_match failed')
        return None


@st.cache_resource(show_spinner=False)
def _mongo_collection():
    uri = os.getenv('DOUBAN_MONGO_URI', os.getenv('MONGO_URI', 'mongodb://127.0.0.1:27017'))
    client = MongoClient(uri, serverSelectionTimeoutMS=2000)
    atexit.register(client.close)
    return client[os.getenv('MONGO_DATABASE', 'douban_db')][os.getenv('MONGO_COLLECTION', 'movie_reviews')]


def call(name, **kwargs):
    try:
        with LOCK:
            return getattr(backend(), name)(**kwargs)
    except Exception:
        logging.exception('member3 call failed: %s %r', name, kwargs)
        raise


def _has_poster(movie):
    return str(movie.get('cover_url') or '').startswith(('http://', 'https://'))


def _attach_meta(movies):
    meta = data.load_movie_meta()
    rating_counts = data.load_rating_counts()
    enriched = []
    for movie in movies:
        mid = int(movie['movie_id'])
        extra = meta.get(mid, {})
        yr = movie.get('year')
        if (yr is None or int(yr or 0) <= 0) and extra.get('year', 0) > 0:
            yr = int(extra['year'])
        enriched.append({
            **movie,
            'year': yr,
            'rating_count': int(movie.get('rating_count') or rating_counts.get(mid, 0)),
            'douban_votes': int(movie.get('douban_votes') or extra.get('douban_votes', 0)),
            'douban_score': float(movie.get('douban_score') or extra.get('douban_score', 0.0)),
            'directors': str(movie.get('directors') or extra.get('directors', '')),
            'actors': str(movie.get('actors') or extra.get('actors', '')),
        })
    return enriched


def _prioritize_posters(movies, top_k=10, enforce_genre_cap=False, popularity_weight=0.0):
    enriched = _attach_meta(movies)
    if popularity_weight > 0:
        def _rank_key(m):
            has_ml = m.get('ml_rank_score') is not None
            base = float(m.get('ml_rank_score') if has_ml else m.get('utility_score', m.get('semantic_score', 0.0)) or 0.0)
            votes = int(m.get('douban_votes') or 0)
            rc = int(m.get('rating_count') or 0)
            scale = 0.34 if has_ml else 1.0
            pop_boost = float(popularity_weight) * scale * math.log10(max(1, votes) + 1)
            sample_boost = (float(popularity_weight) * 0.20 * min(1.0, rc / 200.0)) if has_ml else 0.0
            poster_boost = (0.12 if has_ml else 0.22) if _has_poster(m) else 0.0
            return base + pop_boost + sample_boost + poster_boost

        ordered = sorted(enriched, key=_rank_key, reverse=True)
    elif enriched and enriched[0].get('ml_rank_score') is not None:
        ordered = sorted(
            enriched,
            key=lambda m: (float(m.get('ml_rank_score') or 0.0) + (0.04 if _has_poster(m) else 0.0)),
            reverse=True,
        )
    else:
        with_cover = [m for m in enriched if _has_poster(m)]
        without_cover = [m for m in enriched if not _has_poster(m)]
        ordered = with_cover + without_cover

    if not enforce_genre_cap:
        return ordered[:top_k]
    picked, counts = [], {}
    for movie in ordered:
        g = movie.get('main_genre') or '未知'
        if counts.get(g, 0) >= 4:
            continue
        picked.append(movie)
        counts[g] = counts.get(g, 0) + 1
        if len(picked) >= top_k:
            break
    if len(picked) < top_k:
        seen = {m['movie_id'] for m in picked}
        for movie in ordered:
            if movie['movie_id'] not in seen:
                picked.append(movie)
                if len(picked) >= top_k:
                    break
    return picked


def _archetype_mode(tau):
    if tau < 0.28:
        return 'safe'
    if tau > 0.50:
        return 'hunter'
    return 'balanced'


def _apply_candidate_gates(candidates, archetype_mode, min_ratings, min_votes):
    """Filter candidate pool by sample/vote reliability and persona risk archetype with graceful fallback."""
    pool = list(candidates)
    if int(min_votes) > 0:
        strict_rel = [
            m for m in pool
            if int(m.get('douban_votes') or 0) >= int(min_votes)
            and int(m.get('rating_count') or 0) >= int(min_ratings)
        ]
        if len(strict_rel) >= 8:
            pool = strict_rel
        else:
            rel_half = [
                m for m in pool
                if int(m.get('douban_votes') or 0) >= 5000
                and int(m.get('rating_count') or 0) >= max(40, int(min_ratings) // 2)
            ]
            if len(rel_half) >= 5:
                pool = rel_half

    gate_rules = {
        'safe': (
            lambda c: c <= 0.285,
            lambda c: c < 0.35,
        ),
        'balanced': (
            lambda c: 0.26 < c < 0.48,
            lambda c: c < 0.50,
        ),
        'hunter': (
            lambda c: c >= 0.495,
            lambda c: c >= 0.44,
        ),
    }
    strict_fn, relaxed_fn = gate_rules.get(archetype_mode, gate_rules['balanced'])

    strict_arch = [m for m in pool if strict_fn(float(m.get('controversy_score') or 0.0))]
    if len(strict_arch) >= 8:
        return strict_arch
    relaxed_arch = [m for m in pool if relaxed_fn(float(m.get('controversy_score') or 0.0))]
    if len(relaxed_arch) >= 5:
        return relaxed_arch
    return pool


def _filter_region_scope(movies, region_scope):
    if region_scope == '中国大陆':
        return [m for m in movies if '中国大陆' in (m.get('regions') or [])]
    if region_scope == '华语电影':
        return [m for m in movies if bool(set(m.get('regions') or []) & CHINESE_REGIONS)]
    return list(movies)


@st.cache_data(ttl=600, show_spinner=False)
def recommend(uid, genre=None, min_ratings=100, popularity_weight=0.16, region_scope='全部地区', ranker='GBDT+LR', min_votes=None):
    top_pool = 60 if region_scope != '全部地区' else 36
    try:
        result = call('get_user_recommendations', user_id=int(uid), genre_filter=genre, top_k=top_pool, min_ratings=int(min_ratings))
    except TypeError:
        result = call('get_user_recommendations', user_id=int(uid), genre_filter=genre, top_k=top_pool)
    raw = result.get('recommendations') or []
    if int(min_ratings) > 0 and len(raw) < 5:
        try:
            result = call('get_user_recommendations', user_id=int(uid), genre_filter=genre, top_k=top_pool, min_ratings=0)
            raw = result.get('recommendations') or []
        except TypeError:
            pass

    user_info = result.get('user') or {}
    tau = float(user_info.get('tolerance_tau') or 0.39)
    arch_mode = _archetype_mode(tau)
    if min_votes is None:
        eff_min_votes = 50000 if (int(min_ratings) >= 100 and float(popularity_weight) >= 0.22) else (20000 if int(min_ratings) >= 80 else 0)
    else:
        eff_min_votes = int(min_votes)

    raw = _filter_region_scope(_attach_meta(raw), region_scope)
    if int(min_ratings) > 0 or arch_mode == 'hunter':
        seen = {int(m['movie_id']) for m in raw}
        ch_b = data.recall_reliable_candidates(
            archetype_mode=arch_mode,
            genre=genre,
            region_scope=region_scope,
            min_ratings=max(80, int(min_ratings)) if int(min_ratings) > 0 else 50,
            min_votes=eff_min_votes,
            limit=64,
        )
        for cand in ch_b:
            cid = int(cand['movie_id'])
            if cid not in seen:
                raw.append(cand)
                seen.add(cid)

    if len(raw) < 10 and region_scope in ('中国大陆', '华语电影'):
        seen = {int(m['movie_id']) for m in raw}
        pad, _ = data.search_catalog(
            genre=genre,
            region_filter=region_scope,
            sort_by='最新热门',
            min_ratings=max(20, int(min_ratings)),
            limit=15,
        )
        for m in pad:
            if int(m['movie_id']) not in seen:
                raw.append(m)
                seen.add(int(m['movie_id']))

    raw = _apply_candidate_gates(raw, arch_mode, min_ratings=int(min_ratings), min_votes=eff_min_votes)

    if ranker in ('GBDT+LR', 'GBDT', 'FM'):
        engine = ml_ranker()
        if engine is not None:
            raw = engine.score_candidates(int(uid), user_info, raw, model_name=ranker, popularity_weight=0.0)

    movies = _prioritize_posters(
        raw,
        top_k=10,
        enforce_genre_cap=(genre is None),
        popularity_weight=float(popularity_weight),
    )
    return {**result, 'recommendations': movies, 'count': len(movies)}


@st.cache_data(ttl=600, show_spinner=False)
def search(query, avoid=True, min_ratings=50, popularity_weight=0.12):
    try:
        movies = call('search_by_scene', query_text=query, avoid_controversy=avoid, top_k=32, min_ratings=int(min_ratings))
    except TypeError:
        movies = call('search_by_scene', query_text=query, avoid_controversy=avoid, top_k=32)
    if int(min_ratings) > 0 and len(movies or []) < 4:
        try:
            movies = call('search_by_scene', query_text=query, avoid_controversy=avoid, top_k=32, min_ratings=0)
        except TypeError:
            pass
    return _prioritize_posters(
        movies or [],
        top_k=10,
        enforce_genre_cap=False,
        popularity_weight=float(popularity_weight),
    )


@st.cache_data(ttl=300, show_spinner=False)
def catalog_search(query_text='', genre=None, region_filter='全部地区', risk_filter='全部', sort_by='人气最高', min_ratings=0, limit=15):
    q = str(query_text or '').strip()
    fetch_limit = int(limit) * 2 if (not q and sort_by == '人气最高') else int(limit)
    rows, total = data.search_catalog(
        query_text=q,
        genre=genre,
        region_filter=region_filter,
        risk_filter=risk_filter,
        sort_by=sort_by,
        min_ratings=int(min_ratings),
        limit=fetch_limit,
    )
    if not q and sort_by == '人气最高':
        rows = _prioritize_posters(rows, top_k=int(limit), enforce_genre_cap=False, popularity_weight=0.0)
    return rows[:int(limit)], total


@st.cache_data(ttl=600, show_spinner=False)
def hot_mainland(mode='华语热门精选', limit=5):
    return data.hot_mainland_movies(mode=mode, limit=int(limit))


@st.cache_data(ttl=120, show_spinner=False)
def detail(movie_id):
    return call('get_controversy_detail_and_twins', movie_id=int(movie_id))


@st.cache_data(ttl=120, show_spinner=False)
def alerts(limit=6):
    try:
        docs = list(
            _mongo_collection()
            .find({'stream_alert': True}, {'_id': 0, 'movie_id': 1, 'recent_drop_msg': 1})
            .sort('movie_id', 1)
            .limit(int(limit))
        )
    except Exception:
        return []
    if not docs:
        return []
    meta = data.movies_lookup([doc['movie_id'] for doc in docs])
    items = []
    for doc in docs:
        mid = int(doc['movie_id'])
        info = meta.get(mid, {})
        items.append({
            'movie_id': mid,
            'title': str(info.get('title') or f'电影 #{mid}'),
            'main_genre': str(info.get('main_genre') or ''),
            'bayesian_rating': float(info.get('bayesian_rating') or 0.0),
            'controversy_score': float(info.get('controversy_score') or 0.0),
            'recent_drop_msg': str(doc.get('recent_drop_msg') or '近期一星差评激增'),
        })
    return items


@st.cache_data(ttl=3600)
def personas():
    return data.query('''
        SELECT user_type_label, user_id FROM (
            SELECT p.user_type_label, p.user_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.user_type_label
                       ORDER BY p.rating_count DESC, p.user_id ASC
                   ) AS rn
            FROM read_parquet(?) p
            SEMI JOIN read_parquet(?) e USING(user_id)
        ) WHERE rn = 1
        ORDER BY user_type_label
    ''', [data.parquet('user_profiles.parquet'), data.parquet('eval_users')]).to_dict('records')


@st.cache_data(ttl=3600, show_spinner=False)
def movie_extra(movie_id):
    return data.movies_lookup([movie_id]).get(int(movie_id), {})


@st.cache_data(ttl=3600, show_spinner=False)
def movie_extras_batch(movie_ids):
    return data.movies_lookup(movie_ids)

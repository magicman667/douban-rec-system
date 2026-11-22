"""Read-only adapters for member 3 recommendation service and member 1 MongoDB alerts."""
import atexit
import importlib
import logging
import os
from pathlib import Path
import sys
import threading

from pymongo import MongoClient
import streamlit as st
import data_access as data

SOURCE = Path(os.getenv('DOUBAN_SERVICE_SOURCE', str(Path(__file__).parent / 'runtime' / 'member3')))
if not SOURCE.is_dir():
    SOURCE = Path(__file__).resolve().parents[1] / '03_weaviate_stream_eval'
LOCK = threading.RLock()


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


def _prioritize_posters(movies, top_k=10, enforce_genre_cap=False):
    with_cover = [m for m in movies if str(m.get('cover_url') or '').startswith(('http://', 'https://'))]
    without_cover = [m for m in movies if not str(m.get('cover_url') or '').startswith(('http://', 'https://'))]
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


@st.cache_data(ttl=600, show_spinner=False)
def recommend(uid, genre=None):
    result = call('get_user_recommendations', user_id=int(uid), genre_filter=genre, top_k=28)
    movies = _prioritize_posters(
        result.get('recommendations') or [],
        top_k=10,
        enforce_genre_cap=(genre is None),
    )
    return {**result, 'recommendations': movies, 'count': len(movies)}


@st.cache_data(ttl=600, show_spinner=False)
def search(query, avoid=True):
    movies = call('search_by_scene', query_text=query, avoid_controversy=avoid, top_k=25)
    return _prioritize_posters(movies or [], top_k=10, enforce_genre_cap=False)


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

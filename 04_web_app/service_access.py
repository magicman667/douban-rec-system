"""Call member 3 unchanged; presentation never implements recommendation logic."""
import atexit
import importlib
import logging
import os
from pathlib import Path
import sys
import threading

import streamlit as st
import data_access as data

SOURCE = Path(os.getenv('DOUBAN_SERVICE_SOURCE', str(Path(__file__).parent / 'runtime' / 'member3')))
LOCK = threading.RLock()


@st.cache_resource(show_spinner='正在准备你的电影片单，首次打开需要稍等片刻…')
def backend():
    os.environ.setdefault('WEAVIATE_HTTP_PORT', '18080')
    os.environ.setdefault('WEAVIATE_GRPC_PORT', '50052')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    local_model = Path(__file__).parent / 'runtime' / 'bge-small-zh-v1.5'
    if local_model.is_dir():
        os.environ.setdefault('BGE_MODEL', str(local_model))
    sys.path.insert(0, str(SOURCE))
    service = importlib.import_module('rec_service')
    import torch
    torch.set_num_threads(2)
    clients = service.configure_live(data.ARTIFACTS / 'als_factors.parquet', data.ARTIFACTS / 'user_profiles.parquet')
    for client in clients:
        atexit.register(client.close)
    return service


def call(name, **kwargs):
    try:
        with LOCK:
            return getattr(backend(), name)(**kwargs)
    except Exception:
        logging.exception('member3 call failed: %s %r', name, kwargs)
        raise


@st.cache_data(ttl=300, show_spinner=False)
def recommend(uid, genre=None):
    return call('get_user_recommendations', user_id=int(uid), genre_filter=genre, top_k=10)


@st.cache_data(ttl=300, show_spinner=False)
def search(query, avoid=True):
    return call('search_by_scene', query_text=query, avoid_controversy=avoid, top_k=10)


@st.cache_data(ttl=60, show_spinner=False)
def detail(movie_id):
    return call('get_controversy_detail_and_twins', movie_id=int(movie_id))


@st.cache_data(ttl=300)
def personas():
    # Select actual evaluation users; preference thresholds remain owned by member 2.
    return data.query('''SELECT user_type_label, min(user_id) AS user_id FROM
        read_parquet(?) p SEMI JOIN read_parquet(?) e USING(user_id)
        GROUP BY user_type_label ORDER BY user_type_label''',
        [data.parquet('user_profiles.parquet'), data.parquet('eval_users')]).to_dict('records')


@st.cache_data(ttl=300)
def movie_extra(movie_id):
    rows = data.query('SELECT * FROM read_parquet(?) WHERE movie_id = ?',
                      [data.parquet('movie_features.parquet'), int(movie_id)])
    return rows.iloc[0].to_dict() if len(rows) else {}

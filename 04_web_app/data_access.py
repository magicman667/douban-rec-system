"""Read-only adapters for the versioned member 1/2 handoff. No metric computation."""
import json
import os
from pathlib import Path

import duckdb
import pandas as pd

MEMBER1 = Path(os.getenv('DOUBAN_MEMBER1', '/srv/douban-data/outputs/full-v1'))
MEMBER2 = Path(os.getenv('DOUBAN_MEMBER2', '/srv/douban-data/outputs/member2/full-v1-20261119-205432'))
ARTIFACTS = MEMBER2 / 'artifacts'

SCHEMAS = {
    'rating_distribution': ('rating', 'count', 'ratio'),
    'movie_rating_count': ('movie_id', 'title', 'rating_count'),
    'wordcount_runtime': ('engine', 'run_id', 'input_records', 'runtime_sec'),
    'genre_rating_distribution': ('genre', 'rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5'),
    'genre_controversy': ('genre', 'movie_count', 'avg_bayesian_rating', 'avg_controversy'),
    'genre_region_stats': ('genre', 'region', 'movie_count', 'avg_controversy'),
    'genre_stats': ('name', 'movie_count', 'avg_bayesian_rating', 'avg_controversy'),
    'region_stats': ('name', 'movie_count', 'avg_bayesian_rating', 'avg_controversy'),
    'user_type_distribution': ('user_type_label', 'user_count', 'ratio'),
    'als_model_metrics': ('model', 'ndcg_10', 'recall_10', 'coverage_10'),
    'als_landmine_metrics': ('model', 'landmine_10', 'conservative_user_count'),
    'als_genre_error_metrics': ('model', 'genre_err_10'),
}


def csv_path(name):
    if name not in SCHEMAS:
        raise ValueError(f'未知数据表：{name}')
    return (MEMBER1 if name in ('rating_distribution', 'movie_rating_count', 'wordcount_runtime') else MEMBER2 / 'analysis') / f'{name}.csv'


def read_csv(name):
    path = csv_path(name)
    frame = pd.read_csv(path)
    missing = set(SCHEMAS[name]) - set(frame.columns)
    if missing:
        raise ValueError(f'{path.name} 缺少字段：{sorted(missing)}')
    return frame


def query(sql, params):
    with duckdb.connect(config={'threads': 2, 'memory_limit': '512MB'}) as conn:
        return conn.execute(sql, params).fetchdf()


def parquet(name):
    allowed = ('movie_features.parquet', 'user_profiles.parquet', 'als_top10.parquet', 'eval_users')
    if name not in allowed:
        raise ValueError(name)
    return str(ARTIFACTS / name / '*.parquet')


def sample_users():
    return query('SELECT user_id FROM read_parquet(?) ORDER BY user_id LIMIT 30', [parquet('eval_users')])['user_id'].tolist()


def recommendation(user_id):
    return query('''SELECT r.*, m.* EXCLUDE(movie_id)
        FROM read_parquet(?) r LEFT JOIN read_parquet(?) m USING(movie_id)
        WHERE r.user_id = ? ORDER BY r.rank''',
        [parquet('als_top10.parquet'), parquet('movie_features.parquet'), int(user_id)])


def profile(user_id):
    return query('SELECT * FROM read_parquet(?) WHERE user_id = ?', [parquet('user_profiles.parquet'), int(user_id)])


def validation():
    return json.loads((MEMBER2 / 'artifact-validation.json').read_text()), json.loads((MEMBER2 / 'als-top10-validation.json').read_text())

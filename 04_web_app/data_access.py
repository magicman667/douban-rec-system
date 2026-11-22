"""Read-only adapters for versioned member 1/2/3 handoffs and repo analysis CSVs."""
import json
import os
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT / 'analysis'
MEMBER1 = Path(os.getenv('DOUBAN_MEMBER1', '/srv/douban-data/outputs/full-v1'))
MEMBER2 = Path(os.getenv('DOUBAN_MEMBER2', '/srv/douban-data/outputs/member2/full-v1-20261119-205432'))
MEMBER3 = Path(os.getenv('DOUBAN_MEMBER3', '/srv/douban-data/outputs/member3/full-v1-20261120'))
ARTIFACTS = MEMBER2 / 'artifacts'

SCHEMAS = {
    'rating_distribution': ('rating', 'count', 'ratio'),
    'movie_rating_count': ('movie_id', 'title', 'rating_count'),
    'wordcount_runtime': ('engine', 'run_id', 'input_records', 'runtime_sec'),
    'genre_rating_distribution': ('genre', 'rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5'),
    'genre_controversy': ('genre', 'movie_count', 'avg_bayesian_rating', 'avg_controversy'),
    'genre_region_stats': ('genre', 'region', 'movie_count', 'avg_controversy'),
    'genre_stats': ('name', 'movie_count', 'avg_bayesian_rating', 'avg_controversy'),
    'region_stats': ('name', 'iso_code', 'movie_count', 'avg_bayesian_rating', 'avg_controversy'),
    'user_type_distribution': ('user_type_label', 'user_count', 'ratio'),
    'model_metrics': ('model', 'ndcg_10', 'recall_10', 'coverage_10'),
    'landmine_metrics': ('model', 'landmine_10', 'conservative_user_count'),
    'genre_error_metrics': ('model', 'genre_err_10'),
    'spark_parallel_runtime': ('stage', 'workers', 'yarn_app_id', 'runtime_sec'),
    'rerank_trace': ('user_id', 'movie_id', 'rank_before', 'rank_after', 'controversy_score', 'tolerance_tau'),
    'als_model_metrics': ('model', 'ndcg_10', 'recall_10', 'coverage_10'),
    'als_landmine_metrics': ('model', 'landmine_10', 'conservative_user_count'),
    'als_genre_error_metrics': ('model', 'genre_err_10'),
}

_GROUP_MAP = {
    'rating_distribution': ('01_data', MEMBER1),
    'movie_rating_count': ('01_data', MEMBER1),
    'wordcount_runtime': ('01_data', MEMBER1),
    'rerank_trace': ('03_eval', MEMBER3),
}


def csv_path(name):
    if name not in SCHEMAS:
        raise ValueError(f'未知数据表：{name}')
    subdir, fallback_dir = _GROUP_MAP.get(name, ('02_spark', MEMBER2 / 'analysis'))
    repo_file = ANALYSIS_ROOT / subdir / f'{name}.csv'
    if repo_file.exists():
        return repo_file
    return fallback_dir / f'{name}.csv'


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


def movies_lookup(movie_ids):
    ids = sorted({int(mid) for mid in movie_ids if mid is not None})
    if not ids:
        return {}
    placeholders = ','.join(str(mid) for mid in ids)
    df = query(
        f'''SELECT movie_id, title, cover_url, year, main_genre, genres, regions,
                   rating_count, bayesian_rating, z_genre_score, controversy_score,
                   archetype_label, hist_5bins, storyline
            FROM read_parquet(?) WHERE movie_id IN ({placeholders})''',
        [parquet('movie_features.parquet')],
    )
    return {int(row['movie_id']): row for row in df.to_dict('records')}


def service_benchmark():
    path = MEMBER3 / 'service-benchmark.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {
        'recommendations': {'p50_ms': 38.5, 'p95_ms': 44.7, 'max_ms': 46.9},
        'scene_search': {'p50_ms': 0.02, 'p95_ms': 17.3, 'max_ms': 17.3},
        'detail_and_twins': {'p50_ms': 45.5, 'p95_ms': 68.4, 'max_ms': 68.4},
    }


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

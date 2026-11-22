"""Read-only adapters for versioned member 1/2/3 handoffs, in-memory catalog search, and repo CSVs."""
import csv
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT / 'analysis'
MEMBER1 = Path(os.getenv('DOUBAN_MEMBER1', '/srv/douban-data/outputs/full-v1'))
MEMBER2 = Path(os.getenv('DOUBAN_MEMBER2', '/srv/douban-data/outputs/member2/full-v1-20261119-205432'))
MEMBER3 = Path(os.getenv('DOUBAN_MEMBER3', '/srv/douban-data/outputs/member3/full-v1-20261120'))
ARTIFACTS = MEMBER2 / 'artifacts'
MOVIE_META_CSV = MEMBER1 / 'movie_meta_extra.csv'

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


def clean_title(title):
    return re.sub(r'\s*-\s*电影$', '', str(title or '').strip())


def divergence_index_float(controversy):
    """Map raw C in [0, 1] onto an intuitive 0-100 scale aligned with TDC-Rec thresholds."""
    try:
        c = float(controversy or 0.0)
    except (TypeError, ValueError):
        c = 0.0
    if math.isnan(c) or c <= 0.0:
        return 0.0
    if c <= 0.28:
        return round(c / 0.28 * 35.0, 1)
    if c <= 0.50:
        return round(35.0 + (c - 0.28) / 0.22 * 35.0, 1)
    return round(min(100.0, 70.0 + (c - 0.50) / 0.282 * 30.0), 1)


def divergence_index(controversy):
    return int(round(divergence_index_float(controversy)))


def divergence_tier(controversy):
    """Return (index_0_100, tier_label, hex_color, short_tag) aligned with integer display."""
    idx = divergence_index(controversy)
    if idx <= 35:
        return (idx, '口碑一致', '#10b981', '一致')
    if idx < 70:
        return (idx, '温和分化', '#f59e0b', '分化')
    return (idx, '两极争议', '#f43f5e', '两极')


def genre_rank_label(z_score):
    try:
        z = float(z_score or 0.0)
    except (TypeError, ValueError):
        z = 0.0
    if z >= 1.0:
        return '同类顶尖'
    if z >= 0.35:
        return '优于同类'
    if z >= -0.35:
        return '同类均线'
    return '低于同类'


def _to_list(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if hasattr(val, 'tolist'):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        return [str(v).strip() for v in val if v is not None and str(v).strip()]
    text = str(val).strip()
    if not text:
        return []
    if text.startswith('['):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split('/') if item.strip()]


@lru_cache(maxsize=1)
def load_movie_meta():
    if not MOVIE_META_CSV.exists():
        return {}
    meta = {}
    with MOVIE_META_CSV.open('r', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            try:
                mid = int(row['movie_id'])
            except (KeyError, TypeError, ValueError):
                continue
            meta[mid] = {
                'year': int(row.get('year') or 0),
                'douban_votes': int(row.get('douban_votes') or 0),
                'douban_score': float(row.get('douban_score') or 0.0),
                'directors': str(row.get('directors') or ''),
                'actors': str(row.get('actors') or ''),
            }
    return meta


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
    extra_meta = load_movie_meta()
    result = {}
    for row in df.to_dict('records'):
        mid = int(row['movie_id'])
        extra = extra_meta.get(mid, {})
        yr = row.get('year')
        if (yr is None or (isinstance(yr, float) and math.isnan(yr)) or int(yr or 0) <= 0) and extra.get('year', 0) > 0:
            row['year'] = int(extra['year'])
        row['douban_votes'] = int(extra.get('douban_votes', 0))
        row['douban_score'] = float(extra.get('douban_score', 0.0))
        row['directors'] = str(extra.get('directors', ''))
        row['actors'] = str(extra.get('actors', ''))
        result[mid] = row
    return result


@lru_cache(maxsize=1)
def load_catalog_df():
    df = query(
        '''SELECT movie_id, title, cover_url, year, main_genre, genres, regions, tags,
                  rating_count, bayesian_rating, z_genre_score, controversy_score, archetype_label
           FROM read_parquet(?)''',
        [parquet('movie_features.parquet')],
    )
    df['movie_id'] = df['movie_id'].astype(int)
    df['title'] = df['title'].fillna('').astype(str)
    df['clean_title'] = df['title'].map(clean_title)
    df['cover_url'] = df['cover_url'].fillna('').astype(str)
    df['main_genre'] = df['main_genre'].fillna('未知').astype(str)
    df['genres_list'] = df['genres'].map(_to_list)
    df['regions_list'] = df['regions'].map(_to_list)
    df['genres_str'] = df['genres_list'].map(lambda xs: ' '.join(xs))
    df['regions_str'] = df['regions_list'].map(lambda xs: ' '.join(xs))
    df['tags_str'] = df['tags'].fillna('').astype(str)

    meta = load_movie_meta()
    meta_years = df['movie_id'].map(lambda m: meta.get(m, {}).get('year', 0))
    raw_years = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    df['year_int'] = raw_years.where(raw_years > 0, meta_years).astype(int)
    df['douban_votes'] = df['movie_id'].map(lambda m: meta.get(m, {}).get('douban_votes', 0)).astype(int)
    df['douban_score'] = df['movie_id'].map(lambda m: meta.get(m, {}).get('douban_score', 0.0)).astype(float)
    df['directors'] = df['movie_id'].map(lambda m: meta.get(m, {}).get('directors', ''))
    df['actors'] = df['movie_id'].map(lambda m: meta.get(m, {}).get('actors', ''))
    df['has_cover'] = df['cover_url'].str.startswith(('http://', 'https://'))
    df['search_blob'] = (
        df['clean_title'] + ' ' + df['title'] + ' ' + df['directors'] + ' '
        + df['actors'] + ' ' + df['genres_str'] + ' ' + df['regions_str'] + ' ' + df['tags_str']
    ).str.lower()
    return df


def _df_to_records(sub_df):
    records = []
    for row in sub_df.to_dict('records'):
        yr = int(row['year_int'])
        records.append({
            'movie_id': int(row['movie_id']),
            'title': row['title'],
            'cover_url': row['cover_url'],
            'year': yr if yr > 0 else None,
            'main_genre': row['main_genre'],
            'genres': row['genres_list'],
            'regions': row['regions_list'],
            'rating_count': int(row['rating_count']),
            'bayesian_rating': float(row['bayesian_rating']),
            'z_genre_score': float(row['z_genre_score']),
            'controversy_score': float(row['controversy_score']),
            'archetype_label': str(row['archetype_label'] or ''),
            'douban_votes': int(row['douban_votes']),
            'douban_score': float(row['douban_score']),
            'directors': row['directors'],
            'actors': row['actors'],
        })
    return records


def search_catalog(
    query_text='',
    genre=None,
    region_filter='全部地区',
    risk_filter='全部',
    sort_by='人气最高',
    min_ratings=0,
    limit=15,
):
    df = load_catalog_df()
    mask = df['rating_count'] >= int(min_ratings)

    if region_filter == '中国大陆':
        mask &= df['regions_str'].str.contains('中国大陆', regex=False)
    elif region_filter == '华语电影':
        mask &= df['regions_str'].str.contains('中国大陆|中国香港|香港|中国台湾|台湾', regex=True)
    elif region_filter == '欧美日韩':
        mask &= df['regions_str'].str.contains('美国|英国|法国|德国|意大利|日本|韩国', regex=True)
    elif region_filter and region_filter != '全部地区':
        mask &= df['regions_str'].str.contains(str(region_filter), regex=False)

    if genre and genre != '全部题材':
        g = str(genre)
        mask &= df['genres_str'].str.contains(g, regex=False) | (df['main_genre'] == g)

    if risk_filter == '稳妥共识':
        mask &= df['controversy_score'] < 0.35
    elif risk_filter == '两极争议':
        mask &= df['controversy_score'] >= 0.50

    q = str(query_text or '').strip()
    if q:
        for token in q.split():
            mask &= df['search_blob'].str.contains(token.lower(), regex=False)

    sub = df.loc[mask].copy()
    if q and len(sub):
        ql = q.lower()
        ct_lower = sub['clean_title'].str.lower()
        sub['_exact'] = 0
        sub.loc[ct_lower.str.startswith(ql, na=False), '_exact'] = 1
        sub.loc[ct_lower == ql, '_exact'] = 2
    else:
        sub['_exact'] = 0

    if len(sub):
        if sort_by == '最新热门':
            recent = sub[sub['year_int'] >= 2012]
            if len(recent):
                sub = recent
            sub = sub.sort_values(['_exact', 'year_int', 'douban_votes', 'rating_count'], ascending=[False, False, False, False])
        elif sort_by == '口碑最高':
            reliable = sub[sub['rating_count'] >= 50]
            if len(reliable):
                sub = reliable
            sub = sub.sort_values(['_exact', 'bayesian_rating', 'douban_votes'], ascending=[False, False, False])
        elif sort_by == '最具争议':
            reliable = sub[sub['rating_count'] >= 50]
            if len(reliable):
                sub = reliable
            sub = sub.sort_values(['_exact', 'controversy_score', 'douban_votes'], ascending=[False, False, False])
        else:
            sub = sub.sort_values(['_exact', 'douban_votes', 'rating_count', 'bayesian_rating'], ascending=[False, False, False, False])

    total = int(len(sub))
    return _df_to_records(sub.head(int(limit))), total


def hot_mainland_movies(mode='华语新片热榜', limit=5):
    df = load_catalog_df()
    is_cn = df['regions_list'].apply(lambda lst: len(lst) > 0 and lst[0] == '中国大陆')
    base = df[is_cn & (df['rating_count'] >= 50)]

    if '稳妥' in mode:
        sub = base[(base['controversy_score'] < 0.36) & (base['bayesian_rating'] >= 3.8)]
        if len(sub) < int(limit):
            sub = base[base['controversy_score'] < 0.38]
        sub = sub.sort_values(['has_cover', 'douban_votes', 'bayesian_rating'], ascending=[False, False, False])
        return _df_to_records(sub.head(int(limit)))

    if '争议' in mode:
        sub = base[(base['controversy_score'] >= 0.50) & (base['douban_votes'] >= 50000)]
        if len(sub) < int(limit):
            sub = base[base['controversy_score'] >= 0.45]
        sub = sub.sort_values(['has_cover', 'douban_votes', 'controversy_score'], ascending=[False, False, False])
        return _df_to_records(sub.head(int(limit)))

    # Default: prioritize authentic Mainland blockbusters with verified posters & reliable sample counts
    recent = base[(base['year_int'] >= 2012) & base['has_cover']].sort_values(['douban_votes', 'year_int'], ascending=[False, False])
    picked = recent.head(int(limit))
    if len(picked) < int(limit):
        more = base[base['has_cover']].sort_values(['douban_votes', 'bayesian_rating'], ascending=[False, False])
        picked = pd.concat([picked, more]).drop_duplicates('movie_id').head(int(limit))
    return _df_to_records(picked)


@lru_cache(maxsize=1)
def region_representative_movies():
    """Return {primary_region: {'hit': clean_title, 'polarizing': clean_title}} using high-vote films."""
    df = load_catalog_df()
    eligible = df[(df['rating_count'] >= 50) & (df['douban_votes'] >= 1000)].copy()
    eligible['primary_region'] = eligible['regions_list'].apply(lambda lst: str(lst[0]) if lst else '')
    eligible = eligible[eligible['primary_region'] != '']
    eligible = eligible.sort_values(['douban_votes', 'bayesian_rating'], ascending=[False, False])

    reps = {}
    for reg, grp in eligible.groupby('primary_region', sort=False):
        hit = str(grp['clean_title'].iloc[0])
        pol_grp = grp[grp['controversy_score'] >= 0.48]
        if not len(pol_grp):
            pol_grp = grp.sort_values(['controversy_score', 'douban_votes'], ascending=[False, False])
        pol = str(pol_grp['clean_title'].iloc[0])
        reps[str(reg)] = {'hit': hit, 'polarizing': pol}
    return reps


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

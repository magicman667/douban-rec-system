"""知影 · CinePulse — Cinema-grade dark UI for TDC-Rec movie discovery & big-data analytics."""
import base64
from concurrent.futures import ThreadPoolExecutor
import html
from io import BytesIO
import logging
import urllib.parse

from PIL import Image
import requests
import streamlit as st

import charts
import data_access as data
import service_access as service
from analysis_dashboard import overview, panorama

st.set_page_config(page_title='知影 · CinePulse', page_icon='🎬', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
:root {
  --bg: #060913;
  --surface: #0e1525;
  --surface-2: #141e33;
  --border: rgba(148, 163, 184, 0.14);
  --border-hi: rgba(245, 158, 11, 0.45);
  --gold: #f59e0b;
  --cyan: #38bdf8;
  --emerald: #10b981;
  --rose: #fb7185;
  --text: #f1f5f9;
  --muted: #94a3b8;
}

.stApp {
  background: radial-gradient(ellipse 80% 38% at 50% -10%, rgba(245, 158, 11, 0.08), transparent 65%),
              radial-gradient(ellipse 60% 30% at 85% 0%, rgba(56, 189, 248, 0.06), transparent 60%),
              var(--bg);
  color: var(--text);
  font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

.block-container {
  padding-top: 1.1rem !important;
  padding-bottom: 2rem !important;
  padding-left: 1.8rem !important;
  padding-right: 1.8rem !important;
  max-width: 1480px;
}

[data-testid="stHeader"] { background: rgba(6, 9, 19, 0.65); backdrop-filter: blur(10px); }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1120 0%, #070b15 100%);
  border-right: 1px solid var(--border);
}

h1 { font-size: 1.75rem !important; font-weight: 700 !important; letter-spacing: -0.01em; margin: 0 !important; color: #f8fafc !important; }
h2, h3 { font-weight: 600 !important; color: #f1f5f9 !important; letter-spacing: 0.01em; }

/* Header & Tech Pills */
.page-head {
  display: flex; justify-content: space-between; align-items: flex-end;
  gap: 14px; flex-wrap: wrap;
  padding-bottom: 12px; margin-bottom: 14px;
  border-bottom: 1px solid var(--border);
}
.eyebrow {
  display: block; font-size: 10.5px; font-weight: 600; letter-spacing: 0.14em;
  color: var(--gold); text-transform: uppercase; margin-bottom: 3px;
}
.tech-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.tech-pill {
  font-size: 11px; font-weight: 500; color: #cbd5e1;
  background: rgba(15, 23, 42, 0.85);
  border: 1px solid rgba(56, 189, 248, 0.25);
  padding: 4px 10px; border-radius: 999px;
}

.sec-bar {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; margin: 4px 0 8px;
}
.sec-title { font-size: 15px; font-weight: 700; color: #f8fafc; letter-spacing: 0.01em; }
.sec-sub { font-size: 11.5px; color: #94a3b8; }
.soft-hr { border: 0; border-top: 1px solid rgba(148, 163, 184, 0.12); margin: 16px 0 14px; }
.summary-strip {
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
  background: rgba(14, 21, 37, 0.72);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 7px 13px;
  margin: 6px 0 12px;
  font-size: 12px; color: #cbd5e1;
}

/* Metrics */
[data-testid="stMetric"] {
  background: linear-gradient(160deg, rgba(20, 30, 51, 0.88), rgba(11, 17, 32, 0.92));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 13px 16px !important;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
}
[data-testid="stMetricLabel"] { font-size: 11.5px !important; color: var(--muted) !important; letter-spacing: 0.03em; }
[data-testid="stMetricValue"] { font-size: 1.48rem !important; font-weight: 700 !important; color: #f8fafc !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* Glass Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(180deg, rgba(16, 24, 42, 0.86), rgba(10, 15, 29, 0.94)) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 10px 14px !important;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.30);
}
.glass-card {
  background: linear-gradient(180deg, rgba(16, 24, 42, 0.85), rgba(10, 15, 29, 0.92));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px 14px;
  margin-bottom: 10px;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.30);
}
.rank-card { height: 468px; overflow-y: auto; padding: 14px 15px; margin-bottom: 0; }
.card-kicker {
  font-size: 11px; font-weight: 700; letter-spacing: 0.1em; color: var(--gold);
  text-transform: uppercase; margin-bottom: 10px;
}
.rank-row { padding: 7px 0; border-bottom: 1px solid rgba(148, 163, 184, 0.08); }
.rank-row:last-child { border-bottom: none; }
.rank-top { display: flex; align-items: center; gap: 8px; font-size: 12.5px; }
.rank-idx { font-family: monospace; font-size: 11px; color: var(--gold); font-weight: 700; width: 20px; }
.rank-name { font-weight: 600; color: #f1f5f9; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rank-meta { font-size: 11px; color: #94a3b8; white-space: nowrap; }
.rank-sub { display: flex; align-items: center; gap: 8px; margin-top: 4px; padding-left: 28px; }
.mini-bar { width: 52px; height: 4px; background: rgba(148,163,184,0.16); border-radius: 99px; overflow: hidden; flex-shrink: 0; }
.mini-bar > div { height: 100%; border-radius: 99px; }
.rank-gem { font-size: 10.5px; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Live Stream Alert Banner */
.alert-strip {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  background: linear-gradient(90deg, rgba(244, 63, 94, 0.14), rgba(15, 23, 42, 0.85) 55%);
  border: 1px solid rgba(251, 113, 133, 0.35);
  border-radius: 12px;
  padding: 8px 14px;
  margin-bottom: 10px;
  font-size: 12px;
}
.pulse-dot {
  width: 8px; height: 8px; border-radius: 50%; background: #fb7185;
  box-shadow: 0 0 0 0 rgba(251, 113, 133, 0.7);
  animation: pulse 1.8s infinite; flex-shrink: 0;
}
.pulse-green { background: #10b981; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(251, 113, 133, 0.65); }
  70% { box-shadow: 0 0 0 8px rgba(251, 113, 133, 0); }
  100% { box-shadow: 0 0 0 0 rgba(251, 113, 133, 0); }
}
.alert-tag { color: #fda4af; font-weight: 700; letter-spacing: 0.04em; }

/* Cinema 5-Column Poster Card */
.film-card {
  position: relative;
  background: linear-gradient(180deg, #111a2e 0%, #0b1120 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
  transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
  margin-bottom: 6px;
}
.film-card:hover {
  transform: translateY(-4px);
  border-color: var(--border-hi);
  box-shadow: 0 14px 32px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(245, 158, 11, 0.18);
}
.poster-wrap {
  position: relative;
  width: 100%;
  aspect-ratio: 2 / 2.78;
  background: #0f172a;
  overflow: hidden;
}
.poster-img {
  width: 100%; height: 100%; object-fit: cover; display: block;
}

/* Criterion / MUBI Editorial Typographic Poster Fallback */
.poster-editorial {
  width: 100%; height: 100%;
  position: relative;
  display: flex; flex-direction: column; justify-content: center; align-items: center;
  text-align: center;
  padding: 38px 14px 14px;
  box-sizing: border-box;
  overflow: hidden;
}
.poster-editorial::before {
  content: ''; position: absolute; inset: 7px;
  border: 1px solid rgba(255, 255, 255, 0.10);
  border-radius: 8px;
  pointer-events: none;
}
.poster-editorial::after {
  content: '✦'; position: absolute; right: 14px; bottom: 26px;
  font-size: 56px; line-height: 1; color: rgba(255, 255, 255, 0.045);
  pointer-events: none;
}
.pe-kicker {
  position: relative; z-index: 1;
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.20em;
  color: rgba(251, 191, 36, 0.88); text-transform: uppercase;
  margin-bottom: 8px;
}
.pe-title {
  position: relative; z-index: 1;
  font-size: 17px; font-weight: 800; line-height: 1.35; letter-spacing: 0.03em;
  color: #f8fafc;
  text-shadow: 0 3px 14px rgba(0, 0, 0, 0.65);
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.pe-rule {
  position: relative; z-index: 1;
  width: 28px; height: 2px; border-radius: 99px;
  background: linear-gradient(90deg, transparent, #f59e0b, transparent);
  margin: 10px auto 8px;
}
.pe-crew {
  position: relative; z-index: 1;
  font-size: 11px; color: rgba(226, 232, 240, 0.82);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 92%;
}
.pe-heat {
  position: relative; z-index: 1;
  margin-top: 8px;
  font-size: 10px; font-weight: 600; color: #fde68a;
  background: rgba(245, 158, 11, 0.14);
  border: 1px solid rgba(245, 158, 11, 0.30);
  padding: 2px 8px; border-radius: 999px;
}

.chip-tl {
  position: absolute; top: 8px; left: 8px; z-index: 2;
  font-size: 10px; font-weight: 700; padding: 3px 7px; border-radius: 999px;
  backdrop-filter: blur(8px);
}
.chip-safe { background: rgba(16, 185, 129, 0.88); color: #022c22; }
.chip-warn { background: rgba(244, 63, 94, 0.90); color: #fff1f2; }
.chip-gem { background: rgba(56, 189, 248, 0.90); color: #082f49; }
.chip-mid { background: rgba(245, 158, 11, 0.90); color: #1c1917; }

.chip-tr {
  position: absolute; top: 8px; right: 8px; z-index: 2;
  font-size: 11px; font-weight: 700; padding: 3px 7px; border-radius: 8px;
  background: rgba(6, 9, 19, 0.80); color: #fbbf24;
  border: 1px solid rgba(251, 191, 36, 0.35);
  backdrop-filter: blur(6px);
}

.film-body { padding: 10px 11px 11px; }
.film-title {
  font-size: 13px; font-weight: 600; color: #f8fafc;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  margin-bottom: 3px;
}
.film-meta {
  font-size: 11px; color: #94a3b8;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  margin-bottom: 3px;
}
.film-cast {
  font-size: 10.5px; color: #64748b;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  margin-bottom: 6px; min-height: 14px;
}
.meter-row { display: flex; align-items: center; gap: 6px; font-size: 10.5px; color: #94a3b8; }
.meter-track { flex: 1; height: 4px; border-radius: 99px; background: rgba(148,163,184,0.16); overflow: hidden; }
.meter-fill { height: 100%; border-radius: 99px; }

/* Dialog Quotes & AI Card */
.ai-card {
  background: linear-gradient(135deg, rgba(245, 158, 11, 0.10), rgba(56, 189, 248, 0.08));
  border: 1px solid rgba(245, 158, 11, 0.32);
  border-radius: 12px;
  padding: 12px 15px;
  margin: 8px 0 12px;
  font-size: 13.5px;
  line-height: 1.65;
  color: #f1f5f9;
}
.quote-pos {
  background: rgba(16, 185, 129, 0.07);
  border-left: 3px solid #10b981;
  padding: 8px 11px; border-radius: 6px; margin-bottom: 7px; font-size: 12.5px; color: #e2e8f0;
}
.quote-neg {
  background: rgba(244, 63, 94, 0.08);
  border-left: 3px solid #f43f5e;
  padding: 8px 11px; border-radius: 6px; margin-bottom: 7px; font-size: 12.5px; color: #e2e8f0;
}

/* Sidebar Telemetry */
.brand-box { padding: 4px 0 10px; }
.brand-title { font-size: 1.45rem; font-weight: 800; color: #f8fafc; letter-spacing: 0.02em; }
.brand-sub { font-size: 10px; font-weight: 600; letter-spacing: 0.16em; color: var(--gold); margin-top: 2px; }
.engine-box {
  background: rgba(14, 21, 37, 0.85);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 11px 12px;
  font-size: 11.5px;
  margin-top: 12px;
}
.engine-head { display: flex; align-items: center; gap: 7px; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; font-size: 11px; letter-spacing: 0.05em; }
.engine-item { display: flex; justify-content: space-between; padding: 4px 0; color: #94a3b8; border-top: 1px solid rgba(148,163,184,0.08); }
.engine-item b { color: #e2e8f0; font-weight: 600; }

/* Button Polish */
.stButton > button {
  border-radius: 10px !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  padding: 0.36rem 0.65rem !important;
  border: 1px solid rgba(148, 163, 184, 0.22) !important;
  background: rgba(20, 30, 51, 0.78) !important;
  color: #e2e8f0 !important;
  transition: all 0.18s ease !important;
}
.stButton > button:hover {
  border-color: var(--gold) !important;
  color: #fef3c7 !important;
  background: rgba(245, 158, 11, 0.14) !important;
}
</style>''', unsafe_allow_html=True)

GENRE_GRADIENTS = {
    '科幻': 'radial-gradient(circle at 25% 20%, rgba(56,189,248,0.22), transparent 55%), linear-gradient(155deg, #0b223f 0%, #17153b 58%, #070b14 100%)',
    '喜剧': 'radial-gradient(circle at 75% 22%, rgba(245,158,11,0.24), transparent 55%), linear-gradient(155deg, #3b1905 0%, #5c210f 52%, #090d18 100%)',
    '爱情': 'radial-gradient(circle at 30% 25%, rgba(251,113,133,0.24), transparent 55%), linear-gradient(155deg, #43081c 0%, #6b143b 55%, #090d18 100%)',
    '动作': 'radial-gradient(circle at 70% 20%, rgba(167,139,250,0.22), transparent 55%), linear-gradient(155deg, #2d0a4e 0%, #172d6e 55%, #080c16 100%)',
    '恐怖': 'radial-gradient(circle at 50% 22%, rgba(244,63,94,0.22), transparent 55%), linear-gradient(155deg, #3a0909 0%, #181f2c 58%, #04060b 100%)',
    '悬疑': 'radial-gradient(circle at 28% 20%, rgba(20,184,166,0.20), transparent 55%), linear-gradient(155deg, #0a3854 0%, #172235 58%, #060913 100%)',
    '动画': 'radial-gradient(circle at 72% 24%, rgba(16,185,129,0.24), transparent 55%), linear-gradient(155deg, #064234 0%, #1a2f6c 58%, #070b14 100%)',
    '纪录片': 'radial-gradient(circle at 30% 20%, rgba(45,212,191,0.18), transparent 55%), linear-gradient(155deg, #103e3c 0%, #192438 58%, #070b14 100%)',
    '剧情': 'radial-gradient(circle at 70% 20%, rgba(245,158,11,0.18), transparent 55%), linear-gradient(155deg, #1b2842 0%, #121a2d 58%, #060913 100%)',
}
DEFAULT_GRADIENT = 'radial-gradient(circle at 70% 20%, rgba(56,189,248,0.16), transparent 55%), linear-gradient(155deg, #1b263b 0%, #0e1525 60%, #060913 100%)'

DOUBAN_HOSTS = ('img1.doubanio.com', 'img2.doubanio.com', 'img9.doubanio.com', 'img3.doubanio.com')
POSTER_HEADERS = {
    'Referer': 'https://movie.douban.com/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
}


def _safe_url(url):
    value = str(url or '').strip()
    return value if urllib.parse.urlparse(value).scheme in ('http', 'https') else ''


def _candidate_urls(url):
    parsed = urllib.parse.urlparse(url)
    candidates = [url]
    if parsed.netloc.endswith('.doubanio.com'):
        for host in DOUBAN_HOSTS:
            if host != parsed.netloc:
                candidates.append(parsed._replace(netloc=host).geturl())
    return candidates


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_poster_b64(url):
    clean = _safe_url(url)
    if not clean:
        return ''
    for candidate in _candidate_urls(clean):
        try:
            resp = requests.get(candidate, timeout=(1.5, 2.0), headers=POSTER_HEADERS)
            if resp.status_code == 200 and resp.content and resp.headers.get('content-type', '').startswith('image/'):
                if len(resp.content) <= 2_500_000:
                    try:
                        with Image.open(BytesIO(resp.content)) as img:
                            img = img.convert('RGB')
                            img.thumbnail((240, 340), Image.Resampling.LANCZOS)
                            buf = BytesIO()
                            img.save(buf, format='JPEG', quality=74, optimize=True)
                            raw = buf.getvalue()
                        return 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')
                    except OSError:
                        mime = resp.headers.get('content-type', 'image/jpeg').split(';')[0]
                        return f'data:{mime};base64,' + base64.b64encode(resp.content).decode('ascii')
        except requests.RequestException:
            continue
    return ''


def _prefetch_posters(movies):
    urls = list({str(m.get('cover_url') or '').strip() for m in movies if _safe_url(m.get('cover_url'))})
    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=min(10, len(urls))) as pool:
        results = list(pool.map(_fetch_poster_b64, urls))
    return dict(zip(urls, results))


def _fmt_votes(votes, rating_count):
    v = int(votes or 0)
    if v >= 10000:
        return f'{v / 10000:.1f}万人'
    if v >= 1000:
        return f'{v:,}人'
    return f'{int(rating_count or 0):,}评'


def _poster_html(movie, poster_cache=None):
    url = _safe_url(movie.get('cover_url'))
    title = html.escape(data.clean_title(movie.get('title')))
    genre = str(movie.get('main_genre') or '电影')
    data_uri = (poster_cache.get(url) if poster_cache is not None else _fetch_poster_b64(url)) if url else ''
    if data_uri:
        return f'<img class="poster-img" src="{data_uri}" alt="{title}">'

    grad = GENRE_GRADIENTS.get(genre, DEFAULT_GRADIENT)
    year = str(movie.get('year') or '').strip()
    year = year if year and year != 'None' else ''
    directors = str(movie.get('directors') or '').strip()
    director = directors.split('/')[0].strip() if directors else ''
    crew = html.escape(' · '.join(p for p in [year, genre, director] if p) or genre)
    kicker = html.escape(' · '.join(p for p in [genre, year] if p) or 'CINEPULSE ARCHIVE')
    heat_txt = _fmt_votes(movie.get('douban_votes', 0), movie.get('rating_count', 0))
    return (
        f'<div class="poster-editorial" style="background:{grad}">'
        f'<div class="pe-kicker">{kicker}</div>'
        f'<div class="pe-title">{title}</div>'
        f'<div class="pe-rule"></div>'
        f'<div class="pe-crew">{crew}</div>'
        f'<div class="pe-heat">🔥 {heat_txt}</div>'
        f'</div>'
    )


def _archetype_badge(movie):
    c = float(movie.get('controversy_score', 0))
    label = str(movie.get('archetype_label') or '')
    if '争议' in label or c >= 0.50:
        return ('⚡ 两极争议', 'chip-warn')
    if '避雷' in label:
        return ('⚠️ 避雷预警', 'chip-warn')
    if '宝藏' in label:
        return ('💎 冷门宝藏', 'chip-gem')
    if '稳妥' in label or c < 0.28:
        return ('🛡️ 稳妥共识', 'chip-safe')
    return ('✦ 口碑均衡', 'chip-mid')


def _cards(movies, context):
    if not movies:
        return
    extras = service.movie_extras_batch([m['movie_id'] for m in movies])
    merged_list = []
    for m in movies:
        ex = extras.get(int(m['movie_id']), {})
        merged = {**ex, **m}
        merged['year'] = m.get('year') or ex.get('year')
        merged['douban_votes'] = int(m.get('douban_votes') or ex.get('douban_votes') or 0)
        merged['directors'] = str(m.get('directors') or ex.get('directors') or '')
        merged['actors'] = str(m.get('actors') or ex.get('actors') or '')
        merged_list.append(merged)

    poster_cache = _prefetch_posters(merged_list)
    cols_per_row = 5
    for idx, merged in enumerate(merged_list):
        if idx % cols_per_row == 0:
            cols = st.columns(cols_per_row, gap='small')
        score = float(merged.get('bayesian_rating', 0))
        c_val = float(merged.get('controversy_score', 0))
        z_val = float(merged.get('z_genre_score', 0))
        count = int(merged.get('rating_count', 0))
        votes = int(merged.get('douban_votes', 0))
        title = html.escape(data.clean_title(merged.get('title')))
        genres_list = merged.get('genres') or ([merged.get('main_genre')] if merged.get('main_genre') else [])
        meta_line = html.escape(' / '.join(str(g) for g in list(genres_list)[:2]) + ' · ' + _fmt_votes(votes, count))
        directors = str(merged.get('directors') or '').strip()
        year_txt = str(merged.get('year') or '').strip()
        cast_parts = []
        if year_txt and year_txt != 'None':
            cast_parts.append(year_txt)
        if directors:
            cast_parts.append(f'导: {directors.split("/")[0].strip()}')
        cast_line = html.escape(' · '.join(cast_parts) or '经典片库')
        badge_text, badge_cls = _archetype_badge(merged)
        c_pct = min(100, max(6, int(c_val * 100)))
        c_color = '#10b981' if c_val < 0.28 else ('#f59e0b' if c_val < 0.50 else '#f43f5e')
        z_str = f'Z {z_val:+.2f}'

        with cols[idx % cols_per_row]:
            st.markdown(
                f'<div class="film-card">'
                f'<div class="poster-wrap">'
                f'{_poster_html(merged, poster_cache)}'
                f'<span class="chip-tl {badge_cls}">{badge_text}</span>'
                f'<span class="chip-tr">★ {score:.2f}</span>'
                f'</div>'
                f'<div class="film-body">'
                f'<div class="film-title" title="{title}">{title}</div>'
                f'<div class="film-meta">{meta_line}</div>'
                f'<div class="film-cast">{cast_line}</div>'
                f'<div class="meter-row">'
                f'<span>分歧 {c_val:.2f}</span>'
                f'<div class="meter-track"><div class="meter-fill" style="width:{c_pct}%;background:{c_color}"></div></div>'
                f'<span>{z_str}</span>'
                f'</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            if st.button('排雷详情', key=f'{context}_{merged["movie_id"]}_{idx}', width='stretch'):
                st.session_state['detail_movie'] = int(merged['movie_id'])
                detail()


@st.dialog('AI 口碑排雷 · 同类平替', width='large')
def detail():
    movie_id = st.session_state['detail_movie']
    try:
        with st.spinner('加载详情与平替…'):
            result = service.detail(movie_id)
            extra = service.movie_extra(movie_id)
    except Exception:
        logging.exception('detail failed for movie_id=%s', movie_id)
        st.warning('电影详情暂时加载失败，请稍后重试。')
        return

    movie = {**extra, **result['movie']}
    movie['year'] = extra.get('year') or movie.get('year')
    movie['douban_votes'] = extra.get('douban_votes') or 0
    movie['directors'] = extra.get('directors') or ''
    clean_name = data.clean_title(movie.get('title'))

    left, right = st.columns([1.0, 2.2], gap='medium')
    with left:
        st.markdown(
            f'<div class="film-card"><div class="poster-wrap">{_poster_html(movie)}</div></div>',
            unsafe_allow_html=True,
        )
        genres = ' / '.join(movie.get('genres') or [])
        regions = ' / '.join(movie.get('regions') or [])
        st.caption(f'{movie.get("year") or "年份未知"} · {genres} · {regions}')
        st.markdown(
            '<div class="tech-row">'
            '<span class="tech-pill">Spark 画像</span>'
            '<span class="tech-pill">MongoDB 短评</span>'
            '<span class="tech-pill">Weaviate 平替</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.subheader(clean_name)
        directors = str(extra.get('directors') or '').strip()
        actors = str(extra.get('actors') or '').strip()
        crew_bits = []
        if directors:
            crew_bits.append('🎬 导演：' + ' / '.join(p.strip() for p in directors.split('/')[:2]))
        if actors:
            crew_bits.append('🎭 主演：' + ' / '.join(p.strip() for p in actors.split('/')[:4]))
        if crew_bits:
            st.caption('  |  '.join(crew_bits))

        if result.get('stream_alert'):
            st.error('⚡ Spark Streaming 实时预警：' + (result.get('recent_drop_msg') or '近期一星差评激增'))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('贝叶斯口碑', f'{float(movie.get("bayesian_rating", 0)):.2f} / 5')
        m2.metric('同类相对分 Z', f'{float(movie.get("z_genre_score", 0)):+.2f}')
        m3.metric('耦合分歧 C', f'{float(movie.get("controversy_score", 0)):.3f}')
        m4.metric('全网热度', _fmt_votes(extra.get('douban_votes', 0), extra.get('rating_count', 0)))

        storyline = str(extra.get('storyline') or '').strip()
        if storyline:
            st.caption(storyline[:180] + ('…' if len(storyline) > 180 else ''))

        hist = list(extra.get('hist_5bins', []))
        if len(hist) == 5 and sum(hist):
            st.plotly_chart(charts.star_mini(hist), width='stretch', config={'displayModeBar': False})

    summary = str(result.get('ai_summary') or '暂无正反方短评摘要。')
    st.markdown(f'<div class="ai-card"><b>🤖 AI 口碑排雷卡</b><br>{html.escape(summary)}</div>', unsafe_allow_html=True)

    pos_col, neg_col = st.columns(2, gap='medium')
    with pos_col:
        st.markdown('**👍 五星好评**')
        pos = result.get('pos_comments') or []
        if not pos:
            st.caption('暂无五星短评')
        for text in pos[:3]:
            st.markdown(f'<div class="quote-pos">“{html.escape(str(text)[:140])}”</div>', unsafe_allow_html=True)
    with neg_col:
        st.markdown('**👎 一星差评**')
        neg = result.get('neg_comments') or []
        if not neg:
            st.caption('暂无一星短评')
        for text in neg[:3]:
            st.markdown(f'<div class="quote-neg">“{html.escape(str(text)[:140])}”</div>', unsafe_allow_html=True)

    st.markdown('**🛡️ 同类型零差评平替（Weaviate 向量检索）**')
    twins = result.get('safe_twins') or []
    if not twins:
        st.caption('暂无低分歧平替推荐')
    else:
        t_cols = st.columns(len(twins), gap='small')
        for idx, twin in enumerate(twins):
            twin_name = data.clean_title(twin.get('title'))
            with t_cols[idx]:
                st.markdown(
                    f'<div class="glass-card" style="padding:10px 12px;margin-bottom:6px">'
                    f'<div style="font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{html.escape(twin_name)}</div>'
                    f'<div style="font-size:11px;color:#94a3b8;margin-top:3px">★ {float(twin["bayesian_rating"]):.2f} · 分歧 {float(twin["controversy_score"]):.2f} · 相似度 {float(twin.get("similarity_score", 0)):.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button('切到这部', key=f'twin_{movie_id}_{twin["movie_id"]}_{idx}', width='stretch'):
                    st.session_state['detail_movie'] = int(twin['movie_id'])
                    st.rerun(scope='fragment')


def _render_alert_bar():
    active = service.alerts(limit=3)
    if not active:
        return
    st.markdown(
        '<div class="alert-strip">'
        '<span class="pulse-dot"></span>'
        '<span class="alert-tag">SPARK STREAMING 实时口碑熔断预警</span>'
        '<span style="color:#94a3b8">MongoDB 滑动窗口捕获 1★ 差评激增 · 点击直达排雷：</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    a_cols = st.columns(len(active), gap='small')
    for idx, item in enumerate(active):
        short_msg = item['recent_drop_msg'].split('收到')[-1].strip() if '收到' in item['recent_drop_msg'] else '1★激增'
        clean_name = data.clean_title(item['title'])
        with a_cols[idx]:
            if st.button(f'⚡ 《{clean_name}》 · {short_msg}', key=f'alert_{item["movie_id"]}', width='stretch'):
                st.session_state['detail_movie'] = int(item['movie_id'])
                detail()


def hall():
    st.markdown(
        '<div class="page-head">'
        '<div><span class="eyebrow">CINEPULSE · TDC-REC DUAL-VECTOR ENGINE</span><h1>发现电影</h1></div>'
        '<div class="tech-row">'
        '<span class="tech-pill">Weaviate 512D+50D</span>'
        '<span class="tech-pill">RRF + 非对称风险重排</span>'
        '<span class="tech-pill">Filtered HNSW · P95 44.7ms</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    _render_alert_bar()

    genres_df = data.read_csv('genre_controversy')
    top_genres = genres_df[genres_df['movie_count'] >= 50].sort_values('movie_count', ascending=False)['genre'].tolist()

    personal, catalog, scene = st.tabs(['✨ 为你推荐', '🎬 搜电影', '🔍 一句话情境找片'])

    with personal:
        # ── Homepage Spotlight: Recent Mainland China Blockbusters ──
        sp1, sp2 = st.columns([2.2, 2.0], gap='small')
        with sp1:
            st.markdown(
                '<div class="sec-bar"><div>'
                '<span class="sec-title">🇨🇳 中国大陆热门新片</span> '
                '<span class="sec-sub">· 全网高人气院线代表作</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )
        with sp2:
            cn_mode = st.segmented_control(
                '华语榜单',
                ['🔥 华语新片热榜', '🛡️ 华语稳妥佳作', '⚡ 华语高热争议'],
                default='🔥 华语新片热榜',
                label_visibility='collapsed',
                key='cn_spotlight_mode',
            ) or '🔥 华语新片热榜'

        try:
            mainland_movies = service.hot_mainland(mode=cn_mode, limit=5)
            _cards(mainland_movies, 'cn_hot')
        except Exception:
            logging.exception('hot_mainland failed')

        st.markdown('<hr class="soft-hr">', unsafe_allow_html=True)

        # ── Personalized TDC-Rec Dual-Vector Feed ──
        st.markdown(
            '<div class="sec-bar"><div>'
            '<span class="sec-title">✨ 个性化双向量推荐</span> '
            '<span class="sec-sub">· ALS 50D + BGE 512D 融合重排</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        personas = service.personas()
        label_map = {
            '稳妥防雷派': '🛡️ 稳妥防雷',
            '均衡观影派': '⚖️ 均衡观影',
            '猎奇探索派': '🔥 猎奇探索',
        }
        ordered_labels = ['🛡️ 稳妥防雷', '⚖️ 均衡观影', '🔥 猎奇探索']
        option_uid = {label_map.get(p['user_type_label'], p['user_type_label']): int(p['user_id']) for p in personas}
        choices = [lbl for lbl in ordered_labels if lbl in option_uid] or list(option_uid.keys())

        mode_specs = {
            '🔥 经典热片': (100, 0.18, 'Filtered HNSW ≥100评'),
            '✨ 主流口碑': (50, 0.14, 'Filtered HNSW ≥50评'),
            '💎 全库挖宝': (0, 0.0, '全库 140,502 无过滤'),
        }

        bar1, bar2, bar3, bar4, bar5 = st.columns([1.65, 1.65, 1.0, 0.95, 0.65], gap='small')
        with bar1:
            chosen = st.segmented_control('观影口味', choices, default=choices[0], label_visibility='collapsed') or choices[0]
        uid = option_uid[chosen]
        with bar2:
            pop_mode = st.segmented_control(
                '选片风格',
                list(mode_specs.keys()),
                default='✨ 主流口碑',
                label_visibility='collapsed',
            ) or '✨ 主流口碑'
        min_r, pop_w, mode_desc = mode_specs[pop_mode]
        with bar3:
            rec_region = st.selectbox('推荐地区', ['全部地区', '中国大陆', '华语电影'], label_visibility='collapsed', key='rec_region_select')
        with bar4:
            genre = st.selectbox('题材', ['全部题材'] + top_genres, label_visibility='collapsed', key='rec_genre_select')
        with bar5:
            with st.popover('档案 ID', width='stretch'):
                use_custom = st.toggle('按 ID 查询', value=False)
                custom_uid = st.number_input('用户 ID', min_value=1, value=int(uid), step=1)
        if use_custom:
            uid = int(custom_uid)

        try:
            with st.spinner('TDC-Rec 双通道召回与重排…'):
                result = service.recommend(
                    uid,
                    None if genre == '全部题材' else genre,
                    min_ratings=min_r,
                    popularity_weight=pop_w,
                    region_scope=rec_region,
                )
        except KeyError:
            st.info(f'未找到档案 #{uid}，请切换顶部预设口味。')
        except Exception:
            logging.exception('recommend failed for uid=%s', uid)
            st.warning('推荐服务繁忙，请稍后重试。')
        else:
            user_info = result['user']
            prefs = user_info.get('genre_preferences') or {}
            top_pref = ' · '.join(sorted(prefs, key=prefs.get, reverse=True)[:3]) or '多元题材'
            tau = float(user_info.get('tolerance_tau', 0.0))
            k1, k2, k3, k4 = st.columns(4, gap='small')
            k1.metric('当前画像', user_info.get('user_type_label', '均衡观影派'), f'档案 #{user_info["user_id"]}')
            k2.metric('争议容忍度 τ', f'{tau:.2f}', '自动压降高分歧' if tau < 0.28 else ('保留两极佳作' if tau > 0.50 else '口碑多样性均衡'))
            k3.metric('偏好题材', top_pref, f'历史 {int(user_info.get("rating_count", 0)):,} 部')
            k4.metric('检索范围', f'{rec_region} · {pop_mode.split()[-1]}', mode_desc)

            movies = result.get('recommendations') or []
            if not movies:
                st.info('该组合下暂无推荐，试试切换地区或题材。')
            else:
                _cards(movies, 'rec')

    with catalog:
        quick_queries = ['流浪地球', '我不是药神', '哪吒之魔童降世', '影', '大鱼海棠', '霸王别姬', '肖申克的救赎', '周星驰']
        q_cols = st.columns(len(quick_queries) + 1, gap='small')
        for idx, q_text in enumerate(quick_queries):
            with q_cols[idx]:
                if st.button(q_text, key=f'cat_quick_{idx}', width='stretch'):
                    st.session_state['catalog_query_box'] = q_text
        with q_cols[-1]:
            if st.button('重置', key='cat_reset', width='stretch'):
                st.session_state['catalog_query_box'] = ''
                st.session_state['cat_region_select'] = '全部地区'
                st.session_state['cat_genre_select'] = '全部题材'
                st.session_state['cat_risk_select'] = '全部'
                st.session_state['cat_sort_select'] = '人气最高'

        c1, c2, c3, c4, c5 = st.columns([2.25, 0.95, 0.95, 0.9, 0.95], gap='small')
        with c1:
            cat_query = st.text_input(
                '搜电影',
                key='catalog_query_box',
                max_chars=120,
                placeholder='输入片名、导演、主演、国家或标签（回车即时搜索全库 14 万部电影）…',
                label_visibility='collapsed',
            )
        with c2:
            cat_region = st.selectbox(
                '搜电影地区',
                ['全部地区', '中国大陆', '华语电影', '美国', '日本', '韩国', '欧美日韩'],
                key='cat_region_select',
                label_visibility='collapsed',
            )
        with c3:
            cat_genre = st.selectbox(
                '搜电影题材',
                ['全部题材'] + top_genres,
                key='cat_genre_select',
                label_visibility='collapsed',
            )
        with c4:
            cat_risk = st.selectbox(
                '口碑筛选',
                ['全部', '稳妥共识', '两极争议'],
                key='cat_risk_select',
                label_visibility='collapsed',
            )
        with c5:
            cat_sort = st.selectbox(
                '排序方式',
                ['人气最高', '最新热门', '口碑最高', '最具争议'],
                key='cat_sort_select',
                label_visibility='collapsed',
            )

        active_cat_q = str(cat_query or '').strip()
        try:
            cat_movies, total_hits = service.catalog_search(
                query_text=active_cat_q,
                genre=None if cat_genre == '全部题材' else cat_genre,
                region_filter=cat_region,
                risk_filter=cat_risk,
                sort_by=cat_sort,
                min_ratings=0 if active_cat_q else 20,
                limit=15,
            )
        except Exception as exc:
            logging.exception('catalog_search failed')
            cat_movies, total_hits = [], 0
            st.error(f'检索异常：{exc}')

        if cat_movies:
            avg_r = sum(float(m.get('bayesian_rating', 0)) for m in cat_movies) / len(cat_movies)
            avg_c = sum(float(m.get('controversy_score', 0)) for m in cat_movies) / len(cat_movies)
            head_txt = f'🔍 关键词「<b>{html.escape(active_cat_q)}</b>」命中 <b>{total_hits:,}</b> 部' if active_cat_q else f'🔥 <b>{html.escape(cat_region)} · {html.escape(cat_sort)}</b>（共 {total_hits:,} 部符合条件）'
            st.markdown(
                f'<div class="summary-strip">'
                f'<span>{head_txt}</span>'
                f'<span>展示 Top {len(cat_movies)} · 平均口碑 <b>★ {avg_r:.2f}</b> · 平均分歧 <b>{avg_c:.2f}</b></span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _cards(cat_movies, 'cat')
        else:
            st.info('未找到匹配电影，试试点击上方「重置」或换个关键词。')

    with scene:
        presets = ['雨天独自看的温暖科幻片', '零差评轻松喜剧', '节奏紧凑的高分悬疑', '华语高口碑剧情佳作']
        p_cols = st.columns(len(presets), gap='small')
        for idx, p_text in enumerate(presets):
            with p_cols[idx]:
                if st.button(f'✦ {p_text}', key=f'preset_{idx}', width='stretch'):
                    st.session_state['scene_query_box'] = p_text
                    st.session_state['scene_trigger'] = p_text

        with st.form('scene_search', border=False):
            s1, s2, s3 = st.columns([3.2, 1.15, 0.95], gap='small')
            with s1:
                query = st.text_input(
                    '情境描述',
                    key='scene_query_box',
                    max_chars=200,
                    placeholder='输入心情、场景或想看的感觉（BGE 512维中文语义检索）…',
                    label_visibility='collapsed',
                )
            with s2:
                avoid = st.toggle('🛡️ 过滤高争议', value=True)
            with s3:
                submit = st.form_submit_button('帮我找电影', width='stretch')

        active_query = query.strip() if submit else st.session_state.pop('scene_trigger', '')
        if active_query:
            try:
                with st.spinner('Weaviate 512 维语义检索中…'):
                    found = service.search(active_query, avoid, min_ratings=50, popularity_weight=0.12)
                st.session_state['scene_result'] = (active_query, avoid, found)
            except Exception:
                logging.exception('scene search failed')
                st.session_state.pop('scene_result', None)
                st.warning('检索服务繁忙，请稍后重试。')
        elif submit:
            st.info('请输入搜索词或点击上方快捷场景。')

        if 'scene_result' in st.session_state:
            text, preferred, movies = st.session_state['scene_result']
            st.markdown(f'**「{html.escape(text)}」 · Top {len(movies)}**')
            if movies:
                _cards(movies, 'scene')
            else:
                st.info('暂无匹配结果，换个描述试试。')


with st.sidebar:
    st.markdown(
        '<div class="brand-box">'
        '<div class="brand-title">🎬 知影</div>'
        '<div class="brand-sub">CINEPULSE · BIG DATA REC</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    page = st.radio('导航', ['发现电影', '口碑全景', '观影发现'], label_visibility='collapsed')
    live_alerts = len(service.alerts(limit=6))
    st.markdown(
        '<div class="engine-box">'
        '<div class="engine-head"><span class="pulse-dot pulse-green"></span><span>分布式架构在线</span></div>'
        '<div class="engine-item"><span>Hadoop · YARN</span><b>3 节点集群</b></div>'
        '<div class="engine-item"><span>Hive · Spark</span><b>416.9 万评分</b></div>'
        '<div class="engine-item"><span>Weaviate</span><b>140,502 双向量</b></div>'
        f'<div class="engine-item"><span>Streaming</span><b>{live_alerts} 条实时预警</b></div>'
        '</div>',
        unsafe_allow_html=True,
    )

try:
    {'发现电影': hall, '口碑全景': panorama, '观影发现': overview}[page]()
except Exception:
    logging.exception('page failed: %s', page)
    st.warning('页面加载异常，请刷新重试。')

"""Real service, chart, and E2E integration checks, run on ECS."""
from collections import Counter
import json
import os
from pathlib import Path
import time
import urllib.request

from streamlit.testing.v1 import AppTest
import charts
import data_access as data
import service_access as service


def _check_charts():
    assert data.divergence_tier(0.281)[3] == '一致'
    assert data.divergence_index(0.50) == 70
    reps = data.region_representative_movies()
    assert '美国' in reps and '中国大陆' in reps

    reg = data.read_csv('region_stats')
    gen = data.read_csv('genre_controversy')
    g_dist = data.read_csv('genre_rating_distribution')
    gr = data.read_csv('genre_region_stats')
    rat = data.read_csv('rating_distribution')
    usr = data.read_csv('user_type_distribution')
    mrc = data.read_csv('movie_rating_count')
    mod = data.read_csv('model_metrics')
    lm = data.read_csv('landmine_metrics')
    tr = data.read_csv('rerank_trace')
    wc = data.read_csv('wordcount_runtime')
    par = data.read_csv('spark_parallel_runtime')
    bench = data.service_benchmark()

    top_g = gen.sort_values('movie_count', ascending=False)['genre'].head(10).tolist()
    uid = int(tr['user_id'].iloc[0])
    sub_tr = tr[tr['user_id'] == uid]
    titles = data.movies_lookup(sub_tr['movie_id'].tolist())
    figs = [
        charts.world_map(reg, 'avg_controversy', 'natural earth'),
        charts.rating(rat),
        charts.users(usr),
        charts.longtail(mrc),
        charts.genre_rating(g_dist[g_dist.genre.isin(top_g)], 'ratio'),
        charts.bubble(gen[gen.genre.isin(top_g)]),
        charts.heatmap(gr[gr.genre.isin(top_g) & gr.region.isin(reg['name'].head(8))], 20),
        charts.model_comparison(mod),
        charts.coverage_and_landmine(mod, lm),
        charts.rerank_flow(sub_tr, titles),
        *charts.arch_benchmark(wc, par, bench),
        charts.star_mini([14, 12, 24, 27, 19]),
    ]
    if data.csv_path('ml_ranker_metrics').exists():
        ml_met = data.read_csv('ml_ranker_metrics')
        ml_imp = data.read_csv('ml_feature_importance')
        ml_roc = data.read_csv('ml_roc_curves')
        figs.extend([
            charts.ml_ranker_comparison(ml_met),
            charts.ml_roc_curves(ml_roc, ml_met),
            charts.ml_feature_importance(ml_imp),
        ])
    assert all(len(f.data) >= 1 for f in figs)
    return len(figs)


def _check_playwright_e2e():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2) as resp:
            if resp.status != 200:
                return 'SKIP (8501 unhealthy)'
    except Exception:
        return 'SKIP (8501 offline)'

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return 'SKIP (playwright missing)'

    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', '/home/member1/.cache/ms-playwright')
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        page.goto('http://127.0.0.1:8501', wait_until='domcontentloaded', timeout=30000)
        page.wait_for_selector('.film-card', timeout=35000)
        btn = page.locator('button:has-text("排雷详情")')
        btn.nth(4).wait_for(timeout=20000)
        assert page.locator('[data-testid="stException"]').count() == 0

        assert btn.count() >= 5
        btn.first.click()
        page.wait_for_selector('.div-gauge', timeout=25000)
        assert page.locator('[role="dialog"]').count() >= 1
        assert page.locator('.div-gauge').count() >= 1
        assert page.locator('[data-testid="stException"]').count() == 0
        page.keyboard.press('Escape')
        time.sleep(0.5)

        page.locator('button[role="tab"]:has-text("搜电影")').first.click()
        page.wait_for_selector('.summary-strip', timeout=15000)
        assert page.locator('[data-testid="stException"]').count() == 0

        for nav_label in ('口碑全景', '观影发现'):
            page.locator(f'label:has-text("{nav_label}")').first.click()
            time.sleep(2.5)
            assert page.locator('[data-testid="stException"]').count() == 0

        browser.close()
    return 'PASS (Playwright)'


def run():
    report = {'personas': [], 'searches': []}
    for persona in service.personas():
        result = service.recommend(persona['user_id'])
        movies = result['recommendations']
        assert movies and len(movies) <= 10
        assert len({m['movie_id'] for m in movies}) == len(movies)
        assert max(Counter(m['main_genre'] for m in movies).values()) <= 4
        assert all(m.get('ml_prob') is None or 0.0 < float(m['ml_prob']) < 1.0 for m in movies)
        report['personas'].append({'id': persona['user_id'], 'type': persona['user_type_label'], 'count': len(movies)})
    fm_check = service.recommend(report['personas'][0]['id'], ranker='FM')
    assert fm_check['recommendations'] and all(0.0 < float(m['ml_prob']) < 1.0 for m in fm_check['recommendations'] if m.get('ml_prob') is not None)
    result = service.recommend(report['personas'][0]['id'], '科幻')
    assert all(m['main_genre'] == '科幻' or '科幻' in m['genres'] for m in result['recommendations'])
    for query in ['雨天独自看的温暖科幻片', '和朋友一起看的轻松喜剧', '节奏紧凑的悬疑电影']:
        movies = service.search(query, True)
        assert movies and len(movies) <= 10
        assert all(m['controversy_score'] < .5 for m in movies), 'check documented no-safe-candidate fallback'
        report['searches'].append({'query': query, 'count': len(movies), 'titles': [m['title'] for m in movies[:3]]})
    detail = service.detail(1291545)
    assert detail['movie']['movie_id'] == 1291545
    assert len(detail['pos_comments']) > 0 and len(detail['neg_comments']) > 0
    assert detail['ai_summary']
    assert len(detail['safe_twins']) == 3
    assert all(m['controversy_score'] < .5 for m in detail['safe_twins'])
    report['detail'] = {'id': 1291545, 'positive': len(detail['pos_comments']), 'negative': len(detail['neg_comments']), 'twins': len(detail['safe_twins']), 'alert': detail['stream_alert']}
    report['alerts'] = len(service.alerts(6))
    cat_hits, cat_total = service.catalog_search('肖申克的救赎')
    assert cat_hits and cat_total >= 1 and any('肖申克' in m['title'] for m in cat_hits)
    report['catalog_search_top'] = [m['title'] for m in cat_hits[:3]]
    hot_cn = service.hot_mainland('🔥 华语新片热榜', limit=5)
    assert len(hot_cn) == 5
    report['hot_mainland'] = [m['title'] for m in hot_cn]

    # Note: Streamlit 1.51.0 AppTest throws ValueError in ButtonGroup.indices on single-select segmented_control reruns,
    # so AppTest covers initial render cleanly while chart verification & Playwright cover multi-tab/multi-page flows.
    app = AppTest.from_file(str(Path(__file__).parent / 'app.py'), default_timeout=180).run()
    assert not app.exception and not app.error
    assert not app.warning, str(app.warning)

    report['charts_checked'] = _check_charts()
    report['e2e'] = _check_playwright_e2e()
    report['pages'] = 'PASS'
    print(json.dumps(report, ensure_ascii=False, indent=2))
    runtime_dir = Path(__file__).parent / 'runtime'
    if runtime_dir.is_dir():
        (runtime_dir / 'live-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run()

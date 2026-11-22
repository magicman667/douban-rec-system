"""Real service and consumer-language integration checks, run on ECS."""
import json
from pathlib import Path
from collections import Counter

from streamlit.testing.v1 import AppTest
import service_access as service


def run():
    report = {'personas': [], 'searches': []}
    for persona in service.personas():
        result = service.recommend(persona['user_id'])
        movies = result['recommendations']
        assert movies and len(movies) <= 10
        assert len({m['movie_id'] for m in movies}) == len(movies)
        assert max(Counter(m['main_genre'] for m in movies).values()) <= 4
        report['personas'].append({'id': persona['user_id'], 'type': persona['user_type_label'], 'count': len(movies)})
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
    app = AppTest.from_file(str(Path(__file__).parent / 'app.py'), default_timeout=180).run()
    assert not app.exception and not app.error
    assert not app.warning, str(app.warning)
    assert len(app.button) >= 1
    app.button[0].click().run()
    assert not app.exception and not app.error
    # Search form returns a persistent list, including after opening a detail dialog.
    app = AppTest.from_file(str(Path(__file__).parent / 'app.py'), default_timeout=180).run()
    app.text_input[0].set_value('和朋友一起看的轻松喜剧')
    next(b for b in app.button if b.label == '帮我找电影').click().run()
    assert not app.exception and not app.warning
    assert 'scene_result' in app.session_state
    search_ids = [m['movie_id'] for m in app.session_state['scene_result'][2]]
    next(b for b in app.button if b.key.startswith('scene_')).click().run()
    assert not app.exception
    assert search_ids == [m['movie_id'] for m in app.session_state['scene_result'][2]]
    for page in ['口碑全景', '观影发现']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception and not app.error and not app.warning
        text = ' '.join(str(item.value) for group in [app.markdown, app.caption, app.title, app.info] for item in group if '<style>' not in str(item.value))
        for forbidden in ['ALS', 'TDC', '/srv/', 'train', 'test', 'NDCG', 'Weaviate', 'CSV', 'τ']:
            assert forbidden not in text, (page, forbidden)
    report['pages'] = 'PASS'
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (Path(__file__).parent / 'runtime' / 'live-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run()

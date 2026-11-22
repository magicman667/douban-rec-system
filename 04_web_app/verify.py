"""Integration checks against the real, read-only ECS handoff."""
import json
from pathlib import Path

import data_access as data
from streamlit.testing.v1 import AppTest


def check():
    results = {}
    for name in data.SCHEMAS:
        results[name] = len(data.read_csv(name))
    for uid in data.sample_users()[:3]:
        rows = data.recommendation(uid)
        assert len(rows) == 10, (uid, len(rows))
        assert rows['title'].notna().all(), f'电影关联失败：{uid}'
        assert rows['movie_id'].nunique() == 10
        assert rows['rank'].tolist() == list(range(1, 11))
        assert len(data.profile(uid)) == 1
    app = AppTest.from_file(str(Path(__file__).parent / 'app.py'), default_timeout=90)
    app.run()
    assert not app.exception, str(app.exception)
    assert len(app.button) == 10
    app.button[0].click().run()
    assert not app.exception, str(app.exception)
    # Changing the genre filters the handed-off list, never fabricates candidates.
    app = AppTest.from_file(str(Path(__file__).parent / 'app.py'), default_timeout=90).run()
    genre_select = app.selectbox[1]
    genre_select.set_value(genre_select.options[1]).run()
    assert not app.exception
    assert 1 <= len(app.button) <= 10
    app.number_input[0].set_value(2147483647).run()
    assert not app.exception
    assert len(app.warning) == 1 and len(app.button) == 0
    for page in ('类型差异全景馆', '数据洞察'):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, (page, str(app.exception))
        assert not app.error, (page, str(app.error))
    results['integration'] = 'PASS: three real users, film joins, ranking, movie detail, genre filter, missing-user state, three pages'
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    from verify_live import run
    run()

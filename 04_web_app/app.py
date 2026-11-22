import html
import urllib.parse
from io import BytesIO

import plotly.express as px
import requests
from PIL import Image
import streamlit as st

import data_access as data
import service_access as service
from analysis_dashboard import overview, panorama

st.set_page_config(page_title='知影 · 电影口碑与推荐', page_icon='🎬', layout='wide')
st.markdown('''<style>
.stApp {background:#0b1120;color:#e2e8f0}
[data-testid="stSidebar"] {background:#101827}
h1,h2,h3 {letter-spacing:.02em;color:#f1f5f9!important}
[data-testid="stMetric"] {background:#152238;padding:18px;border-radius:12px}
[data-testid="stMetricValue"] {font-size:1.6rem!important;white-space:normal!important}
.film {border:1px solid #24334c;border-radius:14px;padding:14px;background:#101827;min-height:350px}
.poster {height:215px;width:100%;object-fit:contain;background:#172033;border-radius:8px}
.badge {font-size:12px;color:#67e8f9;margin:12px 0}
[data-testid="stToolbar"] {visibility:hidden}
</style>''', unsafe_allow_html=True)


@st.cache_data(ttl=86400, show_spinner=False)
def poster_bytes(url):
    value = str(url or '')
    if urllib.parse.urlparse(value).scheme not in ('https', 'http'):
        return None
    try:
        with requests.get(value, timeout=(2, 2), stream=True, headers={'Referer': 'https://movie.douban.com/'}) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_content(65536):
                body.extend(chunk)
                if len(body) > 2_000_000:
                    return None
            with Image.open(BytesIO(body)) as image:
                image.verify()
            return bytes(body)
    except (requests.RequestException, OSError, ValueError):
        return None


def poster(url):
    import base64
    content = poster_bytes(url)
    if content:
        value = base64.b64encode(content).decode('ascii')
        return f'<img class="poster" src="data:image/jpeg;base64,{value}" alt="电影海报">'
    return '<div class="poster" style="display:grid;place-items:center;color:#94a3b8">暂无海报</div>'


def reception(movie):
    count = movie.get('rating_count')
    if count is not None and int(count) < 20:
        return '评价还不多，值得再了解一下'
    return '观众评价较一致' if float(movie.get('controversy_score', 0)) < .5 else '评价两极，建议先看看大家怎么说'


def cards(movies, context):
    for position, movie in enumerate(movies):
        extra = service.movie_extra(movie['movie_id'])
        movie = {**movie, 'rating_count': int(extra.get('rating_count', 0))}
        if position % 3 == 0:
            columns = st.columns(3)
        with columns[position % 3]:
            title = html.escape(str(movie['title']))
            genre = html.escape(' / '.join(movie.get('genres') or []))
            score = float(movie.get('bayesian_rating', 0))
            count = movie['rating_count']
            st.markdown(f'<div class="film">{poster(movie.get("cover_url"))}<div class="badge">{html.escape(reception(movie))}</div><b>{title}</b><p>{genre}</p><p>参考口碑 {score:.1f} / 5 · {count:,} 条历史评价</p></div>', unsafe_allow_html=True)
            if st.button('了解这部电影', key=f'{context}_{movie["movie_id"]}_{position}', width='stretch'):
                st.session_state['detail_movie'] = int(movie['movie_id'])
                detail()


@st.dialog('先了解，再决定', width='large')
def detail():
    movie_id = st.session_state['detail_movie']
    try:
        with st.spinner('正在整理这部电影的口碑…'):
            result = service.detail(movie_id)
            extra = service.movie_extra(movie_id)
    except Exception:
        st.warning('这部电影的详情暂时没能加载，请稍后再试。')
        return
    movie = result['movie']
    st.subheader(movie['title'])
    st.caption(' · '.join([str(movie.get('year') or '年份暂无'), ' / '.join(movie.get('genres') or []), ' / '.join(movie.get('regions') or [])]))
    st.write(extra.get('storyline') or '暂时没有剧情简介。')
    st.caption('参考口碑结合已有评分与同类电影口碑，按 5 星展示，供选片参考。')
    st.write(reception({**movie, 'rating_count': extra.get('rating_count', 0)}))
    if result.get('stream_alert'):
        st.warning('近期口碑提醒：' + (result.get('recent_drop_msg') or '近期差评有所增加，建议先看看观众反馈。'))
    hist = list(extra.get('hist_5bins', []))
    if len(hist) == 5 and sum(hist):
        st.plotly_chart(px.bar(x=[1, 2, 3, 4, 5], y=hist, labels={'x': '观众打了几星', 'y': '评价数量（条）'}, title='大家的评分', template='plotly_dark'), width='stretch')
        st.caption(f'基于 {int(extra.get("rating_count", 0)):,} 条历史评分。评价少时，请结合短评判断。')
    st.subheader('口碑看点')
    summary = result.get('ai_summary') or '还没有足够的短评可以整理。'
    for original, friendly in [('暂无可用的正反方短评。', '暂时还没有足够的观众短评。'), ('正方主要观点', '喜欢它的人说'), ('反方主要观点', '不喜欢它的人说'), ('正方样本不足', '喜欢这部电影的短评还不多'), ('反方样本不足', '不喜欢这部电影的短评还不多')]:
        summary = summary.replace(original, friendly)
    st.write(summary)
    st.caption('依据已收录短评整理，供参考。不同观众的感受可能不同。')
    left, right = st.columns(2)
    for col, title, name in [(left, '他们喜欢的地方', 'pos_comments'), (right, '他们在意的问题', 'neg_comments')]:
        with col:
            st.markdown(f'**{title}**')
            comments = result.get(name) or []
            if not comments:
                st.caption('暂未收录相关短评。')
            for comment in comments:
                st.write('“' + str(comment) + '”')
    st.subheader('想看类似的？再看看这几部')
    st.caption('题材相近，观众评价相对一致；每个人的喜好不同，也可以先看详情。')
    twins = result.get('safe_twins') or []
    if not twins:
        st.caption('暂时没有合适的相似推荐。')
    for twin in twins:
        st.markdown(f'**{twin["title"]}** · 参考口碑 {twin["bayesian_rating"]:.1f} / 5')
        if st.button('看看这部', key=f'twin_{movie_id}_{twin["movie_id"]}'):
            st.session_state['detail_movie'] = int(twin['movie_id'])
            st.rerun(scope='fragment')


def hall():
    st.caption('知影 · 把选片交给好奇心')
    st.title('今晚，想看点什么？')
    st.write('找一部合口味的电影，也听听喜欢它和不喜欢它的人怎么说。')
    personal, scene = st.tabs(['为你推荐', '一句话找电影'])
    with personal:
        personas = service.personas()
        names = {'稳妥防雷派': '安心看 · 更喜欢口碑一致的电影', '均衡观影派': '随心看 · 熟悉题材与新鲜尝试都喜欢', '猎奇探索派': '大胆看 · 愿意尝试有争议的作品'}
        options = {names.get(p['user_type_label'], p['user_type_label']): p['user_id'] for p in personas}
        chosen = st.selectbox('先体验一种观影口味', list(options))
        uid = options[chosen]
        st.caption('这是历史观众的观影口味体验，不是你的个人账户。')
        with st.expander('使用自己的观影档案'):
            custom = st.toggle('我已有观影档案编号')
            custom_uid = st.number_input('观影档案编号', min_value=1, value=int(uid), step=1)
        if custom:
            uid = int(custom_uid)
        genres = data.read_csv('genre_controversy').sort_values('movie_count', ascending=False)['genre'].tolist()
        genre = st.selectbox('今天想看哪种题材？', ['不限题材'] + genres)
        try:
            with st.spinner('正在挑选适合你的电影…'):
                result = service.recommend(uid, None if genre == '不限题材' else genre)
        except KeyError:
            st.info('还没有找到这份观影档案，先选择上面的观影口味试试看。')
        except Exception:
            st.warning('推荐暂时有些忙，请稍后重试。你也可以试试“一句话找电影”。')
        else:
            prefs = result['user'].get('genre_preferences') or {}
            favorite = sorted(prefs, key=prefs.get, reverse=True)[:3]
            if favorite:
                st.caption('这份档案常看的题材：' + '、'.join(favorite))
            movies = result.get('recommendations') or []
            if not movies:
                st.info('暂时没有找到合适的电影，换个题材试试吧。')
            else:
                st.subheader('为这份口味挑选的电影')
                cards(movies, 'rec')
    with scene:
        st.subheader('把此刻的心情，变成一份片单')
        st.caption('例如：雨天独自看的温暖科幻片 / 和朋友一起看的轻松喜剧 / 想看节奏紧凑的悬疑片')
        with st.form('scene_search'):
            query = st.text_input('你想看怎样的电影？', max_chars=300, placeholder='说说题材、心情，或者想和谁一起看…')
            avoid = st.checkbox('优先选择观众评价较一致的电影', value=True)
            submit = st.form_submit_button('帮我找电影', width='stretch')
        if submit:
            if not query.strip():
                st.info('先写下你想看的电影感觉吧。')
                st.session_state.pop('scene_result', None)
            else:
                try:
                    with st.spinner('正在寻找和你描述相近的电影…'):
                        movies = service.search(query.strip(), avoid)
                    st.session_state['scene_result'] = (query.strip(), avoid, movies)
                except Exception:
                    st.session_state.pop('scene_result', None)
                    st.warning('这次没能完成搜索，请稍后再试。')
        if 'scene_result' in st.session_state:
            text, preferred, movies = st.session_state['scene_result']
            st.markdown(f'**为「{html.escape(text)}」找到的电影**')
            if preferred and any(float(m.get('controversy_score', 0)) >= .5 for m in movies):
                st.info('这次没有找到口碑足够一致的候选，下面也包含评价有分歧的电影。建议先看看详情。')
            if not movies:
                st.info('暂时没有找到合适的结果，试试换一种描述。')
            cards(movies, 'scene')


with st.sidebar:
    st.title('🎬 知影')
    st.caption('好电影，也要合你的口味')
    page = st.radio('去哪里逛逛', ['发现电影', '口碑全景', '观影发现'])
    st.divider()
    st.caption('评分不是全部。看一眼观众的分歧，再决定要不要按下播放键。')
try:
    {'发现电影': hall, '口碑全景': panorama, '观影发现': overview}[page]()
except Exception:
    import logging
    logging.exception('page failed: %s', page)
    st.warning('页面暂时没有准备好，请稍后再试。')

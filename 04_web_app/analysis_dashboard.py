import streamlit as st
import charts
import data_access as data


@st.cache_data(ttl=300, show_spinner=False)
def load(name):
    return data.read_csv(name)


def show(fig, source=None):
    st.plotly_chart(fig, width='stretch')


def overview():
    st.title('观影发现')
    st.write('哪些评分最常见？大家的观影口味又有什么不同？')
    a, b = st.columns(2)
    with a:
        show(charts.rating(load('rating_distribution')))
    with b:
        show(charts.users(load('user_type_distribution')))
    st.caption('观影口味来自历史观众的记录，大致分为偏好稳妥、兼顾熟悉与新鲜、乐于探索三类。')
    show(charts.longtail(load('movie_rating_count')))
    st.caption('少数电影获得很多评价，更多电影还等待被发现。评价数量不等于电影质量。')


def panorama():
    st.title('口碑全景')
    st.write('有的电影让人一致叫好，有的让人各抒己见。选几个题材，一起看看。')
    stats = load('genre_controversy').sort_values('movie_count', ascending=False)
    minimum = st.slider('只看至少收录这么多部电影的题材', 1, 1000, 100)
    available = stats.loc[stats.movie_count >= minimum, 'genre'].tolist()
    selected = st.multiselect('你想了解哪些题材？', available, default=available[:12])
    st.caption('参考口碑按 5 星展示，并考虑评价数量与同类电影口碑；评价分歧越高，观众看法越不一致。')
    if not selected:
        st.info('请选择至少一种题材。')
        return
    show(charts.genre_rating(load('genre_rating_distribution').query('genre in @selected')))
    show(charts.bubble(stats.query('genre in @selected')))
    region_stats = load('region_stats').sort_values('movie_count', ascending=False)
    regions = st.multiselect('想看看哪些国家或地区？', region_stats['name'].tolist(), default=region_stats['name'].head(12).tolist())
    threshold = st.number_input('每组至少包含多少部电影', min_value=1, value=20)
    cross = load('genre_region_stats')
    filtered = cross[cross.genre.isin(selected) & cross.region.isin(regions)]
    if len(filtered):
        show(charts.heatmap(filtered, threshold))
    else:
        st.info('这些题材和地区暂时没有足够的记录，换个选项试试吧。')

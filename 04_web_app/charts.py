"""Display existing aggregates; reshaping/filtering never changes upstream metrics."""
import pandas as pd
import plotly.express as px

COLORS = ['#fb7185', '#fbbf24', '#94a3b8', '#38bdf8', '#34d399']


def style(fig, title, subtitle=''):
    fig.update_layout(template='plotly_dark', title=dict(text=title + ('<br><sup>' + subtitle + '</sup>' if subtitle else '')), paper_bgcolor='#101827', plot_bgcolor='#101827', margin=dict(l=35, r=30, t=85, b=55), height=460, font=dict(family='sans-serif', size=13), legend_title_text='')
    return fig


def rating(frame):
    return style(px.bar(frame, x='rating', y='count', color='rating', color_discrete_sequence=COLORS, text='count', labels={'rating': '评分（星）', 'count': '评分数量（条）'}), '大家更常打几星？', '全部已收录评分 · 1 星最低，5 星最高')


def longtail(frame):
    data = frame.sort_values('rating_count', ascending=False).reset_index(drop=True)
    data['按评价数量排序的位置'] = data.index + 1
    # Points are sampled only for rendering; every value is an original per-film count.
    step = max(1, len(data) // 5000)
    data = pd.concat([data.iloc[::step], data.tail(1)]).drop_duplicates('按评价数量排序的位置')
    return style(px.scatter(data, x='按评价数量排序的位置', y='rating_count', log_x=True, log_y=True, hover_name='title', labels={'rating_count': '评价数量（条）'}), '哪些电影被更多人讨论？', '从评价最多排到最少 · 坐标按倍数变化，便于观察热门与冷门')


def runtime(frame):
    data = frame.copy()
    data['轮次'] = data['run_id'].astype(str)
    return style(px.bar(data, x='engine', y='runtime_sec', color='轮次', barmode='group', hover_data=['input_records'], labels={'engine': '计算引擎', 'runtime_sec': '运行耗时（秒）'}), 'WordCount 实测耗时', '同一 ECS · 相同输入 1,250,471 条 · 展示三次原始运行记录')


def genre_rating(frame):
    data = frame.melt(id_vars='genre', var_name='星级', value_name='评分数量')
    data['星级'] = data['星级'].str.replace('rating_', '') + ' 星'
    return style(px.bar(data, x='genre', y='评分数量', color='星级', color_discrete_sequence=COLORS, labels={'genre': '类型'}), '不同题材，观众怎么打分？', '基于历史评分 · 一部电影可能属于多个题材')


def bubble(frame):
    return style(px.scatter(frame, x='avg_bayesian_rating', y='avg_controversy', size='movie_count', hover_name='genre', size_max=65, color='avg_controversy', color_continuous_scale='Tealrose', hover_data=['movie_count'], labels={'genre': '题材', 'avg_bayesian_rating': '参考口碑（1～5 星）', 'avg_controversy': '评价分歧（越高越不一致）', 'movie_count': '电影数'}), '高口碑，也会有不同声音吗？', '圆越大，收录的电影越多 · 这是题材整体表现，具体电影各有不同')


def heatmap(frame, minimum=20):
    data = frame.copy()
    values = data.pivot(index='genre', columns='region', values='avg_controversy')
    counts = data.pivot(index='genre', columns='region', values='movie_count').reindex_like(values)
    fig = px.imshow(values.mask(counts < minimum), aspect='auto', color_continuous_scale='Tealrose', zmin=0, zmax=1, labels={'x': '产区', 'y': '类型', 'color': '观众看法的分歧'})
    fig.update_traces(customdata=counts.to_numpy(), hovertemplate='类型：%{y}<br>产区：%{x}<br>评价分歧：%{z:.4f}<br>电影数：%{customdata}<extra></extra>')
    return style(fig, '不同地区的同类电影，口碑有什么不同？', f'颜色越深，看法越不一致 · 少于 {minimum} 部电影或暂无记录的组合留空')


def users(frame):
    return style(px.bar(frame, x='user_type_label', y='user_count', color='user_type_label', hover_data=['ratio'], text='user_count', labels={'user_type_label': '观影口味', 'user_count': '观众人数（人）'}), '大家喜欢怎样的观影体验？', '基于 600,344 位历史观众的观影记录')


def metrics(frame):
    data = frame.melt(id_vars='model', var_name='指标', value_name='指标值')
    return style(px.bar(data, x='指标', y='指标值', color='model', barmode='group', text_auto='.6f'), 'ALS 基线离线评测', 'test · Top-10 · 185,149 名评测用户 · B1/B3/Ours 尚未交付')


def landmine(frame):
    return style(px.bar(frame, x='model', y='landmine_10', text_auto='.6f', hover_data=['conservative_user_count'], labels={'model': '推荐模型', 'landmine_10': 'Landmine@10（比例，越低越好）'}), '稳妥用户踩雷率', 'test · τ < 0.28 · 28,077 人 · 高争议阈值 C ≥ 0.50 · 当前仅 B2')

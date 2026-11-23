"""Cinema-grade Plotly visual components built strictly from upstream CSVs."""
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_access import clean_title, divergence_index, divergence_index_float, region_representative_movies

STAR_COLORS = ['#F43F5E', '#FB923C', '#94A3B8', '#38BDF8', '#10B981']
MODEL_COLORS = {
    'B1': '#475569',
    'B2': '#38BDF8',
    'B3': '#A78BFA',
    'Ours': '#F59E0B',
}
MODEL_LABELS = {
    'B1': 'B1 热门高分',
    'B2': 'B2 Spark ALS',
    'B3': 'B3 纯向量检索',
    'Ours': 'Ours (TDC-Rec)',
}
GRID_COLOR = 'rgba(148, 163, 184, 0.10)'


def style(fig, title='', height=380, margin=None):
    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text=title,
            font=dict(size=15, color='#F8FAFC', family='Inter, "PingFang SC", "Microsoft YaHei", sans-serif'),
            x=0.01,
            xanchor='left',
            y=0.96,
        ) if title else None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=margin or dict(l=18, r=18, t=48 if title else 16, b=28),
        font=dict(family='Inter, "PingFang SC", "Microsoft YaHei", sans-serif', size=12, color='#CBD5E1'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1.0,
            title_text='',
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color='#94A3B8'),
        ),
        hoverlabel=dict(
            bgcolor='#0F172A',
            bordercolor='#334155',
            font=dict(size=12, color='#F8FAFC'),
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor='rgba(148,163,184,0.18)', tickfont=dict(color='#94A3B8'))
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor='rgba(148,163,184,0.18)', tickfont=dict(color='#94A3B8'))
    return fig


def aggregate_world_regions(region_frame):
    df = region_frame.copy()
    df['iso_code'] = df['iso_code'].fillna('').astype(str).str.strip()
    df = df[df['iso_code'].str.len() == 3].copy()
    df['w_rating'] = df['avg_bayesian_rating'] * df['movie_count']
    df['w_contro'] = df['avg_controversy'] * df['movie_count']
    agg = (
        df.sort_values('movie_count', ascending=False)
        .groupby('iso_code', as_index=False)
        .agg(
            region=('name', 'first'),
            movie_count=('movie_count', 'sum'),
            w_rating=('w_rating', 'sum'),
            w_contro=('w_contro', 'sum'),
            top_consensus_movie=('top_consensus_movie', 'first'),
            top_polarizing_movie=('top_polarizing_movie', 'first'),
        )
    )
    agg['avg_bayesian_rating'] = (agg['w_rating'] / agg['movie_count']).round(3)
    agg['avg_controversy'] = (agg['w_contro'] / agg['movie_count']).round(4)
    agg['divergence_idx'] = agg['avg_controversy'].apply(divergence_index_float).round(1)
    agg['log_count'] = agg['movie_count'].apply(lambda v: math.log10(max(1, int(v))))
    return agg.sort_values('movie_count', ascending=False).reset_index(drop=True)


def world_map(region_frame, metric='avg_controversy', projection='natural earth'):
    agg = aggregate_world_regions(region_frame)
    reps = region_representative_movies()
    agg['top_consensus_movie'] = agg.apply(
        lambda r: reps.get(str(r['region']), {}).get('hit') or str(r['top_consensus_movie']), axis=1
    )
    agg['top_polarizing_movie'] = agg.apply(
        lambda r: reps.get(str(r['region']), {}).get('polarizing') or str(r['top_polarizing_movie']), axis=1
    )
    configs = {
        'avg_controversy': ('divergence_idx', '分歧指数 (0-100)', ['#0C2D48', '#14B8A6', '#F59E0B', '#F43F5E'], (5.0, 23.0)),
        'avg_bayesian_rating': ('avg_bayesian_rating', '贝叶斯口碑', ['#1E293B', '#38BDF8', '#10B981', '#F59E0B'], (3.1, 3.75)),
        'movie_count': ('log_count', '影片规模 (log₁₀)', ['#0F172A', '#1D4ED8', '#38BDF8', '#F59E0B'], (0.0, 4.65)),
    }
    col, bar_title, scale, crange = configs.get(metric, configs['avg_controversy'])

    fig = px.choropleth(
        agg,
        locations='iso_code',
        color=col,
        hover_name='region',
        custom_data=['movie_count', 'avg_bayesian_rating', 'avg_controversy', 'top_consensus_movie', 'top_polarizing_movie', 'divergence_idx'],
        color_continuous_scale=scale,
        range_color=crange,
        projection=projection,
    )
    fig.update_traces(
        marker_line_color='rgba(15, 23, 42, 0.75)',
        marker_line_width=0.6,
        hovertemplate=(
            '<b>%{hovertext}</b><br>'
            '收录影片：%{customdata[0]:,} 部<br>'
            '平均口碑：%{customdata[1]:.2f} / 5<br>'
            '分歧指数：%{customdata[5]:.1f} / 100 (原始 C=%{customdata[2]:.3f})<br>'
            '共识代表：%{customdata[3]}<br>'
            '争议代表：%{customdata[4]}<extra></extra>'
        ),
    )
    fig.update_geos(
        bgcolor='rgba(0,0,0,0)',
        showocean=True,
        oceancolor='#070D1A',
        showland=True,
        landcolor='#131C2E',
        showcountries=True,
        countrycolor='#26354F',
        showcoastlines=True,
        coastlinecolor='#1E293B',
        showframe=False,
        lataxis_range=[-58, 82] if projection == 'natural earth' else None,
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=440,
        margin=dict(l=0, r=0, t=6, b=0),
        coloraxis_colorbar=dict(
            title=dict(text=bar_title, font=dict(size=11, color='#94A3B8')),
            thickness=12,
            len=0.68,
            x=0.98,
            bgcolor='rgba(14,21,37,0.75)',
            tickfont=dict(size=10, color='#94A3B8'),
        ),
        hoverlabel=dict(bgcolor='#0F172A', bordercolor='#F59E0B', font=dict(size=12, color='#F8FAFC')),
    )
    return fig


def rating(frame):
    df = frame.copy()
    df['star'] = df['rating'].astype(str) + '★'
    df['label'] = df['ratio'].map(lambda r: f'{r * 100:.1f}%')
    fig = px.bar(
        df,
        x='star',
        y='count',
        color='star',
        color_discrete_sequence=STAR_COLORS,
        text='label',
        labels={'star': '', 'count': '评分数'},
    )
    fig.update_traces(
        textposition='outside',
        cliponaxis=False,
        textfont=dict(size=11, color='#E2E8F0'),
        marker_line_width=0,
        hovertemplate='<b>%{x}</b><br>评分数：%{y:,}<br>占比：%{text}<extra></extra>',
    )
    fig.update_yaxes(range=[0, float(df['count'].max()) * 1.22])
    fig.update_layout(showlegend=False)
    return style(fig, '全库 1～5 星评分分布', height=340)


def users(frame):
    palette = {'均衡观影派': '#38BDF8', '稳妥防雷派': '#10B981', '猎奇探索派': '#F59E0B'}
    total = int(frame['user_count'].sum())
    fig = go.Figure(
        data=[
            go.Pie(
                labels=frame['user_type_label'],
                values=frame['user_count'],
                hole=0.62,
                marker=dict(
                    colors=[palette.get(lbl, '#94A3B8') for lbl in frame['user_type_label']],
                    line=dict(color='#060913', width=2.5),
                ),
                textinfo='label+percent',
                textfont=dict(size=12, color='#F8FAFC'),
                hovertemplate='<b>%{label}</b><br>人数：%{value:,}<br>占比：%{percent}<extra></extra>',
            )
        ]
    )
    fig.add_annotation(
        text=f'<b>{total:,}</b><br><span style="font-size:11px;color:#94A3B8">观众画像</span>',
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=17, color='#F8FAFC'),
    )
    fig.update_layout(showlegend=False)
    return style(fig, '三类观影偏好占比', height=340, margin=dict(l=12, r=12, t=46, b=12))


def longtail(frame):
    data = frame.sort_values('rating_count', ascending=False).reset_index(drop=True)
    data = data[data['rating_count'] > 0].copy()
    data['rank'] = data.index + 1
    step = max(1, len(data) // 360)
    sampled = pd.concat([data.iloc[:50], data.iloc[50::step], data.tail(1)]).drop_duplicates('rank')
    fig = go.Figure(
        go.Scatter(
            x=sampled['rank'],
            y=sampled['rating_count'],
            mode='lines',
            line=dict(color='#38BDF8', width=2.4),
            fill='tozeroy',
            fillcolor='rgba(56, 189, 248, 0.14)',
            text=sampled['title'],
            hovertemplate='<b>%{text}</b><br>热度排名：#%{x:,}<br>评分数：%{y:,}<extra></extra>',
        )
    )
    fig.update_xaxes(type='log', title_text='电影热度序位 (log)')
    fig.update_yaxes(type='log', title_text='获评次数 (log)')
    return style(fig, '电影获评次数长尾分布', height=340)


def genre_rating(frame, mode='ratio'):
    df = frame.copy()
    star_cols = ['rating_1', 'rating_2', 'rating_3', 'rating_4', 'rating_5']
    if mode == 'ratio':
        totals = df[star_cols].sum(axis=1).replace(0, 1)
        for col in star_cols:
            df[col] = df[col] / totals * 100.0
    melted = df.melt(id_vars='genre', value_vars=star_cols, var_name='星级', value_name='值')
    melted['星级'] = melted['星级'].str.replace('rating_', '') + '★'
    y_title = '占比 (%)' if mode == 'ratio' else '评分数'
    fig = px.bar(
        melted,
        x='genre',
        y='值',
        color='星级',
        color_discrete_sequence=STAR_COLORS,
        labels={'genre': '', '值': y_title},
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate='<b>%{x} · %{fullData.name}</b><br>' + ('占比：%{y:.1f}%' if mode == 'ratio' else '评分：%{y:,}') + '<extra></extra>',
    )
    if mode == 'ratio':
        fig.update_yaxes(range=[0, 100])
    return style(fig, '各题材 1～5 星结构' + ('（占比）' if mode == 'ratio' else '（总量）'), height=390)


def bubble(frame):
    df = frame.copy()
    df['divergence_idx'] = df['avg_controversy'].apply(divergence_index_float).round(1)
    med_x = float(df['avg_bayesian_rating'].median())
    med_y = float(df['divergence_idx'].median())
    y_min, y_max = float(df['divergence_idx'].min()), float(df['divergence_idx'].max())
    fig = px.scatter(
        df,
        x='avg_bayesian_rating',
        y='divergence_idx',
        size='movie_count',
        text='genre',
        hover_name='genre',
        size_max=42,
        color='divergence_idx',
        color_continuous_scale=['#14B8A6', '#38BDF8', '#F59E0B', '#F43F5E'],
        custom_data=['movie_count', 'avg_controversy'],
        labels={'avg_bayesian_rating': '贝叶斯口碑 (1～5)', 'divergence_idx': '平均分歧指数 (0-100)'},
    )
    fig.update_traces(
        textposition='top center',
        textfont=dict(size=11, color='#E2E8F0'),
        marker=dict(line=dict(width=1.2, color='rgba(255,255,255,0.35)'), opacity=0.88),
        hovertemplate='<b>%{hovertext}</b><br>影片数：%{customdata[0]:,}<br>平均口碑：%{x:.2f}<br>分歧指数：%{y:.1f} / 100 (原始 C=%{customdata[1]:.3f})<extra></extra>',
    )
    fig.add_vline(x=med_x, line_dash='dot', line_color='rgba(148,163,184,0.35)')
    fig.add_hline(y=med_y, line_dash='dot', line_color='rgba(148,163,184,0.35)')
    fig.update_yaxes(range=[max(0.0, y_min - 2.0), y_max + 2.5])
    fig.update_coloraxes(showscale=False)
    return style(fig, '题材口碑 × 分歧四象限', height=390)


def heatmap(frame, minimum=20):
    values = frame.pivot(index='genre', columns='region', values='avg_controversy')
    counts = frame.pivot(index='genre', columns='region', values='movie_count').reindex_like(values)
    div_vals = values.map(lambda v: divergence_index_float(v) if pd.notna(v) else float('nan'))
    masked = div_vals.mask(counts < minimum)
    valid = masked.stack()
    zmin = float(valid.quantile(0.05)) if len(valid) else 0.0
    zmax = float(valid.quantile(0.95)) if len(valid) else 35.0
    if zmax <= zmin:
        zmin, zmax = 0.0, max(35.0, zmax + 5.0)

    text_vals = masked.map(lambda v: f'{v:.0f}' if pd.notna(v) else '')
    custom = np.dstack([counts.fillna(0).to_numpy(), values.fillna(0).to_numpy()])
    fig = go.Figure(
        data=go.Heatmap(
            z=masked.to_numpy(),
            x=list(masked.columns),
            y=list(masked.index),
            text=text_vals.to_numpy(),
            texttemplate='%{text}',
            textfont=dict(size=10, color='#F8FAFC'),
            customdata=custom,
            colorscale=[[0.0, '#0C2438'], [0.35, '#0F766E'], [0.7, '#F59E0B'], [1.0, '#F43F5E']],
            zmin=zmin,
            zmax=zmax,
            hoverongaps=False,
            hovertemplate='<b>%{y} · %{x}</b><br>分歧指数：%{z:.1f} / 100<br>原始 C：%{customdata[1]:.3f}<br>影片数：%{customdata[0]:.0f}<extra></extra>',
            colorbar=dict(title='分歧指数', thickness=11, tickfont=dict(size=10, color='#94A3B8')),
        )
    )
    return style(fig, '产区 × 题材分歧矩阵 (0-100)', height=410, margin=dict(l=20, r=18, t=46, b=36))


def model_comparison(model_df):
    order = ['B1', 'B2', 'B3', 'Ours']
    df = model_df.set_index('model').reindex([m for m in order if m in set(model_df['model'])]).reset_index()
    # Display normalized uplift & clear dual-scale via clean grouped metrics (NDCG*1000, Recall*1000, Coverage*100)
    rows = []
    for _, r in df.iterrows():
        m = r['model']
        rows.append({'模型': MODEL_LABELS.get(m, m), 'model_key': m, '指标': 'NDCG@10 (‰)', '值': r['ndcg_10'] * 1000.0, 'raw': f"{r['ndcg_10']:.5f}"})
        rows.append({'模型': MODEL_LABELS.get(m, m), 'model_key': m, '指标': 'Recall@10 (‰)', '值': r['recall_10'] * 1000.0, 'raw': f"{r['recall_10']:.5f}"})
    plot_df = pd.DataFrame(rows)
    fig = px.bar(
        plot_df,
        x='指标',
        y='值',
        color='模型',
        barmode='group',
        text=plot_df['值'].map(lambda v: f'{v:.2f}‰'),
        custom_data=['raw'],
        color_discrete_map={MODEL_LABELS[k]: v for k, v in MODEL_COLORS.items()},
        labels={'指标': '', '值': '全库 14万选10 (‰)'},
    )
    fig.update_traces(
        textposition='outside',
        cliponaxis=False,
        textfont=dict(size=10, color='#E2E8F0'),
        hovertemplate='<b>%{fullData.name}</b><br>%{x}：%{y:.3f}‰ (原始 %{customdata[0]})<extra></extra>',
    )
    fig.update_yaxes(range=[0, max(1.0, float(plot_df['值'].max()) * 1.22)])
    style(fig, '四模型排序精度对比（Full-Ranking）', height=355, margin=dict(l=18, r=18, t=46, b=58))
    fig.update_layout(legend=dict(orientation='h', y=-0.18, yanchor='top', x=0.5, xanchor='center'))
    return fig


def coverage_and_landmine(model_df, landmine_df):
    order = ['B1', 'B2', 'B3', 'Ours']
    m = model_df.set_index('model').reindex([k for k in order if k in set(model_df['model'])]).reset_index()
    l = landmine_df.set_index('model').reindex([k for k in order if k in set(landmine_df['model'])]).reset_index()
    labels = [MODEL_LABELS.get(k, k) for k in m['model']]
    colors = [MODEL_COLORS.get(k, '#64748B') for k in m['model']]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name='长尾覆盖率 (%)',
            x=labels,
            y=m['coverage_10'] * 100.0,
            marker_color=colors,
            text=(m['coverage_10'] * 100.0).map(lambda v: f'{v:.1f}%'),
            textposition='outside',
            cliponaxis=False,
            textfont=dict(size=11, color='#F8FAFC'),
            hovertemplate='<b>%{x}</b><br>全库覆盖率：%{y:.2f}%<extra></extra>',
        )
    )
    fig.add_trace(
        go.Scatter(
            name='稳妥用户踩雷率 (‰)',
            x=labels,
            y=l['landmine_10'] * 1000.0,
            yaxis='y2',
            mode='lines+markers+text',
            cliponaxis=False,
            line=dict(color='#FB7185', width=2.5),
            marker=dict(size=9, color='#FB7185', line=dict(color='#060913', width=1.5)),
            text=(l['landmine_10'] * 1000.0).map(lambda v: f'{v:.2f}‰'),
            textposition='top center',
            textfont=dict(size=10, color='#FDA4AF'),
            hovertemplate='<b>%{x}</b><br>稳妥踩雷率：%{y:.3f}‰<extra></extra>',
        )
    )
    fig.update_layout(
        yaxis=dict(title='覆盖率 (%)', gridcolor=GRID_COLOR, range=[0, max(1.0, float((m['coverage_10'] * 100.0).max()) * 1.22)]),
        yaxis2=dict(title='踩雷率 (‰)', overlaying='y', side='right', showgrid=False, range=[0, max(0.1, float((l['landmine_10'] * 1000.0).max()) * 1.26)], tickfont=dict(color='#FDA4AF')),
    )
    style(fig, '长尾覆盖率 ↑ 与稳妥防踩雷 ↓', height=355, margin=dict(l=20, r=46, t=46, b=58))
    fig.update_layout(legend=dict(orientation='h', y=-0.18, yanchor='top', x=0.5, xanchor='center'))
    return fig


def rerank_flow(trace_df, movie_titles):
    dedup = trace_df.sort_values('rank_after').drop_duplicates('rank_after')
    df = dedup[dedup['rank_after'] <= 10].head(10).copy()
    if not len(df):
        df = dedup.head(10).copy()
    fig = go.Figure()
    for _, row in df.iterrows():
        mid = int(row['movie_id'])
        title = clean_title(movie_titles.get(mid, {}).get('title') or f'#{mid}')[:11]
        rb, ra = int(row['rank_before']), int(row['rank_after'])
        delta = rb - ra
        color = '#10B981' if delta > 0 else ('#F43F5E' if delta < 0 else '#64748B')
        width = 2.8 if abs(delta) >= 2 else 1.6
        fig.add_trace(
            go.Scatter(
                x=['重排前 (RRF)', '重排后 (TDC-Rec)'],
                y=[rb, ra],
                mode='lines+markers+text',
                line=dict(color=color, width=width),
                marker=dict(size=7, color=color),
                text=[f'#{rb}', f'#{ra} {title}'],
                textposition=['middle left', 'middle right'],
                textfont=dict(size=10, color='#E2E8F0'),
                showlegend=False,
                hovertemplate=f'<b>{title}</b><br>位次：#{rb} → #{ra}<br>分歧指数：{divergence_index(row["controversy_score"])}/100 (原始 C={row["controversy_score"]:.3f})<extra></extra>',
            )
        )
    fig.update_yaxes(autorange='reversed', title_text='排名位次', dtick=2)
    fig.update_xaxes(range=[-0.22, 1.72])
    return style(fig, 'TDC-Rec 非对称风险重排位次变化', height=360, margin=dict(l=34, r=18, t=46, b=24))


def arch_benchmark(wc_df, parallel_df, bench_dict):
    wc_med = wc_df.groupby('engine', as_index=False)['runtime_sec'].median()
    rows = []
    for _, r in wc_med.iterrows():
        name = 'MapReduce WordCount' if r['engine'] == 'mapreduce' else 'Spark WordCount'
        rows.append({'任务': name, '耗时 (秒)': float(r['runtime_sec']), '类别': 'WordCount 对照'})
    for _, r in parallel_df.iterrows():
        if r['stage'] == 'feature_engineering':
            lbl = f"YARN 特征 ({int(r['workers'])} Worker)"
            rows.append({'任务': lbl, '耗时 (秒)': float(r['runtime_sec']), '类别': '逻辑并行实测'})
    df = pd.DataFrame(rows)
    fig1 = px.bar(
        df,
        x='耗时 (秒)',
        y='任务',
        orientation='h',
        color='类别',
        text=df['耗时 (秒)'].map(lambda v: f'{v:.1f}s'),
        color_discrete_map={'WordCount 对照': '#38BDF8', '逻辑并行实测': '#F59E0B'},
        labels={'任务': '', '耗时 (秒)': '端到端耗时 (s)'},
    )
    fig1.update_traces(textposition='outside', cliponaxis=False, textfont=dict(size=11, color='#F8FAFC'))
    fig1.update_xaxes(range=[0, float(df['耗时 (秒)'].max()) * 1.26])
    style(fig1, '离线批计算与 YARN 并行实测', height=315, margin=dict(l=18, r=36, t=46, b=56))
    fig1.update_layout(legend=dict(orientation='h', y=-0.22, yanchor='top', x=0.5, xanchor='center'))

    lat_rows = [
        {'接口': '双向量推荐', 'P50 (ms)': bench_dict['recommendations']['p50_ms'], 'P95 (ms)': bench_dict['recommendations']['p95_ms']},
        {'接口': '语义情境搜片', 'P50 (ms)': bench_dict['scene_search']['p50_ms'], 'P95 (ms)': bench_dict['scene_search']['p95_ms']},
        {'接口': '排雷与平替', 'P50 (ms)': bench_dict['detail_and_twins']['p50_ms'], 'P95 (ms)': bench_dict['detail_and_twins']['p95_ms']},
    ]
    lat_df = pd.DataFrame(lat_rows).melt(id_vars='接口', var_name='分位', value_name='延迟 (ms)')
    fig2 = px.bar(
        lat_df,
        x='接口',
        y='延迟 (ms)',
        color='分位',
        barmode='group',
        text=lat_df['延迟 (ms)'].map(lambda v: f'{v:.1f}ms'),
        color_discrete_map={'P50 (ms)': '#10B981', 'P95 (ms)': '#F59E0B'},
        labels={'接口': '', '延迟 (ms)': '在线响应 (ms)'},
    )
    fig2.update_traces(textposition='outside', cliponaxis=False, textfont=dict(size=11, color='#F8FAFC'))
    fig2.update_yaxes(range=[0, float(lat_df['延迟 (ms)'].max()) * 1.25])
    style(fig2, 'Weaviate + MongoDB 毫秒级在线延迟', height=315, margin=dict(l=18, r=18, t=46, b=56))
    fig2.update_layout(legend=dict(orientation='h', y=-0.22, yanchor='top', x=0.5, xanchor='center'))
    return fig1, fig2


def star_mini(hist):
    fig = go.Figure(
        go.Bar(
            x=['1★', '2★', '3★', '4★', '5★'],
            y=hist,
            marker_color=STAR_COLORS,
            text=[f'{int(v):,}' for v in hist],
            textposition='outside',
            cliponaxis=False,
            textfont=dict(size=10, color='#CBD5E1'),
            hovertemplate='%{x}：%{y:,} 人<extra></extra>',
        )
    )
    fig.update_yaxes(range=[0, max(1, max(hist)) * 1.25])
    fig.update_layout(showlegend=False)
    return style(fig, '1～5 星口碑波形', height=240, margin=dict(l=12, r=12, t=36, b=18))


ML_SHORT_NAMES = {
    'ALS_Dot': '纯 ALS',
    'L2-LR': 'L2-LR',
    'FM': 'FM 分解机',
    'Spark_YARN_GBT': 'Spark GBT',
    'GBDT': 'Hist-GBDT',
    'GBDT+LR': 'GBDT+LR',
}
ML_PALETTE = {
    'ALS_Dot': '#64748B',
    'L2-LR': '#94A3B8',
    'FM': '#A78BFA',
    'Spark_YARN_GBT': '#10B981',
    'GBDT': '#38BDF8',
    'GBDT+LR': '#F59E0B',
}


def ml_ranker_comparison(metrics_df):
    df = metrics_df.copy()
    df['简称'] = df['model'].map(lambda k: ML_SHORT_NAMES.get(k, str(k)))
    rows = []
    for _, r in df.iterrows():
        rows.append({'模型': r['简称'], '指标': '全局 ROC-AUC', '得分': float(r['roc_auc'])})
        rows.append({'模型': r['简称'], '指标': '分组 GAUC (用户内)', '得分': float(r['gauc'])})
    plot_df = pd.DataFrame(rows)
    y_min = max(0.50, float(plot_df['得分'].min()) - 0.06)
    y_max = min(1.0, float(plot_df['得分'].max()) + 0.04)
    fig = px.bar(
        plot_df,
        x='模型',
        y='得分',
        color='指标',
        barmode='group',
        text=plot_df['得分'].map(lambda v: f'{v:.4f}'),
        color_discrete_map={'全局 ROC-AUC': '#38BDF8', '分组 GAUC (用户内)': '#F59E0B'},
        labels={'模型': '', '得分': '测试集排序 AUC'},
    )
    fig.update_traces(
        textposition='outside',
        cliponaxis=False,
        textfont=dict(size=9.5, color='#F8FAFC'),
        hovertemplate='<b>%{x}</b><br>%{fullData.name}：%{y:.5f}<extra></extra>',
    )
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(range=[y_min, y_max])
    style(fig, '非深度学习精排模型对比（vs 纯 ALS）', height=380, margin=dict(l=22, r=18, t=46, b=76))
    fig.update_layout(legend=dict(orientation='h', y=-0.18, yanchor='top', x=0.5, xanchor='center'))
    return fig


def ml_roc_curves(roc_df, metrics_df):
    auc_map = dict(zip(metrics_df['model'], metrics_df['roc_auc']))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode='lines',
            line=dict(color='rgba(148,163,184,0.30)', width=1.1, dash='dot'),
            name='随机基线 (0.500)',
            hoverinfo='skip',
        )
    )

    groups = {k: g.sort_values('fpr') for k, g in roc_df.groupby('model', sort=False)}
    als_grp = groups.get('ALS_Dot')
    best_key = max((k for k in groups if k != 'ALS_Dot'), key=lambda k: float(auc_map.get(k, 0.0)), default=None)

    # Shaded gain band between pure ALS and best ML model
    if als_grp is not None and best_key is not None:
        best_grp = groups[best_key]
        band_x = np.linspace(0.0, 1.0, 80)
        als_y = np.interp(band_x, als_grp['fpr'].values, als_grp['tpr'].values)
        best_y = np.interp(band_x, best_grp['fpr'].values, best_grp['tpr'].values)
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=als_y,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=best_y,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(56, 189, 248, 0.12)',
                showlegend=False,
                hoverinfo='skip',
            )
        )

    dash_map = {
        'ALS_Dot': 'dash',
        'L2-LR': 'solid',
        'Spark_YARN_GBT': 'solid',
        'FM': 'dashdot',
        'GBDT+LR': 'dot',
        'GBDT': 'solid',
    }
    width_map = {
        'ALS_Dot': 2.2,
        'L2-LR': 1.8,
        'Spark_YARN_GBT': 2.0,
        'FM': 2.1,
        'GBDT+LR': 2.4,
        'GBDT': 2.6,
    }
    draw_order = [k for k in ('ALS_Dot', 'L2-LR', 'Spark_YARN_GBT', 'FM', 'GBDT+LR', 'GBDT') if k in groups]
    for k in groups:
        if k not in draw_order:
            draw_order.append(k)

    for model_key in draw_order:
        grp = groups[model_key]
        auc = float(auc_map.get(model_key, 0.0))
        short = ML_SHORT_NAMES.get(model_key, model_key)
        color = ML_PALETTE.get(model_key, '#CBD5E1')
        fig.add_trace(
            go.Scatter(
                x=grp['fpr'],
                y=grp['tpr'],
                mode='lines',
                name=f'{short} ({auc:.4f})',
                line=dict(
                    color=color,
                    width=width_map.get(model_key, 2.0),
                    dash=dash_map.get(model_key, 'solid'),
                ),
                hovertemplate=f'<b>{short}</b> (AUC={auc:.4f})<br>FPR=%{{x:.3f}} · TPR=%{{y:.3f}}<extra></extra>',
            )
        )

    # Embedded inset: ΔTPR gain over pure ALS (pp) in the naturally empty lower-right quadrant
    inset_min, inset_max = 4.0, 10.5
    if als_grp is not None:
        grid = np.linspace(0.08, 0.65, 58)
        als_ref = np.interp(grid, als_grp['fpr'].values, als_grp['tpr'].values)
        all_deltas = []
        for model_key in draw_order:
            if model_key == 'ALS_Dot':
                continue
            grp = groups[model_key]
            delta_pp = (np.interp(grid, grp['fpr'].values, grp['tpr'].values) - als_ref) * 100.0
            all_deltas.append(delta_pp)
            short = ML_SHORT_NAMES.get(model_key, model_key)
            color = ML_PALETTE.get(model_key, '#CBD5E1')
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=delta_pp,
                    xaxis='x2',
                    yaxis='y2',
                    mode='lines',
                    showlegend=False,
                    line=dict(
                        color=color,
                        width=width_map.get(model_key, 2.0),
                        dash=dash_map.get(model_key, 'solid'),
                    ),
                    hovertemplate=f'<b>{short}</b> 相对纯 ALS<br>FPR=%{{x:.2f}} · ΔTPR=<b>+%{{y:.2f}}pp</b><extra></extra>',
                )
            )
        if all_deltas:
            flat = np.concatenate(all_deltas)
            inset_min = max(0.0, math.floor((float(flat.min()) - 0.5) * 2) / 2)
            inset_max = math.ceil((float(flat.max()) + 0.7) * 2) / 2

    als_auc = float(auc_map.get('ALS_Dot', 0.759))
    best_auc = float(auc_map.get(best_key, 0.8296)) if best_key else als_auc
    gain_pp = (best_auc - als_auc) * 100.0

    fig.update_xaxes(title_text='假阳性率 (FPR)', range=[0.0, 1.0])
    fig.update_yaxes(title_text='真阳性率 (TPR)', range=[0.0, 1.02])
    style(fig, '测试集 ROC 曲线与相对纯 ALS 增益放大（89.8 万未见评分）', height=390, margin=dict(l=24, r=18, t=46, b=96))
    fig.update_layout(
        xaxis2=dict(
            domain=[0.44, 0.96],
            anchor='y2',
            range=[0.08, 0.65],
            dtick=0.2,
            tickfont=dict(size=9.5, color='#94A3B8'),
            gridcolor='rgba(148, 163, 184, 0.12)',
            zeroline=False,
            title=dict(text='FPR', font=dict(size=9.5, color='#94A3B8'), standoff=2),
        ),
        yaxis2=dict(
            domain=[0.11, 0.56],
            anchor='x2',
            range=[inset_min, inset_max],
            ticksuffix='%',
            dtick=2,
            tickfont=dict(size=9.5, color='#94A3B8'),
            gridcolor='rgba(148, 163, 184, 0.12)',
            zeroline=False,
        ),
        shapes=[
            dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0.38,
                y0=0.04,
                x1=0.985,
                y1=0.635,
                fillcolor='rgba(10, 16, 30, 0.90)',
                line=dict(color='rgba(56, 189, 248, 0.32)', width=1),
                layer='below',
            )
        ],
        annotations=[
            dict(
                x=0.24,
                y=0.69,
                xref='x',
                yref='y',
                text=f'<b>+{gain_pp:.2f}pp AUC</b>',
                showarrow=True,
                arrowhead=2,
                ax=-26,
                ay=-34,
                arrowcolor='#38BDF8',
                font=dict(size=10.5, color='#E0F2FE'),
                bgcolor='rgba(14, 23, 42, 0.85)',
                bordercolor='rgba(56, 189, 248, 0.45)',
                borderpad=3,
            ),
            dict(
                xref='paper',
                yref='paper',
                x=0.40,
                y=0.615,
                xanchor='left',
                yanchor='top',
                text='<b>🔍 局部放大：各 ML 相对纯 ALS 的 TPR 增益 (ΔTPR)</b>',
                showarrow=False,
                font=dict(size=10, color='#38BDF8'),
            ),
        ],
        legend=dict(orientation='h', y=-0.27, yanchor='top', x=0.5, xanchor='center', font=dict(size=10)),
    )
    return fig


def ml_feature_importance(imp_df):
    df = imp_df.sort_values('importance', ascending=False).head(10).iloc[::-1].copy()
    max_imp = max(0.01, float(df['importance'].max()))
    fig = px.bar(
        df,
        x='importance',
        y='label_zh',
        orientation='h',
        color='importance',
        text=df['importance'].map(lambda v: f'+{v:.4f}'),
        color_continuous_scale=['#1E293B', '#38BDF8', '#10B981', '#F59E0B'],
        labels={'label_zh': '', 'importance': '置换 AUC 重要度 (ΔAUC)'},
    )
    fig.update_traces(
        textposition='outside',
        cliponaxis=False,
        textfont=dict(size=10.5, color='#F8FAFC'),
        hovertemplate='<b>%{y}</b><br>AUC 贡献：+%{x:.5f}<extra></extra>',
    )
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(range=[0.0, max_imp * 1.24])
    return style(fig, 'GBDT 核心特征重要性 TOP 10（置换 AUC 消融）', height=345, margin=dict(l=20, r=48, t=46, b=36))


# Keep backwards-compatible aliases for any external imports
def runtime(frame):
    data = frame.copy()
    data['轮次'] = data['run_id'].astype(str)
    return style(px.bar(data, x='engine', y='runtime_sec', color='轮次', barmode='group'), 'WordCount 实测耗时', height=320)


def metrics(frame):
    return model_comparison(frame)


def landmine(frame):
    order = ['B1', 'B2', 'B3', 'Ours']
    df = frame.set_index('model').reindex([k for k in order if k in set(frame['model'])]).reset_index()
    fig = px.bar(
        df,
        x=df['model'].map(lambda k: MODEL_LABELS.get(k, k)),
        y=df['landmine_10'] * 1000.0,
        color=df['model'].map(lambda k: MODEL_LABELS.get(k, k)),
        text=(df['landmine_10'] * 1000.0).map(lambda v: f'{v:.2f}‰'),
        color_discrete_map={MODEL_LABELS[k]: v for k, v in MODEL_COLORS.items()},
        labels={'x': '', 'y': 'Landmine@10 (‰)'},
    )
    fig.update_layout(showlegend=False)
    return style(fig, '稳妥用户踩雷率对比', height=320)

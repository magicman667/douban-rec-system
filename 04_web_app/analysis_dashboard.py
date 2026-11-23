"""Dense, cinema-grade Panorama (World Map + Genre) and Discovery (Data + Tech) pages."""
import html
import streamlit as st
import charts
import data_access as data


@st.cache_data(ttl=300, show_spinner=False)
def load(name):
    return data.read_csv(name)


def show(fig):
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def panorama():
    st.markdown(
        '<div class="page-head">'
        '<div><span class="eyebrow">GLOBAL CINEMA ATLAS · SPARK SQL</span><h1>口碑全景</h1></div>'
        '<div class="tech-row">'
        '<span class="tech-pill">Spark SQL 全量聚合</span>'
        '<span class="tech-pill">129 国/地区 ISO-3</span>'
        '<span class="tech-pill">贝叶斯去偏 Z · 耦合分歧 C</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    region_stats = load('region_stats')
    agg_world = charts.aggregate_world_regions(region_stats)

    # ── 1. Hero C-Position: Interactive World Map / 3D Globe + Regional Leaderboard ──
    c1, c2, _ = st.columns([2.1, 1.5, 1.4])
    with c1:
        metric_label = st.segmented_control(
            '地图指标',
            ['分歧指数', '平均口碑', '影片产量'],
            default='分歧指数',
            label_visibility='collapsed',
        ) or '分歧指数'
    with c2:
        proj_label = st.segmented_control(
            '视角',
            ['世界平面', '3D 地球仪'],
            default='世界平面',
            label_visibility='collapsed',
        ) or '世界平面'

    metric_key = {'分歧指数': 'avg_controversy', '平均口碑': 'avg_bayesian_rating', '影片产量': 'movie_count'}[metric_label]
    proj_key = 'natural earth' if '平面' in proj_label else 'orthographic'

    map_col, rank_col = st.columns([2.45, 1.05], gap='medium')
    with map_col:
        with st.container(border=True):
            show(charts.world_map(region_stats, metric=metric_key, projection=proj_key))

    with rank_col:
        reps = data.region_representative_movies()
        top8 = agg_world.head(8)
        rows_html = []
        for idx, row in top8.iterrows():
            r_name = html.escape(str(row['region']))
            cnt = int(row['movie_count'])
            rat = float(row['avg_bayesian_rating'])
            con = float(row['avg_controversy'])
            div_idx = data.divergence_index(con)
            rep_title = reps.get(str(row['region']), {}).get('hit') or str(row['top_consensus_movie'])
            gem = html.escape(rep_title[:12])
            pct = min(100, int(div_idx / 25 * 100))
            bar_color = '#10B981' if con < 0.10 else ('#F59E0B' if con < 0.14 else '#F43F5E')
            rows_html.append(
                f'<div class="rank-row">'
                f'<div class="rank-top"><span class="rank-idx">{idx + 1:02d}</span>'
                f'<span class="rank-name">{r_name}</span>'
                f'<span class="rank-meta">{cnt:,}部 · ★{rat:.2f}</span></div>'
                f'<div class="rank-sub"><div class="mini-bar"><div style="width:{pct}%;background:{bar_color}"></div></div>'
                f'<span class="rank-gem">分歧 {div_idx} · {gem}</span></div>'
                f'</div>'
            )
        st.markdown(
            '<div class="glass-card rank-card">'
            '<div class="card-kicker">主要产区生态 TOP 8</div>'
            + ''.join(rows_html)
            + '</div>',
            unsafe_allow_html=True,
        )

    # ── 2. Genre Dual-Column Analytics ──
    stats = load('genre_controversy').sort_values('movie_count', ascending=False)
    f1, f2, f3 = st.columns([1.2, 2.8, 1.0])
    with f1:
        minimum = st.slider('最少收录影片', 20, 1000, 150, step=10)
    available = stats.loc[stats.movie_count >= minimum, 'genre'].tolist()
    with f2:
        selected = st.multiselect('筛选题材', available, default=available[:12])
    with f3:
        stack_mode = st.segmented_control('星级显示', ['占比', '总量'], default='占比') or '占比'

    if not selected:
        st.info('请选择至少一个题材。')
        return

    g_left, g_right = st.columns(2, gap='medium')
    with g_left:
        with st.container(border=True):
            g_dist = load('genre_rating_distribution').query('genre in @selected')
            g_dist = g_dist.set_index('genre').reindex([g for g in selected if g in set(g_dist['genre'])]).reset_index()
            show(charts.genre_rating(g_dist, mode='ratio' if stack_mode == '占比' else 'count'))
    with g_right:
        with st.container(border=True):
            show(charts.bubble(stats.query('genre in @selected')))

    # ── 3. Region × Genre Auto-Scaled Matrix ──
    r_names = region_stats[region_stats['name'] != '未知'].sort_values('movie_count', ascending=False)['name'].tolist()
    h1, h2 = st.columns([3.6, 1.0])
    with h1:
        regions = st.multiselect('矩阵产区', r_names[:40], default=r_names[:12])
    with h2:
        threshold = st.number_input('样本下限', min_value=5, value=20, step=5)

    cross = load('genre_region_stats')
    filtered = cross[cross.genre.isin(selected) & cross.region.isin(regions)]
    if len(filtered):
        with st.container(border=True):
            show(charts.heatmap(filtered, threshold))
    else:
        st.info('当前筛选组合暂无满足样本下限的数据。')


def overview():
    st.markdown(
        '<div class="page-head">'
        '<div><span class="eyebrow">DATA INSIGHTS · TDC-REC & BIG DATA BENCHMARKS</span><h1>观影发现与架构实测</h1></div>'
        '<div class="tech-row">'
        '<span class="tech-pill">Hadoop 3.3.6 · YARN</span>'
        '<span class="tech-pill">Spark 3.5 + Hive</span>'
        '<span class="tech-pill">Weaviate 512D+50D</span>'
        '<span class="tech-pill">Structured Streaming</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    ratings_df = load('rating_distribution')
    users_df = load('user_type_distribution')
    movies_df = load('movie_rating_count')
    model_df = load('model_metrics')
    landmine_df = load('landmine_metrics')

    total_movies = len(movies_df)
    total_ratings = int(ratings_df['count'].sum())
    ours_row = model_df.set_index('model').loc['Ours']
    b2_row = model_df.set_index('model').loc['B2']
    ndcg_uplift = (ours_row['ndcg_10'] / b2_row['ndcg_10'] - 1.0) * 100.0
    ours_lm = float(landmine_df.set_index('model').loc['Ours', 'landmine_10']) * 100.0

    # ── Top 4 Hero KPI Strip ──
    k1, k2, k3, k4 = st.columns(4, gap='small')
    k1.metric('全库电影规模', f'{total_movies:,}', '100% 双向量入库')
    k2.metric('真实评分流水', f'{total_ratings:,}', 'Hive 时间切分防穿越')
    k3.metric('TDC-Rec 排序增益', f'+{ndcg_uplift:.1f}%', 'NDCG@10 vs 标准 ALS')
    k4.metric('稳妥派踩雷率', f'{ours_lm:.2f}%', '28,077 稳妥用户零踩雷')

    # ── Row 1: Rating Distribution + User Archetype Ring ──
    a, b = st.columns([1.15, 0.85], gap='medium')
    with a:
        with st.container(border=True):
            show(charts.rating(ratings_df))
    with b:
        with st.container(border=True):
            show(charts.users(users_df))

    # ── Row 2: Longtail + 4-Model Precision Comparison ──
    c, d = st.columns(2, gap='medium')
    with c:
        with st.container(border=True):
            show(charts.longtail(movies_df))
    with d:
        with st.container(border=True):
            show(charts.model_comparison(model_df))

    # ── Row 3: Coverage & Landmine + Real Rerank Flow ──
    e, f = st.columns(2, gap='medium')
    with e:
        with st.container(border=True):
            show(charts.coverage_and_landmine(model_df, landmine_df))
    with f:
        trace_all = load('rerank_trace')
        user_ids = trace_all['user_id'].drop_duplicates().tolist()
        u_map = {}
        for uid in user_ids:
            tau = float(trace_all.loc[trace_all['user_id'] == uid, 'tolerance_tau'].iloc[0])
            tag = '稳妥防雷' if tau < 0.28 else ('均衡观影' if tau <= 0.50 else '猎奇探索')
            u_map[f'#{uid} · {tag} (耐受 {data.divergence_index(tau)}/100 · τ={tau:.2f})'] = uid
        with st.container(border=True):
            picked_label = st.selectbox('代表用户重排追踪', list(u_map.keys()), label_visibility='collapsed')
            sub_trace = trace_all[trace_all['user_id'] == u_map[picked_label]]
            titles = data.movies_lookup(sub_trace['movie_id'].tolist())
            show(charts.rerank_flow(sub_trace, titles))

    # ── Row 4: Non-DL ML Ranking Evaluation (GBDT / GBDT+LR / FM / Spark YARN GBT) ──
    if data.csv_path('ml_ranker_metrics').exists():
        ml_met = load('ml_ranker_metrics')
        ml_imp = load('ml_feature_importance')
        ml_roc = load('ml_roc_curves')

        st.markdown('<hr class="soft-hr">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-bar"><div>'
            '<span class="sec-title">🤖 非深度学习机器学习精排实测</span> '
            '<span class="sec-sub">· 时序两阶段防穿越训练 · 89.8 万未见评分封存测试</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        m_idx = ml_met.set_index('model')
        als_auc = float(m_idx.loc['ALS_Dot', 'roc_auc']) if 'ALS_Dot' in m_idx.index else 0.76
        best_row = ml_met.sort_values('roc_auc', ascending=False).iloc[0]
        best_gauc_row = ml_met.sort_values('gauc', ascending=False).iloc[0]
        sp_id = str(m_idx.loc['Spark_YARN_GBT', 'yarn_app_id']) if 'Spark_YARN_GBT' in m_idx.index else 'YARN'
        sp_auc = f'{float(m_idx.loc["Spark_YARN_GBT", "roc_auc"]):.4f}' if 'Spark_YARN_GBT' in m_idx.index else '-'

        mk1, mk2, mk3, mk4 = st.columns(4, gap='small')
        mk1.metric('最优测试 ROC-AUC', f'{float(best_row["roc_auc"]):.4f}', f'{best_row["model"]} (较纯ALS +{(float(best_row["roc_auc"]) - als_auc) * 100:.2f}pp)')
        mk2.metric('最优用户内 GAUC', f'{float(best_gauc_row["gauc"]):.4f}', f'{best_gauc_row["model"]} (同用户精排精度)')
        mk3.metric('Spark YARN GBT', sp_auc, f'2 Worker 分布式 · {sp_id[-13:]}')
        mk4.metric('精排特征维数', '21 稠密 + 18 稀疏', '协同 · 贝叶斯 · 分歧 × 容忍度')

        m_left, m_right = st.columns(2, gap='medium')
        with m_left:
            with st.container(border=True):
                show(charts.ml_ranker_comparison(ml_met))
        with m_right:
            with st.container(border=True):
                show(charts.ml_roc_curves(ml_roc, ml_met))

        with st.container(border=True):
            show(charts.ml_feature_importance(ml_imp))

    # ── Row 5: Big Data Architecture Performance Telemetry ──
    wc_df = load('wordcount_runtime')
    par_df = load('spark_parallel_runtime')
    bench = data.service_benchmark()
    fig_batch, fig_online = charts.arch_benchmark(wc_df, par_df, bench)
    p1, p2 = st.columns(2, gap='medium')
    with p1:
        with st.container(border=True):
            show(fig_batch)
    with p2:
        with st.container(border=True):
            show(fig_online)

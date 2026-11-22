"""Headless report PNG export from the same upstream data as the product."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

import data_access as data


def export(output):
    output.mkdir(parents=True, exist_ok=True)
    font = Path(__file__).parent / 'runtime' / 'NotoSansCJKsc-Regular.otf'
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    elif not any('CJK' in item.name or 'WenQuanYi' in item.name for item in font_manager.fontManager.ttflist):
        raise RuntimeError('缺少中文字体。请先运行 setup.sh，或在 runtime/ 放置 NotoSansCJKsc-Regular.otf。')
    plt.rcParams['axes.unicode_minus'] = False
    records = []

    def save(fig, filename, sources, note):
        fig.text(.02, .015, note, fontsize=8, color='#475569')
        fig.tight_layout(rect=(0, .06, 1, 1))
        path = output / filename
        fig.savefig(path, dpi=200, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        records.append({'file': filename, 'sources': [str(data.csv_path(n)) for n in sources], 'scope': note})

    f = data.read_csv('rating_distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(f.rating, f['count'], color=['#f43f5e', '#fb923c', '#94a3b8', '#38bdf8', '#10b981'])
    ax.set(xlabel='评分（星）', ylabel='评分数量（条）', title='全库评分星级分布')
    save(fig, '01_rating_distribution.png', ['rating_distribution'], '范围：train + test（4,169,028 条真实评分）。')

    f = data.read_csv('movie_rating_count').sort_values('rating_count', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(f) + 1), f.rating_count, color='#0284c7')
    ax.set(xscale='log', yscale='log', xlabel='电影序位（降序，对数轴）', ylabel='获评次数（对数轴）', title='电影获评次数长尾分布')
    save(fig, '02_movie_rating_longtail.png', ['movie_rating_count'], '范围：全库 140,502 部电影；对数坐标仅显示获评数 > 0 的影片。')

    f = data.read_csv('wordcount_runtime')
    fig, ax = plt.subplots(figsize=(10, 6))
    f.pivot(index='engine', columns='run_id', values='runtime_sec').plot.bar(ax=ax, rot=0)
    ax.set(xlabel='计算引擎', ylabel='耗时（秒）', title='MapReduce 与 Spark WordCount 实测耗时')
    ax.legend(title='轮次')
    save(fig, '03_wordcount_runtime.png', ['wordcount_runtime'], '同一 ECS、相同输入 1,250,471 条；展示 3 次原始耗时。')

    stats = data.read_csv('genre_controversy').sort_values('movie_count', ascending=False)
    top = stats.head(15).genre.tolist()
    f = data.read_csv('genre_rating_distribution').set_index('genre').loc[top]
    fig, ax = plt.subplots(figsize=(12, 7))
    f.plot.bar(stacked=True, ax=ax, rot=45, color=['#f43f5e', '#fb923c', '#94a3b8', '#38bdf8', '#10b981'])
    ax.set(xlabel='类型', ylabel='训练评分数量（条）', title='不同类型评分结构（电影数最多的 15 类）')
    ax.legend(['1 星', '2 星', '3 星', '4 星', '5 星'])
    save(fig, '04_genre_rating_stacked.png', ['genre_rating_distribution', 'genre_controversy'], '范围：train；多类型电影展开统计。')

    f = stats.head(20)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(f.avg_bayesian_rating, f.avg_controversy, s=f.movie_count / f.movie_count.max() * 1400 + 30, alpha=.65, color='#0891b2')
    for _, r in f.iterrows():
        ax.annotate(r.genre, (r.avg_bayesian_rating, r.avg_controversy), fontsize=9)
    ax.set(xlabel='平均贝叶斯评分（1～5 星）', ylabel='平均分歧度 C（0～1）', title='类型口碑分歧气泡图（Top 20 类型）')
    save(fig, '05_genre_controversy.png', ['genre_controversy'], '范围：train；气泡面积正比于电影数。')

    regions = data.read_csv('region_stats').query('name != "未知"').sort_values('movie_count', ascending=False).head(12)['name'].tolist()
    f = data.read_csv('genre_region_stats')
    f = f[f.genre.isin(top) & f.region.isin(regions)]
    values = f.pivot(index='genre', columns='region', values='avg_controversy').reindex(index=top, columns=regions)
    counts = f.pivot(index='genre', columns='region', values='movie_count').reindex_like(values)
    masked = values.mask(counts < 20)
    valid = masked.stack()
    vmin = float(valid.quantile(0.05)) if len(valid) else 0.0
    vmax = float(valid.quantile(0.95)) if len(valid) else 0.3
    fig, ax = plt.subplots(figsize=(13, 8))
    image = ax.imshow(masked, aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(regions)), regions, rotation=45, ha='right')
    ax.set_yticks(range(len(top)), top)
    ax.set(xlabel='产区', ylabel='类型', title='类型 × 产区平均分歧度热力图')
    fig.colorbar(image, ax=ax, label='平均分歧度 C')
    save(fig, '06_genre_region_heatmap.png', ['genre_region_stats', 'genre_controversy', 'region_stats'], '范围：train；少于 20 部组合留空；色阶按 5%～95% 分位拉伸。')

    f = data.read_csv('user_type_distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(f.user_type_label, f.user_count, color=['#38bdf8', '#f59e0b', '#10b981'])
    ax.set(xlabel='用户类型', ylabel='用户人数（人）', title='用户观影类型分布')
    save(fig, '07_user_types.png', ['user_type_distribution'], '范围：train 用户画像（600,344 名用户）。')

    f = data.read_csv('model_metrics').set_index('model').reindex(['B1', 'B2', 'B3', 'Ours']).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = range(len(f))
    w = 0.36
    b_ndcg = axes[0].bar([i - w / 2 for i in x], f['ndcg_10'] * 1000, width=w, label='NDCG@10 (‰)', color='#0284c7')
    b_rec = axes[0].bar([i + w / 2 for i in x], f['recall_10'] * 1000, width=w, label='Recall@10 (‰)', color='#f59e0b')
    axes[0].set_xticks(list(x), f['model'])
    axes[0].set(ylabel='千分比 (‰)', title='排序精度对比（NDCG / Recall）')
    axes[0].legend()
    axes[0].bar_label(b_ndcg, fmt='%.2f', fontsize=8)
    axes[0].bar_label(b_rec, fmt='%.2f', fontsize=8)
    b_cov = axes[1].bar(f['model'], f['coverage_10'] * 100, color=['#64748b', '#38bdf8', '#a78bfa', '#f59e0b'])
    axes[1].set(ylabel='全库覆盖率 (%)', title='全库长尾覆盖率 Coverage@10')
    axes[1].bar_label(b_cov, fmt='%.2f%%', fontsize=9)
    save(fig, '08_model_metrics.png', ['model_metrics'], '范围：test，185,149 名用户，全库 140,502 选 10（Full-Ranking）。')

    f = data.read_csv('landmine_metrics').set_index('model').reindex(['B1', 'B2', 'B3', 'Ours']).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(f.model, f.landmine_10 * 1000, color=['#64748b', '#38bdf8', '#a78bfa', '#10b981'])
    ax.bar_label(bars, fmt='%.3f‰')
    ax.margins(y=.25)
    ax.set(xlabel='模型方案', ylabel='Landmine@10 (‰，越低越好)', title='四组模型稳妥用户踩雷率对比')
    save(fig, '09_landmine_metrics.png', ['landmine_metrics'], '范围：test；τ < 0.28（28,077 名稳妥防雷派）；Ours 压降至 0.000‰。')

    trace = data.read_csv('rerank_trace')
    sample_uid = int(trace['user_id'].iloc[0])
    sub = trace[trace['user_id'] == sample_uid].sort_values('rank_after').head(10)
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in sub.iterrows():
        color = '#10b981' if row['rank_before'] > row['rank_after'] else ('#f43f5e' if row['rank_before'] < row['rank_after'] else '#64748b')
        ax.plot([0, 1], [row['rank_before'], row['rank_after']], marker='o', color=color, linewidth=2)
    ax.invert_yaxis()
    ax.set_xticks([0, 1], ['重排前 (RRF)', '重排后 (TDC-Rec)'])
    ax.set(ylabel='推荐位次', title=f'TDC-Rec 重排前后位次流向（用户 #{sample_uid}）')
    save(fig, '10_rerank_flow.png', ['rerank_trace'], '来源：Weaviate 双向量实测重排追踪（analysis/03_eval/rerank_trace.csv）。')

    (output / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'charts')
    args = parser.parse_args()
    print(json.dumps(export(args.output), ensure_ascii=False, indent=2))

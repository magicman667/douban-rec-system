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
    # Bundled Noto font is installed by setup.sh. Never silently export missing CJK glyphs.
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
        fig.savefig(path, dpi=180, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        records.append({'file': filename, 'sources': [str(data.csv_path(n)) for n in sources], 'scope': note})

    f = data.read_csv('rating_distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(f.rating, f['count'], color='#0284c7'); ax.set(xlabel='评分（星）', ylabel='评分数量（条）', title='全库评分星级分布')
    save(fig, '01_rating_distribution.png', ['rating_distribution'], '范围：train + test，描述性统计；纵轴为评分条数。')

    f = data.read_csv('movie_rating_count').sort_values('rating_count', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6)); ax.plot(range(1, len(f)+1), f.rating_count, color='#0284c7'); ax.set(xscale='log', yscale='log', xlabel='电影序位（按获评次数降序，对数轴）', ylabel='获评次数（条，对数轴）', title='电影获评次数长尾')
    save(fig, '02_movie_rating_longtail.png', ['movie_rating_count'], '范围：全库电影；零获评电影不显示于对数轴。')

    f = data.read_csv('wordcount_runtime')
    fig, ax = plt.subplots(figsize=(10, 6)); f.pivot(index='engine', columns='run_id', values='runtime_sec').plot.bar(ax=ax, rot=0); ax.set(xlabel='计算引擎', ylabel='耗时（秒）', title='MapReduce 与 Spark WordCount 实测耗时'); ax.legend(title='轮次')
    save(fig, '03_wordcount_runtime.png', ['wordcount_runtime'], '同一 ECS、相同输入 1,250,471 条；展示三次原始耗时，未重新计算上游指标。')

    stats = data.read_csv('genre_controversy').sort_values('movie_count', ascending=False)
    top = stats.head(15).genre.tolist()
    f = data.read_csv('genre_rating_distribution').set_index('genre').loc[top]
    fig, ax = plt.subplots(figsize=(12, 7)); f.plot.bar(stacked=True, ax=ax, rot=45); ax.set(xlabel='类型', ylabel='训练评分数量（条）', title='不同类型评分结构（电影数最多的 15 类）'); ax.legend(['1 星', '2 星', '3 星', '4 星', '5 星'])
    save(fig, '04_genre_rating_stacked.png', ['genre_rating_distribution', 'genre_controversy'], '范围：train；多类型电影可重复归入不同类型，保留上游标签。')

    f = stats.head(20)
    fig, ax = plt.subplots(figsize=(12, 8)); ax.scatter(f.avg_bayesian_rating, f.avg_controversy, s=f.movie_count / f.movie_count.max() * 1400 + 25, alpha=.5, color='#0891b2')
    for _, r in f.iterrows(): ax.annotate(r.genre, (r.avg_bayesian_rating, r.avg_controversy), fontsize=9)
    ax.set(xlabel='平均贝叶斯评分（1～5 星）', ylabel='平均分歧度 C（0～1）', title='类型口碑分歧（电影数最多的 20 类）')
    save(fig, '05_genre_controversy.png', ['genre_controversy'], '范围：train；气泡面积随电影数增加。图中标签保留上游原值。')

    regions = data.read_csv('region_stats').sort_values('movie_count', ascending=False).head(12)['name'].tolist()
    f = data.read_csv('genre_region_stats'); f = f[f.genre.isin(top) & f.region.isin(regions)]
    values = f.pivot(index='genre', columns='region', values='avg_controversy').reindex(index=top, columns=regions)
    counts = f.pivot(index='genre', columns='region', values='movie_count').reindex_like(values)
    fig, ax = plt.subplots(figsize=(13, 8)); image = ax.imshow(values.mask(counts < 20), aspect='auto', cmap='YlOrRd', vmin=0, vmax=1); ax.set_xticks(range(len(regions)), regions, rotation=45, ha='right'); ax.set_yticks(range(len(top)), top); ax.set(xlabel='产区', ylabel='类型', title='类型 × 产区平均分歧度'); fig.colorbar(image, ax=ax, label='平均分歧度 C（0～1）')
    save(fig, '06_genre_region_heatmap.png', ['genre_region_stats', 'genre_controversy', 'region_stats'], '范围：train；电影数最多的 15 类型、12 产区；少于 20 部的组合及缺失组合留空。')

    f = data.read_csv('user_type_distribution')
    fig, ax = plt.subplots(figsize=(10, 6)); ax.bar(f.user_type_label, f.user_count, color=['#0ea5e9', '#a78bfa', '#34d399']); ax.set(xlabel='用户类型', ylabel='用户人数（人）', title='用户观影类型分布')
    save(fig, '07_user_types.png', ['user_type_distribution'], '范围：train 用户画像；600,344 名用户，比例由上游计算。')

    f = data.read_csv('als_model_metrics')
    fig, ax = plt.subplots(figsize=(10, 6)); row = f.iloc[0]; names = ['ndcg_10', 'recall_10', 'coverage_10']; bars = ax.bar(['NDCG@10', 'Recall@10', 'Coverage@10'], [row[n] for n in names], color='#0284c7'); ax.bar_label(bars, fmt='%.6f'); ax.margins(y=.2); ax.set(ylabel='指标值（比例）', title='B2 ALS 单模型实测结果')
    save(fig, '08_b2_model_metrics.png', ['als_model_metrics'], '范围：test，185,149 名用户；Coverage 分母 140,502 部电影；四模型对比待交付。')

    f = data.read_csv('als_landmine_metrics')
    fig, ax = plt.subplots(figsize=(10, 6)); bars = ax.bar(f.model, f.landmine_10, color='#0d9488'); ax.bar_label(bars, fmt='%.6f'); ax.margins(y=.3); ax.set(xlabel='模型', ylabel='Landmine@10（比例，越低越好）', title='B2 ALS 稳妥用户踩雷率')
    save(fig, '09_b2_landmine.png', ['als_landmine_metrics'], '范围：test；τ < 0.28，28,077 名稳妥用户；高争议阈值 C ≥ 0.50；四模型对比待交付。')
    (output / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'charts')
    args = parser.parse_args(); print(json.dumps(export(args.output), ensure_ascii=False, indent=2))

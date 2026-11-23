# TDC-Rec 加权 RRF 备用方案

组员二整理。该方案保存在 `backup/tdc-fusion-v3` 分支，供查看和试用。可以在 GitHub 直接浏览这个分支。已有项目的同学如需本地试跑，可以单独克隆一份。

```bash
git clone --branch backup/tdc-fusion-v3 --single-branch https://github.com/magicman667/douban-rec-system.git douban-tdc-backup
```

## 改动位置

在 TDC-Rec 合并 ALS 和语义候选时，按用户已有的评分条数选择 RRF 参数。B2 的 ALS 模型、B3 的语义检索和原有争议惩罚沿用原实现。

| 用户已有评分 | ALS 权重 | 语义权重 | RRF 的 `k` |
|---|---:|---:|---:|
| 0 条 | 50% | 50% | 60 |
| 1–9 条 | 70% | 30% | 100 |
| 10 条及以上 | 60% | 40% | 60 |

参数在 [fusion_policy.py](../../../03_weaviate_stream_eval/fusion_policy.py)，服务入口在 [rec_service.py](../../../03_weaviate_stream_eval/rec_service.py)。在网页试用时选择「规则重排 (对照组)」，可查看采用加权 RRF 的推荐。网页还会按自身规则筛选和排序。

## 离线结果

在 20,000 名旧样本用户上选参，然后用 50,000 名独立用户对照。相对上一版加强算法，NDCG@10 从 **0.003108** 升到 **0.004284**（+37.8%），Recall@10 从 **0.484%** 升到 **0.599%**（+23.9%）。类型误差从 **0.4662** 升到 **0.5988**。两版在 7,538 名稳妥用户中均未出现踩雷。

[总指标](metrics.csv) · [按评分历史分组](metrics_by_history.csv) · [配对置信区间](paired_deltas.csv) · [调参明细](dev_grid.csv) · [评测脚本](../../../03_weaviate_stream_eval/benchmark_fusion_tuning.py)

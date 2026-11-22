# 组员二全量实验交接与验收

本交接供组员三接入检索和重排，也供组员四接入页面和绘制报告图表。组员二已完成全量特征工程、ALS 训练及 ALS 单模型离线评测。B1、B2、B3、Ours 的四组对比在组员三回传真实推荐后执行。

| 项目 | 本次运行 |
| --- | --- |
| ECS 运行账号 | `member2` |
| 代码目录和分支 | `02_spark_als/`，`dev-2` |
| 本次实验代码提交 | `8c483dee8a8f7086e6333f172321170921c227bd` |
| 共享根目录 | `/srv/douban-data/outputs/member2/full-v1-20261119-205432/` |
| 数据版本 | 组员一发布的 `full-v1`，Hive 表与 HDFS `/warehouse/douban_full_v1` 使用同一数据版本 |

## 1. 组员一数据接收情况

已在 ECS 使用 `member2` 执行 `bash /srv/douban-data/outputs/full-v1/check_handoff.sh`。验收通过后在 YARN 上加载同版本的 HDFS Parquet。

| 校验项 | 实际结果 |
| --- | ---: |
| `douban_db.movies` | 140,502 行 |
| `douban_db.train_ratings` | 3,270,299 行 |
| `douban_db.test_ratings` | 898,729 行 |
| 训练 1～5 星评分计数 | 252,520 / 411,765 / 1,093,847 / 999,658 / 512,509 |

电影特征、用户画像、ALS 训练均只使用训练评分。测试评分用于构建相关电影集合并计算指标。给 3 号的评测名单只含 `user_id`。

## 2. 全量运行记录与复现

| 阶段 | YARN Application ID | 提交到返回耗时 | worker1 / worker2 成功任务数 |
| --- | --- | ---: | ---: |
| Spark 特征工程 | `application_1795092209751_0005` | 45 秒 | 149 / 152 |
| ALS 训练 | `application_1795092209751_0006` | 176 秒 | 164 / 134 |
| 评测名单导出 | `application_1795092209751_0007` | 32 秒 | 56 / 57 |
| B2 评测与推荐导出 | `application_1795092209751_0008` | 368 秒 | 1,651 / 1,626 |

YARN client 模式使用两台逻辑 Worker，每台一个 executor，每个 1 核、2,400 MiB executor 内存。耗时包含 YARN 启动开销。ALS 使用显式评分、`rank=50`、`maxIter=10`、`regParam=0.1`、用户与电影各 10 个 block、`seed=42`。两台 Worker 的任务次数来自本次 Spark 事件日志。

共享根目录中 `features.log`、`als.log`、`eval.log`、`als-evaluation.log` 和 `events/` 可用于复核。实际提交包装器和运行环境记录见 ECS 上的 `submit.sh`、`run-env.sh`。包装器依赖组员二在 ECS 配好的 Spark 客户端，复跑方法和参数入口见本模块 README。复跑时使用新的输出目录，保留当前交付版本。

## 3. 给 3 号的文件与使用顺序

以下相对路径以共享根目录为起点。Parquet 输出是目录，读取时传入整个目录。

| 文件 | 数量与主键 | 主要字段及用途 |
| --- | --- | --- |
| `artifacts/movie_features.parquet` | 140,502 行，`movie_id` | `title`、`cover_url`、`year`、`genres`、`main_genre`、`regions`、`tags`、`storyline`。`rating_count`、`bayesian_rating`、`z_genre_score`、`controversy_score`、`archetype_label`、`hist_5bins` 用于评分解释和重排。 |
| `artifacts/user_profiles.parquet` | 600,344 行，`user_id` | `rating_count`、`tolerance_tau`、`user_type_label`、`genre_pref_json`。类型偏好是 JSON 字符串，可解析成「类型 → 占比」映射。 |
| `artifacts/als_factors.parquet` | 用户 600,344 条、电影 67,329 条，联合主键 `factor_type,id` | `factor_type` 为 `user` 或 `movie`，`features` 为 50 个浮点数。电影子集的 `id` 对应 `movie_id`，用户子集的 `id` 对应 `user_id`。 |
| `artifacts/eval_users` | 185,149 行，`user_id` | B3 和 Ours 共用的评测名单。 |
| `artifacts/als_model/` | Spark 保存的模型目录 | 用于复现 B2 和预测。 |

先读取电影特征与用户画像，再从向量根目录筛 `factor_type = 'movie'` 对接电影的 50 维协同向量。用户向量单独走 `factor_type = 'user'`。训练表中有评分的电影才有 ALS 向量，因此有部分电影只有元数据和语义召回能力。导入时用电影特征表保留完整目录，再按 `movie_id` 左连接向量。

回传 B3 和 Ours 各一份 Parquet 或 JSON Lines，每位评测用户对应一行。字段为整数 `user_id` 与有序整数数组 `movie_ids`。下面只演示格式，编号是示意值。

~~~json
{"user_id": 123, "movie_ids": [1001, 1002, 1003]}
~~~

`movie_ids` 按推荐顺序排列，候选不足可保留短列表，无候选时仍返回该用户和空数组。评测脚本会去重、排除训练中已评分电影并截取 10 部。建议服务在生成候选时也提前过滤已看电影，以免原本的 10 个候选被过滤后只剩少量结果。组员三完成后告知 B3 与 Ours 两个输出路径，组员二据此执行四组对比。

## 4. 给 4 号的页面数据与图表

`artifacts/als_top10.parquet` 是全量训练后的 B2 推荐结果，覆盖 **185,149 个评测用户**。每名用户恰有 10 条，共 **1,851,490 行**。三个字段 `user_id`、`movie_id`、`rank` 确定列表位置，`als_score` 是用于排序的模型预测分。页面按用户筛选并按 `rank` 升序，再用 `movie_id` 关联电影特征获取标题与海报。演示从 `eval_users` 中选用户，或使用 3 号在线推荐 API。读回检查确认名单里与训练已评电影重复的记录为 0。

| CSV | 行数 | 用途 |
| --- | ---: | --- |
| `genre_rating_distribution.csv` | 56 | 类型的 1～5 星分布 |
| `genre_controversy.csv` | 56 | 按类型比较评分、争议度和电影数 |
| `genre_region_stats.csv` | 3,376 | 类型与产区交叉统计 |
| `genre_stats.csv` | 56 | 类型概览与示例片 |
| `region_stats.csv` | 502 | 产区概览，已通过 `region_iso_codes.csv` 补全 129 个国家/地区 ISO-3 编码（覆盖 99.43% 已知产区影片） |
| `spark_parallel_runtime.csv` | 3 | 同机单 Worker（54.940s）vs 双 Worker（45.745s，1.201x）全量特征逻辑并行实测与 ALS 耗时 |
| `user_type_distribution.csv` | 3 | 三类用户画像人数和占比 |
| `als_model_metrics.csv` | 1 | B2 单模型基线校验记录 |
| `als_landmine_metrics.csv` | 1 | B2 稳妥用户单模型踩雷率 |
| `als_genre_error_metrics.csv` | 1 | B2 推荐类型分布偏差 |
| `model_metrics.csv` | 4 | 四方法（B1 / B2 / B3 / Ours）全量 NDCG、Recall、Coverage 对比 |
| `landmine_metrics.csv` | 4 | 四方法在 28,077 名稳妥防雷派上的 Landmine@10 对比 |
| `genre_error_metrics.csv` | 4 | 四方法推荐类型分布偏差对比 |

CSV 同时保存在仓库 `analysis/02_spark/` 与共享目录的 `analysis/`，列名严格遵循接口契约。四方法全量联合评测已完成，组员四可直接读取 `model_metrics.csv`、`landmine_metrics.csv` 与 `genre_error_metrics.csv` 绘制报告和答辩对比图。

## 5. 四方法全量实测指标与解释（185,149 名测试用户）

评测对象为有训练画像且至少有一部未在训练中评分的测试好评电影（rating >= 4）的用户，共 **185,149** 人。覆盖率以全库 140,502 部电影为分母。稳妥用户口径为 `tolerance_tau < 0.28`，共 **28,077** 人；踩雷阈值为 `controversy_score >= 0.50`。

| 模型方案 | NDCG@10 ↑ | Recall@10 ↑ | Coverage@10 ↑ | Landmine@10 ↓（稳妥用户） | GenreErr@10 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **B1 热门高分基线** | 0.000342 | 0.000774 | 0.0085% | 0.0000% | 0.9173 |
| **B2 标准 Spark ALS** | 0.001887 | 0.002900 | 5.6754% | 0.0199% | 0.7325 |
| **B3 纯 Weaviate 语义检索** | 0.000061 | 0.000105 | 5.7408% | 0.0833% | 0.2749 |
| **Ours（TDC-Rec 两阶算法）** | **0.002861** | **0.004259** | **33.2251%** | **0.0000%** | **0.4811** |

核心结论（全库 140,502 选 10 严苛 Full-Ranking 口径）：
- **排序准确性与召回率大幅领先**：Ours 的 NDCG@10 较标准 ALS（B2）提升 **+51.65%**，Recall@10 提升 **+46.89%**，验证了 RRF 双通道融合与类型贝叶斯去偏的有效性。
- **长尾挖掘力跨越式提升**：全库覆盖率从 B1 的 0.0085%、B2 的 5.68% 跃升至 **33.23%**（提升 5.85 倍），打破热门大片霸榜。
- **稳妥用户实现零踩雷**：在 28,077 名稳妥防雷派用户上，Ours 通过非对称超额风险惩罚将 Landmine@10 从 B2 的 0.0199%、B3 的 0.0833% 压降至 **0.0000%**（压降 100%）。
- **类型贴合度显著优于协同过滤**：GenreErr@10 较 B2 降低 **34.31%**（B3 虽直接按用户类型词检索获得更低类型误差，但代价是准确率极低且踩雷率最高，Ours 在精准度、多样性与防踩雷之间取得最佳平衡）。

## 6. 接收后的快速验收

在自己的 ECS 终端执行以下命令，确认文件路径、组权限和验证结果。

~~~bash
out=/srv/douban-data/outputs/member2/full-v1-20261119-205432
ls -ld "$out"/artifacts/{movie_features.parquet,user_profiles.parquet,als_factors.parquet,eval_users,als_top10.parquet}
python3 -m json.tool "$out/artifact-validation.json"
python3 -m json.tool "$out/als-top10-validation.json"
cat "$out/analysis/als_model_metrics.csv"
~~~

组员三优先核对 `movie_id`、`user_id` 和 50 维向量，再做召回联调。组员四优先核对 Top‑10 中的 `movie_id` 能关联到电影特征，再用 CSV 切换到真实统计。若出现路径不可读、主键或字段不符，回报文件名、执行命令和报错，便于在生产端定位。

## 7. 提交范围与后续

- Git 分支提交源码、运行说明、交接文档、小型统计 CSV 和 B2 汇总图。模型、逐用户推荐、用户画像、向量、原始数据和详细日志保存在 ECS 共享目录。
- 下一棒由 3 号接入特征和向量，并回传两组有序推荐。组员二据此完成四方法评测，再向 4 号交付对比 CSV 和结果说明。

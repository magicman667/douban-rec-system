# 02_spark_als — 2号成员专属目录（Spark 特征挖掘与 ALS 协同模型）

> **负责人**：2号成员  
> **开发分支**：`dev-2`  
> **只维护范围**：`02_spark_als/` 与 `analysis/02_spark/`

## 核心研发任务
1. **类型与产区差异统计**：只读 Hive 中的 `train` 训练集，对多类型与合拍片产区执行 `explode` 展开，用 Spark SQL 统计 20+ 种电影类型及全球/华语产区的均分、分歧度与 1～5 星波形分布。
2. **第一阶 TDC-Rec 离线画像并行计算**：
   - 同类型贝叶斯平滑相对分 `z_genre_score` $Z(i \mid g)$
   - “香农熵—双峰”耦合分歧指数 `controversy_score` $C_i \in [0, 1]$
   - 四大口碑象限标签 `archetype_label`（大众稳妥神作 / 高分两极争议 / 小众冷门宝藏 / 低分避雷片）
   - 用户类型偏好 `genre_pref_json` $p_u(g)$ 与争议容忍度 `tolerance_tau` $\tau_u$
3. **Spark MLlib ALS 训练**：训练隐语义模型（`rank=50`），导出用户与电影 50 维协同隐向量 `als_factors.parquet`。
4. **统计交接产物**（输出至 `analysis/02_spark/`）：`genre_stats.csv`、`region_stats.csv`、`genre_rating_distribution.csv`、`genre_controversy.csv`、`genre_region_stats.csv`、`user_type_distribution.csv`。4 号从这些文件绘图。
5. **离线评测**：在时间切分的测试集上，对同一批用户比较 B1、B2、B3 和 Ours 的 Top-10，输出 `model_metrics.csv`、`landmine_metrics.csv`、`genre_error_metrics.csv`。

## 接 1 号并运行特征与 ALS

以 `member2` 登录 ECS 后，先运行 1 号的交接检查。它会校验 Hive 行数和训练 Parquet 的 Spark 聚合。

```bash
bash /srv/douban-data/outputs/full-v1/check_handoff.sh
```

从项目根目录运行以下命令。将 `out` 指向与 3 号约定、`member2` 可写的新版本目录。Spark 集群环境使用 1 号交接的 `spark-submit` 与配置。已发布的 Parquet 快照与 Hive 表使用同一份清洗产物。

```bash
snapshot=/srv/douban-data/outputs/full-v1/parquet
out=/path/to/shared/member2/full-v1

spark-submit --master yarn 02_spark_als/feature_engineering.py \
  --input-parquet "$snapshot" --output "$out" --analysis-dir analysis/02_spark
spark-submit --master yarn 02_spark_als/als_train.py \
  --input-parquet "$snapshot" --output "$out"
```

`--input-parquet` 接受作业和执行器都能读取的路径。YARN 执行器看不到本机挂载的快照时，使用 `hdfs:///warehouse/douban_full_v1`。也可在已有 Hive 元数据配置的 Spark 环境省略 `--input-parquet`，从 `douban_db` 读取。验收时留存作业日志和 Spark DAG。单 Worker 与双 Worker 的性能对比在相同数据和 ALS 参数下运行，分别使用独立输出目录。

| 输出 | 主键和字段 |
| --- | --- |
| `movie_features.parquet` | `movie_id`，含元数据、评分数、贝叶斯均分、Z、C、标签和五档评分数 |
| `user_profiles.parquet` | `user_id`，含训练评分数、容忍度、用户类型和类型偏好 JSON |
| `als_factors.parquet` | `factor_type` 与 `id` 联合标识向量。按 `factor_type` 分区，值为 `user` 或 `movie`，向量列 `features` 长度为 50 |
| `als_model/` | Spark 保存的训练模型，用于产出 B2 |

电影分歧度采用指导书公式，类型均值和标准差按同主类型作品的训练评分加权。无训练评分的电影保留元数据，贝叶斯评分取类型均值，Z 和 C 取零。四类标签依次判断零评分或 Z 为负、C 至少 0.50、评分数少于 20、其余。用户容忍度采用正评电影平均 C 加高争议电影正向打分溢价（除以四星跨度），截断至 1。无正评用户使用中性值 0.39。类型偏好对正评电影的多类型展开后归一化。产区 ISO 可用 `--region-codes` 提供 `region,iso_code` 映射 CSV，未知映射保留空值。

## 与 3 号联调和评测

用 `eval_offline.py --export-cohort 路径` 导出仅含 `user_id` 的评测用户表，交给 3 号。联调建议 3 号把同一批用户的 B3 和 Ours 推荐分别导出为 Parquet 或 JSON Lines，每个用户每个方法一条记录。字段是 `user_id` 和按推荐顺序排列的 `movie_ids`。如无推荐，可提供空数组。下面的命令在三方确认输出位置后使用。

```bash
spark-submit --master yarn 02_spark_als/eval_offline.py \
  --input-parquet "$snapshot" --artifacts "$out" --export-cohort "$out/eval_users"

spark-submit --master yarn 02_spark_als/eval_offline.py \
  --input-parquet "$snapshot" --artifacts "$out" --analysis-dir analysis/02_spark \
  --b3 /path/to/b3_lists.parquet --ours /path/to/ours_lists.parquet
```

B1 按训练贝叶斯评分和评分量排序，B2 使用保存的 ALS 模型。四组统一过滤训练时已评分的电影并截取 Top-10。测试评分至少 4 星作为相关，评测用户须有训练画像及至少一部未在训练中评分的测试好评电影。覆盖率以全量电影表为分母。稳妥用户取 `tolerance_tau < 0.28`，高争议电影暂用 `controversy_score >= 0.50`，踩雷率按 10 个推荐位计。类型误差是推荐类型分布与训练喜好分布的总变差距离。正式跑分前和组员核对推荐回传字段与这些评测口径。

### 单独交付 ALS 推荐与指标

`--als-only` 只评 B2，无需等待 3 号的推荐服务。附加 `--als-recommendations` 可同时导出评测用户的 Top‑10，用于 4 号读取和展示。Parquet 每行是一部候选电影，字段为 `user_id`、`movie_id`、从 1 开始的 `rank`、`als_score`。按用户查询后用 `rank` 排序。已评电影不入榜，可推荐电影不足时名单可以少于 10 部。此名单覆盖评测用户，模型与向量表另行覆盖全部训练用户。

```bash
spark-submit --master yarn 02_spark_als/eval_offline.py \
  --input-parquet "$snapshot" --artifacts "$out" --analysis-dir analysis/02_spark \
  --als-only --als-recommendations "$out/als_top10.parquet"
```

单独评测输出 `als_model_metrics.csv`、`als_landmine_metrics.csv` 和 `als_genre_error_metrics.csv`。四组对比继续用前面的命令。

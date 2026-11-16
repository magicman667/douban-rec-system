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
4. **统计交接产物**（输出至 `../analysis/02_spark/`）：
   - `genre_rating_distribution.csv` + `charts/04_genre_rating_stacked.png`
   - `genre_controversy.csv` + `charts/05_genre_controversy_bubble.png`
   - `genre_region_stats.csv` + `charts/06_genre_region_heatmap.png`
   - `user_type_distribution.csv` + `charts/07_user_type_distribution.png`

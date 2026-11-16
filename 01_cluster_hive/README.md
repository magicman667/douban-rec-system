# 01_cluster_hive — 1号成员专属目录（集群底座与多模入库）

> **负责人**：组长兼 1号（[@magicman667](https://github.com/magicman667)）  
> **开发分支**：`dev-1`  
> **只维护范围**：`01_cluster_hive/` 与 `analysis/01_data/`

## 核心研发任务
1. **3 节点完全分布式集群编排**：使用 Docker Compose 搭建 `1 Master + 2 Workers` 的 Hadoop 3.3.6（HDFS + YARN）+ Spark 3.5 + Hive 3.1.3 集群并完成进程与 Web UI 截图存证。
2. **数据清洗、防穿越切分与多模入库**：
   - 清洗豆瓣 `movies.csv`、`ratings.csv`、`comments.csv`，剔除文本内换行符，将字符串 `USER_MD5` 映射为连续正整数 `user_id`（满足 Spark MLlib ALS 输入要求）。
   - 严格按每位用户打分时间 `RATING_TIME` **前 80% 存为 `train`、后 20% 存为 `test`**（杜绝未来数据穿越）。
   - 切出评分人数最多的 2,000 部电影生成 `data_small` 供全组秒级联调。
   - 结构化表映射入 **Apache Hive**；每部电影高赞 Top-5 五星好评与 Top-5 一星差评写入 **MongoDB**（`douban_db.movie_reviews`）。
3. **Hadoop Streaming MapReduce 基准对比**：编写 Python Mapper/Reducer 统计“电影类型 + 中文标签”词频（WordCount），并与同口径 Spark WordCount 实测运行耗时对比。
4. **全量数据总体统计图表**（输出至 `../analysis/01_data/`）：
   - `rating_distribution.csv` + `charts/01_rating_distribution.png`
   - `movie_rating_count.csv` + `charts/02_movie_rating_longtail.png`
   - `wordcount_runtime.csv` + `charts/03_wordcount_runtime.png`

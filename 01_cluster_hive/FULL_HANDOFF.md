# 1 号全量数据交接

## 接棒前核对

工作分支为 dev-1。ECS 原始数据仍可在项目忽略的 data/ 中找到。生产数据盘挂载点为 /srv/douban-data。执行部署前用 findmnt 确认挂载点。原始数据、用户映射、清洗文件、服务卷、运行日志和含真实评论的证据保存在 ECS。

本次清洗执行了全部下载的全量文件。实际隔离 392 条无效评分，保留 140502 部电影、600344 位评分用户、3270299 条训练评分和 898729 条测试评分。下载源提供的全量 comments.csv 实际解析为 1048575 行，按下载文件的实际行数记录。

## 发布与验收

1. 在数据盘创建 processed/full-v1，迁入已验证的清洗产物并留存 counts.json。重新清洗时执行 python3 01_cluster_hive/clean.py 原始CSV目录 新的输出目录，再运行 verify_split.py。
2. 在仓库根目录执行 docker compose build master，然后执行 bash 01_cluster_hive/deploy.sh。HDFS/YARN、Hive 和 WordCount 证据写入数据盘 outputs/full-v1。
3. 执行 bash 01_cluster_hive/validate_services.sh，导入 MongoDB 全量 Top-5 评论，验证模拟事件写入与读回，将总体统计 CSV 放到相同的交接目录。真实流的 JSON 输入字段为 movie_id、rating 和 event_time。
4. 2 号用自己的 ECS 账号查询 localhost:10000 的 HiveServer2，验收 douban_db.movies、douban_db.train_ratings 和 douban_db.test_ratings。核对上面的行数，并使用 Spark 针对训练表执行一次聚合。绝对路径 /warehouse/douban_full_v1 下保存 Hadoop 文件系统中的 Parquet。
5. 2 号验收通过后开始全量特征和 ALS。MongoDB 默认地址在 ECS 上是 mongodb://127.0.0.1:27017，在容器网内用 mongodb://mongo:27017。Weaviate 由后续模块通过内部服务名 weaviate 访问。

电影、评分和评论 CSV 保持原始文件名；电影和评分读取 UTF-8，短评读取 GB18030。相同用户同一时间的评分使用 rating_id 排序。单条历史评分留在训练集。数据字典直接见 tables.sql，用户映射位于 processed/full-v1/user_mapping.csv。

## 成员账号验收

在 ECS 的成员终端中执行 bash /srv/douban-data/outputs/full-v1/check_handoff.sh。脚本通过本机 HiveServer2 核对三张表的行数，并用 Spark 聚合从这批 Hive 存储位置导出的全量训练 Parquet。共享交接目录包含 parquet/、counts.json 和客户端运行时。执行验收前，先确认该目录的 Parquet 三个子目录均已发布。

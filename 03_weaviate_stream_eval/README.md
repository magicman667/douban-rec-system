# 03_weaviate_stream_eval — 3号成员专属目录（Weaviate 检索、流式预警与离线评测）

> **负责人**：3号成员  
> **开发分支**：`dev-3`  
> **只维护范围**：`03_weaviate_stream_eval/` 与 `analysis/03_eval/`

## 核心研发任务
1. **Weaviate 双命名向量与 MongoDB 容器编排**：使用 `BAAI/bge-small-zh-v1.5` 将“片名+类型+产区+标签+简介”编码为 512 维中文语义向量，与 2 号导出的 50 维 ALS 隐向量一并灌入 Weaviate（持久化挂载 `./weaviate_data` 与 `./mongo_data`）。
2. **Spark Structured Streaming 实时口碑熔断**：编写微批流处理脚本监测模拟实时评分流，滑动窗口内 1 星差评激增时自动更新 MongoDB `stream_alert` 与 `recent_drop_msg` 预警字段。
3. **封装 `rec_service.py` 在线服务接口**：实现第二阶 RRF 融合 + 非对称风险重排、一句话情境排雷搜片、“Weaviate 同类型低分歧平替检索 + 15 行大模型好差评总结（缓存至 MongoDB）”三大契约函数。
4. **独立后台量化评估 `eval_offline.py`**（输出至 `../analysis/03_eval/`）：
   - 对比 `B1 热门高分`、`B2 标准 ALS`、`B3 纯向量检索`、`Ours (TDC-Rec)` 四组消融模型
   - `model_metrics.csv` + `charts/08_model_metrics.png`
   - `landmine_metrics.csv` + `charts/09_landmine_metrics.png`
   - （可选）`rerank_trace.csv` + `charts/10_rerank_flow.png`

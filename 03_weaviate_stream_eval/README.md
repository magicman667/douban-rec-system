# 三号模块 Weaviate 检索与 TDC Rec 推荐服务

本目录接收二号的 `movie_features.parquet`、`user_profiles.parquet` 和 `als_factors.parquet`，生成 512 维 BGE 中文语义向量，导入 Weaviate 双命名向量集合，并向四号提供三个冻结接口。流式预警由一号维护，离线指标由二号统一计算；三号仅回传 B3 与 Ours 的逐用户有序推荐。

## 文件说明

- `core.py`：RRF、非对称风险效用、类型上限和数据对象。
- `vector_import.py`：校验二号 Schema，生成 BGE 向量并建立 `semantic` 与 `collaborative` 双向量集合。
- `live_backend.py`：Weaviate v4 查询、二号用户画像和用户 ALS 向量加载、MongoDB 短评适配器。
- `rec_service.py`：供四号调用的推荐、情境搜索、争议详情和平替接口，并可导出二号评测所需 JSONL。
- `repository.py`：可测试的存储协议和内存实现。
- `tests/`：不依赖数据库和模型下载的算法与接口测试。

## 环境与测试

```bash
cd 03_weaviate_stream_eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

ECS 无 GPU 时应先从 PyTorch CPU 仓库安装 `torch`，避免默认拉取 CUDA 依赖：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`。不要提交 `.env`、Parquet、模型权重或 Weaviate 数据卷。

## 全量向量导入

```bash
python vector_import.py \
  --movie-features /srv/douban-data/outputs/member2/full-v1-20261119-205432/artifacts/movie_features.parquet \
  --als-factors /srv/douban-data/outputs/member2/full-v1-20261119-205432/artifacts/als_factors.parquet \
  --host 127.0.0.1
```

导入前会校验关键字段、50 维 ALS 向量和 512 维 BGE 向量。集合已存在时会重建，正式复跑应先备份或更换集合名。

本次 ECS 实验使用独立 Weaviate v1.40.6，HTTP 为 `127.0.0.1:18080`，gRPC 为 `50052`。语义向量已分批保存到：

```text
/srv/douban-data/outputs/member3/full-v1-20261120/semantic_vectors.parquet
```

复跑时可直接使用 `--semantic-vectors` 读取已验证的 512 维向量，无需重新编码：

```bash
python vector_import.py \
  --movie-features /srv/douban-data/outputs/member2/full-v1-20261119-205432/artifacts/movie_features.parquet \
  --als-factors /srv/douban-data/outputs/member2/full-v1-20261119-205432/artifacts/als_factors.parquet \
  --semantic-vectors /srv/douban-data/outputs/member3/full-v1-20261120/semantic_vectors.parquet \
  --http-port 18080 --grpc-port 50052
```

## 四号调用

```python
import sys
sys.path.insert(0, "/home/member3/work/douban-rec-system/03_weaviate_stream_eval")

import rec_service

weaviate_client, mongo_client = rec_service.configure_live(
    "/srv/douban-data/outputs/member2/full-v1-20261119-205432/artifacts/als_factors.parquet",
    "/srv/douban-data/outputs/member2/full-v1-20261119-205432/artifacts/user_profiles.parquet",
)

result = rec_service.get_user_recommendations(123, top_k=10)
scene = rec_service.search_by_scene("雨天独自看的温暖科幻片", avoid_controversy=True, top_k=10)
detail = rec_service.get_controversy_detail_and_twins(1292052)
```

进程退出前调用 `weaviate_client.close()` 与 `mongo_client.close()`。若 MongoDB 中没有 AI 摘要，当前版本生成可复现的本地正反方摘要并写回缓存，不依赖或泄露大模型密钥。

本次独立服务端口需先设置：

```bash
export WEAVIATE_HTTP_PORT=18080
export WEAVIATE_GRPC_PORT=50052
export HF_ENDPOINT=https://hf-mirror.com
```

## 二号评测回传

`rec_service.export_eval_recommendations()` 输出 JSON Lines，每行固定为整数 `user_id` 与有序整数数组 `movie_ids`。`mode="b3"` 输出纯语义检索，`mode="ours"` 输出完整 TDC-Rec；评测指标仍由二号脚本计算。

```python
rec_service.export_eval_recommendations(user_ids, "b3.jsonl", mode="b3")
rec_service.export_eval_recommendations(user_ids, "ours.jsonl", mode="ours")
```

## 实现口径

融合公式严格使用 `61 × [0.5/(60+rank_ALS) + 0.5/(60+rank_Vec)]`。效用分加入 `0.20 × sigmoid(Z)`、`0.15 × p_u(g)`、超额争议风险的 `1.2 × max(0,C-τ)^1.5` 惩罚以及探索型用户的争议奖励，最终限制单一主类型最多四部。推荐候选池默认至少为 Top K 的 20 倍。

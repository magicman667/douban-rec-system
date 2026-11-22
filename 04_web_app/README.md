# 4 号前端与数据可视化

## 2026 年 11 月 21 日 在线服务与大众化界面更新

网页现在包含“发现电影”“口碑全景”“观影发现”。面向观众的页面不展示 ALS、TDC-Rec、训练/测试集、服务器路径、公式或实验跑分。报告图片导出与后台数据契约继续保留。

- “为你推荐”调用 3 号 `get_user_recommendations`，题材条件传给服务，不再仅过滤离线 Top-10。
- 三种口味从真实评测用户中选择代表 ID，界面明确为体验档案，不冒充登录账户。
- “一句话找电影”调用 `search_by_scene`；搜索结果保留在会话中；安全候选回退时明确提示。
- 电影详情调用 `get_controversy_detail_and_twins`，展示评论、缓存/本地提要、近期口碑提示与相似作品。提要统一称“口碑看点”，不保证由大模型生成。
- 详情的剧情、评分直方图和样本数从 2 号电影特征只读补齐。3 号公开电影字段不含这些信息。
- `service_access.py` 统一管理客户端、锁和缓存。调用失败完整日志写在服务运行日志中，页面只显示易懂提示。

三号家目录对 member4 不可读，ECS 无法访问 GitHub。本次通过本机取得正式提交 `7ab02bd824d8b8d24f0721e67f1899d3c515c9a5`，原版文件存放于 `runtime/member3/`。文件校验见 `runtime/member3/source-manifest.json`。没有修改三号目录或数据库结构，没有重新导入向量。

独立环境准备入口：`bash setup_live.sh`。若 ECS 无法访问 GitHub，在本机克隆 dev-3，在本机运行 `python prepare_service.py --source-only --source-repo <仓库目录>`，将生成的 `runtime/member3/` 上传。模型下载使用与三号相同的 `BAAI/bge-small-zh-v1.5`，固定公开版本，校验记录在模型目录。可设置 `DOUBAN_SERVICE_SOURCE` 与 `BGE_MODEL` 使用团队可读的现有位置。

在线验证：`.venv/bin/python verify_live.py`。客户端在进程结束时关闭。首次配置需要加载全量画像和向量及中文模型，后续复用；不要同时启动多份服务进程。

以下为前一版记录，涉及 ALS 离线大厅与技术评估页的描述已由上述更新替代。当前公众页面不展示后台评测；报告 PNG 导出仍可使用。

当前版本接入 1、2 号的 full-v1 真实交付，在 `dev-4` 开发。只修改本目录，只读共享数据，不修改上游算法、数据或其他成员代码。

## 在 ECS 上启动

现有环境和字体已经配置，可直接运行：

```bash
cd /home/member4/work/douban-rec-system/04_web_app
bash run.sh
```

服务只监听 `127.0.0.1:8501`。在 VS Code Remote-SSH 的“端口”面板转发 8501，然后在浏览器打开本地转发地址。如果已有服务，不要重复启动。检查：

```bash
curl http://127.0.0.1:8501/_stcore/health
```

也可在 Windows PowerShell 建立隧道，保持该终端运行：

```powershell
ssh -N -L 18501:127.0.0.1:8501 bigdata-ecs-member4
```

随后访问 `http://127.0.0.1:18501`。不要在 ECS 上输入 Windows 的隧道命令。

新环境运行 `bash setup.sh` 安装隔离依赖和开源 Noto CJK 字体。运行文件、字体、PNG、浏览器 QA 和虚拟环境不提交 Git。

## 已实现

- 推荐大厅：评测用户示例、自定义用户 ID、训练画像与类型偏好、真实 B2 ALS Top-10、海报、原始名次、题材筛选、电影详情与训练星级分布。
- 类型差异全景馆：多类型选择、训练评分堆叠图、分歧气泡图、类型×产区热力图、低样本遮罩及上游原始概览。
- 数据洞察：全库评分结构、获评次数长尾、WordCount 三轮耗时、用户类型分布、B2 指标与踩雷率、验证记录。
- 9 张中文报告 PNG、来源清单及真实数据集成验证。

题材筛选只过滤既有 Top-10，不生成新推荐。ALS 预测分不作为电影星级显示。示例列表取真实评测用户 ID，并非三个用户类型的精选代表。

## 数据来源

默认只读路径：

```text
DOUBAN_MEMBER1=/srv/douban-data/outputs/full-v1
DOUBAN_MEMBER2=/srv/douban-data/outputs/member2/full-v1-20261119-205432
```

可用同名环境变量切换版本。CSV 来源为 member1 根目录及 member2 的 `analysis/`；Parquet 为 member2 的 `artifacts/`。DuckDB 限制 2 线程、512MB 查询内存，按用户筛选，无需 Spark 服务或数据库账号。页面缓存 5 分钟。

训练特征、类型/产区统计和用户画像来自 train；全库评分分布包含 train+test，仅供描述；B2 推荐效果来自 test。前端只做读取、关联、过滤、图表整形和显示，不重新计算评分、分歧、画像及评测指标。

## 图片与验证

```bash
cd /home/member4/work/douban-rec-system/04_web_app
.venv/bin/python verify.py
.venv/bin/python export_charts.py
```

PNG 位于 `charts/`，来源、筛选范围记录于 `charts/manifest.json`。1～7 是真实统计图；8、9 是 B2 单模型图，不是四组模型对比。绘图脚本仅从既有聚合值绘图，不重新计算模型指标。长尾按原始电影获评次数排序；交互图至多约 5000 点用于控制渲染体积，静态图保留全量点。WordCount 展示三次实测耗时，未在前端重算中位数。

## 待上游交付

1. 3 号 `rec_service.py` 调用方式及完整返回样例：在线 TDC-Rec、情境搜片、AI 口碑总结与平替。
2. 1 号 `mongo_store.py` 可调用源码及运行配置：`get_active_alerts(limit)`。当前 dev-4 只含模块说明，无法验证预警接口；页面显示待接入，不查询或修改 MongoDB 集合。
3. 2 号 `model_metrics.csv`、`landmine_metrics.csv`：同口径四组对比；当前仅 ALS 指标可用。
4. 地图所需可靠 ISO/坐标映射、类型高频标签 CSV。当前产区有异常标签且 ISO 为空，保留原值，不画伪地图或编造词云。

尚未录制最终视频、制作完整报告和 PPT。需在线链路与四方法评测齐备后再做最终取证。

## 文件说明

- `app.py`：页面入口、推荐大厅与电影详情。
- `analysis_dashboard.py`：类型全景及数据洞察。
- `data_access.py`：版本路径、字段校验、只读 Parquet 查询。
- `charts.py`：交互式图表。
- `export_charts.py`：中文静态报告图片及来源清单。
- `verify.py`：真实 CSV/Parquet 与 Streamlit 页面集成验证。
- `setup.sh` / `run.sh`：依赖准备与启动。

## 数据质量边界

上游存在中英文/繁简体类型异名、异常产区标签。页面保留原始口径并允许筛选，不合并类别、不填补缺失度量。热力图低于阈值的组合留空，缺失不补零。B2 指标较低，展示实测值，不推断完整算法效果。

# FootballMind — VOI Agent

面向足球胜平负研究的信息价值智能体：发现来源、获取正文、语义编码、评估候选、采用证据、更新预测，并决定继续或停止。

本仓库是现有项目的协作开发快照，包含前后端、采集适配器、研究与训练代码、测试，以及可离线运行的冻结演示包。没有重新训练或替换模型。

长期开发的统一配置、环境诊断和冻结实验入口见 [研究底座使用说明](docs/development/research-foundation.md)。本机差异写进被 Git 忽略的 `config/workspace.local.json`，路径不再依赖修改在线模型源码。

## 快速启动（冻结历史演示，无需 API Key）

推荐 Python 3.13，当前开发环境为 Windows。首次安装 PyTorch 等依赖需要下载较大的软件包。

```powershell
git clone https://github.com/CHINAITENGINEER/VOI-agent-for-FootballMind.git
cd VOI-agent-for-FootballMind
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m footballmind doctor --mode offline
.\.venv\Scripts\python.exe -m footballmind serve --page defense
```

默认打开 http://127.0.0.1:8766/defense.html 。历史模式使用仓库内冻结模型、正文和已编码向量，不需要重新抓取新闻或下载 Qwen 权重。启动器自动检查冻结包，端口占用时可改用后续端口，以启动输出为准。

## 在线全文分析

```powershell
.\.venv\Scripts\python.exe -m footballmind doctor --mode online
.\.venv\Scripts\python.exe -m footballmind serve --page live
```

默认页面为 http://127.0.0.1:8766/live.html 。在线全文编码还需要本地 Qwen3-Embedding-0.6B 权重，仓库不包含该大型第三方模型。复制 `config/workspace.example.json` 为 `config/workspace.local.json` 并配置 `encoder_dir` 与 `cache_dir`。启动器会先检查本地编码文件与依赖；缺失时不会报告在线就绪。文件就绪检查不执行完整正文推理，来源可用性由实际请求另行记录。详见 [协作与模型环境](docs/COLLABORATION.md)。

在线采集依赖公开来源的可用性。研究代码保留部分旧版 DeepSeek 路径，其凭据只能通过环境变量配置；全文向量 f/g 路径直接进行本地推理，不需要为它设置 DeepSeek Key。

## 目录

| 路径 | 内容 |
|---|---|
| `demo/` | HTTP 服务、最新在线页、冻结研究页、旧版本兼容入口 |
| `football_agent/` | 信息选择、状态、策略、运行与语义协议 |
| `football_research/` | f/g、语义编码、训练、回放、数据适配与评估 |
| `football_platform/` | 信息流与采集日志合同 |
| `football_spider/` | 来源采集器、身份、存储与平台工具 |
| `tools/` | 数据构建、训练和诊断脚本 |
| `tests/` | 自动化回归和本地样例 |
| `config/` | 来源配置与合同证据 |
| `var/releases/footballmind-defense-20261025-rc1/` | 原样保留的冻结模型和离线演示材料 |
| `docs/` | 研究、历史设计和开发文档 |

`demo/README.md` 描述早期回放入口；当前启动方式以本文件为准。历史研究文档保留原日期与路径，不应当作最新产品能力承诺。

## 模型口径

- f 主演示模型为证据集合模型；GRU 是独立架构研究对照。
- 全文 MLP 与 state-Ridge 为不同策略头。state-Ridge 不使用候选正文向量，不能把其策略结果归给全文排序。
- 运行时参数冻结，只更新证据与状态。概率变化不等于预测改善。
- 冻结包的 `bundle-config.json` 区分演示默认策略和开发集选定策略，模型绑定、SHA 与停止规则须一起保留。

## 开发检查

```powershell
python -m pytest -q tests/demo/test_defense_budget.py tests/demo/test_live_candidate_time.py tests/demo/test_live_postread_model.py
python -m demo --help
```

完整测试中部分用例依赖原开发机归档或可选库，不能将本地资源缺失当成算法结果。CI 仅运行不依赖外部采集的针对性测试。

开发入口默认选择 `state_ridge`，它是原开发集完整策略选择项，主要控制继续/停止；全文 MLP 可显式选择。冻结历史演示内部的原默认值保持在原 bundle 中，所有策略仍是研究模型。

## 数据和凭据

没有上传 `.env`、密钥、数据库、完整 E 盘数据、Qwen 权重、后台日志或演示视频。冻结演示包保留少量原始来源正文与出处，供项目研究复现，不将第三方正文声明为本项目原创。完整语料和训练资源由团队单独管理。

协作分支、复现步骤和环境限制见 [docs/COLLABORATION.md](docs/COLLABORATION.md)。

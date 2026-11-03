# 研究底座第一阶段

本轮完成统一路径配置、离线诊断、冻结开发材料回放以及现有服务接线。它为下一轮数据关联、f/g实验提供可复跑入口；没有重训模型、改善数据关联或建立新的盲测集。

## 工作副本

使用 Git 仓库根目录执行下列命令。原比赛材料和 E 盘语料仍保持原样。当前分支为 `feature/research-foundation-v1`，不要在旧非 Git 目录继续产生另一套源码修改。

## 配置

将 `config/workspace.example.json` 复制为 `config/workspace.local.json`。示例字段均有实际用途：

| 字段 | 用途 |
|---|---|
| workspace_root | 相对配置文件所在目录解析，示例 `..` 指向仓库根目录 |
| release_dir | 固定 f/g 模型包，默认工作区 `var/releases/footballmind-defense-20261025-rc1` |
| encoder_dir | 本地 Qwen 快照目录，末级目录必须与固定 revision 相同 |
| cache_dir | 新正文编码缓存 |
| runs_dir | 服务运行、日志目录 |
| experiments_dir | 冻结回放的输出根目录 |
| data_root | 后续数据集入口的统一路径，本轮不重建语料 |
| default_head | `state_ridge`、`fulltext_mlp_seed20261027` 或 `fixed_order` |

除 `workspace_root` 外，相对路径都基于解析后的工作区。配置不会创建目录，不会加载模型或联网。未知字段、拼错的路径字段和未知策略直接报错，避免静默回退。配置优先级为显式 `--config`、`FOOTBALLMIND_CONFIG`、工作区 local 文件、默认目录。local 配置被 Git 忽略。

路径配置不允许改变 f/g SHA 或编码器 revision，这一入口目前专用于既有冻结模型。支持新模型类型属于后续独立版本。

## 诊断和启动

```powershell
python -m footballmind doctor --mode offline
python -m footballmind doctor --mode online
python -m footballmind doctor --mode online --deep
python -m footballmind serve --page defense
python -m footballmind serve --page live
```

也可以 `python -m footballmind --config config/workspace.local.json doctor --mode online`。`--config` 可以放在 doctor/serve/experiment 后。

doctor 总是检查冻结包文件摘要和运行依赖。`--deep` 额外读取大型编码器权重并验证 SHA。默认只检查编码器文件存在与版本目录，不将此检查表述成已执行正文推理。doctor 不探测来源，不下载模型，不训练，报告包含 `weights_loaded=false`、`inference_executed=false` 和 `sources.status=not_probed`。

服务新增 `GET /api/live-g/readiness`，只做本地检查。它与可能刷新赛程的 `GET /api/live-g/catalog` 分开，启动检查不再触发来源发现。启动器确认编码文件和当前配置一致后才复用在线服务；缺失编码器时会立即给出原因，而不是等待一轮采集失败。

在线开发默认使用原开发集选出的状态 Ridge。它不区分当前候选正文，不能用其结果宣称全文排序有效。需要检查全文评分时显式选择 MLP。原冻结演示的默认值、模型权重和停止阈值均未改写。

## 可重跑的冻结实验

```powershell
python -m footballmind experiment plan config/experiments/frozen-replay.example.json
python -m footballmind experiment run config/experiments/frozen-replay.example.json
```

plan 只核验配置、固定 manifest SHA、完整输入文件摘要及比赛角色，不加载模型，不生成输出。示例使用 Brentford–Arsenal 的既有 g_dev 材料，三个策略、最多采用3篇，默认不结算。

run 加载固定 f/g 并复用原推理函数，逐个运行所选比赛与策略，输出配置、输入/代码摘要、版本、完整预测与失败记录。输出目录必须全新且不能覆盖模型输入；想重跑时更换实验ID和输出目录。运行期间阻断 socket 连接与 DNS，不会在线补文章或下载权重。失败的 after/gain 保持 null，退出码为非零。

配置省略 `case_ids` 时使用指定角色中的全部比赛。`settle=true` 时，先完成所有策略预测，再通过既有独立结算入口读取赛果；这些结果属于历史开发诊断，不能称为新独立测试。

冻结开发包是160场 g_dev 演示数据。这一命令没有接入全部1,520场研究数据，也不是新训练命令。全量数据入口、滚动时间验证和下一轮 f/g 训练将在后续阶段基于本轮配置扩展。

## 验证

```powershell
python -m pytest -q tests/foundation tests/demo/test_defense_api.py tests/demo/test_defense_budget.py tests/demo/test_live_candidate_time.py tests/demo/test_live_postread.py tests/demo/test_live_postread_model.py
```

配置、诊断与实验单测不需要外部模型/API；真实 Qwen 集成测试在本地模型未安装时显式跳过。不要把跳过写成完成了真实编码验证。

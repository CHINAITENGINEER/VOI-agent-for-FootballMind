# FootballMind 独立开发交接文档

版本：`handoff-20261103-v1`。本文面向没有参与过原项目、拿走代码后自行开发和维护的人。无需提交原仓库 PR，也无需原项目的 API Key、数据库或 E 盘，便可完成下面的冻结推理复现。

## 1. 先明确这份项目交付包含什么

FootballMind 研究状态条件下的信息选择：公开来源形成正文候选池，g 结合已有证据和预测状态评分，策略决定采用或停止，f 用已采用证据更新胜平负概率。

当前仓库同时保存了研究代码和冻结推理包。可复现范围如下：

| 能力 | 仅凭仓库是否可以完成 | 资源条件 |
|---|---|---|
| 本地历史前端、冻结 f/g 推理 | 可以 | 安装固定 Python 依赖 |
| 示例比赛、三种策略的数值复现 | 可以 | 下面的 run 和 verify 命令 |
| 全部160场 g_dev 开发样本回放 | 可以 | 复制实验配置并省略 case_ids |
| 历史示例的独立赛果结算 | 可以 | 显式设置 settle=true，预测全部结束后再结算 |
| 新正文的本地语义编码 | 需要补充资源 | 指定版本 Qwen 权重 |
| 实时赛程和新闻获取 | 依赖外部来源 | 联网、来源可用、在线编码文件就绪 |
| 重训原1,520场正文实验 | 仓库不能单独完成 | 缺完整语料、全量输入表和向量等资源 |

冻结包包含 **160场 g_dev 开发比赛、342篇正文、预计算向量、冻结 f/g 权重**。这些是开发演示材料，不是新盲测集。项目曾整理的1,520场、3,001篇正文以及24,911场结构化比赛，不能写成已全部包含在这个仓库中。

## 2. 获取固定版本并独立维护

```powershell
git clone --branch handoff-20261103-v1 --single-branch https://github.com/CHINAITENGINEER/VOI-agent-for-FootballMind.git FootballMind-independent
cd FootballMind-independent
git switch -c my-development
```

tag 用于固定交接版本，避免以后 main 更新导致复现输入变化。独立维护可保留本地历史；如果不需要关联原仓库，可以执行：

```powershell
git remote remove origin
```

之后按自己的仓库地址设置新的 origin。无需加入原仓库协作者，也不要求回传改动。现有文本、模型与第三方依赖的许可各自保留；仓库没有统一开源许可证，本文不代替再分发授权。

## 3. 十分钟阅读顺序

1. 先完成第4节命令，得到数值校验 PASS。
2. 看第6节，理解 f、g、策略和冻结代码的分工。
3. 根据目标选择第8节在线采集，或第9节自建数据与训练。
4. 开始改代码前阅读第10节扩展点，避免改错实际加载的实现。

## 4. 从干净克隆开始复现

原验证环境为 Windows、Python 3.13，推理可在 CPU 上运行。无需 CUDA、Qwen 权重或 API Key。首次安装 PyTorch 等软件包仍需要联网下载；这里的“离线复现”指安装依赖后的实验过程不访问来源/API。

在仓库根目录执行，不需要激活虚拟环境或修改系统执行策略：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-replay.txt pytest==9.1.1
.\.venv\Scripts\python.exe -m footballmind --config config/workspace.example.json doctor --mode offline
.\.venv\Scripts\python.exe -m footballmind --config config/workspace.example.json experiment plan config/experiments/frozen-replay.example.json
.\.venv\Scripts\python.exe -m footballmind --config config/workspace.example.json experiment run config/experiments/frozen-replay.example.json
.\.venv\Scripts\python.exe -m footballmind --config config/workspace.example.json verify var/experiments/frozen-development-replay-v1/predictions.jsonl
```

显式使用 example 配置，可以排除原开发机环境变量或本机 local 配置的影响。该配置全部采用工作区相对目录，无需创建 E 盘。

预期成功条件：

- doctor：`offline_ready=true`，冻结包364个文件摘要无失败。
- plan：1场比赛、3个策略、最多采用3篇，不加载模型，不创建实验输出。
- run：`status=COMPLETED`、`completed_predictions=3`、`failed_predictions=0`。
- verify：`status=PASS`、`reference_predictions=3`、`matched_predictions=3`。
- 回放报告：`network_requests=0`、`training_calls=0`、`outcomes_used_in_prediction=false`。

doctor 中 `online_artifacts_ready=false` 是未安装可选编码器时的正常结果，不影响本节。

标准答案保存在 [`docs/reproduction/reference-predictions-v1.json`](docs/reproduction/reference-predictions-v1.json)。概率绝对容差为 `1e-6`，采用文章、停止原因、预算与模型 SHA 必须一致。UUID、运行耗时和本地路径不作为相同结果的判据。verify 只比较标准文件列出的3条记录；输入中有更多比赛时，会明确报告额外未比较行数。

### 示例结果便于人工核对

比赛：Brentford–Arsenal，2025-01-01；以下概率顺序始终是主胜、平、客胜。

初始 f：`[0.12954601645469666, 0.19393886625766754, 0.6765151619911194]`。

| 策略 | 采用篇数 | 最终主胜 | 最终平局 | 最终客胜 | 停止原因 |
|---|---:|---:|---:|---:|---|
| state_ridge | 0 | 0.1295460165 | 0.1939388663 | 0.6765151620 | predicted_gain_below_frozen_threshold |
| fixed_order | 3 | 0.1352660805 | 0.2003333420 | 0.6644005775 | budget_exhausted |
| fulltext_mlp_seed20261027 | 1 | 0.1340866536 | 0.2000818700 | 0.6658314466 | predicted_gain_below_frozen_threshold |

Ridge在此场直接停止是保存的策略行为。MLP采用的是月度最佳球员投票正文；概率变化不能解释为预测改善。这份样例用于验证链路和数值，没有筛成“预测成功案例”。

### 打开前端

```powershell
.\.venv\Scripts\python.exe -m footballmind --config config/workspace.example.json serve --page defense
```

默认地址为 `http://127.0.0.1:8766/defense.html`。端口已被占用时以启动器输出为准。前端可切换比赛、策略和采用上限，查看正文与状态更新。前端可选20篇上限；复现本节或原预算3比较时，必须选3，不能把不同预算的结果直接混比。

需要便于停止的前台进程时运行：

```powershell
.\.venv\Scripts\python.exe -m demo --config config/workspace.example.json --host 127.0.0.1 --port 8780
```

该进程用 Ctrl+C 停止。serve 使用后台启动器，日志写到配置的 `runs_dir/server`，`last-start.json` 保存PID和实际地址。

### 重跑与批量

原输出目录存在时，run会拒绝覆盖。复制实验配置，修改 `experiment_id` 和 `output_dir` 后重跑。删除 `case_ids` 字段可以使用冻结角色的全部160场；三个策略共480条预测。运行耗时主要还包含每次冻结包验证，不代表模型自身的纯推理速度。

默认 `settle=false`。若研究历史结果，可在新实验配置中设为true；系统完成所有预测后再读取赛果，写独立 settlements 文件。不要把赛果、after或gain加入预测输入。

## 5. 文件、版本与输出

| 路径 | 实际职责 |
|---|---|
| `footballmind/config.py` | 路径解析、策略默认值、当前固定模型标识 |
| `footballmind/doctor.py` | 文件、依赖与摘要诊断，不执行正文推理 |
| `footballmind/experiment.py` | 冻结开发包的配置驱动回放 |
| `footballmind/reproduction.py` | 标准预测的独立数值校验 |
| `demo/server.py` | 本地HTTP路由与应用装配 |
| `demo/static/defense.*` | 冻结研究前端 |
| `demo/static/live.*` | 在线全文前端 |
| `demo/live_postread.py` | 在线采集、预取、编码、决策与采用过程 |
| `demo/live_postread_model.py` | 本地编码与冻结f/g适配 |
| `demo/live_sources.py`、`live_*_body.py` | 来源入口与正文适配 |
| `football_research/acquisition/` | 可修改的研究算法实现 |
| `tools/` | 数据构建、训练、评估等历史研究脚本 |
| `var/releases/footballmind-defense-20261025-rc1/` | 保留字节与摘要的冻结模型、正文和内部实现 |

主要实验输出包括 `config.json`、`plan.json`、`provenance.json`、`predictions.jsonl`、`failures.jsonl`、`summary.json`、`artifacts.json`。`plan`记录比赛、角色、预算和输入摘要；`provenance`记录解释器、依赖及代码摘要。成功与失败分别存放，失败没有有效after/gain。

## 6. f、g到底如何工作

### f：证据条件下的胜平负预测器

当前冻结主模型为 `set_residual_mlp`，共26,823参数，其中26,819为正文训练阶段可训练参数。固定的全文编码器不计入这一参数量。它使用先验、正文向量和元数据构建证据表示，输出三个概率。

研究代码另有GRU架构对照。集合模型汇总采用集合；GRU逐篇更新隐藏状态。输入增加不等于模型参数被更新，运行时两者都固定参数。

研究接口见 `football_research/acquisition/sequential_evidence_f.py` 的 `SequenceEvidenceNetwork`、`FrozenF.predict_prefixes()`。

### g：针对某一固定f的信息价值模型

目标标签为：`gain = log(p_after[y]) - log(p_before[y])`。正值表示该次已实现的log loss下降。这个值是历史监督，不能进入预测输入，也不是文章永久有用/无用的真值。

当前完整特征共1140维，其中状态349维，候选正文256维，差值256维，逐元素乘积256维，候选元数据22维，最大已采用相似度1维。状态还包含已采用正文的平均向量和当前f隐藏状态。

- `state_ridge`：只用状态部分；不读取当前候选向量，同状态候选同分，主要体现继续/停止控制。
- `fulltext_mlp_seed20261027`：1140→128→64→1，154,369参数，读取候选正文及其与状态的关系。
- `fixed_order`：按固定来源顺序采用，不使用g评分。

新开发入口默认state-Ridge；冻结历史展示内部默认全文MLP。两者默认值各有记录，不要把state-Ridge结果归给全文排序。

### 策略与推理边界

在线 `evaluate()` 输出评分和推荐，不自动采用；`predict()` 只对明确传入的已采用正文计算。只有成功的采用步骤才能形成有效after。g看过的候选池与f使用的集合分开保存。来源获取和编码已经发生，不能计作后续筛选节省的成本。

### 很容易改错的位置

**冻结演示与当前在线适配实际加载包内 `src/fm_defense_core/`，不是直接加载 `football_research/acquisition/` 的同名实现。** 因此修改研究源码不会自动改变冻结运行结果。

阅读顺序：`demo/live_postread_model.py:_load_release()` → 包内 `footballmind_defense.py:FreezeApplication` → 包内 `src/fm_defense_core/`。开发新算法应在研究模块实现，验证后构建新模型包和适配器；不要直接改旧包文件再篡改manifest以“通过复现”。

## 7. 当前结论的范围

原研究主策略在320场测试角色上的Log Loss点估计：state-Ridge 1.021563，固定顺序1.022446，随机1.022450；平均f更新次数约0.70625对1.85。这是状态停止控制结果，更新次数下降不等于总成本同比下降，也没有证明稳定优于随机。

全文MLP训练拟合较强，开发/测试尚未胜过零收益回归基线。原测试角色有既往开发暴露，历史正文版本和来源时钟也有未知。不要将这些数字作为全新盲测或已建立的全文排序优势。

本仓库的160场开发演示只能复现其包含的推理结果，不能重算上述320场主表。要重算主表，必须另外取得原评估数据、向量、分组和报告输入。

## 8. 可选的在线采集与新正文编码

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item config/workspace.example.json config/workspace.local.json
```

自行准备 `Qwen/Qwen3-Embedding-0.6B` 的快照 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`，把实际快照目录写进 `encoder_dir`，缓存位置写进 `cache_dir`。快照末级目录与revision一致；诊断不会自动下载。原生向量1024维，当前f/g取前256维再L2归一化，编码器采用冻结的全文合同。

```powershell
.\.venv\Scripts\python.exe -m footballmind --config config/workspace.local.json doctor --mode online --deep
.\.venv\Scripts\python.exe -m footballmind --config config/workspace.local.json serve --page live
```

在线全文f/g路径不要求DeepSeek。旧API语义路径仍保留，需要开发时单独阅读其文档，不能把旧模型Key当成新全文入口前置条件。

`/api/live-g/readiness`只检查本地状态；`/api/live-g/catalog`可能刷新赛程；真正开始在线运行会请求公开来源。模型文件就绪不代表网站可访问、正文提取成功或预测有效。站点变化、网络失败和空正文需要保留真实状态。

## 9. 自己准备数据和重新训练

统一 `data_root` 当前只是配置入口，尚未把所有旧数据脚本重构为可迁移流程。`tools/build_full_history_sequence_dataset.py` 等仍包含原机器路径和旧实验依赖。不要直接使用默认参数运行，或把历史路径缺失解释为没有训练数据。

主要训练表的合同从 `football_research/acquisition/sequential_postread_g.py:EvidencePanel` 和训练脚本实际读取逻辑确认：

| 文件 | 关键内容 |
|---|---|
| `schema.json` | `document_meta_names`，元数据名称和顺序 |
| `matches.jsonl` | 唯一physical_match_id、split_role、match_date、prior等 |
| `documents.jsonl` | document_id、text_sha256、原文引用、来源、时间等 |
| `sequences.jsonl` | physical_match_id、split_role、eligible、document_ids、document_meta、逐比赛metadata_context |
| `outcomes.jsonl` | 独立赛果；outcome_class为0主胜、1平、2客胜 |
| 向量NPZ或缓存目录 | text_sha256索引的1024维float32有限向量，带编码合同与摘要 |

表不是只凑几个列名就可以用。元数据会依据实际已有证据重新计算，向量必须与原文SHA相配，分组不能跨比赛混入。先阅读读取函数，再建立自己的版本化数据，保留缺失和失败。

训练主线代码：

1. `tools/train_full_history_sequence_f.py`：f训练、模型选择、冻结和评估；读取 `--data`、`--embeddings`、`--output` 等显式路径。
2. `tools/train_full_history_sequence_g.py`：固定f后生成对应gain并训练g；读取 `--data`、`--vectors`、`--f-model`、`--output`。
3. `tools/evaluate_full_history_sequence_policies.py`：固定模型与候选池下的策略对照；需要与训练绑定一致的输入。

这些是研究脚本，不属于本交接版“一条命令重训全部实验”的保证。除数据外，还需要补齐研究绘图等依赖并处理旧来源绑定。不要复制旧f的gain当新f标签；不同f版本必须重新计算before/after。不要按收益正负删样本，也不要把大量同场转移当作大量独立比赛。

## 10. 独立开发的推荐切入点

| 想做什么 | 从哪里改 | 怎么验证 |
|---|---|---|
| 改历史前端与证据展示 | `demo/static/defense.*`、`demo/defense.py` | 冻结样例可运行，概率仍来自后端 |
| 改在线状态与交互 | `demo/static/live.*`、`demo/live_postread.py` | 区分获取、编码、评分、采用和失败 |
| 增加来源 | `demo/live_sources.py`、正文projection适配器 | 缓存真实HTML回归，不把目录当文章 |
| 改研究f | `sequential_evidence_f.py` 与对应训练脚本 | 固定数据和分组，比较无正文、集合与顺序模型 |
| 改g或停止规则 | `sequential_postread_g.py` 与训练/策略脚本 | 分别测排序、停止和组合，加入同预算随机 |
| 接自己的模型 | 新模型版本、模型包与adapter | 旧包复现仍通过；新包有自己SHA、schema与测试 |

当前 `footballmind/config.py` 和在线 adapter 固定了旧模型SHA/revision。支持新模型时需要明确增加版本绑定和加载规则，不能仅把新文件放到旧路径期待自动生效。

本地入口为回环地址开发服务，运行记录和任务恢复仍是研究原型形态。将其部署为多人公网服务，需要另行实现认证、持久化任务及部署运维。

## 11. 测试与常见问题

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/foundation tests/demo/test_defense_api.py tests/demo/test_defense_budget.py tests/demo/test_live_candidate_time.py tests/demo/test_live_postread.py tests/demo/test_live_postread_model.py
```

未安装Qwen时，真实编码集成测试会跳过，不是失败，也不能写成完成真实编码验证。当前CI额外运行真实冻结样例与数值verify。原全量测试中部分模块需要外部归档或采集依赖，不能把本交接版针对性通过说成全项目测试通过。

| 症状 | 处理方式 |
|---|---|
| Dependency differs | 使用规定环境：numpy2.3.5、torch2.11.0、scikit-learn1.8.0、scipy1.16.3、joblib1.5.3、threadpoolctl3.6.0。改依赖属于新环境验证，不修改校验器假装一致 |
| Use a new experiment output directory | 修改实验配置的ID和输出目录，不覆盖旧结果 |
| encoder_missing或online_artifacts_ready=false | 离线复现不受影响；在线编码另配Qwen |
| Frozen bundle integrity failed | 恢复固定tag中的原文件，检查换行或传输损坏 |
| 修改研究算法但前端没变 | 查看第6节，实际运行的是冻结包内部实现 |
| 某场采用0篇 | 查看策略评分与stop_reason，不能只凭0篇判断故障 |
| 找不到原E盘语料 | 仓库未包含全量资源，改用自建数据或独立补充的数据包 |

复现通过的标准是固定输入和模型产生一致选择与概率。预测有效性则需要独立比赛上的完整策略评估，这是两项不同的验收。

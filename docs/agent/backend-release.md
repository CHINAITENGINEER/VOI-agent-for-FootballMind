# FootballMind 最终后端操作说明（Round182）

交付范围：用户指定比赛，系统自动生成和选择信息动作，保存原始返回、证据、成本、
重决策及终止原因，输出分析报告。当前真实部署是三场经过官方身份核验的比赛；
支持范围由部署配置声明。没有要求用户先提出分析问题或挑选伤病、天气等特征。

## 运行环境和入口

在项目目录或解压后的发布包目录运行，验证环境为 Windows、Python 3.13。
`football_agent/requirements-frozen.txt` 固定了本次实际使用的后端依赖及传递依赖。
需要新环境时先创建独立虚拟环境，再执行：

```powershell
python -m pip install -r football_agent/requirements-frozen.txt
python -m football_agent.footballmind matches
```

Round182 在全新、禁止继承系统包的 Windows / Python 3.13 虚拟环境中按固定依赖安装。
Windows 解析中国原始时间还需要 `tzdata`，已纳入冻结依赖；上一版已有环境掩盖了这项
遗漏。本次是同一台机器上的隔离安装验证，未声称跨系统或全新物理机器验证。
后端不需要启动网页、
PostgreSQL、Codex CLI 或配置 LLM 密钥。旧 `Start-FootballMind.cmd` 是保留的历史
前端启动方式，本次入口为上述模块命令。

## 指定一场比赛

```powershell
python -m football_agent.footballmind analyze --match "克罗地亚 vs 西班牙" --output var/my-analysis-001
```

也可使用完整比赛 ID，例如 `spftll:match:2981455`。`matches` 列出原名、官方英文名、
初态冻结时间、开球时间、预测截止时间以及此刻是否可采集。只接受完整 ID 或明确的
主客队名称组合；比赛过期、名称歧义、身份或原始数据校验失败时明确报错。
每次运行必须使用新的输出目录，避免覆盖之前的归档。
完成时终端会直接打印 Markdown 和 JSON 报告位置，`result.json.artifacts` 同时保存
入口。每次新采集有独立 `collection_id` 和运行 ID，同场同配置再次运行也不会覆盖
或挤掉另一条轨迹；独立比赛数仍按比赛 ID 去重。

当前计划的三场比赛开球时间为 2026-10-06 14:00 UTC、18:45 UTC、18:45 UTC，
初始概率是 11:53:56 UTC 冻结的市场报价投影。运行时会显示这份快照的时间，
不会将它解释成刚更新的赔率。超过各自赛前截止时间后只能重放既存记录；新增比赛
需要在后台走现有采集、初态生成、官方身份绑定流程，再更新受 SHA 约束的配置。
当前包不代表已经实现任意比赛名称的开放实体解析和自动注册。

默认每场最多三次请求，来源和预算在 `config/footballmind.json` 中配置。
这条真实路径尚无经真实文本轨迹训练验证的数值 VoI 模型，采用 UNKNOWN VoI 下的
有限探索。它能从实际返回产生下一步候选，但不能把探索行为解释成模型已证明
该请求有正收益；未建模的文字不会凭空改变概率。

## 重放自己的新分析及冻结案例

精确重放刚才产生的分析，指定它的输出目录：

```powershell
python -m football_agent.footballmind replay --run var/my-analysis-001 --output var/my-replay-001
```

`--run` 会核对该次分析的部署快照、比赛身份、单场计划、创建时间、运行配置及原始归档，
不会猜测最新运行。它与 `--match` 互斥。使用下面的 `--match` 命令则明确选择部署自带
的冻结案例，不会自动选择自己刚产生的运行：

```powershell
python -m football_agent.footballmind replay --match "克罗地亚 vs 西班牙" --output var/replay-croatia-001
python -m football_agent.footballmind replay --match "苏格兰 vs 斯洛文尼" --output var/replay-scotland-001
```

回放从原始归档重新解析正文、重新生成候选并执行原决策，核对整个运行及分析报告。
`result.json` 说明运行模式；内部 `replay/analysis.json` 和 `replay/report.md` 是分析产物，
`replay/acceptance.json` 是一致性验收。回放零网络，不是新预测。

真实三场归档分别保留正文读取、目录链接发现、无结果检索、失败请求和实际追问。
其中只有两份正文含目标比赛实质信息，概率均未更新。失败没有从展示中删除。

## 演示自主信息选择的算法机制

```powershell
python -m football_agent.footballmind mechanism --case two-step --output var/mechanism-two-001
python -m football_agent.footballmind mechanism --case one-step --output var/mechanism-one-001
python -m football_agent.footballmind mechanism --case one-request --output var/mechanism-budget-001
python -m football_agent.footballmind mechanism --case actual-no-link --output var/mechanism-empty-001
```

四个案例共用同一套运行、预算、信息更新和原始返回回放机制，均为明确标记的合成
受控案例：两步策略可为后续正文价值先读无直接预测价值的目录；单步策略不读；
只剩一次请求时不买无法完成的链；目录实际没返回链接时，不执行模型假想的 URL。
它们证明机制行为，不提供真实足球收益结论。

Round179 原集成演示曾依赖测试替换初态模型。Round181 将确定性原模型恢复为独立
冻结文件，严格核对其策略声明后回放。保留原运行字节；恢复不产生新的独立实验样本。

## 官方赛果接入

```powershell
python -m football_agent.footballmind settle --output var/settlement-001
python -m football_agent.footballmind settle --run var/my-analysis-001 --output var/my-settlement-001
```

此命令实际联网取得官方终场状态和同事件比分并进行审计，不接受用户手写结果替代
官方核验。未完赛、延期、来源失败或比分冲突保留各自状态；拿到真实标签也不会自动
调参、训练或部署新模型。当前仅实现该部署比赛所属赛事的常规时间结果规则。
不带 `--run` 时审计部署自带的三场冻结运行；指定 `--run` 时只审计该次分析，不能把
另一条旧运行的结算误认成自己的新运行已经结算。
没有启动轮询或后台任务，不需要等赛事结束才能验证本次交付。

## 实验复核及发布包

```powershell
python -m football_research.evaluation.release_evidence verify --root . --output var/parallel/agent/round181/evidence
python -m football_agent.release_bundle verify --root . --output var/integrity-check-001.json
python tools/verify_footballmind_release.py --root . --output var/release-check-001
```

后两个命令在带有 `release-manifest.json` 的冻结包中使用。独立验证脚本拒绝网络请求，
检查所有项目模块均从给定发布包加载，并运行三个真实回放、四个机制案例和实验重算。
其输出目录应是新的目录。发布包的散列清单用于发现文件变化，不是密码学签名。
验证脚本另支持 `--forbid-source-root <原工作区>` 来禁止读取原工作区作为后备数据，
`--installation-report <pip的report.json>` 核对隔离环境和安装版本，以及
`--analysis-run <已保存的新分析>` 验证显式选择路径。安装报告可重复提供。

实验表见 `var/parallel/agent/round181/evidence/experiment-table.md`，可声明范围见
`docs/aic/claims-and-limitations.md`，完整方案及答辩脚本在 `docs/aic/`。
真实历史重构、合成机制和实际前瞻查询分别报告；Round175 与 Round177 的旧综合成本
口径不同，新表提供明确的成本权重，不能直接混用旧数字。

Round182 完成约定的最后一次阻断修复与验收，不重置原十轮预算，不再预留常规开发
轮次。真实策略收益尚未获得证据。已有实验仍保持 Round181 冻结口径；本轮对同一场
克罗地亚比赛追加的两次失败请求属于操作链路验证，没有增加独立比赛、正文训练样本
或已证实的效益。后续反馈修复或独立算法研究属于另行确定的任务范围。

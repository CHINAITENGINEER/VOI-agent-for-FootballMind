# 比赛驱动的信息获取运行底座

更新：2026-10-04。当前先完成底层边界；人工体验暂缓，算法研究留给后续独立任务。

## 实际入口与独立运行循环

当前命令行和比赛服务的实际入口为 `ExperienceService`，由 `runtime_cli.py` 与 `demo/experience_server.py` 调用。`autonomous_runtime_loop.py` 是另一个可注入 provider 的状态绑定循环，当前由离线实验和测试调用；它没有自动替换前者。修改后者不等于实际服务已获得相同保障，因此两条路径分别验证执行、时钟、成本和回放。

`execution_receipt_bridge` 的新运行请求在接受时保存完整规范化请求、摘要及行动级 `acquisition_id`。来源、路由、参数、候选、状态、目标与截止时间受该绑定约束；只有等价时区表示被规范化。重新规划先检查原菜单中已经执行的动作，随后才允许新候选目录变化。旧未绑定回执保留 `legacy_unbound` 标记，不能据此宣称路由已验证。

来源观测时间、操作可用时间和接收完成时间分别记录。缺失 `captured_at` 时回执保留未知，不能从请求或来源时间推断成功；已知越过预测截止时间的返回为 `LATE`，费用仍入账，Actor 状态不变。`QueryExecutionEnvelope` 要求显式 capture；调用异常仅记录本地失败事件，不把异常时间解释为来源事实。其持久账本验证每条 query 的一次计划与至多一次终结，并复核计划/回执的目标、来源版本、时钟和成本。

系统的停止声明也有边界：`stop`、`defer`、`ask_user` 和执行失败保持不同含义。未知决策在 provider 调用前拒绝；候选菜单不完整、估值未知或预算不足不等于所有潜在信息没有价值。以上验证证明自主选择与执行链路的约束，不证明真实信息价值已校准。

## 策略身份迁移（Round127）

策略身份不仅用于创建时记录，也用于恢复前的语义守卫。当前选择器是 `value-aware-selection-v2`；旧任务若保存的是 `value-aware-selection-v1`，恢复路径调用显式迁移函数，先验证保存的版本、白名单配置和指纹，再比较当前 v2 的模型描述与资源约束。v1 没有把 `value_budget` 写入身份，所以迁移只允许当前策略声明无成本上限；带成本上限的当前策略、模型描述变化、额外配置字段和指纹不一致都保持暂停。

成功迁移在同一状态事务中替换 `config.selection_policy`，并追加 `selection_policy_migrated` 事件，记录迁移方向、前后指纹和 `unbounded_only` 约束。事件写入发生在 `pending` 规划操作和任何 provider/source 调用之前；因此离线回放与故障诊断可以区分“已迁移后继续”与“语义不确定而暂停”。迁移不修改历史候选、来源回执或 VoI 标签，也不把 v1 的声明数值升级为校准结果。

## 迁移选择的回放边界（Round128）

回放按当前任务配置验证新选择；对历史选择，只额外接受迁移事件明确登记的 `from_fingerprint`，并要求事件的 `to_fingerprint` 与当前 v2 身份相同。事件缺失、目标版本不一致或旧指纹未登记时，仍报告 `REPLAY_RUN_POLICY_MISMATCH`。旧选择的自身版本/配置指纹仍由选择回放校验，迁移事件只是历史身份的有限谱系证明，不是任意策略替换通行证。

## 任务与模块分工

用户只提供比赛。系统拥有分析目标、关键未知、候选信息、查询与停止决策。比赛身份仍有歧义时，先检索，再仅确认日期、赛事或球队；不要求用户设计研究问题。

服务可脱离浏览器运行。历史类名仍为 ExperienceService，它不依赖前端服务器；前端、命令行及后续实验程序复用同一预算与执行机制。

| 模块 | 职责 |
| --- | --- |
| football_agent/match_task.py | 比赛任务目标、身份与自主问题契约 |
| football_agent/match_proposals.py | 单次模型调用，解释当前证据并生成多个可执行候选 |
| football_agent/acquisition_boundary.py | 冻结输入、候选动作、策略接口、选择结果约束 |
| football_agent/experience.py | 状态持久化、预算预约、调度、证据入账、取消和恢复 |
| football_agent/experience_sources.py | 公开来源获取与原始归档 |
| football_agent/local_evidence.py | 对本任务已归档结构化响应做显式定位、过滤与分页 |
| football_agent/acquisition_replay.py | 离线重建选择与实际请求的关联 |
| football_agent/runtime_cli.py | 无界面的创建、检查、执行和回放入口 |

流程：比赛 → 已有证据快照 → 共享分析与候选动作 → 独立策略选择或停止 → 执行器核验及预约预算 → 获取与归档 → 更新证据快照。

已有数据平台及球队/赛事契约继续使用 football_spider，不另建球队数据库。当前生产动作仍为 search_web/read_url；registered source_catalog 和 structured_sources.py 将已跑通的足彩公开接口接入同一执行链，复用平台 normalize_spftll。其他数据源需按自己的凭据、成本、覆盖和解析契约逐项接入。

## 已接入的结构化来源

source_catalog 随新比赛任务冻结，提供近期赛程及当前足彩选场两个可选 read_url 入口。它只说明能力、覆盖和未知成本，不包含预先获取的比赛事实，也不规定查询顺序。用户仍只输入比赛，候选生成器及选择策略决定是否调用。

详情与赔率链接由已取得的合法源比赛 ID 派生。未观察的 ID、猜测的链接、额外日期或历史参数不被目录授权；返回的详情/赔率 ID 必须与请求一致。没有新增无文档历史查询接口或隐藏批量爬取。

一次实际请求仍先预约预算，再归档原始响应。规范化保存来源、比赛 ID、时间、比分周期、球队 ID 质量、缺失字段，以及赔率的市场/结算语义。空数据、业务错误、错误字段和访问失败分别保留；不会因 HTTP 200 就把内容记为事实已核验。

标准化结果以完整 JSON 记录进入证据文本。算法状态提供同一可见窗口内的 structured_records，便于直接读取字段；未知比分保留 null，未核验发布时间保留未知。原始响应与规范化结果不回填为历史赛前已知数据，赛后赛果也不作为赛前预测证明。

当前证据文本最多 12,000 字符，策略快照最多 4,500 字符，按完整记录边界截断并注明范围。保留源顺序而不暗中用相关性规则选记录；总条数不代表所有行都已向策略展示。

query_acquired 是新的显式本地信息动作。它只能引用本任务已经成功归档的 platform_structured 文档，支持源比赛 ID、球队文字精确/包含匹配和基于过滤结果的 offset/limit 分页。它重新读取同一份不可变原始字节并重建规范化记录，不发 HTTP、不读取其他任务、不会把未显示的记录偷偷加入当前证据。结果携带父文档、观测时间、原始摘要、覆盖范围、过滤条件和 next_offset；字符窗口不足时明确报告被跳过的 offset，不能把截断冒充完整结果。空结果仍只是“本次已归档响应没有匹配记录”，不是比赛不存在。

本地查询有独立的有限读取预算和失败回执。HTTP 用尽时，只要仍有本地读取和后续推理预算，Agent 仍可选择该动作；执行时间、重建失败、取消、截止时间和中断都会留在账本里。零 HTTP 不等于零信息获取成本，local_query_capability 的处理耗时和决策损失仍未校准。分页、源比赛 ID 和球队名是执行约束，不是相关性排序或球队别名推断。

来源目录、公开参数及解析版本绑定任务。配置变化会在下一次规划或排队获取前阻止执行。原始归档在 archive/raw/spftll；离线回放重新读取原始字节，校验摘要并复算规范化结果，而不只比较展示元数据。目录注册本身不保证来源此刻可用或与目标比赛相关。

## 算法接入接口

使用 ExperienceService 的 proposer_factory 与 selection_policy。结构化生成器可使用 MatchCandidateProvider(folder, effort)；算法实现 SelectionPolicy。

策略提供 version、configuration，以及 select(context, batch) → PolicyChoice。configuration 是可 JSON 序列化的公开配置；权重摘要、训练截止、特征版本和随机种子应登记在其中，不存凭据。会被归档脱敏改变语义的配置会被拒绝。

select 是本地选择函数。网络或模型获取仍由受计费的 provider/tool 执行，不在选择器里另发隐含请求。隐藏可变状态不应取代公开快照；需要恢复的状态或训练产物必须显式版本化。

context 含 task_id、snapshot_digest 和 state。state 每次返回独立副本，包括比赛、历史分析、已返回证据片段、来源与发布时间、原始摘要/请求关联、剩余预算、已尝试动作；不含未来返回、最终赛果标签或未获取正文。

batch.candidates 是冻结候选列表。每项含 candidate_id 和 response，后者包含动作、目标未知及有用返回判据。所有候选共享同一 assessment、analysis_plan 和 match_identity，避免按动作更换当前信念。候选可提出新搜索词和已观察 URL，不限定伤停、天气等固定维度。

PolicyChoice 必须绑定原 task_id 和 snapshot_digest。以下输出互斥：

1. candidate_id：选择本批次的一项；执行前再次核对预算、来源可见性和引文。
2. terminal_action='stop' 或 'defer'：独立结束，不追加模型调用，也不要求模型先提出停止候选。停止不提升事实支持程度。

reason 是解释；diagnostics_json 可记录算法估计及其单位/依据。底座不自动认证自报估计为已校准数值 VoI。默认 ProposedChoiceBaseline 仍采纳模型定性推荐，只是兼容基线。

策略版本与公开配置绑定任务。恢复时策略或候选模式不同会暂停，不默默切回默认策略；截止时间和费用不重置。当前 provider 模式会绑定，生成器的具体版本与原始返回另外归档；更换生成器也应创建新实验任务。

## 候选与成本

结构化 provider 一次返回共享分析及 1—6 个候选；通用边界最多接受 32 个。数量上限是运行资源限制，不是固定研究维度；没有有意义的替代动作时允许一个候选，不凑数量。

新候选 value_status 复用 query_value_contract.VALUE_UNMEASURED。来源调用已知 HTTP 请求下界为 1；重定向和失败按实际预约计费。缺乏依据的金钱、时延估计及等价决策损失为 null，不填零。

llm_calls 延续原含义：每次规划 provider 调用前预约一次尝试，失败及超时保留。结构化 provider 应封装单次规划模型调用；候选展开和本地 selector 不增加模型次数。selector_elapsed_seconds 单独记录选择耗时；远端 provider 内部请求数未报告时仍为未知。

provider-outcome-v1 为规划调用提供统一失败边界。它按 process_start、stream_parse、stream_validation、turn_completion、message_parse、schema_validation、candidate_validation、policy_selection 和 decision_validation 区分“调用没有形成可用决策”的阶段；记录只含有限错误码、错误摘要哈希和本任务内的 diagnostic 相对路径，不把 provider 原文、stderr 或凭据放入 public state。每次尝试的预算仍视为已扣除；远端可能已经完成而本地未确认的情况保留 remote_outcome=unknown，服务不自动重复同一个逻辑尝试。operation、failure.json 与 last_provider_outcome 共享该记录，因此恢复时可以区分“没有选择”与“选择返回未接纳”。

来源返回后还会生成 return-observation-v1。它只投影执行器可验证的技术事实：transport_status、HTTP 状态、内容类型、证据表面（正文、结构化记录、本地结构化记录或发现元数据）、覆盖范围和来源信号。它明确把 HTTP 200、页面标题、命中球队文字和规范化记录与“回答了当前未知”分开；semantic_relevance、information_gain、decision_answered 保持 unassessed/null，truth_status 保持 unverified，交给后续 Agent 评估与算法策略。结构化响应被窗口截断、搜索摘要可能只是导航、空结构化结果和访问失败都保留不同的技术状态。该投影写入 acquisition result 与 public state，离线回放按同一原始文档重新计算并校验，派生观测被篡改时回放失败；它不增加来源请求，也不宣称 VoI 校准。

### 返回准入与任务生命周期（Round167）

实际 `ExperienceService` 在拿到工具返回后先把原始 `result.json` 持久化，再在同一状态锁内生成 `admission.json` 和 `acquisition_admissions` 记录。准入使用本地 `evaluated_at` 与任务 deadline/cancel 状态；它与来源的 `observed_at`、`source_published_at` 和执行器 capture 是不同的时钟。取消或任务截止已经发生时，返回仍保留原始正文、请求回执和实际资源成本，但状态为 rejected，不进入 `documents`、`pending_return` 或 Actor 可见 evidence state。已经准入的历史文档不会因为任务后来结束而消失。

回放要求 acquisition link 的准入版本、准入摘要、事件序号、资源操作完成时间和原始结果逐项一致；它保留旧任务无准入标记的兼容路径，但不会替旧记录补造时间。动作账本把拒绝返回的 `admission_status/reason` 与资源成本分开投影，`rejected_late_return` 不被解释为信息已进入当前状态。这样可以同时表达“请求已经发生并应计费”和“该返回不能影响当时的 Agent 决策”，避免把 transport completion 或归档存在误当成可用证据。

current_assessment 目前是证据支持的文字分析，不是校准概率。将其变为可用于损失计算的信念状态属于算法研究，不在底层虚构。

## 问题级实际返回评估（Round120）

新建无界面比赛任务默认启用 match-return-assessment-v1。用户仍只给比赛；Agent 自主形成问题、提出候选、获取信息，再于下一次正常规划调用中评估刚收到的实际结果。评估与下一步候选生成共用一次调用，不增加隐藏模型调用，也不预设伤停、天气等查询顺序。

获取结果携带 return_context，绑定原选择操作、原决策摘要、原 active_question_id 对应的问题文本、unknown_to_resolve、useful_if 和实际返回摘要。即使后续 analysis_plan 修改了问题或复用了 ID，旧动作的原始判据仍保持原样。空返回和失败也有上下文，防止把“未取得信息”当作“事件没有发生”。

pending_return_assessment 只在实际获取后出现。下一次规划必须提交共享 return_assessment，再选择继续、停止或澄清比赛身份。所有候选使用同一份返回评估，外部 SelectionPolicy 独立停止时保留该评估；取消、截止或未通过校验的规划不接纳评估，原待评估记录与已花费预算继续保留。

| outcome | 契约含义 | 执行校验边界 |
| --- | --- | --- |
| resolved | Agent 认为来源回答了原问题 | 需要可见正文引用，remaining_unknowns 为空 |
| partial / irrelevant | Agent 认为部分回答或与原问题无关 | 需要可见正文引用，并保留未知 |
| conflicting | Agent 报告来源内的冲突 | 至少两段不同的可见引用，冲突本身未独立验证 |
| discovery_only | 只获得可继续导航的发现元数据 | 不作为正文事实引用，保留未知 |
| unavailable | 未取得可用来源结果 | 不解释为语义否定，保留未知 |
| insufficient_context | 空结果或现有内容不足以判断 | 保留未知，不制造信息收益 |

引用只接受这次获取对应文档中可见且唯一的原文片段。其他文档、未展示窗口和搜索摘要不会成为本次正文引用。网页读取与 query_acquired 共用该契约；本地查询仍有自己的预算，不把同一份原始响应当作新的独立来源。

接受的评估写入 operations/<id>/return-assessment.json 与 question_return_assessments。记录始终标记 ACTOR_PROVISIONAL_NOT_GOLD、world_truth_verified=false、independent_semantic_validation=false、realized_decision_gain=null、numeric_voi_training_eligible=false。严格引用检查只证明定位与可见性，不证明相关性、蕴含关系、矛盾或世界真值。技术层 return_observation 的 unassessed/null 不被改写。

离线回放从已接受动作和实际回执重新构造原问题上下文，逐次核对规划前的待评估状态、历史评估、接受记录和最终状态；不会只相信已存的语义标签。结果校验也不等于策略收益验证。

兼容边界：runtime_cli 创建的新任务默认启用；直接使用 ExperienceService 的既有调用方维持原协议，可显式设置 enable_return_assessment=True 创建新任务。恢复时以任务保存的能力版本为准，不为旧任务补造历史评估。前端和人工体验流程本轮未扩展。

## 归档与恢复

每个规划操作保存 input.json、result.json、proposal.json、policy-output.json、selection.json、decision.json，分别对应事前快照、原始 provider 返回、候选、策略输出、选择绑定和最终接受决策。模型原始推荐不会被算法选择覆盖。

来源获取另存 acquisitions/<id>/input.json、selection-link.json、result.json。实际请求及证据关联产生它的选择操作，排队动作被改变时不执行；原始正文仍归档在 archive。

取消后不派发新请求。中断且结果未确认时保留 pending 和已预约预算，恢复不自动重复。回放只重建选择并核对关联，不发请求、不重跑模型，也不替代来源真实性和算法效果评估。对正在写入的任务，先等待停止或暂停后再完整回放。

## 无界面命令

创建任务（零模型和来源调用）：

    python -m football_agent.runtime_cli create --match '主队 vs 客队，赛事与日期'

ID 替换为创建返回的真实 run_id：

    python -m football_agent.runtime_cli show ID
    python -m football_agent.runtime_cli step ID
    python -m football_agent.runtime_cli run ID
    python -m football_agent.runtime_cli cancel ID
    python -m football_agent.runtime_cli verify ID

仅 step/run 调用模型或工具。默认目录 var/runtime，与人工体验目录分开。使用其他目录时把 --root 路径 放在子命令前。任务有效期从创建开始，不因查看、重启或恢复延长。

直接核对既有操作目录：

    python -m football_agent.acquisition_replay var/runtime/runs/ID

## 后续算法线程边界

聚焦信息的事前价值、证据条件下的概率更新、成本约束下的选择与停止。复用已有 decision_value.py、inquiry_value.py、query_value_contract.py、empirical_context_policy.py 中适用定义；区分显式构造模型与真实经验校准。

算法线程先实现版本化 SelectionPolicy 并接入对照，无需重做前端、爬虫或任务账本。估值未测量不等于零价值，事后概率变化不等于事前价值。真实对照、独立标签与校准证据仍需专门设计。

已落地：结构化证据的本地定位/分页、模型失败分类、技术返回观测，以及绑定原问题的实际返回评估。尚待检验：真实模型对新契约的遵循情况、开放网页相关性与实际任务完成率、其他来源接入。真实多步策略收益、数值 VoI 校准和泛化效果留待算法研究。接口成立与这些研究结论成立分开验收。

## 真实 Provider 契约回执（Round121）

真实 CLI Provider 现在额外生成 provider-contract-v1 回执。回执绑定公开输入摘要、协议版本与协议指纹、响应 Schema、提示词摘要、流式响应摘要、输出文件摘要、事件摘要和能力开关；tool_events 必须为空，防止模型在声明的问答回合中暗含来源访问。ExperienceService 在接纳真实回合前复核公开输入与协议绑定，并把回执相对路径和契约摘要写入 operation/provider outcome，离线审计可以发现回执或输入被替换。

CLI 在 turn.completed 后清理 response.json 时，回执会记录 output_file_present=false，但仍以已验证的 agent message 流作为响应来源；这保持了既有清理回合语义，同时让缺失文件成为显式事实。

该回执只证明运输和边界遵循：语义正确性仍由 ExperienceService 的决策校验处理，来源事实与世界真值没有被认证，numeric_voi_calibrated=false。因此它为后续真实 Provider 和算法线程提供可追溯输入边界，但不会把“模型回了一段合规 JSON”升级成“Agent 已经有效选择了最有价值的信息”。
该回执只证明运输和边界遵循：语义正确性仍由 ExperienceService 的决策校验处理，来源事实与世界真值没有被认证，numeric_voi_calibrated=false。因此它为后续真实 Provider 和算法线程提供可追溯输入边界，但不会把“模型回了一段合规 JSON”升级成“Agent 已经有效选择了最有价值的信息”。

## 本地价值选择策略（Round122）

新增 value-aware-selection-v1 作为可插拔 SelectionPolicy：Agent/Provider 只负责提出共享分析与候选信息动作，最终选择由本地策略根据候选级 QueryValueAssessment、可见资源预算和当前候选集合完成。它优先选择声明模型内净价值为正且资源可执行的候选；若仍有未测量的开放世界候选，则返回 defer 并保留 WITHHELD_OPEN_WORLD_VALUE_UNKNOWN；只有所有可执行候选都已被声明模型覆盖且净价值非正时，才返回 MODEL_SCOPED_STOP。

该策略不会把候选缺失、预算耗尽或未校准估值转换成零价值，也不会替用户提出问题。候选仍由 Agent 根据当前比赛状态生成，策略只负责在同一批候选中做可回放的价值门控。numeric_voi_calibrated 继续为 false；现阶段的数值只代表显式声明模型或历史校准输入，不能解释为比赛结果概率或开放世界全局最优。

## 注册候选价值模型与运行时接线（Round123）

Round123 将 Round122 的本地价值选择策略接入 headless match runtime。候选生成仍由 Agent 完成，候选选择由 ValueAwareSelectionPolicy 完成；两者之间通过 CandidateValueModel 连接。

价值模型是一个显式版本化的规则文件，不由当前 LLM 临时发明数值。每条规则按 action、query_contains、url_contains 和 engine 匹配候选，返回声明模型下的 expected value 与 cost。没有命中规则时，适配器返回 VALUE_UNMEASURED，策略进入 defer，不会把未知价值当成零。

运行时通过 --selection-policy value-aware --value-model PATH 显式启用；策略身份会持久化模型摘要和哈希，任务恢复时模型发生漂移会被边界拒绝。当前数值 VoI 仍是声明模型，不代表已经完成真实校准。

## 上下文条件与返回后的再选择（Round124）

CandidateValueModel 的可选 when 对本次 SelectionContext 的公开快照进行条件匹配。候选由 Agent 提出，估值函数每次读取当前快照；同一信息动作可随上下文命中不同规则，进而改变排序。注册规则仍属于声明模型的接口验证，数值没有经过经验拟合，规则本身不是自主学习算法。

条件直接使用运行时已经公开的字段，例如 match_identity.status、current_assessment、documents，以及启用返回评估后的 pending_return_assessment。未公开的字段不会被补造：比赛出线、战术或轮换状态若未被有来源的上下文投影暴露，条件就不满足。调用方提供的独立 match_context 示例仅是构造测试，当前运行时没有自动生成这一字段。LLM 的结构化评语也不自动升级为已核实事实。

匹配规则：对象按声明键递归匹配，额外键不影响匹配；数组按长度与顺序逐项匹配；布尔值与数字区分，JSON 数字按数值比较。字段缺失与显式 null 区分，空子对象仍要求该键存在且值为对象。条件只接受有限 JSON 值，嵌套深度上限为 16，配置在进入任务前校验。

没有非空 when 的模型保持 Round123 原模型哈希及描述符；包含条件的模型额外声明 context_match_mode=exact_recursive，条件内容纳入模型哈希，沿用既有恢复时的策略身份核对。规则仍按 first_match 处理，广泛条件应排在具体条件之后。

运行时构造测试从用户仅提交比赛开始：本地策略覆盖候选生成器的推荐，执行所选信息动作，来源返回进入下一次公开快照，随后失效的条件回到 VALUE_UNMEASURED 并暂缓。该测试只证明返回与选择之间的接线及离线回放，未证明返回语义相关、风险下降或真实模型行为。MODEL_SCOPED_STOP 只对已声明模型内的候选成立；未知价值保持未知。

## 当前返回评估进入同次信息选择（Round125）

公开快照在调用候选生成器之前建立，question_return_assessments 因而只包含此前已接纳的评估。本次规划刚产生的 return_assessment 要等最终决策被接纳才提交历史。若价值模型只读公开历史，就会遗漏刚读完正文后的判断，直到下一次规划才生效。

current_return_evidence(public, response) 为选择器提供本次返回评估的派生视图。它复用问题级返回校验：任务、实际 acquisition、原始问题、上下文摘要、关联文档及可见引用须一致。视图保留获取前的问题和有用判据，不用新分析计划改写旧动作目的；它复制已有数据，不提前提交历史或改变公开快照。

注册模型增加可选 when_return，与 when 共同限制规则。when 继续匹配公开快照，when_return 匹配当前返回视图；两者均使用有类型的递归 JSON 匹配。例如下列条件只要求本次对原 lineup 问题的暂定评估为 partial，不规定下一步动作或数值：

```json
{
  "when_return": {
    "context": {"question": {"id": "lineup"}},
    "assessment": {"outcome": "partial"}
  }
}
```

空 when_return 对象仍要求存在通过校验的非空当前评估。首次规划、已消费返回、未启用返回评估的旧任务均不命中，不从历史回填。无规则覆盖时按显式 default 处理，默认 VALUE_UNMEASURED；缺失评估不自动解释为零价值。已声明当前评估但绑定失配、正文引文不可见或能力契约改变时，估值前报错。

视图固定携带 ACTOR_PROVISIONAL_NOT_GOLD、world_truth_verified=false、independent_semantic_validation=false、realized_decision_gain=null、numeric_voi_training_eligible=false。它可以作为选择器的条件输入，尚不是独立核实的事实或已实现的信息收益。当前注册规则只验证接口，未来估计器需要从观测与这类带不确定性的评语中估计事前价值；本轮没有把评语直接变成训练标签。

只有使用 when_return 的模型描述符才增加 current_return_evidence_version 和暂定判断标记；条件纳入模型哈希。旧 when 模型及无条件模型的身份保持不变，沿用既有任务恢复核对。候选必须共享本次返回评估的边界继续有效；独立停止、取消和失败仍按原提交规则处理。

构造集成验证覆盖同一旧历史下，本次 partial 与 irrelevant 评估分别触发不同候选选择。三次正常规划、两次构造获取后，下一步动作排队；本次评估参与选择没有增加模型调用。原提议、选择、实际返回和已接纳评估可离线核对。该结果证明执行链及时使用返回信息，不代表规则价值已经学习、来源语义已独立验证或真实决策损失已经下降。

## 停止边界与资源约束（Round126）

Round125 的选择器会遍历一批候选。停止或暂缓是控制动作，不是待估值的信息查询；如果把候选中的 `stop`、`defer` 也交给价值估计器，它们可能污染“所有信息都没有正价值”的判断。Round126 的 value-aware-selection-v2 将控制动作排除在信息价值评估之外，并在诊断中标为 `terminal_control` / `NOT_APPLICABLE`。选择器自行产生停止或暂缓，不能从 Agent 的控制候选推断世界已经足够确定。

可评估动作目前包括 `search_web`、`read_url`、`query_acquired`，以及经过比赛身份校验的 `ask_user`。其他动作在选择前报错，不隐式当作零成本信息。每个候选的可执行性同时受来源槽位、`final_decision_only` 和声明价值成本上限影响；这些是运行约束，不是信息价值观测。

停止结果必须带范围：

- `BUDGET_EXHAUSTED` 表示资源或最后一次规划限制阻止执行，保留未知；
- `WITHHELD_OPEN_WORLD_VALUE_UNKNOWN` 表示仍有可执行候选尚未测量其价值；
- `NO_FEASIBLE_REGISTERED_ACTION` 表示本次候选集只有控制动作，不能证明信息已经充分；
- `MODEL_SCOPED_STOP` 只在至少一个可执行信息候选、所有可执行候选都被同一声明模型覆盖且最大净价值不为正时成立。

诊断显式记录 `evaluated_candidate_ids`、被忽略的控制候选、排除原因、`candidate_generation_complete=false` 和 `global_optimum_claim=false`。已选择一个正净价值动作也只说明它在当前已提出且已建模的候选集合内胜出；未测量的开放世界动作仍保留未知，不能声称全局最优。

价值成本上限现在进入 SelectionPolicy 的持久化配置和指纹。恢复时上限改变会暂停任务，避免同一份候选在不同资源语义下被重放成另一种选择。旧 Round125 模型快照仍可离线回放；正在创建的新任务使用 v2 身份。数值仍是声明假设，未校准为真实 VoI。

## 正式 QueryValueContract 投影（Round129）

Round126—128 分别固定了候选边界、策略身份迁移和历史回放，但比赛运行时此前只在 `diagnostics_json` 中写入本地停止字段；通用 `QueryValueContract` 的 `QueryValueDecision` 与 `StopCertificate` 主要由上下文查询循环使用。Round129 将两者接到同一份归档选择输出，保持原有排序与门控行为，只增加可回放的协议投影。

运行时的正净价值信息候选生成 `QUERY` 决策并绑定被选候选 ID及全部已评估的 `QueryValueAssessment`。仍有未测量候选时生成 `DEFER` 与 `WITHHELD_OPEN_WORLD_VALUE_UNKNOWN/open_world` 证书；来源槽位、最后一次规划限制或声明成本上限阻止全部候选时生成 `DEFER` 与 `BUDGET_EXHAUSTED/operational_budget`；候选全部由声明有限模型覆盖且最大净价值不为正时生成 `STOP` 与 `MODEL_SCOPED_STOP/declared_finite_model`。只有 `stop/defer` 控制候选时，运行时保持 `defer`，并生成 `NO_FEASIBLE_REGISTERED_ACTION/proposed_candidate_set` 证书；这表示候选缺口，不表示信息价值为零。

运行时资源不是单一可加总的效用单位。`StopCertificate.remaining_budget` 因而接受标量或命名映射，并对映射键和值做有限、非负、排序后的归一化。`http_calls`、`local_reads` 等槽位原样进入回放证书，不建立未经声明的跨资源汇率。上下文查询循环仍使用标量预算时，旧序列化格式保持兼容。

`ask_user` 是比赛身份澄清动作，不是来源 acquisition query。若它被声明模型选中，诊断保留 `query_value_decision=null` 和明确的省略原因；正式 query 投影范围标记为 `acquisition_actions_only`。这样用户提交的比赛身份问题不会与“Agent 选择下一条外部信息”混成同一类动作。

本轮的 179 项定向回归和 7 项确定性审计只验证协议接线、资源保真和停止范围；没有真实模型请求、来源请求或数值 VoI 校准。它证明的是选择账本的一致性，不是信息收益、概率校准或比赛结果性能。

## 动作级成本与证据账本（Round130）

`action-value-ledger-v1` 将已通过选择边界和来源回放核验的记录按选择操作聚合。一行对应一次已接受的规划选择，字段分成四个时点：

1. **选择前**：`candidate_value_assessments`、`selection_basis`、`selection_scope`、候选的声明 `expected_information_value/expected_cost/expected_net_value`、`prospective_digest` 和规划前独立资源快照；未测量价值保持 `VALUE_UNMEASURED`，终止候选为 `NOT_APPLICABLE`。
2. **执行中**：`selection_resource_cost` 和 `resource_cost` 分别记录一次规划与实际预约的 `llm_calls/http_calls/local_reads`。来源重定向、失败和 pending 预约都按已预约槽位计数，本地查询的 HTTP 成本为零但 `local_reads` 为一。
3. **返回后**：`return_observation` 只包含技术返回覆盖、状态和证据表面；`return_context` 绑定原问题、未知项和 useful-if，不因后续计划改写。没有可用文档的获取尝试也保留 acquisition 状态。
4. **再选择**：`return_assessment` 保存当前 Agent 的暂定 outcome、可见引用和剩余未知，关联 `assessment_operation_id` 与 `next_selection_operation_id`。该评估带 `ACTOR_PROVISIONAL_NOT_GOLD`、`world_truth_verified=false`、`realized_decision_gain=null`，不能直接成为 VoI 训练标签。

账本由 `ExperienceService.export()` 暴露，也作为离线 replay report 的摘要字段；公共构建入口先执行回放校验，避免只相信单个账本文件。`entry_id` 绑定运行与选择/决策摘要，`entry_digest` 和 `ledger_digest` 绑定当前投影；`state_digest` 变化后导出会把账本标为需要重新生成。该设施让后续算法可以观察“事前为什么选、实际花了什么、返回覆盖了什么、下一步怎么变”，但没有把事后状态变化冒充事前 VoI，也没有跨资源汇率或收益校准。

## 自主信息选择的离线评估契约（Round131）

Round130 的动作账本可以记录一次选择及其执行链，但不能单独证明“这个动作比另一个动作更值得”。Round131 增加 `value-evaluation-contract-v1`，把评估器输入与 Agent 的公开状态分离。

评估记录分为四个边界：

1. **事前公开快照**：固定 `prediction_cutoff`、内容摘要、候选 ID、候选生成完整性、选择操作、`query/stop/defer` 决策及选择时间。快照递归拒绝 `result`、比分、标签、隐藏返回、事后状态和 `decision_loss_after` 等字段；`snapshot_digest` 不匹配时整条记录阻断。
2. **候选级技术返回**：每个候选最多一行独立评估记录，绑定 `return_digest`、观察时间、声明资源成本和固定决策损失单位。返回摘要只证明评估器看到了某个技术结果，不证明来源陈述正确。
3. **独立评估标签**：每行带 `evaluator_id` 和 `label_status`。只有同一个外部评估器为所有候选提供 `INDEPENDENT_EVALUATOR_GOLD`，并且结果可用时间晚于返回，才满足独立标签条件。`ACTOR_PROVISIONAL_NOT_GOLD` 可以被保存用于审计，但不能当作训练标签。
4. **覆盖资格**：候选生成明确完整且所有菜单项都有独立标签时才是 `counterfactual_eligible`。只看到所选动作，或候选生成仍然是开放世界时保持 `partial_observation`；候选越界、摘要篡改、返回早于选择或结果早于返回时为 `blocked`。

契约只输出资格和原因清单，不计算 loss reduction、realized decision gain 或数值 VoI。`numeric_voi_training_eligible=false` 是固定边界，不会因为 Agent 的暂定语义评估、技术返回状态或完整菜单自动变成 true。后续研究若要训练估值器，必须另行登记独立结果来源、固定损失、反事实覆盖和校准数据切分，并把这些资料放在评估器域，不能回写 Agent 的事前状态。

## 动作账本到评估 Episode 的桥接（Round132）

Round132 将已验证的 `action-value-ledger-v1` 适配为 `value-evaluation-contract-v1` episode。`ledger_entry_to_evaluation_episode()` 是纯函数：它只复制选择前的公开快照、候选 ID、候选生成完整性、选择操作、决策、预测截止时间和选择时间，并重新绑定 `snapshot_digest`。账本中的技术返回、正文、结果、Agent 暂定评语和独立评估字段不会进入公共 episode。

每条 episode 都带 `evaluator_sidecar.required=true` 与 `runtime_can_read=false`。因此没有独立评估器时，`candidate_evaluations=[]`、`evaluation_outcome=null`，契约报告 `partial_observation`；这不是把未知收益记为零。独立 evaluator 后续可在离线域按 `entry_id`、`entry_digest`、`prospective_digest`、`state_digest` 和候选菜单摘要加入评估记录，运行时本身不读取或改写这些记录。

`Experience.export()` 在已验证账本上生成完整 episode 列表；`acquisition_replay` 默认只报告数量和确定性摘要，显式请求账本时才返回 episode 内容。两条路径使用相同适配器，保证恢复、导出和离线回放看到同一份事前选择边界。Round132 只建立数据契约，不启动 evaluator runner、不计算实际 VoI、不声称候选集合覆盖开放世界。

# 第三轮：公开状态、主动获取与价值停止

2026-09-22。已接通真实平台 `FrozenEvidenceEnvironment`、研究 `PartialPredictor` / `ValueEstimator`、Agent持久执行器和浏览器。概率与预计改善均来自研究服务。四入口为 fixed、pure_llm、all_candidates、llm_gate；外部LLM当前缺模型配置/凭据，实际请求0次，数值路线独立运行。

## 当前展示

- 真实主动获取：http://127.0.0.1:8769/?run=3aa5fc27-76c6-42bb-a274-05f11f4e54c9
- 正式 v2 主动获取：http://127.0.0.1:8770/?run=758b785d-39db-4a19-9bdf-0d4f50b17711
- 同场固定顺序：http://127.0.0.1:8769/?run=f0ea6676-53c0-4baf-818e-2705430343bd
- 构造机制展示：http://127.0.0.1:8768/?run=fca05525-841b-4b86-869f-4cbceabfcf23

真实比赛 canonical ID `654a90d2-65a9-5cae-8494-f2fb685a122b`（Burnley—Newcastle United），cutoff `2025-12-29T19:30:00Z`。当前正式研究发布为 `active-information-v2` fold2；v1 fold2与fold0轨迹继续保留作兼容/对照。这是展示场景选择，不是对模型效果的筛选报告。

| 决策位置 | 候选 | 预计log-loss改善 | 换算成本 | net_gain | 支持比赛数 | 执行 |
|---|---|---:|---:|---:|---:|---|
| 空mask | results_view | -0.01697919 | 0.005 | -0.02197919 | 80 | 跳过 |
| 空mask | xg_view | 0.00833387 | 0.0075 | 0.00083387 | 80 | 获取 |
| 取得xG后 | results_view | -0.00111759 | 0.005 | -0.00611759 | 53 | 停止 |

`xg_view`模拟揭示费0.03 simulation_credit；结果视图一次揭示Elo/近期表现两组，模拟费0.02。正式 v2 fold2 验证段冻结lambda.money=0.25 log-loss/credit。固定顺序取得两动作累计0.05；数值策略取得一次累计0.03。实际HTTP请求0，费用差不代表供应商节费或策略统计优势。原Understat底层响应含全部三组，此处测量明确标注的缓存投影揭示。

数值路线最终概率48.3982877% / 22.2300877% / 29.3716245%；固定路线61.1499234% / 10.4289238% / 28.4211529%。两者分别与直接研究部分预测器完全一致，输入mask不同。未强制概率随信息改变。

## 运行命令

工作目录 `D:\a1-足彩`，使用已有Python环境，不新增共享依赖。

```powershell
python -X utf8 -m demo.acquisition_server --port 8769 --mode real --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda --research-artifact var/parallel/agent/round3/real/backend-fold-2.json --store-dir var/parallel/agent/round3/real/ui-fold2-runs

python -X utf8 -m football_agent.acquisition_cli run --mode real --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda --research-artifact var/parallel/agent/round3/real/backend-fold-2.json --match-id 654a90d2-65a9-5cae-8494-f2fb685a122b --cutoff 2025-12-29T19:30:00Z --strategy all_candidates

python -X utf8 -m football_agent.acquisition_cli validate-synthetic
python -X utf8 -m var.parallel.agent.round3.real.verify_bridge
python -m pytest tests/agent tests/demo -q --junitxml=var/parallel/agent/round3/regression-final.xml
```

CLI `show --run-id UUID --store-dir DIR`读取记录；`resume --run-id UUID`另带创建时相同mode/bundle/research-artifact/store-dir参数继续。配置hash变化暂缓，不自动换模型。服务端配置路径只在启动参数读取，HTTP不接收模型路径、隐藏组、标签或任意动作覆盖。默认真实模式缺配置显示BACKEND_UNCONFIGURED，明确构造模式用 `--mode synthetic_test`。

## 实现与边界

`football_agent/acquisition.py`为独立研究运行器，旧两轮执行入口保留。backend为可替换接口：configuration/matches/start/actions/predict/estimate/acquire。策略/LLM/估计器/部分预测器只收公开字典；环境独占完整数据与标签。运行器不调用旧PlatformTools完整FeatureIndex。桥接复用平台VisibleState/ActionSpec与研究 `to_research_state`，缺失/过期/失败mask分别保留。

动作目录只公开能力、成本及前置条件，不提前泄漏目标组值或将来返回状态。未获取组不进入run.groups和事件snapshot。工具完整返回在保存blob前递归校验隐藏/标签/路径字段；失败状态或越权揭示暂缓。改变环境内部隐藏值与标签的探针见验收，文件路径不传给LLM。

幂等键包含action ID、参数、目录/证据版本、刷新窗口和cutoff。NO_DATA/失败/无变化只消耗当前动作和预算，继续其他合法动作；显式新版本可再查询。retry_after记录到持久退避，避免同动作立即重试；固定cutoff发生改变时拒绝，并要求另开明确关联的时点运行。现有round2 live父子流程保留；本轮新动作环境为历史受控揭示，没有新增线上live承诺。

调用前持久化费用预留与request ID，返回后持久化输出引用，应用状态后清除pending。崩溃窗口恢复为INTERRUPTED_CALL_REQUIRES_RECONCILIATION，不猜测结果、不自动重复付费。RunStore沿用OS跨进程锁与原子写、hash审计。可确认的模拟实收费结算到模拟账户，超预算立即停止；未知线上费用保留unknown和本地非货币预留。LLM调用数、usage、计算调用数、时延、模型/成本/配置版本均入账。

少量候选全枚举；llm_gate仍估计完整合法目录以测量正价值动作遗漏。LLM提出stop时会先记录该审计，再以LLM_PROPOSAL_STOP_AFTER_VALUE_AUDIT停止；不伪装成数值非正。pure_llm保留纯提案停止对照。LLM理由不参与数值VoI估计。详情见[提案器](llm-proposer.md)与[研究桥接](research-bridge.md)。

## 冻结版本与证据

- 平台COMPLETE：understat-2025-historical_reconstruction-a3b3d45811fd5cda，feature_version=prematch-season-v2.0，COMPLETE SHA256 `e0a3aa4c9c750e4aa8453c5654c2466de97c7e3ffbd803fc543e1c706793e497`。继续使用研究已训练UUID空间，不混入9b013510映射增强版。
- 数据版本：`platform-history-research-v1.0:a3065657f0a6e7e8385e783a9169e23d35e419469c54061e1da91ee4806493ad`。
- 正式 v2 fold2部分模型：`partial-subset-logistic-1.0:fb968fc75bd79f62de8d8ef5abe8c57a37aae93c15222c5b08d059110e33721a`；artifact SHA `c027bc039e49f7a4161cd272caf3922cae626fd2d903effec3185f862a51671e`；配置 SHA `1161136fa55365f7e0080fa33f54d0a87caa95d9a86f3da975a1eaf526fae08e`。
- 估计器：`empirical-mask-action-shrinkage-v1:3f28ebabe006605999da`；artifact SHA `37afccc703a57126f92a741ddf7414301c2860805d7c042464c63307ed9ff92d`。两者训练截止均2025-11-29T12:00:00Z，早于目标cutoff。
- 原始观测引用：`observation:a9447ae0-3252-4207-a437-63bfc700f3a5`；各可见组保留match_version/statistic引用、group版本和可用时间，详见完整run的visible_lineage。
- 环境seed=7冻结在成本配置；执行器seed单独保存，本版数值选择和固定顺序均为确定性执行。

`var/parallel/agent/round3/browser-real-active/`保存 v1 真实浏览器、完整同场对照、截图与实际进程重启前后记录比较。正式 v2 浏览器保存于 `var/parallel/agent/round3/browser-real-v2-final4/`，主 run `758b785d-39db-4a19-9bdf-0d4f50b17711`，固定对照 `cfc5ad85-d01d-44ee-a834-007105ddb79d`，数值对照 `122d90be-b582-4a50-b72e-0dd8dce29c0d`，8项检查和0 JS错误。正式 v2 运行及完整价值摘要另保存在 `var/parallel/agent/round3/real-v2/{v2-validation,all_candidates-run,fixed-run,cli-v2}.json`，桥接主验证 run `396d68f0-a68a-42ed-ab9f-3dfa0d7d9dc3`，固定 run `13fc9912-5803-4a04-8fcc-4c7d0a95417a`，CLI 重跑 `90ada41f-85b1-42da-b70d-e0c2dc2686e4`。

## 分项验收与限制

- 真实历史：fold0立即停止、fold2主动取xG后停止、固定两次揭示三组、cache_views零新增外部费对照均已执行。`real/fold-2-validation.json`验证direct概率一致。
- 恢复：真实运行中途重建backend，2动作累计0.05不重复收费，31事件hash通过；浏览器主服务实际进程重启后记录/配置/预算/事件逐字一致。证据 `real/integrity-validation.json`、`browser-real-active/process-restart.json`。
- 隔离构造探针：改真实环境内未揭示xG和目标比分，获取前状态/预测/目录/估计全部一致；取得xG后才变化。明确标constructed_probe_on_real_history。
- 构造机制：立即停止、状态改变后选不同动作、NO_DATA后继续、预算耗尽、正增益低于成本五类，另覆盖隐藏泄漏、目录外动作、LLM提案遗漏、实际费超预留、崩溃收费窗口及版本刷新。`synthetic-acceptance/validation.json`。
- 浏览器：真实fold2、真实fold0、构造各8项核心检查、各0 JavaScript错误；先前未配置场景记录保留，状态切换见demo-configuration-transition.json。
- LLM真实状态NEEDS_CONFIGURATION、外部LLM请求0；构造provider测试不算实调用。伤停/天气NOT_IMPLEMENTED；不产生虚假观测。

本轮Agent新增外部采集请求0。已零网络消费平台round2成功fixture交接：spftll:3012999、观测3b3ff4de-5361-4b6a-92c3-8109b18be48b，7条Agent证据，实际费用UNKNOWN；其杯赛scoreperiod未知，只证明fixture刷新工具成功。旧“完整赛季等待COMPLETE”已经关闭，旧空响应记录继续作为历史审计。

数据是探索性历史重建，3h赛果/24h xG/168h赛程可见假设保留；不是前瞻效果证明。数值为有限一步经验改善估计，支持度和不确定性原样显示；不宣称LLM必要或策略已获验证优势。研究效果以研究冻结评估报告为准。

剩余便利性依赖：平台公开赛程枚举；当前使用服务端35场纯赛程/cutoff列表，执行仍经平台身份/时点验证。生产DB/API凭据、真实LLM配置及线上动态动作不影响已交付历史闭环。

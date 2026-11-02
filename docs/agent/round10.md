# Agent 第十轮：前瞻策略合同的离线定界

本轮新增一个纯离线、只读公开元数据的合同检查器，用来判断第九轮策略的工件、时间、域和工具口径是否足以支持后续前瞻研究声明。它不启动预测、不查询外部接口、不创建调度，也不把历史回放成功升级成真实前瞻实验完成。

## 对第九轮现状的判断

第九轮已经验证同一冻结预测器、校准器、模拟工具目录与实际 Agent 执行的一致性。这支持历史受控回放中的工程正确性。切换到全新时间段还缺少以下独立证据：

1. **信息模式仍为历史重建。** 预测器、残差训练和 lambda 验证记录均声明 `historical_reconstruction`。历史可用时间假设与真正收到响应、入库并可被 Agent 使用的时间是不同概念。
2. **工具与费用仍是模拟视图。** `results_view/xg_view` 是同一 Understat 数据源的受控揭示，采用 `simulation_credit/projection_call`。它们不是两次独立真实供应商请求，`.02/.03` 也不是经过测量的 API 价格。
3. **工件绑定尚未覆盖完整部署协议。** 第九轮校准器绑定联合预测器 SHA256 与版本，但没有新的前瞻合同要求的目录内容哈希、工具返回编码哈希及真实来源关系。改变工具、字段、缺失语义或计费范围后，旧校准值的含义需要重新审查。
4. **训练域与新赛事域没有正式合同。** 历史联赛模型与其他年龄组、杯赛、小组赛、国家队或另一实体命名空间之间的迁移，应当作为新的研究问题，不能只凭“都是足球”获得同域资格。
5. **尚未冻结新的选赛和部署窗口。** 当前工件没有给出某个真实未来时间段的冻结时点、结构化选赛规则和预测提前量。文件创建或修改时间也不能替代实际参数冻结、数据可用和 lambda 选择日志。
6. **公开档案能力与策略效果是两种证据。** 项目中存在真实采集和归档模块，不等于已经存在与这两个模拟动作、训练域及返回编码相匹配的 operational 概率决策实验。

这些缺口不要求重写旧模型或修改旧报告状态。本轮保留第九轮工件和状态，通过独立新合同显式列出缺失条件。

**本版是要求相同 operational 时间语义的严格元数据 profile。** `blocked` 不表示任何历史训练模型都不适合前瞻研究。历史训练模型可以在未来数据上接受真正的前瞻评估；这里的阻断表示本项目尚未用独立证据证明历史特征构造、在线来源、工具返回、缺失和成本语义的兼容性。若以后完成这种兼容性验证，应由新的明确协议版本表达可接受的迁移条件，而不是把旧 `historical_reconstruction` 字段改名为 operational。

## 小型 API

实现：`football_agent/prospective_contract.py`，仅依赖 Python 标准库。

```python
from football_agent.prospective_contract import (
    validate_prospective_contract,
    validate_runtime_metadata,
)

readiness = validate_prospective_contract(contract, evidence=public_verifier_metadata)
runtime_check = validate_runtime_metadata(
    contract,
    public_fixture_and_receipt_metadata,
    evidence=public_verifier_metadata,
)
```

两个函数均为纯函数，不读取文件、调用网络、运行模型、检查真实返回内容或修改输入。调用者应先使用已有工件、归档和事件验证器，产生允许公开的元数据，再传入合同检查器。

返回状态有四种：

| 状态 | 含义 |
| --- | --- |
| `retrospective_only` | 当前明确是历史探索；可以列出前瞻缺口，但没有发现核心配置身份或结构错误 |
| `metadata_contract_ready` | 提交的公开元数据满足本版合同约束；不表示已核验原始来源真实性、已启用策略或已完成实验 |
| `runtime_metadata_admissible` | 在满足合同的前提下，这条公开赛事及收据元数据符合固定窗口和域 |
| `blocked` | 结构/工件绑定错误，或请求前瞻模式时还有证据缺口 |

`reasons` 中每项包含 `code/path/message/kind`；`kind` 为 `invalid` 或 `prospective_gap`。历史合同允许保留后一类缺口；将同一合同的模式改为 `prospective_operational` 时，所有缺口都阻断就绪状态。输出另包含合同与证据元数据的确定性 SHA256，以及 `experiment_executed=false`、外部请求数和新增独立比赛数均为零的作用范围说明。

**SHA256 在这里仅绑定内容。** 伪造的来源声明也能计算出 SHA256，因此检查器不宣称独立认证原始来源。`metadata_contract_ready` 只是对调用者提交并署明验证器出处的元数据给出的限定判断。测试中的 ready 路径全部使用明确标注的构造元数据，它们不是新 operational 数据。

## 合同与证据字段

合同 `schema` 固定为 `prospective-policy-contract-v1`，`mode` 为 `retrospective_exploratory` 或 `prospective_operational`。所有对象采用字段白名单，旧配置中的任意额外字段不会被静默透传。

| 合同字段 | 要求 |
| --- | --- |
| `artifacts` | predictor、calibrator、catalog 的内容 SHA256 与版本，逐一对照独立 evidence 映射 |
| `value_target` | 固定为 `empirical_fixed_predictor_one_step_log_loss_reduction`，不表示二步计划或 Bayes 最优信息价值 |
| `policy` | 显式 `horizon=1`、mean/conservative、四维 `lambda_cost`、四维固定报告效用权重及 lambda 来源 |
| `cost_units` | money、requests、latency_ms、tokens 的明确单位；与每个动作成本逐一匹配 |
| `domain` | sport、category、competition_ids、stage_types、entity_namespace；训练、校准、lambda 验证、来源与部署须同域 |
| `freeze_at` | 带时区的政策冻结时点 |
| `deployment` | prediction cutoff 的 `[start,end)` 窗口，以及正数 `min_lead_hours/max_lead_hours` |
| `selection` | `all_eligible` 或 `frozen_fixture_ids`；固定赛事列表、`time_field=prediction_cutoff` 和规则冻结时点 |
| `tools` | 显式动作 ID，须与学习过的动作及绑定目录一致 |
| `information_semantics` | operational 模式、实际 observed/ingested/registered 时间规则、动作前意图归档、赛果单独存储 |

自由文本“会无偏选场”不算结构化选赛规则。`all_eligible` 不接受手选 ID 列表；`frozen_fixture_ids` 必须给出事先冻结的唯一 ID 集合。比赛预测窗口必须严格晚于政策冻结时点。

`evidence` 由以下四部分组成：

- `verifier`：已有验证器名称、版本及报告 SHA256。
- `artifacts`：独立 predictor/calibrator/catalog 身份、可用时点、训练与数据可用上界、information mode；校准器还给出 reference predictor 哈希/版本和 reference catalog 哈希；预测器与目录都声明返回编码哈希。
- `lambda_selection`：所选权重、固定报告权重、来源类型、选择时间、数据可用上界、时间切分清单哈希及所用三个工件哈希。
- `sources`：来源域、operational 归档类别、归档清单哈希、既有验证器的核验时点、一条公开收据时间元数据，以及请求/返回/成本和动作前意图的归档能力。

`max_data_available_at` 可以是已有验证器确认的保守上界，不要求捏造一个精确到秒的原始发布时间；但该上界必须满足时间约束。历史语义下验证过的可用上界仍保留 `historical_reconstruction` 标记，不能直接升级为 operational。

lambda 来源支持 `heldout_temporal_validation` 和 `fixed_preregistered`。前者必须绑定验证数据、切分、域和实际使用的工件，且所有数据在选择时已可用；后者明确没有数据选择依赖，允许先预登记常数，再在实际工件存在后完成最终绑定。策略 lambda 与报告效用权重分别核对，避免在测试结果出现后通过更改费用权重改变结论。

时间检查要求：所有模型/校准/诊断/选择依赖在冻结前已可用；模型拟合不能早于自身训练数据可用上界；绑定工件不能早于所引用工件存在；来源归档证明必须在政策冻结前完成。源收据保持 observation → ingestion → registration → operational availability 的顺序。部署窗口、冻结时点、来源或数据可用上界缺失时，返回具体缺口。

本版对真实费用资格采用明确且保守的范围：动作必须声明 `real_external_request`，请求单位为 `HTTP_request`，各维成本为 finite 的 known/estimated 值。unknown 费用保持 unknown 并阻断本版带完整费用口径的 utility 合同；它不会被当成零。若未来需要仅按请求数评估、保留未计量货币费用的前瞻协议，应新增明确的效用范围版本，而不重标当前工件。

## 运行时公开元数据边界

运行时只接受比赛 ID、赛事/类别/阶段/命名空间、开赛与预测时间、赛程实际可用时间、合同 SHA256、information mode，以及收据的证据 ID、source/action ID 和时间哈希元数据。

运行时会先复验完整合同，再检查比赛域、冻结 ID 或固定选赛规则、预测提前量、证据实际可用时间不晚于 cutoff、source/action 绑定和重复证据 ID。历史合同不会获得 operational runtime 资格。

`outcomes`、标签、比分、hidden returns、未购买 groups、响应正文及 `joint_return_hypothesis` 在任意嵌套层级都会被拒绝。非公开值不会被回显或散列进审计输出。验证器不读取“查询之后才知道的内容”来判断查询之前是否值得调用。

该函数是元数据入场检查，并不替代后续每次动作的预算执行、真实成本记账、provider failure 处理、模型 support 检查或参数校准；这些职责仍属于已有或未来明确接入的执行协议。

## 七折真实工件的只读 readiness audit

机器工件：`var/parallel/agent/round10/prospective-contract-audit-v1.json`。保存了七折的公开 contract/evidence、原始历史模式和仅改模式后的前瞻审计结果，以及所有源文件和验证器源码哈希，根代理可独立离线重算。

输入为第九轮原始 `residual-evsi-v1` 工件，而非测试 fixture：

| 输入 | SHA256 |
| --- | --- |
| `report.json` | `39b18c7f61ed08f381ab69b035affa7836761e368b67258ccbe28f87080c4633` |
| `catalog.json` | `86a88c07ef07244c4d644d9f575457824f2e6a4191cdd8a164c7029fcf117be5` |
| 第九轮独立验证报告 | `33507254d337be3a89009f3c0de4f21ef162d962d86c67457f0633218fbd6d58` |

所有 predictor/calibrator 文件先逐一核对原报告 SHA256，再抽取公开元数据。没有记录的真实工件可用时点、选择墙钟时间、正式训练/部署域、未来窗口与冻结规则均保留空值。没有为该目录提交同域 operational 来源证明，因此 `sources=[]`；这不表示整个项目不存在真实采集档案。

结果为 **7/7 `retrospective_only`，强制改为前瞻模式后 7/7 `blocked`**。原历史合同没有结构或工件身份错误，前瞻模式记录了 14 类缺口，包括 historical mode、模拟费用和视图、缺少 operational 来源、返回编码与目录绑定不全，以及新的域/部署/选赛/冻结条件尚未给出。没有修改任何旧模型、旧校准器或旧状态。

本轮只增加合同与审计，不新增独立比赛、不启动实验、不发出外部请求，也不据此声称下一阶段已具备真实信息价值收益。

## 验证

专项覆盖内容绑定、时间与时区、校准器引用、同域要求、固定选赛、lambda 与报告效用权重分离、历史模式重标失败、来源证明、费用口径、runtime 私有字段拒绝、证据晚于 cutoff，以及无文件/网络副作用。

```powershell
python -m pytest tests/agent/test_prospective_contract.py --import-mode=importlib -q
```

当前专项为 47 项通过；最终项目级检查由第十轮总验收记录。新增源码仅 `football_agent/prospective_contract.py`，测试与本文分别为 `tests/agent/test_prospective_contract.py`、`docs/agent/round10.md`。

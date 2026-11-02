# FootballMind 基线、概率接口与时间回测

**第二轮（2026-09-22）已交付**：不可变预测登记/结果追加/评分、批量特征训练与滚动评估、模型元数据；真实Understat条件历史重建380比赛/290测试场/3480测试预测，前瞻71条另列为68缺标签/3待赛。完整真实结果、版本、运行命令与限制见 [round2-results.md](round2-results.md)，账本说明见 [round2-ledger.md](round2-ledger.md)。本轮产物 `var/parallel/baselines/round2/`。

下文保留第一轮运行说明：当时真实本地档案产物为 `var/parallel/baselines/real-asof-final/`，严格历史可评分样本为 **0**，结果标为功能验证。第一轮覆盖见 `data-audit.md`，运行证据见 `results.md`，原产物未覆盖。

目标固定为常规时间含补时的主胜、平局、客胜，排除加时与点球。预测顺序为 H、D、A。时间均为 UTC aware。

## 直接运行

从 `D:\a1-足彩` 执行。本机现有 Python 环境已具备依赖，不需要数据库或外部网络。

```powershell
python -X utf8 -m pytest tests/research/baselines tests/research/evaluation -q

# 使用已经冻结、验证过的真实时点切片复跑全部基础模型
python -X utf8 -m football_research.baselines run-local --dataset var/parallel/baselines/real-asof-final/dataset.jsonl --output var/parallel/baselines/replay

# 从真实原始档案重新按观测时点截取；先筛未来观测，再做实体推断、去重和冲突隔离
python -X utf8 -m football_research.baselines run-local --archive-root football_spider/platform_data/raw --cutoff 2026-09-20T11:22:02Z --output var/parallel/baselines/fresh-asof

# 独立训练与预测
python -X utf8 -m football_research.baselines train --dataset var/parallel/baselines/real-asof-final/dataset.jsonl --model frequency --as-of 2026-09-20T11:22:02Z --competition spftll:competition:36 --output var/parallel/baselines/model.json
python -X utf8 -m football_research.baselines predict --dataset var/parallel/baselines/real-asof-final/dataset.jsonl --artifact var/parallel/baselines/model.json --match-id spftll:match:3003895 --cutoff 2026-09-20T11:22:02Z --output var/parallel/baselines/prediction.json

# 严格时间滚动回测：当前输入如实返回 NO_ELIGIBLE_TEST_PREDICTIONS
python -X utf8 -m football_research.baselines backtest --dataset var/parallel/baselines/real-asof-final/dataset.jsonl --validation-start 2026-08-01T00:00:00Z --validation-days 14 --test-days 14 --lead-hours 24 1 --output var/parallel/baselines/backtest

# 逐条校验真实概率、训练标签、输入时点、模型文件；并消费已有 Elo 快照
python -X utf8 -m football_research.evaluation.verify_local --run-dir var/parallel/baselines/real-asof-final --output var/parallel/baselines/verification.json --elo-feature football_spider/platform_data/features/9cb07bdf-ca47-43ba-b292-638a52d11d64.parquet

# 只读审计与可携带数据导出
python -X utf8 -m football_research.data_adapters audit --output var/parallel/baselines/audit-latest.json
python -X utf8 -m football_research.data_adapters export --as-of 2026-09-20T11:22:02Z --output var/parallel/baselines/export.jsonl
```

`run-local` 的默认截止时点固定为本次验收时点，属于可重放实验。新赛前预测应明确传入新的 `--cutoff`。`--dataset` 表示消费已冻结输入，不会从更新中的原始目录补充数据；应使用与预测时点相符的投影。不同截止时点的新预测先运行带 `as_of` 的适配器。

## 模型、实际输入与参数

所有模型提供 `fit(rows, as_of=...)`、`predict_proba(row)`、`to_state()`、`from_state()`；状态为 JSON，没有 pickle。模型文件包含实际参数、输入字段、训练截止与样本 ID、输入哈希、算法源码 SHA-256、优化状态；`run-report.json` 包含运行依赖版本。

| CLI / 类型 | 版本 | 输入 | 本次默认参数 | 实测范围 |
|---|---|---|---|---|
| frequency / LeagueFrequency | league-frequency-1.0 | 历史联赛 ID、常规比分 | 对称 Dirichlet alpha=1 | 英超66、德甲40、西甲34个训练结果；12场预测 |
| market / MarketBaseline | market-proportional-1.0 | 同一观测的完整1X2报价、观测时间、来源时间、周期/盘口 | 逆赔率比例去水；最大年龄86400秒 | 18场预测，不使用历史赛果拟合 |
| poisson / PoissonModel | poisson-attack-defence-1.0 | 联赛、球队ID、历史主客进球 | regularization=5；max_iterations=500；tail_tolerance=1e-12；half_life_days=null | 3个联赛分别拟合；12场预测 |
| dixon_coles / DixonColesModel | dixon-coles-1.0 | 同Poisson | 同上；联合估计低比分相关rho，约束修正因子为正 | 3个联赛分别拟合；12场预测 |
| elo / EloMapping | elo-davidson-mapping-1.0 | D已有主客Elo、时点与血缘 | 固定映射home_advantage=65，draw_parameter=0.7，rating_scale=400 | 一条真实特征导出；仅固定映射，无评级算法复制 |
| logistic / LogisticBaseline | multinomial-logistic-1.0 | 显式数值特征白名单、各比赛自身的历史赛前时点与血缘 | regularization=1；random_state=1729；最多1000迭代 | 构造机制测试；真实历史特征待接入 |
| tree / TreeBaseline | decision-tree-1.0 | 同上 | max_depth=3；min_samples_leaf=5；random_state=1729；叶概率alpha=1 | 构造机制测试；真实历史特征待接入 |

Poisson/DC 按联赛建进攻/防守和主客截距。未知球队使用零进攻/防守偏移，未知联赛使用训练进球均值；这些回退记录于状态，稀疏历史应审慎解释。当前时间权重关闭，训练窗口为截止前的扩展历史；历史交锋选择偏差、长时间跨度及球队状态变化均未得到完整联赛数据控制。

Dixon–Coles 低比分相关模型的出处为 [Dixon & Coles (1997), JRSS Series C](https://academic.oup.com/jrsssc/article-abstract/46/2/265/6990546)。本实现额外使用正则化及预测外推时的有效因子范围保护；不宣称复现论文的收益实验。

逻辑回归和树模型的中位数填补、缺失指示、标准化仅由训练窗拟合。三分类训练要求H/D/A均出现。优化失败明确抛出异常，不把未收敛状态标为成功。真实训练特征尚缺，当前这三种特征模型的训练通过 Python API 的映射行接口使用；CLI 的数据文件只包含比赛和报价，尚无历史数值特征数据记录装配。

## Agent 调用与契约

```python
from dataclasses import asdict
from football_research.baselines import PredictionService
from football_research.data_adapters import load_jsonl

dataset = load_jsonl("var/parallel/baselines/real-asof-final/dataset.jsonl")
fixture = next(asdict(m) for m in dataset.matches if m.match_id == "spftll:match:3003895")
service = PredictionService.load("var/parallel/baselines/real-asof-final/models/1ac913520e-poisson.json")
prediction = service.predict(fixture, prediction_cutoff="2026-09-20T11:22:02Z")
payload = prediction.model_dump(mode="json")
```

`payload` 包含：`match_id`、`prediction_cutoff`、三个指定概率字段、`model_version`、`dataset_version`、`input_manifest`、`information_mode`。验证有限非负、每项≤1、总和误差≤1e-10。目标赛程必须在时点前已知、预测早于开球、训练时点不晚于预测、目标不出现在训练集中、信息模式相同。最终标签仅用于评分，不进入目标预测行。

特征快照调用：`service.predict(fixture, prediction_cutoff=..., features=...)`。需提供相同 `match_id/cutoff/mode`、`input_manifest` 和显式 `features_available_at` 或 `elo_available_at`；两个时间字段如果同时存在均要通过。历史特征训练行另需自身的赛前 `prediction_cutoff` 与 `feature_input_manifest`。即使全部数值缺失，也需显式有证据的快照。

Elo 固定映射入口为 `PredictionService.fixed_elo_mapping(as_of=..., dataset_version=...)`，然后传入 `load_feature_snapshot(path)` 得到的特征。其 canonical ID 不做名称猜测；与来源 ID 的连接需要平台已确认映射。验收脚本使用既有平台验收报告中的显式 source ID / canonical ID 对应关系。

公共基础模型复用当前 `football_spider/models/contracts.py`（SCHEMA_VERSION=1.0）。现有包初始化含绝对导入问题，私有 `_contracts.py` 桥接直接加载该现有文件，不修改 `sys.path`。当前 `Prediction` 为此 `Contract` 上的薄验证层；B 尚未发布统一概率类型，接管请求见 `docs/parallel/requests/baselines/001-contracts.md`。不改共享契约或根入口。

## 回测约束与产物

时间分区按 canonical kickoff，匹配同场所有快照。训练截止再下调到最早验证/测试预测时点，防止24小时前预测跨越名义窗口；标签严格早于预测。验证标签在最早测试预测前再次过滤。同场不同快照分区一致，重复快照、比分冲突及改期未解析状态拒绝。

默认扩展训练窗、固定超参数、无融合、无概率校准拟合。代码没有用测试成绩选模型或调整参数。验证成绩单独输出。当前没有自动调参/融合/校准训练器；后续接入必须使用 `TemporalFold` 的对应窗口。

主指标为自然对数 log loss（数值下限1e-15），Brier 为三类别平方误差之和（不除以3），另有准确率。结果按预测提前量分别列出；各类别校准图与分箱计数保存 PNG/JSON。样本缺失时保持空分数，不生成有误导性的真实校准曲线。

逐场对比使用相同 `match_id + prediction_cutoff` 交集，显式报告未配对覆盖。损失差=左模型减右模型；每场先平均快照，再整UTC日历周bootstrap（2000次、固定seed=0）。至少30场且8周才给95%区间；小样本区间为null，方法不覆盖跨周相关和模型选择不确定性。

研究JSONL是内部不可变投影传输（metadata/match/odds），不替代平台公共模型或版本仓库。完整延期/修订时间线应由C/D读取协议接入；当前适配对冲突整场隔离。显式历史重建选项仅用于未核验可见性假设下的功能测试，正式验收没有使用该选项。

## 修改边界与后续接通

新增代码仅在 `football_research/baselines/`、`evaluation/`、`data_adapters/`；测试位于 `tests/research/baselines/`、`evaluation/`；文档位于本目录和F独占协作记录；生成物位于 `var/parallel/baselines/`。未改采集、migration、Agent、展示、根依赖或总入口。

接下来需要：完整联赛赛季与赛程版本、可信的历史结果可用时间、能匹配赛果的赛前赔率快照、带逐行时点与血缘的Elo/数值特征。新数据到位即可用同一回测入口评分。数据库和新来源继续由对应角色维护。

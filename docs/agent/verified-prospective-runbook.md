# 已核实比赛的真实正文采集与独立结果接入

所有命令从 `D:\a1-足彩` 运行。下面的配置属于来源与运行部署；最终使用者提供比赛，
无需拟调查问题、排列特征或挑文章。当前 verified 路径支持已有 spftll 赛前报价、
已显式映射并核实的 UEFA Nations League 2026/27 成年男子联赛阶段比赛。

## 本轮已经实际完成的运行

`var/parallel/agent/round180/live-batch/` 内包含三场原始运行。初始概率来自 Round179
真实报价，新的官方身份只进入本轮，不回填此前历史。

- 哈萨克－法罗群岛：`match-2981549`
- 克罗地亚－西班牙：`match-2981455`
- 苏格兰－斯洛文尼：`match-2981506`

每场目录中的 `report.md` 是中文报告，`analysis.json` 是结构结果，`run.json` 是完整
决策轨迹。市场和官方身份原始数据分别保存在 `prior-archive` 与 `identity-archive`，
本次工具实际返回在 `source-archive`。目录、通用内容、失败、预算耗尽与未解决项均保留。

只读验收不访问网络：

```powershell
python var/parallel/agent/round180/verify_round180.py --test-xml var/parallel/agent/round180/regression.xml --test-xml var/parallel/agent/round180/platform-verification.xml
```

## 用已冻结计划采集

这条命令会实际联网，必须使用新的输出目录。旧采集不要重跑覆盖或换种子挑成功轨迹。
当计划内的开球/赛前截止时间已经过期时会拒绝，不会用现在的信息补装成历史赛前获取。

```powershell
python -m football_research.data_adapters.prospective_text collect --archive var/parallel/agent/round179/prospective/capture --plan var/parallel/agent/round179/prospective/initializations.json --identity-bundle var/parallel/agent/round180/identity/identity-bundle.json --identity-archive var/parallel/agent/round180/identity/archive --allowed-host www.uefa.com --news-provider bing_rss --requests-per-match 3 --seed 180 --output <新的运行目录>
```

本轮对应的一次性执行脚本 `var/parallel/agent/round180/run_collection.py` 已运行完成，
保留预先声明的计划和代码哈希，再次执行会拒绝重复联网。当前文字查询的真实 VoI 尚未
训练，采集使用已存在的有限探索；没有给工具人为添加正收益，也没有因此修改市场概率。

## 赛事实际结束后接入结果

无需手抄比分或制作标签 JSON。以下程序最多四次请求：一次官方总赛程，三次目标比赛页。
它记录意图、失败、原始数据与实际时间，然后核验身份、完场状态、终场时间和 90 分钟比分。
现在提前运行只会保留 PENDING，不会等待赛事，也没有后台定时任务。

```powershell
python -m football_research.data_adapters.verified_fixtures capture-results --identity-bundle var/parallel/agent/round180/identity/identity-bundle.json --identity-archive var/parallel/agent/round180/identity/archive --output var/parallel/agent/round180/identity/results-after-finish --route direct --max-http-requests 4

python -m football_research.data_adapters.prospective_text audit --batch var/parallel/agent/round180/live-batch --result-archive var/parallel/agent/round180/identity/results-after-finish --output var/parallel/agent/round180/episodes-settled.json
```

第二条命令完全离线，再次从原始结果重算，不信任可编辑的核验报告本身。只有 VERIFIED
行可以接入。PENDING、POSTPONED、CANCELLED、DISPUTED、UNCONFIRMED 均不生成标签。
终场信号或比分缺一不可。来源发生失败时保留原始失败，另一次明确采集使用新目录，
不要覆盖或把重复请求当新比赛。

标签接入后可以形成真正的训练候选；三场小样本仍不足以证明信息策略收益。
Round179 和 Round180 是同三场比赛的重复观测，分组拆分必须按比赛身份执行，不能跨训练
与确认集。不能把事后补抓新闻当成这次赛前已经取得的正文。

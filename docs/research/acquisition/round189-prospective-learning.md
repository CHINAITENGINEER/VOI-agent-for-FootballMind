# Round189：把信息获取接到事先登记的真实学习阶段

本项目标是使现有开放信息采集器在未来真实学习中使用正确的预测器和信息价值校准器，
并保留能够回放的实验边界。用户仍只给比赛，研究用途由实验监督层登记，
不要求用户选择伤病、天气或查询问题，也不把分组用途传入 Agent 的候选生成输入。

## 为什么需要这一步

一个工具返回后，预测概率发生变化，不等于获取这条信息降低了实际预测风险。
需要先用 A 组比赛学习正文预测器 f，再在 f 已登记并冻结后采集独立的 B 组比赛，
用 B 的独立结果校准信息获取收益，最后对更晚的 C 组确认效果。
这些比赛必须按真实物理赛事划分；运行次数、文档版本和数据源本地 ID 都不是独立样本。

现有模型支持这条路径，但原 `collect_prepared_batch` 只会构造市场初始概率策略。
这意味着只生成分组说明还不够：采集入口必须实际加载 f 和 critic，并把它们绑定到本次运行。

## 实现

- `prospective_learning.py` 登记官方 `source_event_id`、本地 match ID、球队/赛事事实、
  截止时间、用途和已存在模型的完整文件 SHA。延期可以改变开球钟，不能改变已登记用途。
- `f_train` 继续使用原市场初始概率与有限未知信息探索；`calibration` 实际加载冻结 f；
  `confirmation` 实际加载相同 f 和冻结 critic。未知支持保留为未知，不填人为价值。
- f、critic 的训练比赛必须能映射到显式提供的历史库存，而且分别属于正确用途。
  当前比赛与它们按官方赛事身份隔离，替换本地 ID 不能绕过。
- 统计截止 `training_as_of` 不被称为模型物理冻结时刻。登记时读取现存模型字节，
  采集时以真实时钟复验，要求登记严格早于采集，模型内容与登记一致。
- 批次在第一次请求前保存学习上下文和模型原字节；每个完成的 run 也保留本地副本。
  回放从本地副本加载并核对原 SHA，不依赖训练机器原来的绝对路径。
- 原始身份和市场报价仍由既有原始归档校验。默认未指定学习计划的旧路径保留。

登记范围明确限于调用方提供的历史清单，无法证明一个未披露的历史不存在。
文件 SHA 是可复验的一致性承诺，不是外部可信时间戳。真正的效果确认仍需冻结实验和独立新样本。

## 运行入口

现有初始化与官方身份核验完成后，登记未来采集计划：

```powershell
python -m football_research.data_adapters.prospective_learning prepare `
  --initialization-plan INITIALIZATIONS.json --identity IDENTITIES.json `
  --purpose f_train --output LEARNING_PLAN.json
```

后续 B 组使用 `--purpose calibration --text-model FROZEN_F.json`，
并用可重复的 `--previous-plan` 提供先前登记；C 组再加 `--text-risk-model FROZEN_CRITIC.json`。
这些是监督层命令参数，不是面向最终用户的研究问题输入。

```powershell
python -m football_research.data_adapters.prospective_text collect `
  --archive MARKET_ARCHIVE --plan INITIALIZATIONS.json --output NEW_BATCH `
  --identity-bundle IDENTITIES.json --identity-archive IDENTITY_ARCHIVE `
  --allowed-host www.uefa.com --learning-plan LEARNING_PLAN.json
```

原 `--news-provider`、`--source-profile-version`、`--news-projection-version`、
`--evidence-version` 继续按已登记能力选择。回放入口不变。

已有真实采集可以显式登记为已暴露的 f 训练身份：

```powershell
python -m football_research.data_adapters.prospective_learning import-training-exposure `
  --run EXISTING_RUN_FOLDER --run ANOTHER_RUN_FOLDER --output TRAINING_EXPOSURE.json
```

此入口重新核验 run/config 摘要、真实完成时间、官方身份原始字节与市场报价原始字节。
它不追认历史预登记，不补采样概率，也不把已登记的校准/确认比赛改为训练。
监督记录是否可用仍由原语料与独立赛果网关判断。之后可以把输出作为 `--previous-plan`。

本轮已实际导入已有 9 个 run，归并为 6 场官方比赛；登记时间是北京时间
2026-10-07 02:51:14.217580，保存于 `var/parallel/agent/round189/existing-training-exposure.json`。
这些是未来可使用的训练身份，不是 6 场已具备有效标签的训练样本，更不是新增比赛。

## 本轮真实数据结果

截至北京时间 2026-10-07 02:37:20，旧市场归档中有 15 场距离开球超过半小时，
其中 14 场在未来 72 小时，但没有可直接新增的合格独立比赛。
美国—加拿大、芝加哥—温哥华虽有历史常规时间赔率，均超过初始化时的一小时新鲜度约束。
02:32:55 的实际市场更新返回 HTTP 200、`data=[]`；这不算新增数据。

另外两次官方页面发现返回 403 和 301，未得到足够的官方比赛身份、成年男子组别及赛制证据。
共 3 次新 HTTP 请求，未执行新采集批次、未轮询未到期赛果、未训练真实模型。
现有已核实欧国联赛程的下一比赛日是 11 月 12 日；52 个后续官方 URL 只是已观察的引用，
不计为已绑定或可立即运行的比赛。

真实监督仍是此前已归档的 1 场独立有结果比赛、2 条实际响应监督记录。
已有六场比赛不能事后分成“f 冻结后采集的 B 组”；旧数据的使用必须保留实际发生时钟。
本轮机制测试不能替代真实查询收益、类别覆盖或统计效力证据。

本项验收、测试计数和独立审查见 `var/parallel/agent/round189/acceptance.json`。

最终相关回归为 1007 项通过、0 失败、0 跳过；新增 17 项契约检查和 7 项端到端检查
已包含在其中。独立审查额外核验旧档导入的 9 个边界，未发现限定范围内剩余阻断。
`python var/parallel/agent/round189/verify_round189.py` 已通过，总验收状态为 `passed`。

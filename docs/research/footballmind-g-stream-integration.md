# FootballMind 信息流与 `g` 研究证据

本次接入把 `footballmind-information-stream/` 作为不可变原始包，通过
`tools/prepare_footballmind_information_panel.py` 生成派生面板，再交给现有的
`f`/`g` 历史重建内核。适配器不会修改原始 JSONL、HTML 或结果文件。

## 时间语义

原始包中的正文、菜单和查询记录是在赛后重建的。正文的来源发布时间可以作为
赛前可用性的上界，但正文版本没有被证明在原时刻可见；菜单不是原始 Agent 菜单，
`selected_by_collector` 也不是历史策略轨迹。

派生层保留两种结果时钟：

- `observed`：结果文件在事后真正被观测的时间；
- `source_reconstruction`：来源记录中重建的结果可用时间，用于离线按时间划分
  A/B/C，不能解释为实际在线观测时间。

默认使用 `source_reconstruction`，并在 `label_provenance` 中同时保存
`actual_result_observed_at`、`source_result_available_at`、`result_clock` 和
`result_available_at_basis`。旧格式案例继续要求结果行的主客队、联赛、开球时间和
结果可用时间逐项严格相等；只有 `footballmind-information-stream/v1` 分支允许
外部结果缺少供应商 ID，并通过物理比赛、队名、联赛和开球时间绑定。

## 本批离线结果

```text
fixtures: 26
documents: 62
discovery menus: 31
attempts: 62
context snapshots: 26
results: 26
```

正文 URL 与正文哈希和此前的 26 场语料完全重合，独立比赛数没有增加；这批数据
增加的是候选菜单、读取尝试和溯源记录。因此它适合验证信息揭示回放、内容对 `f`
的响应、时间泄漏门禁和数据绑定，不适合宣称在线 OPE 或跨联赛泛化。

固定 `f` 和 A-only 选择 `f` 的两次 B 组留一验证都选择了常数 `g` 基线。A-only
选择允许正文影响预测（选择 `ridge_0.01`），但 `g` 仍无法在仅 5 场 B 组上可靠
区分候选。主运行继续保持 `empirical_g=deferred`，不把这批事后重建数据接入
线上动作选择。

## 可复核命令

```powershell
python tools/validate_footballmind_information_stream.py footballmind-information-stream `
  --report var/submission/p8-g-stream-v3/validation-report.json

python tools/prepare_footballmind_information_panel.py footballmind-information-stream `
  --output var/submission/p8-g-stream-v4 --result-clock source_reconstruction

python tools/train_footballmind_single_case.py `
  --cases var/submission/p8-g-stream-v3/cases.json `
  --output var/submission/p8-g-stream-v3/model-selected `
  --dataset-version footballmind-information-stream-20261008-reconstructed-selected-v3 `
  --g-fit-scheme b_group_loo_v1 --f-fit-scheme a_group_loo_v1

python tools/audit_footballmind_information_stream.py footballmind-information-stream `
  var/submission/p8-g-stream-v3 `
  --fixed-report var/submission/p8-g-stream-v3/model-fixed/report.json `
  --selected-report var/submission/p8-g-stream-v3/model-selected/report.json `
  --prior-cases var/submission/p1-single-match/corpus-probe/cases-with-interviews-routed.json `
  --output var/submission/p8-g-stream-v3/stream-audit-v2.json
```

最终机器可读结论位于
`var/submission/p8-g-stream-v3/stream-audit-v2.json`：这批资料可用于可重放的离线
揭示实验，当前不应把经验 `g` 接入主决策链。

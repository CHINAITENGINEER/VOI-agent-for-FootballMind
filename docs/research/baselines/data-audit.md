# FootballMind 真实数据覆盖与时间语义审计

生成时间（UTC）：2026-09-20T11:34:12.420308+00:00

冻结数据版本：`spftll-research-1.0.0:a2d006c4ab19b2e1a25ce50d204b58214fec4f305d79349761189415f3e8a76e`

结论：现有档案支持已观测历史赛果训练与后续赛前预测；历史可见性完整的常规时间正式回测样本见下表。
历史重建仅用于功能验证，结果公布延迟和赛程提前可见性是显式假设，不代表已核实的历史系统状态。

| 项目 | 实际数量 |
|---|---:|
| SQLite 当前比赛 | 29 |
| SQLite 当前完整比分 | 1 |
| SQLite 历史行 / 去重比赛 | 1640 / 1503 |
| 验证哈希的原始观测 | 61 |
| 导出比赛 / 常规时间标签 | 714 / 317 |
| 完整胜平负赔率快照 | 127 |
| 严格 operational 历史回测合格比赛 | 0 |
| 最新输入时点后已知五大联赛赛程 | 19 |

## 联赛和年份覆盖

来源没有赛季字段，因此年份分布只表示开赛年份。没有完整赛程分母，缺赛数量保持未知。

| 来源赛事 ID | 来源名称 | 比赛 | 常规时间赛果 | UTC 首场 | UTC 末场 |
|---|---|---:|---:|---|---|
| spftll:competition:103 | 欧冠杯 | 15 | 0 | 2025-03-04 | 2026-09-10 |
| spftll:competition:11 | 法甲 | 10 | 8 | 2026-09-12 | 2026-09-20 |
| spftll:competition:113 | 欧罗巴杯 | 16 | 0 | 2026-07-23 | 2026-09-17 |
| spftll:competition:12 | 法乙 | 2 | 0 | 2026-09-14 | 2026-09-18 |
| spftll:competition:122 | 瑞典甲 | 6 | 0 | 2005-04-24 | 2019-08-11 |
| spftll:competition:13 | 芬超 | 3 | 0 | 2026-09-13 | 2026-09-18 |
| spftll:competition:1312 | 德电信杯 | 1 | 0 | 2026-08-15 | 2026-08-15 |
| spftll:competition:1366 | 国际赛 | 3 | 0 | 2026-09-24 | 2026-09-24 |
| spftll:competition:142 | 意丙 | 2 | 0 | 2004-12-12 | 2005-04-17 |
| spftll:competition:144 | 英锦标赛 | 3 | 0 | 2026-09-22 | 2026-09-22 |
| spftll:competition:15 | 韩职 | 2 | 0 | 2026-09-19 | 2026-09-20 |
| spftll:competition:16 | 荷甲 | 44 | 0 | 2016-01-24 | 2026-09-20 |
| spftll:competition:17 | 荷乙 | 1 | 0 | 2026-09-18 | 2026-09-18 |
| spftll:competition:1864 | 欧国联 | 5 | 0 | 2026-09-24 | 2026-09-24 |
| spftll:competition:192 | 亚冠精英 | 7 | 0 | 2026-09-14 | 2026-09-16 |
| spftll:competition:21 | 美职 / 美职业 | 4 | 0 | 2026-09-13 | 2026-10-07 |
| spftll:competition:2187 | 欧协联 | 10 | 0 | 2026-05-07 | 2026-08-27 |
| spftll:competition:22 | 挪超 | 6 | 0 | 2026-09-12 | 2026-09-20 |
| spftll:competition:23 | 葡超 | 6 | 0 | 2026-09-12 | 2026-09-20 |
| spftll:competition:25 | 日职 | 5 | 0 | 2026-09-13 | 2026-09-20 |
| spftll:competition:26 | 瑞典超 / 瑞超 | 41 | 0 | 2010-03-21 | 2026-09-20 |
| spftll:competition:284 | 日乙 | 2 | 0 | 2026-09-13 | 2026-09-19 |
| spftll:competition:31 | 西甲 | 89 | 83 | 2010-01-17 | 2026-10-21 |
| spftll:competition:33 | 西乙 | 9 | 0 | 2026-05-02 | 2026-06-20 |
| spftll:competition:34 | 意甲 | 73 | 68 | 2011-10-02 | 2026-09-20 |
| spftll:competition:350 | 亚冠乙 | 1 | 0 | 2026-09-17 | 2026-09-17 |
| spftll:competition:36 | 英超 | 114 | 110 | 2010-03-14 | 2026-09-20 |
| spftll:competition:37 | 英冠 | 14 | 0 | 2006-03-04 | 2026-09-20 |
| spftll:competition:385 | 英社盾 | 1 | 0 | 2026-08-16 | 2026-08-16 |
| spftll:competition:4 | 巴甲 / 巴西甲 | 4 | 0 | 2026-09-13 | 2026-09-20 |
| spftll:competition:40 | 意乙 | 15 | 0 | 2021-09-17 | 2026-05-08 |
| spftll:competition:401 | 亚运男足 | 5 | 0 | 2026-09-15 | 2026-09-20 |
| spftll:competition:405 | 亚运女足 | 3 | 0 | 2026-09-14 | 2026-09-21 |
| spftll:competition:41 | 球会友谊 | 94 | 0 | 2013-07-27 | 2026-08-16 |
| spftll:competition:51 | 德国杯 | 3 | 0 | 2026-04-22 | 2026-08-22 |
| spftll:competition:58 | 荷超杯 | 1 | 0 | 2026-08-02 | 2026-08-02 |
| spftll:competition:59 | 荷兰杯 | 1 | 0 | 2024-01-17 | 2024-01-17 |
| spftll:competition:704 | 西超杯 | 3 | 0 | 2020-01-12 | 2026-01-08 |
| spftll:competition:73 | 瑞典杯 | 2 | 0 | 2026-03-22 | 2026-08-27 |
| spftll:competition:8 | 德甲 | 51 | 48 | 2016-11-18 | 2026-09-20 |
| spftll:competition:81 | 西杯 | 6 | 0 | 2010-01-07 | 2024-01-18 |
| spftll:competition:83 | 意杯 | 9 | 0 | 2014-08-17 | 2026-09-15 |
| spftll:competition:84 | 英联杯 | 14 | 0 | 2008-08-26 | 2026-09-17 |
| spftll:competition:89 | 解放者杯 | 4 | 0 | 2026-09-15 | 2026-09-18 |
| spftll:competition:9 | 德乙 | 1 | 0 | 2026-09-18 | 2026-09-18 |
| spftll:competition:90 | 英足总杯 | 3 | 0 | 2026-04-25 | 2026-05-16 |

## 时间、缺失、重复和实体语义

- 时间统一为 UTC；来源无时区的比赛时间按既有来源契约使用 Asia/Shanghai。
- 最终比分、排名、统计与赔率只在真实观测后可用；SQLite created_time/updated_time 不作为字段首次可见时间。
- 历史交锋和近况是对当前目标球队有选择的样本，具有选择偏差；重复历史条目按来源比赛 ID 去重。
- 同一 ID 的开赛时间、球队、赛事或最终比分冲突整场隔离，保留审计原因；延期旧版本不被静默覆盖用于回放。
- 球队使用来源 ID；同 ID 多显示名作为别名证据输出，不做模糊名称合并，不混入女子/青年/杯赛未知阶段赛果。
- 当前英超、德甲、西甲、意甲、法甲常规时间规则复用既有 normalizer 的接受范围；任何非空加时字段再排除。
- 赔率完整三项必须来自同一观测；源更新时间没有证据时留空，不把收盘或当前价格回填到较早时点。
- 每个输入保留内容 SHA-256、观测 ID 和观测时间；冻结 JSONL 与版本可用于重复实验。

## 排除与外部条件

身份/时间/赛果冲突隔离：`[{"match_id": "3049657", "identity_versions": 2, "result_versions": 1, "reason": "Quarantined: replay needs per-version history reader"}]`

解析或报价缺失问题数量：5。完整细节见同名 JSON 审计。

Understat 现有文件为 HTML 能力探测；ClubElo、FBref 探测没有可用数据集。尚缺历史赛程版本、来源公布时间、完整联赛赛季和历史赔率序列。

真实数据预测暂未有相应最终赛果供评分。若运行显式重建回测，报告应称为真实记录上的功能验证，不能据此宣称稳定预测能力。

## 冻结的预测时点切片

整体档案审计冻结了 714 场比赛、317 个常规时间标签和 127 个完整赔率快照；该全量清单供覆盖审计使用。真实预测闭环另外在 UTC `2026-09-20T11:22:02+00:00` 切片：先排除较晚观测，再进行来源 ID 推断、去重、赛果和身份冲突检查以及输入清单聚合，得到 **263 场比赛、140 个已可用常规时间标签、56 个完整赔率快照**。此切片的数据版本为 `spftll-research-1.0.0:4c0b2f8761f834ec23f42fb04cad37c3cebb8b1dc068f0aa52afa37ee1116680`。严格 operational 历史回测合格比赛仍为 **0**。

这样较晚采集的新增赛果、球队别名、改期或错误修订不会影响较早预测时点，也不会进入该时点的输入血缘。对应文件为 `var/parallel/baselines/audit-cutoff.json` 和 `audit-cutoff-dataset.jsonl`。完整审计的 `frozen/dataset.jsonl` 继续保留原始 a2d006 版本。

复核命令（工作目录为仓库根）：

```powershell
python -X utf8 -m football_research.data_adapters audit --workspace . --as-of 2026-09-20T11:22:02+00:00 --output var/parallel/baselines/audit-cutoff.json
python -X utf8 -m football_research.data_adapters export --workspace . --as-of 2026-09-20T11:22:02+00:00 --output var/parallel/baselines/audit-cutoff-dataset.jsonl
python -X utf8 -m pytest tests/research/baselines/test_data_adapters.py -q
```

适配器验证：17 项构造样本测试通过，覆盖未来冲突隔离、真实观测可见性、零比分、常规/加时边界、完整同次赔率、哈希校验、JSONL 重放和 SQLite 只读。构造样本未加入真实导出。

Elo 只消费既有 FeatureStore 导出的 Parquet。`load_feature_snapshot` 校验原始输入清单 SHA-256、比赛 ID/时点/模式、参数与 Elo 血缘，记录 Parquet 文件哈希并保留原 canonical ID；不进行球队评级更新。`features_available_at` 是由清单证明的输入逻辑可用时间上界，非 Parquet 物理生成时间；该读接口没有重新运行仓库特征计算。既有 `9cb07bdf-ca47-43ba-b292-638a52d11d64.parquet` 实读成功，UTC 截止时点 `2026-09-20T11:21:58.902054+00:00`，主客 Elo 分别为 `1600.8830423836503`、`1463.5164538906354`，覆盖仍为不完整历史样本。

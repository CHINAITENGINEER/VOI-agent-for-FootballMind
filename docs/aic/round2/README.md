# round2 交付物清单与核验入口

生成时间：2026-10-25（北京时间）
性质：第二轮独立审计与最终路演方案。**未覆盖**第一轮报告（对话产出）、原始实验结果或项目源代码。

## 文件清单

| 文件 | 大小 | SHA-256（前 16 位） | 对应交付物 |
|---|---:|---|---|
| `01-audit-report.md` | 29,250 B | `2A9698F5837704E4` | 交付物 1：第二轮审计报告 |
| `02-evidence-and-literature.md` | 26,585 B | `080BFDA8E6C75D6F` | 交付物 2、3：事实证据清单 + 文献对照矩阵 |
| `03-innovation-and-pitch.md` | 35,561 B | `4F0BA8DACC58BF1C` | 交付物 4、5、6：创新点 + PPT + 讲稿 + 视频脚本 |
| `04-defense-and-action.md` | 29,157 B | `349DB65806662D0E` | 交付物 7、8：问答手册 + 行动清单 + 版本记录 |

## 逐项核验入口

| 若要核验 | 打开 | 再对照 |
|---|---|---|
| 某个数字对不对 | `01-audit-report.md` §二（数字一致性审计表） | 表中"原始文件"列给出的路径 |
| 某篇文献是否真实 | `02-evidence-and-literature.md` §4 | 表中链接的 arXiv abs 页 / DOI |
| 某项能力属于哪一级证据 | `02-evidence-and-literature.md` §3 | §3 各表"文件"列 |
| f / g 的数学定义 | `02-evidence-and-literature.md` §2 | `football_research\acquisition\state_candidate_gain.py`、`entity_evidence_updates.py` |
| PPT 某页该放什么 | `03-innovation-and-pitch.md` §三 | — |
| 补做哪个实验、怎么做 | `03-innovation-and-pitch.md` §五 | `E:\FootballMind\experiments\postread-reversal-corrected-20261025-v2\run-script.py` |
| 评委问什么怎么答 | `04-defense-and-action.md` §一、§二 | 各题"证据"行 |
| 今天先做什么 | `04-defense-and-action.md` §三 P0 | — |

## 强约束（本轮新增，后续所有材料适用）

1. 所有数字必须标注：**A 集/ B 集** + **比赛级/行级** + 预算 + 指标。禁止跨集累加或比较绝对数值。
2. 禁止表述："g 的精度是 0.544"、"首次提出"、"已提高准确率"、"最优停止"、"跨语言能力"、"稳定盈利"、"多智能体协作形成算法创新"。
3. 引用两步前瞻的 `loss+cost` 优势时，**必须同页并列**纯 log loss 的反证（0.900448 vs 0.908912）。
4. 引用 700 个对照点零误差时，**必须说明它是复现一致性证据，不是模型有效性证据**。
5. 佐证材料按官方定义属**资质类**（专利/软著/论文/合作/用户反馈/试点证明）；项目暂无此类材料，不得用实验日志替代。

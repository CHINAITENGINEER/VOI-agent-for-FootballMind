# FootballMind：E 盘比赛与证据语料库

本轮目标是把少量比赛、少量文章扩展成可重复使用的数据与向量底座。实际统计以 `E:/FootballMind/corpus-v1/reports/delivery.json` 为准；本轮不把下载、向量化称为重新训练 f 或 g。

## 文件位置

- 数据根目录：`E:/FootballMind/corpus-v1`。
- 比赛公开字段：`structured/matches.jsonl`。
- 独立目标：`structured/labels/results.jsonl`；赛后统计另存 `structured/labels/postmatch_statistics.jsonl`。
- 历史赔率：`structured/market_snapshots.jsonl`，区分 closing 与来源其他快照，准确报价时刻未知。
- 新闻发现库存：`news/discovery.jsonl`，保留选中与未选中 URL。
- 原始网页：`raw/news/`；正文：`news/bodies/`；文章记录：`news/articles.jsonl`；失败：`news/failures.jsonl`。
- 比赛—文章候选关联：`research/match_article_candidates.jsonl`。这是待核验候选关系，不能直接当作本场事实。
- 按比赛分组的候选文章包：`research/match_article_bags.jsonl`，不含赛果，可用于后续比较不同读取顺序。
- 向量、行号映射和特征定义：`vectors/`。
- 本地编码器和下载来源：`models/`。缓存和临时目录也在 E 盘。

## 数据选择

比赛来自 Football-Data.co.uk 五大联赛 CSV，从 2014/15 到已发布的 2026/27 部分赛季。下载原件、来源 URL、响应状态和摘要与清洗记录一同保存。未以本场赛后射门、进球等字段构造本场特征。

新闻来自 Liverpool、Arsenal、Chelsea、Crystal Palace 公开官网。网站地图提供发现库存；按站点与主题分层，对 URL 哈希作确定性抽样。主要覆盖伤停、赛前发布会、首发、训练和比赛前瞻，也保留其他主题。抽样不使用赛果或 gain。不是把所有新闻、所有球队、所有日期都收齐。

正文要求命中站点正文 DOM，不能用导航或搜索摘要代替；单独呈现的导语只有在实际 DOM 中存在同文段落时才补入正文。失败、访问限制和空壳页面单列。女足、青年队、票务和训练相册等材料保留，男子英超候选关联另作类型筛选和适用性核验。

这些是当前取得的历史页面版本。`datePublished` 是来源声明，不能证明当时页面版本与当前一致。原文、请求记录与抓取时间保留；历史首次可见时间未知，不能改写为一条当时实际执行的 Agent 轨迹。这个限制不阻断历史研究，但必须与严格前瞻验证分层。

## 向量与信息边界

编码器为 `sentence-transformers/all-MiniLM-L6-v2`，固定 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，本地 CPU 推理，384 维。不是本轮训练的新神经网络。英文官网正文使用该英文编码器；跨语言检索不在本批验收范围。

标题与正文分别编码。全文按最多 254 个内容 token 分段，步长 222，相邻段重叠 32 token；不把长文静默截成第一段。每段 attention-mask mean pooling 后 L2 归一化，文章向量为分段向量之和再归一化。原文始终保留，向量不能代替事实抽取或引文核验。

`ExternalFootballCorpus.unread_candidate` 只提供标题元数据及标题向量。`read_evidence` 要求调用方显式登记已读取文章，才提供正文和正文向量。历史文章标题是否在原决策菜单中可见尚未证实，历史模拟必须明确菜单假设。这个适配器用于接线，不自动接管既有线上 Agent，也不是对恶意调用方的安全隔离。

数值特征 11 维，包含 Elo、历史比赛数与最近五场积分/进失球。按日期处理，同一天所有比赛的特征先生成，再更新当天赛果，避免同场或同日赛果进入特征。Elo 固定参数未调参。标准化尚未拟合，后续只能在训练分区拟合。赛季与球队名称映射仍应在跨源融合时审阅。

## 重现命令（项目根目录运行）

```powershell
$env:PYTHONPATH = '.'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HOME = 'E:/FootballMind/corpus-v1/cache/huggingface'
$env:TEMP = 'E:/FootballMind/corpus-v1/tmp'
$env:TMP = $env:TEMP

python tools/build_e_drive_football_corpus.py --stage structured --root E:/FootballMind/corpus-v1 --first-season 2014 --last-season 2026
python tools/build_e_drive_football_corpus.py --stage model --root E:/FootballMind/corpus-v1
# 已存在发现清单时复用同一抽样；网络收集不要与离线重解析同时运行。
python tools/build_e_drive_news_corpus.py --root E:/FootballMind/corpus-v1 --per-host 220
python tools/build_e_drive_news_corpus.py --root E:/FootballMind/corpus-v1 --offline-reparse
python tools/vectorize_e_drive_football_corpus.py --root E:/FootballMind/corpus-v1 --stage all
python tools/validate_e_drive_football_corpus.py --root E:/FootballMind/corpus-v1
python tools/search_e_drive_football_corpus.py 'injured player ruled out before the next match' --root E:/FootballMind/corpus-v1 --top-k 5
python tools/search_e_drive_football_corpus.py 'injury team news and starting lineup' --root E:/FootballMind/corpus-v1 --match-id 'fd:E0:2425:2025-01-18:Brentford:Liverpool'
```

成功且摘要匹配的原件复用，不再请求。离线重解析不请求网络。网络失败不被假设成永久失败，因此重新执行网络收集可能再次请求失败 URL；无须为重新向量化重跑采集。

## 接下来如何用于 f 与 g

1. 先用结构化时间划分评估基础 f，再评估文本增量，不能把赛果文件与正文拼到模型输入。现有结构化时间划分不等于已经完成文本研究分组。
2. 核验文章与目标比赛、性别/年龄组、时间范围的关联。保留弱证据、无关和重复文章，避免只收录会改变概率的正例。
3. 固定同一版本 f，构建读取前状态、候选菜单、读取后状态与预测。正文向量只能在对应读取动作后使用。f 不响应时分清：未进入模型、没有支持特征、实际计算后预测不变。
4. 用独立赛果计算有正有负也可以为零的损失变化，再训练 g 的即时价值及后续价值。没有收益标签的文章不能直接算 g 监督样本。
5. 按比赛、重复文章簇及时间分组，避免同场不同文章或转载跨训练/验证。负收益不能用于删除样本，未知收益也不能填零。

批量语料能解决覆盖与表示问题，但不能单凭语料规模证明 f 准确或 g 会选择。下一阶段的验收应看真实文本增量与信息选择对照，而不是继续累计下载字节数。

## 首批实际验收（2026-10-20）

21,886 场、65 个 CSV、679 篇不同正文、2,092 个正文片段；标题与正文向量均为 384 维。官网类别/关键词过滤后，候选关联涉及 166 场，其中 82 场至少两篇、35 场至少三篇。这里的关联仍待语义适用性复核。

本地带比赛范围的查询已实际返回 Brentford–Liverpool（2025-01-18）的官方首发、Jota 赛前身体状态评估，以及较早的赛后反应文章。第三类保留为背景/低相关候选，不强行标记为本场有效事实。全库检索会包含女足等保留材料，比赛用途应使用 `--match-id` 或调用方明确的候选集合，不能把全库相似度当作当前比赛的信息价值。

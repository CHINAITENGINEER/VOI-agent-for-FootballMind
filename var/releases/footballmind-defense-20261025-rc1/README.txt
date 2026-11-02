FootballMind 答辩冻结研究包

运行目录可移动，依赖版本见 requirements-lock.txt。不会自动下载模型或访问网络。
python footballmind_defense.py --command verify
python footballmind_defense.py --command catalog
python footballmind_defense.py --command run --output run.json
python footballmind_defense.py --command run --model state_ridge --settle --output comparison.json

默认set f+fulltext_mlp_seed20261027，阈值0.01，预算3，为全文研究候选。
原开发集完整策略优选state_ridge阈值0也保留；它不按候选全文区分同状态文章，不声称最佳全文筛选。
仅加载本地已有正文和对应Qwen向量，f/g重新计算真实分数及概率；不是播放旧预测JSON。
默认选择开发组按日期及比赛ID排序的第一场，没有按胜负或gain选案例。
run_case/catalog/get_case不加载赛果字段；显式settle_run才读取settlement/outcomes.json。
manifest验证失败、缺失正文/向量、未知模型、依赖不符均报错，无随机假数据或固定概率fallback。

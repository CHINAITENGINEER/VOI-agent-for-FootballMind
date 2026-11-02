# Round197：可发现新闻进入决策菜单，冻结 f/g 接入独立比赛

本轮继续落实“只输入比赛，机器依据当前可见信息选择下一次查询”。完成两个主线缺口：真实官方新闻链接可以进入候选菜单；同一冻结 f/g 可以通过正式独立 C 登记进入公开 collector，完成选择、返回、更新和回放。真实训练数据仍不足，没有发布模型效果结论。

## 实际数据请求及阻点

在请求前保存 `var/parallel/agent/round197/baseline-and-budget.json`，冻结三场已观察、未登记、七天内的芬兰比赛及请求上限，顺序按开球时间：FC Inter–KuPS（3085213）、VPS–IF Gnistan（3085214）、AC Oulu–HJK（3085215）。沿正式 `footballmind start-batch` 入口运行。

三次 GetInfo 都为 HTTP 200、业务 code=0、data=null。实际请求共 3 次，均停在准备阶段，没有市场快照、Actor 请求、正文、独立标签或训练样本。保留所有原始回执，不用旧赛程中的信息伪装最新详情，不重跑已经登记的其他比赛，不追加自动重试。

三次空返回不能增加独立物理比赛支持。真实库存仍为 14 run、10 场物理比赛、43 次实际 Actor 返回；6 场独立有标签比赛，类别为主胜0/平2/客胜4，35 有标签前缀、22 待标签前缀。真实 f 仍 not_ready，真实 B=0，真实 g 未拟合，真实 C 未启动。

这次运行还暴露出当前入口对市场初始化的依赖：官方已知赛程不等于详情和赔率接口已供给。后续应把来源供应状态和算法能力分开报告，优先解决供给与合法先验初始化，而不是放宽训练支持门槛或把空返回计为样本。

## 新闻候选 v10

R196 保存的真实官方首页有 16 条 `/uutiset/YYYY/MM/DD/slug` 链接；v9 只识别该来源 `/uutiset/slug`，通用日期规则又只认根目录日期路径，实际日期文章全部被漏掉。

新增显式 v10，保留 v1–v9 行为。只在 profile4 已核验的芬兰赛事共享来源入口接纳同域、已观察的有效日期路径；父页必须成功且时间在决策截止内。遵循原始 DOM 次序及既有上限，16 条中前12条进入菜单、后4条截断。没有按主题、球队名称、比赛 ID 或已知结果对这些链接设置信息价值。

12 条入选链接均能通过实际 NewsDiscoveryTools.prepare 和严格 system TLS 授权；删除父页后授权失败。零网络复验只证明这些链接可被选择、可被合法读取，不能证明文章已经读取、与比赛相关或者能够提升预测。有限均匀探索也不保证在 4 请求内选中文章。

未来 profile4 准备与 collector 默认 v10。既有归档继续按各自冻结的 v1–v9 重放。

## 正式 C 入口

新增 `context_confirmation.prepare_context_confirmation` 和 `validate_context_confirmation`：

1. 使用已保存的 f/g 字节，核对两者共同的 f、来源类型、训练截止和实际拟合时点；登记必须先于执行。
2. 检查 A 预测器训练、B 收益模型训练和当前 C 的物理比赛互不重叠；B 必须绑定同一个 f。比赛换本地 ID 或重复登记不会产生新的独立 C。
3. 登记实际 seed、请求与探索预算、损失/成本系数、工具成本、来源能力、传输路由、候选和正文版本。
4. 正式 collector 在请求前按实际配置重验，将 f/g 与登记快照保存到本地。运行器依据 g 的预期有符号收益减成本作选择，缺支持仍 UNKNOWN，仅有限探索。
5. 回放从归档重建来源、候选、预算、种子、投影和模型。g 原字节保存，不因重新排版 JSON 改变登记 SHA。
6. 统一库存识别新 B/C，后续 A/B 不能把当前 C 比赛改成训练用途。原 C 上下文不进入 Agent 可见状态，确认标签也不回填收益监督。

公开采集接口新增 `context_confirmation_path`、`context_value_model_path`，与已有 `context_registration_path`（B）互斥，共用 `context_predictor_path`。CLI 对应 `--context-confirmation`、`--context-value-model`、`--context-predictor`。准备登记 API 的 `policy_spec` 由 `context_collection.confirmation_policy_spec` 按实际 collector 参数生成，防止登记与运行采用不同默认设置。

此 C 是独立比赛的冻结策略执行入口。它还不等于一份确定样本量、策略分配、基线比较及统计检验的效果试验登记；`independent_effect_confirmed` 和 `production_ready` 都为 false。真实 f 未就绪时不启用真实 B/C。

## 可回放的构造执行

`var/parallel/agent/round197/confirmation-mechanism/` 保存一次完整 C 构造执行。模型实际拟合，调用正式 collector，没有替换预测方法或注入运行器策略。只模拟 2027-02-10 时钟和来源返回，实际网络请求为零。

本次三个动作依次为 UNKNOWN 搜索探索、UNKNOWN 正文探索、g 计算出正净收益后选择读取。首篇正文使 f 从市场先验更新，全部返回保留原始字节和实际模拟返回时点，完整运行可零网精确重放。为隔离选择机制，成本换算 λ_money=0；这不是推荐的实际研究成本，更不是真实预测收益。最后一次重复内容只有浮点量级概率差异，不当成实质信息改善。

生成器脚本使用构造测试帮助函数；已保存 run 的正式回放不依赖测试帮助函数。无需重造、重采、改写任何 R196 运行。

## 验证和后续优先级

最终相关合并回归记录在 `var/parallel/agent/round197/regression-final.xml`，总验收及源码/产物摘要在同目录 `acceptance.json`。复验命令：

```powershell
python var/parallel/agent/round197/verify_round197.py
```

下一关键工作仍是合法赛前信息和独立赛果供给：优先处理“无市场详情时系统无法启动”的初始化耦合，再积累足够的 A 比赛、冻结 f、采独立 B、训练 g，最后进入独立策略效果试验。现有最早独立赛果检查时点为北京时间10月8日02:10（旧 Gnistan–Inter），然后10月9日01:10与02:10（HJK–VPS、KuPS–AC Oulu）；到时必须读取真实完场证据，不按时钟自动赋标签。

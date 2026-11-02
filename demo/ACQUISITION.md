# 第三轮主动信息获取展示

独立入口保留此前工具回放服务：

```powershell
Set-Location -LiteralPath 'D:\a1-足彩'
python -m demo.acquisition_server --port 8768 --mode synthetic_test
```

打开 `http://127.0.0.1:8768/`，默认从服务端目录填入比赛与历史截止点。选择“全候选数值选择”后新建运行，逐步执行，或“运行至停止”。页面依次展示公开 mask、部分预测器概率、候选可用性、预计损失改善、成本单位及换算、净增益、估计不确定性/支持度，再展示执行和停止事件。概率与数值来自运行器返回，页面不计算新的预测或增益。

`--mode synthetic_test` 是构造替身，明确显示测试标签。场景可设 `--scenario immediate_stop|no_data|cost_exceeds_gain`；默认 `normal`。真实入口使用服务端配置：

```powershell
python -m demo.acquisition_server --port 8769 --mode real --bundle PATH_TO_PLATFORM_BUNDLE --research-artifact PATH_TO_FITTED_ARTIFACT
```

路径仅在服务端命令行接收；HTTP 拒绝模型文件、完整特征、隐藏信息、标签、动作覆盖及费用换算等客户端字段。真实配置缺失时，页面保留 `BACKEND_UNCONFIGURED`，不会自动进入构造场景。平台/研究工厂配置格式以 `football_agent.research_bridge.make_backend` 及各服务发布的接口为准。

四种策略为 `fixed`、`pure_llm`、`all_candidates`、`llm_gate`。“同场对照”只启动固定顺序和全候选数值策略，两者比赛、截止点、预算、seed 相同，各自独立计费；单场演示不是效果优越性的结论。LLM 配置由 `LLMProposer.from_env` 复用服务端环境变量。默认 `--llm-max-calls 5 --llm-max-output-tokens 512`，持久请求账本位于 run 目录相邻 `llm/demo-ledger.json`；数值策略不发 LLM 请求。实际配置、provider/model、prompt版本及缺配置状态在服务能力中可查。

运行配置与状态保存在 `var/parallel/agent/round3/runs/`。页面 URL 含 `?run=RUN_ID`，刷新或重新启动服务后可读取同一记录。跨进程锁与恢复计费由实际运行器执行。每个播放位置仅展示对应事件 snapshot；初始位置没有后续概率或证据，不回填最终结果。JSON 下载包含公开决策轨迹与审计；隐藏档案不通过页面提供。

测试命令：

```powershell
python -m pytest tests/demo/test_round3_server.py -q
python tests/demo/browser_round3.py --url http://127.0.0.1:8768 --output var/parallel/agent/round3/browser-synthetic
```

浏览器验证会实际创建运行和同场对照，覆盖公开初始状态、真实工具返回的概率、停止原因、回放隔离、下载、刷新恢复、移动布局与 JavaScript 错误。纯 UI 测试不声称完成研究效果验证或外部 API 成功获取。

## 已接通的真实历史展示（2026-09-22）

平台正式历史导出和研究向前验证工件已接通。当前 `8769` 运行 fold 2 配置：

```powershell
python -m demo.acquisition_server --port 8769 --mode real --bundle var/parallel/collectors/round2/exports/understat-2025-historical_reconstruction-a3b3d45811fd5cda --research-artifact var/parallel/agent/round3/real/backend-fold-2.json --store-dir var/parallel/agent/round3/real/ui-fold2-runs
```

打开 `http://127.0.0.1:8769/?run=3aa5fc27-76c6-42bb-a274-05f11f4e54c9` 可查看真实 Burnley / Newcastle United 历史受控揭示：数值策略从空 mask 开始，选择 `xg_view`，更新部分预测，再依据非正净增益停止；模拟费为 `0.03 simulation_credit`。同场公平对照固定顺序取得 `results_view` 和 `xg_view`，模拟费 `0.05 simulation_credit`。这些费用属于历史受控实验；不构成外部 API 实测节费或策略效果证据。

`var/parallel/agent/round3/browser-real-active/` 保存 8 项浏览器检查、零 JavaScript 错误、完整公平对照、桌面/移动截图和实际进程重启检查。重启后记录、模型配置、费用账本及事件全部保持一致，无新增工具调用。此前真实 fold 0 的立即价值停止位于 `browser-real/`；旧缺配置验收保留在 `browser-unconfigured/`。`demo-configuration-transition.json` 记录从未配置到真实接通的状态变化。

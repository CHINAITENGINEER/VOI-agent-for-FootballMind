"""Assemble the frozen research results; never train, predict, or modify inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.experiment.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def read(relative):
        p = root / relative
        inputs[str(p)] = sha(p)
        return json.loads(p.read_text(encoding='utf-8'))

    def write(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def table(name, rows):
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (out / name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)

    coverage = read('final-analysis/data-coverage.json')
    gpu = read('gpu-return-verification-v1/verification.json')
    imported = read('gpu-import-v1/provenance.json')
    models = {k: read(f'g-models/{k}/model.json') for k in ['set', 'gru']}
    fmodels = {k: read(f'f-models/selected-{v}/model.json') for k, v in [('set', 'set_residual_mlp'), ('gru', 'gru')]}
    fmetrics = read('f-models/evaluation/selected-model-metrics.json') + read('f-models/evaluation/text-subset-baseline-metrics.json')
    policy = [dict(f_architecture=k, **row) for k in ['set', 'gru'] for row in read(f'policy-evaluation/{k}/strategy-comparison.json')['summary']]
    transfer = [dict(f_architecture=k, **row) for k in ['set', 'gru'] for row in read(f'legacy-g-transfer-v1/{k}/comparison.json')['summary']]
    comparison = read('code-checks/g-training/parent-comparison-tables.json')
    order = read('order-sensitivity-v1/summary.json')
    replay = {k: read(f'recorded-replay-new-f-v2/{k}/summary.json') for k in ['set', 'gru']}
    evidence = {name: read(path) for name, path in {
        'f_final_code': 'f-models/final-frozen-code-verification.json',
        'f_numeric_reproduction': 'f-models/evaluation-verification-v1/verification.json',
        'g_training': 'code-checks/g-training/completion-receipt.json',
        'g_policy': 'code-checks/g-training/policy-independent-verification.json',
        'split': 'final-analysis/split-integrity.json',
        'labels': 'final-analysis/label-binding-validation.json',
        'order': 'order-sensitivity-v1/verification.json',
        'replay': 'recorded-replay-new-f-v2/replay-verification.json',
    }.items()}
    metrics = [r for r in fmetrics if r['role'] in ['validation', 'test']]
    budget3 = [r for r in policy if r['budget'] == 3 and r['strategy'] in ['none', 'all', 'fixed', 'random', 'new_g']]
    paired = [r for r in comparison['paired_results'] if r['previous_development_exposure'] == 'all' and r['comparison'] == 'new_g_vs_random']
    table('coverage-by-role.csv', coverage['role_counts'])
    table('f-heldout-metrics.csv', metrics)
    table('strategy-budget3.csv', budget3)
    table('strategy-all-budgets.csv', policy)
    table('g-auxiliary-gain-metrics.csv', comparison['gain_metrics'])
    table('paired-policy-intervals.csv', comparison['paired_results'])
    table('legacy-g-transfer-diagnostic.csv', transfer)
    modelrows = []
    for k in ['set', 'gru']:
        fp = root / ('f-models/selected-set_residual_mlp/model.pt' if k == 'set' else 'f-models/selected-gru/model.pt')
        gp = root / f'g-models/{k}/g-models.joblib'
        assert sha(fp) == models[k]['f_model_sha256'] == fmodels[k]['f_model_sha256']
        assert sha(gp) == models[k]['g_model_sha256']
        modelrows.append(dict(f_architecture=k, f_path=str(fp), f_sha256=sha(fp),
                              f_total_parameters=fmodels[k]['parameters']['total'],
                              f_trainable_parameters=fmodels[k]['parameters']['trainable'],
                              g_bundle=str(gp), g_sha256=sha(gp),
                              selected_g=models[k]['selected'], g_configurations=models[k]['configurations']))
    write('model-bindings.json', modelrows)
    write('verification-references.json', evidence)
    write('results.json', dict(schema='footballmind-full-history-research-delivery-v1',
        created_at=datetime.now(timezone.utc).isoformat(), coverage=coverage,
        gpu_return=gpu, imported_vectors=imported['imported_texts'], vector_origins=imported['origin_counts'],
        f_metrics=metrics, policy_budget3=budget3, paired_intervals=paired,
        gain_metrics=comparison['gain_metrics'], models=modelrows,
        order_summary=order, recorded_replay=replay, input_hashes=inputs,
        formal_runtime_changed=False, inference_updates_parameters=False,
        current_test_used_to_tune=False, full_system_monetary_cost='UNKNOWN',
        research_effectiveness='not_established', scientific_promotion=False))

    lines = [
        'FootballMind 全文＋状态＋顺序研究训练交付（2026-10-25）',
        '',
        '结论：GPU 返回包验收及导入完成；新集合 f、GRU f、两套配套 g 已真实训练并冻结，完成策略评估。',
        '本轮没有证明新模型在测试集优于市场先验，也没有证明全文筛选优于同预算随机筛选。保留研究模型，不替换正式运行模型。',
        '这里的“完成”指列出的训练、推理及数值复算已运行，不代表长期研究目标或所有来源审阅已经完成。',
        '',
        '1. GPU 与向量',
        f"原 ZIP：{gpu['zip_path']}；SHA256={gpu['zip_sha256']}。",
        f"完成 {gpu['completed_texts']} / {gpu['expected_texts']} 篇，其中主历史库 3001、原轨迹诊断 5。",
        f"8 个 CPU 对照通过，最小余弦={gpu['minimum_control_cosine']:.16f}，最大逐元素差={gpu['maximum_control_absolute_difference']:.16g}。",
        f"此前溢出的 15446-token 正文已完整编码，峰值 allocated={gpu['longest_peak_allocated_gib']:.6f} GiB，reserved={gpu['longest_peak_reserved_gib']:.6f} GiB。",
        '修复将 attention 的 query 计算分块，保留完整 K/V 和全局 causal 位置；没有把正文切段平均、截短正文或换模型。',
        '八个对照只证明这八篇的数值一致性；最长正文不在这八个 CPU 对照中。其完整性由原文 SHA、token 数和执行记录支持。',
        '使用 Qwen/Qwen3-Embedding-0.6B，revision=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3；595776512 个冻结参数。',
        '全文原样、无标题/结果注入，last nonpadding pooling，原始 1024 维 L2 向量；本轮预测头取原生前 256 维后再 L2。不是 TF-IDF。',
        f"正式研究向量包：{root / 'gpu-import-v1/embeddings.npz'}；来源计数={imported['origin_counts']}。",
        '已存在的 CPU 向量保留原字节；其余由验证通过的 GPU 向量补齐。旧 CPU 长任务已停止，未删除缓存。',
        '',
        '2. 比赛与正文去向',
        f"主注册表 {coverage['master_registry_unique_ids']} 个唯一 ID；其中 {coverage['structured_supervised_matches']} 场具备结构监督，{coverage['eligible_sequence_matches']} 场可形成正文序列。",
        '主注册表含未结算、缺先验或未跨源确认的记录，不能把 26271 全部宣传为可训练的独立比赛。',
        f"完整正文 {coverage['full_text_unique_bodies']}；互斥去向：{coverage['body_routes_mutually_exclusive']}。",
        '按 physical_match_id 分组；本轮六个角色交集为 0，合格正文的精确 SHA 跨角色重叠为 0。近似转载及未记录的历史实验暴露不能因此宣布全部排除。',
        '角色 | 结构监督比赛 | 正文序列比赛 | 正文数',
    ]
    for r in coverage['role_counts']:
        lines.append(f"{r['role']} | {r['structured_supervised_matches']} | {r['eligible_sequence_matches']} | {r['eligible_document_occurrences']}")
    lines += [
        'f_train：2022-08-07 前；f_dev：2022-08-07 至 2022 年末；g_fit：2023—2024；g_dev：2025 上半年；validation：2025 下半年；test：2026 起。',
        '全部源顺序前缀共 4521 条，含每场 s0；这些不是 4521 场比赛。g_fit/g_dev 与 f_train/f_dev 比赛分开。',
        '测试 320 场中 55 场已有项目开发暴露记录，另 265 场仅能称“未记录暴露”；不称全新盲测。验证集对应 39/154。',
        '来源扫描范围、未解析格式和每场未进入训练的具体原因见 data/source-discovery-scope.json 与 data/missing-routing.jsonl。',
        '已覆盖识别到的数据源族和规范化索引；尚不能声称磁盘上每个任意格式文件都已穷尽解析。未解决的材料保留清单，不伪造正文或时间。',
        '',
        '3. 新 f 的训练与固定参数推理',
        '基础校准器只有 4 个参数，在 16399 场 f_train 上用固定 LBFGS/penalty 拟合，在 948 场 f_dev 上评估；没有用该开发集选择校准器，正文训练阶段冻结。',
        '集合 f：26823 总参数 / 26819 可训练；GRU f：67079 / 67075。全文编码器冻结，不把编码器参数计作本轮训练参数。',
        '训练正文是 294 场 521 篇，开发正文是 74 场 137 篇；不能说文本模型在 24911 场正文上训练。',
        '输入是 256 维全文表示、22 维元数据、先验和已有证据；GRU 隐状态 64。集合对照屏蔽依赖采用顺序的元数据，最终集合置换不变。',
        '集合/GRU 的元数据使用存在差异，比较结果不能纯归因为循环结构本身。',
        '三种 seed=11/29/47；AdamW lr=.001、weight_decay=.01、dropout=.1、最多 200 epoch、patience=25；每场先对前缀平均，再按比赛等权。',
        'f_dev 前缀 log loss 选择：集合 seed47 epoch1、GRU seed29 epoch1；选后冻结，不用 heldout 调参。',
        '每一步导出 used_input_ids、概率、隐藏状态及 f SHA。推理期间只改变证据和隐藏状态，参数与 buffer 不变；outcome 不传入 predict。',
        '',
        '同一正文比赛集合，全部正文的最终 log loss（越低越好）：',
        'model | split | matches | log_loss | brier | accuracy',
    ]
    for r in metrics:
        lines.append(f"{r['model']} | {r['role']} | {r['physical_matches']} | {r['final_log_loss']:.12f} | {r['final_brier_score']:.12f} | {r['final_accuracy']:.9f}")
    lines += [
        '验证集新 f 有改善，但测试集原始市场先验最好；不把验证改善称作已证明的泛化提升。',
        '旧 f 只在有相同正文/claim 的共同子集比较，validation 19 场、test 6 场；详见 f-models/evaluation/old-f-common-subset-metrics.json。',
        '',
        '4. 与各自冻结 f 绑定的 g',
        '每套 g：479 场/7600 转移用于拟合，160 场/2846 转移用于开发选择。10,446 转移不是 10,446 场比赛。',
        '每个候选均有 s0 标签，并有 3 个有界来源/随机前缀顺序，前缀深度最多 3；预算特征=1/2/3/6。完整候选池未截断。',
        'g 输入 1140 维：状态349、候选全文256、差值256、逐元素乘积256、候选元数据22、最大已采用相似度1。',
        'state349包含当前概率/熵/计数/预算、已采用正文均值256、元数据均值22、当前 f 隐状态64。',
        '输入没有 outcome、p_after、gain 或未来前缀。标签另存：gain=log(p_after[y])-log(p_before[y])。',
        '每场总权重受控。标准化器只拟合 g_fit。三个 MLP seed=20261025/26/27，1140→128→64→1，154369 参数；同时训练零、均值、状态 Ridge、全文 Ridge、局部元数据 Ridge。',
        'MLP epoch 根据 g_dev 比赛等权 MSE 选择；最终模型与阈值根据 g_dev 完整预算3策略损失选择。阈值候选[-.01,-.003,0,.003,.01]。',
    ]
    for k, m in models.items():
        lines.append(f"{k}: 正={m['positive']}，零={m['zero']}，负={m['negative']}；实际选中 {m['selected']['model']}，阈值={m['selected']['threshold']}。")
    lines += [
        '零样本计数为 0 是本轮有效标签的实际结果，代码未筛除零或负收益。4540 条失败清单都是 input_preparation/sequence_not_eligible，不是 4540 次 f 执行失败；gain=null。',
        '集合 f 配套的 state_ridge 在 334 个多候选决策状态中全部候选同分。它只能学习继续/停止，当前无法声称学会候选正文排序。',
        'GRU 配套局部元数据模型在测试 320 场只采用 10 次，115 场英超全部停止于未采用状态。收益优势可能来自少读，不代表识别了更好的文章。',
        '',
        '辅助收益回归指标（比赛等权 MSE；训练/开发预算1/2/3/6，验证/测试预算3，跨角色并非相同转移协议）：',
        'f | g | split | matches | rows | macro MSE',
    ]
    for r in comparison['gain_metrics']:
        lines.append(f"{r['f_architecture']} | {r['model']} | {r['split_role']} | {r['matches']} | {r['rows']} | {r['match_macro_mse']:.12f}")
    lines += [
        '两套全文 MLP 都能显著拟合训练标签，但开发/测试没有胜过零基线。这是训练—评估落差的直接证据；不能凭参数少认定容量不足。',
        '',
        '5. 完整策略比较',
        '同一个 f、同一候选池、同一比赛、采用预算3；随机每场10种seed先平均，再按比赛等权。all为完整池无上限参照，不冒称与预算3成本相同。',
        'f | split | strategy | matches | final_log_loss | mean_updates',
    ]
    for r in budget3:
        lines.append(f"{r['f_architecture']} | {r['split_role']} | {r['strategy']} | {r['matches']} | {r['mean_final_log_loss']:.12f} | {r['mean_updates']:.6f}")
    lines += ['配对 bootstrap：比赛为抽样单元，2000次，seed20261025；差值=新g损失减随机损失，负数更好。']
    for r in paired:
        lines.append(f"{r['f_architecture']} / {r['split_role']}: 差={r['mean_difference']:.12f}；95%区间={r['ci95']}。")
    lines += [
        '区间包含0，未证明稳定优于同预算随机；验证集新g点估计反而不如随机。旧g迁移在46场/85篇严格共同池单列，包含曾用于开发的比赛，不混入主独立比较。',
        'oracle只作事后分析，保留未算完状态；集合2050个精确结果/2个状态限制，GRU2044/8。未完成项没有伪造最优分数。',
        '成本：全文获取与编码已发生，不能当作后续g节省的成本；历史发现/获取金额与真实API总金额未知。各步 g/f 时间、采用次数及未知字段保留，不能据此声称总费用下降。',
        '',
        '6. 顺序与实际记录回放',
        '完整 513 场 / 1041 正文出现，source顺序+5固定随机排列；两个f共6156行。集合f最终L1=0；GRU在多文324场最大L1均值约.001374，最大.0122995，无最高概率类别变化。',
        '这些是历史采用顺序对照，不是事实发生顺序；后读不能解释为事件更新。重算前缀的元数据后再推理，结果不因顺序被人为要求变化。',
        '真实运行扫描单列：40个已执行run/18场，6个可重放run/5场；两套新f均6/6完成，原日志不改写。重复控制版本属于同一场。',
        '已结算可比较只有两场：Monterrey–Pachuca 与 Leeds–Manchester United。其余保持未知；新f离线结果不倒填成原赛前预测。',
        '',
        '7. 数据限制与下一步',
        '3001篇中2915篇无结构化claims；已有334条claims的时间持续范围全未知，294条缺条件注释。神经模型仍读取这些完整正文，未因缺claim把向量置零。',
        '历史source_available_at全部未知；2882条缺精确kickoff。使用来源声明日期早于比赛日的历史重建规则，不冒充真实赛前版本证明。',
        '110篇距目标比赛超过30天，14篇超过90天，最远1083天；例如2025 Brighton–Fulham池含2022同对手发布会。它是历史候选关联，不能自动视为本场球队状态。',
        '这种旧文可以作为历史或无关负例保留；尚未证明每篇旧文有害，也没有依据测试收益删文重训。',
        '22个元数据字段中球队/主客关联主要为词面别名与来源归属，不等于已经理解球员条件性状态、事件持续时间或谁与谁冲突。',
        '本轮可以确认：真实语义向量、固定参数前缀预测、新f专属gain、分组训练、策略和逐场复算已运行。',
        '本轮不能确认：全文g有必要、GRU优于集合f、参数更大更准、线上经济优势、所有历史时钟/近重复泄漏都排除。',
        '下一项最高优先级：改善当前状态与候选正文的实体/时间适用性及重复冲突表达；先在训练/开发材料建立明确监督与消融，再冻结下一轮协议。保留旧文/无关/冲突负例，不再靠加宽评分头争取训练拟合。',
        '若继续开发使用本轮测试案例，就必须记录其开发暴露，下一轮不能再把同一批称为未见测试。',
        '',
        '8. 文件与复现',
        f'总目录：{root}',
        'data/matches.jsonl + missing-routing.jsonl：每场去向；documents.jsonl：全文引用；sequences.jsonl：前缀输入；outcomes.jsonl：独立结果文件。',
        'f-models：6个真实候选训练与选中checkpoint、训练/学习曲线；evaluation：所有导出概率；evaluation-verification-v1：独立数值复算。',
        'g-models/{set,gru}：labels.jsonl、features.npy、feature-ids.json、g-models.joblib、model.json、训练/校准曲线。',
        'policy-evaluation/{set,gru}：逐场路径、全部候选/未选择项、停止、gain预测、预算曲线、oracle与失败清单。',
        'legacy-g-transfer-v1：旧g相同池跨f迁移；recorded-trajectories-v2及recorded-replay-new-f-v2：真实旧记录与新f离线回放。',
        'final-analysis：按比赛、来源、联赛、球队、claims、开发暴露的错误与数据覆盖；order-sensitivity-v1：完整池重排诊断。',
        '具体绝对路径与文件摘要见 artifact-index.csv / source-inputs.json / model-bindings.json。',
        '专项测试：f固定前缀/参数不变/集合对称/GRU路径；g标签绑定/无结果输入/按场加权/策略停止；GPU分块及导入分别有独立回执。未宣称全仓库测试通过。',
        '复算没有改源数据、重训选中模型或重新挑选阈值；正式 f/g/runtime 未替换。',
    ]
    (out / 'readable-report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # Standalone scientific figures: only read exported numerical results.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fontfile = Path('C:/Windows/Fonts/msyh.ttc')
    if fontfile.exists():
        plt.rcParams['font.family'] = FontProperties(fname=fontfile).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fnames = [('no_text_market_prior','原始市场先验'),('no_text_calibrated_prior','校准先验'),('selected','集合 f'),('selected-gru','GRU f')]
    for role, color, marker in [('validation','#25816c','o'),('test','#c16c32','s')]:
        ax=axes[0,0]
        values=[next(r['final_log_loss'] for r in metrics if r['role']==role and r['model']==name) for name,_ in fnames]
        ax.plot(range(4),values,marker=marker,color=color,label=role)
        for i,v in enumerate(values):ax.annotate(f'{v:.5f}',(i,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    axes[0,0].set(xticks=range(4),xticklabels=[x[1] for x in fnames],ylabel='最终 log loss，越低越好',title='同一正文比赛：验证193场 / 测试320场')
    axes[0,0].legend()
    ax=axes[0,1]
    strategies=[('none','不加正文'),('all','全部正文*'),('fixed','固定顺序'),('random','随机'),('new_g','选中的 g')]
    for k,color in [('set','#25816c'),('gru','#8a61a6')]:
        values=[next(r['mean_final_log_loss'] for r in budget3 if r['f_architecture']==k and r['split_role']=='test' and r['strategy']==name) for name,_ in strategies]
        ax.plot(range(5),values,'o-',color=color,label=k)
    ax.set(xticks=range(5),xticklabels=[x[1] for x in strategies],ylabel='最终 log loss，越低越好',title='测试集策略：预算3；*全部正文无上限参照')
    ax.legend()
    ax=axes[1,0]
    for k,color in [('set','#25816c'),('gru','#8a61a6')]:
        subset=[r for r in comparison['gain_metrics'] if r['f_architecture']==k]
        ratios=[]
        for role in ['g_fit','g_dev','validation','test']:
            z=next(r for r in subset if r['split_role']==role and r['model']=='zero')
            m=next(r for r in subset if r['split_role']==role and r['model']!='zero')
            ratios.append(m['match_macro_mse']/z['match_macro_mse'])
        ax.plot(range(4),ratios,'o-',color=color,label=k+' 的全文 MLP')
    ax.axhline(1,color='#555',linestyle='--',label='零基线参考')
    ax.set(xticks=range(4),xticklabels=['训练','开发','验证','测试'],ylabel='MLP MSE / 零基线 MSE',title='小于1才优于零基线；开发选定MLP，不按测试挑选')
    ax.legend(fontsize=9)
    ax=axes[1,1]
    for i,r in enumerate(paired):
        mean=r['mean_difference'];lo,hi=r['ci95']
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,color='#25816c' if r['f_architecture']=='set' else '#8a61a6')
    ax.axvline(0,color='#555',linestyle='--')
    ax.set(yticks=range(len(paired)),yticklabels=[r['f_architecture']+' / '+r['split_role'] for r in paired],xlabel='新 g 损失 − 随机损失；左侧更好',title='逐场配对95%区间：均跨过0')
    for ax in axes.flat:ax.grid(alpha=.18)
    fig.suptitle('真实训练已完成；当前结果尚未证明全文筛选的稳定优势',fontsize=17)
    fig.savefig(out/'research-summary.png',dpi=160)
    plt.close(fig)

    artifact_rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or out in p.parents:
            continue
        # Keep a bounded navigable index, not a duplicate corpus or API archives.
        rel=p.relative_to(root)
        if len(rel.parts)<=3 and p.suffix in {'.json','.csv','.png','.joblib','.pt','.txt'} and 'encoding-texts' not in rel.parts:
            artifact_rows.append(dict(path=str(p),relative=str(rel),bytes=p.stat().st_size,sha256=sha(p)))
    table('artifact-index.csv',artifact_rows)
    code_names=['run_kaggle_semantic_encoding_v2.py','import_kaggle_semantic_encoding.py',
        'train_full_history_sequence_f.py','train_full_history_sequence_g.py',
        'evaluate_full_history_sequence_policies.py','verify_full_history_g_training.py',
        'evaluate_legacy_g_transfer_subset.py','diagnose_f_order_sensitivity.py',Path(__file__).name]
    code=[dict(path=str(Path(__file__).resolve().parent/n),sha256=sha(Path(__file__).resolve().parent/n)) for n in code_names]
    write('source-inputs.json',dict(read_only_input_hashes=inputs,code=code))
    assert all(sha(p)==digest for p,digest in inputs.items())
    write('report-checks.json',dict(input_hashes_unchanged=True,inputs=len(inputs),model_hash_checks=4,
        no_training_or_inference_in_report=True,full_candidate_pool_retained=True,
        note='Historical checkpoints and earlier acceptance receipts are point-in-time; final evaluation and verification are separate artifacts.'))
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()}
    write('manifest.json',dict(schema='research-delivery-manifest-v1',files=manifest))
    print(json.dumps(dict(status='complete',output=str(out),files=len(manifest)+1),ensure_ascii=False))


if __name__=='__main__':
    main()

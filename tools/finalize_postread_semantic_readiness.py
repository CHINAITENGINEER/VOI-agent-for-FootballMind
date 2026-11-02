"""Summarize completed stability runs and the separate training preparation."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path("E:/FootballMind/experiments")
STABILITY = EXPERIMENTS / "postread-qwen-stability-20261025-v1"
READY = EXPERIMENTS / "postread-semantic-training-ready-20261025-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main():
    stability = read(STABILITY / "stability-summary.json")
    independent = read(STABILITY / "verification-independent/verification.json")
    panel = read(READY / "supervised-panel/readiness-report.json")
    context = read(READY / "context-sidecar/checks.json")["summary"]
    corpus = read(READY / "corpus-queue/preparation.json")
    encoded = read(READY / "corpus-queue/encoding-status.json")
    smoke = read(READY / "corpus-queue/encoding-smoke.json")
    assert independent["completed_folds"] == 36 and not independent["pending_folds"]
    assert independent["status"] == "passed_with_recorded_refit_drift"
    assert panel["status"] == "passed"
    assert context["sample_rows"] == 289 and context["source_files_unchanged"]
    assert corpus["inputs_unchanged"] and corpus["candidate_rows"] == 2179
    assert smoke["fresh_encoded"] == 4 and encoded["fresh_encoded"] == 0
    assert encoded["verified_cache_hits"] == 20
    result = {
        "schema": "postread-semantic-stability-and-preparation-v1",
        "stability_completed": True, "stability_verification_status": independent["status"],
        "stability": stability,
        "supervised_panel_ready": True, "supervised_matches": 51,
        "supervised_bodies": 92, "supervised_transitions": 289,
        "unique_frozen_text_vectors": 110, "native_vector_dimension": 1024,
        "gains": {"positive": 60, "zero": 114, "negative": 115},
        "context_sidecar_ready": True, "context_automatically_added_to_old_models": False,
        "corpus_matches": corpus["independent_matches"], "corpus_bodies": corpus["candidate_rows"],
        "corpus_known_contaminated_excluded": corpus["excluded_known_contaminated"],
        "corpus_token_counts": corpus["encoding_status_counts"],
        "corpus_encoding_status": encoded["status_counts"],
        "large_corpus_all_vectors_complete": False,
        "large_corpus_same_f_supervised_labels_complete": False,
        "scalable_entrypoints_ready": True,
        "large_training_started": False, "waiting_for_user_training_protocol": True,
        "new_source_or_model_http": 0, "new_api_calls": 0,
        "f_retrained": False, "runtime_modified": False,
        "main_next_work": "Use the user's training protocol to finish historical applicability/claims and fixed-f state replay; resume bulk local encoding; train only after label-ready joins.",
        "historical_data_does_not_require_live_action_propensity": True,
        "warnings": ["Do not count cumulative corpus versions or repeated state transitions as new matches.",
                     "Do not mix labels from a different f SHA with current f targets.",
                     "Do not call repeated development splits new independent test evidence.",
                     "The 256-vs32 representation comparison changes dimensionality as well as encoder.",
                     "Current context-sidecar fields have not yet been evaluated as new model inputs."]}
    write(READY / "readiness.json", result)
    report = f"""先完成稳定性，再准备扩大训练：交付说明

第一项已完成，第二项已做到数据与程序入口可运行；未启动新的大规模训练。

固定对象：当前 f SHA 34ce858f02165c20779a05b6bed20f1c121008ea7fbe389b64022508f0d7b583；
Qwen/Qwen3-Embedding-0.6B revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3；
110 个既有全文/已采用集合向量不变，51 场 / 92 篇 / 289 条转移不变。

稳定性结论
- 完成6个既有指定分组和30个既有随机重分组；随机重分组复用同一批比赛，不是30批新测试集。
- 同维度、同MLP形状：Qwen32在6个指定分组中的5个低于词频MLP的测试gain MSE；同种子12/18组更低。
- 但Qwen32 MLP在6/6个分组都没有低于恒零收益基线；全部6个比赛配对bootstrap区间包含0。
- Qwen256 Ridge在30次随机重分组里28次低于词频32维Ridge，16次低于恒零；存在表示维度差异，不能全归因于语义编码。
- 完整策略与回归MSE分别看：每场最多采用1篇时，新MLP最终log loss相对旧MLP为3组较低、1组相同、2组较高；相对随机5组较低、1组较高。
- 这些是同一开发材料上的稳定性证据，未证明独立未来泛化，也没有据此扩大评分头或上线。
- 独立复算1224组指标及117480条逐模型预测记录（不是117480条独立数据）；指标差0。
  重新拟合一个旧词频MLP时有微小数值漂移，测试MSE差约-4.1488e-8，已记录；Qwen三种子的重训复算一致。

状态与正文检查
- 289条candidate/adopted/input_id/向量绑定均一致，没有查到错配。
- 旧state57中35列恒0，claim29中11列恒0；显式球队/球员身份、实际剩余采用预算和条件/时间关系不足。
- 旧candidate_status_conflicts是合并证据后的多状态计数，其中2条只继承旧多状态，不是候选新引入冲突。
- context-sidecar已补存可追溯目标身份、实体/引句、候选与已采用证据、旧与新增多状态；不改旧特征/模型。
  只有11/51场有精确UTC kickoff，实际预算未指定均null；不推造字段，不把多状态当已裁定矛盾。

可立即使用的监督数据
- supervised-panel：51场，289条，原train39/241、validation6/24、test6/24；正60、零114、负115全部保留。
- 110×1024 float32唯一向量用内存映射；candidate/adopted索引分开，空已采用集合=-1。
- state57、claims29和32/256/1024维文本块可批量加载，完整输入宽度214/1110/4182。
- 默认不打开labels；显式include_labels=True才读监督目标。训练权重/归一化仍应只用训练折拟合。
- 三种维度的批量特征与原接口逐元素相同，8个冻结Qwen头按原整批预测最大差0。
- batch19的float32 body-only Ridge有1.1176e-8浮点差，其余7个头小于4.2e-17，单独记录。

扩展正文池准备
- 原2185篇/1220场逐条body ID、引用和SHA已核验；沿用既有6篇赛后栏目污染排除，留下2179篇/1218场。
- 已完整落盘2179篇原文、源引用、时钟、比赛关联；内容不传外部API，不使用gain筛选或编码。
- 共1544005 tokens；2178篇不超过8192，1篇15446 tokens完整保留并标over_token_limit，没有截尾。
- 20篇已有可验证向量（16个缓存复用+4篇真实本地新增编码）；2158篇待编码，1篇需另定长文编码契约。
- 这批有历史prior/result关联，但还缺逐篇适用性/claim与同一f的状态收益标签；不能把2179篇叫现成训练行。
- 当前2026抓取钟不冒充历史source_available_at；历史重建允许披露假设，不要求补造实时action_id/propensity。
- 其他大标签集绑定另一f SHA，已单列；没有拼到当前289条里。

可直接运行的入口（在 D:/a1-足彩 执行）
python tools/prepare_postread_corpus_semantic.py encode --limit 100
  最多新增100次本地全文编码；缓存命中不消耗limit，进程中断后从已保存SHA缓存继续。
  当前无遗留编码进程；全池编码没有在后台悄悄启动。

from football_research.acquisition.semantic_training_dataset import SemanticTrainingDataset
data = SemanticTrainingDataset(
    'E:/FootballMind/experiments/postread-semantic-training-ready-20261025-v1/supervised-panel',
    split='train', dimension=256, include_labels=True)
for batch in data.iter_batches(64):
    x = batch['features']['fulltext_state']
    y = batch['realized_gain']

请先看：
{STABILITY / 'readable-report.txt'}
{STABILITY / 'split-stability.png'}
{STABILITY / 'training-curves.png'}
{READY / 'supervised-panel/README.txt'}
{READY / 'context-sidecar/readable-report.txt'}
{READY / 'data-inventory/readable-report.txt'}
{READY / 'corpus-queue/preparation.json'}

验证：32项针对性pytest通过；首次默认Windows临时目录出现2项MAX_PATH失败，短路径重跑32/32通过。
另有context 10项独立检查、data-inventory 10项检查、全量稳定性独立复算，证据分别在各子目录。
本轮有研究评分头稳定性重训与4篇本地编码；没有重新训练f、正式g切换、runtime/前端改动或网络/API请求。
"""
    (READY / "READ_FIRST.txt").write_text(report, encoding="utf-8")
    sources = [
        "tools/compare_postread_qwen_stability.py", "tools/summarize_postread_qwen_stability.py",
        "tools/verify_postread_qwen_stability.py", "tools/build_postread_semantic_context_sidecar.py",
        "tools/prepare_semantic_training_panel.py", "tools/prepare_postread_corpus_semantic.py",
        "tools/finalize_postread_semantic_readiness.py",
        "football_research/acquisition/semantic_training_dataset.py",
        "tests/acquisition/test_postread_corpus_semantic_preparation.py",
    ]
    (READY / "code").mkdir(exist_ok=True)
    for relative in sources:
        shutil.copyfile(ROOT / relative, READY / "code" / Path(relative).name)
    # Bind root summaries and child manifests; child manifests bind their artifacts.
    for directory in (STABILITY, READY):
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != "manifest.json")
        files += sorted(p for p in directory.glob("*/manifest.json"))
        if directory == READY:
            files += sorted(p for p in (READY / "code").iterdir() if p.is_file())
            files += sorted(p for p in (READY / "corpus-queue").iterdir() if p.is_file() and not p.name.startswith("."))
        write(directory / "manifest.json", {str(p.relative_to(directory)).replace("\\", "/"): digest(p) for p in files})
    print(json.dumps({"stability_complete": True, "readiness": str(READY),
                      "supervised": [51, 289], "unlabeled_pool": [1218, 2179],
                      "all_corpus_vectors_done": False, "large_training_started": False}))


if __name__ == "__main__":
    main()

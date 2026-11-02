"""Rebuild and exercise paired full-body g -> frozen f from existing local inputs.

No training, HTTP, external API, model replacement or production activation.
Labels are used after inference, exclusively for parity and strategy evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import socket
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_research.acquisition.postread_selector import (
    BodyCandidate, PostreadSelector, PostreadSession, sha256, text_sha256,
)

EXPERIMENTS = Path("E:/FootballMind/experiments")
BASE = EXPERIMENTS / "unified-f-postread-g-20261025-v1"
EXPANDED = EXPERIMENTS / "unified-f-postread-g-expansion-20261025-v1"
DEFAULT_DATA = EXPERIMENTS / "postread-clean-expansion-20261025-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def save(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def save_lines(path, rows):
    with Path(path).open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")


def network_forbidden(*args, **kwargs):
    raise RuntimeError("This independent replay is strictly offline")


def load_inputs(dataset):
    records = {}
    sources = [BASE/"replay/semantic-records.json", EXPANDED/"semantic-reviews.jsonl", dataset/"semantic-reviews.jsonl"]
    for path in sources:
        for r in (lines(path) if path.suffix == ".jsonl" else read(path)):
            key = r["input"]["input_id"]
            if key in records and any(records[key][k] != r[k] for k in ("input", "response")):
                raise ValueError("Conflicting reviewed semantic input: "+key)
            records[key] = r
    projections = {}
    priors = {}
    for path in [BASE/"replay/projected-states.jsonl", EXPANDED/"replayed-states.jsonl", dataset/"replayed-states.jsonl"]:
        sources.append(path)
        for r in lines(path):
            mid = r["physical_match_id"]
            prior = r["projection"]["prior"]
            if mid in priors and priors[mid] != prior:
                raise ValueError("Prior differs between local f replay sources")
            priors[mid] = prior
            projections[(mid, r.get("discovered_mask"), r["acquired_mask"])] = r
    samples, documents = lines(dataset/"samples.jsonl"), lines(dataset/"documents.jsonl")
    feature_rows = lines(dataset/"features.jsonl")
    features = {r["sample_id"]: r for r in feature_rows}
    groups = defaultdict(list)
    for s in samples:
        groups[s["physical_match_id"]].append(s)
    cases = {}
    for mid, group in groups.items():
        docs = sorted([d for d in documents if d["physical_match_id"] == mid], key=lambda d: d["candidate_index"])
        n = len(docs)
        if [d["candidate_index"] for d in docs] != list(range(n)):
            raise ValueError("Candidate indexes do not completely cover this match")
        if {s["candidate_count"] for s in group} != {n}:
            raise ValueError("Candidate pool differs between samples and documents")
        if len({s["split"] for s in group}) != 1:
            raise ValueError("Physical match crosses dataset split")
        candidates = []
        for d in docs:
            example = next(s for s in group if s["candidate_index"] == d["candidate_index"])
            r = records[d["semantic_input_id"]]
            if d["text"] != r["input"]["text"] or d["text_sha256"] != text_sha256(d["text"]):
                raise ValueError("Document and actual semantic input body differ")
            if d["source_url"] != r["input"]["url"]:
                raise ValueError("Document URL differs from semantic input")
            clocks = {k: d.get(k) for k in ("source_published_at", "source_available_at", "captured_at")}
            clocks.update({k: example.get(k) for k in ("kickoff", "prediction_cutoff")})
            clocks.update(example.get("clocks", {}))
            candidates.append(BodyCandidate.from_record(candidate_id=example["candidate_text_id"],
                action_id=example["action_id"], body_sha256=d["text_sha256"], record=r, clocks=clocks))
        # The original EntityEvidenceBank orders body inputs by input ID. Later
        # historical pools explicitly use menu order; preserve both contracts.
        order = sorted(candidates, key=lambda c: c.input_id) if group[0]["source_family"] == "understat_archived_panel" else candidates
        cases[mid] = dict(candidates=candidates, prior=priors[mid], split=group[0]["split"],
                          evidence_order=[c.candidate_id for c in order], group=group)
    return records, samples, features, cases, sources


def make_session(selector, case, sample, records, features, *, acquired_mask=None):
    dm = sample["discovered_mask"]
    am = sample["acquired_mask"] if acquired_mask is None else acquired_mask
    cs = [c for i, c in enumerate(case["candidates"]) if dm & (1 << i)]
    adopted = [c.candidate_id for i, c in enumerate(case["candidates"]) if am & (1 << i)]
    titles = [records[k] for k in sample["known_input_ids"] if records[k]["input"]["channel"] == "title"]
    state = features[sample["sample_id"]]["state_numeric"]
    hours = None if state[8] == 1 else state[7]
    return PostreadSession(selector, physical_match_id=sample["physical_match_id"], prior=case["prior"],
        candidates=cs, known_title_records=titles, hours_to_kickoff=hours, adopted_ids=adopted,
        evidence_order=[c for c in case["evidence_order"] if c in {d.candidate_id for d in cs}])


def run_path(factory, *, strategy, budget, model_name=None, fixed_order=None):
    """Selection function has no result labels or precomputed after/gain table."""
    session = factory()
    initial = session.prediction()
    decisions = []
    ids = list(session.candidates)
    if strategy == "none":
        reason = "no_information_baseline"
    elif fixed_order is not None or strategy in ("all", "fixed"):
        chosen = list(fixed_order) if fixed_order is not None else ids[:len(ids) if strategy == "all" else budget]
        for cid in chosen:
            session.adopt(cid)
        reason = "all_candidates_used" if len(chosen) == len(ids) else "budget_exhausted"
    else:
        while True:
            decision = session.choose(model_name=model_name, remaining_updates=max(0, budget-len(session.trace)))
            decisions.append(decision)
            if decision["selected_candidate_id"] is None:
                reason = decision["stop_reason"]
                break
            session.adopt(decision["selected_candidate_id"])
    return dict(selected_candidate_ids=[r["candidate_id"] for r in session.trace],
                unselected_candidate_ids=[cid for cid in ids if cid not in session.adopted_ids],
                g_read_body_ids=ids, f_adopted_body_ids=session.adopted_ids,
                processing_count=len(session.trace), budget=budget,
                budget_compliant=len(session.trace) <= budget,
                stop_reason=reason, p_before=initial["probabilities"], p_after=session.prediction()["probabilities"],
                decisions=decisions, f_updates=session.trace,
                costs=dict(current_network_requests=0, current_api_calls=0,
                    archived_body_pool_size=len(ids), archived_semantic_reviews=len(ids),
                    g_score_calls=sum(len(d["candidates"]) for d in decisions), f_updates=len(session.trace),
                    historical_discovery_cost=None, historical_acquisition_cost=None,
                    historical_semantic_review_cost=None, g_reading_money=None, f_update_money=None,
                    total_money=None, incremental_cost_loss_units=0.,
                    cost_assumption="zero incremental adoption cost for this controlled comparison, not measured real cost"))


def settle(path, y):
    pb, pa = path["p_before"], path["p_after"]
    lb, la = -math.log(pb[y]), -math.log(pa[y])
    return {**path, "outcome_class": y, "loss_before": lb, "final_log_loss": la, "realized_gain": lb-la}


def compare_strategies(selector, cases, records, features, labels, models):
    by_match = defaultdict(list)
    for lab in labels.values():
        by_match[lab["physical_match_id"]].append(lab)
    results = []
    for mid, case in sorted(cases.items()):
        full_mask = (1 << len(case["candidates"]))-1
        sample = next(s for s in case["group"] if s["discovered_mask"] == full_mask and s["acquired_mask"] == 0)
        factory = lambda: make_session(selector, case, sample, records, features)
        ys = {r["outcome_class"] for r in by_match[mid]}
        if len(ys) != 1:
            raise ValueError("Conflicting outcomes for one physical match")
        y = next(iter(ys))
        ids = list(factory().candidates)
        for budget in (1, 2, 3):
            common = dict(physical_match_id=mid, split=case["split"], budget=budget, candidate_count=len(ids))
            for strategy in ("none", "all", "fixed", *models):
                path = run_path(factory, strategy=strategy, budget=budget, model_name=models.get(strategy))
                results.append({**common, "strategy": strategy, **settle(path, y)})
            # Enumerate controlled-random sequences exactly; do not average
            # probabilities before log loss (that would be a different policy).
            permutations = list(itertools.permutations(ids, min(budget, len(ids))))
            random_paths = [settle(run_path(factory, strategy="random", budget=budget, fixed_order=seq), y) for seq in permutations]
            results.append({**common, "strategy": "random_exact", "sequence_count": len(random_paths),
                "sequence_paths": random_paths, "processing_count": min(budget, len(ids)),
                "p_before": random_paths[0]["p_before"], "p_after": None,
                "final_log_loss": float(np.mean([r["final_log_loss"] for r in random_paths])),
                "realized_gain": float(np.mean([r["realized_gain"] for r in random_paths])),
                "stop_reason": "budget_or_pool_exhausted", "budget_compliant": True})
            # Oracle is explicitly separated and sees outcome only here.
            oracle = []
            for size in range(min(budget, len(ids))+1):
                for subset in itertools.combinations(ids, size):
                    oracle.append(settle(run_path(factory, strategy="oracle", budget=budget, fixed_order=subset), y))
            best = min(oracle, key=lambda r: (r["final_log_loss"], r["processing_count"]))
            results.append({**common, "strategy": "oracle_analysis_only", **best, "uses_outcome": True})
    aggregates = []
    for split in ("train", "validation", "test"):
        for budget in (1, 2, 3):
            for strategy in sorted({r["strategy"] for r in results}):
                rr = [r for r in results if (r["split"], r["budget"], r["strategy"]) == (split, budget, strategy)]
                if rr:
                    aggregates.append(dict(split=split, budget=budget, strategy=strategy, matches=len(rr),
                        mean_log_loss=float(np.mean([r["final_log_loss"] for r in rr])),
                        mean_realized_gain=float(np.mean([r["realized_gain"] for r in rr])),
                        mean_f_updates=float(np.mean([r["processing_count"] for r in rr])),
                        all_budget_compliant=all(r["budget_compliant"] for r in rr)))
    return results, aggregates


def render(output, aggregates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    strategies = ("none", "all", "fixed", "random_exact", "title_local", "fulltext_ridge", "fulltext_mlp")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for split, ax in zip(("validation", "test"), axes):
        for name in strategies:
            rr = [r for r in aggregates if r["split"] == split and r["strategy"] == name]
            ax.plot([r["budget"] for r in rr], [r["mean_log_loss"] for r in rr], marker="o", label=name,
                    linestyle="--" if name == "all" else "-")
        n = next(r["matches"] for r in aggregates if r["split"] == split)
        ax.set(title=f"{split}: {n} matches (previously viewed)", xlabel="Maximum f body updates",
               ylabel="Mean final log loss (lower is better)", xticks=[1, 2, 3])
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Fresh paired f inference from archived bodies; fixed saved g models\nAll-candidate reference may exceed budget; no new training or network")
    fig.tight_layout(); fig.savefig(output/"strategy-comparison.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--f-model", type=Path, default=BASE/"replay/frozen-f.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir or args.dataset/"training"
    args.output.mkdir(parents=True, exist_ok=False)
    socket.socket.connect = socket.socket.connect_ex = socket.getaddrinfo = socket.create_connection = network_forbidden
    start = time.perf_counter()
    selector = PostreadSelector(model_dir, args.f_model)
    records, samples, features, cases, source_paths = load_inputs(args.dataset)
    source_paths += [args.dataset/n for n in ("features.jsonl", "samples.jsonl", "documents.jsonl", "labels.jsonl")]
    source_paths += list(model_dir.glob("*.json")) + [model_dir/"models.joblib", model_dir/"predictions.jsonl", args.f_model]
    source_hashes = {str(p): sha256(p) for p in source_paths}
    labels = {r["sample_id"]: r for r in lines(args.dataset/"labels.jsonl")}
    saved_predictions = {(r["model"], r["sample_id"]): r["predicted_gain"] for r in lines(model_dir/"predictions.jsonl")}
    verification_rows, all_scores = [], []
    for sample in samples:
        sid, mid = sample["sample_id"], sample["physical_match_id"]
        session = make_session(selector, cases[mid], sample, records, features)
        cid = sample["candidate_text_id"]
        actual = session.feature(cid, sample_id=sid, split=sample["split"])
        expected = features[sid]
        for key in expected:
            if key in ("state_numeric", "semantic_numeric"):
                if not np.allclose(actual[key], expected[key], atol=1e-13, rtol=0):
                    raise ValueError(f"Feature reproduction failed: {sid} {key}")
            elif actual[key] != expected[key]:
                raise ValueError(f"Feature reproduction failed: {sid} {key}")
        errors = []
        for name in selector.model_names:
            score = selector.score(actual, model_name=name)
            error = abs(score-saved_predictions[name, sid])
            errors.append(error)
            all_scores.append(dict(sample_id=sid, physical_match_id=mid, split=sample["split"], model=name,
                                   predicted_gain=score, saved_predicted_gain=saved_predictions[name, sid], max_abs_error=error))
        before = session.prediction()
        update = session.adopt(cid)
        lab = labels[sid]
        if lab["f_model_sha256"] != selector.f_sha256:
            raise ValueError("Label was generated using a different f")
        error_f = max(np.max(np.abs(np.array(update[k])-lab[k])) for k in ("p_before", "p_after"))
        actual_gain = math.log(update["p_after"][lab["outcome_class"]])-math.log(update["p_before"][lab["outcome_class"]])
        gain_error = abs(actual_gain-lab["realized_gain"])
        if max(errors) > 1e-10 or error_f > 1e-13 or gain_error > 1e-13:
            raise ValueError("Frozen prediction replay differs: "+sid)
        if set(update["used_input_ids_after"])-set(before["used_input_ids"]) != {session.candidates[cid].input_id}:
            raise ValueError("Exactly one candidate semantic body must enter f")
        verification_rows.append(dict(sample_id=sid, physical_match_id=mid, split=sample["split"], **update,
            original_realized_gain=lab["realized_gain"], recomputed_realized_gain=actual_gain,
            feature_reproduction=True, max_f_error=float(error_f), max_g_error=max(errors), gain_error=gain_error))
    save_lines(args.output/"transition-replay.jsonl", verification_rows)
    save_lines(args.output/"score-replay.jsonl", all_scores)
    selection = read(model_dir/"selection-frozen.json")
    mse = selection["validation_match_mse"]
    mlp = min((name for name in mse if name.startswith("mlp_")), key=lambda name: mse[name])
    models = {"title_local": "title_local_ridge", "fulltext_ridge": "fulltext_state_ridge", "fulltext_mlp": mlp}
    results, aggregate = compare_strategies(selector, cases, records, features, labels, models)
    save_lines(args.output/"strategy-paths.jsonl", results)
    save(args.output/"strategy-comparison.json", dict(aggregates=aggregate, models=models,
        cohort="51 historical matches; previously viewed validation/test; not fresh generalization evidence",
        formal_runtime_enabled=False, current_http=0, current_api_calls=0, training=False))
    with (args.output/"strategy-comparison.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)
    vocabulary = []
    for mid, case in cases.items():
        for c in case["candidates"]:
            vocabulary.append(dict(physical_match_id=mid, split=case["split"], candidate_id=c.candidate_id,
                body_sha256=c.body_sha256, **selector.vocabulary_coverage(c.record["input"]["text"])))
    save_lines(args.output/"vocabulary-coverage.jsonl", vocabulary)
    render(args.output, aggregate)
    verification = dict(status="passed", transitions=len(samples), matches=len(cases),
        bodies=sum(len(c["candidates"]) for c in cases.values()), model_score_checks=len(all_scores),
        max_f_prediction_error=max(r["max_f_error"] for r in verification_rows),
        max_g_prediction_error=max(r["max_abs_error"] for r in all_scores),
        max_gain_error=max(r["gain_error"] for r in verification_rows),
        f_model_sha256=selector.f_sha256, g_model_sha256=selector.g_sha256,
        chosen_fulltext_mlp=mlp, overall_validation_winner=selection["selected_model"],
        current_http=0, current_api_calls=0, retraining=False, formal_runtime_modified=False,
        no_new_empirical_rows=True, all_source_files_unchanged=all(sha256(p)==h for p,h in source_hashes.items()),
        network_disabled_in_process=True, elapsed_seconds=time.perf_counter()-start, source_hashes=source_hashes)
    assert verification["all_source_files_unchanged"]
    save(args.output/"verification.json", verification)
    report = ["全文加状态 g 与配套 f 独立入口验收", "",
        f"实际重建 {len(cases)} 场、{verification['bodies']} 篇正文、{len(samples)} 条转移。",
        f"13 个保存模型共核对 {len(all_scores)} 次分数；每条转移重新执行同一冻结 f，未读取旧 after 充当执行。",
        f"f 最大概率误差：{verification['max_f_prediction_error']:.3g}；g 最大分数误差：{verification['max_g_prediction_error']:.3g}。",
        f"全文神经网络按既有验证集 MSE 选择：{mlp}。总体既有验证集最优仍为标题模型 {selection['selected_model']}。",
        "候选全文和已保存语义审阅在 g 决策前可见；仅选中正文进入 f；每次采用均保存 used_input_ids。",
        "没有网络请求、API、重新训练、修改旧模型、启用在线策略或新增 empirical 训练行。", "",
        "同条件历史策略比较（平均最终 log loss 越低越好）："]
    for r in aggregate:
        if r["split"] in ("validation", "test") and r["budget"] == 2:
            report.append(f"{r['split']} | {r['matches']} 场 | {r['strategy']} | loss={r['mean_log_loss']:.9f} | 平均采用正文={r['mean_f_updates']:.3f}")
    report += ["", "解释边界：",
        "旧历史资料的原始赛前版本和部分报价时钟未被证明；这不是前瞻赛前实验。",
        "验证/测试集曾被查看，不把这次复现作为全新独立泛化证据。",
        "现有全文表示为英文 TF-IDF + 32维 SVD，没有多语言覆盖证明。",
        "up-front 全文读取和语义审阅成本未知；本次仅设增量采用成本为零进行对照，不声称总成本节省。",
        "配套 f SHA 与当前线上 f 不同，新入口拒绝混用；尚未替换前端在线模型。",
        "当前入口验证路径可复现，不表示全文模型已经胜过标题模型、随机或全量采用。"]
    (args.output/"readable-report.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    save(args.output/"manifest.json", {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    print(json.dumps({k:v for k,v in verification.items() if k != "source_hashes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

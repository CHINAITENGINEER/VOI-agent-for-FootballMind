"""Replay a dated public corpus through the semantic VoI controller.

This is a historical reconstruction: publication dates are retained, while
the later capture clock and target result never enter the actor's public state.
The transport is local and replay-only; it performs no network request.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_footballmind_semantic_case import build_provider_factory
from football_agent.semantic_scenario_report import write_semantic_report
from football_agent.semantic_scenario_runtime import SemanticScenarioRuntime, replay_math, save


class HistoricalPanelTransport:
    """A capability-bound local replay of already archived article bodies."""
    def __init__(self, panel, clock):
        self.panel = panel
        self.clock = clock
        self.seed = panel["case"]["source_profile_seed"]
        self.by_url = {row["url"]: row for row in panel["documents"]}
        self.links = [{"url": row["url"], "title": row["title"]} for row in panel["documents"]]
        self.executions = []

    def configuration(self):
        return {"provider": "historical_corpus_replay", "network_requests": 0,
                "information_mode": "historical_reconstruction",
                "body_source": "existing_archived_corpus_only"}

    def prepare(self, candidate, public):
        tool = candidate.get("tool")
        if tool == "read_url":
            visible = set(public.get("task", {}).get("seed_urls", []))
            visible.update(item.get("url") for doc in public.get("documents", []) for item in doc.get("links", []))
            if candidate.get("url") not in visible:
                raise ValueError("historical replay read bypassed observed link capability")
        elif tool != "search_web":
            raise ValueError("unsupported historical replay tool")
        return deepcopy(candidate)

    def execute(self, prepared, *, run_id, request_id):
        self.executions.append(deepcopy(prepared))
        if prepared["tool"] == "search_web":
            return {"http_attempts": 1, "document": {"document_id": "historical-search-" + request_id,
                "url": "historical://search/" + request_id, "kind": "discovery", "status": "FAILED",
                "text": "", "links": [], "observed_at": self.clock().isoformat(),
                "available_at": self.clock().isoformat(), "source_published_at": None},
                "latency_seconds": 0., "money": None, "money_status": "not_measured",
                "historical_replay": True}
        source = self.by_url.get(prepared["url"])
        if source is None:
            raise ValueError("historical replay URL is not in selected panel")
        document = deepcopy(source)
        document["kind"] = "article"
        document["status"] = "OK"
        document["observed_at"] = self.clock().isoformat()
        document["available_at"] = self.clock().isoformat()
        return {"http_attempts": 1, "document": document, "latency_seconds": 0.,
                "money": None, "money_status": "not_measured", "historical_replay": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT/"var/submission/p7-historical-semantic/case-selection/panel.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", choices=("codex-cli", "deepseek"), default="codex-cli")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default=None)
    parser.add_argument("--observation-thinking", choices=("enabled", "disabled"), default=None,
                        help="仅覆盖 DeepSeek 正文分类阶段；省略则继承全局配置")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--provider-timeout", type=int, default=180)
    parser.add_argument("--candidate-discovery-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--directional-evidence", action="store_true")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--semantic-calls", type=int, default=10)
    parser.add_argument("--request-loss-cost", type=float, default=.0001)
    parser.add_argument("--rediscoveries", type=int, default=0,
                        help="Optional bounded history-aware rediscovery rounds; default keeps historical replay conservative")
    parser.add_argument("--planning-horizon", type=int, choices=(1, 2), default=1,
                        help="有限前视深度；2 会把一个最优后续查询计入首个查询价值")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new historical output folder")
    api_key = None
    if args.provider == "deepseek":
        from football_agent.deepseek_provider import resolve_api_key
        api_key = resolve_api_key()
        if not api_key:
            parser.error("DEEPSEEK_API_KEY is not configured for --provider deepseek")
    try:
        provider_factory = build_provider_factory(provider=args.provider, model=args.model,
            timeout_seconds=args.provider_timeout, max_output_tokens=args.max_output_tokens,
            thinking=args.thinking, api_key=api_key, output=args.output,
            observation_thinking=args.observation_thinking)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    fixture = panel["fixture"]
    docs = panel["documents"]
    # Use only chronological cases before the selected source index for the
    # historical base distribution. The selected target outcome never enters it.
    corpus = json.loads((ROOT/"var/submission/p1-single-match/corpus-probe/cases-with-interviews-routed.json").read_text(encoding="utf-8"))
    selected_index = json.loads((ROOT/"var/submission/p7-historical-semantic/case-selection/selection.json").read_text(encoding="utf-8"))["selected_source_index"]
    from collections import Counter
    counts = Counter(row["outcome"] for row in corpus[:selected_index])
    baseline = [(counts.get(i, 0) + .5) / (selected_index + 1.5) for i in range(3)]
    from datetime import timedelta
    kickoff = datetime.fromisoformat(fixture["kickoff"].replace("Z", "+00:00"))
    historical_cutoff = (kickoff - timedelta(hours=6)).isoformat()
    case = {"match": {"match_id": fixture["match_id"], "physical_match_id": fixture["physical_match_id"],
                       "match": fixture["home_name"] + " vs " + fixture["away_name"],
                       "kickoff": fixture["kickoff"], "prediction_cutoff": historical_cutoff},
            "source_profile": {"teams": [{"side": "home", "canonical_name": fixture["home_name"]},
                                           {"side": "away", "canonical_name": fixture["away_name"]}],
                               "allowed_hosts": ["www.leedsunited.com"],
                               "seed_urls": [docs[0]["url"]], "status": "historical_reconstruction"},
            "structured_baseline": {"probabilities": baseline},
            "baseline_scope": "chronological_pre_case_outcome_frequency_laplace_v1_excludes_selected_outcome"}
    now = lambda: datetime.fromisoformat(case["match"]["prediction_cutoff"].replace("Z", "+00:00")) - __import__("datetime").timedelta(hours=6)
    reconstructed_index = "https://www.leedsunited.com/en/historical-reconstructed-menu"
    case["source_profile"]["seed_urls"] = [reconstructed_index]
    transport = HistoricalPanelTransport({"case": {"source_profile_seed": reconstructed_index}, "documents": docs}, now)
    runtime = SemanticScenarioRuntime(case=case, output=args.output,
        provider_factory=provider_factory,
        transport=transport, seed_urls=[reconstructed_index], bootstrap_url=reconstructed_index,
        request_budget=args.requests, semantic_call_budget=args.semantic_calls,
        request_loss_cost=args.request_loss_cost, max_extensions=1,
        max_rediscoveries=args.rediscoveries,
        planning_horizon=args.planning_horizon,
        candidate_discovery_version=args.candidate_discovery_version,
        require_directional_evidence=args.directional_evidence,
        clock=now, progress=lambda message: print(message, flush=True))
    # The first replay page is an index whose observed links are the selected
    # corpus records; no article body is visible until the solver selects it.
    original = transport.by_url[docs[0]["url"]]
    transport.by_url[reconstructed_index] = {"document_id": "historical-index", "url": reconstructed_index,
        "title": "Historical reconstructed fixture menu", "kind": "index", "status": "OK", "text": "",
        "links": transport.links, "observed_at": now().isoformat(), "available_at": now().isoformat(),
        "source_published_at": None, "structured_news": {"segments": []}}
    # Keep the original article body available under its URL only after the
    # index has exposed it; transport uses this separate map for that purpose.
    transport.by_url[docs[0]["url"] + "#article"] = original
    result = runtime.run()
    # If the runtime selected the seed URL as an article (it should not), the
    # body remains inaccessible because the index document contains no body.
    write_semantic_report(result, args.output)
    if result["cycles"]:
        save(args.output/"math-replay.json", replay_math(result))
    save(args.output/"historical-panel-manifest.json", {"panel": str(args.panel.resolve()),
        "network_requests": 0, "target_result_in_public_state": False,
        "source_network_requests": 0,
        "semantic_provider_real_requests": sum(call.get("receipt", {}).get("requests", 0)
            for call in result["semantic_calls"] if call.get("receipt", {}).get("real_provider_call") is True),
        "network_requests_scope": "source_corpus_only_excludes_semantic_provider",
        "source_capture_clock_is_later_than_publication": True,
        "historical_decision_clock": now().isoformat()})
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "initial_prediction": result["initial_prediction"],
                      "final_prediction": result["final_prediction"],
                      "resources": result["resources"], "terminal": result["terminal"]},
                     ensure_ascii=False, indent=2))
    if result["status"] == "incomplete":
        sys.exit(1)


if __name__ == "__main__":
    main()

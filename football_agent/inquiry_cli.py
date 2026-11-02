"""Run/resume or export a bounded, operational open-information inquiry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_spider.storage.raw_archive import RawArchive
from .inquiry import InquiryWorkflow
from .inquiry_provider import CodexInquiryProvider
from .inquiry_tools import OpenWebTools
from .inquiry_hypotheses import get_protocol
from .inquiry_selection import get_selector
from .inquiry_continuation import get_continuation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--step", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    root = Path(args.root).resolve()
    protocol = get_protocol(config.get("inquiry_protocol"), decision_problem=config.get("decision_problem"))
    provider = CodexInquiryProvider(root / "provider", timeout_seconds=config.get("provider_timeout_seconds", 120), protocol=protocol)
    if config.get("transport_route") is not None:
        from .inquiry_transport import RoutedOpenWebTools
        tools = RoutedOpenWebTools(RawArchive(root / "evidence"), config["allowed_hosts"], route=config["transport_route"])
    else:
        tools = OpenWebTools(RawArchive(root / "evidence"), config["allowed_hosts"])
    selector = get_selector(config.get("decision_selector"))
    continuation = get_continuation(config.get("continuation_policy"))
    workflow = InquiryWorkflow(root / "runs", provider, tools, protocol=protocol, selector=selector,
                               continuation=continuation, source_catalog=config.get("source_catalog"))
    if args.resume:
        run = workflow.get(args.resume)
    elif args.export_only:
        parser.error("--export-only requires --resume")
    else:
        run = workflow.start(config["task"], **config.get("budget", {}))
        if config.get("inherit_evidence"):
            run = workflow.inherit_evidence(run["run_id"], **config["inherit_evidence"])
        print(json.dumps({"started": run["run_id"], "root": str(root)}, ensure_ascii=True), flush=True)
    if not args.export_only:
        if args.resume and continuation and run["state"] == "WAITING":
            run = workflow.resume_wait(run["run_id"])
        if args.step:
            run = workflow.step(run["run_id"])
        else:
            while run["state"] == "READY":
                run = workflow.step(run["run_id"])
                print(json.dumps({"run_id": run["run_id"], "state": run["state"], "reason": run["reason"],
                                  "llm_calls": run["budget"]["llm_calls"], "http_attempts": run["budget"]["http_attempts"]}), flush=True)
    audit = workflow.export(run["run_id"], root / "reports" / (run["run_id"] + ".md"))
    print(json.dumps({"run_id": run["run_id"], "state": run["state"], "audit": audit,
                      "decisions": run["decisions"], "claims": run["claims"]}, ensure_ascii=True), flush=True)
    if run["reason"] in {"PROVIDER_FAILED_OR_RESPONSE_UNKNOWN", "INVALID_INQUIRY_RESPONSE",
                         "ACQUISITION_REQUIRES_RECONCILIATION", "INVALID_ACQUISITION_RECEIPT",
                         "INTERRUPTED_CALL_REQUIRES_RECONCILIATION", "CONFIGURATION_CHANGED"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

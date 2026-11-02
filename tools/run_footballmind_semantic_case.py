"""Run a bounded real semantic-scenario case, or replay its frozen mathematics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_agent.inquiry_provider import CodexInquiryProvider
from football_agent.news_discovery import NewsDiscoveryTools
from football_agent.semantic_scenario_runtime import SemanticScenarioRuntime, replay_math, save
from football_agent.semantic_scenario_report import write_semantic_report
from football_agent.semantic_scenario_protocol import ObservationProtocol, DirectionalObservationProtocol
from football_agent.semantic_match_adapter import attach_match_state
from football_spider.storage.raw_archive import RawArchive


def _deepseek_provider_class():
    """Load the optional direct API adapter only when it is selected.

    Keeping this import lazy preserves the existing Codex-only installation:
    replay and ``--provider codex-cli`` do not require an API client or a
    DeepSeek-specific dependency.
    """
    from football_agent.deepseek_provider import DeepSeekInquiryProvider
    return DeepSeekInquiryProvider


def build_provider_factory(*, provider, model, timeout_seconds, max_output_tokens,
                           thinking, api_key, output, observation_thinking=None):
    """Build the runtime's two-argument provider factory.

    The runtime deliberately knows nothing about provider selection.  The
    CLI owns the selection and freezes it in the provider configuration and
    receipts, so a replay remains independent of credentials or network.
    """
    if provider == "codex-cli":
        if thinking not in (None, "disabled") or observation_thinking is not None:
            raise ValueError("--thinking / --observation-thinking are only supported with --provider deepseek")
        return lambda workspace, protocol: CodexInquiryProvider(
            workspace, protocol=protocol, model=model,
            timeout_seconds=timeout_seconds)
    if provider != "deepseek":
        raise ValueError("Unsupported provider: " + str(provider))
    if observation_thinking not in (None, "enabled", "disabled"):
        raise ValueError("observation_thinking must be enabled, disabled, or unspecified")
    DeepSeekInquiryProvider = _deepseek_provider_class()
    thinking = thinking or "enabled"
    # The adapter validates the model, deadline and token budget.  Construct a
    # probe before transport creation so a missing/invalid credential cannot
    # result in any source HTTP request.
    DeepSeekInquiryProvider(output / "provider-preflight", protocol=None,
                            model=model, timeout_seconds=timeout_seconds,
                            max_tokens=max_output_tokens, thinking=thinking,
                            api_key=api_key)
    def factory(workspace, protocol):
        is_observation = getattr(protocol, "version", None) in {
            ObservationProtocol.version, DirectionalObservationProtocol.version}
        # An explicit, per-run stage choice; never a response-dependent retry
        # or a silent fallback. Discovery and probability proposals keep their
        # original mode. Every call records its actual effective configuration.
        effective = observation_thinking if is_observation and observation_thinking is not None else thinking
        return DeepSeekInquiryProvider(workspace, protocol=protocol, model=model,
            timeout_seconds=timeout_seconds, max_tokens=max_output_tokens,
            thinking=effective, api_key=api_key)
    return factory


def resolve_bootstrap_url(case, explicit=None):
    """Return a case-owned discovery entry without inheriting another fixture.

    A semantic run may only start from a URL that the case explicitly bound to
    its source profile.  This keeps the first transport action auditable and
    makes an incomplete arbitrary-match case fail closed instead of borrowing
    the historical Arsenal–Leeds entry.
    """
    profile = case.get("source_profile", {}) if isinstance(case, dict) else {}
    seeds = profile.get("seed_urls", [])
    if not isinstance(seeds, list) or not seeds or not all(
            isinstance(item, str) and item for item in seeds):
        raise ValueError("case_not_ready: source_profile.seed_urls must contain at least one explicit source URL")
    bootstrap = explicit or profile.get("bootstrap_url")
    if not isinstance(bootstrap, str) or not bootstrap:
        raise ValueError(
            "case_not_ready: no bootstrap source is declared; add source_profile.bootstrap_url "
            "or pass --bootstrap-url explicitly"
        )
    if bootstrap not in seeds:
        raise ValueError("case_not_ready: bootstrap URL must be present in source_profile.seed_urls")
    allowed_hosts = profile.get("allowed_hosts", [])
    if allowed_hosts:
        from urllib.parse import urlsplit
        if urlsplit(bootstrap).hostname not in set(allowed_hosts):
            raise ValueError("case_not_ready: bootstrap URL host is outside source_profile.allowed_hosts")
    return seeds, bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=ROOT/"var/submission/p1-single-match/initial-state")
    parser.add_argument("--bootstrap-url", help="显式指定案例已登记且已观察过的启动来源；必须属于 source_profile.seed_urls")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provider", choices=("codex-cli", "deepseek"), default="codex-cli",
                        help="语义推理通道；默认保留 Codex CLI，DeepSeek 使用直接 API")
    parser.add_argument("--model", default=None, help="Explicit provider model request; server identity may remain unreported")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default=None,
                        help="DeepSeek 默认 enabled（low 强度）；仅 --provider deepseek 生效")
    parser.add_argument("--observation-thinking", choices=("enabled", "disabled"), default=None,
                        help="仅覆盖 DeepSeek 正文分类阶段；省略时继承 --thinking，发现与概率提案不变")
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="DeepSeek 单次响应 token 上限（默认8192）")
    parser.add_argument("--match", help="可选：只输入注册案例中的‘主队 vs 客队’，由系统附加自主未知状态")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--semantic-calls", type=int, default=12)
    parser.add_argument("--request-loss-cost", type=float, default=.001)
    parser.add_argument("--provider-timeout", type=int, default=180,
                        help="每次语义 Provider 调用的硬截止时间（秒，1..180）；超时会保留诊断并失败闭合")
    parser.add_argument("--extensions", type=int, default=1)
    parser.add_argument("--rediscoveries", type=int, default=1,
                        help="Bounded history-aware candidate rediscovery rounds after the first observation")
    parser.add_argument("--planning-horizon", type=int, choices=(1, 2), default=1,
                        help="有限前视深度；2 会为首个查询计入一个最优后续查询的条件价值")
    parser.add_argument("--candidate-discovery-version", type=int, choices=(1, 2), default=2,
                        help="2：从已见目录自主提名建模候选；1：旧查询加目录前缀协议")
    parser.add_argument("--projection-version", type=int, choices=(7, 8), default=8,
                        help="8：保留最多320条已见索引链接；7：旧归档兼容")
    parser.add_argument("--resume-from", type=Path,
                        help="仅继续带初始化绑定的失败 run.json，累计预算不重置；输出必须使用新目录")
    parser.add_argument("--legacy-observation", action="store_true",
                        help="Use the frozen legacy supports/refutes observation contract (historical compatibility only)")
    parser.add_argument("--replay", type=Path, help="Saved run.json; zero network/provider calls")
    args = parser.parse_args()
    if args.replay:
        result = replay_math(json.loads(args.replay.read_text(encoding="utf-8")))
        if args.output:
            if args.output.exists():
                raise FileExistsError(args.output)
            save(args.output/"replay.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args.output:
        parser.error("--output is required for live execution")
    if not 1 <= args.provider_timeout <= 180:
        parser.error("--provider-timeout must be between 1 and 180 seconds")
    if not 1 <= args.max_output_tokens <= 32768:
        parser.error("--max-output-tokens must be between 1 and 32768")
    if args.provider == "codex-cli" and (args.thinking not in (None, "disabled") or args.observation_thinking is not None):
        parser.error("--thinking / --observation-thinking are only supported with --provider deepseek")
    api_key = None
    if args.provider == "deepseek":
        # Resolve before constructing NewsDiscoveryTools.  This is a local
        # credential lookup only; the key is never printed or persisted by the
        # runner.  Replay returned above intentionally bypasses this branch.
        from football_agent.deepseek_provider import resolve_api_key
        api_key = resolve_api_key()
        if not api_key:
            parser.error("DEEPSEEK_API_KEY is not configured for --provider deepseek")
    case = json.loads((args.case_dir/"case.json").read_text(encoding="utf-8"))
    if args.match:
        case = attach_match_state(case, args.match)
    try:
        seeds, bootstrap = resolve_bootstrap_url(case, args.bootstrap_url)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        provider_factory = build_provider_factory(
            provider=args.provider, model=args.model,
            timeout_seconds=args.provider_timeout,
            max_output_tokens=args.max_output_tokens,
            thinking=args.thinking, api_key=api_key, output=args.output,
            observation_thinking=args.observation_thinking)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    transport = NewsDiscoveryTools(RawArchive(args.output/"source-archive"),
        case["source_profile"]["allowed_hosts"], index_urls=seeds, projection_version=args.projection_version)
    runtime = SemanticScenarioRuntime(case=case, output=args.output,
        provider_factory=provider_factory,
        transport=transport, seed_urls=seeds, bootstrap_url=bootstrap,
        request_budget=args.requests, semantic_call_budget=args.semantic_calls,
        request_loss_cost=args.request_loss_cost, max_extensions=args.extensions,
        max_rediscoveries=args.rediscoveries,
        planning_horizon=args.planning_horizon,
        candidate_discovery_version=args.candidate_discovery_version,
        resume_from=args.resume_from,
        require_directional_evidence=not args.legacy_observation,
        progress=lambda message: print(message, flush=True))
    result = runtime.run()
    write_semantic_report(result, args.output)
    if result["cycles"]:
        save(args.output/"math-replay.json", replay_math(result))
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
        "initial_prediction": result["initial_prediction"], "final_prediction": result["final_prediction"],
        "resources": result["resources"], "terminal": result["terminal"].get("stop_reason"),
        "failures": result["failures"]}, ensure_ascii=False, indent=2))
    if result["status"] == "incomplete":
        sys.exit(1)


if __name__ == "__main__":
    main()

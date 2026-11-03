"""A dependency-free, loopback-only UI over the actual Agent workflow.

The HTTP layer neither calculates probabilities nor invents tool outcomes.
Mutations are serialized because workflow tools and run files have mutable state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import ipaddress
import json
from pathlib import Path
import re
import threading
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit


STATIC_DIR = Path(__file__).with_name("static")
RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
SCENARIOS = ("normal", "empty", "timeout", "rate_limited", "duplicate", "stale", "conflict", "budget", "model_unavailable")
MAX_BODY_BYTES = 64 * 1024


def jsonable(value: Any) -> Any:
    """Serialize existing contract types without imposing another public schema."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


def make_workflow(mode: str, scenario: str, store_dir: Path, archive_replay: bool = True, *,
                  model_id: str = "market", model_catalog=None, information_mode: str = "operational",
                  time_mode: str = "archive_replay", platform_export=None, service=None,
                  archive_root: str | Path = "football_spider/platform_data/raw", platform_reader: bool = False) -> Any:
    from football_agent.adapters import ArchiveReplayTools, LocalArchiveTools
    from football_agent.engine import Workflow
    from football_agent.testing import SyntheticTools

    if mode == "test":
        tools = SyntheticTools(scenario=scenario)
    elif platform_reader or platform_export is not None or service is not None or time_mode == "live":
        from football_agent.frozen_features import platform_tools
        tools = platform_tools(model_id=model_id, model_catalog=model_catalog, information_mode=information_mode,
                              time_mode=time_mode, export_path=platform_export, service=service, archive_root=archive_root)
    else:
        adapter = ArchiveReplayTools if archive_replay else LocalArchiveTools
        tools = adapter(archive_root, model_id=model_id, model_catalog=model_catalog, information_mode=information_mode)
    return Workflow(tools, store_dir=store_dir)


def make_budget(values: dict[str, Any] | None) -> Any:
    if values is None:
        return None
    if not isinstance(values, dict):
        raise ValueError("budget 必须是 JSON 对象")
    from football_agent.types import Budget

    accepted = {field.name for field in fields(Budget) if field.init} if is_dataclass(Budget) else set(inspect.signature(Budget).parameters)
    unknown = set(values) - accepted
    if unknown:
        raise ValueError("未知预算字段: " + ", ".join(sorted(unknown)))
    return Budget(**values)


class DemoApplication:
    def __init__(self, store_dir: str | Path | None = None, factory: Callable[..., Any] = make_workflow,
                 *, model_catalog=None, platform_export=None, service=None,
                 archive_root: str | Path = "football_spider/platform_data/raw", workspace_config=None):
        from football_agent.model_catalog import ModelCatalog
        from footballmind.config import load_config
        self.workspace_config = workspace_config or load_config()
        self.store_dir = Path(store_dir if store_dir is not None else self.workspace_config.runs_dir).resolve()
        self.factory = factory
        self.model_catalog = model_catalog if model_catalog is not None else ModelCatalog()
        self.platform_export = str(Path(platform_export).resolve()) if platform_export is not None else None
        self.service = service
        self.archive_root = str(Path(archive_root).resolve())
        self.lock = threading.RLock()
        self.workflows: dict[tuple, Any] = {}
        self.run_owners: dict[str, Any] = {}
        self._research_g = None
        self._live_g = None
        self._defense = None

    def defense(self):
        if self._defense is None:
            from .defense import DefenseApplication
            self._defense = DefenseApplication(release_dir=self.workspace_config.release_dir)
        return self._defense

    def live_g(self):
        with self.lock:
            if self._live_g is None:
                from .live_postread import LivePostreadApplication
                self._live_g = LivePostreadApplication(self.store_dir / "live-g", workspace_config=self.workspace_config)
            return self._live_g

    def research_g(self):
        if self._research_g is None:
            from .research_g import ResearchGApplication
            self._research_g = ResearchGApplication(self.store_dir / "research-g")
        return self._research_g

    def workflow(self, mode: str = "real", scenario: str = "normal", archive_replay: bool = True, *,
                 model_id: str = "market", information_mode: str = "operational", time_mode: str = "archive_replay",
                 platform_export=None, archive_root=None, platform_reader: bool = False, use_service: bool = True) -> Any:
        if mode not in ("real", "test"):
            raise ValueError("mode 仅支持 real 或 test")
        if scenario not in SCENARIOS:
            raise ValueError("未知测试场景")
        if type(archive_replay) is not bool:
            raise ValueError("archive_replay 必须是布尔值")
        if time_mode not in ("archive_replay", "live"):
            raise ValueError("time_mode 仅支持 archive_replay 或 live")
        if information_mode not in ("operational", "historical_reconstruction"):
            raise ValueError("information_mode 仅支持 operational 或 historical_reconstruction")
        if not isinstance(model_id, str):
            raise ValueError("model_id 必须是服务端模型目录中的 ID")
        if mode == "real":
            self.model_catalog.resolve(model_id)
        export = platform_export if platform_export is not None else self.platform_export
        archive = archive_root if archive_root is not None else self.archive_root
        key = (mode, scenario if mode == "test" else "normal", archive_replay if mode == "real" else False,
               model_id, information_mode, time_mode, export, str(archive), platform_reader, use_service)
        if key not in self.workflows:
            options = dict(model_id=model_id, model_catalog=self.model_catalog, information_mode=information_mode,
                           time_mode=time_mode, platform_export=export, service=self.service if use_service else None, archive_root=archive,
                           platform_reader=platform_reader)
            # Small injected factories used by embedders may implement the original signature.
            signature = inspect.signature(self.factory)
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
                options = {k: v for k, v in options.items() if k in signature.parameters}
            self.workflows[key] = self.factory(mode, key[1], self.store_dir, key[2], **options)
        return self.workflows[key]

    def run_workflow(self, run_id: str) -> Any:
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("run_id 格式错误")
        if run_id in self.run_owners:
            return self.run_owners[run_id]
        from football_agent.audit import RunStore
        # Loading the verified journal is read-only; recovery runs only after restoring its tools.
        record = RunStore(self.store_dir).load(run_id)
        configuration = record.get("tool_configuration") or next((event.get("data", {}).get("capabilities", {}) for event in record.get("events", []) if event.get("type") == "initialized"), {})
        if record.get("test_mode") or record.get("mode") in ("test", "synthetic"):
            scenario = record.get("scenario") or configuration.get("scenario", "normal")
            workflow = self.workflow("test", scenario if scenario in SCENARIOS else "normal")
        else:
            workflow = self.workflow("real", archive_replay=configuration.get("refresh_mechanism") == "archive_replay",
                                     model_id=configuration.get("model_id", "market"),
                                     information_mode=configuration.get("information_mode", "operational"),
                                     time_mode=configuration.get("time_mode", "archive_replay"),
                                     platform_export=configuration.get("export_path", configuration.get("platform_export")),
                                     archive_root=configuration.get("archive_root", self.archive_root),
                                     platform_reader=bool(configuration.get("reader_version")),
                                     use_service=configuration.get("service") is not None)
            current = workflow.tools.capabilities()
            for key in ("model_id", "model_version", "artifact_sha256", "information_mode", "time_mode", "export_path", "export_sha256", "reader_version", "configuration_hash"):
                if key in configuration and current.get(key) != configuration[key]:
                    raise ValueError(f"RUN_CONFIGURATION_CHANGED: {key} 与已存配置不一致")
        if hasattr(workflow, "policy"):
            from football_agent.engine import Workflow
            from football_agent.policy import FixedOrderPolicy, ToolPlanningPolicy
            version = record.get("policy_version", FixedOrderPolicy.version)
            parameters = record.get("policy_configuration", {})
            if version == FixedOrderPolicy.version:
                policy = FixedOrderPolicy(**parameters)
            elif version == ToolPlanningPolicy.version and not parameters:
                policy = ToolPlanningPolicy()
            else:
                raise ValueError("RUN_CONFIGURATION_CHANGED: 已存策略需要原服务端策略工厂")
            if workflow.policy.version != version or getattr(workflow.policy, "configuration", {}) != parameters:
                workflow = Workflow(workflow.tools, store_dir=self.store_dir, policy=policy)
        self.run_owners[run_id] = workflow
        return workflow

    def get(self, path: str, query: dict[str, list[str]]) -> Any:
        with self.lock:
            if path.startswith('/api/defense/'):
                if path == '/api/defense/catalog' and not query:return self.defense().catalog()
                if path == '/api/defense/case' and set(query)=={'case_id'}:return self.defense().case(query['case_id'][0])
                if path == '/api/defense/evidence' and set(query)=={'case_id','document_id'}:return self.defense().evidence(query['case_id'][0],query['document_id'][0])
                match=re.fullmatch(r'/api/defense/runs/(df-[a-f0-9]{32})',path)
                if match and not query:return self.defense().get(match[1])
                raise FileNotFoundError('冻结研究演示接口不存在')
            if path.startswith("/api/live-g/"):
                if query:
                    raise ValueError("Live GET accepts no query parameters")
                if path == "/api/live-g/catalog":
                    return self.live_g().catalog()
                if path == "/api/live-g/readiness":
                    return self.live_g().readiness()
                match = re.fullmatch(r"/api/live-g/runs/(lg-[a-f0-9]{32})(/view|/report)?", path)
                if match:
                    return self.live_g().get(match[1])
                raise FileNotFoundError("在线实验接口不存在")
            if path.startswith("/api/research-g/"):
                if query:
                    raise ValueError("Research configuration is server-owned; GET accepts no query parameters")
                if path == "/api/research-g/catalog":
                    return self.research_g().catalog()
                research_match = re.fullmatch(r"/api/research-g/runs/(rg-[a-f0-9]{32})(/view|/report)?", path)
                if research_match:
                    return self.research_g().get(research_match[1], report=research_match[2] == "/report")
                raise FileNotFoundError("研究运行接口不存在")
            mode = query.get("mode", ["real"])[0]
            scenario = query.get("scenario", ["normal"])[0]
            replay_value = query.get("archive_replay", ["true"])[0]
            if replay_value not in ("true", "false"):
                raise ValueError("archive_replay 仅支持 true 或 false")
            archive_replay = replay_value == "true"
            permitted = {"mode", "scenario", "archive_replay", "model_id", "information_mode", "time_mode", "cutoff"}
            if set(query) - permitted:
                raise ValueError("未知查询配置字段；模型与数据路径仅在服务端配置")
            options = dict(model_id=query.get("model_id", ["market"])[0],
                           information_mode=query.get("information_mode", ["operational"])[0],
                           time_mode=query.get("time_mode", ["archive_replay"])[0])
            if path == "/api/models":
                return {"models": self.model_catalog.list_models()}
            if path == "/api/config":
                capabilities = jsonable(self.workflow(mode, scenario, archive_replay, **options).capabilities())
                return {"mode": mode, "test_mode": mode == "test", "archive_replay": archive_replay, "scenarios": list(SCENARIOS), "capabilities": capabilities, "models": self.model_catalog.list_models(), "display_timezone": "Asia/Shanghai"}
            if path == "/api/matches":
                cutoff = query.get("cutoff", [None])[0]
                workflow = self.workflow(mode, scenario, archive_replay, **options)
                raw_matches = workflow.list_matches(cutoff=cutoff)
                # Keep injected/fake workflows byte-for-byte compatible with
                # the legacy boundary.  The production ToolResult receives
                # the versioned UI metadata and nullable eligibility fields.
                if is_dataclass(raw_matches) and not isinstance(raw_matches, type):
                    from .ui_projection import build_matches_view
                    return build_matches_view(raw_matches, mode=mode,
                                              information_mode=options["information_mode"],
                                              time_mode=options["time_mode"], cutoff=cutoff)
                return raw_matches
            blob = re.fullmatch(r"/api/runs/([^/]+)/blobs/([a-f0-9]{64})", path)
            if blob:
                from football_agent.audit import RunStore
                store = RunStore(self.store_dir)
                record = store.load(blob[1])
                reference = f"blobs/{blob[2]}.json"
                if not any(event.get("data", {}).get(key) == reference for event in record["events"] for key in ("input_ref", "output_ref")):
                    raise FileNotFoundError("运行审计未引用该内容")
                return store.get_blob(blob[1], reference)
            view_match = re.fullmatch(r"/api/runs/([^/]+)/view", path)
            if view_match:
                from .ui_projection import build_run_view
                record = self.run_workflow(view_match[1]).get_run(view_match[1])
                return build_run_view(record)
            match = re.fullmatch(r"/api/runs/([^/]+)(/report)?", path)
            if match:
                record = self.run_workflow(match[1]).get_run(match[1])
                if match[2]:
                    # The complete record preserves citations, event snapshots and costs.
                    return record
                return record
            raise FileNotFoundError("接口不存在")

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        with self.lock:
            if path == '/api/defense/runs':return self.defense().start(payload)
            defense_match=re.fullmatch(r'/api/defense/runs/(df-[a-f0-9]{32})/settle',path)
            if defense_match:return self.defense().settle(defense_match[1],payload)
            if path == "/api/live-g/refresh":
                return self.live_g().refresh(payload)
            if path == "/api/live-g/runs":
                return self.live_g().start(payload)
            live_match = re.fullmatch(r"/api/live-g/runs/(lg-[a-f0-9]{32})/stop", path)
            if live_match:
                return self.live_g().stop(live_match[1], payload)
            if path == "/api/research-g/runs":
                return self.research_g().start(payload)
            research_match = re.fullmatch(r"/api/research-g/runs/(rg-[a-f0-9]{32})/action", path)
            if research_match:
                return self.research_g().action(research_match[1], payload)
            if path == "/api/runs":
                permitted = {"mode", "scenario", "archive_replay", "model_id", "information_mode", "time_mode", "match_id", "prediction_cutoff", "budget"}
                if set(payload) - permitted:
                    raise ValueError("未知运行配置字段；模型文件与数据路径仅在服务端配置")
                match_id = payload.get("match_id")
                cutoff = payload.get("prediction_cutoff")
                if not isinstance(match_id, str) or not match_id.strip() or len(match_id) > 200:
                    raise ValueError("match_id 必须是非空字符串")
                if not isinstance(cutoff, str):
                    raise ValueError("prediction_cutoff 必须带时区")
                parsed = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("prediction_cutoff 必须带时区")
                workflow = self.workflow(payload.get("mode", "real"), payload.get("scenario", "normal"), payload.get("archive_replay", True),
                                         model_id=payload.get("model_id", "market"), information_mode=payload.get("information_mode", "operational"),
                                         time_mode=payload.get("time_mode", "archive_replay"))
                record = workflow.start(match_id, cutoff, budget=make_budget(payload.get("budget")))
                self.run_owners[record["run_id"]] = workflow
                return record
            match = re.fullmatch(r"/api/runs/([^/]+)/action", path)
            if match:
                if set(payload) - {"action", "topic"}:
                    raise ValueError("动作仅接受 action 与 topic；运行配置来自已存审计记录")
                workflow = self.run_workflow(match[1])
                action = payload.get("action", "next")
                if action not in ("next", "run", "query", "wait", "resume", "stop", "defer"):
                    raise ValueError("未知动作")
                if action == "run":
                    return workflow.run_to_completion(match[1])
                topic = payload.get("topic")
                if action == "query" and (not isinstance(topic, str) or not topic.strip() or len(topic) > 100):
                    raise ValueError("指定查询需要非空 topic")
                return workflow.step(match[1], action=None if action == "next" else action, topic=topic)
            raise FileNotFoundError("接口不存在")


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "FootballMindLocal/1.0"

    @property
    def app(self) -> DemoApplication:
        return self.server.application  # type: ignore[attr-defined]

    def _local_request(self) -> bool:
        host = self.headers.get("Host", "")
        allowed = {f"{self.server.server_address[0]}:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host.lower() not in allowed:
            self._error(403, "仅接受本地展示地址")
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + item for item in allowed}:
            self._error(403, "请求来源与本地展示地址不符")
            return False
        return True

    def _send(self, status: int, body: bytes, mime: str, download: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: Any, download: str | None = None) -> None:
        body = json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(200, body, "application/json; charset=utf-8", download)

    def _error(self, status: int, message: str) -> None:
        self._send(status, json.dumps({"status": "FAILED", "error": message}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _exception(self, exc: Exception) -> None:
        if type(exc).__name__ == "RunBusyError":
            self._error(409, str(exc))
        elif isinstance(exc, (FileNotFoundError, KeyError)):
            self._error(404, "运行记录或资源不存在")
        elif isinstance(exc, (ValueError, TypeError)):
            self._error(400, str(exc))
        else:
            self.log_error("%s: %s", type(exc).__name__, str(exc))
            self._error(500, f"{type(exc).__name__}: 工作流请求失败，请检查服务日志")

    def do_GET(self) -> None:
        if not self._local_request():
            return
        target = urlsplit(self.path)
        path = unquote(target.path)
        try:
            if path.startswith("/api/"):
                result = self.app.get(path, parse_qs(target.query))
                download = f"footballmind-{path.split('/')[3]}.json" if path.endswith("/report") else None
                self._json(result, download)
                return
            # Explicit allowlist: neither run files nor workspace paths are static assets.
            query = parse_qs(target.query)
            workspace = query.get("workspace", [""])[0]
            root_page = {
                "research": "index.html",
                "legacy": "index.html",
                "live": "live.html",
                "defense": "defense.html",
            }.get(workspace, "live.html")
            # Keep existing run bookmarks explicit; unrelated query parameters
            # must not silently switch the default UI to the old application.
            if not workspace and query.get("run"):
                root_page = "live.html" if query["run"][0].startswith("lg-") else "index.html"
            assets = {
                "/": (root_page, "text/html"),
                "/live.html": ("live.html", "text/html"),
                "/live.js": ("live.js", "text/javascript"),
                "/live-progress.js": ("live-progress.js", "text/javascript"),
                "/defense.html": ("defense.html", "text/html"),
                "/defense.js": ("defense.js", "text/javascript"),
                "/defense.css": ("defense.css", "text/css"),
                "/live.css": ("live.css", "text/css"),
                "/app.js": ("app.js", "text/javascript"),
                "/data-adapter.js": ("data-adapter.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if path not in assets:
                raise FileNotFoundError(path)
            filename, mime = assets[path]
            self._send(200, (STATIC_DIR / filename).read_bytes(), mime + "; charset=utf-8")
        except Exception as exc:
            self._exception(exc)

    def do_POST(self) -> None:
        if not self._local_request():
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("请求需使用 application/json")
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("请使用 Content-Length 请求正文")
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > MAX_BODY_BYTES:
                self._error(413, "请求正文为空或超过 64 KiB")
                return
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("请求正文必须是 JSON 对象")
            self._json(self.app.post(unquote(urlsplit(self.path).path), payload))
        except Exception as exc:
            self._exception(exc)


def create_server(host: str, port: int, app: DemoApplication | None = None) -> ThreadingHTTPServer:
    if host != "localhost":
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("展示服务仅支持 localhost 或 IPv4 回环地址") from exc
        if not address.is_loopback or address.version != 4:
            raise ValueError("展示服务仅支持 localhost 或 IPv4 回环地址")
    server = ThreadingHTTPServer((host, port), DemoHandler)
    server.daemon_threads = True
    server.application = app or DemoApplication()  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="FootballMind 本地工具工作流与事件回放")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--store-dir", help="Override the configured run directory")
    parser.add_argument("--config", help="Workspace JSON configuration file")
    parser.add_argument("--model-catalog", help="服务端模型白名单 JSON")
    parser.add_argument("--platform-export", help="服务端配置的已完成平台导出")
    parser.add_argument("--archive-root", default="football_spider/platform_data/raw")
    parser.add_argument("--enable-platform-refresh", action="store_true", help="启用既有spftll采集服务，每次最多2个请求")
    args = parser.parse_args()
    from footballmind.config import load_config
    workspace = load_config(args.config)
    from football_agent.model_catalog import ModelCatalog
    service = None
    if args.enable_platform_refresh:
        from football_agent.collector_refresh import PlatformRefreshAdapter
        service = PlatformRefreshAdapter(args.archive_root)
    server = create_server(args.host, args.port, DemoApplication(args.store_dir, model_catalog=ModelCatalog(args.model_catalog),
                          platform_export=args.platform_export, archive_root=args.archive_root, service=service,
                          workspace_config=workspace))
    print(f"FootballMind: http://{args.host}:{server.server_port} (Ctrl+C 停止)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

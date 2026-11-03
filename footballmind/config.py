"""Resolve explicit workspace paths without loading models or creating files."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

SCHEMA = "footballmind-workspace-v1"
RELEASE_ID = "footballmind-defense-20261025-rc1"
ENCODER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
F_SHA256 = "b36b2cf87f379034224c314794efc6ec59266dd4cf4752383965c41fb63661e0"
G_SHA256 = "5af947711279765838adb02e7dc09c2a9525a89e90db3e23e4f68a865ed88eff"
DEFAULT_HEAD = "state_ridge"
ALLOWED_HEADS = frozenset({"fulltext_mlp_seed20261027", "state_ridge", "fixed_order"})
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_FIELDS = ("release_dir", "encoder_dir", "cache_dir", "runs_dir", "experiments_dir", "data_root")
CONFIG_FIELDS = frozenset({"schema", "workspace_root", "default_head", *PATH_FIELDS})


@dataclass(frozen=True)
class WorkspaceConfig:
    workspace_root: Path
    release_dir: Path
    encoder_dir: Path
    cache_dir: Path
    runs_dir: Path
    experiments_dir: Path
    data_root: Path
    default_head: str = DEFAULT_HEAD
    config_path: Path | None = None
    release_id: str = RELEASE_ID
    encoder_revision: str = ENCODER_REVISION
    f_sha256: str = F_SHA256
    g_sha256: str = G_SHA256

    def as_dict(self) -> dict:
        """Resolved non-secret configuration, suitable for run provenance."""
        return {
            "schema": SCHEMA,
            "workspace_root": str(self.workspace_root),
            **{name: str(getattr(self, name)) for name in PATH_FIELDS},
            "default_head": self.default_head,
            "config_path": str(self.config_path) if self.config_path is not None else None,
            "release_id": self.release_id,
            "encoder_revision": self.encoder_revision,
            "f_sha256": self.f_sha256,
            "g_sha256": self.g_sha256,
        }

    def fingerprint(self) -> str:
        """Hash effective configuration; moving only its source file is immaterial."""
        values = self.as_dict()
        values.pop("config_path")
        return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()


def _string(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain a null byte")
    return value


def _resolve(value: str | os.PathLike, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate workspace configuration field: {key}")
        result[key] = value
    return result


def load_config(path=None, *, environ: Mapping[str, str] | None = None,
                workspace_root: str | os.PathLike | None = None) -> WorkspaceConfig:
    """Load explicit path > environment > local config > portable defaults.

    An explicit configuration filename is interpreted relative to the caller's
    current directory. Within the file, ``workspace_root`` is relative to that
    file's directory; all other paths are relative to the resolved workspace.
    Without a file's workspace_root, the checkout (or supplied root) is used.
    The function never creates directories, accesses a network, or loads weights.
    """
    env = os.environ if environ is None else environ
    project_root = Path(workspace_root).expanduser().resolve() if workspace_root is not None else PROJECT_ROOT
    config_path = None
    if path is not None:
        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("Configuration path must be a filename")
        if isinstance(path, str):
            _string(path, "Configuration path")
        config_path = Path(path).expanduser().resolve()
    elif "FOOTBALLMIND_CONFIG" in env:
        config_path = Path(_string(env["FOOTBALLMIND_CONFIG"], "FOOTBALLMIND_CONFIG")).expanduser().resolve()
    else:
        local = project_root / "config/workspace.local.json"
        if local.is_file():
            config_path = local

    raw = {}
    if config_path is not None:
        # Explicit paths must exist; no fallback to a different machine's model.
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
        if not isinstance(raw, dict):
            raise ValueError("Workspace configuration must be a JSON object")
        unknown = set(raw) - CONFIG_FIELDS
        if unknown:
            raise ValueError("Unknown workspace configuration fields: " + ", ".join(sorted(unknown)))
        if "schema" in raw and raw["schema"] != SCHEMA:
            raise ValueError("Unsupported workspace configuration schema")
        for name in ("workspace_root", *PATH_FIELDS):
            if name in raw:
                _string(raw[name], name)

    root = _resolve(raw["workspace_root"], config_path.parent) if "workspace_root" in raw else project_root
    head = _string(raw.get("default_head", DEFAULT_HEAD), "default_head")
    if head not in ALLOWED_HEADS:
        raise ValueError("default_head must name an approved frozen strategy")
    defaults = {
        "release_dir": f"var/releases/{RELEASE_ID}",
        "encoder_dir": f"var/models/Qwen3-Embedding-0.6B/snapshots/{ENCODER_REVISION}",
        "cache_dir": "var/cache/embeddings/live-postread-qwen-v1",
        "runs_dir": "var/runs",
        "experiments_dir": "var/experiments",
        "data_root": "var/data",
    }
    paths = {name: _resolve(raw.get(name, value), root) for name, value in defaults.items()}
    return WorkspaceConfig(workspace_root=root, config_path=config_path, default_head=head, **paths)

"""Configuration portability, precedence and no-side-effect contract."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from footballmind.config import ENCODER_REVISION, RELEASE_ID, SCHEMA, load_config


def write(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_portable_defaults_are_independent_of_current_directory(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_config(environ={}, workspace_root=checkout)
    assert config.release_dir == checkout / "var/releases" / RELEASE_ID
    assert config.encoder_dir == checkout / "var/models/Qwen3-Embedding-0.6B/snapshots" / ENCODER_REVISION
    assert config.default_head == "state_ridge"
    assert config.config_path is None
    assert not checkout.exists()


def test_explicit_root_relative_to_config_and_paths_relative_to_root(tmp_path):
    file = write(tmp_path / "settings/workspace.json", {
        "schema": SCHEMA, "workspace_root": "../project",
        "encoder_dir": "models/encoder", "cache_dir": "../cache",
        "default_head": "fulltext_mlp_seed20261027",
    })
    config = load_config(file, environ={})
    assert config.workspace_root == tmp_path / "project"
    assert config.encoder_dir == tmp_path / "project/models/encoder"
    assert config.cache_dir == tmp_path / "cache"
    assert config.config_path == file
    assert not config.workspace_root.exists()
    assert not config.cache_dir.exists()


def test_precedence_explicit_environment_local_default(tmp_path):
    local = write(tmp_path / "config/workspace.local.json", {"runs_dir": "local-runs"})
    environment = write(tmp_path / "environment.json", {"runs_dir": "env-runs"})
    explicit = write(tmp_path / "explicit.json", {"runs_dir": "explicit-runs"})
    assert load_config(explicit, environ={"FOOTBALLMIND_CONFIG": str(environment)}, workspace_root=tmp_path).runs_dir == tmp_path / "explicit-runs"
    assert load_config(environ={"FOOTBALLMIND_CONFIG": str(environment)}, workspace_root=tmp_path).runs_dir == tmp_path / "env-runs"
    assert load_config(environ={}, workspace_root=tmp_path).runs_dir == tmp_path / "local-runs"
    assert load_config(environ={}, workspace_root=tmp_path).config_path == local


def test_absent_workspace_root_in_file_uses_checkout_not_file_directory(tmp_path):
    file = write(tmp_path / "external/settings.json", {"runs_dir": "runs"})
    root = tmp_path / "checkout"
    assert load_config(file, environ={}, workspace_root=root).runs_dir == root / "runs"


def test_explicit_missing_config_does_not_fall_back(tmp_path):
    write(tmp_path / "config/workspace.local.json", {})
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.json", environ={}, workspace_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_config(environ={"FOOTBALLMIND_CONFIG": str(tmp_path / "absent.json")}, workspace_root=tmp_path)


@pytest.mark.parametrize("values", [
    [], {"typo_cache_dir": "cache"}, {"cache_dir": None}, {"runs_dir": 42},
    {"workspace_root": []}, {"schema": "future"}, {"default_head": "unfrozen"},
    {"default_head": False}, {"default_head": ""}, {"encoder_dir": "  "},
    {"release_id": "other-release"}, {"f_sha256": "forged"}, {"cache_dir": "bad\x00path"},
])
def test_invalid_config_is_rejected(tmp_path, values):
    with pytest.raises(ValueError):
        load_config(write(tmp_path / "settings.json", values), environ={})


def test_duplicate_fields_are_rejected(tmp_path):
    file = tmp_path / "settings.json"
    file.write_text('{"runs_dir":"one","runs_dir":"two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(file, environ={})


def test_fingerprint_tracks_effective_values_and_config_is_immutable(tmp_path):
    one = write(tmp_path / "one.json", {"runs_dir": "runs"})
    two = write(tmp_path / "two.json", {"runs_dir": "runs"})
    a = load_config(one, environ={}, workspace_root=tmp_path)
    b = load_config(two, environ={}, workspace_root=tmp_path)
    assert a.fingerprint() == b.fingerprint()
    assert len(a.fingerprint()) == 64
    c = load_config(write(tmp_path / "three.json", {"runs_dir": "other"}), environ={}, workspace_root=tmp_path)
    assert a.fingerprint() != c.fingerprint()
    assert a.as_dict()["config_path"] == str(one)
    assert a.as_dict()["schema"] == SCHEMA
    with pytest.raises(FrozenInstanceError):
        a.default_head = "fixed_order"


def test_empty_environment_config_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="FOOTBALLMIND_CONFIG"):
        load_config(environ={"FOOTBALLMIND_CONFIG": ""}, workspace_root=tmp_path)

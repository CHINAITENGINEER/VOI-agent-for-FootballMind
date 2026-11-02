"""A moved handoff must preserve the raw evidence and detect changed inputs."""
import json
from pathlib import Path
import shutil

import pytest

from football_agent.release_bundle import freeze_files, verify_bundle


def test_move_preserves_inventory_and_corrupted_raw_bytes_are_rejected(tmp_path):
    source = tmp_path/"source"
    (source/"raw").mkdir(parents=True)
    (source/"raw/body.gz").write_bytes(b"unaltered source bytes")
    (source/"runner.py").write_text("pass\n", encoding="utf-8")
    result = freeze_files(source, tmp_path/"package", ["raw/body.gz", "runner.py"])
    relocated = tmp_path/"elsewhere"/"package"
    relocated.parent.mkdir()
    shutil.move(str(tmp_path/"package"), relocated)
    assert verify_bundle(relocated) == result
    (relocated/"raw/body.gz").write_bytes(b"tampered source bytes")
    with pytest.raises(ValueError, match="Release file changed"):
        verify_bundle(relocated)


def test_missing_file_and_escaping_manifest_are_rejected(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    (source/"model.json").write_text("{}", encoding="utf-8")
    frozen = tmp_path/"frozen"
    freeze_files(source, frozen, ["model.json"])
    (frozen/"model.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_bundle(frozen)
    manifest_path = frozen/"release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../source/model.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="inside its root"):
        verify_bundle(frozen)


def test_frozen_output_never_overwrites_or_accepts_external_inputs(tmp_path):
    (tmp_path/"item").write_bytes(b"preserved")
    with pytest.raises(ValueError, match="already exists"):
        freeze_files(tmp_path, tmp_path, ["item"])
    with pytest.raises(ValueError, match="inside its root"):
        freeze_files(tmp_path, tmp_path/"new", ["../item"])
    assert not (tmp_path/"new").exists()
    assert (tmp_path/"item").read_bytes() == b"preserved"

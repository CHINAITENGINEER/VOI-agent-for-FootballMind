"""Checks the source-to-body binding used by the offline corpus queue."""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "prepare_postread_corpus_semantic",
    Path(__file__).resolve().parents[2] / "tools/prepare_postread_corpus_semantic.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def source(tmp_path, article_id="a", text="A body\u0085containing unicode separators\u2028kept verbatim"):
    p = tmp_path / "articles.jsonl"
    p.write_text("\n" + json.dumps({"article_id": article_id, "body_text": text}) + "\n", encoding="utf-8")
    candidate = {"candidate_text_id": article_id, "text_sha256": M.text_sha(text),
                 "text_reference": {"path": str(p), "jsonl_record": 1,
                                    "article_id": article_id, "body_sha256": M.text_sha(text)}}
    return candidate, text


def test_resolves_nonempty_record_and_preserves_all_text(tmp_path):
    row, text = source(tmp_path)
    assert M.extract_bodies([row]) == {M.text_sha(text): text}


def test_rejects_correct_position_but_wrong_article_id(tmp_path):
    row, _ = source(tmp_path)
    row["candidate_text_id"] = "wrong"
    with pytest.raises(ValueError, match="Article ID"):
        M.extract_bodies([row])


def test_rejects_changed_text_at_same_id(tmp_path):
    row, _ = source(tmp_path)
    row["text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Body differs"):
        M.extract_bodies([row])


def test_deduplicates_only_exact_text(tmp_path):
    row, text = source(tmp_path)
    assert len(M.extract_bodies([row, row])) == 1
    assert M.text_sha(text) != M.text_sha(text.replace("\u0085", "\n"))


def test_immutable_preparation_does_not_overwrite(tmp_path):
    p = tmp_path / "body.txt"
    M.immutable_bytes(p, b"original")
    M.immutable_bytes(p, b"original")
    with pytest.raises(ValueError):
        M.immutable_bytes(p, b"changed")
    assert p.read_bytes() == b"original"


def test_feature_whitelist_has_no_labels_or_after():
    assert not {"outcome", "gain", "result", "p_after", "prediction_after", "prior_choice"}.intersection(M.FIELDS)


def test_os_lock_is_exclusive_and_released(tmp_path):
    p = tmp_path / "writer.lock"
    with M.writer_lock(p):
        with pytest.raises(OSError):
            with M.writer_lock(p):
                pass
    with M.writer_lock(p):
        pass

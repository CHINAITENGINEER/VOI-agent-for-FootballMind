"""The competition evidence check is offline and read-only."""

from tools.check_footballmind_semantic_submission import run_acceptance


def test_semantic_submission_evidence_passes_without_external_calls():
    result = run_acceptance()
    assert result["status"] == "passed"
    assert result["network_calls"] == 0
    assert result["semantic_calls"] == 0
    assert result["cycles"] == 6
    assert result["observations"] == 5
    assert result["supports_match_specific"] == 1
    assert result["no_answer"] == 4
    assert result["directional_evidence_used_for_voi"] is False


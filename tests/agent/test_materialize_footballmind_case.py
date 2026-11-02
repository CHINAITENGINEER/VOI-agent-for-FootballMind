import pytest

from tools.materialize_footballmind_case import materialize


def bundle():
    return {
        "request": {
            "schema": "autonomous-match-request-v1",
            "target_key": "match-test",
            "mode": "prematch",
            "match": {"match_id": "m1", "home": "Santos", "away": "Flamengo",
                      "competition": "Brasileirao", "kickoff": "2026-10-10T12:00:00Z",
                      "prediction_cutoff": "2026-10-09T12:00:00Z"},
        },
        "state": {"schema": "autonomous-match-state-v1", "unknown_information": []},
    }


def profile():
    return {"teams": [{"side": "home", "canonical_name": "Santos", "input_name": "Santos"},
                       {"side": "away", "canonical_name": "Flamengo", "input_name": "Flamengo"}],
            "allowed_hosts": ["example.org"], "seed_urls": ["https://example.org/fixture"],
            "bootstrap_url": "https://example.org/fixture", "status": "verified"}


def test_materializer_requires_explicit_fixture_bound_inputs():
    case = materialize(bundle(), profile(), {"probabilities": [.5, .3, .2], "source": "frozen-test",
                                            "match_id": "m1", "prediction_cutoff": "2026-10-09T20:00:00+08:00"})
    assert case["schema"] == "footballmind-semantic-case-v2"
    assert case["source_profile"]["bootstrap_url"] == "https://example.org/fixture"
    assert case["structured_baseline"]["probabilities"] == [.5, .3, .2]
    assert case["text_model_prior"] is None


@pytest.mark.parametrize("bad", [
    {"probabilities": [.5, .3, .3]},
    {"probabilities": [.5, .5]},
])
def test_materializer_rejects_invalid_baseline(bad):
    bad = {**bad, "source": "frozen-test", "match_id": "m1",
           "prediction_cutoff": "2026-10-09T12:00:00Z"}
    with pytest.raises(ValueError, match="probabilities"):
        materialize(bundle(), profile(), bad)


def test_materializer_rejects_source_mismatch_without_fallback():
    wrong = profile()
    wrong["teams"][0]["canonical_name"] = "Palmeiras"
    wrong["teams"][0]["input_name"] = "Palmeiras"
    baseline = {"probabilities": [.5, .3, .2], "source": "frozen-test", "match_id": "m1",
                "prediction_cutoff": "2026-10-09T12:00:00Z"}
    with pytest.raises(ValueError, match="does not match match bundle"):
        materialize(bundle(), wrong, baseline)


def test_materializer_rejects_outcome_leakage():
    with pytest.raises(ValueError, match="outcome or evaluator"):
        materialize(bundle(), profile(), {"probabilities": [.5, .3, .2], "source": "frozen-test",
                                          "match_id": "m1", "prediction_cutoff": "2026-10-09T12:00:00Z",
                                          "outcome": "home"})


def test_materializer_rejects_unbound_baseline_context():
    with pytest.raises(ValueError, match="match_id"):
        materialize(bundle(), profile(), {"probabilities": [.5, .3, .2], "source": "frozen-test",
                                          "prediction_cutoff": "2026-10-09T12:00:00Z"})


def test_materializer_rejects_bootstrap_outside_declared_sources():
    wrong = profile()
    wrong["bootstrap_url"] = "https://other.example/fixture"
    with pytest.raises(ValueError, match="bootstrap_url must be present"):
        materialize(bundle(), wrong, {"probabilities": [.5, .3, .2], "source": "frozen-test",
                                     "match_id": "m1", "prediction_cutoff": "2026-10-09T12:00:00Z"})


def test_materializer_rejects_unadmitted_source_profile():
    wrong = profile()
    wrong["status"] = "partial"
    with pytest.raises(ValueError, match="admitted fixture-bound"):
        materialize(bundle(), wrong, {"probabilities": [.5, .3, .2], "source": "frozen-test",
                                     "match_id": "m1", "prediction_cutoff": "2026-10-09T12:00:00Z"})

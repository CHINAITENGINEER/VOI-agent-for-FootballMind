"""A valid raw evidence archive cannot enter the existing numerical f gateway."""
from datetime import datetime, timezone

import pytest

from football_research.data_adapters.prospective_training import _registered_runs
from test_evidence_collection import collect, inputs, no_network, transport


def test_valid_official_evidence_run_is_rejected_before_numeric_prior_loading(tmp_path, monkeypatch):
    prepared, _, plan, path = inputs(tmp_path)
    calls = transport(monkeypatch)
    report = collect(tmp_path, prepared, path, requests_per_match=1)
    assert report['status'] == 'collected_prior_unavailable'
    assert report['initial_probabilities'] is report['final_probabilities'] is None
    assert len(calls) == 1

    def numeric_prior_must_not_be_loaded(*args, **kwargs):
        pytest.fail('Evidence-only runs must be rejected before a numerical prior is requested')

    monkeypatch.setattr('football_research.data_adapters.prospective_training.verify_prior_archive',
                        numeric_prior_must_not_be_loaded)
    with pytest.raises(ValueError, match='Evidence-only acquisition has no numerical prior'):
        _registered_runs([tmp_path/'run'], [plan], datetime.now(timezone.utc))
    assert len(calls) == 1

"""Failures and missing local models must be visible to scripts using the CLI."""
from footballmind.__main__ import main


def test_failed_experiment_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr('footballmind.experiment.run_experiment', lambda *a: {'status': 'FAILED'})
    assert main(['experiment', 'run', 'unused.json']) == 1
    assert 'FAILED' in capsys.readouterr().out


def test_partial_experiment_returns_nonzero(monkeypatch):
    monkeypatch.setattr('footballmind.experiment.run_experiment', lambda *a: {'status': 'COMPLETED_WITH_FAILURES'})
    assert main(['experiment', 'run', 'unused.json']) == 1


def test_doctor_mode_checks_the_requested_capability(monkeypatch):
    monkeypatch.setattr('footballmind.doctor.diagnose', lambda *a, **kw: {
        'offline_ready': True, 'online_artifacts_ready': False})
    assert main(['doctor', '--mode', 'offline']) == 0
    assert main(['doctor', '--mode', 'online']) == 1


def test_missing_explicit_config_fails_instead_of_falling_back(tmp_path, capsys):
    assert main(['doctor', '--config', str(tmp_path / 'missing.json')]) == 1
    assert 'FileNotFoundError' in capsys.readouterr().err

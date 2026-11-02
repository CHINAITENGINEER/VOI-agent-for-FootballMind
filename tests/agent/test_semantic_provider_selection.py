"""The semantic runner selects one explicit provider without changing the runtime contract."""
from pathlib import Path

import pytest

from tools import run_footballmind_semantic_case as cli
from football_agent.semantic_scenario_protocol import (
    ObservationProtocol, DirectionalObservationProtocol, DirectionalProposalProtocol,
    CandidateDiscoveryProtocol, DirectionalActionExtensionProtocol,
)


class FakeDeepSeek:
    instances = []

    def __init__(self, workspace, **kwargs):
        self.workspace = Path(workspace)
        self.kwargs = kwargs
        self.__class__.instances.append(self)


def test_codex_factory_preserves_existing_defaults(tmp_path):
    factory = cli.build_provider_factory(
        provider="codex-cli", model=None, timeout_seconds=17,
        max_output_tokens=8192, thinking="disabled", api_key=None,
        output=tmp_path / "run")
    provider = factory(tmp_path / "workspace", object())
    assert provider.timeout_seconds == 17
    assert provider.model is None


def test_deepseek_factory_preflights_then_binds_key_without_printing_it(tmp_path, monkeypatch):
    FakeDeepSeek.instances = []
    monkeypatch.setattr(cli, "_deepseek_provider_class", lambda: FakeDeepSeek)
    factory = cli.build_provider_factory(
        provider="deepseek", model="deepseek-flash", timeout_seconds=21,
        max_output_tokens=4096, thinking="enabled", api_key="local-secret",
        output=tmp_path / "run")
    assert len(FakeDeepSeek.instances) == 1
    assert FakeDeepSeek.instances[0].kwargs == {
        "protocol": None, "model": "deepseek-flash", "timeout_seconds": 21,
        "max_tokens": 4096, "thinking": "enabled", "api_key": "local-secret",
    }
    provider = factory(tmp_path / "workspace", "protocol")
    assert provider.workspace == (tmp_path / "workspace").resolve()
    assert len(FakeDeepSeek.instances) == 2
    assert FakeDeepSeek.instances[1].kwargs["protocol"] == "protocol"
    # The factory never writes a credential into the output directory.
    assert not list((tmp_path / "run").rglob("*"))


def test_thinking_is_rejected_for_codex(tmp_path):
    with pytest.raises(ValueError, match="only supported"):
        cli.build_provider_factory(
            provider="codex-cli", model=None, timeout_seconds=17,
            max_output_tokens=8192, thinking="enabled", api_key=None,
            output=tmp_path / "run")


def test_deepseek_default_is_pro_with_low_effort(tmp_path):
    factory = cli.build_provider_factory(provider='deepseek', model=None,
        timeout_seconds=120, max_output_tokens=8192, thinking=None,
        api_key='local-test-key', output=tmp_path/'run')
    config = factory(tmp_path/'call', None).configuration()
    assert config['model'] == 'deepseek-v4-pro'
    assert config['thinking'] == 'enabled'
    assert config['reasoning_effort'] == 'low'


def test_unknown_provider_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported provider"):
        cli.build_provider_factory(
            provider="other", model=None, timeout_seconds=17,
            max_output_tokens=8192, thinking="disabled", api_key=None,
            output=tmp_path / "run")


@pytest.mark.parametrize("override", [None, "enabled", "disabled"])
def test_observation_thinking_is_explicit_and_does_not_change_other_stages(tmp_path, override):
    factory = cli.build_provider_factory(provider="deepseek", model=None, timeout_seconds=120,
        max_output_tokens=8192, thinking="enabled", api_key="local-test-key", output=tmp_path/"run",
        observation_thinking=override)
    for protocol in (ObservationProtocol(), DirectionalObservationProtocol(), DirectionalProposalProtocol(),
                     CandidateDiscoveryProtocol(), DirectionalActionExtensionProtocol()):
        provider = factory(tmp_path/protocol.version, protocol)
        is_observation = isinstance(protocol, (ObservationProtocol, DirectionalObservationProtocol))
        assert provider.configuration()["thinking"] == (override if is_observation and override else "enabled")
        assert provider.configuration()["model"] == "deepseek-v4-pro"
    assert not list(tmp_path.rglob("*"))


def test_observation_override_requires_deepseek_and_valid_value(tmp_path):
    common = dict(model=None, timeout_seconds=120, max_output_tokens=8192, thinking=None,
                  api_key="local-test-key", output=tmp_path/"run")
    with pytest.raises(ValueError, match="only supported"):
        cli.build_provider_factory(provider="codex-cli", observation_thinking="disabled", **common)
    with pytest.raises(ValueError, match="observation_thinking"):
        cli.build_provider_factory(provider="deepseek", observation_thinking="automatic", **common)


def test_live_missing_key_fails_before_case_or_source_access(tmp_path, monkeypatch, capsys):
    import sys
    from football_agent import deepseek_provider
    monkeypatch.setattr(deepseek_provider, 'resolve_api_key', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['run', '--provider', 'deepseek', '--output', str(tmp_path/'out'),
                                    '--case-dir', str(tmp_path/'missing-case')])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert 'DEEPSEEK_API_KEY is not configured' in capsys.readouterr().err
    assert not (tmp_path/'out').exists()


def test_math_replay_needs_no_provider_key(tmp_path, monkeypatch):
    import sys
    from football_agent import deepseek_provider
    archive = tmp_path/'frozen.json'
    archive.write_text('{}')
    monkeypatch.setattr(deepseek_provider, 'resolve_api_key',
                        lambda: pytest.fail('Replay accessed credentials'))
    monkeypatch.setattr(cli, 'replay_math', lambda run: {'status': 'passed'})
    monkeypatch.setattr(sys, 'argv', ['run', '--provider', 'deepseek', '--replay', str(archive)])
    cli.main()


def test_historical_missing_key_fails_before_loading_panel(tmp_path, monkeypatch, capsys):
    import sys
    from tools import run_historical_semantic_case as historical
    from football_agent import deepseek_provider
    monkeypatch.setattr(deepseek_provider, 'resolve_api_key', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['run', '--provider', 'deepseek', '--output', str(tmp_path/'out'),
                                    '--panel', str(tmp_path/'missing-panel')])
    with pytest.raises(SystemExit) as error:
        historical.main()
    assert error.value.code == 2
    assert 'DEEPSEEK_API_KEY is not configured' in capsys.readouterr().err
    assert not (tmp_path/'out').exists()

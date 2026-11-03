"""Configured runtime and startup checks must not start source collection."""
from dataclasses import replace
import json
from pathlib import Path
import sys

from footballmind.config import load_config
from demo.live_postread import LivePostreadApplication
from demo.server import DemoApplication
import start_footballmind_defense as launcher


def test_live_readiness_uses_injected_paths_without_refresh(tmp_path, monkeypatch):
    config = load_config(workspace_root=tmp_path, environ={})
    seen = []
    def readiness(**kwargs):
        seen.append(kwargs['workspace_config'])
        return {'ready': False, 'missing_files': ['encoder']}
    monkeypatch.setattr('demo.live_postread.LivePostreadModel.readiness', readiness)
    monkeypatch.setattr('demo.live_g.LiveGApplication.catalog', lambda *a: (_ for _ in ()).throw(AssertionError('No discovery')))
    app = LivePostreadApplication(config.runs_dir, backend=object(), workspace_config=config)
    status = app.readiness()
    assert seen == [config]
    assert status['default_model_name'] == 'state_ridge'
    assert status['service_alive'] and not status['encoder_ready']
    assert status['sources_status'] == 'not_probed' and status['network_requests'] == 0
    assert not config.runs_dir.exists()


def test_server_forwards_workspace_to_live_and_frozen_apps(tmp_path):
    config = load_config(workspace_root=tmp_path, environ={})
    app = DemoApplication(workspace_config=config)
    assert app.store_dir == config.runs_dir
    assert app.defense().release_dir == config.release_dir
    assert app.live_g().workspace_config == config


def test_readiness_route_does_not_call_live_catalog(tmp_path):
    app = DemoApplication(tmp_path)
    class Live:
        def readiness(self):
            return {'encoder_ready': False, 'network_requests': 0}
        def catalog(self):
            raise AssertionError('Would trigger source refresh')
    app._live_g = Live()
    assert app.get('/api/live-g/readiness', {}) == {'encoder_ready': False, 'network_requests': 0}


def test_launcher_rejects_live_service_with_missing_encoder(tmp_path, monkeypatch, capsys):
    config = load_config(workspace_root=tmp_path, environ={})
    config.release_dir.mkdir(parents=True)
    (config.release_dir / 'footballmind_defense.py').write_text('')
    monkeypatch.setattr(launcher, 'load_config', lambda *a: config)
    monkeypatch.setattr(launcher.importlib.util, 'find_spec', lambda *a: object())
    monkeypatch.setattr('demo.live_postread_model.LivePostreadModel.readiness',
                        lambda **kw: {'ready': False, 'missing_files': ['model.safetensors']})
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Must not launch')))
    monkeypatch.setattr(sys, 'argv', ['launcher', '--page', 'live', '--no-browser'])
    assert launcher.main() == 1
    assert 'No service or acquisition was started' in capsys.readouterr().out
    assert not config.runs_dir.exists()


def test_launcher_checks_encoder_and_config_fingerprint_on_reuse(tmp_path, monkeypatch):
    config = load_config(workspace_root=tmp_path, environ={})
    state = {'ready': False, 'fingerprint': config.fingerprint()}
    urls = []
    class Response:
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return json.dumps(self.value).encode()
    class Opener:
        def open(self, url, **kw):
            urls.append(url)
            if url.endswith('/api/defense/catalog'):
                return Response({'frozen_research_demo': True, 'release_ready': True,
                    'release_id': config.release_id, 'workspace_root': str(launcher.ROOT),
                    'release_dir': str(config.release_dir)})
            assert url.endswith('/api/live-g/readiness')
            return Response({'pipeline_mode': 'postread_fulltext_v1',
                'runtime_bindings': {'reload_required': False},
                'encoder_ready': state['ready'], 'configuration_fingerprint': state['fingerprint']})
    monkeypatch.setattr(launcher, 'build_opener', lambda *a: Opener())
    assert launcher.health(8766, 'live', config) is None
    state['ready'] = True
    assert launcher.health(8766, 'live', config)
    state['fingerprint'] = 'different-workspace'
    assert launcher.health(8766, 'live', config) is None
    assert all(not u.endswith('/api/live-g/catalog') for u in urls)

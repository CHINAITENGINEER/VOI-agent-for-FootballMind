import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from football_research.data_adapters.phase4_historical_panel import PanelAdapter, paragraphs
from football_spider.storage.raw_archive import atomic_write, json_bytes


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return type("Response", (), {"status_code": 200, "content": b"<html>archived</html>",
                                    "headers": {"Content-Type": "text/html", "Location": "/next"},
                                    "close": lambda self: None})()

    def close(self):
        pass


class HistoricalPanelTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = {"request_budget": 6, "random_seed": 20261009,
            "budget": {"per_fixture_actual_http_max": 2, "pilot_http_max": 4},
            "pilot_order": ["understat:1", "understat:2"], "allowed_hosts": ["example.com"],
            "selected_fixtures": [{"physical_match_id": "understat:" + str(x)} for x in (1,2,3)]}
        atomic_write(self.root / "frozen-plan.json", json_bytes(self.plan))
        self.session = FakeSession()
        self.adapter = PanelAdapter(self.root, session=self.session)
        self.adapter.transport.interval = 0
        self.fixture = {"physical_match_id": "understat:1"}

    def tearDown(self):
        self.adapter.close()
        self.temp.cleanup()

    def test_journal_receipt_raw_identity_and_real_clock_order(self):
        obs, attempt, raw = self.adapter.fetch(self.fixture, "https://example.com/one?q=two")
        self.assertEqual(raw, b"<html>archived</html>")
        run = self.adapter.journal.read(self.adapter.run_id)
        self.assertEqual(run["budget"]["charged_requests"], 1)
        self.assertEqual(obs.context["journal_request_id"], attempt["journal_request_id"])
        self.assertEqual(attempt["response_headers"]["location"], "/next")
        self.assertLessEqual(attempt["response_at"], attempt["archive_save_started_at"])
        self.assertLessEqual(attempt["archive_save_started_at"], attempt["archive_save_returned_at"])
        self.assertLessEqual(attempt["archive_save_returned_at"], attempt["collector_handoff_at"])
        self.assertIsNone(attempt["available_at"])
        self.assertIsNone(run["fixture"])
        self.assertFalse(self.session.calls[0][1]["allow_redirects"])

    def test_recreated_adapter_cannot_reset_owner_cap_or_repeat_request(self):
        self.adapter.fetch(self.fixture, "https://example.com/one")
        with self.assertRaisesRegex(ValueError, "already archived"):
            self.adapter.fetch(self.fixture, "https://example.com/one")
        self.adapter.close()
        self.adapter = PanelAdapter(self.root, session=self.session)
        self.adapter.transport.interval = 0
        self.adapter.fetch(self.fixture, "https://example.com/two")
        with self.assertRaisesRegex(ValueError, "per-fixture"):
            self.adapter.fetch(self.fixture, "https://example.com/three")
        self.assertEqual(len(self.session.calls), 2)

    def test_expansion_and_candidate_gate_prevent_http(self):
        with self.assertRaisesRegex(ValueError, "pilot gate"):
            self.adapter.fetch({"physical_match_id": "understat:3"}, "https://example.com/one")
        self.adapter.candidate_menu = [{"candidate_id": "known", "url": "https://example.com/known"}]
        with self.assertRaisesRegex(ValueError, "not exposed"):
            self.adapter.fetch(self.fixture, "https://example.com/unknown", candidate_id="unknown")
        self.assertEqual(self.session.calls, [])

    def test_freeze_preserves_empty_pre_read_title_and_exact_offsets(self):
        parent = {"discovery_id": "parent", "visible_candidates": [], "observed_at": "2026-10-09T00:00Z"}
        candidates = [{"candidate_id": "one", "url": "https://example.com/one", "title": "", "snippet": None, "rank": 10602}]
        menu = self.adapter.freeze_menu(self.fixture, parent, candidates, "URL only")
        self.assertEqual(menu["visible_candidates"][0]["title"], "")
        self.assertEqual(menu["visible_candidates"][0]["rank"], 10602)
        self.assertFalse(menu["original_agent_menu_proven"])
        body = "Heading\n\nText with 中文 and quotes.\n"
        for p in paragraphs(body):
            self.assertEqual(body[p["start"]:p["end"]], p["text"])

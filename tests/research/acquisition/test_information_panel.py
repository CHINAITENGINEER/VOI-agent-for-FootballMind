"""Finite-panel boundary checks with actual archived synthetic response bytes."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import shutil

import pytest

from football_agent.inquiry_tools import OpenWebTools
from football_agent.learned_voi_runtime import _save
from football_research.data_adapters.information_panel import (
    _digest, freeze_panel_design, seal_panel, load_panel,
)
from football_spider.storage.raw_archive import RawArchive
from football_spider.models.contracts import Observation
from test_verified_fixtures import NOW, KICKOFF, identity, manifest
from test_verified_collection import verified_inputs, _collect


@pytest.fixture
def actor(tmp_path):
    identity_archive, binding = identity(tmp_path)
    folder = tmp_path / "actor"
    shutil.copytree(identity_archive.root, folder / "identity-archive")
    archive = RawArchive(folder / "source-archive")
    index_url = "https://www.uefa.com/news/"

    def action(key, url):
        return {"action_id": key, "information_key": "page:" + key, "tool": "read_url", "url": url,
                "source": "open_web", "available": True, "cost": {"requests": 1., "money": .01}, "cost_status": "simulated"}

    actions = [action(key, f"https://www.uefa.com/news/{key}") for key in ("a", "b", "used")]
    index_action = action("index", index_url)
    at = NOW + timedelta(minutes=2, seconds=30)
    tools = OpenWebTools(archive, ["www.uefa.com"], extraction_version=3, clock=lambda: at)
    index_body = "<html><main>Already known source index." + "".join(
        '<a href="' + a["url"] + '">' + a["action_id"] + "</a>" for a in actions) + "</main></html>"

    def receipt(a, body, observed):
        prepared = {"source": "open_page", "tool": "read_url", "endpoint": a["url"], "params": {}}
        obs = archive.save("open_page", a["url"], {}, body, status=200, observed_at=observed,
            context={"requested_at": (observed-timedelta(seconds=1)).isoformat(), "inquiry_run_id": "actor", "inquiry_request_id": a["action_id"]})
        return {"status": "OK", "document": tools.project(obs, prepared),
                "receipt": {"observation": obs.model_dump(mode="json"), "prepared": prepared}}

    first_receipt = receipt(index_action, index_body.encode(), at)
    first_public = {"decision_at": (at-timedelta(seconds=1)).isoformat(), "documents": [], "attempts": [],
        "prediction": {"probabilities": [.5, .3, .2], "model_version": "archived-test"}, "state": {"match_id": "spftll:match:m1"}}
    first = {"step": 0, "selected_action": index_action, "generated_actions": [index_action],
             "public_before": first_public, "public_digest": _digest(first_public)}
    second_public = deepcopy(first_public)
    second_public.update(decision_at=(NOW+timedelta(minutes=3, seconds=10)).isoformat(),
        documents=[first_receipt["document"]], attempts=[{"action_id": "index", "status": "OK"}],
        prediction={"probabilities": [.6, .2, .2], "model_version": "archived-test-updated"})
    second = {"step": 1, "selected_action": actions[-1], "generated_actions": actions,
              "public_before": second_public, "public_digest": _digest(second_public)}
    at = NOW+timedelta(minutes=3, seconds=11)
    second_receipt = receipt(actions[-1], b"<html>Already executed different response.</html>", at)
    match = {"match_id": "spftll:match:m1", "home_team": "克罗地亚", "away_team": "西班牙",
             "home_aliases": binding["aliases"]["home"], "away_aliases": binding["aliases"]["away"],
             "kickoff_at": KICKOFF.isoformat(), "prediction_cutoff": (KICKOFF-timedelta(minutes=1)).isoformat()}
    config = {"schema": "learned-voi-open-execution-v2", "policy_manifest": manifest(),
        "created_at": (NOW+timedelta(minutes=2)).isoformat(), "evidence_projection": {"match": match},
        "initial_state": {"match_id": match["match_id"], "prediction_cutoff": match["prediction_cutoff"]},
        "supervisor_metadata": {"verified_fixture_identity": binding, "transport": tools.configuration(),
                                "generator": {"seed_urls": [index_url]}}}
    run = {"schema": config["schema"], "run_id": "actor", "configuration_digest": _digest(config),
           "initial_probabilities": [.5, .3, .2], "cycles": [first, second],
           "finished_at": (NOW+timedelta(minutes=4)).isoformat(), "terminal_selection": {"decision": "defer"}}
    for name, value in (("configuration", config), ("run", run), ("cycle-000", first), ("cycle-001", second),
                        ("receipt-000", first_receipt), ("receipt-001", second_receipt)):
        _save(folder / (name + ".json"), value)
    return folder


def freeze(actor):
    output = actor.parent / "panel" / "design.json"
    design = freeze_panel_design(actor, step=1, frozen_at=NOW+timedelta(minutes=5), output=output)
    return output, design


def capture(design, archive, action_id, *, text=None, http_status=200, url=None):
    index = [a["action_id"] for a in design["candidates"]].index(action_id)
    at = NOW+timedelta(minutes=6+index)
    tools = OpenWebTools(archive, ["www.uefa.com"], extraction_version=3, clock=lambda: at)
    action = next(a for a in design["candidates"] if a["action_id"] == action_id)
    request = tools.prepare(action, design["public_before"])
    if url is not None:
        request["endpoint"] = url
    content = text if text is not None else "Croatia vs Spain on 1 October 2026. Private response " + action_id + "."
    body = ("<html><article>" + content + "</article></html>").encode() if content else b""
    request_id = "research-" + action_id
    decision = at-timedelta(seconds=1)
    obs = archive.save(request["source"], request["endpoint"], request["params"], body,
        status=http_status, observed_at=at, context={"requested_at": decision.isoformat(),
            "inquiry_run_id": design["panel_id"], "inquiry_request_id": request_id})
    return {"action_id": action_id, "decision_at": decision.isoformat(), "returned_at": (at+timedelta(seconds=1)).isoformat(), "prepared": request,
        "result": {"document": tools.project(obs, request), "observation": obs.model_dump(mode="json"),
                   "request_id": request_id, "http_attempts": 1, "latency_seconds": .4,
                   "money": None, "money_status": "not_measured"}}


def seal(actor, texts=None, statuses=None):
    design_path, design = freeze(actor)
    archive = RawArchive(design_path.parent / "source-archive")
    rows = [capture(design, archive, key, text=(texts or {}).get(key), http_status=(statuses or {}).get(key, 200)) for key in ("a", "b")]
    path = design_path.parent / "panel.json"
    panel = seal_panel(design_path, rows, panel_as_of=NOW+timedelta(minutes=8), archive_folder=archive.root, output=path)
    return path, panel, rows


def test_reveal_is_isolated_from_actor_and_preserves_known_history_prediction(actor):
    original = {p.name: p.read_bytes() for p in actor.glob("*.json")}
    path, panel, _ = seal(actor)
    assert panel["audit"]["fully_observed_panel"] and panel["audit"]["multi_article_panel"]
    assert all(c["qualification"]["target_scope"] == "match_specific" for c in panel["captures"])
    environment = load_panel(path)
    before = environment.public_state()
    assert "Private response" not in json.dumps(before)
    assert before["prediction"]["probabilities"] == [.6, .2, .2]
    assert before["initial_probabilities"] == [.5, .3, .2]
    assert before["documents"][0]["text"].startswith("Already known")
    assert before["attempts"] == [{"action_id": "index", "status": "OK"}]
    assert panel["design"]["actor_public_before_sha256"] == _digest(panel["design"]["actor_public_before"])
    returned = environment.reveal("b")
    assert returned["network_requests"] == 0 and returned["actual_resources"]["money"] is None
    after = environment.public_state()
    assert "Private response b" in json.dumps(after) and "Private response a" not in json.dumps(after)
    assert after["panel_budget"]["requests_spent"] == 1
    assert {p.name: p.read_bytes() for p in actor.glob("*.json")} == original
    with pytest.raises(ValueError, match="already revealed"):
        environment.reveal("b")
    with pytest.raises(ValueError, match="No actual captured"):
        environment.reveal("not-in-menu")


def test_runtime_written_terminal_snapshot_uses_bound_bytes_without_fabricating_digest(verified_inputs, monkeypatch):
    # Run the actual collector/runtime/archive code. Only network transport is
    # replaced by local source bytes; the terminal cycle schema is not handmade.
    verified_inputs["body"] = verified_inputs["body"].replace(b"</html>",
        b'<a href="https://www.uefa.com/news/a">Croatia fitness</a>'
        b'<a href="https://www.uefa.com/news/b">Spain preparation</a>'
        b'<a href="https://www.uefa.com/news/c">Croatia preparations</a></html>')
    folder, _, _ = _collect(verified_inputs, monkeypatch)
    run = json.loads((folder/"run.json").read_text(encoding="utf-8"))
    terminal = run["cycles"][-1]
    assert terminal["selected_action"] is None and terminal["execution_status"] == "not_requested"
    assert "public_digest" not in terminal
    original = (folder/f"cycle-{terminal['step']:03}.json").read_bytes()
    design = freeze_panel_design(folder, step=terminal["step"], frozen_at=datetime.now(timezone.utc),
                                 output=verified_inputs["root"]/"runtime-panel-design.json")
    assert len(design["candidates"]) == 2
    assert design["public_digest_basis"] == "derived_from_bound_terminal_snapshot"
    assert design["actor_public_before"] == terminal["public_before"]
    assert design["actor_public_before_sha256"] == _digest(terminal["public_before"])
    assert (folder/f"cycle-{terminal['step']:03}.json").read_bytes() == original


def test_missing_digest_on_executed_cycle_remains_invalid(actor):
    run = json.loads((actor/"run.json").read_text(encoding="utf-8"))
    del run["cycles"][1]["public_digest"]
    _save(actor/"run.json", run)
    _save(actor/"cycle-001.json", run["cycles"][1])
    with pytest.raises(ValueError, match="snapshot binding"):
        freeze(actor)


def test_caller_mutation_of_returned_document_and_cost_cannot_modify_panel(actor):
    path, _, _ = seal(actor)
    environment = load_panel(path)
    returned = environment.reveal("a")
    visible = environment.public_state()
    returned["document"]["text"] = "Caller-local cleanup"
    returned["declared_cost"]["money"] = 99
    returned["qualification"]["article"] = False
    assert environment.public_state() == visible
    assert load_panel(path).reveal("a")["declared_cost"]["money"] == .01


def test_sampling_cannot_be_reordered_after_seeing_content(actor):
    with pytest.raises(ValueError, match="first two"):
        freeze_panel_design(actor, step=1, candidate_ids=["b", "a"], frozen_at=NOW+timedelta(minutes=5), output=actor.parent/"bad.json")


@pytest.mark.parametrize("frozen", [NOW+timedelta(minutes=3), KICKOFF])
def test_design_refuses_earlier_actor_or_expired_plan(actor, frozen):
    with pytest.raises(ValueError, match="completed run.*prediction cutoff"):
        freeze_panel_design(actor, step=1, frozen_at=frozen, output=actor.parent/"bad.json")


def test_missing_capture_stays_unknown_not_an_empty_return(actor):
    design_path, design = freeze(actor)
    archive = RawArchive(design_path.parent/"source-archive")
    rows = [capture(design, archive, "a")]
    output = design_path.parent/"partial.json"
    panel = seal_panel(design_path, rows, panel_as_of=NOW+timedelta(minutes=8), archive_folder=archive.root, output=output)
    assert panel["audit"]["missing_action_ids"] == ["b"]
    assert not panel["audit"]["fully_observed_panel"]
    with pytest.raises(ValueError, match="unknown is not an empty"):
        load_panel(output).reveal("b")


def test_failed_return_counts_observed_but_does_not_manufacture_article(actor):
    path, panel, _ = seal(actor, texts={"b": ""}, statuses={"b": 503})
    assert panel["audit"]["fully_observed_panel"] and not panel["audit"]["multi_article_panel"]
    failure = load_panel(path).reveal("b")
    assert failure["document"]["status"] == "FAILED"
    assert failure["actual_resources"]["http_attempts"] == 1
    assert failure["qualification"]["document_role"] == "failed_return"


def test_team_context_and_unscoped_articles_are_retained_without_value_judgment(actor):
    _, panel, _ = seal(actor, texts={"a": "Croatia has announced a squad update.", "b": "Other sport elsewhere."})
    assert panel["audit"]["multi_article_panel"]
    assert [row["qualification"]["target_scope"] for row in panel["captures"]] == ["team_context", "unscoped"]


def test_reprinted_identical_body_is_not_two_independent_information_units(actor):
    _, panel, _ = seal(actor, texts={"a": "Same Croatia story.", "b": "Same Croatia story."})
    assert panel["audit"]["fully_observed_panel"]
    assert panel["audit"]["distinct_article_contents"] == 1
    assert not panel["audit"]["multi_article_panel"]


def test_content_already_visible_is_not_counted_as_new_article_information(actor):
    known = json.loads((actor/"receipt-000.json").read_text(encoding="utf-8"))["document"]["text"]
    _, panel, _ = seal(actor, texts={"a": known})
    assert panel["audit"]["fully_observed_panel"]
    assert panel["captures"][0]["qualification"]["already_visible_content"]
    assert panel["audit"]["distinct_article_contents"] == 1


def test_unread_body_cannot_change_first_reveal_via_global_duplicate_flags(actor):
    design_path, design = freeze(actor)
    archive = RawArchive(design_path.parent/"source-archive")
    same_a = capture(design, archive, "a", text="Same source story.")
    b = capture(design, archive, "b", text="Same source story.")
    alternate_archive = RawArchive(design_path.parent/"alternate-source-archive")
    different_a = capture(design, alternate_archive, "a", text="Completely different source story.")
    obs = Observation.model_validate(b["result"]["observation"])
    alternate_archive.save(obs.source, obs.endpoint, obs.params, archive.read_body(obs), status=obs.http_status,
        observed_at=obs.observed_at, observation_id=obs.id, request_id=obs.request_id, context=obs.context)
    panels = []
    for index, (a, store) in enumerate(((same_a, archive), (different_a, alternate_archive))):
        output = design_path.parent/f"panel-{index}.json"
        panel = seal_panel(design_path, [a, b], panel_as_of=NOW+timedelta(minutes=8), archive_folder=store.root, output=output)
        panels.append((panel, load_panel(output)))
    assert panels[0][0]["audit"]["distinct_article_contents"] == 1
    assert panels[1][0]["audit"]["distinct_article_contents"] == 2
    assert panels[0][1].public_state() == panels[1][1].public_state()
    assert panels[0][1].reveal("b") == panels[1][1].reveal("b")
    assert panels[0][1].public_state() == panels[1][1].public_state()
    assert panels[0][1].reveal("a")["qualification"]["duplicate_article_content"]
    assert not panels[1][1].reveal("a")["qualification"]["duplicate_article_content"]


def test_all_actual_same_panel_attempts_must_be_disclosed(actor):
    design_path, design = freeze(actor)
    archive = RawArchive(design_path.parent/"source-archive")
    capture(design, archive, "a", text="First actual failed request.", http_status=503)
    latest_a = capture(design, archive, "a")
    b = capture(design, archive, "b")
    with pytest.raises(ValueError, match="omit actual archived attempts"):
        seal_panel(design_path, [latest_a, b], panel_as_of=NOW+timedelta(minutes=8), archive_folder=archive.root, output=design_path.parent/"bad.json")


def test_other_research_runs_may_share_archive_without_becoming_panel_cost(actor):
    design_path, design = freeze(actor)
    archive = RawArchive(design_path.parent/"source-archive")
    rows = [capture(design, archive, "a"), capture(design, archive, "b")]
    archive.save("open_page", "https://www.uefa.com/other", {}, b"Other run data", status=200,
                 observed_at=NOW+timedelta(minutes=6), context={"inquiry_run_id": "unrelated-run"})
    panel = seal_panel(design_path, rows, panel_as_of=NOW+timedelta(minutes=8), archive_folder=archive.root, output=design_path.parent/"panel.json")
    assert panel["audit"]["actual_http_attempts"] == 2


def test_directory_returns_complete_the_panel_without_becoming_articles(actor):
    run = json.loads((actor/"run.json").read_text(encoding="utf-8"))
    cycle = run["cycles"][1]
    for action, url in zip(cycle["generated_actions"][:2], ("https://www.uefa.com/en/news", "https://www.uefa.com/fixtures")):
        action["url"] = url
        cycle["public_before"]["documents"][0]["links"].append({"url": url, "title": "Directory"})
    # New links must come from actual raw extraction, not patched public text.
    archive = RawArchive(actor/"source-archive")
    first = json.loads((actor/"receipt-000.json").read_text(encoding="utf-8"))
    at = NOW+timedelta(minutes=2, seconds=30)
    tools = OpenWebTools(archive, ["www.uefa.com"], extraction_version=3, clock=lambda: at)
    prepared = first["receipt"]["prepared"]
    content = "<html>Already known source index." + "".join('<a href="'+a["url"]+'">'+a["action_id"]+'</a>' for a in cycle["generated_actions"]) + "</html>"
    obs = archive.save("open_page", prepared["endpoint"], {}, content.encode(), status=200, observed_at=at,
                       context={"requested_at": (at-timedelta(seconds=1)).isoformat()})
    first.update(document=tools.project(obs, prepared), receipt={"observation": obs.model_dump(mode="json"), "prepared": prepared})
    cycle["public_before"]["documents"] = [first["document"]]
    cycle["public_digest"] = _digest(cycle["public_before"])
    for name, value in (("run", run), ("cycle-001", cycle), ("receipt-000", first)):
        _save(actor/(name+".json"), value)
    _, panel, _ = seal(actor)
    assert panel["audit"]["fully_observed_panel"]
    assert not panel["audit"]["multi_article_panel"]
    assert all(c["qualification"]["document_role"] == "index_page" for c in panel["captures"])


@pytest.mark.parametrize("as_of", [NOW+timedelta(minutes=4), NOW+timedelta(minutes=6), KICKOFF])
def test_panel_cannot_backdate_its_actual_availability_or_pass_cutoff(actor, as_of):
    design_path, design = freeze(actor)
    archive = RawArchive(design_path.parent/"source-archive")
    rows = [capture(design, archive, "a"), capture(design, archive, "b")]
    with pytest.raises(ValueError, match="snapshot|backdated"):
        seal_panel(design_path, rows, panel_as_of=as_of, archive_folder=archive.root, output=design_path.parent/"bad.json")


@pytest.mark.parametrize("mutation", ["body", "observation_clock", "prepared_url", "label", "unknown_action", "missing_return_clock", "early_return"])
def test_modified_or_unbound_actual_capture_cannot_be_sealed(actor, mutation):
    design_path, design = freeze(actor)
    archive = RawArchive(design_path.parent/"source-archive")
    row = capture(design, archive, "a")
    if mutation == "body": row["result"]["document"]["text"] = "invented"
    elif mutation == "observation_clock": row["result"]["observation"]["observed_at"] = (NOW+timedelta(minutes=5, seconds=59)).isoformat()
    elif mutation == "prepared_url": row["prepared"]["endpoint"] += "-other"
    elif mutation == "label": row["outcome"] = 0
    elif mutation == "missing_return_clock": del row["returned_at"]
    elif mutation == "early_return": row["returned_at"] = row["decision_at"]
    else: row["action_id"] = "unknown"
    with pytest.raises(ValueError):
        seal_panel(design_path, [row], panel_as_of=NOW+timedelta(minutes=8), archive_folder=archive.root, output=design_path.parent/"bad.json")


def test_zero_network_loader_rechecks_raw_bytes_and_immutable_outputs(actor, monkeypatch):
    path, panel, _ = seal(actor)
    monkeypatch.setattr(OpenWebTools, "execute", lambda *a, **k: pytest.fail("offline load made HTTP"))
    assert load_panel(path).reveal("a")["network_requests"] == 0
    with pytest.raises(FileExistsError):
        seal_panel(path.parent/"design.json", panel["captures"], panel_as_of=panel["panel_as_of"], archive_folder=panel["archive_folder"], output=path)
    obs = panel["captures"][0]["result"]["observation"]
    (Path(panel["archive_folder"])/obs["payload_path"]).write_bytes(gzip.compress(b"changed raw"))
    with pytest.raises(ValueError, match="checksum"):
        load_panel(path)


def test_loader_rejects_late_snapshot_even_with_rehashed_envelope(actor):
    path, panel, _ = seal(actor)
    panel["panel_as_of"] = KICKOFF.isoformat()
    panel["panel_sha256"] = _digest({k: v for k, v in panel.items() if k != "panel_sha256"})
    path.write_text(json.dumps(panel), encoding="utf-8")
    with pytest.raises(ValueError, match="cutoff"):
        load_panel(path)

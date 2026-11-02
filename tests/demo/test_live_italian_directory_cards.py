"""Response-bound Genoa/Venezia cards survive navigation-label projection."""
from datetime import datetime, timezone

import pytest

from demo.live_sources import LiveNewsTools, article_link_reason, discovery_queries, profile, search_team_binding
from football_agent.news_discovery import NewsDiscoveryTools
from football_spider.storage.raw_archive import RawArchive


AT = datetime(2026, 10, 25, 13, 9, tzinfo=timezone.utc)
GENOA = "https://genoacfc.it/news/"
VENEZIA = "https://www.veneziafc.it/news"
GENOA_ARTICLE = "https://genoacfc.it/2026/10/24/genoa-cfc-i-convocati-per-la-giornata-n-8/"
VENEZIA_ARTICLE = "https://www.veneziafc.it/news/i-convocati-per-genoa-venezia-fc"


def genoa_card(title="Genoa CFC, i convocati per la giornata n. 8", date="24.10.26", url=GENOA_ARTICLE):
    return f'''<article><div class="article-content"><div class="wrap-txt">
    <a href="/category/news/">News</a><h3>{title}</h3><p class="label-small-regular">{date}</p>
    </div><a href="{url}">Continua</a></div></article>'''


def venezia_card(title="I convocati per Genoa - Venezia FC", accessible=None, date="2026-10-24T15:18:27.627Z"):
    accessible = accessible if accessible is not None else title + " - 24/10/2026"
    return f'''<a href="{VENEZIA_ARTICLE}" aria-label="{accessible}">
    <div aria-hidden="true"><img alt="Not a headline"><h3>{title}</h3>
    <time datetime="{date}">24/10/2026</time></div></a>'''


def project(tmp_path, url, html, *, status=200, generic=False):
    from urllib.parse import urlsplit
    archive = RawArchive(tmp_path)
    observation = archive.save("open_page", url, {}, html.encode(), status=status, observed_at=AT)
    cls = NewsDiscoveryTools if generic else LiveNewsTools
    tools = cls(archive, [urlsplit(url).hostname], index_urls=[url], projection_version=13, clock=lambda: AT)
    return tools.project(observation, {"tool": "read_url", "endpoint": url, "params": {}, "source": "open_page"})


def test_genoa_button_binds_to_unique_same_card_heading_and_calendar_date(tmp_path):
    doc = project(tmp_path, GENOA, genoa_card())
    link = next(x for x in doc["links"] if x["url"] == GENOA_ARTICLE)
    assert link["title"] == "Genoa CFC, i convocati per la giornata n. 8"
    assert link["original_observed_link_title"] == "Continua"
    assert link["title_source"] == "same_card_observed_heading"
    assert link["source_publication_date"] == "2026-10-24"
    assert link["source_published_at"] is None  # A day is not an invented UTC timestamp.
    assert link["source_available_at"] is None
    assert article_link_reason(link["url"], link["title"], [GENOA], set()) is None
    category = next(x for x in doc["links"] if "/category/" in x["url"])
    assert article_link_reason(category["url"], category["title"], [GENOA], set()) is not None
    assert doc["official_directory_card_projection"]["target_article_requests"] == 0


def test_venezia_accessible_name_corroborates_inner_heading_and_explicit_timestamp(tmp_path):
    doc = project(tmp_path, VENEZIA, venezia_card())
    link = doc["links"][0]
    assert link["title"] == "I convocati per Genoa - Venezia FC"
    assert link["original_observed_link_title"] == ""
    assert link["title_source"] == "same_anchor_observed_accessible_name"
    assert link["source_published_at"] == "2026-10-24T15:18:27.627000+00:00"
    assert link["source_available_at"] is None
    assert article_link_reason(link["url"], link["title"], [VENEZIA], set()) is None


@pytest.mark.parametrize("html", [
    genoa_card(date="23.10.26"),
    genoa_card().replace("<h3>", "<h3>Other heading</h3><h3>"),
    genoa_card().replace("</div></article>", '<a href="https://genoacfc.it/2026/10/24/second-article/">Continua</a></div></article>'),
    genoa_card().replace("<article>", "<article hidden>"),
    genoa_card(url="https://elsewhere.example/2026/10/24/match-preview/"),
])
def test_genoa_ambiguous_hidden_or_cross_host_card_does_not_supply_title(tmp_path, html):
    doc = project(tmp_path, GENOA, html)
    assert not any(x.get("title_source") == "same_card_observed_heading" for x in doc["links"])


@pytest.mark.parametrize("html", [
    venezia_card(accessible="Unrelated accessible label"),
    venezia_card(date="2026-10-23T15:18:27Z"),
    venezia_card(date="2026-10-24T15:18:27"),
    venezia_card().replace('<a href=', '<a hidden href='),
    venezia_card().replace('<time datetime=', '<time>24/10/2026</time><time datetime='),
    venezia_card().replace('aria-label=', 'data-label='),
])
def test_venezia_unbound_or_inconsistent_metadata_does_not_supply_title(tmp_path, html):
    doc = project(tmp_path, VENEZIA, html)
    assert not any(x.get("title_source") == "same_anchor_observed_accessible_name" for x in doc["links"])


def test_conflicting_versions_of_one_card_do_not_overwrite_title(tmp_path):
    doc = project(tmp_path, GENOA, genoa_card() + genoa_card(title="A conflicting title from another card"))
    assert not any(x.get("title_source") == "same_card_observed_heading" for x in doc["links"])
    assert any(x["reason"] == "conflicting_same_url_card_metadata" for x in doc["official_directory_card_projection"]["rejected"])


def test_unregistered_host_and_failed_response_do_not_gain_card_projection(tmp_path):
    doc = project(tmp_path / "other", "https://example.com/news", venezia_card())
    assert "official_directory_card_projection" not in doc
    doc = project(tmp_path / "failed", VENEZIA, venezia_card(), status=503)
    assert "official_directory_card_projection" not in doc


def test_generic_v13_replay_contract_remains_unchanged(tmp_path):
    genoa = project(tmp_path / "genoa", GENOA, genoa_card(), generic=True)
    assert next(x for x in genoa["links"] if x["url"] == GENOA_ARTICLE)["title"] == "Continua"
    venezia = project(tmp_path / "venezia", VENEZIA, venezia_card(), generic=True)
    assert venezia["links"][0]["title"] == ""


def test_chinese_source_names_resolve_same_scoped_directories_and_native_queries():
    binding = {"canonical": {"home_name": "热那亚", "away_name": "威尼斯", "competition_name": "意甲", "kickoff": "2026-10-25T14:00:00Z"}}
    selected = profile(binding)
    assert selected["seed_urls"] == [GENOA, VENEZIA]
    assert not selected["unknown_teams"]
    queries = discovery_queries(binding["canonical"])
    assert all("热那亚" not in q and "威尼斯" not in q for q in queries)
    assert '"Genoa" "Venezia"' in queries[0]
    assert binding["canonical"]["home_name"] == "热那亚"  # No identity mutation.
    assert search_team_binding("Genoa prepare for the weekend", "https://football-italia.net/preview", binding["canonical"], [])


@pytest.mark.parametrize("title", ["Serie B Fem. 26/27: Venezia FC - Vicenza 0-1",
    "Primavera 26/27: Sudtirol - Venezia FC 1-2", "Il programma delle gare del Settore Giovanile"])
def test_explicit_italian_squad_names_do_not_enter_mens_menu(title):
    assert article_link_reason(VENEZIA_ARTICLE, title, [VENEZIA], set()) == "directory_other_squad_label"


def test_commercial_or_ticket_context_is_not_automatically_assigned_negative_value():
    for title in ["Serie A: la prevendita del Settore Ospiti per Genoa - Venezia FC",
                  "ENEL È OFFICIAL ENERGY PARTNER DEL VENEZIA FC"]:
        assert article_link_reason(VENEZIA_ARTICLE, title, [VENEZIA], set()) is None


@pytest.mark.parametrize("competition", ["English Premier League", "Serie A Women", "Serie A U19"])
def test_chinese_alias_never_bypasses_association_and_squad_scope(competition):
    binding = {"canonical": {"home_name": "热那亚", "away_name": "威尼斯", "competition_name": competition}}
    selected = profile(binding)
    assert GENOA not in selected["seed_urls"] and VENEZIA not in selected["seed_urls"]
    assert selected["unknown_teams"] == ["热那亚", "威尼斯"]


def test_archived_genoa_venezia_responses_restore_six_candidates_without_network(monkeypatch):
    """Local optional regression against the exact two user-reported responses."""
    import hashlib
    import json
    from pathlib import Path
    import requests
    from demo.live_g import _directory_candidate_reason, candidate_time_scope
    from football_spider.models.contracts import Observation

    folder = Path(__file__).resolve().parents[2] / "var/parallel/agent/round2/runs/live-g/runs/lg-b802605ada4a419093a2b5ed1c858cd9"
    if not (folder / "run.json").exists():
        pytest.skip("Original local incident archive is not part of the source distribution")
    before = (folder / "run.json").read_bytes()
    run = json.loads(before)
    monkeypatch.setattr(requests.Session, "request", lambda *a, **kw: pytest.fail("Network forbidden in archived replay"))
    archive = RawArchive(folder / "sources")
    entries = run["source_profile"]["seed_urls"]
    tools = LiveNewsTools(archive, run["source_profile"]["allowed_hosts"], index_urls=entries,
                         projection_version=13, clock=lambda: datetime.now(timezone.utc))
    admitted, seen, restored = [], set(), []
    for attempt in run["attempts"]:
        if attempt["purpose"] != "directory_discovery":
            continue
        observation = Observation.model_validate(attempt["response"]["observation"])
        assert hashlib.sha256(archive.read_body(observation)).hexdigest() == observation.content_hash
        document = tools.project(observation, {"tool": "read_url", "endpoint": attempt["action"]["url"],
                                               "params": {}, "source": "open_page"})
        restored.append(document["official_directory_card_projection"]["cards"])
        for link in document["links"]:
            reason = _directory_candidate_reason(link["url"], link["title"], run["fixture"], entries, seen)
            scope = candidate_time_scope(link, attempt["handoff_at"])
            if reason is None and scope["status"] not in {"stale", "future"}:
                admitted.append(link)
                seen.add(link["url"])
    assert restored == [2, 31]
    assert len(admitted) == 6
    assert sum("genoacfc.it" in x["url"] for x in admitted) == 1
    assert any(x["url"] == VENEZIA_ARTICLE for x in admitted)
    assert not any("Primavera" in x["title"] or "Serie B Fem." in x["title"] or "Settore Giovanile" in x["title"] for x in admitted)
    assert (folder / "run.json").read_bytes() == before

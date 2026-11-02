"""Read-only projection for the football-data.co.uk historical crawl.

The source CSVs contain match results and bookmaker prices, but they do not
provide a trustworthy event-level observation timestamp or a timezone for the
declared local kickoff clock.  This adapter therefore deliberately returns a
historical projection instead of :class:`ResearchMatch` objects.  Callers
must not treat its rows as prospective fixtures or as g trajectories.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Iterator, Mapping


SCHEMA = "footballmind-football-data-co-uk-projection-v1"
COMPETITIONS = {
    "E0": {"name": "Premier League", "country": "england", "group": "europe_top5"},
    "SP1": {"name": "La Liga", "country": "spain", "group": "europe_top5"},
    "I1": {"name": "Serie A", "country": "italy", "group": "europe_top5"},
    "D1": {"name": "Bundesliga", "country": "germany", "group": "europe_top5"},
    "F1": {"name": "Ligue 1", "country": "france", "group": "europe_top5"},
    "CHN": {"name": "Chinese Super League", "country": "china", "group": "asia"},
}


def _sha(value: str, size: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:size]


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return value or "unknown"


def _source_manifest(root: Path) -> dict[tuple[str, str], dict]:
    manifest = json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))
    result = {}
    for item in manifest.get("files", []):
        result[(str(item.get("competition_code")), str(item.get("season_code")))] = dict(item)
    return result


def _parse_int(value, *, field: str, allow_none: bool = True) -> int | None:
    if value in (None, "", "NA", "N/A") and allow_none:
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"negative {field}: {value!r}")
    return parsed


def _outcome(result: str | None) -> int | None:
    return {"H": 0, "D": 1, "A": 2}.get(result)


def _price(value):
    if value in (None, "", "NA", "N/A"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 1 else None


def _date_iso(value: str) -> tuple[str, str]:
    """Return (ISO date, source-declared date string) without guessing locale."""
    raw = str(value or "")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat(), raw
        except ValueError:
            continue
    raise ValueError(f"invalid local date: {raw!r}")


def _match_id(code: str, season: str, date_local: str, home: str, away: str) -> str:
    return f"fd:{code}:{season}:{date_local}:{_slug(home)}:{_slug(away)}"


def iter_rows(root: str | Path) -> Iterator[dict]:
    """Yield normalized source rows after verifying source-file hashes.

    The crawl's normalized JSONL is the only file read here.  Every row keeps
    the raw file digest and source path so a consumer can reproduce it from
    the archived bytes.
    """
    root = Path(root).resolve()
    normalized = root / "normalized-matches.jsonl"
    if not normalized.is_file():
        raise FileNotFoundError(normalized)
    source_by_key = _source_manifest(root)
    verified = set()
    for number, line in enumerate(normalized.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        code, season = str(row.get("competition_code")), str(row.get("season_code"))
        if code not in COMPETITIONS:
            raise ValueError(f"unsupported competition code at row {number}: {code}")
        source = source_by_key.get((code, season))
        if source is None:
            raise ValueError(f"missing source manifest entry at row {number}: {code}/{season}")
        source_path = root.parent.parent.parent.parent / str(source["path"]) if str(source["path"]).startswith("var/") else root / Path(str(source["path"])).name
        # The manifest path is workspace-relative in the archived artifact.
        if not source_path.is_file():
            source_path = root / Path(str(source["path"])).relative_to("var/external/football-data-co-uk-20261024-v2") if str(source["path"]).startswith("var/external/") else source_path
        if not source_path.is_file():
            raise FileNotFoundError(f"raw source not found for row {number}: {source.get('path')}")
        if (code, season) not in verified:
            actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual != source.get("sha256"):
                raise ValueError(f"raw source sha mismatch for {code}/{season}")
            verified.add((code, season))
        # The upstream CSVs contain a small number of entirely blank trailing
        # rows. They remain in the raw/normalized crawl but cannot be a match
        # identity; skip them explicitly rather than manufacturing IDs.
        if not row.get("date_local") and not row.get("home_team") and not row.get("away_team"):
            continue
        home, away = str(row.get("home_team") or ""), str(row.get("away_team") or "")
        if not home or not away or home == away:
            raise ValueError(f"invalid teams at row {number}")
        local_date, source_date_local = _date_iso(row.get("date_local"))
        code_meta = COMPETITIONS[code]
        match_id = _match_id(code, season, local_date, home, away)
        source_path_rel = str(source.get("path"))
        yield {
            "schema": SCHEMA,
            "match_id": match_id,
            "provider": "football-data.co.uk",
            "competition_code": code,
            "competition_name": code_meta["name"],
            "competition_group": code_meta["group"],
            "country": code_meta["country"],
            "season": season,
            "home_team": home,
            "away_team": away,
            "home_team_id": f"fdteam:{_sha(home.casefold())}",
            "away_team_id": f"fdteam:{_sha(away.casefold())}",
            "date_local": local_date,
            "source_date_local": source_date_local,
            "time_local": row.get("time_local"),
            "kickoff_timezone": None,
            "kickoff_at_utc": None,
            "kickoff_time_basis": row.get("kickoff_time_basis", "source_declared_local_clock_timezone_not_normalized"),
            "source_available_at": None,
            "source_observed_at": None,
            "historical_reconstruction": True,
            "home_goals": _parse_int(row.get("home_goals"), field="home_goals"),
            "away_goals": _parse_int(row.get("away_goals"), field="away_goals"),
            "result": row.get("result") if row.get("result") in {"H", "D", "A"} else None,
            "outcome_class": _outcome(row.get("result")),
            "avg_1x2": {side: _price((row.get("avg_1x2") or {}).get(side)) for side in ("home", "draw", "away")},
            "b365_1x2": {side: _price((row.get("b365_1x2") or {}).get(side)) for side in ("home", "draw", "away")},
            "odds_observed_at": None,
            "odds_source_timestamp": None,
            "strict_prequery_eligible": False,
            "identity_basis": "source_composite_code_season_local_date_teams",
            "source_file": source_path_rel,
            "source_url": source.get("url"),
            "source_sha256": source.get("sha256"),
        }


def build_projection(root: str | Path, output: str | Path | None = None) -> dict:
    """Build a read-only project projection without making network requests."""
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "projected"
    source_normalized_rows = sum(1 for line in (root / "normalized-matches.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    rows = list(iter_rows(root))
    rows.sort(key=lambda item: (item["date_local"], item["competition_code"], item["match_id"]))
    matches, labels, odds = [], [], []
    for row in rows:
        matches.append({k: row[k] for k in (
            "schema", "match_id", "provider", "competition_code", "competition_name",
            "competition_group", "country", "season", "home_team", "away_team",
            "home_team_id", "away_team_id", "date_local", "source_date_local", "time_local", "kickoff_timezone",
            "kickoff_at_utc", "kickoff_time_basis", "source_available_at", "source_observed_at",
            "historical_reconstruction", "identity_basis", "source_file", "source_url", "source_sha256")})
        labels.append({k: row[k] for k in (
            "match_id", "home_goals", "away_goals", "result", "outcome_class",
            "provider", "source_file", "source_sha256", "label_available_at") if k in row} |
            {"label_available_at": None, "information_mode": "historical_reconstruction"})
        odds.append({"match_id": row["match_id"], "avg_1x2": row["avg_1x2"], "b365_1x2": row["b365_1x2"],
                     "odds_observed_at": None, "odds_source_timestamp": None,
                     "strict_prequery_eligible": False, "provider": row["provider"],
                     "source_file": row["source_file"], "source_sha256": row["source_sha256"],
                     "information_mode": "historical_reconstruction"})
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in (("matches.jsonl", matches), ("labels-results.jsonl", labels), ("market-snapshots.jsonl", odds)):
        path = output / filename
        path.write_text("".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values), encoding="utf-8")
    counts = Counter(row["competition_code"] for row in rows)
    report = {
        "schema": "footballmind-football-data-co-uk-projection-report-v1",
        "root": str(root),
        "output": str(output),
        "rows": len(rows),
        "source_normalized_rows": source_normalized_rows,
        "skipped_blank_rows": source_normalized_rows - len(rows),
        "matches": len(matches),
        "labels": sum(row["outcome_class"] is not None for row in rows),
        "complete_avg_1x2": sum(all(value is not None for value in row["avg_1x2"].values()) for row in rows),
        "complete_b365_1x2": sum(all(value is not None for value in row["b365_1x2"].values()) for row in rows),
        "rows_by_competition_code": dict(sorted(counts.items())),
        "strict_prequery_eligible": 0,
        "network_requests": 0,
        "g_training_rows": 0,
        "limitations": [
            "kickoff_at_utc is null because source timezone is not declared",
            "source_available_at and odds_observed_at are null",
            "historical reconstruction only; no prospective Agent trajectory",
            "CHN is Chinese Super League only; China League One is not present",
        ],
    }
    (output / "projection-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# football-data.co.uk 历史投影\n\n"
        "这是只读历史研究输入，不是实时赛程接口，不是 prospective Agent 轨迹，也不产生 g 标签。\n\n"
        "`kickoff_at_utc`、`source_available_at` 与 `odds_observed_at` 保持 null；来源 CSV 没有足以证明这些字段的时间信息。\n"
        "CHN 仅代表中超，不包含中甲。原始 CSV 与 source-manifest 位于上级 crawl 目录。\n",
        encoding="utf-8")
    manifest_files = []
    for path in sorted(output.iterdir()):
        if path.name == "projection-manifest.json" or not path.is_file():
            continue
        manifest_files.append({"path": path.name, "bytes": path.stat().st_size,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"schema": "footballmind-football-data-co-uk-projection-manifest-v1",
                "source_root": str(root), "network_requests": 0,
                "historical_only": True, "files": manifest_files}
    (output / "projection-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["SCHEMA", "COMPETITIONS", "iter_rows", "build_projection"]

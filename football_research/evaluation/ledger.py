"""Append-only forecast and result evidence, with explicit replay provenance.

Content hashes detect changed bytes, not a malicious administrator with access to
all files. Prospective issuance is a local process attestation, not a third-party
timestamp. A caller-supplied prediction is always registered as a replay.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping

from football_research.baselines.api import Prediction, canonical_json, content_hash, utc
from .metrics import per_prediction_losses, reliability_table, score_by_lead_time, score_predictions

LEDGER_VERSION = "footballmind.forecast-ledger.v1"
MODES = {"operational", "historical_reconstruction"}
RESULT_STATUSES = {"final", "scheduled", "postponed", "cancelled", "disputed"}


def _now():
    return datetime.now(timezone.utc)


def _json(value):
    return json.loads(canonical_json(value))


@contextmanager
def _lock(path: Path, timeout=15):
    """OS-released interprocess lock, including Windows and concurrent threads."""
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        end = time.monotonic() + timeout
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= end:
                    raise TimeoutError("Forecast ledger is busy")
                time.sleep(.01)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_new(path: Path, record: dict):
    """Called under the ledger lock; committed files are never overwritten."""
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".uncommitted-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path: Path):
    record = json.loads(path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in record.items() if k != "content_sha256"}
    if record.get("schema") != LEDGER_VERSION or content_hash(payload) != record.get("content_sha256"):
        raise ValueError(f"Ledger integrity check failed: {path.name}")
    expected_id = record.get("forecast_id", record.get("label_version"))
    if expected_id != path.stem:
        raise ValueError("Ledger filename identity mismatch")
    return record


def _sealed(payload):
    payload = _json(payload)
    return {**payload, "content_sha256": content_hash(payload)}


class ForecastLedger:
    def __init__(self, path):
        self.path = Path(path)
        for name in ("forecasts", "results"):
            (self.path / name).mkdir(parents=True, exist_ok=True)

    def _records(self, folder):
        return [_read(path) for path in sorted((self.path / folder).glob("*.json"))]

    def forecasts(self):
        with _lock(self.path / ".lock"):
            return self._records("forecasts")

    def results(self):
        with _lock(self.path / ".lock"):
            return self._records("results")

    def import_forecast(self, prediction, *, kickoff, source_reference=None):
        """Import is always replay, even if an input claims an earlier issued_at."""
        raw = prediction.model_dump(mode="json") if hasattr(prediction, "model_dump") else dict(prediction)
        claimed = raw.get("issued_at")
        if claimed is not None and utc(claimed) > _now():
            raise ValueError("Externally claimed issued_at is in the future")
        return self._register(raw, kickoff=kickoff, registration_kind="replay", issued_at=None,
                              generation_started_at=None, max_latency_seconds=None,
                              source_reference=source_reference, claimed_issued_at=claimed)

    def issue_forecast(self, service, fixture: Mapping, *, prediction_cutoff=None,
                       features=None, max_latency_seconds=300):
        """Generate and freeze in this process; the clock is never caller supplied.

        A bounded latency permits an explicitly frozen cutoff slightly before
        actual computation. Both generation and registration must finish before
        kickoff. Historical reconstruction remains a separate information mode.
        """
        if not math.isfinite(max_latency_seconds) or max_latency_seconds <= 0:
            raise ValueError("max_latency_seconds must be finite and positive")
        started = _now()
        cutoff = started if prediction_cutoff is None else utc(prediction_cutoff)
        if cutoff > started or (started - cutoff).total_seconds() > max_latency_seconds:
            raise ValueError("Prospective cutoff must precede generation within the latency bound")
        if started >= utc(fixture["kickoff"]):
            raise ValueError("Prospective generation must precede kickoff")
        prediction = service.predict(fixture, prediction_cutoff=cutoff, features=features)
        issued = _now()
        if issued < started or issued >= utc(fixture["kickoff"]) or (issued - cutoff).total_seconds() > max_latency_seconds:
            raise ValueError("Prospective generation exceeded its prematch time window")
        raw = prediction.model_dump(mode="json")
        if raw["match_id"] != fixture["match_id"] or utc(raw["prediction_cutoff"]) != cutoff:
            raise ValueError("Service returned a different match or cutoff")
        return self._register(raw, kickoff=fixture["kickoff"], registration_kind="prospective",
                              issued_at=issued, generation_started_at=started,
                              max_latency_seconds=max_latency_seconds,
                              source_reference="in_process_PredictionService.predict", claimed_issued_at=None)

    def _register(self, raw, *, kickoff, registration_kind, issued_at, generation_started_at,
                  max_latency_seconds, source_reference, claimed_issued_at):
        # Transport enrichments (team names, model alias) do not alter the public contract.
        fields = Prediction.model_fields
        prediction = Prediction.model_validate({k: v for k, v in raw.items() if k in fields}).model_dump(mode="json")
        for field in ("model_version", "dataset_version"):
            if not prediction[field]:
                raise ValueError(f"{field} is required")
        if not prediction["input_manifest"]:
            raise ValueError("Prediction input_manifest is required")
        cutoff, kickoff = utc(prediction["prediction_cutoff"]), utc(kickoff)
        if cutoff >= kickoff:
            raise ValueError("Forecast cutoff must precede kickoff")
        prediction["prediction_cutoff"] = cutoff.isoformat()
        slot = {key: prediction[key] for key in ("match_id", "prediction_cutoff", "model_version", "dataset_version", "information_mode")}
        slot["registration_kind"] = registration_kind
        identity = content_hash(slot)
        immutable_input = {"prediction": prediction, "kickoff": kickoff.isoformat()}
        path = self.path / "forecasts" / f"{identity}.json"
        with _lock(self.path / ".lock"):
            if path.exists():
                existing = _read(path)
                if existing["input_sha256"] != content_hash(immutable_input):
                    raise ValueError("Forecast slot conflict: frozen probabilities, manifest or kickoff changed")
                return {"status": "duplicate", "forecast_id": identity, "record": existing}
            registered = _now()
            if cutoff > registered:
                raise ValueError("Forecast cutoff is in the future")
            if registration_kind == "prospective" and (registered >= kickoff or registered < issued_at or (registered - cutoff).total_seconds() > max_latency_seconds):
                raise ValueError("Prospective registration must finish before kickoff with a monotonic wall clock")
            record = _sealed({"schema": LEDGER_VERSION, "forecast_id": identity, **immutable_input,
                              "input_sha256": content_hash(immutable_input), "registration_kind": registration_kind,
                              "issued_at": issued_at, "generation_started_at": generation_started_at,
                              "registered_at": registered, "max_latency_seconds": max_latency_seconds,
                              "claimed_issued_at_unverified": claimed_issued_at,
                              "source_reference": source_reference,
                              "issuance_evidence": "local_clock_wrapped_predict_call" if registration_kind == "prospective" else "import_has_no_generation_attestation"})
            _write_new(path, record)
            return {"status": "registered", "forecast_id": identity, "record": record}

    def settle_result(self, result: Mapping):
        """Append one immutable label version. Corrections name superseded hashes."""
        return self.settle_results([result])[0]

    def settle_results(self, results):
        """Append a batch under one lock and one integrity-checked snapshot read.

        Commits occur per row. On a failed row, the successfully committed prefix
        remains immutable; retrying the same/corrected batch deduplicates that
        prefix. Later rows can supersede versions committed earlier in the batch.
        This intentionally provides resumability, not all-or-nothing rollback.
        """
        with _lock(self.path / ".lock"):
            forecasts, existing = self._records("forecasts"), self._records("results")
            index = {"kickoffs": defaultdict(set), "versions": {}, "revisions": {},
                     "first_available": {}, "first_final_available": {}}
            for forecast in forecasts:
                prediction = forecast["prediction"]
                index["kickoffs"][prediction["match_id"], prediction["information_mode"]].add(forecast["kickoff"])
            for record in existing:
                self._index_result(index, record)
            return [self._settle_result_locked(result, index) for result in results]

    @staticmethod
    def _index_result(index, record):
        row = record["result"]
        key = row["match_id"], row["information_mode"]
        available = utc(row["available_at"])
        index["versions"][record["label_version"]] = record
        index["revisions"][(*key, row["source"], row["revision_id"])] = record["label_version"]
        index["first_available"][key] = min(available, index["first_available"].get(key, available))
        if row["status"] == "final":
            index["first_final_available"][key] = min(available, index["first_final_available"].get(key, available))

    def _settle_result_locked(self, result, index):
        row = dict(result)
        for field in ("match_id", "source", "source_reference", "revision_id", "information_mode", "status", "available_at", "observed_at"):
            if not row.get(field):
                raise ValueError(f"Result requires {field}")
        for field in ("match_id", "source", "source_reference", "revision_id", "information_mode", "status"):
            if not isinstance(row[field], str):
                raise ValueError(f"Result {field} must be a string")
        if row["information_mode"] not in MODES or row["status"] not in RESULT_STATUSES:
            raise ValueError("Unknown information mode or result status")
        if row.get("score_period") != "regulation":
            raise ValueError("Result requires regulation-time semantics")
        now = _now()
        available, observed = utc(row["available_at"]), utc(row["observed_at"])
        # Legacy records retain their exact payload/hash and historical semantics.
        # New contracts preserve receipt and commit separately; operational
        # availability is a lower bound on both, not an upper bound on receipt.
        explicit_clocks = row.get('time_contract_version') == 'separate-clocks-v2' or any(
            row.get(name) is not None for name in ('ingested_at', 'source_published_at', 'operational_available_at'))
        ingested = utc(row['ingested_at']) if row.get('ingested_at') else None
        published = utc(row['source_published_at']) if row.get('source_published_at') else None
        operational_available = utc(row['operational_available_at']) if row.get('operational_available_at') else max(observed, ingested or observed)
        if max(available, observed, ingested or observed, operational_available) > now:
            raise ValueError('Result timestamps must not be after current clock')
        if published and published > observed:
            raise ValueError('Source publication must not follow actual observation')
        if explicit_clocks:
            if operational_available < max(observed, ingested or observed):
                raise ValueError('Operational availability precedes observation or ingestion')
            if row['information_mode'] == 'operational' and available < operational_available:
                raise ValueError('Operational result availability precedes evidence')
            for name, stamp in (('ingested_at', ingested), ('source_published_at', published)):
                if stamp is not None:
                    row[name] = stamp.isoformat()
            if row.get('operational_available_at') is not None:
                row['operational_available_at'] = operational_available.isoformat()
        elif available > observed:
            raise ValueError('Legacy result requires available_at <= observed_at; later ingestion needs separate clocks')
        row.update(available_at=available.isoformat(), observed_at=observed.isoformat())
        if row["status"] == "final":
            for name in ("home_goals", "away_goals"):
                if type(row.get(name)) is not int or row[name] < 0:
                    raise ValueError("Final regulation scores must be nonnegative integers")
        elif row.get("home_goals") is not None or row.get("away_goals") is not None:
            raise ValueError("Non-final results must not carry score labels")
        supersedes = row.get("supersedes", [])
        if not isinstance(supersedes, (list, tuple)) or any(not isinstance(x, str) for x in supersedes):
            raise ValueError("supersedes must contain label-version hashes")
        row["supersedes"] = sorted(set(supersedes))
        key = row["match_id"], row["information_mode"]
        known_kickoffs = index["kickoffs"][key]
        if row.get("kickoff") is None:
            if len(known_kickoffs) != 1:
                raise ValueError("Result needs explicit kickoff when no unique forecast schedule exists")
            row["kickoff"] = next(iter(known_kickoffs))
        row["kickoff"] = utc(row["kickoff"]).isoformat()
        if row["status"] == "final" and available < utc(row["kickoff"]):
            raise ValueError("Final result is available before kickoff")
        if row["information_mode"] == "historical_reconstruction" and not row.get("availability_assumption"):
            raise ValueError("Reconstruction labels require availability_assumption")
        versions = index["versions"]
        for version in row["supersedes"]:
            previous = versions.get(version, {}).get("result")
            if previous is None or any(previous[k] != row[k] for k in ("match_id", "information_mode")):
                raise ValueError("Superseded result must exist for the same match and information mode")
            if utc(previous["available_at"]) > available or utc(previous["observed_at"]) > observed:
                raise ValueError("A revision must not predate the result it supersedes")
        row = _json(row)
        version = content_hash(row)
        if version in versions:
            return {"status": "duplicate", "label_version": version, "record": versions[version]}
        if (*key, row["source"], row["revision_id"]) in index["revisions"]:
            raise ValueError("Source revision identity reused with different content")
        first_available = min(available, index["first_available"].get(key, available))
        first_final = index["first_final_available"].get(key)
        if row["status"] == "final":
            first_final = min(available, first_final) if first_final is not None else available
        recorded = _now()
        if recorded < max(observed, available, ingested or observed, operational_available) or recorded < now:
            raise ValueError("Result recording requires a monotonic wall clock")
        record = _sealed({"schema": LEDGER_VERSION, "label_version": version, "result": row,
                          "recorded_at": recorded, "first_available_at": first_available,
                          "first_final_label_available_at": first_final})
        _write_new(self.path / "results" / f"{version}.json", record)
        self._index_result(index, record)
        return {"status": "appended", "label_version": version, "record": record}

    def score(self, *, as_of=None, result_grace_hours=3):
        """Score label versions actually in this ledger by as_of; never mutate it."""
        now = _now()
        at = now if as_of is None else utc(as_of)
        if at > now or not math.isfinite(result_grace_hours) or result_grace_hours < 0:
            raise ValueError("Invalid score as_of or result grace interval")
        with _lock(self.path / ".lock"):
            forecasts, versions = self._records("forecasts"), self._records("results")
        forecasts = [r for r in forecasts if utc(r["registered_at"]) <= at]
        versions = [r for r in versions if utc(r['recorded_at']) <= at and all(
            utc(r['result'][key]) <= at for key in
            ('observed_at', 'available_at', 'ingested_at', 'operational_available_at', 'source_published_at')
            if r['result'].get(key) is not None)]
        results_by_match = defaultdict(list)
        for version in versions:
            row = version["result"]
            results_by_match[row["match_id"], row["information_mode"]].append(version)
        groups, items = defaultdict(list), []
        for forecast in forecasts:
            prediction = forecast["prediction"]
            group = (forecast["registration_kind"], prediction["information_mode"], prediction["model_version"])
            groups[group]  # include empty-score models in reports
            candidates = results_by_match[prediction["match_id"], prediction["information_mode"]]
            superseded = {v for r in candidates for v in r["result"]["supersedes"]}
            active = [r for r in candidates if r["label_version"] not in superseded]
            item = {"forecast_id": forecast["forecast_id"], "forecast_content_sha256": forecast["content_sha256"],
                    "match_id": prediction["match_id"], "prediction_cutoff": prediction["prediction_cutoff"],
                    "kickoff": forecast["kickoff"], "issued_at": forecast["issued_at"],
                    "registration_kind": group[0], "information_mode": group[1], "model_version": group[2],
                    "label_versions": sorted(r["label_version"] for r in active)}
            item["cutoff_lead_hours"] = (utc(forecast["kickoff"]) - utc(prediction["prediction_cutoff"])).total_seconds() / 3600
            item["issued_lead_hours"] = ((utc(forecast["kickoff"]) - utc(forecast["issued_at"])).total_seconds() / 3600
                                         if forecast["issued_at"] else None)
            semantics = {tuple(r["result"].get(k) for k in ("status", "home_goals", "away_goals", "score_period", "kickoff")) for r in active}
            if not active:
                item["status"] = "pending" if at < utc(forecast["kickoff"]) + timedelta(hours=result_grace_hours) else "no_label"
                item["label_due_at"] = (utc(forecast["kickoff"]) + timedelta(hours=result_grace_hours)).isoformat()
                item["reason"] = ("fixture_not_started" if at < utc(forecast["kickoff"]) else
                                  "awaiting_regulation_result" if item["status"] == "pending" else "result_overdue_or_missing")
            elif len(semantics) > 1:
                item["status"] = "conflict"
            else:
                selected = max(active, key=lambda r: (r["result"]["observed_at"], r["result"]["available_at"], r["label_version"]))
                row = selected["result"]
                item.update(label_version=selected["label_version"], result_available_at=row["available_at"], source=row["source"], first_available_at=min(r["first_available_at"] for r in candidates))
                final_times = [r["result"]["available_at"] for r in candidates if r["result"]["status"] == "final"]
                item["first_final_label_available_at"] = min(final_times) if final_times else None
                if row["status"] != "final":
                    item["status"] = ("pending" if at < utc(row["kickoff"]) + timedelta(hours=result_grace_hours) else "no_label") if row["status"] == "scheduled" else row["status"]
                elif row["kickoff"] != forecast["kickoff"]:
                    item["status"] = "schedule_changed"
                else:
                    scored = {**prediction, "kickoff": forecast["kickoff"], "home_goals": row["home_goals"], "away_goals": row["away_goals"]}
                    groups[group].append(scored)
                    item.update(status="scored", home_goals=row["home_goals"], away_goals=row["away_goals"], **per_prediction_losses(scored))
            items.append(item)
        reports = [{"registration_kind": k[0], "information_mode": k[1], "model_version": k[2],
                    **score_predictions(rows), "by_lead_time": score_by_lead_time(rows), "calibration": reliability_table(rows)}
                   for k, rows in sorted(groups.items())]
        return {"schema": LEDGER_VERSION, "as_of": at.isoformat(), "n_forecasts": len(forecasts),
                "n_result_versions": len(versions), "status_counts": dict(sorted(Counter(r["status"] for r in items).items())),
                "scoring_scope": "registration_kind and information_mode remain separate; no pooled score",
                "result_policy": "active explicitly superseded versions; unresolved differences excluded",
                "groups": reports, "forecasts": items,
                "forecast_snapshot_sha256": content_hash(sorted((r["forecast_id"], r["content_sha256"]) for r in forecasts)),
                "result_snapshot_sha256": content_hash(sorted((r["label_version"], r["content_sha256"]) for r in versions))}

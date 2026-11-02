"""Finite learned text responses, conditioned on the actual acquisition history.

The model defines an ordered acquisition process, not an order-invariant joint
distribution of every page on the internet.  Its likelihood at step t is
q(z_t | Y, original_p0, action_t, observed_history_t).  A single-response log
cannot support a later history.  All words/clusters come from eligible training
text; no football-topic importance or hand-written injury value is supplied.

Operational failures, missing data and navigation pages have a shared marginal
mass across outcome classes.  They consume an attempt and enter history, but
cannot move match probabilities.  Unseen responses also enter history, marking
it unsupported rather than silently returning to the empty-history model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .expected_voi import evaluate_candidate, unknown_candidate
from .state import parse_time

VERSION = "ordered-text-return-kernel-v1"
ENCODER_VERSION = "training-only-tfidf-kmeans-v1"
STATE_VERSION = "ordered-text-return-state-v1"
ORIGINS = {"real_observational", "constructed_mechanism"}
NEUTRAL = {"failure", "no_data", "index", "discovery"}
_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be nonempty text")
    return value.strip()


def _p(values):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("Three outcome probabilities required")
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Strictly positive finite probabilities required")
    if abs(sum(values)-1.) > 1e-9:
        raise ValueError("Probabilities must sum to one")
    return list(map(float, values))


def _action(action):
    if not isinstance(action, dict):
        raise ValueError("Action mapping required")
    tool = action.get("tool")
    if tool in {"read_url", "open_page"}:
        url = _text(action.get("url"), "Action URL")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("A public HTTP(S) action URL without credentials is required")
        request = unquote(parsed.path + " " + parsed.query).replace("-", " ").replace("_", " ")
        if not _TOKEN.findall(request):
            request = "root_page"
        return {"tool": tool, "url": url}, tool+":"+parsed.hostname.lower(), request
    if tool == "search_web":
        query = _text(action.get("query"), "Query")
        source = action.get("source", "open_web")
        return {"tool": tool, "query": query, "source": source}, tool+":"+_text(source, "Search source"), query
    raise ValueError("Unsupported text acquisition tool")


def _response(response):
    if not isinstance(response, dict):
        raise ValueError("Response mapping required")
    status, kind, text = response.get("status"), response.get("kind"), response.get("text", "")
    if not isinstance(text, str):
        raise ValueError("Response text must be a string")
    if status in {"FAILED", "FAILURE", "TIMEOUT", "PROVIDER_ERROR", "HTTP_ERROR", "NETWORK_ERROR"}:
        return "failure", ""
    if status == "NO_DATA":
        return "no_data", ""
    if status != "OK":
        raise ValueError("Unknown response status")
    if kind in {"index", "discovery"}:
        return kind, ""
    if kind != "article" or not text.strip():
        return "unsupported", ""
    return "article", text


class _Codebook:
    """Training uses sklearn; inference uses the exact saved vocabulary/IDF."""

    def __init__(self, payload=None):
        self.payload = deepcopy(payload) if payload else None

    def fit(self, texts, clusters):
        unique = sorted(set(texts))
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=2048,
                                   token_pattern=_TOKEN.pattern)
        matrix = vectorizer.fit_transform(unique).toarray()
        distinct = np.unique(matrix, axis=0)
        n = min(clusters, len(distinct))
        if n == 1:
            centers = distinct.mean(axis=0, keepdims=True)
        else:
            centers = KMeans(n_clusters=n, random_state=0, n_init=5).fit(distinct).cluster_centers_
        self.payload = {"version": ENCODER_VERSION,
                        "vocabulary": vectorizer.get_feature_names_out().tolist(),
                        "idf": vectorizer.idf_.tolist(), "centers": centers.tolist()}
        return self

    def encode(self, text):
        if self.payload is None:
            return None, None
        counts = Counter(_TOKEN.findall(text.lower()))
        vocabulary = self.payload["vocabulary"]
        vector = np.asarray([(1.+math.log(counts[word])) if counts[word] else 0. for word in vocabulary])
        vector *= np.asarray(self.payload["idf"])
        norm = float(np.linalg.norm(vector))
        if norm == 0.:
            return None, None
        vector /= norm
        centers = np.asarray(self.payload["centers"])
        index = int(np.argmin(np.sum((centers-vector)**2, axis=1)))
        return str(index), vector


class TextReturnModel:
    version = VERSION
    state_version = STATE_VERSION

    def __init__(self):
        self.metadata, self.parameters, self.records = {}, {}, []
        self.action_codebooks, self.text_codebook = {}, _Codebook()
        self.model_version = ""
        self._contexts, self._kernel_cache = None, {}

    def fit(self, records, *, as_of, dataset_version, evidence_origin="real_observational",
            bandwidth=.2, dirichlet_mass=1., min_matches=12, min_class_matches=1,
            text_clusters=8, action_clusters=8, min_action_similarity=.15):
        if evidence_origin not in ORIGINS:
            raise ValueError("Explicit real_observational or constructed_mechanism origin required")
        if not isinstance(records, (list, tuple)) or not records:
            raise ValueError("Independent settled outcomes and eligible response records are required")
        for value in (bandwidth, dirichlet_mass, min_matches, min_class_matches):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Positive finite kernel, smoothing and support parameters required")
        if not 0 < min_action_similarity <= 1:
            raise ValueError("Positive action similarity threshold required")
        if any(type(v) is not int or not 1 <= v <= 32 for v in (text_clusters, action_clusters)):
            raise ValueError("Cluster counts must be integers between one and 32")
        as_of = parse_time(as_of)
        validated, texts, requests, outcomes = [], [], defaultdict(list), {}
        for item in records:
            match_id = _text(item.get("match_id"), "Training match ID")
            label = item.get("outcome")
            if type(label) is not int or label not in (0, 1, 2):
                raise ValueError("An independently settled H/D/A outcome is required; null labels cannot train")
            cutoff = parse_time(item["prediction_cutoff"])
            result_at = parse_time(item["result_available_at"])
            if not cutoff < result_at <= as_of:
                raise ValueError("Settlement must follow prediction cutoff and precede training cutoff")
            for key in ("available_at", "observed_at", "decision_at"):
                if item.get(key) is not None and parse_time(item[key]) > cutoff:
                    raise ValueError("Late text cannot train a pre-match response model")
            if match_id in outcomes and outcomes[match_id] != label:
                raise ValueError("Conflicting independent outcomes for one match")
            outcomes[match_id] = label
            history = item.get("history", [])
            if not isinstance(history, list) or len(history) > 32:
                raise ValueError("A finite ordered observed history is required")
            steps = history + [{"action": item["action"], "response": item["response"]}]
            for step in steps:
                _, channel, request = _action(step["action"])
                requests[channel].append(request)
                kind, text = _response(step["response"])
                if kind == "unsupported":
                    raise ValueError("Unclassified response cannot be treated as training article text")
                if kind == "article":
                    texts.append(text)
            validated.append({"match_id": match_id, "outcome": label,
                              "prior_probabilities": _p(item["prior_probabilities"]),
                              "action": deepcopy(item["action"]), "response": deepcopy(item["response"]),
                              "history": deepcopy(history), "result_available_at": result_at.isoformat(),
                              "prediction_cutoff": cutoff.isoformat()})
        self.parameters = {"bandwidth": float(bandwidth), "dirichlet_mass": float(dirichlet_mass),
                           "min_matches": float(min_matches), "min_class_matches": float(min_class_matches),
                           "text_clusters": text_clusters, "action_clusters": action_clusters,
                           "min_action_similarity": float(min_action_similarity)}
        self.action_codebooks = {key: _Codebook().fit(value, action_clusters) for key, value in sorted(requests.items())}
        self.text_codebook = _Codebook().fit(texts, text_clusters) if texts else _Codebook()
        stored, seen = [], set()
        for item in validated:
            action, channel, token, vector = self._encode_action(item["action"])
            response_token = self._encode_response(item["response"])
            history = [self._history_entry(step["action"], step["response"]) for step in item["history"]]
            if token is None or response_token is None or any(x["unsupported"] for x in history):
                raise ValueError("Training vocabulary failed to represent an observed training response")
            record = {"match_id": item["match_id"], "outcome": item["outcome"],
                      "prior_probabilities": item["prior_probabilities"], "channel": channel,
                      "action_token": token, "action_vector": vector.tolist(),
                      "history_key": self._history_key(history), "response_token": response_token,
                      "action_digest": _digest(action), "response_digest": _digest(item["response"]),
                      "result_available_at": item["result_available_at"], "prediction_cutoff": item["prediction_cutoff"]}
            identity = _digest(record)
            if identity not in seen:
                stored.append(record)
                seen.add(identity)
        self.records = stored
        self.metadata = {"training_as_of": as_of.isoformat(), "trained_as_of": as_of.isoformat(),
                         "dataset_version": _text(dataset_version, "Dataset version"),
                         "training_match_ids": sorted(outcomes), "training_records": len(stored),
                         "training_matches": len(outcomes), "evidence_origin": evidence_origin,
                         "encoder_version": ENCODER_VERSION, "production_ready": False,
                         "calibration_status": "not_established",
                         "label_contract": "independent_settlement_sidecar_validated_by_dataset_gateway",
                         "joint_scope": "ordered_acquisition_process_conditioned_on_declared_actions",
                         "order_invariance_established": False,
                         "history_scope": "exact_ordered_action_channel_cluster_and_response_tokens",
                         "support_scope": "unsmoothed_match_capped_past_conditional_coverage",
                         "unknown_policy": "retain_attempt_and_probability_then_refuse_unmatched_history",
                         "neutral_assumption": "failure_no_data_index_discovery_independent_of_Y_given_history_action",
                         "selection_scope": "observed_generated_menu_support_not_open_world_identification",
                         "text_scope": "training_only_bag_of_words_clusters_not_verified_semantic_truth",
                         "guarantee": "internal_Bayesian_coherence_not_calibration_or_positive_realized_gain"}
        self.model_version = self.version+":"+_digest(self._payload())[:24]
        self._contexts, self._kernel_cache = None, {}
        return self

    def _encode_action(self, action):
        clean, channel, request = _action(action)
        book = self.action_codebooks.get(channel)
        token, vector = book.encode(request) if book else (None, None)
        return clean, channel, token, vector

    def _encode_response(self, response):
        kind, text = _response(response)
        if kind in NEUTRAL:
            return kind
        if kind == "article":
            token, _ = self.text_codebook.encode(text)
            return "text:"+token if token is not None else None
        return None

    def _history_entry(self, action, response):
        clean, channel, token, _ = self._encode_action(action)
        response_token = self._encode_response(response)
        return {"action": clean, "channel": channel, "action_token": token,
                "response_token": response_token, "response_digest": _digest(response),
                "unsupported": token is None or response_token is None}

    @staticmethod
    def _history_key(history):
        return [[x["channel"], x["action_token"], x["response_token"]] for x in history]

    def start(self, prior_probabilities, match_id, prediction_cutoff):
        if not self.model_version:
            raise ValueError("Fit or load a frozen text response model first")
        state = {"schema": self.state_version, "model_version": self.model_version,
                 "match_id": _text(match_id, "Match ID"),
                 "prediction_cutoff": parse_time(prediction_cutoff).isoformat(),
                 "prior_probabilities": _p(prior_probabilities),
                 "current_probabilities": _p(prior_probabilities), "history": []}
        self._state(state)
        return state

    def _state(self, state):
        required = {"schema", "model_version", "match_id", "prediction_cutoff",
                    "prior_probabilities", "current_probabilities", "history"}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("Exact public text-state fields are required")
        if state["schema"] != self.state_version or state["model_version"] != self.model_version:
            raise ValueError("Text-state/model version mismatch")
        if state["match_id"] in self.metadata["training_match_ids"]:
            raise ValueError("Runtime state cannot reuse a training match")
        if parse_time(state["prediction_cutoff"]) < parse_time(self.metadata["training_as_of"]):
            raise ValueError("Text model was trained after this prediction cutoff")
        _p(state["prior_probabilities"])
        _p(state["current_probabilities"])
        if not isinstance(state["history"], list) or len(state["history"]) > 32:
            raise ValueError("Invalid ordered acquisition history")
        for entry in state["history"]:
            required_entry = {"action", "channel", "action_token", "response_token", "response_digest", "unsupported"}
            if not isinstance(entry, dict) or set(entry) != required_entry or type(entry["unsupported"]) is not bool:
                raise ValueError("Invalid acquired response entry")
            _, channel, token, _ = self._encode_action(entry["action"])
            if (channel, token) != (entry["channel"], entry["action_token"]):
                raise ValueError("History action differs from its frozen encoding")
        # A changed probability vector is not itself new evidence. Reconstruct
        # it from the frozen p0 and every admitted ordered return, including the
        # point at which an unsupported response stopped numerical updating.
        replay = {**state, "history": [], "current_probabilities": list(state["prior_probabilities"])}
        for entry in state["history"]:
            if not entry["unsupported"]:
                q, _, _ = self._kernel(replay, entry["action"])
                if q is None or entry["response_token"] not in q:
                    raise ValueError("History claims an update without conditional response support")
                joint = np.asarray(replay["current_probabilities"])*q[entry["response_token"]]
                replay["current_probabilities"] = (joint/float(joint.sum())).tolist()
            replay["history"].append(entry)
        if max(abs(p-q) for p, q in zip(replay["current_probabilities"], state["current_probabilities"])) > 1e-10:
            raise ValueError("Current probabilities disagree with frozen observed history")
        return deepcopy(state)

    def predict(self, state):
        clean = self._state(state)
        return {"probabilities": clean["current_probabilities"], "model_version": self.model_version,
                "history_length": len(clean["history"]),
                "unsupported_observation": any(x["unsupported"] for x in clean["history"])}

    def _kernel(self, state, action):
        clean, _, _ = _action(action)
        key = _digest([state["prior_probabilities"],
                       [[entry["channel"], entry["action_token"], entry["response_token"], entry["unsupported"]]
                        for entry in state["history"]], clean])
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self._kernel_uncached(state, action)
        return self._kernel_cache[key]

    def _history_neighbors(self, channel, key):
        """Exact ordered context in v1; alternatives require an explicit model version."""
        if self._contexts is None:
            self._contexts = defaultdict(list)
            for row in self.records:
                self._contexts[(row["channel"], _canonical(row["history_key"]))].append(row)
        return [(row, 1.) for row in self._contexts.get((channel, _canonical(key)), [])]

    def _history_support(self, observations):
        """Additional model-specific support gate; v1 changes no support semantics."""
        return {}, None

    def _kernel_uncached(self, state, action, *, request_vector=None):
        clean, channel, token, vector = self._encode_action(action)
        if request_vector is not None:
            vector = request_vector
        if any(x["unsupported"] for x in state["history"]):
            return None, "unsupported_observed_history", {}
        if channel not in self.action_codebooks:
            return None, "unseen_action_source", {}
        if token is None:
            return None, "out_of_vocabulary_action", {}
        key = self._history_key(state["history"])
        matches = self._history_neighbors(channel, key)
        if not matches:
            return None, "no_matching_acquisition_history", {}
        candidates, history_observations = [], []
        for row, history_weight in matches:
            similarity = float(np.dot(vector, np.asarray(row["action_vector"])))
            if similarity+1e-12 < self.parameters["min_action_similarity"]:
                continue
            delta = np.asarray(row["prior_probabilities"])-np.asarray(state["prior_probabilities"])
            weight = similarity*math.exp(-float(np.dot(delta, delta))/(2*self.parameters["bandwidth"]**2))*history_weight
            if weight > 0:
                candidates.append((row, weight))
                history_observations.append((row["match_id"], history_weight))
        if not candidates:
            return None, "insufficient_action_similarity", {}
        rows_per_match = Counter(row["match_id"] for row, _ in candidates)
        weighted = [(row, weight/rows_per_match[row["match_id"]]) for row, weight in candidates]
        match_weights, class_weights = defaultdict(float), [0., 0., 0.]
        for row, weight in weighted:
            match_weights[row["match_id"]] += weight
            class_weights[row["outcome"]] += weight
        mass = math.fsum(match_weights.values())
        square_mass = math.fsum(w*w for w in match_weights.values())
        effective = mass*mass/square_mass if square_mass else 0.
        support = {"matching_matches": len(match_weights), "effective_matches": effective,
                   "class_weight": class_weights, "max_match_weight": max(match_weights.values()),
                   "smoothing_counts_as_support": False, "history_length": len(key)}
        extra, history_reason = self._history_support(history_observations)
        support.update(extra)
        if history_reason is not None:
            return None, history_reason, support
        if effective+1e-9 < self.parameters["min_matches"] or min(class_weights)+1e-9 < self.parameters["min_class_matches"]:
            return None, "insufficient_conditional_support", support
        neutral = defaultdict(float)
        semantic = defaultdict(lambda: np.zeros(3))
        for row, weight in weighted:
            category = row["response_token"]
            if category in NEUTRAL:
                neutral[category] += weight
            else:
                semantic[category][row["outcome"]] += weight
        q = {category: np.full(3, weight/mass) for category, weight in neutral.items()}
        neutral_mass = sum(neutral.values())/mass
        if semantic:
            class_article = sum(semantic.values(), np.zeros(3))
            support["semantic_class_weight"] = class_article.tolist()
            if min(class_article)+1e-9 < self.parameters["min_class_matches"]:
                return None, "insufficient_semantic_class_support", support
            alpha = self.parameters["dirichlet_mass"]
            article_mass = float(class_article.sum())
            for category, counts in semantic.items():
                pooled = float(counts.sum())/article_mass
                q[category] = (1.-neutral_mass)*(counts+alpha*pooled)/(class_article+alpha)
        if not q or np.max(np.abs(sum(q.values(), np.zeros(3))-1.)) > 1e-9:
            raise ValueError("Conditional response likelihood is not normalized")
        return q, "supported_frozen_conditional_kernel", support

    def enumerate(self, state, action, *, cost=0., loss="log_loss"):
        state = self._state(state)
        clean, _, _, _ = self._encode_action(action)
        q, reason, support = self._kernel(state, clean)
        return self._enumeration(state, action, clean, q, reason, support, cost, loss)

    def enumerate_hypothetical_request_class(self, state, action, *, cost=0., loss="log_loss"):
        """Plan an undiscovered request class, never an actual observation.

        The normalized frozen cluster center averages training-only request
        features. It avoids treating a training URL's unique identifier as a
        known future request. Existing concrete-request inference is unchanged.
        This abstraction is not a claim of calibrated within-class invariance.
        """
        state = self._state(state)
        clean, channel, token, _ = self._encode_action(action)
        if token is None:
            q, reason, support = None, "unsupported_hypothetical_request_class", {}
        else:
            key = "request-class:"+_digest([state["prior_probabilities"],
                [[x["channel"], x["action_token"], x["response_token"], x["unsupported"]]
                 for x in state["history"]], channel, token])
            if key not in self._kernel_cache:
                center = np.asarray(self.action_codebooks[channel].payload["centers"][int(token)])
                norm = float(np.linalg.norm(center))
                self._kernel_cache[key] = ((None, "unsupported_hypothetical_request_class", {}) if norm <= 0 else
                    self._kernel_uncached(state, action, request_vector=center/norm))
            q, reason, support = self._kernel_cache[key]
        result = self._enumeration(state, action, clean, q, reason, support, cost, loss)
        result.update(planning_only=True, request_representation="frozen_normalized_action_cluster_centroid",
                      request_class={"channel": channel, "action_token": token})
        result["decision_digest"] = _digest({k: v for k, v in result.items() if k != "decision_digest"})
        return result

    def _enumeration(self, state, action, clean, q, reason, support, cost, loss):
        aid = action.get("action_id") or "text-"+_digest(clean)[:20]
        prior = state["current_probabilities"]
        branches = []
        if q is None:
            value = unknown_candidate(aid, prior, reason=reason, cost=cost, loss=loss,
                                      predictor_version=self.model_version)
        else:
            p = np.asarray(prior)
            for token, likelihood in sorted(q.items()):
                joint = p*likelihood
                probability = float(joint.sum())
                if probability > 0:
                    branches.append({"branch_id": token, "probability": probability,
                                     "posterior": (joint/probability).tolist()})
            value = evaluate_candidate(aid, prior, branches, cost=cost, loss=loss,
                                       predictor_version=self.model_version,
                                       response_model_version=self.model_version,
                                       calibration_cutoff=self.metadata["training_as_of"])
        result = {"schema": self.version, "status": "supported" if q is not None else "unknown",
                  "reason": reason, "model_version": self.model_version, "state_digest": _digest(state),
                  "action": clean, "action_digest": _digest(clean), "prediction_before": prior,
                  "branches": branches, "value": value.as_dict(), "support": support,
                  "evidence_origin": self.metadata["evidence_origin"], "production_ready": False}
        result["decision_digest"] = _digest(result)
        return result

    def update(self, state, action, response, *, decision=None):
        state = self._state(state)
        clean, _, _, _ = self._encode_action(action)
        decision = self.enumerate(state, action) if decision is None else deepcopy(decision)
        if decision.get("planning_only"):
            raise ValueError("A hypothetical request-class plan cannot update actual observed evidence")
        if decision.get("decision_digest") != _digest({k: v for k, v in decision.items() if k != "decision_digest"}):
            raise ValueError("Frozen response decision digest mismatch")
        if (decision.get("model_version") != self.model_version or decision.get("state_digest") != _digest(state)
                or decision.get("action_digest") != _digest(clean)):
            raise ValueError("Response does not belong to this frozen model/state/action")
        entry = self._history_entry(action, response)
        token = entry["response_token"]
        branch = next((b for b in decision["branches"] if b["branch_id"] == token), None)
        if decision["status"] != "supported" or branch is None:
            entry["unsupported"] = True
            status, reason = "unknown", (decision["reason"] if decision["status"] != "supported"
                                         else "unseen_or_out_of_vocabulary_response")
        else:
            state["current_probabilities"] = _p(branch["posterior"])
            status, reason = "updated", "actual_response_matches_frozen_branch"
        state["history"].append(entry)
        return {"status": status, "reason": reason, "state": state, "observed_token": token,
                "prediction_after": state["current_probabilities"],
                "predicted_response_probability": branch["probability"] if status == "updated" else None,
                "model_version": self.model_version, "decision_digest": decision["decision_digest"]}

    def _payload(self):
        return {"schema": self.version, "parameters": self.parameters, "metadata": self.metadata,
                "records": self.records, "text_codebook": self.text_codebook.payload,
                "action_codebooks": {key: book.payload for key, book in sorted(self.action_codebooks.items())}}

    def save(self, path):
        if not self.model_version:
            raise ValueError("Cannot save an unfitted text model")
        payload = self._payload()
        data = {**payload, "model_version": self.model_version, "content_sha256": _digest(payload)}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        def reject(value):
            raise ValueError("Non-finite JSON constant: "+value)
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate JSON field: "+key)
                result[key] = value
            return result
        data = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject, object_pairs_hook=pairs)
        expected = {"schema", "parameters", "metadata", "records", "text_codebook", "action_codebooks",
                    "model_version", "content_sha256"}
        if not isinstance(data, dict) or set(data) != expected or data["schema"] != cls.version:
            raise ValueError("Unexpected frozen text model schema")
        payload = {k: v for k, v in data.items() if k not in {"model_version", "content_sha256"}}
        if data["content_sha256"] != _digest(payload) or data["model_version"] != cls.version+":"+_digest(payload)[:24]:
            raise ValueError("Frozen text model content mismatch")
        if data["metadata"].get("evidence_origin") not in ORIGINS or data["metadata"].get("production_ready") is not False:
            raise ValueError("Unvalidated text models cannot claim production readiness")
        model = cls()
        model.parameters, model.metadata, model.records = data["parameters"], data["metadata"], data["records"]
        model.text_codebook = _Codebook(data["text_codebook"])
        model.action_codebooks = {key: _Codebook(value) for key, value in data["action_codebooks"].items()}
        model.model_version = data["model_version"]
        return model


__all__ = ["TextReturnModel", "VERSION", "ENCODER_VERSION", "STATE_VERSION"]

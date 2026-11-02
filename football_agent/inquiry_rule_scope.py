"""Witnessed, source-reported competition-rule scope for a match inquiry.

Canonical endpoints and verbatim mentions are checked. Entity resolution,
relation entailment, section scope and freshness remain unverified. A rule can
be available for reasoning without proving qualification or rotation.
"""
from __future__ import annotations

from copy import deepcopy
import re
import unicodedata

from jsonschema import Draft202012Validator

from .inquiry_applicability import merge_bindings, validate_witnesses
from .inquiry_contract import ID, obj, normalized
from .inquiry_evidence_state import (assess_requirements as assess_evidence,
                                     project_requirements as project_evidence)
from .inquiry_probability import ReturnProbabilityProtocol


VERSION = "rule-scope-inquiry-v1"
MENTION = {"type": "string", "minLength": 1, "maxLength": 160}


def witness(*roles):
    return obj({"document_id": ID, "quote": {"type": "string", "minLength": 8, "maxLength": 1200},
                **{role + "_mention": deepcopy(MENTION) for role in roles}})


BINDING = obj({
    "binding_id": ID, "claim_id": ID, "fixture_claim_id": ID, "hypothesis_id": ID,
    "competition_id": ID, "campaign_key": ID,
    "season": {"type": "string", "pattern": "^[0-9]{4}(?:[-/][0-9]{2,4})?$"}, "team_id": ID, "fixture_id": ID,
    "rule_edition": witness("competition", "season"),
    "team_in_edition": witness("competition", "season", "team"),
    "fixture_in_edition": witness("competition", "season"),
    "team_in_fixture": witness("team"),
    "explanation": {"type": "string", "minLength": 1, "maxLength": 1000},
})
SCHEMA = deepcopy(ReturnProbabilityProtocol.schema)
SCHEMA["properties"]["rule_scope_bindings"] = {"type": "array", "maxItems": 6, "items": BINDING}
SCHEMA["required"].append("rule_scope_bindings")
SEMANTICS = {
    "version": VERSION, "scope": "source_reported_competition_rules_for_qualification_match",
    "checked": ["immutable_ids", "typed_endpoints", "acquired_quote_anchors", "consistent_literal_mentions"],
    "entity_resolution": "unverified_source_reported_aliases",
    "fixture_identity": "source_claim_not_independently_resolved",
    "relation_entailment": "unverified", "rule_section_scope": "unverified",
    "freshness": "unverified", "qualification_conclusion": "not_established_by_rule_binding",
    "rotation_conclusion": "not_supported_by_rule_binding", "information_value": None,
}
PROMPT = ReturnProbabilityProtocol.prompt + """
Rule-scope extension (a new, explicit route for competition-wide rules):
Legacy context_bindings STILL require identical source and target subjects. Do not place a
competition-subject rule -> team-subject qualification link there. Do not rewrite the rule's
subject as a team to pass that check. Use rule_scope_bindings instead, or keep the scope unknown.
Only SOURCE CLAIM / competition_rules / campaign can be linked to a qualification / match
hypothesis. These links supply a source-reported rule INPUT, never qualification truth or
rotation intent. For that target the rule claim must have a context link, not supports/challenges.

Every rule_scope_binding has explicit competition_id, campaign_key, season, team_id and fixture_id.
competition_id/campaign_key equal the rule claim's subject/scope_key. team_id/fixture_id equal the
target hypothesis's subject/scope_key. fixture_claim_id names a separately anchored source_claim
of kind other, same target team and match scope/key, also linked as context. Its text must describe
the actual source's fixture identity and retain unresolved aliases, month/year or timezones.
An article saying only 'the 27th' needs dated context to identify the month/year; do not silently
certify cross-source match identity or resolve differing kickoff times.

Supply FOUR role-specific acquired witnesses, each with document_id and exact quote:
* rule_edition: competition_mention + season_mention, from a document cited by the rule claim.
  Its quote must overlap/contain/be contained in a rule-claim citation, binding this edition anchor
  to that claim; include a separately anchored edition citation when needed.
* team_in_edition: competition_mention + season_mention + team_mention in the SAME quote, from a
  document cited by the rule claim. A nearby edition title and participation list may form one quote.
* fixture_in_edition: competition_mention + season_mention, directly anchored to a fixture_claim citation.
* team_in_fixture: team_mention, directly anchored to a fixture_claim citation. For both roles the
  exact quote must contain or be contained in that citation, not merely occur on the same page.
  A different match on the same page is not a witness for this fixture. The fixture statement may
  be cited again for this distinct role; a shared document is not independent corroboration.
Each mention must occur literally in its own quote. All season_mention values equal season.
season is a complete year or year range (for example 2026 or 2026-27), never a numeric prefix.
Use the same source-language competition/team surface names across witnesses; only Unicode
normalization, whitespace and case differences are mechanically recognized. No transliteration,
fuzzy aliases or implicit expansion of an acronym is performed. If no common literal names are
available, leave the link unresolved and consider obtaining an identity/alias witness later.
Quoted co-occurrence alone does not prove a membership or scope relation: explain the relation,
preserve ambiguity, and do not combine a K1 header with a K2 operative clause. Document-level
co-location does not establish section scope. Rules for another edition/group need new evidence.

Bindings are immutable. Repeating an identical binding is allowed; reusing its ID for changed
claims/endpoints/witnesses is not. A binding ID cannot collide with legacy context_bindings.
The runtime marks a witnessed rule as a source-reported input, with identity, section, semantics
and freshness unverified. It does not alter the original claims or promote the hypothesis direction.
standings still requires an effective time through its existing checks, and rotation still needs
its own evidence. Already acquired rules may change which information is worth seeking next;
the runtime does not prescribe rules -> standings -> coach as a fixed sequence.
You may reflect after adding a genuinely new rule_scope_binding to inspect the computed state;
that spends a model call and no HTTP. You may also use the new input and select a query in the
same response, or stop/defer/wait as justified. Do not refresh inherited rules merely to fill a
protocol field. Outcome probabilities remain subjective or abstained, never numerical VoI.
"""


def literal_key(value):
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def _contains_mention(quote, mention, *, season=False):
    text, value = normalized(quote), normalized(mention)
    if not value:
        return False
    left = r"(?<!\d)" if value[0].isdigit() else ""
    right = r"(?!\d|[-/]\d)" if season else r"(?!\d)" if value[-1].isdigit() else ""
    return re.search(left + re.escape(value) + right, text) is not None


def _merged(previous, updates):
    return merge_bindings(previous, updates)


def _context_link(claim, identity):
    return [link["relation"] for link in claim["hypothesis_links"] if link["hypothesis_id"] == identity] == ["context"]


def _anchored_to_claim(witness, claim):
    return any(c["document_id"] == witness["document_id"] and
               (normalized(c["quote"]) in normalized(witness["quote"]) or normalized(witness["quote"]) in normalized(c["quote"]))
               for c in claim["citations"])


def validate_rule_binding(binding, hypotheses, claims, documents):
    hypothesis = hypotheses.get(binding["hypothesis_id"])
    claim = claims.get(binding["claim_id"])
    fixture = claims.get(binding["fixture_claim_id"])
    if not hypothesis or not claim or not fixture:
        raise ValueError("Rule scope requires existing hypothesis, rule and fixture claims")
    evidence, match = claim["evidence"], fixture["evidence"]
    if (hypothesis["hypothesis_type"] != "qualification" or hypothesis["scope"] != "match"
            or "competition_rules" not in hypothesis["required_evidence"]):
        raise ValueError("Rule scope only supplies a match qualification prerequisite")
    if (claim["status"] != "source_claim" or evidence["kind"] != "competition_rules" or evidence["scope"] != "campaign"
            or fixture["status"] != "source_claim" or match["kind"] != "other" or match["scope"] != "match"):
        raise ValueError("Rule scope requires a campaign rule and a source-reported match identity")
    if (binding["competition_id"] != evidence["subject"] or binding["campaign_key"] != evidence["scope_key"]
            or binding["team_id"] != hypothesis["subject"] or binding["fixture_id"] != hypothesis["scope_key"]
            or match["subject"] != binding["team_id"] or match["scope_key"] != binding["fixture_id"]):
        raise ValueError("Rule scope endpoints differ from rule, fixture or target identities")
    if not _context_link(claim, hypothesis["hypothesis_id"]) or not _context_link(fixture, hypothesis["hypothesis_id"]):
        raise ValueError("Rule and fixture links must remain context, not qualification support")
    rule_docs = {c["document_id"] for c in claim["citations"]}
    fixture_docs = {c["document_id"] for c in fixture["citations"]}
    for role in ("rule_edition", "team_in_edition", "fixture_in_edition", "team_in_fixture"):
        item = binding[role]
        allowed = rule_docs if role in {"rule_edition", "team_in_edition"} else fixture_docs
        if item["document_id"] not in allowed:
            raise ValueError("Scope witness is not anchored to its rule or fixture claim")
        if role in {"fixture_in_edition", "team_in_fixture"} and not _anchored_to_claim(item, fixture):
            raise ValueError("Fixture witness must directly anchor the cited fixture, not another match on the page")
        validate_witnesses([item], documents)
        for key, mention in item.items():
            if key.endswith("_mention") and not _contains_mention(item["quote"], mention, season=key == "season_mention"):
                raise ValueError("Entity mention is absent from its own scope witness")
        if "season_mention" in item and normalized(item["season_mention"]) != normalized(binding["season"]):
            raise ValueError("Witness edition differs from the declared season")
    competition_mentions = [literal_key(binding[role]["competition_mention"])
                            for role in ("rule_edition", "team_in_edition", "fixture_in_edition")]
    if len(set(competition_mentions)) != 1 or not competition_mentions[0]:
        raise ValueError("Scope witnesses name different competitions or unresolved aliases")
    if literal_key(binding["team_in_edition"]["team_mention"]) != literal_key(binding["team_in_fixture"]["team_mention"]):
        raise ValueError("Scope witnesses name different teams or unresolved aliases")
    edition = binding["rule_edition"]
    if not _anchored_to_claim(edition, claim):
        raise ValueError("Rule edition witness must be directly anchored to a rule-claim citation")


def _projected_bindings(bindings):
    # These are temporary projection inputs, not rewritten claims or saved
    # legacy bindings. Their own witnessed endpoints remain in rule_scope.
    return [{"binding_id": b["binding_id"], "hypothesis_id": b["hypothesis_id"], "claim_id": b["claim_id"],
             "information_as_of": None, "rule_scope": {k: b[k] for k in
             ("competition_id", "campaign_key", "season", "team_id", "fixture_id", "fixture_claim_id")}} for b in bindings]


def project_requirements(hypotheses, claims, bindings):
    result = project_evidence(hypotheses, claims, bindings)
    rule_ids = {b["binding_id"] for b in bindings if "rule_scope" in b}
    for requirement in result:
        for slot in requirement["slots"]:
            for path in slot["evidence_paths"]:
                if path["binding_id"] in rule_ids:
                    path.update(via="witnessed_rule_scope", relation_semantics="source_reported_unverified",
                                identity_verification="unverified", section_scope_verification="unverified")
    return result


def assess_requirements(candidate, hypotheses, claims, bindings, requirements):
    result = assess_evidence(candidate, hypotheses, claims, bindings, requirements)
    intent = candidate["inquiry_intent"]
    by_id = {h["hypothesis_id"]: h for h in hypotheses}
    for target in result["targets"]:
        hypothesis = by_id[target["hypothesis_id"]]
        if (hypothesis["hypothesis_type"] != "qualification" or intent["scope"] != "campaign"
                or "competition_rules" not in intent["expected_evidence"]):
            continue
        linked = any(b["hypothesis_id"] == hypothesis["hypothesis_id"] and "rule_scope" in b
                     and b["rule_scope"]["competition_id"] == intent["subject"]
                     and b["rule_scope"]["campaign_key"] == intent["scope_key"] for b in bindings)
        if linked:
            field = "potential_missing_fills" if "competition_rules" in target["missing_before"] else "already_present_types"
            target[field] = sorted(set(target[field]) | {"competition_rules"})
        elif intent["subject"] != hypothesis["subject"]:
            target["requires_applicability_link"] = sorted(set(target["requires_applicability_link"]) | {"competition_rules"})
        target["scope_verification"] = "witnessed_source_relation_unverified" if linked else "scope_witnesses_required"
    return result


class RuleScopeProtocol(ReturnProbabilityProtocol):
    version, schema, prompt = VERSION, SCHEMA, PROMPT
    projection_fields = (*ReturnProbabilityProtocol.projection_fields, "rule_scope_bindings")
    project_requirements = staticmethod(project_requirements)
    assess_requirements = staticmethod(assess_requirements)

    def validate(self, response, public):
        Draft202012Validator(self.schema).validate(response)
        base = deepcopy(response)
        updates = base.pop("rule_scope_bindings")
        rules = _merged(public.get("rule_scope_bindings", []), updates)
        legacy = _merged(public.get("context_bindings", []), response["context_bindings"])
        if {b["binding_id"] for b in rules} & {b["binding_id"] for b in legacy}:
            raise ValueError("Rule-scope and legacy binding IDs must be distinct")
        newly_bound = {b["binding_id"] for b in updates} - {b["binding_id"] for b in public.get("rule_scope_bindings", [])}
        if base["decision"]["kind"] == "reflect" and newly_bound:
            # Legacy validation does not know this new kind of reflection
            # delta. Its non-query shape is identical to defer; actual action
            # and evidence are unchanged, and the new delta is checked here.
            base["decision"]["kind"] = "defer"
        ReturnProbabilityProtocol().validate(base, public)
        hypotheses = {h["hypothesis_id"]: h for h in public.get("hypotheses", []) + response["hypotheses"]}
        claims = {c["claim_id"]: c for c in public["claims"] + response["claim_updates"]}
        documents = {d["document_id"]: d for d in public["documents"]}
        for binding in rules:
            validate_rule_binding(binding, hypotheses, claims, documents)
        return response

    def public(self, run):
        return {**super().public(run), "rule_scope_bindings": deepcopy(run.get("rule_scope_bindings", [])),
                "rule_scope_semantics": deepcopy(SEMANTICS)}

    def advance(self, run, response):
        rules = _merged(run.get("rule_scope_bindings", []), response["rule_scope_bindings"])
        legacy = _merged(run.get("context_bindings", []), response["context_bindings"])
        run["rule_scope_bindings"] = rules
        run["context_bindings"] = [*run.get("context_bindings", []), *_projected_bindings(rules)]
        try:
            assessment = super().advance(run, response)
        finally:
            run["context_bindings"] = legacy
        assessment["rule_scope"] = {"bindings": deepcopy(rules), "semantics": deepcopy(SEMANTICS),
            "effect": "rule_input_availability_only_not_qualification_or_rotation_truth"}
        return assessment

    def contract(self, run, candidate):
        ids = set(candidate["inquiry_intent"]["hypothesis_ids"])
        return {**super().contract(run, candidate), "rule_scope_before": [deepcopy(b)
            for b in run.get("rule_scope_bindings", []) if b["hypothesis_id"] in ids],
            "rule_scope_semantics": deepcopy(SEMANTICS)}

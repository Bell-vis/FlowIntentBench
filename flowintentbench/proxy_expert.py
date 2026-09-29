"""Proxy flow-expert orchestration with explicit information firewalls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FORBIDDEN_AUDITOR_KEYS = frozenset({"model_answers", "model_scores", "accepted_O", "reference_findings"})
FORBIDDEN_ANALYST_KEYS = frozenset({"accepted_O", "existing_f2_roles", "reference_findings", "other_proxy_answers"})
FORBIDDEN_ADJUDICATOR_KEYS = frozenset({
    "producer_identity", "producer_agent_id", "producer_model_family",
    "tested_model_identity", "tested_model_score", "model_score",
    "other_adjudicator_verdict", "orchestration_provenance",
    "ground_truth", "reference_findings", "accepted_operationalizations",
    "reference_membership", "benchmark_scores",
    # These are evaluator routing/membership annotations, not executed
    # scientific evidence. Keep the actual selected field and all numerical
    # evidence, but never reveal the hidden branch/reference inventory.
    "field_selection_policy", "selection_resolution",
    "identity_reference_ids", "reference_ids_by_field",
})
FORBIDDEN_SCIENTIFIC_REVIEWER_KEYS = frozenset({
    "ground_truth", "accepted_o", "accepted_operationalization", "reference_branch",
    "reference_findings", "curator_decision", "previous_expert_verdict", "scq_result",
    "srac_adjudication", "benchmark_score", "evaluated_model_answer", "model_answer",
    "model_score", "evaluator_output",
})

ROLE_ALLOWED_KEYS = {
    "auditor": frozenset({
        "question", "context", "dataset_contract", "scientific_family_contract",
        "condition", "principal_dimensions", "resolved_dimensions",
        "unresolved_dimensions", "case_input", "dataset_identity",
        "dataset_metadata", "grounding_evidence_bundle", "scientific_target",
        "responsibility_contract", "evaluation_representability_summary",
        "runtime_contract", "scientific_support_claims", "claim_support_links",
        "evidence_references", "claim_provenance", "old_repository_evidence",
        "new_external_evidence", "new_external_sources", "known_limitations",
        # Evidence-conditioned condition eligibility view. These are
        # construction evidence fields, not adjudication or GT outcomes.
        "dataset_context", "conditions", "current_evidence_records",
        "candidate_source_records", "current_unresolved_claims",
    }),
    "analyst": frozenset({"question", "context", "dataset_contract", "condition", "runtime_contract", "fixed_o", "fixed_evidence_context", "scientific_target", "principal_dimensions", "resolved_dimensions", "unresolved_dimensions"}),
    "adjudicator": frozenset({
        "question", "context", "scientific_target", "scientific_scope",
        "candidate_scientific_O", "materialized_G_of_O",
        "fixed_evidence_context", "candidate_F",
        "structured_scientific_evidence",
    }),
    "scientific_reviewer": frozenset({
        "scientific_review_packet_version", "review_input_status", "dataset_identity", "family_identity",
        "scientific_target", "dataset_context", "observable_semantics", "conditions",
        "scientific_claims", "evidence_records", "source_records", "claim_support_links",
        "provenance_summary", "runtime_contract",
        # The same canonical Flow Expert may be invoked in a presentation-only
        # authoring or post-rewrite fidelity mode.  These fields expose the
        # frozen semantic contract and final wording, never GT or adjudication.
        "question_presentation_authoring_packet_version",
        "post_rewrite_semantic_fidelity_packet_version",
        "final_flow_expert_wording_review_packet_version",
        "task", "case_identity", "semantic_contract_sha256",
        "scientific_scope", "fixed_operationalization",
        "unresolved_operationalization_dimensions", "finding_responsibility",
        "candidate_count", "presentation_requirements", "model_visible_text",
        "semantic_visibility", "review_requirements", "evidence_context",
        "repair_feedback",
        # Combustor target comparison is another mode of the same canonical
        # Flow Expert.  The alternatives and selection contract are the
        # scientific object being reviewed; dropping them would leave the
        # reviewer with evidence but no targets to compare.  These fields are
        # construction inputs only and carry no GT, SRAC, curator, or model
        # evaluation result.
        "family_id", "dataset_id", "previous_target", "candidate_targets",
        "dataset_capability_profile", "selection_contract",
    }),
}


@dataclass(frozen=True)
class ProxyReviewProfile:
    review_profile_id: str
    proposer_pool: tuple[Mapping[str, Any], ...]
    adjudicator_pool: tuple[Mapping[str, Any], ...]
    auditor_pool: tuple[Mapping[str, Any], ...] = ()
    min_open_case_proposal_sources: int = 2
    prefer_distinct_model_families: bool = True
    candidate_source_blinding: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"review_profile_id": self.review_profile_id, "proposer_pool": [dict(x) for x in self.proposer_pool], "adjudicator_pool": [dict(x) for x in self.adjudicator_pool], "auditor_pool": [dict(x) for x in self.auditor_pool], "min_open_case_proposal_sources": self.min_open_case_proposal_sources, "prefer_distinct_model_families": self.prefer_distinct_model_families, "candidate_source_blinding": self.candidate_source_blinding}


@dataclass(frozen=True)
class AdjudicationCandidateView:
    question: str = ""
    context: Any = None
    scientific_target: Any = None
    scientific_scope: Any = None
    candidate_scientific_O: Any = None
    materialized_G_of_O: Any = None
    fixed_evidence_context: Any = None
    candidate_F: Any = None
    structured_scientific_evidence: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in {
            "question": self.question, "context": self.context,
            "scientific_target": self.scientific_target,
            "scientific_scope": self.scientific_scope,
            "candidate_scientific_O": self.candidate_scientific_O,
            "materialized_G_of_O": self.materialized_G_of_O,
            "fixed_evidence_context": self.fixed_evidence_context,
            "candidate_F": self.candidate_F,
            "structured_scientific_evidence": self.structured_scientific_evidence,
        }.items() if value is not None}


def _strip_forbidden(value: Any, forbidden: set[str], *, nested: bool = True) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in forbidden or any(token in normalized for token in ("producer_agent_id", "producer_model_family", "tested_model_identity", "tested_model_score", "other_adjudicator_verdict")):
                continue
            result[str(key)] = _strip_forbidden(item, forbidden, nested=nested)
        return result
    if isinstance(value, list):
        return [_strip_forbidden(item, forbidden, nested=nested) for item in value]
    if isinstance(value, tuple):
        return [_strip_forbidden(item, forbidden, nested=nested) for item in value]
    return value


def build_firewalled_payload(role: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {"auditor": FORBIDDEN_AUDITOR_KEYS, "analyst": FORBIDDEN_ANALYST_KEYS, "adjudicator": FORBIDDEN_ADJUDICATOR_KEYS, "scientific_reviewer": FORBIDDEN_SCIENTIFIC_REVIEWER_KEYS}.get(role)
    allowed = ROLE_ALLOWED_KEYS.get(role)
    if forbidden is None or allowed is None:
        raise ValueError(f"unknown proxy role: {role}")
    visible = {str(key): value for key, value in payload.items() if str(key) in allowed}
    return _strip_forbidden(visible, set(forbidden))


def build_adjudication_candidate_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the only payload shape accepted by an adjudicator."""
    # A serialized SRACProposal is orchestration metadata, not a science view.
    # Reject it at the boundary instead of silently producing an empty view;
    # callers must explicitly project candidate content into the allowlist.
    # Identity is recursively stripped, while orchestration structure itself
    # (proposal IDs, condition, proposal payloads) is rejected.  This keeps
    # backwards-compatible identity scrubbing without retaining a generic
    # candidate wrapper as a scientific API.
    raw_proposal_keys = {"proposal_id", "condition", "proposed_operationalizations", "proposed_findings", "proposal_kind", "adjudication_id", "materialization_id"}

    def contains_raw(value: Any) -> bool:
        if isinstance(value, Mapping):
            if any(str(key).casefold() in raw_proposal_keys for key in value):
                return True
            return any(contains_raw(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_raw(item) for item in value)
        return False

    if isinstance(payload, AdjudicationCandidateView):
        return build_firewalled_payload("adjudicator", payload.to_dict())
    if not isinstance(payload, Mapping):
        raise ValueError("ADJUDICATOR_VIEW_MAPPING_REQUIRED")
    if contains_raw(payload):
        raise ValueError("RAW_PROPOSAL_NOT_ALLOWED")
    if "candidate" in payload:
        # A legacy wrapper is never projected: it contributes no
        # model-visible content.  Raw proposal structure was already rejected
        # recursively above; identity-only wrappers are scrubbed for backwards
        # compatibility with older audit fixtures.
        return {}
    # Unknown orchestration keys are dropped at the boundary.  Scientific
    # fields are explicitly enumerated by ROLE_ALLOWED_KEYS; the generic
    # ``candidate`` escape hatch is intentionally absent.
    return build_firewalled_payload("adjudicator", payload)


def prepare_adjudicator_invocation(
    proposal: Any,
    adjudicator: Mapping[str, Any],
    scientific_view: AdjudicationCandidateView | Mapping[str, Any],
    invoke: Any,
) -> Any:
    """Guard and project a candidate before invoking an adjudicator model."""

    producer = str(getattr(proposal, "producer_agent_id", "") or (proposal.get("producer_agent_id", "") if isinstance(proposal, Mapping) else ""))
    adjudicator_id = str(adjudicator.get("agent_id", ""))
    enforce_self_adjudication_guard(producer, adjudicator_id)
    view = build_adjudication_candidate_view(scientific_view)
    if not callable(invoke):
        raise ValueError("invoke must be callable")
    return invoke(view)


def validate_adjudicator_semantic_coverage(
    *,
    condition: str,
    proposal: Any,
    firewalled_view: Mapping[str, Any],
    adjudication: Mapping[str, Any] | None = None,
    requested_scientific_view: Mapping[str, Any] | None = None,
    allowed_finding_roles: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate that the scientific object being judged survived projection.

    This validator checks coverage only.  It never decides whether an O or F
    is scientifically valid and never checks reference membership.
    """

    normalized = str(condition).upper().replace("_", "-")
    errors: list[str] = []
    if requested_scientific_view is not None:
        # The requested and actual scientific views must agree on every
        # contract-bearing field.  Identity/orchestration keys are excluded
        # by construction and are not part of this comparison.
        import json

        def canonical(value: Any) -> str:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )

        for key in (
            "question",
            "context",
            "scientific_target",
            "scientific_scope",
            "candidate_scientific_O",
            "materialized_G_of_O",
            "fixed_evidence_context",
            "candidate_F",
            "structured_scientific_evidence",
        ):
            if key in requested_scientific_view and key in firewalled_view:
                if canonical(requested_scientific_view[key]) != canonical(
                    firewalled_view[key]
                ):
                    errors.append(f"ADJUDICATOR_SCIENTIFIC_VIEW_DRIFT:{key}")
            elif key in requested_scientific_view and requested_scientific_view[key] is not None:
                errors.append(f"ADJUDICATOR_SCIENTIFIC_VIEW_DROPPED:{key}")
    candidate_o = firewalled_view.get("candidate_scientific_O")
    materialized = firewalled_view.get("materialized_G_of_O")
    if normalized in {"O2-F1", "O3-F1"}:
        if not isinstance(candidate_o, Mapping) or not candidate_o:
            errors.append("O_ADJUDICATION_CANDIDATE_MISSING")
        if not isinstance(materialized, Mapping) or not materialized:
            errors.append("O_ADJUDICATION_G_OF_O_MISSING")
    elif normalized == "O1-F2":
        if not isinstance(candidate_o, Mapping) or not candidate_o:
            errors.append("F2_FIXED_O_MISSING")
        if not isinstance(materialized, Mapping) or not materialized:
            errors.append("F2_FIXED_G_OF_O_MISSING")
        candidates = firewalled_view.get("candidate_F")
        if not isinstance(candidates, list) or not candidates:
            errors.append("F2_CANDIDATE_FINDINGS_MISSING")
            candidate_ids: set[str] = set()
        else:
            candidate_ids = {
                str(item.get("finding_id"))
                for item in candidates
                if isinstance(item, Mapping) and str(item.get("finding_id", "")).strip()
            }
            if len(candidate_ids) != len(candidates):
                errors.append("F2_CANDIDATE_FINDING_IDS_INVALID")
        proposed = (
            proposal.get("proposed_findings", ())
            if isinstance(proposal, Mapping)
            else getattr(proposal, "proposed_findings", ())
        )
        expected_ids = {
            str(item.get("finding_id"))
            for item in proposed
            if isinstance(item, Mapping) and str(item.get("finding_id", "")).strip()
        }
        if candidate_ids != expected_ids:
            errors.append("F2_FIREWALL_CANDIDATE_SET_MISMATCH")
        if adjudication is not None:
            allowed = {
                "finding_support": {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"},
                "finding_relevance": {"RELEVANT", "IRRELEVANT", "UNCERTAIN"},
                "o_f_consistency": {"CONSISTENT", "INCONSISTENT", "UNCERTAIN"},
            }
            for field, labels in allowed.items():
                values = adjudication.get(field)
                if not isinstance(values, Mapping) or set(map(str, values)) != candidate_ids:
                    errors.append(f"F2_{field.upper()}_COVERAGE_MISMATCH")
                    continue
                if any(str(value).upper() not in labels for value in values.values()):
                    errors.append(f"F2_{field.upper()}_LABEL_INVALID")
            roles = adjudication.get("finding_role")
            if not isinstance(roles, Mapping):
                roles = {}
            role_ids = set(map(str, roles))
            if not role_ids <= candidate_ids:
                errors.append("F2_FINDING_ROLE_COVERAGE_MISMATCH")
            elif any(not str(value).strip() for value in roles.values()):
                errors.append("F2_FINDING_ROLE_LABEL_INVALID")
            # A role is required only for a candidate that could receive F2
            # credit.  Explicitly rejected candidates do not need a semantic
            # role label and are represented as NOT_APPLICABLE by omission.
            if isinstance(adjudication.get("finding_support"), Mapping) and isinstance(adjudication.get("finding_relevance"), Mapping) and isinstance(adjudication.get("o_f_consistency"), Mapping):
                for finding_id in candidate_ids:
                    eligible = (
                        str(adjudication["finding_support"].get(finding_id, "")).upper() == "SUPPORTED"
                        and str(adjudication["finding_relevance"].get(finding_id, "")).upper() == "RELEVANT"
                        and str(adjudication["o_f_consistency"].get(finding_id, "")).upper() == "CONSISTENT"
                    )
                    if eligible and finding_id not in role_ids:
                        errors.append("F2_FINDING_ROLE_REQUIRED_FOR_CREDIT")
                    if (
                        eligible
                        and finding_id in role_ids
                        and allowed_finding_roles is not None
                        and str(roles[finding_id]) not in set(allowed_finding_roles)
                    ):
                        errors.append("F2_FINDING_ROLE_NOT_IN_CONTRACT")
    else:
        errors.append("ADJUDICATOR_COVERAGE_CONDITION_UNSUPPORTED")
    return {
        "status": "PASS" if not errors else "FAIL",
        "condition": normalized,
        "errors": errors,
    }


# Explicit name used by integration callers; both names enforce the same
# structural candidate-view boundary.
build_adjudicator_payload = build_adjudication_candidate_view


def enforce_self_adjudication_guard(producer_agent_id: str, adjudicator_agent_id: str) -> None:
    if producer_agent_id and adjudicator_agent_id and producer_agent_id == adjudicator_agent_id:
        raise ValueError("SELF_ADJUDICATION_FORBIDDEN")


def audit_proxy_independence(profile: ProxyReviewProfile) -> dict[str, Any]:
    proposer_ids = {str(item.get("agent_id")) for item in profile.proposer_pool if item.get("agent_id")}
    adjudicator_ids = {str(item.get("agent_id")) for item in profile.adjudicator_pool if item.get("agent_id")}
    proposer_families = {str(item.get("model_family")) for item in profile.proposer_pool if item.get("model_family")}
    adjudicator_families = {str(item.get("model_family")) for item in profile.adjudicator_pool if item.get("model_family")}
    families = proposer_families | adjudicator_families
    overlap = sorted(proposer_ids & adjudicator_ids)
    status = "SELF_ADJUDICATION_FORBIDDEN" if overlap else "MULTI_FAMILY_PROXY_VALIDATED" if len(families) > 1 and len(proposer_families) >= profile.min_open_case_proposal_sources else "MODEL_FAMILY_DIVERSITY_LIMITED"
    return {"status": status, "distinct_model_family_count": len(families), "proposer_model_families": sorted(proposer_families), "adjudicator_model_families": sorted(adjudicator_families), "overlapping_agent_ids": overlap, "self_adjudication_forbidden": True, "candidate_source_blinding": profile.candidate_source_blinding}


def orchestrate_proxy_dry_run(profile: ProxyReviewProfile, condition: str) -> dict[str, Any]:
    from .srac_router import route_srac_condition
    route = route_srac_condition(condition)
    return {"review_profile_id": profile.review_profile_id, "condition": route.condition, "route": route.to_dict(), "orchestration_status": audit_proxy_independence(profile)["status"], "proposer_call_count": len(profile.proposer_pool) if route.independent_O_elicitation or route.independent_F_elicitation else 0, "adjudicator_call_count": 0, "live_model_calls": False}


__all__ = ["ProxyReviewProfile", "AdjudicationCandidateView", "build_firewalled_payload", "build_adjudication_candidate_view", "build_adjudicator_payload", "prepare_adjudicator_invocation", "validate_adjudicator_semantic_coverage", "enforce_self_adjudication_guard", "audit_proxy_independence", "orchestrate_proxy_dry_run"]

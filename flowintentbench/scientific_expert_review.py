"""Canonical, evidence-conditioned Flow Expert scientific review.

This module owns the tool-free reviewer interface and its information
boundary.  It produces advisory atomic scientific observations only;
condition eligibility and benchmark lifecycle decisions remain elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .case_design import FindingRequirementCategory, OperationalizationDimension
from .context import EvidenceRecord, SourceRecord
from .grounding import (
    SUPPORT_CLAIM_TYPES,
    SUPPORT_RELATIONS,
    FamilyScientificEvidenceResolution,
)
from .live_agents import LiveModelCaller
from .schema import CaseContext


FLOW_SCIENTIFIC_REVIEWER_PROFILE = "flow-scientific-reviewer-gpt-5.6-sol"
FLOW_SCIENTIFIC_REVIEWER_MAX_OUTPUT_TOKENS = 4000
SCIENTIFIC_REVIEW_PACKET_VERSION = "scientific-expert-review-packet-v1"
SCIENTIFIC_REVIEW_CONTRACT_VERSION = "scientific-expert-review-contract-v2"
SCIENTIFIC_REVIEW_API_VERSION = "scientific-expert-review-api-v1"
SCIENTIFIC_REVIEW_RUNTIME_CONTRACT = {
    "tools": [],
    "python": False,
    "web": False,
    "network": "unavailable",
}
ATOMIC_OBSERVATION_FIELDS = (
    "scientific_target_supported",
    "fixed_o_supported",
    "unresolved_o_is_legitimate_scientific_choice",
    "finding_contract_supported",
    "materialization_scientifically_meaningful",
    "question_target_preserved",
    "question_scope_preserved",
    "fixed_o_question_faithful",
    "unresolved_o_exposed_by_question",
)
ADVISORY_RECOMMENDATIONS = frozenset(
    {"ELIGIBLE", "NOT_ELIGIBLE", "MORE_EVIDENCE_REQUIRED", "REVISE_QUESTION"}
)
QUESTION_RECOMMENDATIONS = frozenset(
    {
        "KEEP",
        "REVISE_WORDING",
        "REVISE_TARGET",
        "REVISE_OPERATIONALIZATION_SPACE",
        "NOT_SCIENTIFICALLY_SUPPORTABLE",
    }
)
_FORBIDDEN_PACKET_KEYS = frozenset(
    {
        "ground_truth",
        "accepted_o",
        "accepted_operationalization",
        "reference_branch",
        "reference_findings",
        "curator_decision",
        "previous_expert_verdict",
        "scq_result",
        "srac_adjudication",
        "benchmark_score",
        "evaluated_model_answer",
        "model_answer",
        "model_score",
        "evaluator_output",
        "support_status",
        "curator_status",
        "reference_answer",
        "tested_model_output",
        "accepted_branch",
        "accepted_branches",
        "accepted_label",
        "accepted_labels",
        "g_of_o",
        "ground_truth_values",
        "reference_branches",
        "reference_value",
        "reference_values",
        "validity_label",
        "validity_status",
    }
)
_FORBIDDEN_PACKET_KEY_FRAGMENTS = (
    "ground_truth",
    "accepted_o",
    "accepted_branch",
    "accepted_operationalization",
    "branch_id",
    "curator_decision",
    "g_of_o",
    "previous_expert_verdict",
    "reference_answer",
    "reference_branch",
    "reference_finding",
    "reference_o",
    "reference_value",
    "source_case_id",
    "evaluated_model_answer",
    "tested_model_output",
    "evaluator_output",
)
_PACKET_FIELDS = frozenset(
    {
        "scientific_review_packet_version",
        "review_input_status",
        "dataset_identity",
        "family_identity",
        "scientific_target",
        "dataset_context",
        "observable_semantics",
        "conditions",
        "scientific_claims",
        "evidence_records",
        "evidence_family_bindings",
        "source_records",
        "claim_support_links",
        "provenance_summary",
        "runtime_contract",
    }
)
_CONDITION_FIELDS = frozenset(
    {
        "condition",
        "case_id",
        "question",
        "scientific_target",
        "finding_goal",
        "operationalization_responsibility",
        "finding_responsibility",
        "finding_openness",
        "fixed_o_dimensions",
        "unresolved_o_dimensions",
        "finding_requirements",
        "authored_operationalizations",
        "materialization_route",
    }
)
_FINDING_REQUIREMENT_FIELDS = frozenset(
    {
        "category",
        "question_fragment",
        "statement",
        "requirement_type",
        "adequate_core_sets", "supporting_roles", "role_by_category", "novel_role_policy",
        "mandatory_roles",
        "alternative_role_groups",
        "scientific_entity_type",
    }
)
_AUTHORED_OPERATIONALIZATION_FIELDS = frozenset(
    {
        "case_id",
        "candidate_index",
        "decision_status",
        "decisions",
        "scientific_result_values_visible",
    }
)
_EVIDENCE_FAMILY_BINDING_FIELDS = frozenset(
    {"evidence_id", "dataset_id", "family_id", "concept_id"}
)
_ALTERNATIVE_ROLE_GROUP_FIELDS = frozenset(
    {"group_id", "min_required", "roles"}
)
_OPERATIONALIZATION_DECISION_FIELDS = frozenset(
    {"dimension_id", "canonical_id", "description"}
)
_MATERIALIZATION_ROUTE_FIELDS = frozenset(
    {
        "execution_status",
        "handler_id",
        "effective_o_condition",
        "branch_count",
        "branch_statuses",
        "scientific_result_values_visible",
    }
)
_CLAIM_FIELDS = frozenset(
    {"claim_id", "claim_type", "statement", "critical", "evidence_ids", "source_ids"}
)
_SOURCE_FIELDS = frozenset(
    {"source_id", "source_type", "title", "authors", "year", "venue", "doi", "report_id"}
)
_EVIDENCE_FIELDS = frozenset(
    {
        "evidence_id",
        "dataset_id",
        "evidence_type",
        "statement",
        "source_id",
        "locator",
        "eligible_for_context",
        "context_facts",
    }
)
_LINK_FIELDS = frozenset(
    {"claim_id", "evidence_record_id", "source_id", "support_relation", "supported_statement"}
)
_PROVENANCE_FIELDS = frozenset(
    {
        "family_binding_status",
        "bound_identity",
        "evidence_provenance_closure",
        "claim_evidence_sufficiency",
        "unresolved_claim_ids",
        "visible_claim_count",
        "evidence_count",
        "source_count",
    }
)
_OUTPUT_FIELDS = frozenset({"condition_reviews", "question_reviews"})
_CONDITION_REVIEW_FIELDS = frozenset(
    {
        "condition",
        *ATOMIC_OBSERVATION_FIELDS,
        "eligibility_recommendation",
        "rationale",
        "finding_contract_condition_specific_rationale",
        "scientific_ambiguities",
        "evidence_ids",
        "source_ids",
    }
)
_QUESTION_REVIEW_FIELDS = frozenset(
    {"condition", "recommendation", "rationale", "evidence_ids"}
)
_OPERATIONALIZATION_DIMENSIONS = frozenset(
    item.value for item in OperationalizationDimension
)
_FINDING_REQUIREMENT_CATEGORIES = frozenset(
    item.value for item in FindingRequirementCategory
)


class ScientificExpertReviewError(ValueError):
    """Raised when a scientific review packet violates its firewall."""


def _nonempty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ScientificExpertReviewError(f"{label} must be a non-empty string")
    return text


def _string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ScientificExpertReviewError(f"{label} must be a list")
    result = [_nonempty(item, label) for item in value]
    if len(result) != len(set(result)):
        raise ScientificExpertReviewError(f"{label} must not contain duplicates")
    return result


def _assert_firewall(value: Any, path: str = "packet") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            normalized = normalized.replace("-", "_").replace(" ", "_")
            compact = "".join(character for character in normalized if character.isalnum())
            if normalized in _FORBIDDEN_PACKET_KEYS or any(
                fragment in normalized
                or "".join(character for character in fragment if character.isalnum())
                in compact
                for fragment in _FORBIDDEN_PACKET_KEY_FRAGMENTS
            ):
                raise ScientificExpertReviewError(
                    f"forbidden scientific-review field at {path}.{key}"
                )
            _assert_firewall(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_firewall(item, f"{path}[{index}]")


def _unexpected_fields(value: Mapping[str, Any], allowed: frozenset[str]) -> list[str]:
    return sorted(str(key) for key in set(value) - allowed)


def _closed_object(
    value: Any,
    *,
    allowed: frozenset[str],
    field: str,
    errors: list[dict[str, str]],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append({"code": "OBJECT_REQUIRED", "field": field})
        return None
    unexpected = _unexpected_fields(value, allowed)
    if unexpected:
        errors.append(
            {"code": "UNKNOWN_FIELD", "field": f"{field}:{','.join(unexpected)}"}
        )
    return value


def _validate_string_sequence(
    value: Any,
    *,
    field: str,
    errors: list[dict[str, str]],
) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        errors.append({"code": "STRING_LIST_REQUIRED", "field": field})
        return []
    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        errors.append({"code": "DUPLICATE_LIST_VALUE", "field": field})
    return normalized


def _materialization_view(
    condition: Mapping[str, Any], dataset_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    authored = condition.get("authored_operationalizations")
    route = condition.get("materialization_route")
    if authored is not None or route is not None:
        authored_rows = [dict(item) for item in authored or () if isinstance(item, Mapping)]
        route_value = dict(route) if isinstance(route, Mapping) else {}
        return authored_rows, route_value

    raw = condition.get("materialization")
    if not isinstance(raw, Mapping):
        return [], {"execution_status": "NOT_AVAILABLE", "branch_count": 0, "branch_statuses": []}
    branches = [
        item
        for item in raw.get("branches", ()) or ()
        if isinstance(item, Mapping)
    ]
    authored_rows: list[dict[str, Any]] = []
    raw_branch_statuses: list[str] = []
    for index, branch in enumerate(branches, start=1):
        execution = branch.get("execution")
        execution = execution if isinstance(execution, Mapping) else {}
        status = str(branch.get("status") or execution.get("status") or "NOT_AVAILABLE")
        raw_branch_statuses.append(status)
        decisions = branch.get("effective_operationalization")
        if not isinstance(decisions, Mapping):
            decisions = execution.get("effective_operationalization")
        decision_rows = [
            {
                "dimension_id": str(dimension),
                "canonical_id": str(value),
                "description": str(value),
            }
            for dimension, value in (decisions.items() if isinstance(decisions, Mapping) else ())
        ]
        authored_rows.append(
            {
                "candidate_index": index,
                "decision_status": "AUTHORED_EXECUTABLE_CANDIDATE",
                "decisions": decision_rows,
                "scientific_result_values_visible": False,
            }
        )
    expected_effective_condition = (
        "O1-F1"
        if str(condition.get("condition", "")) == "O1-F2"
        else str(condition.get("condition", ""))
    )

    def has_result(value: Any) -> bool:
        if isinstance(value, Mapping):
            return bool(value)
        return isinstance(value, Sequence) and not isinstance(
            value, (str, bytes)
        ) and bool(value)

    execution_recorded = (
        str(raw.get("status", "")) == "MATERIALIZED"
        and str(raw.get("case_id", "")) == str(condition.get("case_id", ""))
        and str(raw.get("effective_o_condition", ""))
        == expected_effective_condition
        and bool(str(raw.get("handler_id", "")).strip())
        and bool(branches)
        and all(
            status == "MATERIALIZED"
            and str(branch.get("effective_o_condition", ""))
            == expected_effective_condition
            and isinstance(branch.get("effective_operationalization"), Mapping)
            and bool(branch.get("effective_operationalization"))
            and isinstance(branch.get("execution"), Mapping)
            and str(branch["execution"].get("status", "")) == "MATERIALIZED"
            and str(branch["execution"].get("dataset_id", "")) == dataset_id
            and has_result(branch.get("G_of_O"))
            and has_result(branch["execution"].get("G_of_O"))
            for branch, status in zip(branches, raw_branch_statuses, strict=True)
        )
    )
    if str(raw.get("status", "")) == "MATERIALIZED" and not execution_recorded:
        raise ScientificExpertReviewError(
            f"materialization execution integrity mismatch for {condition.get('case_id', '')}"
        )
    route_declared = bool(str(raw.get("handler_id", "")).strip())
    return authored_rows, {
        "execution_status": (
            "MATERIALIZED_RESULTS_WITHHELD"
            if execution_recorded
            else "DETERMINISTIC_ROUTE_DECLARED_AVAILABLE"
            if route_declared
            else "DETERMINISTIC_ROUTE_NOT_DECLARED"
        ),
        "handler_id": str(raw.get("handler_id", "")),
        "effective_o_condition": str(raw.get("effective_o_condition", "")),
        "branch_count": len(branches),
        "branch_statuses": [
            "MATERIALIZED_RESULT_WITHHELD"
            if execution_recorded
            else "AUTHORED_O_ROUTE_DECLARED"
            if route_declared
            else "AUTHORED_O_ROUTE_NOT_DECLARED"
            for _ in branches
        ],
        "scientific_result_values_visible": False,
    }


def _condition_view(condition: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    authored, route = _materialization_view(condition, dataset_id)
    case_id = _nonempty(condition.get("case_id"), "case_id")
    authored = [{**item, "case_id": case_id} for item in authored]
    metadata = condition.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    operationalization_responsibility = _nonempty(
        condition.get(
            "operationalization_responsibility",
            metadata.get("operationalization_responsibility"),
        ),
        "operationalization_responsibility",
    )
    finding_openness = _nonempty(
        condition.get("finding_openness", metadata.get("finding_openness")),
        "finding_openness",
    )
    finding_responsibility = str(condition.get("finding_responsibility", "")).strip()
    if not finding_responsibility:
        finding_responsibility = "F2" if finding_openness.casefold() == "open" else "F1"
    finding_requirements = condition.get("finding_requirements", ()) or ()
    if not isinstance(finding_requirements, Sequence) or isinstance(
        finding_requirements, (str, bytes)
    ):
        raise ScientificExpertReviewError("finding_requirements must be a list")
    return {
        "condition": _nonempty(condition.get("condition"), "condition"),
        "case_id": case_id,
        "question": _nonempty(condition.get("question"), "question"),
        "scientific_target": _nonempty(
            condition.get("scientific_target"), "condition.scientific_target"
        ),
        "finding_goal": _nonempty(condition.get("finding_goal"), "finding_goal"),
        "operationalization_responsibility": operationalization_responsibility,
        "finding_responsibility": finding_responsibility,
        "finding_openness": finding_openness,
        "fixed_o_dimensions": _string_list(
            condition.get("fixed_o_dimensions", condition.get("fixed_dimensions", ())),
            "fixed_o_dimensions",
        ),
        "unresolved_o_dimensions": _string_list(
            condition.get(
                "unresolved_o_dimensions", condition.get("unresolved_dimensions", ())
            ),
            "unresolved_o_dimensions",
        ),
        "finding_requirements": [
            dict(item) for item in finding_requirements if isinstance(item, Mapping)
        ],
        "authored_operationalizations": authored,
        "materialization_route": route,
    }


def build_scientific_review_packet(
    *,
    dataset_id: str,
    family_id: str,
    concept_id: str,
    scientific_target: str,
    dataset_context: Mapping[str, Any],
    observable_semantics: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    evidence_resolution: FamilyScientificEvidenceResolution | Mapping[str, Any],
) -> dict[str, Any]:
    """Build the only model-visible Flow Expert packet, failing closed."""

    dataset = _nonempty(dataset_id, "dataset_id")
    family = _nonempty(family_id, "family_id")
    concept = _nonempty(concept_id, "concept_id")
    target = _nonempty(scientific_target, "scientific_target")
    _assert_firewall(dataset_context, "dataset_context")
    _assert_firewall(observable_semantics, "observable_semantics")
    try:
        visible_context = CaseContext.model_validate(dataset_context).model_dump(mode="json")
    except Exception as exc:
        raise ScientificExpertReviewError(f"dataset_context is invalid: {exc}") from exc
    resolution = (
        evidence_resolution
        if isinstance(evidence_resolution, FamilyScientificEvidenceResolution)
        else FamilyScientificEvidenceResolution.from_mapping(evidence_resolution)
    )
    identity_matches = (
        resolution.dataset_id == dataset
        and resolution.family_id == family
        and resolution.concept_id == concept
        and resolution.family_binding_status == "PASS"
        and not resolution.mismatch_claim_ids
        and not resolution.dataset_mismatch_evidence_ids
    )
    declared_provenance_closes = not (
        resolution.link_errors
        or resolution.duplicate_source_ids
        or resolution.duplicate_evidence_ids
        or resolution.duplicate_claim_ids
        or any(
            item.resolution_status != "RESOLVED"
            for item in resolution.provenance_resolutions
        )
    )
    if not identity_matches or not declared_provenance_closes:
        raise ScientificExpertReviewError(
            f"{resolution.reason_code}: evidence is not bound to {dataset}/{family}/{concept}"
        )
    visible_evidence = {
        "source_records": [
            item.model_dump(mode="json") for item in resolution.source_records
        ],
        "evidence_records": [
            item.model_dump(mode="json") for item in resolution.evidence_records
        ],
        "claim_support_links": [
            item.to_dict() for item in resolution.claim_support_links
        ],
    }
    condition_rows = [_condition_view(item, dataset) for item in conditions]
    condition_ids = [item["condition"] for item in condition_rows]
    if not condition_rows:
        raise ScientificExpertReviewError("at least one condition is required")
    if len(condition_ids) != len(set(condition_ids)):
        raise ScientificExpertReviewError("condition values must be unique")

    scientific_claims = [
        {
            "claim_id": claim.claim_id,
            "claim_type": claim.claim_type,
            "statement": claim.statement,
            "critical": claim.critical,
            "evidence_ids": sorted(
                {
                    link.evidence_record_id
                    for link in resolution.claim_support_links
                    if link.claim_id == claim.claim_id
                }
            ),
            "source_ids": sorted(
                {
                    link.source_id
                    for link in resolution.claim_support_links
                    if link.claim_id == claim.claim_id
                }
            ),
        }
        for claim in resolution.scientific_support_claims
    ]
    packet = {
        "scientific_review_packet_version": SCIENTIFIC_REVIEW_PACKET_VERSION,
        "review_input_status": (
            "EVIDENCE_COMPLETE"
            if not resolution.unresolved_claim_ids
            else "EVIDENCE_INCOMPLETE"
        ),
        "dataset_identity": {"dataset_id": dataset},
        "family_identity": {"family_id": family, "concept_id": concept},
        "scientific_target": target,
        "dataset_context": visible_context,
        "observable_semantics": dict(observable_semantics),
        "conditions": condition_rows,
        "scientific_claims": scientific_claims,
        "evidence_records": visible_evidence["evidence_records"],
        "evidence_family_bindings": [
            {
                "evidence_id": item.evidence_id,
                "dataset_id": resolution.dataset_id,
                "family_id": resolution.family_id,
                "concept_id": resolution.concept_id,
            }
            for item in resolution.evidence_records
        ],
        "source_records": visible_evidence["source_records"],
        "claim_support_links": visible_evidence["claim_support_links"],
        "provenance_summary": {
            "family_binding_status": resolution.family_binding_status,
            "bound_identity": {
                "dataset_id": resolution.dataset_id,
                "family_id": resolution.family_id,
                "concept_id": resolution.concept_id,
            },
            "evidence_provenance_closure": "PASS",
            "claim_evidence_sufficiency": (
                "COMPLETE" if not resolution.unresolved_claim_ids else "INCOMPLETE"
            ),
            "unresolved_claim_ids": list(resolution.unresolved_claim_ids),
            "visible_claim_count": len(scientific_claims),
            "evidence_count": len(resolution.evidence_records),
            "source_count": len(resolution.source_records),
        },
        "runtime_contract": dict(SCIENTIFIC_REVIEW_RUNTIME_CONTRACT),
    }
    _assert_firewall(packet)
    validation = validate_scientific_review_packet(packet)
    if validation["status"] != "PASS":
        raise ScientificExpertReviewError(
            "invalid scientific review packet: " + ", ".join(validation["reason_codes"])
        )
    return packet


def validate_scientific_review_packet(packet: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Validate identity, firewall, family binding, and provenance before a call."""

    errors: list[dict[str, str]] = []
    if not isinstance(packet, Mapping):
        return {
            "status": "INVALID",
            "reason_codes": ["PACKET_NOT_OBJECT"],
            "errors": [{"code": "PACKET_NOT_OBJECT", "field": "packet"}],
        }
    try:
        _assert_firewall(packet)
    except ScientificExpertReviewError as exc:
        errors.append({"code": "FIREWALL_VIOLATION", "field": str(exc)})
    unexpected_packet_fields = _unexpected_fields(packet, _PACKET_FIELDS)
    if unexpected_packet_fields:
        errors.append(
            {
                "code": "UNKNOWN_PACKET_FIELD",
                "field": ",".join(unexpected_packet_fields),
            }
        )
    if packet.get("scientific_review_packet_version") != SCIENTIFIC_REVIEW_PACKET_VERSION:
        errors.append({"code": "PACKET_VERSION_INVALID", "field": "scientific_review_packet_version"})
    if packet.get("runtime_contract") != SCIENTIFIC_REVIEW_RUNTIME_CONTRACT:
        errors.append({"code": "RUNTIME_CONTRACT_INVALID", "field": "runtime_contract"})
    if packet.get("review_input_status") not in {"EVIDENCE_COMPLETE", "EVIDENCE_INCOMPLETE"}:
        errors.append({"code": "REVIEW_INPUT_STATUS_INVALID", "field": "review_input_status"})
    dataset_identity = packet.get("dataset_identity")
    family_identity = packet.get("family_identity")
    if isinstance(dataset_identity, Mapping) and set(dataset_identity) != {"dataset_id"}:
        errors.append({"code": "IDENTITY_SCHEMA_INVALID", "field": "dataset_identity"})
    if isinstance(family_identity, Mapping) and set(family_identity) != {"family_id", "concept_id"}:
        errors.append({"code": "IDENTITY_SCHEMA_INVALID", "field": "family_identity"})
    dataset_id = str(dataset_identity.get("dataset_id", "")).strip() if isinstance(dataset_identity, Mapping) else ""
    family_id = str(family_identity.get("family_id", "")).strip() if isinstance(family_identity, Mapping) else ""
    concept_id = str(family_identity.get("concept_id", "")).strip() if isinstance(family_identity, Mapping) else ""
    if not dataset_id or not family_id or not concept_id:
        errors.append({"code": "REVIEW_IDENTITY_INCOMPLETE", "field": "dataset_identity/family_identity"})
    if not isinstance(packet.get("scientific_target"), str) or not str(
        packet.get("scientific_target", "")
    ).strip():
        errors.append({"code": "SCIENTIFIC_TARGET_INVALID", "field": "scientific_target"})
    dataset_context = packet.get("dataset_context")
    if not isinstance(dataset_context, Mapping):
        errors.append({"code": "DATASET_CONTEXT_INVALID", "field": "dataset_context"})
    else:
        try:
            CaseContext.model_validate(dataset_context)
        except Exception:
            errors.append({"code": "DATASET_CONTEXT_INVALID", "field": "dataset_context"})
    if not isinstance(packet.get("observable_semantics"), Mapping):
        errors.append({"code": "OBSERVABLE_SEMANTICS_INVALID", "field": "observable_semantics"})
    provenance = packet.get("provenance_summary")
    if isinstance(provenance, Mapping):
        unexpected = _unexpected_fields(provenance, _PROVENANCE_FIELDS)
        if unexpected:
            errors.append(
                {"code": "UNKNOWN_FIELD", "field": f"provenance_summary:{','.join(unexpected)}"}
            )
    bound = provenance.get("bound_identity") if isinstance(provenance, Mapping) else None
    if isinstance(bound, Mapping) and set(bound) != {
        "dataset_id",
        "family_id",
        "concept_id",
    }:
        errors.append(
            {"code": "IDENTITY_SCHEMA_INVALID", "field": "provenance_summary.bound_identity"}
        )
    if not isinstance(bound, Mapping) or (
        str(bound.get("dataset_id", "")) != dataset_id
        or str(bound.get("family_id", "")) != family_id
        or str(bound.get("concept_id", "")) != concept_id
    ):
        errors.append({"code": "EVIDENCE_FAMILY_MISMATCH", "field": "provenance_summary.bound_identity"})

    conditions = packet.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        errors.append({"code": "CONDITIONS_INVALID", "field": "conditions"})
        conditions = []
    condition_ids: list[str] = []
    for index, condition in enumerate(conditions):
        if not isinstance(condition, Mapping):
            errors.append({"code": "CONDITION_NOT_OBJECT", "field": f"conditions[{index}]"})
            continue
        condition_id = str(condition.get("condition", "")).strip()
        condition_ids.append(condition_id)
        required = (
            "case_id", "question", "scientific_target", "finding_goal",
            "operationalization_responsibility", "finding_responsibility", "finding_openness",
            "fixed_o_dimensions", "unresolved_o_dimensions", "finding_requirements",
            "authored_operationalizations", "materialization_route",
        )
        if not condition_id or any(field not in condition for field in required):
            errors.append({"code": "CONDITION_CONTRACT_INCOMPLETE", "field": f"conditions[{index}]"})
        if condition_id not in {"O1-F1", "O2-F1", "O3-F1", "O1-F2"}:
            errors.append(
                {"code": "CONDITION_ENUM_INVALID", "field": f"conditions[{index}].condition"}
            )
        unexpected = _unexpected_fields(condition, _CONDITION_FIELDS)
        if unexpected:
            errors.append(
                {
                    "code": "UNKNOWN_FIELD",
                    "field": f"conditions[{index}]:{','.join(unexpected)}",
                }
            )
        for name in (
            "case_id",
            "question",
            "scientific_target",
            "finding_goal",
            "operationalization_responsibility",
            "finding_responsibility",
            "finding_openness",
        ):
            if not isinstance(condition.get(name), str) or not str(condition.get(name, "")).strip():
                errors.append(
                    {"code": "CONDITION_VALUE_INVALID", "field": f"conditions[{index}].{name}"}
                )
        controlled_values = {
            "operationalization_responsibility": {
                "user_specified",
                "partially_specified",
                "model_selected",
            },
            "finding_responsibility": {"F1", "F2"},
            "finding_openness": {"bounded", "open"},
        }
        for name, allowed in controlled_values.items():
            if condition.get(name) not in allowed:
                errors.append(
                    {"code": "CONDITION_ENUM_INVALID", "field": f"conditions[{index}].{name}"}
                )
        fixed = _validate_string_sequence(
            condition.get("fixed_o_dimensions"),
            field=f"conditions[{index}].fixed_o_dimensions",
            errors=errors,
        )
        unresolved = _validate_string_sequence(
            condition.get("unresolved_o_dimensions"),
            field=f"conditions[{index}].unresolved_o_dimensions",
            errors=errors,
        )
        if set(fixed) & set(unresolved):
            errors.append(
                {"code": "O_DIMENSION_OVERLAP", "field": f"conditions[{index}]"}
            )
        if any(
            dimension not in _OPERATIONALIZATION_DIMENSIONS
            for dimension in (*fixed, *unresolved)
        ):
            errors.append(
                {
                    "code": "O_DIMENSION_ENUM_INVALID",
                    "field": f"conditions[{index}]",
                }
            )
        finding_requirements = condition.get("finding_requirements")
        if not isinstance(finding_requirements, list) or any(
            not isinstance(item, Mapping) for item in finding_requirements
        ):
            errors.append(
                {
                    "code": "OBJECT_LIST_REQUIRED",
                    "field": f"conditions[{index}].finding_requirements",
                }
            )
        else:
            for requirement_index, requirement in enumerate(finding_requirements):
                unexpected = _unexpected_fields(
                    requirement, _FINDING_REQUIREMENT_FIELDS
                )
                if unexpected:
                    errors.append(
                        {
                            "code": "UNKNOWN_FIELD",
                            "field": (
                                f"conditions[{index}].finding_requirements"
                                f"[{requirement_index}]:{','.join(unexpected)}"
                            ),
                        }
                    )
                requirement_field = (
                    f"conditions[{index}].finding_requirements[{requirement_index}]"
                )
                requirement_type = requirement.get("requirement_type")
                if requirement_type not in {None, "FAMILY_ROLE_CONTRACT"}:
                    errors.append(
                        {"code": "FINDING_REQUIREMENT_INVALID", "field": requirement_field}
                    )
                elif requirement_type == "FAMILY_ROLE_CONTRACT":
                    if not isinstance(requirement.get("scientific_entity_type"), str) or not str(
                        requirement.get("scientific_entity_type", "")
                    ).strip():
                        errors.append(
                            {"code": "FINDING_REQUIREMENT_INVALID", "field": requirement_field}
                        )
                    mandatory = requirement.get("mandatory_roles")
                    alternatives = requirement.get("alternative_role_groups", [])
                    mandatory_valid = (
                        isinstance(mandatory, list)
                        and all(
                            isinstance(role, str) and bool(role.strip())
                            for role in mandatory
                        )
                        and len(mandatory) == len(set(mandatory))
                    )
                    alternatives_valid = isinstance(alternatives, list)
                    core_sets = requirement.get("adequate_core_sets", [])
                    core_sets_valid = isinstance(core_sets, list) and all(
                        isinstance(group, list) and bool(group)
                        and all(isinstance(role, str) and bool(role.strip()) for role in group)
                        and len(group) == len(set(group)) for group in core_sets
                    )
                    if not mandatory_valid or not alternatives_valid or not core_sets_valid or not (mandatory or alternatives or core_sets):
                        errors.append(
                            {"code": "FINDING_REQUIREMENT_INVALID", "field": requirement_field}
                        )
                    if isinstance(alternatives, list):
                        for group_index, group in enumerate(alternatives):
                            group_field = (
                                f"{requirement_field}.alternative_role_groups[{group_index}]"
                            )
                            if not isinstance(group, Mapping) or set(group) != set(
                                _ALTERNATIVE_ROLE_GROUP_FIELDS
                            ):
                                errors.append(
                                    {"code": "FINDING_REQUIREMENT_INVALID", "field": group_field}
                                )
                                continue
                            roles = group.get("roles")
                            minimum = group.get("min_required")
                            roles_valid = (
                                isinstance(roles, list)
                                and bool(roles)
                                and all(
                                    isinstance(role, str) and bool(role.strip())
                                    for role in roles
                                )
                                and len(roles) == len(set(roles))
                            )
                            if (
                                not isinstance(group.get("group_id"), str)
                                or not str(group.get("group_id", "")).strip()
                                or not isinstance(minimum, int)
                                or isinstance(minimum, bool)
                                or not roles_valid
                                or minimum < 1
                                or minimum > len(roles or ())
                            ):
                                errors.append(
                                    {"code": "FINDING_REQUIREMENT_INVALID", "field": group_field}
                                )
                elif not (
                    isinstance(requirement.get("category"), str)
                    and requirement.get("category") in _FINDING_REQUIREMENT_CATEGORIES
                    and isinstance(requirement.get("statement"), str)
                    and str(requirement.get("statement", "")).strip()
                ):
                    errors.append(
                        {"code": "FINDING_REQUIREMENT_INVALID", "field": requirement_field}
                    )

        authored = condition.get("authored_operationalizations")
        if not isinstance(authored, list) or any(
            not isinstance(item, Mapping) for item in authored
        ):
            errors.append(
                {
                    "code": "OBJECT_LIST_REQUIRED",
                    "field": f"conditions[{index}].authored_operationalizations",
                }
            )
        else:
            if not authored:
                errors.append(
                    {
                        "code": "AUTHORED_OPERATIONALIZATIONS_EMPTY",
                        "field": f"conditions[{index}].authored_operationalizations",
                    }
                )
            candidate_indices: list[int] = []
            candidate_signatures: list[tuple[tuple[str, str, str], ...]] = []
            principal_dimensions = set(fixed) | set(unresolved)
            for candidate_offset, candidate in enumerate(authored):
                candidate_field = (
                    f"conditions[{index}].authored_operationalizations"
                    f"[{candidate_offset}]"
                )
                unexpected = _unexpected_fields(
                    candidate, _AUTHORED_OPERATIONALIZATION_FIELDS
                )
                if unexpected:
                    errors.append(
                        {
                            "code": "UNKNOWN_FIELD",
                            "field": f"{candidate_field}:{','.join(unexpected)}",
                        }
                    )
                if candidate.get("case_id") != condition.get("case_id"):
                    errors.append(
                        {
                            "code": "OPERATIONALIZATION_CASE_BINDING_MISMATCH",
                            "field": candidate_field,
                        }
                    )
                candidate_index = candidate.get("candidate_index")
                if not isinstance(candidate_index, int) or isinstance(
                    candidate_index, bool
                ) or candidate_index < 1:
                    errors.append(
                        {"code": "CANDIDATE_INDEX_INVALID", "field": candidate_field}
                    )
                else:
                    candidate_indices.append(candidate_index)
                if candidate.get("decision_status") not in {
                    "AUTHORED_EXECUTABLE_CANDIDATE",
                    "PARTIAL_AUTHORED_O_ONLY",
                }:
                    errors.append(
                        {"code": "DECISION_STATUS_INVALID", "field": candidate_field}
                    )
                if candidate.get("scientific_result_values_visible") is not False:
                    errors.append(
                        {"code": "SCIENTIFIC_RESULT_VISIBILITY_INVALID", "field": candidate_field}
                    )
                decisions = candidate.get("decisions")
                if not isinstance(decisions, list) or any(
                    not isinstance(item, Mapping) for item in decisions
                ):
                    errors.append(
                        {"code": "OBJECT_LIST_REQUIRED", "field": f"{candidate_field}.decisions"}
                    )
                    continue
                decision_dimensions: list[str] = []
                for decision_offset, decision in enumerate(decisions):
                    decision_field = f"{candidate_field}.decisions[{decision_offset}]"
                    unexpected = _unexpected_fields(
                        decision, _OPERATIONALIZATION_DECISION_FIELDS
                    )
                    if unexpected:
                        errors.append(
                            {
                                "code": "UNKNOWN_FIELD",
                                "field": f"{decision_field}:{','.join(unexpected)}",
                            }
                        )
                    for decision_key in _OPERATIONALIZATION_DECISION_FIELDS:
                        if not isinstance(decision.get(decision_key), str) or not str(
                            decision.get(decision_key, "")
                        ).strip():
                            errors.append(
                                {"code": "O_DECISION_INVALID", "field": f"{decision_field}.{decision_key}"}
                            )
                    decision_dimensions.append(str(decision.get("dimension_id", "")))
                if len(decision_dimensions) != len(set(decision_dimensions)):
                    errors.append(
                        {"code": "DUPLICATE_O_DIMENSION", "field": f"{candidate_field}.decisions"}
                    )
                expected_dimensions = (
                    set(fixed)
                    if candidate.get("decision_status") == "PARTIAL_AUTHORED_O_ONLY"
                    else principal_dimensions
                )
                if set(decision_dimensions) != expected_dimensions:
                    errors.append(
                        {"code": "O_DIMENSION_COVERAGE_INVALID", "field": f"{candidate_field}.decisions"}
                    )
                candidate_signatures.append(
                    tuple(
                        sorted(
                            (
                                str(decision.get("dimension_id", "")),
                                str(decision.get("canonical_id", "")),
                                str(decision.get("description", "")),
                            )
                            for decision in decisions
                        )
                    )
                )
            if len(candidate_indices) != len(set(candidate_indices)):
                errors.append(
                    {
                        "code": "DUPLICATE_CANDIDATE_INDEX",
                        "field": f"conditions[{index}].authored_operationalizations",
                    }
                )
            if sorted(candidate_indices) != list(range(1, len(authored) + 1)):
                errors.append(
                    {
                        "code": "CANDIDATE_INDEX_SEQUENCE_INVALID",
                        "field": f"conditions[{index}].authored_operationalizations",
                    }
                )
            if len(candidate_signatures) != len(set(candidate_signatures)):
                errors.append(
                    {
                        "code": "DUPLICATE_AUTHORED_OPERATIONALIZATION",
                        "field": f"conditions[{index}].authored_operationalizations",
                    }
                )

        route = condition.get("materialization_route")
        if not isinstance(route, Mapping):
            errors.append(
                {"code": "OBJECT_REQUIRED", "field": f"conditions[{index}].materialization_route"}
            )
        else:
            route_field = f"conditions[{index}].materialization_route"
            unexpected = _unexpected_fields(route, _MATERIALIZATION_ROUTE_FIELDS)
            if unexpected:
                errors.append(
                    {
                        "code": "UNKNOWN_FIELD",
                        "field": f"{route_field}:{','.join(unexpected)}",
                    }
                )
            if route.get("execution_status") not in {
                "MATERIALIZED_RESULTS_WITHHELD",
                "DETERMINISTIC_ROUTE_DECLARED_AVAILABLE",
                "DETERMINISTIC_ROUTE_NOT_DECLARED",
            }:
                errors.append({"code": "EXECUTION_STATUS_INVALID", "field": route_field})
            handler_id = route.get("handler_id")
            if not isinstance(handler_id, str):
                errors.append({"code": "HANDLER_ID_INVALID", "field": route_field})
            elif route.get("execution_status") in {
                "MATERIALIZED_RESULTS_WITHHELD",
                "DETERMINISTIC_ROUTE_DECLARED_AVAILABLE",
            } and not handler_id.strip():
                errors.append({"code": "HANDLER_ID_INVALID", "field": route_field})
            elif (
                route.get("execution_status") == "DETERMINISTIC_ROUTE_NOT_DECLARED"
                and handler_id.strip()
            ):
                errors.append({"code": "HANDLER_ID_INVALID", "field": route_field})
            expected_effective_condition = (
                "O1-F1" if condition_id == "O1-F2" else condition_id
            )
            if route.get("effective_o_condition") != expected_effective_condition:
                errors.append({"code": "EFFECTIVE_CONDITION_MISMATCH", "field": route_field})
            branch_count = route.get("branch_count")
            if not isinstance(branch_count, int) or isinstance(branch_count, bool) or branch_count < 0:
                errors.append({"code": "BRANCH_COUNT_INVALID", "field": route_field})
            raw_branch_statuses = route.get("branch_statuses")
            if not isinstance(raw_branch_statuses, list) or any(
                not isinstance(item, str) or not item.strip()
                for item in raw_branch_statuses
            ):
                errors.append(
                    {
                        "code": "STRING_LIST_REQUIRED",
                        "field": f"{route_field}.branch_statuses",
                    }
                )
                branch_statuses: list[str] = []
            else:
                branch_statuses = [item.strip() for item in raw_branch_statuses]
            if isinstance(branch_count, int) and not isinstance(branch_count, bool):
                if branch_count != len(branch_statuses) or branch_count != len(authored or []):
                    errors.append({"code": "MATERIALIZATION_COUNT_MISMATCH", "field": route_field})
            expected_branch_status = {
                "MATERIALIZED_RESULTS_WITHHELD": "MATERIALIZED_RESULT_WITHHELD",
                "DETERMINISTIC_ROUTE_DECLARED_AVAILABLE": "AUTHORED_O_ROUTE_DECLARED",
                "DETERMINISTIC_ROUTE_NOT_DECLARED": "AUTHORED_O_ROUTE_NOT_DECLARED",
            }.get(str(route.get("execution_status", "")))
            if expected_branch_status is not None and any(
                status != expected_branch_status for status in branch_statuses
            ):
                errors.append(
                    {"code": "BRANCH_STATUS_INVALID", "field": f"{route_field}.branch_statuses"}
                )
            if route.get("scientific_result_values_visible") is not False:
                errors.append(
                    {"code": "SCIENTIFIC_RESULT_VISIBILITY_INVALID", "field": route_field}
                )
    if len(condition_ids) != len(set(condition_ids)):
        errors.append({"code": "DUPLICATE_CONDITION", "field": "conditions"})

    sources = packet.get("source_records")
    evidence = packet.get("evidence_records")
    evidence_bindings = packet.get("evidence_family_bindings")
    claims = packet.get("scientific_claims")
    links = packet.get("claim_support_links")
    if not all(
        isinstance(value, list)
        for value in (sources, evidence, evidence_bindings, claims, links)
    ):
        errors.append({"code": "PROVENANCE_COLLECTION_INVALID", "field": "scientific provenance collections"})
        sources, evidence, evidence_bindings, claims, links = [], [], [], [], []
    source_ids = [str(item.get("source_id", "")) for item in sources if isinstance(item, Mapping)]
    evidence_ids = [str(item.get("evidence_id", "")) for item in evidence if isinstance(item, Mapping)]
    claim_ids = [str(item.get("claim_id", "")) for item in claims if isinstance(item, Mapping)]
    for name, values in (("source", source_ids), ("evidence", evidence_ids), ("claim", claim_ids)):
        if any(not value for value in values) or len(values) != len(set(values)):
            errors.append({"code": "PROVENANCE_ID_INVALID", "field": f"{name}_ids"})
    source_set, evidence_set, claim_set = set(source_ids), set(evidence_ids), set(claim_ids)
    bound_evidence_ids: list[str] = []
    for index, binding in enumerate(evidence_bindings):
        binding_field = f"evidence_family_bindings[{index}]"
        if not isinstance(binding, Mapping):
            errors.append({"code": "EVIDENCE_FAMILY_MISMATCH", "field": binding_field})
            continue
        if set(binding) != set(_EVIDENCE_FAMILY_BINDING_FIELDS):
            errors.append({"code": "EVIDENCE_FAMILY_MISMATCH", "field": binding_field})
        evidence_id = str(binding.get("evidence_id", "")).strip()
        bound_evidence_ids.append(evidence_id)
        if (
            not evidence_id
            or str(binding.get("dataset_id", "")) != dataset_id
            or str(binding.get("family_id", "")) != family_id
            or str(binding.get("concept_id", "")) != concept_id
        ):
            errors.append({"code": "EVIDENCE_FAMILY_MISMATCH", "field": binding_field})
    if (
        len(bound_evidence_ids) != len(set(bound_evidence_ids))
        or set(bound_evidence_ids) != evidence_set
    ):
        errors.append(
            {"code": "EVIDENCE_FAMILY_MISMATCH", "field": "evidence_family_bindings"}
        )
    for index, record in enumerate(sources):
        if not isinstance(record, Mapping):
            errors.append({"code": "SOURCE_NOT_OBJECT", "field": f"source_records[{index}]"})
            continue
        unexpected = _unexpected_fields(record, _SOURCE_FIELDS)
        if unexpected:
            errors.append(
                {"code": "UNKNOWN_FIELD", "field": f"source_records[{index}]:{','.join(unexpected)}"}
            )
        try:
            SourceRecord.model_validate(record)
        except Exception:
            errors.append(
                {"code": "SOURCE_RECORD_INVALID", "field": f"source_records[{index}]"}
            )
    evidence_source: dict[str, str] = {}
    for index, record in enumerate(evidence):
        if not isinstance(record, Mapping):
            errors.append({"code": "EVIDENCE_NOT_OBJECT", "field": f"evidence_records[{index}]"})
            continue
        unexpected = _unexpected_fields(record, _EVIDENCE_FIELDS)
        if unexpected:
            errors.append(
                {"code": "UNKNOWN_FIELD", "field": f"evidence_records[{index}]:{','.join(unexpected)}"}
            )
        try:
            EvidenceRecord.model_validate(record)
        except Exception:
            errors.append(
                {"code": "EVIDENCE_RECORD_INVALID", "field": f"evidence_records[{index}]"}
            )
        evidence_id = str(record.get("evidence_id", ""))
        source_id = str(record.get("source_id", ""))
        evidence_source[evidence_id] = source_id
        if str(record.get("dataset_id", "")) != dataset_id:
            errors.append({"code": "EVIDENCE_DATASET_MISMATCH", "field": evidence_id})
        if source_id not in source_set:
            errors.append({"code": "EVIDENCE_SOURCE_UNKNOWN", "field": evidence_id})
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            errors.append({"code": "CLAIM_NOT_OBJECT", "field": f"scientific_claims[{index}]"})
            continue
        unexpected = _unexpected_fields(claim, _CLAIM_FIELDS)
        if unexpected:
            errors.append(
                {"code": "UNKNOWN_FIELD", "field": f"scientific_claims[{index}]:{','.join(unexpected)}"}
            )
        if (
            not isinstance(claim.get("claim_id"), str)
            or not str(claim.get("claim_id", "")).strip()
            or claim.get("claim_type") not in SUPPORT_CLAIM_TYPES
            or not isinstance(claim.get("statement"), str)
            or not str(claim.get("statement", "")).strip()
            or not isinstance(claim.get("critical"), bool)
        ):
            errors.append(
                {"code": "SCIENTIFIC_CLAIM_INVALID", "field": f"scientific_claims[{index}]"}
            )
        for field, known in (("evidence_ids", evidence_set), ("source_ids", source_set)):
            citations = _validate_string_sequence(
                claim.get(field), field=f"{claim.get('claim_id')}.{field}", errors=errors
            )
            if any(item not in known for item in citations):
                errors.append({"code": "CLAIM_PROVENANCE_UNKNOWN", "field": f"{claim.get('claim_id')}.{field}"})
    link_tuples: list[tuple[str, str, str]] = []
    links_by_claim: dict[str, list[tuple[str, str]]] = {}
    for index, link in enumerate(links):
        if not isinstance(link, Mapping):
            errors.append({"code": "LINK_NOT_OBJECT", "field": f"claim_support_links[{index}]"})
            continue
        unexpected = _unexpected_fields(link, _LINK_FIELDS)
        if unexpected:
            errors.append(
                {"code": "UNKNOWN_FIELD", "field": f"claim_support_links[{index}]:{','.join(unexpected)}"}
            )
        if (
            link.get("support_relation") not in SUPPORT_RELATIONS
            or not isinstance(link.get("supported_statement"), str)
            or not str(link.get("supported_statement", "")).strip()
        ):
            errors.append(
                {"code": "CLAIM_SUPPORT_LINK_INVALID", "field": f"claim_support_links[{index}]"}
            )
        claim_id = str(link.get("claim_id", ""))
        evidence_id = str(link.get("evidence_record_id", ""))
        source_id = str(link.get("source_id", ""))
        link_tuples.append((claim_id, evidence_id, source_id))
        links_by_claim.setdefault(claim_id, []).append((evidence_id, source_id))
        if claim_id not in claim_set or evidence_id not in evidence_set or source_id not in source_set:
            errors.append({"code": "LINK_PROVENANCE_UNKNOWN", "field": f"claim_support_links[{index}]"})
        if evidence_source.get(evidence_id) != source_id:
            errors.append({"code": "LINK_EVIDENCE_SOURCE_MISMATCH", "field": f"claim_support_links[{index}]"})
    if len(link_tuples) != len(set(link_tuples)):
        errors.append({"code": "DUPLICATE_CLAIM_SUPPORT_LINK", "field": "claim_support_links"})
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        claim_id = str(claim.get("claim_id", ""))
        linked = links_by_claim.get(claim_id, [])
        linked_evidence = {item[0] for item in linked}
        linked_sources = {item[1] for item in linked}
        declared_evidence = {str(item) for item in claim.get("evidence_ids", ()) or ()}
        declared_sources = {str(item) for item in claim.get("source_ids", ()) or ()}
        if declared_evidence != linked_evidence or declared_sources != linked_sources:
            errors.append(
                {"code": "CLAIM_LINK_CLOSURE_MISMATCH", "field": claim_id}
            )
    if isinstance(provenance, Mapping):
        expected_counts = {
            "visible_claim_count": len(claims),
            "evidence_count": len(evidence),
            "source_count": len(sources),
        }
        if any(provenance.get(field) != count for field, count in expected_counts.items()):
            errors.append({"code": "PROVENANCE_COUNT_MISMATCH", "field": "provenance_summary"})
        if provenance.get("family_binding_status") != "PASS" or provenance.get("evidence_provenance_closure") != "PASS":
            errors.append({"code": "PROVENANCE_NOT_CLOSED", "field": "provenance_summary"})
        unresolved = provenance.get("unresolved_claim_ids")
        if not isinstance(unresolved, list) or any(str(item) not in claim_set for item in unresolved):
            errors.append({"code": "UNRESOLVED_CLAIM_IDS_INVALID", "field": "provenance_summary"})
            unresolved = []
        sufficiency = provenance.get("claim_evidence_sufficiency")
        expected_sufficiency = "INCOMPLETE" if unresolved else "COMPLETE"
        expected_input_status = (
            "EVIDENCE_INCOMPLETE" if unresolved else "EVIDENCE_COMPLETE"
        )
        if sufficiency not in {"COMPLETE", "INCOMPLETE"}:
            errors.append(
                {"code": "CLAIM_EVIDENCE_SUFFICIENCY_INVALID", "field": "provenance_summary"}
            )
        if sufficiency != expected_sufficiency:
            errors.append(
                {"code": "CLAIM_EVIDENCE_STATE_MISMATCH", "field": "provenance_summary"}
            )
        if packet.get("review_input_status") != expected_input_status:
            errors.append(
                {"code": "REVIEW_INPUT_STATE_MISMATCH", "field": "review_input_status"}
            )
    return {
        "status": "PASS" if not errors else "INVALID",
        "reason_codes": sorted({item["code"] for item in errors}),
        "errors": errors,
    }


def _citation_ids(value: Any, *, field: str, known: set[str], errors: list[dict[str, str]]) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append({"code": "INVALID_CITATION_LIST", "field": field})
        return []
    result = [item.strip() for item in value]
    if len(result) != len(set(result)):
        errors.append({"code": "DUPLICATE_CITATION_ID", "field": field})
    unknown = sorted(set(result) - known)
    if unknown:
        errors.append(
            {
                "code": "UNKNOWN_PROVENANCE_ID",
                "field": field,
                "ids": ",".join(unknown),
            }
        )
    return result


def validate_scientific_expert_output(
    output: Mapping[str, Any] | Any,
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact atomic observations and all cited provenance IDs."""

    expected_conditions = {
        str(item.get("condition"))
        for item in packet.get("conditions", ())
        if isinstance(item, Mapping)
    }
    known_evidence = {
        str(item.get("evidence_id"))
        for item in packet.get("evidence_records", ())
        if isinstance(item, Mapping)
    }
    known_sources = {
        str(item.get("source_id"))
        for item in packet.get("source_records", ())
        if isinstance(item, Mapping)
    }
    evidence_source = {
        str(item.get("evidence_id")): str(item.get("source_id"))
        for item in packet.get("evidence_records", ())
        if isinstance(item, Mapping)
    }
    errors: list[dict[str, str]] = []
    if not isinstance(output, Mapping):
        return {
            "status": "INCOMPLETE",
            "reason_codes": ["OUTPUT_NOT_OBJECT"],
            "errors": [{"code": "OUTPUT_NOT_OBJECT", "field": "output"}],
            "condition_reviews": [],
            "question_reviews": [],
            "provenance_validation_status": "NOT_EVALUATED",
        }
    try:
        _assert_firewall(output, "output")
    except ScientificExpertReviewError as exc:
        errors.append({"code": "OUTPUT_POLICY_FIELD_FORBIDDEN", "field": str(exc)})
    unexpected_output = _unexpected_fields(output, _OUTPUT_FIELDS)
    if unexpected_output:
        errors.append(
            {"code": "UNKNOWN_OUTPUT_FIELD", "field": ",".join(unexpected_output)}
        )
    raw_reviews = output.get("condition_reviews")
    if not isinstance(raw_reviews, list):
        raw_reviews = []
        errors.append({"code": "CONDITION_REVIEWS_MISSING", "field": "condition_reviews"})
    seen: set[str] = set()
    reviews: list[dict[str, Any]] = []
    for index, item in enumerate(raw_reviews):
        field = f"condition_reviews[{index}]"
        if not isinstance(item, Mapping):
            errors.append({"code": "CONDITION_REVIEW_NOT_OBJECT", "field": field})
            continue
        unexpected = _unexpected_fields(item, _CONDITION_REVIEW_FIELDS)
        if unexpected:
            errors.append(
                {"code": "UNKNOWN_OUTPUT_FIELD", "field": f"{field}:{','.join(unexpected)}"}
            )
        condition = str(item.get("condition", "")).strip()
        if not condition:
            errors.append({"code": "CONDITION_MISSING", "field": f"{field}.condition"})
            continue
        if condition in seen:
            errors.append({"code": "DUPLICATE_CONDITION", "field": condition})
            continue
        seen.add(condition)
        if condition not in expected_conditions:
            errors.append({"code": "UNKNOWN_CONDITION", "field": condition})
        row: dict[str, Any] = {"condition": condition}
        for name in ATOMIC_OBSERVATION_FIELDS:
            value = item.get(name)
            if not isinstance(value, bool):
                errors.append(
                    {
                        "code": "MISSING_OR_INVALID_ATOMIC_FIELD",
                        "field": f"{field}.{name}",
                    }
                )
            row[name] = value if isinstance(value, bool) else None
        recommendation = item.get("eligibility_recommendation")
        if recommendation not in ADVISORY_RECOMMENDATIONS:
            errors.append(
                {"code": "INVALID_ADVISORY_ENUM", "field": f"{field}.eligibility_recommendation"}
            )
        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append({"code": "RATIONALE_MISSING", "field": f"{field}.rationale"})
        ambiguities = item.get("scientific_ambiguities")
        if not isinstance(ambiguities, list) or any(
            not isinstance(value, str) or not value.strip() for value in ambiguities
        ):
            errors.append(
                {"code": "SCIENTIFIC_AMBIGUITIES_INVALID", "field": f"{field}.scientific_ambiguities"}
            )
        cited_evidence = _citation_ids(
            item.get("evidence_ids"),
            field=f"{field}.evidence_ids",
            known=known_evidence,
            errors=errors,
        )
        cited_sources = _citation_ids(
            item.get("source_ids"),
            field=f"{field}.source_ids",
            known=known_sources,
            errors=errors,
        )
        provenance_required = (
            recommendation == "ELIGIBLE"
            or bool(known_evidence)
            or bool(known_sources)
        )
        if provenance_required and (not cited_evidence or not cited_sources):
            errors.append(
                {"code": "REVIEW_PROVENANCE_REQUIRED", "field": field}
            )
        expected_cited_sources = {
            evidence_source[evidence_id]
            for evidence_id in cited_evidence
            if evidence_id in evidence_source
        }
        if expected_cited_sources != set(cited_sources):
            errors.append(
                {"code": "REVIEW_CITATION_LINK_MISMATCH", "field": field}
            )
        row.update(
            {
                "eligibility_recommendation": recommendation,
                "rationale": rationale if isinstance(rationale, str) else "",
                "finding_contract_condition_specific_rationale": str(
                    item.get("finding_contract_condition_specific_rationale", "")
                ),
                "scientific_ambiguities": list(ambiguities)
                if isinstance(ambiguities, list)
                else [],
                "evidence_ids": cited_evidence,
                "source_ids": cited_sources,
            }
        )
        reviews.append(row)
    missing = sorted(expected_conditions - seen)
    if missing:
        errors.append(
            {"code": "REQUIRED_CONDITIONS_MISSING", "field": ",".join(missing)}
        )

    question_rows: list[dict[str, Any]] = []
    raw_questions = output.get("question_reviews", [])
    if not isinstance(raw_questions, list):
        errors.append({"code": "QUESTION_REVIEWS_NOT_LIST", "field": "question_reviews"})
        raw_questions = []
    question_seen: set[str] = set()
    for index, item in enumerate(raw_questions):
        field = f"question_reviews[{index}]"
        if not isinstance(item, Mapping):
            errors.append({"code": "QUESTION_REVIEW_NOT_OBJECT", "field": field})
            continue
        unexpected = _unexpected_fields(item, _QUESTION_REVIEW_FIELDS)
        if unexpected:
            errors.append(
                {"code": "UNKNOWN_OUTPUT_FIELD", "field": f"{field}:{','.join(unexpected)}"}
            )
        condition = str(item.get("condition", "")).strip()
        if condition in question_seen:
            errors.append({"code": "DUPLICATE_QUESTION_CONDITION", "field": condition})
            continue
        question_seen.add(condition)
        if condition not in expected_conditions:
            errors.append({"code": "UNKNOWN_QUESTION_CONDITION", "field": condition})
        recommendation = item.get("recommendation")
        if recommendation not in QUESTION_RECOMMENDATIONS:
            errors.append({"code": "INVALID_QUESTION_ENUM", "field": f"{field}.recommendation"})
        question_rows.append(
            {
                "condition": condition,
                "recommendation": recommendation,
                "rationale": str(item.get("rationale", "")),
                "evidence_ids": _citation_ids(
                    item.get("evidence_ids"),
                    field=f"{field}.evidence_ids",
                    known=known_evidence,
                    errors=errors,
                ),
            }
        )
    reason_codes = sorted({item["code"] for item in errors})
    provenance_failed = any(
        item["code"]
        in {
            "DUPLICATE_CITATION_ID",
            "INVALID_CITATION_LIST",
            "REVIEW_CITATION_LINK_MISMATCH",
            "REVIEW_PROVENANCE_REQUIRED",
            "UNKNOWN_PROVENANCE_ID",
        }
        for item in errors
    )
    return {
        "status": "PASS" if not errors else "INCOMPLETE",
        "reason_codes": reason_codes,
        "errors": errors,
        "condition_reviews": reviews,
        "question_reviews": question_rows,
        "provenance_validation_status": "FAIL" if provenance_failed else "PASS",
    }


SCIENTIFIC_REVIEW_INSTRUCTION = (
    "Use only the supplied scientific claims, evidence records, source records, and authored "
    "condition descriptions. Do not request tools or external information. Return JSON with "
    "condition_reviews containing exactly one object for every supplied condition. Each object "
    "must contain condition; the nine required boolean atomic fields scientific_target_supported, "
    "fixed_o_supported, unresolved_o_is_legitimate_scientific_choice, finding_contract_supported, "
    "materialization_scientifically_meaningful, question_target_preserved, "
    "question_scope_preserved, fixed_o_question_faithful, and unresolved_o_exposed_by_question; "
    "eligibility_recommendation; rationale; scientific_ambiguities; evidence_ids; and source_ids. "
    "eligibility_recommendation must be exactly one of ELIGIBLE, NOT_ELIGIBLE, "
    "MORE_EVIDENCE_REQUIRED, or REVISE_QUESTION. Do not invent qualified variants. "
    "REVISE_TARGET and REVISE_OPERATIONALIZATION_SPACE are question_reviews enums only; "
    "never place either value in condition_reviews. When a condition has target or "
    "Operationalization-space drift, use REVISE_QUESTION in its condition review and, if "
    "useful, add the more specific recommendation in question_reviews. "
    "The recommendation is advisory only. Operationalization responsibility and Finding "
    "responsibility are independent contracts and must be judged separately. Judge "
    "fixed_o_supported only for fixed_o_dimensions; an "
    "unresolved O dimension must not count as an unsupported fixed choice. "
    "unresolved_o_exposed_by_question is true only when every listed unresolved dimension remains "
    "open in the question; it is false if the question fixes even one of them, and true when the "
    "unresolved_o_dimensions list is empty because no unresolved choice is concealed. For F1, the benchmark "
    "fixes the explicitly listed required Finding roles under the selected Effective O; that list must "
    "be complete, scientifically relevant, and faithful to the question demand, but it does not require "
    "a unique pre-operationalization numerical answer. For F2, Finding selection is open to the "
    "evaluated agent; the visible FAMILY_ROLE_CONTRACT specifies adequate-core scientific semantics "
    "rather than a fixed list of extra Findings. An absent, incomplete, or contradictory adequate-core "
    "contract does not become valid merely because the request is open. O1-F2 inherits exactly the "
    "fixed Operationalization of its paired O1-F1 condition, so an O1-F1 effective_o_condition in "
    "its materialization route is expected and is not an O mismatch. finding_contract_supported "
    "requires the declared F1/F2 responsibility, openness, Finding requirements, adequate-core "
    "semantics when applicable, and question demand to agree. Across conditions in the same "
    "packet, compare the scientific object induced by the authored Operationalizations. If two "
    "plausible choices answer different physical questions rather than operationalizing the same "
    "target, do not assume target invariance merely because both choices are listed; record the "
    "ambiguity and recommend target or Operationalization-space revision. Different valid O "
    "choices may yield different G(O), rankings, or selected features; those outcome differences "
    "are diagnostic-only and are not by themselves target drift. Judge materialization "
    "meaning from the visible authored operationalizations and materialization route, never from an "
    "invisible result. A scientifically plausible authored candidate alone is insufficient when the "
    "packet says the deterministic route is not declared. Question drift "
    "belongs in the four question fields and does not by itself make an executed O scientifically "
    "meaningless. Optional question_reviews recommendation must be exactly one of KEEP, "
    "REVISE_WORDING, REVISE_TARGET, REVISE_OPERATIONALIZATION_SPACE, or "
    "NOT_SCIENTIFICALLY_SUPPORTABLE. Every question_reviews object must contain condition, "
    "recommendation, a non-empty rationale, and evidence_ids as a JSON list; use an empty list "
    "when no evidence citation applies. Cite only evidence_ids and source_ids present in the packet. "
    "Do not make curator, SCQ, "
    "SRAC, Ground Truth, or benchmark policy decisions."
)

# Compatibility alias for callers/tests that previously inspected the module.
_REVIEW_INSTRUCTION = SCIENTIFIC_REVIEW_INSTRUCTION


def run_scientific_expert_review(
    repository_root: str | Path,
    packet: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run one canonical tool-free Flow Expert call and validate its output."""

    packet_validation = validate_scientific_review_packet(packet)
    if packet_validation["status"] != "PASS":
        return {
            "status": "INPUT_REJECTED",
            "invocation_status": "NOT_RUN",
            "call": None,
            "parsed_result": {},
            "validated_review": {
                "status": "NOT_EVALUATED",
                "reason_codes": packet_validation["reason_codes"],
                "condition_reviews": [],
                "question_reviews": [],
                "provenance_validation_status": "FAIL",
            },
            "input_validation": packet_validation,
            "execution_mode": "LIVE_MODEL_CALL",
            "live_model_calls": False,
            "proxy_only": True,
        }
    root = Path(repository_root).resolve()
    caller = LiveModelCaller(
        root,
        FLOW_SCIENTIFIC_REVIEWER_PROFILE,
        config_path=config_path,
        api_key=api_key,
        timeout=timeout,
        max_output_tokens=FLOW_SCIENTIFIC_REVIEWER_MAX_OUTPUT_TOKENS,
    )
    dataset_id = str(packet.get("dataset_identity", {}).get("dataset_id", ""))
    family_id = str(packet.get("family_identity", {}).get("family_id", ""))
    concept_id = str(packet.get("family_identity", {}).get("concept_id", ""))
    identity: dict[str, Any] = {
        "dataset_id": dataset_id,
        "family_id": family_id,
        "concept_id": concept_id,
        "review_index": "scientific-constraint-review",
    }
    if run_id is not None:
        identity["scientific_review_run_id"] = run_id
    call = caller.call(
        role="scientific_reviewer",
        visible_payload=packet,
        instruction=SCIENTIFIC_REVIEW_INSTRUCTION,
        identity=identity,
        tool_free=True,
    )
    parsed = call.get("parsed_result")
    if call.get("invocation_status") != "SUCCESS" or not isinstance(parsed, Mapping):
        return {
            "status": "FAILED",
            "invocation_status": call.get("invocation_status", "PROVIDER_ERROR"),
            "call": call,
            "parsed_result": parsed or {},
            "validated_review": {
                "status": "NOT_EVALUATED",
                "reason_codes": ["INVOCATION_NOT_SUCCESSFUL"],
                "condition_reviews": [],
                "question_reviews": [],
                "provenance_validation_status": "NOT_EVALUATED",
            },
            "execution_mode": "LIVE_MODEL_CALL",
            "live_model_calls": True,
            "proxy_only": True,
        }
    validated = validate_scientific_expert_output(parsed, packet)
    return {
        "status": "SUCCESS" if validated["status"] == "PASS" else "INCOMPLETE",
        "invocation_status": "SUCCESS",
        "call": call,
        "parsed_result": dict(parsed),
        "validated_review": validated,
        "execution_mode": "LIVE_MODEL_CALL",
        "live_model_calls": True,
        "proxy_only": True,
    }


__all__ = [
    "ADVISORY_RECOMMENDATIONS",
    "ATOMIC_OBSERVATION_FIELDS",
    "FLOW_SCIENTIFIC_REVIEWER_PROFILE",
    "FLOW_SCIENTIFIC_REVIEWER_MAX_OUTPUT_TOKENS",
    "QUESTION_RECOMMENDATIONS",
    "SCIENTIFIC_REVIEW_PACKET_VERSION",
    "SCIENTIFIC_REVIEW_CONTRACT_VERSION",
    "SCIENTIFIC_REVIEW_API_VERSION",
    "SCIENTIFIC_REVIEW_INSTRUCTION",
    "SCIENTIFIC_REVIEW_RUNTIME_CONTRACT",
    "ScientificExpertReviewError",
    "build_scientific_review_packet",
    "run_scientific_expert_review",
    "validate_scientific_expert_output",
    "validate_scientific_review_packet",
]


def review_constructed_families(
    repository_root, manifest_path, *, config_path=None, max_workers=4, timeout=240.0
):
    """Review explicit dataset-local families with the existing scientific expert.

    Only source evidence and method descriptions enter the review packet;
    computed values, validity labels and GT are withheld by its normal firewall.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .case_repository import load_case_record
    from .grounding import resolve_family_scientific_evidence
    from .scientific_run_snapshots import canonical_json_sha256
    from .review_lineage import live_review_call_matches
    from .agent_profile import load_agent_profile

    root = Path(repository_root).resolve()
    profile_hash = load_agent_profile(
        FLOW_SCIENTIFIC_REVIEWER_PROFILE, repository_root=root
    ).profile_sha256
    manifest_path = (root / manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    groups = {}
    for row in manifest["cases"]:
        if not str(row.get("origin", "")).startswith("FROZEN_REFERENCE"):
            groups.setdefault(row["family_id"], []).append(row)
    output = manifest_path.parent / "scientific_reviews"
    output.mkdir(exist_ok=True)

    def review(family_id, rows):
        first = load_case_record(root, rows[0]["case_id"])
        definition = first["family"]
        dataset_id = first["dataset_id"]
        construction = root / "datasets" / dataset_id / "construction"
        sources = json.loads((construction / "sources.json").read_text())["sources"]
        evidence = []
        for name in ["context_evidence.json", "operationalization_evidence.json"]:
            evidence.extend(json.loads((construction / name).read_text()))
        claims, links = [], []
        descriptions = [
            (
                "context",
                "DATASET_CONTEXT",
                "ctx_001",
                "src_001",
                first["case_context"].get("physical_setting", definition["scope"]),
            ),
            (
                "observable",
                "OBSERVABLE_SEMANTICS",
                "ctx_001",
                "src_001",
                "The target uses the inspected stored arrays and declared sample geometry; undeclared physical units and time histories are not inferred.",
            ),
            (
                "target",
                "SCIENTIFIC_TARGET_VALIDITY",
                "op_001",
                "src_002",
                definition["scientific_coherence"]
                + " The target is: "
                + definition["scientific_target"],
            ),
            (
                "operationalization",
                "OPERATIONALIZATION_RATIONALE",
                "op_001",
                "src_002",
                "The candidate clauses define explicit spatial statistics or geometry on the supplied data. Their scientific adequacy for this target is submitted for independent assessment.",
            ),
            (
                "same_target",
                "SAME_TARGET_RATIONALE",
                "op_001",
                "src_002",
                definition["o_role_coherence"],
            ),
            (
                "findings",
                "FINDING_VERIFIABILITY",
                "op_001",
                "src_002",
                "The numerical recipe computes the declared properties with explicit scalar or spatial tolerances. Numerical executability alone does not establish that they form an adequate scientific characterization.",
            ),
        ]
        for suffix, kind, eid, sid, statement in descriptions:
            cid = family_id + "_" + suffix
            claims.append(
                dict(
                    claim_id=cid,
                    claim_type=kind,
                    statement=statement,
                    support_status="UNKNOWN",
                    supporting_source_ids=[sid],
                    supporting_evidence_record_ids=[eid],
                    critical=True,
                )
            )
            links.append(
                dict(
                    claim_id=cid,
                    evidence_record_id=eid,
                    source_id=sid,
                    support_relation="CONTEXTUAL",
                    supported_statement=statement,
                )
            )
        resolution = resolve_family_scientific_evidence(
            dataset_id,
            family_id,
            definition.get("concept_id", family_id),
            sources,
            evidence,
            claims,
            links,
            evidence_family_id=family_id,
            evidence_concept_id=definition.get("concept_id", family_id),
        )
        conditions = []
        for row in rows:
            case = load_case_record(root, row["case_id"])
            metadata = case["metadata"]
            unresolved = metadata["unresolved_operationalization_dimensions"]
            fixed = [
                d for d in definition["principal_dimensions"] if d not in unresolved
            ]
            options = (
                list(definition["operations"].values())
                if unresolved
                else list(definition["operations"].values())[:1]
            )
            authored = [
                dict(
                    case_id=row["case_id"],
                    candidate_index=i + 1,
                    decision_status="AUTHORED_EXECUTABLE_CANDIDATE",
                    decisions=[
                        dict(dimension_id=d, canonical_id=text, description=text)
                        for d, text in op["clauses"].items()
                    ],
                    scientific_result_values_visible=False,
                )
                for i, op in enumerate(options)
            ]
            requirements = list(metadata["explicit_finding_requirements"])
            if row["condition"] == "O1-F2":
                requirements.append(
                    dict(
                        requirement_type="FAMILY_ROLE_CONTRACT",
                        **definition["finding_requirement_contract"],
                    )
                )
            conditions.append(
                dict(
                    condition=row["condition"],
                    case_id=row["case_id"],
                    question=case["case_input"]["scientific_question"],
                    scientific_target=metadata["scientific_target"],
                    finding_goal=metadata["finding_goal"],
                    metadata=metadata,
                    fixed_o_dimensions=fixed,
                    unresolved_o_dimensions=unresolved,
                    finding_requirements=requirements,
                    authored_operationalizations=authored,
                    materialization_route=dict(
                        execution_status="DETERMINISTIC_ROUTE_DECLARED_AVAILABLE",
                        handler_id="deterministic_case_materializer_v1",
                        effective_o_condition="O1-F1"
                        if row["condition"] == "O1-F2"
                        else row["condition"],
                        branch_count=len(authored),
                        branch_statuses=["AUTHORED_O_ROUTE_DECLARED"] * len(authored),
                        scientific_result_values_visible=False,
                    ),
                )
            )
        packet = build_scientific_review_packet(
            dataset_id=dataset_id,
            family_id=family_id,
            concept_id=definition.get("concept_id", family_id),
            scientific_target=definition["scientific_target"],
            dataset_context=first["case_context"],
            observable_semantics=first["case_input"]["flow_data"]["data_metadata"],
            conditions=conditions,
            evidence_resolution=resolution,
        )
        packet_path = output / (family_id + "_packet.json")
        packet_path.write_text(json.dumps(packet, indent=2) + "\n")
        result_path = output / (family_id + ".json")
        existing = json.loads(result_path.read_text()) if result_path.exists() else {}
        digest = canonical_json_sha256(packet)
        if (
            existing.get("input_packet_sha256") == digest
            and existing.get("review", {}).get("status") == "SUCCESS"
            and (
                live_review_call_matches(
                    existing["review"].get("call", {}),
                    packet,
                    SCIENTIFIC_REVIEW_INSTRUCTION,
                    profile_hash,
                )
                or (
                    len(existing["review"].get("condition_calls", []))
                    == len(packet["conditions"])
                    and all(
                        live_review_call_matches(
                            call.get("call", {}),
                            {**packet, "conditions": [condition]},
                            SCIENTIFIC_REVIEW_INSTRUCTION,
                            profile_hash,
                        )
                        for call, condition in zip(
                            existing["review"]["condition_calls"], packet["conditions"]
                        )
                    )
                )
            )
        ):
            result = existing["review"]
        else:
            # Keep each call bounded to one condition when a full-family call
            # is unavailable; combine only the validated real model outputs.
            condition_calls = []
            merged = {"condition_reviews": [], "question_reviews": []}
            for condition in packet["conditions"]:
                subpacket = {**packet, "conditions": [condition]}
                subpath = output / (
                    family_id
                    + "_"
                    + condition["condition"].lower().replace("-", "_")
                    + "_call.json"
                )
                subhash = canonical_json_sha256(subpacket)
                saved = json.loads(subpath.read_text()) if subpath.is_file() else {}
                call = saved.get("review", {})
                if (
                    saved.get("input_packet_sha256") != subhash
                    or call.get("status") != "SUCCESS"
                    or not live_review_call_matches(
                        call.get("call", {}),
                        subpacket,
                        SCIENTIFIC_REVIEW_INSTRUCTION,
                        profile_hash,
                    )
                ):
                    call = run_scientific_expert_review(
                        root, subpacket, config_path=config_path, timeout=timeout
                    )
                    subpath.write_text(
                        json.dumps(
                            {
                                "input_packet_sha256": subhash,
                                "packet": subpacket,
                                "review": call,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                condition_calls.append(call)
                for key in merged:
                    merged[key].extend(call.get("parsed_result", {}).get(key, []))
            invoked = all(
                call.get("invocation_status") == "SUCCESS" for call in condition_calls
            )
            validated = (
                validate_scientific_expert_output(merged, packet)
                if invoked
                else {
                    "status": "NOT_EVALUATED",
                    "reason_codes": ["INVOCATION_NOT_SUCCESSFUL"],
                }
            )
            result = {
                "status": "SUCCESS" if validated["status"] == "PASS" else "INCOMPLETE",
                "invocation_status": "SUCCESS" if invoked else "INCOMPLETE",
                "parsed_result": merged,
                "validated_review": validated,
                "condition_calls": condition_calls,
                "execution_mode": "LIVE_MODEL_CALLS_BY_CONDITION",
                "live_model_calls": True,
                "proxy_only": True,
            }
            result_path.write_text(
                json.dumps(dict(input_packet_sha256=digest, review=result), indent=2)
                + "\n"
            )
        return dict(
            family_id=family_id,
            invocation_status=result.get("invocation_status"),
            status=result.get("status"),
            validated_review=result.get("validated_review"),
            review_path=str(result_path.relative_to(root)),
        )

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(review, k, v): k for k, v in groups.items()}
        for future in as_completed(futures):
            fid = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = dict(family_id=fid, status="NOT_ESTABLISHED", error=str(exc))
            rows.append(row)
            summary = dict(
                family_count=len(groups),
                reviewed_count=len(rows),
                families=sorted(rows, key=lambda r: r["family_id"]),
                curator_confirmation=None,
                formal_release="NOT_ESTABLISHED_BY_PROXY_REVIEW",
            )
            (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                "scientific",
                len(rows),
                "/",
                len(groups),
                fid,
                row["status"],
                flush=True,
            )
    return summary

"""Robustness and human-calibration utilities for scientific expert review.

The functions in this module operate only on JSON-like review packets and
review outputs.  They do not invoke a model and cannot advance any benchmark
lifecycle gate.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .scientific_run_snapshots import (
    canonical_json_sha256,
    create_scientific_run,
    write_immutable_json,
)
from .scientific_expert_review import (
    ATOMIC_OBSERVATION_FIELDS,
    validate_scientific_expert_output,
    validate_scientific_review_packet,
)


ATOMIC_REVIEW_FIELDS = tuple(ATOMIC_OBSERVATION_FIELDS)

CITATION_FIELDS = ("evidence_ids", "source_ids")

MUTATION_TYPES = (
    "TARGET_DRIFT",
    "SCIENTIFIC_SCOPE_DRIFT",
    "UNSUPPORTED_FIXED_O",
    "LEGITIMATE_O2_OPENNESS",
    "O2_UNRESOLVED_DIMENSION_ACCIDENTALLY_FIXED",
    "F1_F2_RESPONSIBILITY_CONFUSION",
    "WRONG_FAMILY_EVIDENCE",
    "MISSING_CRITICAL_EVIDENCE",
    "UNSUPPORTED_OBSERVABLE",
    "FABRICATED_PROVENANCE_ID",
)

_MUTATION_EXPECTATIONS: Mapping[str, Mapping[str, Any]] = {
    "TARGET_DRIFT": {
        "expected_contract_check": "QUESTION_TARGET_DRIFT_DETECTED",
        "accepted_signals": {"question_target_preserved": (False,)},
    },
    "SCIENTIFIC_SCOPE_DRIFT": {
        "expected_contract_check": "QUESTION_SCOPE_DRIFT_DETECTED",
        "accepted_signals": {"question_scope_preserved": (False,)},
    },
    "UNSUPPORTED_FIXED_O": {
        "expected_contract_check": "UNSUPPORTED_FIXED_O_DETECTED",
        "accepted_signals": {"fixed_o_supported": (False,)},
    },
    "LEGITIMATE_O2_OPENNESS": {
        "expected_contract_check": "LEGITIMATE_O2_OPENNESS_PRESERVED",
        "observational_only": True,
    },
    "O2_UNRESOLVED_DIMENSION_ACCIDENTALLY_FIXED": {
        "expected_contract_check": "O2_OPENNESS_COLLAPSE_DETECTED",
        "accepted_signals": {"unresolved_o_exposed_by_question": (False,)},
    },
    "F1_F2_RESPONSIBILITY_CONFUSION": {
        "expected_contract_check": "FINDING_RESPONSIBILITY_DRIFT_DETECTED",
        "accepted_signals": {"finding_contract_supported": (False,)},
    },
    "WRONG_FAMILY_EVIDENCE": {
        "expected_contract_check": "FAMILY_EVIDENCE_BINDING_REJECTED",
        "pre_invocation_rejection_allowed": True,
        "accepted_recommendations": ("MORE_EVIDENCE_REQUIRED", "NOT_ELIGIBLE"),
    },
    "MISSING_CRITICAL_EVIDENCE": {
        "expected_contract_check": "EVIDENCE_INSUFFICIENCY_RECOGNIZED",
        "observational_only": True,
    },
    "UNSUPPORTED_OBSERVABLE": {
        "expected_contract_check": "UNSUPPORTED_OBSERVABLE_DETECTED",
        "accepted_signals": {"fixed_o_supported": (False,)},
    },
    "FABRICATED_PROVENANCE_ID": {
        "expected_contract_check": "UNKNOWN_PROVENANCE_REJECTED",
        "pre_invocation_rejection_allowed": True,
        "accepted_recommendations": ("MORE_EVIDENCE_REQUIRED", "NOT_ELIGIBLE"),
    },
}

_PRE_INVOCATION_REJECTION_STATUSES = {
    "INPUT_REJECTED",
    "PACKET_INVALID",
    "PROVENANCE_INVALID",
    "VALIDATION_FAILED",
}

_INFRASTRUCTURE_STATUSES = {
    "PROVIDER_TIMEOUT",
    "PROVIDER_ERROR",
    "TOOL_ERROR",
    "INFRASTRUCTURE_INVALID",
    "NOT_RUN",
}

_CALIBRATION_FORBIDDEN_KEYS = {
    "accepted_branch",
    "accepted_branches",
    "benchmark_score",
    "curator_decision",
    "curator_status",
    "eligibility_recommendation",
    "evaluation_output",
    "ground_truth",
    "gt",
    "model_answer",
    "model_output",
    "model_review",
    "previous_expert_verdict",
    "reference_answer",
    "reference_findings",
    "scq_result",
    "srac_adjudication",
}

_SAFE_CLAIM_FIELDS = (
    "claim_id",
    "claim_type",
    "statement",
    "critical",
    "evidence_ids",
    "source_ids",
)
_SAFE_EVIDENCE_FIELDS = (
    "evidence_id",
    "dataset_id",
    "evidence_type",
    "statement",
    "source_id",
    "locator",
)
_SAFE_SOURCE_FIELDS = (
    "source_id",
    "source_type",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "report_id",
)
_SAFE_LINK_FIELDS = (
    "claim_id",
    "evidence_record_id",
    "source_id",
    "support_relation",
    "supported_statement",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sequence(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence")
    return list(value)


def _non_empty_string(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _condition_id(value: Mapping[str, Any]) -> str:
    return str(value.get("condition") or value.get("condition_id") or "").strip()


def _condition_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _sequence(packet.get("conditions"), field="review packet conditions")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("every review packet condition must be an object")
        condition = _condition_id(row)
        if not condition:
            raise ValueError("every review packet condition requires condition or condition_id")
        if condition in seen:
            raise ValueError(f"duplicate review packet condition: {condition}")
        seen.add(condition)
        result.append(dict(row))
    return result


def _review_rows(review: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = review.get("condition_reviews")
    rows = _sequence(raw, field="condition_reviews")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("every condition review must be an object")
        condition = _condition_id(row)
        if not condition:
            raise ValueError("every condition review requires condition or condition_id")
        if condition in result:
            raise ValueError(f"duplicate condition review: {condition}")
        result[condition] = dict(row)
    return result


def _field_name(mapping: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        if name in mapping:
            return name
    return names[0]


def _condition_for_mutation(
    conditions: Sequence[Mapping[str, Any]],
    *,
    prefix: str | None = None,
    finding_openness: str | None = None,
) -> int:
    for index, row in enumerate(conditions):
        condition = _condition_id(row).upper()
        if prefix and condition.startswith(prefix.upper()):
            return index
        if finding_openness and str(row.get("finding_openness", "")).casefold() == finding_openness.casefold():
            return index
    raise ValueError(f"review packet has no condition matching {prefix or finding_openness}")


def _condition_with_fixed_o(conditions: Sequence[Mapping[str, Any]]) -> int:
    for index, row in enumerate(conditions):
        fixed = row.get("fixed_o_dimensions", row.get("fixed_dimensions", ()))
        if isinstance(fixed, Sequence) and not isinstance(fixed, (str, bytes, bytearray)) and fixed:
            return index
    raise ValueError("review packet has no condition with fixed O dimensions")


def _f_condition(conditions: Sequence[Mapping[str, Any]]) -> int:
    for preferred in ("O1-F2", "O1-F1"):
        for index, row in enumerate(conditions):
            if _condition_id(row).upper().startswith(preferred):
                return index
    for index, row in enumerate(conditions):
        if _condition_id(row).upper().endswith(("F1", "F2")):
            return index
    raise ValueError("review packet has no condition with explicit Finding responsibility")


def _replace_conditions(packet: dict[str, Any], conditions: Sequence[Mapping[str, Any]]) -> None:
    packet["conditions"] = [dict(item) for item in conditions]


def _first_evidence_reference(packet: Mapping[str, Any]) -> str:
    for claim_field in ("scientific_claims", "scientific_support_claims"):
        claims = packet.get(claim_field)
        if isinstance(claims, Sequence) and not isinstance(claims, (str, bytes, bytearray)):
            for claim in claims:
                if isinstance(claim, Mapping):
                    values = claim.get("evidence_ids")
                    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                        for value in values:
                            if str(value).strip():
                                return str(value)
    links = packet.get("claim_support_links") or ()
    if isinstance(links, Sequence) and not isinstance(links, (str, bytes, bytearray)):
        for link in links:
            if isinstance(link, Mapping):
                value = link.get("evidence_record_id", link.get("evidence_id"))
                if str(value or "").strip():
                    return str(value)
    records = packet.get("evidence_records") or packet.get("current_evidence_records") or ()
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
        for record in records:
            if isinstance(record, Mapping) and str(record.get("evidence_id", "")).strip():
                return str(record["evidence_id"])
    raise ValueError("review packet has no evidence reference to ablate")


def _remove_evidence_reference(packet: dict[str, Any], evidence_id: str) -> list[str]:
    affected_claim_ids: list[str] = []
    for record_field in ("evidence_records", "current_evidence_records"):
        records = packet.get(record_field)
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
            packet[record_field] = [
                copy.deepcopy(item)
                for item in records
                if not isinstance(item, Mapping) or str(item.get("evidence_id")) != evidence_id
            ]
    bindings = packet.get("evidence_family_bindings")
    if isinstance(bindings, Sequence) and not isinstance(
        bindings, (str, bytes, bytearray)
    ):
        packet["evidence_family_bindings"] = [
            copy.deepcopy(item)
            for item in bindings
            if not isinstance(item, Mapping)
            or str(item.get("evidence_id", "")) != evidence_id
        ]
    links = packet.get("claim_support_links")
    if isinstance(links, Sequence) and not isinstance(links, (str, bytes, bytearray)):
        affected_claim_ids = sorted(
            {
                str(item.get("claim_id"))
                for item in links
                if isinstance(item, Mapping)
                and str(item.get("evidence_record_id", item.get("evidence_id"))) == evidence_id
                and item.get("claim_id")
            }
        )
        packet["claim_support_links"] = [
            copy.deepcopy(item)
            for item in links
            if not isinstance(item, Mapping)
            or str(item.get("evidence_record_id", item.get("evidence_id"))) != evidence_id
        ]
    remaining_links = packet.get("claim_support_links") or ()
    links_by_claim: dict[str, list[Mapping[str, Any]]] = {}
    for link in remaining_links if isinstance(remaining_links, Sequence) else ():
        if isinstance(link, Mapping):
            links_by_claim.setdefault(str(link.get("claim_id", "")), []).append(link)
    for claim_field in ("scientific_claims", "scientific_support_claims"):
        claims = packet.get(claim_field)
        if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes, bytearray)):
            continue
        projected: list[Any] = []
        for claim in claims:
            if not isinstance(claim, Mapping):
                projected.append(copy.deepcopy(claim))
                continue
            value = dict(copy.deepcopy(claim))
            claim_links = links_by_claim.get(str(value.get("claim_id", "")), [])
            value["evidence_ids"] = sorted(
                {str(item.get("evidence_record_id", item.get("evidence_id", ""))) for item in claim_links}
            )
            value["source_ids"] = sorted(
                {str(item.get("source_id", "")) for item in claim_links}
            )
            projected.append(value)
        packet[claim_field] = projected
    remaining_evidence = packet.get("evidence_records") or packet.get("current_evidence_records") or ()
    referenced_sources = {
        str(item.get("source_id", ""))
        for item in remaining_evidence
        if isinstance(item, Mapping)
    }
    for source_field in ("source_records", "candidate_source_records"):
        sources = packet.get(source_field)
        if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, bytearray)):
            packet[source_field] = [
                copy.deepcopy(item)
                for item in sources
                if not isinstance(item, Mapping)
                or str(item.get("source_id", "")) in referenced_sources
            ]
    summary = packet.get("provenance_summary")
    if isinstance(summary, Mapping):
        updated = dict(summary)
        updated["evidence_count"] = len(remaining_evidence)
        remaining_sources = packet.get("source_records") or packet.get("candidate_source_records") or ()
        updated["source_count"] = len(remaining_sources)
        existing_unresolved = {
            str(item) for item in updated.get("unresolved_claim_ids", ()) or ()
        }
        claims = packet.get("scientific_claims") or packet.get("scientific_support_claims") or ()
        structurally_unresolved = {
            str(item.get("claim_id", ""))
            for item in claims
            if isinstance(item, Mapping)
            and not links_by_claim.get(str(item.get("claim_id", "")))
        }
        updated["unresolved_claim_ids"] = sorted(existing_unresolved | structurally_unresolved)
        if updated["unresolved_claim_ids"]:
            updated["claim_evidence_sufficiency"] = "INCOMPLETE"
        else:
            updated["claim_evidence_sufficiency"] = "COMPLETE"
        packet["provenance_summary"] = updated
    if isinstance(packet.get("provenance_summary"), Mapping) and packet["provenance_summary"].get(
        "unresolved_claim_ids"
    ):
        packet["review_input_status"] = "EVIDENCE_INCOMPLETE"
    else:
        packet["review_input_status"] = "EVIDENCE_COMPLETE"
    return affected_claim_ids


def validate_review_packet_provenance(review_packet: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only explicit packet identity and provenance closure."""

    canonical_validation = validate_scientific_review_packet(review_packet)
    dataset_id, _, _ = _packet_identity(review_packet)
    sources = review_packet.get("source_records") or review_packet.get("candidate_source_records") or ()
    evidence = review_packet.get("evidence_records") or review_packet.get("current_evidence_records") or ()
    claims = review_packet.get("scientific_claims") or review_packet.get("scientific_support_claims") or ()
    links = review_packet.get("claim_support_links") or ()
    errors: list[dict[str, Any]] = [
        {"code": str(item.get("code", "CANONICAL_PACKET_INVALID")), "field": item.get("field")}
        for item in canonical_validation.get("errors", ())
        if isinstance(item, Mapping)
    ]

    def indexed_ids(rows: Any, field: str, collection: str) -> set[str]:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            errors.append({"code": "COLLECTION_NOT_LIST", "collection": collection})
            return set()
        values: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or not str(row.get(field, "")).strip():
                errors.append(
                    {"code": "IDENTIFIER_MISSING", "collection": collection, "index": index}
                )
                continue
            values.append(str(row[field]))
        duplicates = sorted({value for value in values if values.count(value) > 1})
        for value in duplicates:
            errors.append(
                {"code": "DUPLICATE_IDENTIFIER", "collection": collection, "identifier": value}
            )
        return set(values)

    source_ids = indexed_ids(sources, "source_id", "source_records")
    evidence_ids = indexed_ids(evidence, "evidence_id", "evidence_records")
    claim_ids = indexed_ids(claims, "claim_id", "scientific_claims")
    for record in evidence if isinstance(evidence, Sequence) else ():
        if not isinstance(record, Mapping):
            continue
        record_id = str(record.get("evidence_id", ""))
        source_id = str(record.get("source_id", ""))
        if source_id not in source_ids:
            errors.append(
                {"code": "UNKNOWN_SOURCE_ID", "evidence_id": record_id, "source_id": source_id}
            )
        record_dataset = record.get("dataset_id")
        if dataset_id and record_dataset is not None and str(record_dataset) != str(dataset_id):
            errors.append(
                {
                    "code": "EVIDENCE_DATASET_MISMATCH",
                    "evidence_id": record_id,
                    "expected_dataset_id": dataset_id,
                    "actual_dataset_id": record_dataset,
                }
            )
    for index, link in enumerate(links if isinstance(links, Sequence) else ()):
        if not isinstance(link, Mapping):
            errors.append({"code": "LINK_NOT_OBJECT", "index": index})
            continue
        claim_id = str(link.get("claim_id", ""))
        evidence_id = str(link.get("evidence_record_id", link.get("evidence_id", "")))
        source_id = str(link.get("source_id", ""))
        if claim_id not in claim_ids:
            errors.append({"code": "UNKNOWN_CLAIM_ID", "claim_id": claim_id})
        if evidence_id not in evidence_ids:
            errors.append({"code": "UNKNOWN_EVIDENCE_ID", "evidence_id": evidence_id})
        if source_id not in source_ids:
            errors.append({"code": "UNKNOWN_SOURCE_ID", "source_id": source_id})
        matching_source = next(
            (
                str(record.get("source_id", ""))
                for record in evidence
                if isinstance(record, Mapping) and str(record.get("evidence_id", "")) == evidence_id
            ),
            None,
        )
        if matching_source is not None and matching_source != source_id:
            errors.append(
                {
                    "code": "LINK_SOURCE_MISMATCH",
                    "evidence_id": evidence_id,
                    "link_source_id": source_id,
                    "evidence_source_id": matching_source,
                }
            )
    for claim in claims if isinstance(claims, Sequence) else ():
        if not isinstance(claim, Mapping):
            continue
        for field, known in (("evidence_ids", evidence_ids), ("source_ids", source_ids)):
            values = claim.get(field)
            if values is None:
                continue
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                errors.append({"code": "CLAIM_PROVENANCE_NOT_LIST", "field": field})
                continue
            unknown = sorted({str(value) for value in values} - known)
            for value in unknown:
                errors.append(
                    {
                        "code": "UNKNOWN_EVIDENCE_ID" if field == "evidence_ids" else "UNKNOWN_SOURCE_ID",
                        "identifier": value,
                    }
                )
    return {
        "status": "PASS" if not errors else "FAIL",
        "family_binding_status": "PASS"
        if not any(
            item["code"]
            in {"EVIDENCE_DATASET_MISMATCH", "EVIDENCE_FAMILY_MISMATCH"}
            for item in errors
        )
        else "FAIL",
        "evidence_provenance_closure": "PASS" if not errors else "FAIL",
        "errors": errors,
        "visible_claim_count": len(claim_ids),
        "evidence_count": len(evidence_ids),
        "source_count": len(source_ids),
        "canonical_packet_validation": canonical_validation,
    }


def ablate_scientific_evidence(
    review_packet: Mapping[str, Any],
    *,
    evidence_id: str,
    critical_reason: str,
) -> dict[str, Any]:
    """Remove one explicitly selected evidence record without inventing truth."""

    selected_id = _non_empty_string(evidence_id, field="evidence_id")
    reason = _non_empty_string(critical_reason, field="critical_reason")
    packet = copy.deepcopy(dict(review_packet))
    records = packet.get("evidence_records") or packet.get("current_evidence_records") or ()
    available = {
        str(item.get("evidence_id"))
        for item in records
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    if selected_id not in available:
        raise ValueError(f"unknown evidence_id for ablation: {selected_id}")
    source_hash = canonical_json_sha256(packet)
    affected_claim_ids = _remove_evidence_reference(packet, selected_id)
    provenance = validate_review_packet_provenance(packet)
    return {
        "ablation_id": f"evidence-ablation-{canonical_json_sha256([source_hash, selected_id])[:12]}",
        "source_packet_sha256": source_hash,
        "ablated_packet_sha256": canonical_json_sha256(packet),
        "removed_evidence_id": selected_id,
        "affected_claim_ids": affected_claim_ids,
        "newly_unresolved_claim_ids": list(
            packet.get("provenance_summary", {}).get("unresolved_claim_ids", ())
        ),
        "critical_reason": reason,
        "expected_behavior": "OBSERVE_ATOMIC_AND_CONFIDENCE_CHANGE_WITHOUT_PREASSIGNED_TRUTH",
        "scientific_truth_assigned": False,
        "provenance_validation": provenance,
        "packet": packet,
    }


def _mutate_target(packet: dict[str, Any]) -> str:
    conditions = _condition_rows(packet)
    index = 0
    row = dict(conditions[index])
    row["scientific_target"] = "domain-wide thermal plume evolution"
    row["question"] = "Characterize the domain-wide thermal plume evolution in this flow field."
    conditions[index] = row
    _replace_conditions(packet, conditions)
    return _condition_id(row)


def _mutate_scope(packet: dict[str, Any]) -> str:
    conditions = _condition_rows(packet)
    index = 0
    row = dict(conditions[index])
    question = str(row.get("question", "")).strip()
    row["question"] = f"{question} Generalize the result beyond the supplied dataset and declared physical scope.".strip()
    conditions[index] = row
    _replace_conditions(packet, conditions)
    return _condition_id(row)


def _mark_authored_o_unmaterialized(row: dict[str, Any]) -> None:
    authored = list(row.get("authored_operationalizations", ()) or ())
    route = dict(row.get("materialization_route", {}) or {})
    route.update(
        {
            "execution_status": "DETERMINISTIC_ROUTE_NOT_DECLARED",
            "handler_id": "",
            "branch_count": len(authored),
            "branch_statuses": ["AUTHORED_O_ROUTE_NOT_DECLARED" for _ in authored],
            "scientific_result_values_visible": False,
        }
    )
    row["materialization_route"] = route


def _mutate_unsupported_fixed_o(packet: dict[str, Any]) -> str:
    conditions = _condition_rows(packet)
    index = _condition_with_fixed_o(conditions)
    row = dict(conditions[index])
    field = _field_name(row, ("fixed_o_dimensions", "fixed_dimensions"))
    fixed = list(row.get(field, ()) or ())
    if not fixed:
        raise ValueError("unsupported fixed-O mutation requires a fixed dimension")
    fixed_dimension = str(fixed[0])
    authored_rows = []
    for raw_candidate in row.get("authored_operationalizations", ()) or ():
        candidate = dict(raw_candidate) if isinstance(raw_candidate, Mapping) else {}
        decisions = []
        replaced = False
        for raw_decision in candidate.get("decisions", ()) or ():
            decision = dict(raw_decision) if isinstance(raw_decision, Mapping) else {}
            if str(decision.get("dimension_id", "")) == fixed_dimension:
                decision.update(
                    {
                        "canonical_id": "pressure_gradient_selected_region",
                        "description": (
                            "Define the target regions as face-connected locations selected "
                            "by pressure-gradient magnitude."
                        ),
                    }
                )
                replaced = True
            decisions.append(decision)
        if not replaced:
            raise ValueError("fixed dimension is absent from an authored Operationalization")
        candidate["decisions"] = decisions
        authored_rows.append(candidate)
    row["authored_operationalizations"] = authored_rows
    row["question"] = (
        f"{str(row.get('question', '')).strip()} Define the target regions using "
        "pressure-gradient magnitude."
    ).strip()
    _mark_authored_o_unmaterialized(row)
    conditions[index] = row
    _replace_conditions(packet, conditions)
    return _condition_id(row)


def _preserve_legitimate_o2(packet: dict[str, Any]) -> str:
    conditions = _condition_rows(packet)
    index = _condition_for_mutation(conditions, prefix="O2")
    row = dict(conditions[index])
    field = _field_name(row, ("unresolved_o_dimensions", "unresolved_dimensions"))
    unresolved = list(row.get(field, ()) or ())
    if not unresolved:
        raise ValueError("O2 positive control requires at least one unresolved dimension")
    conditions[index] = row
    _replace_conditions(packet, conditions)
    return _condition_id(row)


def _mutate_o2_fixed(packet: dict[str, Any]) -> str:
    conditions = _condition_rows(packet)
    index = _condition_for_mutation(conditions, prefix="O2")
    row = dict(conditions[index])
    unresolved_field = _field_name(row, ("unresolved_o_dimensions", "unresolved_dimensions"))
    unresolved = list(row.get(unresolved_field, ()) or ())
    if not unresolved:
        raise ValueError("O2 collapse mutation requires an unresolved dimension")
    dimension = unresolved[0]
    question = str(row.get("question", "")).strip()
    replacements = {
        "criterion": (
            r"using an appropriate speed-based criterion",
            "using a median-split speed criterion",
        ),
        "property_measure": (
            r"use an appropriate measure to compare their strength",
            "use peak speed to compare their strength",
        ),
    }
    pattern, replacement = replacements.get(
        str(dimension),
        (r"$", f" For {dimension}, use a median-split definition."),
    )
    mutated_question, replacement_count = re.subn(
        pattern,
        replacement,
        question,
        count=1,
        flags=re.IGNORECASE,
    )
    if replacement_count == 0:
        mutated_question = f"{question} Use a median-split speed criterion.".strip()
    row["question"] = mutated_question
    conditions[index] = row
    _replace_conditions(packet, conditions)
    return _condition_id(row)


def _mutate_finding_responsibility(packet: dict[str, Any]) -> str:
    conditions = _condition_rows(packet)
    index = _f_condition(conditions)
    row = dict(conditions[index])
    current = str(row.get("finding_openness") or row.get("finding_responsibility") or "F1").upper()
    if "OPEN" in current or current == "F2" or _condition_id(row).upper().endswith("F2"):
        row["finding_openness"] = "bounded"
        row["finding_responsibility"] = "F1"
    else:
        row["finding_openness"] = "open"
        row["finding_responsibility"] = "F2"
    conditions[index] = row
    _replace_conditions(packet, conditions)
    return _condition_id(row)


def _mutate_wrong_family_evidence(
    packet: dict[str, Any],
    replacement_packet: Mapping[str, Any] | None,
) -> str:
    evidence_field = (
        "evidence_records" if "evidence_records" in packet else "current_evidence_records"
    )
    records = list(packet.get(evidence_field, ()) or ())
    if not records:
        raise ValueError("wrong-family evidence mutation requires evidence records")
    if replacement_packet is not None and all(
        isinstance(replacement_packet.get(field), list)
        for field in (
            "scientific_claims",
            "evidence_records",
            "evidence_family_bindings",
            "source_records",
            "claim_support_links",
        )
    ):
        for field in (
            "scientific_claims",
            "evidence_records",
            "evidence_family_bindings",
            "source_records",
            "claim_support_links",
        ):
            packet[field] = copy.deepcopy(replacement_packet[field])
        sibling_summary = replacement_packet.get("provenance_summary")
        if isinstance(sibling_summary, Mapping):
            summary = dict(packet.get("provenance_summary", {}) or {})
            for field in (
                "claim_evidence_sufficiency",
                "unresolved_claim_ids",
                "visible_claim_count",
                "evidence_count",
                "source_count",
            ):
                summary[field] = copy.deepcopy(sibling_summary.get(field))
            packet["provenance_summary"] = summary
        packet["review_input_status"] = replacement_packet.get(
            "review_input_status", packet.get("review_input_status")
        )
    else:
        bindings = list(packet.get("evidence_family_bindings", ()) or ())
        if not bindings or not isinstance(bindings[0], Mapping):
            raise ValueError("wrong-family mutation requires explicit evidence bindings")
        wrong_binding = dict(copy.deepcopy(bindings[0]))
        wrong_binding["family_id"] = "__controlled_wrong_family__"
        wrong_binding["concept_id"] = "__controlled_wrong_concept__"
        bindings[0] = wrong_binding
        packet["evidence_family_bindings"] = bindings
    return _condition_id(_condition_rows(packet)[0])


def _mutate_missing_evidence(packet: dict[str, Any]) -> str:
    evidence_id = _first_evidence_reference(packet)
    _remove_evidence_reference(packet, evidence_id)
    return _condition_id(_condition_rows(packet)[0])


def _mutate_unsupported_observable(packet: dict[str, Any]) -> str:
    conditions = _condition_rows(packet)
    index = _condition_with_fixed_o(conditions)
    row = dict(conditions[index])
    authored_rows = list(row.get("authored_operationalizations", ()) or ())
    if authored_rows and all(isinstance(item, Mapping) for item in authored_rows):
        mutated_rows = []
        for raw_authored in authored_rows:
            authored = dict(raw_authored)
            decisions = [
                dict(item)
                for item in authored.get("decisions", ()) or ()
                if isinstance(item, Mapping)
            ]
            if not decisions:
                raise ValueError("unsupported observable mutation requires authored decisions")
            decisions[0].update(
                {
                    "canonical_id": "pressure_gradient_magnitude",
                    "description": "Use pressure-gradient magnitude as the analysis observable.",
                }
            )
            authored["decisions"] = decisions
            mutated_rows.append(authored)
        row["authored_operationalizations"] = mutated_rows
    else:
        raise ValueError("unsupported observable mutation requires authored Operationalizations")
    _mark_authored_o_unmaterialized(row)
    conditions[index] = row
    _replace_conditions(packet, conditions)
    return _condition_id(row)


def _mutate_fabricated_provenance(packet: dict[str, Any]) -> str:
    claim_field = "scientific_claims" if "scientific_claims" in packet else "scientific_support_claims"
    claims = list(packet.get(claim_field, ()) or ())
    if not claims:
        raise ValueError("fabricated provenance mutation requires scientific claims")
    first = dict(claims[0]) if isinstance(claims[0], Mapping) else {}
    first["evidence_ids"] = list(first.get("evidence_ids", ()) or ()) + ["fabricated_evidence_id"]
    first["source_ids"] = list(first.get("source_ids", ()) or ()) + ["fabricated_source_id"]
    claims[0] = first
    packet[claim_field] = claims
    return _condition_id(_condition_rows(packet)[0])


def build_semantic_mutation_suite(
    review_packet: Mapping[str, Any],
    *,
    wrong_family_packet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ten controlled contract challenges from one sanitized packet."""

    _condition_rows(review_packet)
    source_hash = canonical_json_sha256(review_packet)
    mutators = {
        "TARGET_DRIFT": lambda value: _mutate_target(value),
        "SCIENTIFIC_SCOPE_DRIFT": lambda value: _mutate_scope(value),
        "UNSUPPORTED_FIXED_O": lambda value: _mutate_unsupported_fixed_o(value),
        "LEGITIMATE_O2_OPENNESS": lambda value: _preserve_legitimate_o2(value),
        "O2_UNRESOLVED_DIMENSION_ACCIDENTALLY_FIXED": lambda value: _mutate_o2_fixed(value),
        "F1_F2_RESPONSIBILITY_CONFUSION": lambda value: _mutate_finding_responsibility(value),
        "WRONG_FAMILY_EVIDENCE": lambda value: _mutate_wrong_family_evidence(
            value, wrong_family_packet
        ),
        "MISSING_CRITICAL_EVIDENCE": lambda value: _mutate_missing_evidence(value),
        "UNSUPPORTED_OBSERVABLE": lambda value: _mutate_unsupported_observable(value),
        "FABRICATED_PROVENANCE_ID": lambda value: _mutate_fabricated_provenance(value),
    }
    mutations: list[dict[str, Any]] = []
    for index, mutation_type in enumerate(MUTATION_TYPES, start=1):
        packet = copy.deepcopy(dict(review_packet))
        condition = mutators[mutation_type](packet)
        expectation = _MUTATION_EXPECTATIONS[mutation_type]
        mutations.append(
            {
                "mutation_id": f"semantic-mutation-{index:02d}",
                "mutation_type": mutation_type,
                "condition": condition,
                "source_packet_sha256": source_hash,
                "mutated_packet_sha256": canonical_json_sha256(packet),
                "expected_contract_check": expectation["expected_contract_check"],
                "scientific_truth_assigned": False,
                "preflight_validation": validate_review_packet_provenance(packet),
                "packet": packet,
            }
        )
    return {
        "suite_version": "flow-expert-semantic-mutation-v1",
        "source_packet_sha256": source_hash,
        "mutation_count": len(mutations),
        "mutations": mutations,
    }


def audit_semantic_mutation_reviews(
    mutation_suite: Mapping[str, Any],
    review_attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit known contract signals without assigning ambiguous scientific truth."""

    mutations = _sequence(mutation_suite.get("mutations"), field="mutations")
    attempts: dict[str, Mapping[str, Any]] = {}
    for attempt in review_attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("every mutation review attempt must be an object")
        mutation_id = _non_empty_string(attempt.get("mutation_id"), field="mutation_id")
        if mutation_id in attempts:
            raise ValueError(f"duplicate mutation review attempt: {mutation_id}")
        attempts[mutation_id] = attempt
    rows: list[dict[str, Any]] = []
    for raw in mutations:
        if not isinstance(raw, Mapping):
            raise ValueError("every mutation must be an object")
        mutation_id = _non_empty_string(raw.get("mutation_id"), field="mutation_id")
        mutation_type = _non_empty_string(raw.get("mutation_type"), field="mutation_type")
        if mutation_type not in _MUTATION_EXPECTATIONS:
            raise ValueError(f"unknown mutation type: {mutation_type}")
        expectation = _MUTATION_EXPECTATIONS[mutation_type]
        attempt = attempts.get(mutation_id)
        preflight = raw.get("preflight_validation")
        if (
            expectation.get("pre_invocation_rejection_allowed")
            and isinstance(preflight, Mapping)
            and preflight.get("status") == "FAIL"
        ):
            if attempt is not None:
                invocation_status = str(
                    attempt.get("invocation_status") or attempt.get("status") or ""
                ).upper()
                submitted = attempt.get("submitted_to_model")
                if invocation_status not in _PRE_INVOCATION_REJECTION_STATUSES | {"NOT_RUN"} or submitted is True:
                    rows.append(
                        {
                            "mutation_id": mutation_id,
                            "mutation_type": mutation_type,
                            "status": "PREFLIGHT_REJECTION_BYPASSED",
                            "detected": False,
                            "disagreement_fields": ["preflight_invocation_boundary"],
                        }
                    )
                    continue
            rows.append(
                {
                    "mutation_id": mutation_id,
                    "mutation_type": mutation_type,
                    "status": "EXPECTED_INPUT_REJECTION",
                    "detected": True,
                    "disagreement_fields": [],
                }
            )
            continue
        if attempt is None:
            rows.append(
                {
                    "mutation_id": mutation_id,
                    "mutation_type": mutation_type,
                    "status": "NOT_RUN",
                    "detected": None,
                    "disagreement_fields": [],
                }
            )
            continue
        invocation_status = str(attempt.get("invocation_status") or attempt.get("status") or "").upper()
        if invocation_status in _INFRASTRUCTURE_STATUSES:
            rows.append(
                {
                    "mutation_id": mutation_id,
                    "mutation_type": mutation_type,
                    "status": "INFRASTRUCTURE_INVALID",
                    "detected": None,
                    "disagreement_fields": [],
                }
            )
            continue
        if expectation.get("pre_invocation_rejection_allowed") and invocation_status in _PRE_INVOCATION_REJECTION_STATUSES:
            rows.append(
                {
                    "mutation_id": mutation_id,
                    "mutation_type": mutation_type,
                    "status": "EXPECTED_INPUT_REJECTION",
                    "detected": True,
                    "disagreement_fields": [],
                }
            )
            continue
        review = attempt.get("review") if isinstance(attempt.get("review"), Mapping) else attempt
        packet = raw.get("packet")
        contract_validation = (
            validate_scientific_expert_output(review, packet)
            if isinstance(packet, Mapping)
            else {"status": "INCOMPLETE", "reason_codes": ["MUTATION_PACKET_MISSING"]}
        )
        if contract_validation.get("status") != "PASS":
            rows.append(
                {
                    "mutation_id": mutation_id,
                    "mutation_type": mutation_type,
                    "status": "REVIEW_INCOMPLETE",
                    "detected": False,
                    "disagreement_fields": list(contract_validation.get("reason_codes", ())),
                }
            )
            continue
        review_rows = _review_rows(review)
        condition = str(raw.get("condition") or "")
        condition_review = review_rows.get(condition)
        if condition_review is None:
            rows.append(
                {
                    "mutation_id": mutation_id,
                    "mutation_type": mutation_type,
                    "status": "REVIEW_INCOMPLETE",
                    "detected": False,
                    "disagreement_fields": ["condition_review"],
                }
            )
            continue
        if expectation.get("observational_only"):
            rows.append(
                {
                    "mutation_id": mutation_id,
                    "mutation_type": mutation_type,
                    "status": "OBSERVED_WITHOUT_FROZEN_SCIENTIFIC_TRUTH",
                    "detected": None,
                    "disagreement_fields": [],
                    "atomic_observations": {
                        field: condition_review.get(field) for field in ATOMIC_REVIEW_FIELDS
                    },
                    "eligibility_recommendation": condition_review.get(
                        "eligibility_recommendation"
                    ),
                }
            )
            continue
        signal_results: list[bool] = []
        disagreement_fields: list[str] = []
        for field, accepted in expectation.get("accepted_signals", {}).items():
            match = condition_review.get(field) in accepted
            signal_results.append(match)
            if not match:
                disagreement_fields.append(field)
        recommendation_values = expectation.get("accepted_recommendations")
        if recommendation_values:
            recommendation_match = str(condition_review.get("eligibility_recommendation", "")).upper() in recommendation_values
            signal_results.append(recommendation_match)
            if not recommendation_match:
                disagreement_fields.append("eligibility_recommendation")
        detected = any(signal_results) if expectation.get("signal_mode") == "ANY" else all(signal_results)
        rows.append(
            {
                "mutation_id": mutation_id,
                "mutation_type": mutation_type,
                "status": "DETECTED" if detected else "NOT_DETECTED",
                "detected": detected,
                "disagreement_fields": disagreement_fields,
            }
        )
    evaluable = [
        row
        for row in rows
        if isinstance(row["detected"], bool)
        and row["status"] not in {"EXPECTED_INPUT_REJECTION", "PREFLIGHT_REJECTION_BYPASSED"}
    ]
    scored = [
        row
        for row in rows
        if row["status"] != "OBSERVED_WITHOUT_FROZEN_SCIENTIFIC_TRUTH"
    ]
    complete = bool(scored) and all(
        row["status"] in {"DETECTED", "EXPECTED_INPUT_REJECTION"} for row in scored
    )
    any_model_attempt = any(
        row["status"]
        not in {"NOT_RUN", "EXPECTED_INPUT_REJECTION"}
        for row in rows
    )
    return {
        "status": (
            "PASS"
            if complete and evaluable and all(row["detected"] for row in evaluable)
            else "NOT_RUN"
            if not any_model_attempt
            else "PARTIAL"
        ),
        "mutation_count": len(rows),
        "evaluable_count": len(evaluable),
        "detected_count": sum(row["detected"] is True for row in evaluable),
        "preflight_rejection_count": sum(
            row["status"] == "EXPECTED_INPUT_REJECTION" for row in rows
        ),
        "observational_count": sum(
            row["status"] == "OBSERVED_WITHOUT_FROZEN_SCIENTIFIC_TRUTH"
            for row in rows
        ),
        "infrastructure_invalid_count": sum(row["status"] == "INFRASTRUCTURE_INVALID" for row in rows),
        "rows": rows,
        "scientific_truth_assigned": False,
    }


def compare_evidence_ablation_behavior(
    baseline_review: Mapping[str, Any],
    ablated_review: Mapping[str, Any],
    *,
    condition: str,
) -> dict[str, Any]:
    """Report whether critical-evidence removal changes confidence."""

    baseline = _review_rows(baseline_review).get(condition)
    ablated = _review_rows(ablated_review).get(condition)
    if baseline is None or ablated is None:
        raise ValueError(f"condition review missing for ablation comparison: {condition}")
    changed_fields = [
        field for field in ATOMIC_REVIEW_FIELDS if baseline.get(field) != ablated.get(field)
    ]
    recommendation = str(ablated.get("eligibility_recommendation", "")).upper()
    explicit_uncertainty = recommendation == "MORE_EVIDENCE_REQUIRED" or any(
        baseline.get(field) is True and ablated.get(field) in {False, None, "UNCERTAIN"}
        for field in ATOMIC_REVIEW_FIELDS
    )
    baseline_confident = str(baseline.get("eligibility_recommendation", "")).upper() == "ELIGIBLE"
    false_confidence = baseline_confident and recommendation == "ELIGIBLE" and not changed_fields
    return {
        "condition": condition,
        "status": "RESPONDED_TO_ABLATION" if explicit_uncertainty else "NO_UNCERTAINTY_SHIFT",
        "changed_atomic_fields": changed_fields,
        "baseline_recommendation": baseline.get("eligibility_recommendation"),
        "ablated_recommendation": ablated.get("eligibility_recommendation"),
        "explicit_uncertainty_or_more_evidence": explicit_uncertainty,
        "false_confidence_case": false_confidence,
        "scientific_truth_assigned": False,
    }


def _normalized_field_value(field: str, value: Any) -> Any:
    if field in CITATION_FIELDS:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return ()
        return tuple(sorted({str(item) for item in value}))
    return value


def _standalone_review_contract_errors(review: Mapping[str, Any]) -> list[str]:
    rows = review.get("condition_reviews")
    if not isinstance(rows, list) or not rows:
        return ["CONDITION_REVIEWS_MISSING"]
    errors: list[str] = []
    conditions: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"ROW_{index}_NOT_OBJECT")
            continue
        condition = str(row.get("condition", "")).strip()
        if not condition:
            errors.append(f"ROW_{index}_CONDITION_MISSING")
        conditions.append(condition)
        for field in ATOMIC_REVIEW_FIELDS:
            if not isinstance(row.get(field), bool):
                errors.append(f"ROW_{index}_{field}_INVALID")
        for field in CITATION_FIELDS:
            value = row.get(field)
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                errors.append(f"ROW_{index}_{field}_INVALID")
            elif len(value) != len(set(value)):
                errors.append(f"ROW_{index}_{field}_DUPLICATE")
        if not isinstance(row.get("rationale"), str) or not str(row.get("rationale", "")).strip():
            errors.append(f"ROW_{index}_RATIONALE_MISSING")
        ambiguities = row.get("scientific_ambiguities")
        if not isinstance(ambiguities, list) or any(
            not isinstance(item, str) or not item.strip() for item in ambiguities
        ):
            errors.append(f"ROW_{index}_SCIENTIFIC_AMBIGUITIES_INVALID")
    if len(conditions) != len(set(conditions)):
        errors.append("DUPLICATE_CONDITION")
    return errors


def compare_atomic_review_repeatability(
    runs: Sequence[Mapping[str, Any]],
    *,
    expected_packet_sha256: str | None = None,
    required_successful_runs: int = 3,
) -> dict[str, Any]:
    """Compare independent reviews exactly; never vote or synthesize truth."""

    if required_successful_runs < 2:
        raise ValueError("required_successful_runs must be at least 2")
    seen_run_ids: set[str] = set()
    successful: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    excluded: list[dict[str, Any]] = []
    packet_hashes: set[str] = set()
    model_identity_hashes: set[str] = set()
    profile_identity_hashes: set[str] = set()
    for raw in runs:
        if not isinstance(raw, Mapping):
            raise ValueError("every repeatability run must be an object")
        metadata = raw.get("run_metadata") if isinstance(raw.get("run_metadata"), Mapping) else raw
        run_id = _non_empty_string(metadata.get("run_id"), field="run_id")
        if run_id in seen_run_ids:
            raise ValueError(f"repeatability runs must be independent; duplicate run_id: {run_id}")
        seen_run_ids.add(run_id)
        packet_hash = _non_empty_string(metadata.get("packet_sha256"), field="packet_sha256")
        packet_hashes.add(packet_hash)
        if isinstance(metadata.get("model_identity"), Mapping):
            model_identity_hashes.add(canonical_json_sha256(metadata["model_identity"]))
        if isinstance(metadata.get("profile_identity"), Mapping):
            profile_identity_hashes.add(canonical_json_sha256(metadata["profile_identity"]))
        if expected_packet_sha256 and packet_hash != expected_packet_sha256:
            raise ValueError(f"repeatability packet hash mismatch for run {run_id}")
        invocation_status = str(metadata.get("invocation_status", "")).upper()
        review = raw.get("review")
        if invocation_status != "SUCCESS" or not isinstance(review, Mapping):
            excluded.append({"run_id": run_id, "invocation_status": invocation_status or "UNKNOWN"})
            continue
        contract_errors = _standalone_review_contract_errors(review)
        declared_validation = raw.get("review_validation_status")
        if declared_validation is not None and str(declared_validation).upper() != "PASS":
            contract_errors.append(f"DECLARED_{str(declared_validation).upper()}")
        if contract_errors:
            excluded.append(
                {
                    "run_id": run_id,
                    "invocation_status": invocation_status,
                    "exclusion_reason": "REVIEW_CONTRACT_INVALID",
                    "contract_errors": sorted(set(contract_errors)),
                }
            )
            continue
        declared_review_hash = metadata.get("review_sha256")
        if declared_review_hash is not None and declared_review_hash != canonical_json_sha256(review):
            raise ValueError(f"repeatability review hash mismatch for run {run_id}")
        successful.append((run_id, metadata, review))
    if len(packet_hashes) > 1:
        raise ValueError("repeatability runs do not bind the same packet")
    if len(model_identity_hashes) > 1:
        raise ValueError("repeatability runs do not use the same model identity")
    if len(profile_identity_hashes) > 1:
        raise ValueError("repeatability runs do not use the same profile identity")
    if len(successful) < required_successful_runs:
        return {
            "status": "INSUFFICIENT_SUCCESSFUL_RUNS",
            "required_successful_run_count": required_successful_runs,
            "successful_run_count": len(successful),
            "excluded_runs": excluded,
            "exact_agreement": None,
            "disagreement_fields": [],
            "condition_level_instability": [],
            "majority_vote_used": False,
        }
    selected = successful[:required_successful_runs]
    row_maps = [(run_id, _review_rows(review)) for run_id, _, review in selected]
    condition_sets = [set(rows) for _, rows in row_maps]
    if any(value != condition_sets[0] for value in condition_sets[1:]):
        raise ValueError("repeatability reviews do not contain the same conditions")
    comparisons: list[dict[str, Any]] = []
    material_disagreements: list[dict[str, Any]] = []
    citation_disagreements: list[dict[str, Any]] = []
    for condition in sorted(condition_sets[0]):
        rows = [(run_id, mapping[condition]) for run_id, mapping in row_maps]
        for field in (*ATOMIC_REVIEW_FIELDS, *CITATION_FIELDS):
            values = {
                run_id: _normalized_field_value(field, row.get(field))
                for run_id, row in rows
            }
            agreement = len({repr(value) for value in values.values()}) == 1
            comparison = {
                "condition": condition,
                "field": field,
                "agreement": agreement,
                "values_by_run": values,
            }
            comparisons.append(comparison)
            if not agreement:
                if field in CITATION_FIELDS:
                    citation_disagreements.append(comparison)
                else:
                    material_disagreements.append(comparison)
    unstable_conditions = sorted(
        {
            item["condition"]
            for item in (*material_disagreements, *citation_disagreements)
        }
    )
    status = (
        "FLOW_EXPERT_UNSTABLE"
        if material_disagreements or citation_disagreements
        else "STABLE"
    )
    return {
        "status": status,
        "required_successful_run_count": required_successful_runs,
        "successful_run_count": len(successful),
        "compared_run_ids": [run_id for run_id, _, _ in selected],
        "excluded_runs": excluded,
        "packet_sha256": next(iter(packet_hashes), None),
        "exact_agreement": not material_disagreements and not citation_disagreements,
        "atomic_field_agreement": not material_disagreements,
        "citation_agreement": not citation_disagreements,
        "disagreement_fields": material_disagreements,
        "citation_disagreements": citation_disagreements,
        "condition_level_instability": unstable_conditions,
        "human_calibration_queue": unstable_conditions,
        "majority_vote_used": False,
        "extra_successful_runs_not_compared": [
            run_id for run_id, _, _ in successful[required_successful_runs:]
        ],
    }


def write_immutable_scientific_expert_run(
    output_root: str | Path,
    *,
    review_packet: Mapping[str, Any],
    review: Mapping[str, Any] | None,
    invocation_status: str,
    model_identity: Mapping[str, Any],
    profile_identity: Mapping[str, Any],
    invocation_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one validation attempt with immutable packet/review bindings."""

    status = _non_empty_string(invocation_status, field="invocation_status").upper()
    model = dict(model_identity)
    profile = dict(profile_identity)
    if not model:
        raise ValueError("model_identity must not be empty")
    if not profile:
        raise ValueError("profile_identity must not be empty")
    invocation = dict(invocation_metadata or {})
    forbidden = _find_forbidden_keys(
        invocation,
        {
            "access_token",
            "api-key",
            "api_key",
            "authorization",
            "credential",
            "credentials",
            "token",
        },
    )
    if forbidden:
        raise ValueError(f"invocation metadata contains credential fields: {forbidden}")
    run_id, run_dir, created_at = create_scientific_run(output_root)
    packet = copy.deepcopy(dict(review_packet))
    review_value = copy.deepcopy(dict(review)) if review is not None else None
    packet_hash = write_immutable_json(run_dir / "review_packet.json", packet)
    review_hash = (
        write_immutable_json(run_dir / "scientific_expert_review.json", review_value)
        if review_value is not None
        else None
    )
    invocation_record = {
        "run_id": run_id,
        "created_at": created_at,
        "invocation_status": status,
        "model_identity": model,
        "profile_identity": profile,
        "metadata": invocation,
    }
    invocation_hash = write_immutable_json(run_dir / "invocation.json", invocation_record)
    manifest = {
        "run_id": run_id,
        "created_at": created_at,
        "invocation_status": status,
        "model_identity": model,
        "profile_identity": profile,
        "packet_sha256": packet_hash,
        "review_sha256": review_hash,
        "invocation_sha256": invocation_hash,
        "artifact_hash_algorithm": "CANONICAL_JSON_SHA256",
        "scientific_truth_assigned": False,
        "lifecycle_gate_advanced": False,
    }
    write_immutable_json(run_dir / "run_manifest.json", manifest)
    return {**manifest, "run_directory": str(run_dir)}


def _find_forbidden_keys(value: Any, forbidden: set[str], path: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child = f"{path}.{key_text}"
            if key_text.casefold() in forbidden:
                failures.append(child)
            failures.extend(_find_forbidden_keys(item, forbidden, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            failures.extend(_find_forbidden_keys(item, forbidden, f"{path}[{index}]"))
    return failures


def _project_fields(value: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: copy.deepcopy(value[field]) for field in fields if field in value}


def _sanitized_evidence(packet: Mapping[str, Any]) -> dict[str, Any]:
    claims = packet.get("scientific_claims") or packet.get("scientific_support_claims") or ()
    evidence = packet.get("evidence_records") or packet.get("current_evidence_records") or ()
    sources = packet.get("source_records") or packet.get("candidate_source_records") or ()
    links = packet.get("claim_support_links") or ()
    return {
        "scientific_claims": [
            _project_fields(item, _SAFE_CLAIM_FIELDS)
            for item in claims
            if isinstance(item, Mapping)
        ],
        "evidence_records": [
            _project_fields(item, _SAFE_EVIDENCE_FIELDS)
            for item in evidence
            if isinstance(item, Mapping)
        ],
        "source_records": [
            _project_fields(item, _SAFE_SOURCE_FIELDS)
            for item in sources
            if isinstance(item, Mapping)
        ],
        "claim_support_links": [
            _project_fields(item, _SAFE_LINK_FIELDS)
            for item in links
            if isinstance(item, Mapping)
        ],
    }


def _packet_identity(packet: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    dataset_identity = packet.get("dataset_identity")
    family_identity = packet.get("family_identity")
    dataset_id = (
        dataset_identity.get("dataset_id")
        if isinstance(dataset_identity, Mapping)
        else packet.get("dataset_id")
    )
    family_id = (
        family_identity.get("family_id")
        if isinstance(family_identity, Mapping)
        else packet.get("family_id")
    )
    concept_id = (
        family_identity.get("concept_id")
        if isinstance(family_identity, Mapping)
        else packet.get("concept_id")
    )
    return dataset_id, family_id, concept_id


def _responsibility_from_condition(condition: Mapping[str, Any]) -> tuple[Any, Any]:
    condition_id = _condition_id(condition).upper()
    operationalization = condition.get("operationalization_responsibility")
    finding = condition.get("finding_responsibility") or condition.get("finding_openness")
    if operationalization is None:
        operationalization = (
            "user_specified"
            if condition_id.startswith("O1")
            else "partially_specified"
            if condition_id.startswith("O2")
            else "model_selected"
            if condition_id.startswith("O3")
            else None
        )
    if finding is None:
        finding = "bounded" if condition_id.endswith("F1") else "open" if condition_id.endswith("F2") else None
    return operationalization, finding


def _condition_from_item(item: Mapping[str, Any], packet: Mapping[str, Any]) -> Mapping[str, Any]:
    requested = _non_empty_string(item.get("condition"), field="calibration item condition")
    for row in _condition_rows(packet):
        if _condition_id(row) == requested:
            return row
    raise ValueError(f"calibration condition not present in packet: {requested}")


def audit_human_calibration_set(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check selection coverage without exposing strata in the human packet."""

    rows = _sequence(items, field="calibration items")
    if not 8 <= len(rows) <= 12:
        raise ValueError("human calibration set must contain 8 to 12 items")
    strata: set[str] = set()
    concepts: set[str] = set()
    disagreement_available = False
    disagreement_selected = False
    for item in rows:
        if not isinstance(item, Mapping):
            raise ValueError("every calibration item must be an object")
        stratum = _non_empty_string(item.get("selection_stratum"), field="selection_stratum")
        strata.add(stratum)
        packet = item.get("review_packet")
        if not isinstance(packet, Mapping):
            raise ValueError("every calibration item requires review_packet")
        _, family_id, concept_id = _packet_identity(packet)
        concepts.add(str(concept_id or family_id or ""))
        if item.get("model_disagreement_available") is True:
            disagreement_available = True
        if item.get("selected_for_model_disagreement") is True:
            disagreement_selected = True
    required = {
        "SUPPORTED_ORIGINAL",
        "LEGITIMATE_O2_OPENNESS",
        "TARGET_DRIFT",
        "EVIDENCE_INSUFFICIENCY",
    }
    missing = sorted(required - strata)
    if len({value for value in concepts if value}) < 2:
        missing.append("MULTIPLE_SCIENTIFIC_CONCEPTS")
    if disagreement_available and not disagreement_selected:
        missing.append("AVAILABLE_MODEL_DISAGREEMENT")
    return {
        "status": "PASS" if not missing else "INCOMPLETE",
        "item_count": len(rows),
        "selection_strata": sorted(strata),
        "distinct_concept_count": len({value for value in concepts if value}),
        "missing_coverage": missing,
        "model_disagreement_included_when_available": not disagreement_available or disagreement_selected,
    }


def build_blinded_human_calibration_packet(
    items: Sequence[Mapping[str, Any]],
    *,
    calibration_packet_id: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the human view without model outputs or expected mutation labels."""

    coverage = audit_human_calibration_set(items)
    if coverage["status"] != "PASS":
        raise ValueError(f"human calibration set coverage incomplete: {coverage['missing_coverage']}")
    packet_id = _non_empty_string(calibration_packet_id, field="calibration_packet_id")
    output_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(items, start=1):
        item = dict(raw)
        review_packet = item.get("review_packet")
        if not isinstance(review_packet, Mapping):
            raise ValueError("every calibration item requires review_packet")
        packet_validation = validate_scientific_review_packet(review_packet)
        if packet_validation.get("status") != "PASS":
            raise ValueError(
                "human calibration requires a valid canonical review packet: "
                + ", ".join(packet_validation.get("reason_codes", ()))
            )
        condition = _condition_from_item(item, review_packet)
        item_id = f"calibration-{index:03d}"
        if item_id in seen:
            raise ValueError(f"duplicate calibration_item_id: {item_id}")
        seen.add(item_id)
        dataset_id, family_id, concept_id = _packet_identity(review_packet)
        operationalization_responsibility, finding_responsibility = _responsibility_from_condition(condition)
        entry = {
            "calibration_item_id": item_id,
            "dataset_id": dataset_id,
            "family_id": family_id,
            "concept_id": concept_id,
            "condition": _condition_id(condition),
            "scientific_target": condition.get("scientific_target") or review_packet.get("scientific_target"),
            "question": condition.get("question"),
            "operationalization_responsibility": operationalization_responsibility,
            "finding_responsibility": finding_responsibility,
            "finding_goal": condition.get("finding_goal"),
            "fixed_o_dimensions": copy.deepcopy(
                condition.get("fixed_o_dimensions", condition.get("fixed_dimensions", []))
            ),
            "unresolved_o_dimensions": copy.deepcopy(
                condition.get("unresolved_o_dimensions", condition.get("unresolved_dimensions", []))
            ),
            "finding_requirements": copy.deepcopy(condition.get("finding_requirements", [])),
            "authored_operationalizations": copy.deepcopy(
                condition.get("authored_operationalizations", [])
            ),
            "materialization_route": copy.deepcopy(condition.get("materialization_route", {})),
            "dataset_context": copy.deepcopy(review_packet.get("dataset_context", {})),
            "observable_semantics": copy.deepcopy(
                review_packet.get("observable_semantics", review_packet.get("dataset_metadata", {}))
            ),
            "sanitized_evidence": _sanitized_evidence(review_packet),
            "atomic_judgment_fields": list(ATOMIC_REVIEW_FIELDS),
            "judgments": {field: None for field in ATOMIC_REVIEW_FIELDS},
            "scientific_rationale": None,
        }
        forbidden = _find_forbidden_keys(entry, _CALIBRATION_FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(f"blinded calibration item contains forbidden fields: {forbidden}")
        output_items.append(entry)
    result = {
        "calibration_packet_id": packet_id,
        "created_at": created_at or _utc_now(),
        "blind_to_model_review": True,
        "item_count": len(output_items),
        "atomic_judgment_fields": list(ATOMIC_REVIEW_FIELDS),
        "instructions": (
            "Label each atomic field independently as true, false, or UNCERTAIN and optionally provide scientific rationale."
        ),
        "items": output_items,
        "lifecycle_gate_advanced": False,
    }
    forbidden = _find_forbidden_keys(result, _CALIBRATION_FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(f"blinded calibration packet contains forbidden fields: {forbidden}")
    return result


def validate_external_human_labels(
    calibration_packet: Mapping[str, Any],
    human_labels: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    packet_id = _non_empty_string(
        calibration_packet.get("calibration_packet_id"), field="calibration_packet_id"
    )
    if human_labels.get("calibration_packet_id") != packet_id:
        raise ValueError("human labels do not bind the calibration packet")
    _non_empty_string(human_labels.get("reviewer_id"), field="reviewer_id")
    _non_empty_string(human_labels.get("completed_at"), field="completed_at")
    expected_items = {
        str(item.get("calibration_item_id"))
        for item in _sequence(calibration_packet.get("items"), field="calibration packet items")
        if isinstance(item, Mapping)
    }
    rows: dict[str, dict[str, Any]] = {}
    for raw in _sequence(human_labels.get("labels"), field="human labels"):
        if not isinstance(raw, Mapping):
            raise ValueError("every human label must be an object")
        item_id = _non_empty_string(raw.get("calibration_item_id"), field="calibration_item_id")
        if item_id not in expected_items:
            raise ValueError(f"human labels contain unknown calibration item: {item_id}")
        if item_id in rows:
            raise ValueError(f"duplicate human label: {item_id}")
        judgments = raw.get("judgments")
        if not isinstance(judgments, Mapping):
            raise ValueError(f"human label {item_id} requires judgments")
        if set(judgments) != set(ATOMIC_REVIEW_FIELDS):
            raise ValueError(f"human label {item_id} must label every atomic field")
        normalized: dict[str, Any] = {}
        for field, value in judgments.items():
            if isinstance(value, bool):
                normalized[field] = value
            elif str(value).upper() == "UNCERTAIN":
                normalized[field] = "UNCERTAIN"
            else:
                raise ValueError(f"invalid human label for {item_id}.{field}: {value!r}")
        rows[item_id] = {
            "judgments": normalized,
            "scientific_rationale": raw.get("scientific_rationale"),
        }
    missing = sorted(expected_items - set(rows))
    if missing:
        raise ValueError(f"human labels are incomplete: {missing}")
    return rows


def compare_external_human_labels(
    calibration_packet: Mapping[str, Any],
    model_reviews: Mapping[str, Mapping[str, Any]],
    human_labels: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare independent labels after collection; never create a new gate."""

    if human_labels is None:
        return {
            "status": "PENDING",
            "human_labels_supplied": False,
            "item_count": calibration_packet.get("item_count"),
            "field_level_agreement": None,
            "condition_level_agreement": None,
            "disagreements": [],
            "model_uncertainty_vs_human": [],
            "false_confidence_cases": [],
            "lifecycle_gate_advanced": False,
        }
    labels = validate_external_human_labels(calibration_packet, human_labels)
    expected_items = set(labels)
    if set(model_reviews) != expected_items:
        missing = sorted(expected_items - set(model_reviews))
        extra = sorted(set(model_reviews) - expected_items)
        raise ValueError(f"model review item mismatch; missing={missing}, extra={extra}")
    comparisons: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    model_uncertainty: list[dict[str, Any]] = []
    false_confidence: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    for item_id in sorted(expected_items):
        model = model_reviews[item_id]
        human = labels[item_id]["judgments"]
        if model.get("model_review_status", "PASS") != "PASS":
            raise ValueError(
                f"model review is not comparable for {item_id}: "
                f"{model.get('model_review_status')}"
            )
        explicit_model_uncertainty = (
            model.get("model_uncertainty_explicit") is True
            or model.get("eligibility_recommendation") == "MORE_EVIDENCE_REQUIRED"
            or bool(model.get("scientific_ambiguities", ()) or ())
        )
        if explicit_model_uncertainty:
            model_uncertainty.append(
                {
                    "calibration_item_id": item_id,
                    "eligibility_recommendation": model.get(
                        "eligibility_recommendation"
                    ),
                    "scientific_ambiguities": list(
                        model.get("scientific_ambiguities", ()) or ()
                    ),
                    "human_uncertain_fields": sorted(
                        field
                        for field, value in human.items()
                        if value == "UNCERTAIN"
                    ),
                }
            )
        item_agrees = True
        for field in ATOMIC_REVIEW_FIELDS:
            model_value = model.get(field)
            if not isinstance(model_value, bool):
                raise ValueError(f"invalid model review value for {item_id}.{field}: {model_value!r}")
            human_value = human[field]
            agreement = model_value == human_value
            row = {
                "calibration_item_id": item_id,
                "field": field,
                "model": model_value,
                "human": human_value,
                "agreement": agreement,
            }
            comparisons.append(row)
            if not agreement:
                item_agrees = False
                disagreements.append(row)
            if not agreement and not explicit_model_uncertainty:
                false_confidence.append(row)
        condition_rows.append(
            {"calibration_item_id": item_id, "all_atomic_fields_agree": item_agrees}
        )
    agreement_count = sum(row["agreement"] for row in comparisons)
    condition_agreement_count = sum(row["all_atomic_fields_agree"] for row in condition_rows)
    return {
        "status": "COMPLETE",
        "human_labels_supplied": True,
        "reviewer_id": human_labels.get("reviewer_id"),
        "completed_at": human_labels.get("completed_at"),
        "item_count": len(condition_rows),
        "field_level_agreement": {
            "agreement_count": agreement_count,
            "comparison_count": len(comparisons),
            "rate": agreement_count / len(comparisons) if comparisons else None,
        },
        "condition_level_agreement": {
            "agreement_count": condition_agreement_count,
            "comparison_count": len(condition_rows),
            "rate": condition_agreement_count / len(condition_rows) if condition_rows else None,
        },
        "disagreements": disagreements,
        "model_uncertainty_vs_human": model_uncertainty,
        "false_confidence_cases": false_confidence,
        "scientific_validation_claim": False,
        "lifecycle_gate_advanced": False,
    }


__all__ = [
    "ATOMIC_REVIEW_FIELDS",
    "CITATION_FIELDS",
    "MUTATION_TYPES",
    "ablate_scientific_evidence",
    "audit_human_calibration_set",
    "audit_semantic_mutation_reviews",
    "build_blinded_human_calibration_packet",
    "build_semantic_mutation_suite",
    "compare_atomic_review_repeatability",
    "compare_evidence_ablation_behavior",
    "compare_external_human_labels",
    "validate_external_human_labels",
    "validate_review_packet_provenance",
    "write_immutable_scientific_expert_run",
]

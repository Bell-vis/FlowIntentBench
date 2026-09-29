"""Case-grounded family invariants and fail-closed release checks."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


def _canonical_target(value: Any) -> str | None:
    if isinstance(value, Mapping):
        canonical = value.get("canonical_id")
        if canonical:
            return str(canonical)
        value = value.get("description")
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.casefold()
    if "high-speed" in text or "high speed" in text:
        return "high_speed_flow_region"
    if "turbulence" in text:
        return "turbulence_activity_hotspot"
    if "heterogeneity" in text or "concentration" in text:
        return "concentration_heterogeneity"
    if "density" in text:
        return "density_feature"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or None


def _clause_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        canonical = value.get("canonical_id")
        return str(canonical) if canonical else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def normalize_resolved_clauses(value: Any) -> dict[str, dict[str, Any]]:
    """Return a stable dimension -> canonical clause map."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = {
            str(item.get("dimension_id", item.get("dimension", ""))): item
            for item in value if isinstance(item, Mapping) and item.get("dimension_id", item.get("dimension"))
        }
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for dimension, clause in value.items():
        if isinstance(clause, Mapping):
            item = dict(clause)
        else:
            item = {"description": str(clause)}
        canonical = _clause_id(clause)
        if canonical:
            item.setdefault("canonical_id", canonical)
        normalized[str(dimension)] = item
    return normalized


def canonicalize_operationalization_clause(dimension: str, statement: str) -> dict[str, Any]:
    """Canonicalize authored clause text using the shared construction vocabulary."""

    text = str(statement).casefold()
    if dimension == "feature_definition":
        canonical = "face_connected_region" if any(token in text for token in ("face-connected", "shared grid faces", "6-neighbor")) else "legacy_feature_definition"
    elif dimension == "criterion":
        canonical = "speed_gt_0_20" if "0.20" in text and (">" in text or "above" in text or "greater than" in text) else "speed_q90_nonzero" if "90th percentile" in text else "legacy_criterion"
    elif dimension == "property_measure":
        canonical = "peak_speed" if "peak speed" in text else "mean_speed" if "mean speed" in text else "legacy_property_measure"
    elif dimension == "aggregation_or_representation":
        canonical = "mean_region_coordinate" if "mean spatial" in text or "mean coordinate" in text else "peak_point_coordinate" if "peak-speed location" in text else "legacy_representation"
    else:
        canonical = "legacy_" + re.sub(r"[^a-z0-9]+", "_", str(dimension)).strip("_")
    return {"canonical_id": canonical, "description": statement, "provenance": "LEGACY_DERIVED"}


def validate_family_definition(family: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("family_id", "dataset_id", "scientific_target", "principal_dimensions"):
        if not family.get(field):
            errors.append(f"missing {field}")
    principal = set(str(x) for x in family.get("principal_dimensions", []))
    unresolved = set(str(x) for x in family.get("o2_unresolved_dimensions", family.get("unresolved_dimensions", [])))
    if not unresolved <= principal:
        errors.append("O2 unresolved dimensions must be principal dimensions")
    if len(unresolved) > 2:
        errors.append("O2 may leave at most two principal dimensions unresolved")
    case_records = family.get("controlled_case_records", []) or []
    if isinstance(case_records, Sequence) and case_records:
        for record in case_records:
            if not isinstance(record, Mapping) or str(record.get("dataset_id")) != str(family.get("dataset_id")):
                errors.append("controlled case belongs to another dataset")
    if not family.get("concept_id") and family.get("family_id"):
        errors.append("missing concept_id for dataset concept family")
    return errors


def validate_reference_space_semantics(reference: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate sparse reference-O metadata without assuming Cartesian closure."""

    if not isinstance(reference, Mapping):
        return {"status": "NOT_EVALUABLE", "errors": ["reference space metadata is missing"], "accepted_count": 0, "rejected_count": 0}
    accepted = reference.get("accepted", reference.get("candidate_reference_operationalizations", [])) or []
    rejected = reference.get("explicit_rejected", reference.get("rejected", [])) or []
    errors: list[str] = []
    enforce_anchor_schema = bool(reference.get("require_anchor_schema", False)) or any(
        isinstance(item, Mapping) and "reference_o_id" in item for item in accepted
    ) if isinstance(accepted, Sequence) and not isinstance(accepted, (str, bytes)) else False
    if enforce_anchor_schema and isinstance(accepted, Sequence) and not isinstance(accepted, (str, bytes)):
        for index, item in enumerate(accepted):
            if not isinstance(item, Mapping):
                errors.append(f"accepted[{index}] must be a reference anchor object")
                continue
            for key in ("reference_o_id", "family_id", "source_case_id", "provenance", "validity_status"):
                if not str(item.get(key, "")).strip():
                    errors.append(f"accepted[{index}] missing anchor field {key}")
            if item.get("validity_status") not in {"ENUMERATED_VALID", "VALID", "LEGACY_VALID"}:
                errors.append(f"accepted[{index}] must carry an explicit valid status")
    for index, item in enumerate(rejected):
        if not isinstance(item, Mapping) or not str(item.get("rationale", "")).strip():
            errors.append(f"explicit_rejected[{index}] requires a rejection rationale")
    omitted = reference.get("omitted_combinations", reference.get("omitted_semantics"))
    if omitted is None:
        errors.append("omitted combinations must be explicitly marked UNENUMERATED")
    elif str(omitted).upper() != "UNENUMERATED":
        errors.append("omitted combinations must remain UNENUMERATED")
    accepted_count = len(accepted) if isinstance(accepted, Sequence) and not isinstance(accepted, (str, bytes)) else 0
    return {
        "status": "FAIL" if errors else ("NOT_EVALUABLE_NO_REFERENCE_ANCHOR" if accepted_count == 0 else "PASS"),
        "errors": errors,
        "accepted_count": accepted_count,
        "rejected_count": len(rejected) if isinstance(rejected, Sequence) and not isinstance(rejected, (str, bytes)) else 0,
        "reference_space_is_non_exhaustive": True,
        "cartesian_product_required": False,
    }


def validate_reference_space(reference: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compatibility alias for callers that use the shorter validator name."""

    return validate_reference_space_semantics(reference)


def resolved_dimension_invariance(
    branches: Sequence[Mapping[str, Any]],
    *,
    baseline_clauses: Mapping[str, Any] | None = None,
    principal_dimensions: Sequence[str] | None = None,
    unresolved_dimensions: Sequence[str] | None = None,
    family_id: str | None = None,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Validate actual accepted branches against the family baseline."""

    if not branches:
        return {"status": "NOT_EVALUABLE", "reason": "no accepted branches", "branch_count": 0}
    branch_maps = [normalize_resolved_clauses(item.get("resolved_operationalization_clauses", item.get("resolved_clauses", {}))) for item in branches]
    if not any(branch_maps):
        # Legacy callers may provide only dimension names; this remains a
        # structural check but is explicitly marked as derived.
        branch_maps = [{str(x): {"canonical_id": str(x), "provenance": "LEGACY_DERIVED"} for x in item.get("resolved_dimensions", [])} for item in branches]
    baseline = normalize_resolved_clauses(baseline_clauses) if baseline_clauses else branch_maps[0]
    # When a case explicitly supplies principal dimensions, its openness is
    # authoritative.  In particular O3 may intentionally resolve nothing;
    # falling back to the family baseline in that case creates false missing
    # dimension errors.  Legacy callers without case metadata retain the
    # baseline-derived behavior.
    has_case_dimensions = principal_dimensions is not None
    principal = {str(item) for item in (principal_dimensions if has_case_dimensions else baseline.keys())}
    unresolved = {str(item) for item in (unresolved_dimensions or ())}
    resolved_required = principal - unresolved
    baseline_missing = sorted(resolved_required - set(baseline))
    if baseline_missing:
        return {
            "status": "FAIL",
            "failure_code": "INVALID_FAMILY_BASELINE",
            "family_id": family_id,
            "case_id": case_id,
            "resolved_operationalization_clauses": baseline,
            "branch_count": len(branches),
            "drifts": [],
            "missing_baseline_dimensions": baseline_missing,
        }
    dimensions = set(resolved_required) if has_case_dimensions else set(resolved_required or baseline)
    drifts: list[dict[str, Any]] = []
    branch_results: list[dict[str, Any]] = []
    for index, current in enumerate(branch_maps):
        missing_for_branch: list[str] = []
        drifted_for_branch: list[str] = []
        for dimension in dimensions:
            if dimension not in current:
                missing_for_branch.append(dimension)
                drifts.append({"branch_index": index, "dimension": dimension, "expected": baseline[dimension], "actual": None, "code": "MISSING_RESOLVED_DIMENSION", "family_id": family_id, "case_id": branches[index].get("case_id", case_id) if isinstance(branches[index], Mapping) else case_id, "branch_id": branches[index].get("branch_id") if isinstance(branches[index], Mapping) else None})
            elif (
                current[dimension].get("canonical_id") != baseline[dimension].get("canonical_id")
                or (
                    current[dimension].get("normalized_scientific_meaning")
                    and baseline[dimension].get("normalized_scientific_meaning")
                    and current[dimension].get("normalized_scientific_meaning") != baseline[dimension].get("normalized_scientific_meaning")
                )
            ):
                drifted_for_branch.append(dimension)
                drifts.append({"branch_index": index, "dimension": dimension, "expected": baseline[dimension], "actual": current[dimension], "code": "RESOLVED_DIMENSION_DRIFT", "family_id": family_id, "case_id": branches[index].get("case_id", case_id) if isinstance(branches[index], Mapping) else case_id, "branch_id": branches[index].get("branch_id") if isinstance(branches[index], Mapping) else None})
        branch_results.append({
            "branch_id": branches[index].get("branch_id") if isinstance(branches[index], Mapping) else None,
            "case_id": branches[index].get("case_id", case_id) if isinstance(branches[index], Mapping) else case_id,
            "missing_resolved_dimensions": sorted(missing_for_branch),
            "drifted_resolved_dimensions": sorted(drifted_for_branch),
            "status": "PASS" if not missing_for_branch and not drifted_for_branch else "FAIL",
        })
    return {
        "status": "FAIL" if drifts else "PASS",
        "failure_code": next((item.get("code") for item in drifts), None),
        "resolved_operationalization_clauses": baseline,
        "branch_count": len(branches),
        "drifts": drifts,
        "branch_results": branch_results,
        "principal_dimensions": sorted(principal),
        "unresolved_dimensions": sorted(unresolved),
        "resolved_required_dimensions": sorted(dimensions),
    }


def audit_o3_target_alignment_from_cases(
    family: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
    *, required_conditions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare actual O1/O2/O3 case targets, preserving historical drift."""

    baseline = family.get("model_visible_scientific_target") or family.get("scientific_target")
    baseline_id = _canonical_target(baseline)
    rows: list[dict[str, Any]] = []
    for case in cases:
        metadata = case.get("metadata", case.get("case_construction_metadata", {}))
        if not isinstance(metadata, Mapping):
            metadata = {}
        question = case.get("scientific_question")
        if question is None and isinstance(case.get("case_input"), Mapping):
            question = case["case_input"].get("scientific_question")
        target_value = metadata.get("model_visible_scientific_target") or question or metadata.get("scientific_target")
        target_id = _canonical_target(target_value)
        # A concept-level high-speed baseline is explicit even when legacy
        # metadata contains only prose such as "ventilation regions".
        if str(family.get("concept_id", "")) == "high_speed_region" and isinstance(question, str):
            target_id = "high_speed_flow_region" if re.search(r"high[- ]speed", question, re.I) else "generic_flow_region"
        rows.append({"case_id": case.get("case_id"), "condition": case.get("condition"), "target": target_id, "provenance": "AUTHORED" if metadata.get("model_visible_scientific_target") else "LEGACY_DERIVED"})
    drift = [row for row in rows if row.get("target") and baseline_id and row["target"] != baseline_id]
    present = {row.get("condition") for row in rows}
    missing = sorted(set(required_conditions or ()) - present)
    unknown = [row for row in rows if not row.get("target")]
    status = "FAIL" if drift else "NOT_EVALUABLE" if not rows or missing or unknown or not baseline_id else "PASS"
    return {"status": status, "baseline_target": baseline_id, "cases": rows, "drift_cases": drift, "missing_conditions": missing, "failure_code": "O3_TARGET_DRIFT" if drift else None}


def evaluate_family_release_eligibility(
    family: Mapping[str, Any], audits: Mapping[str, Any],
    *, conditions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Authoritative, derived release predicate for a dataset family."""

    required = {
        "grounding_status": "GROUNDED",
        "curator_status": "CONFIRMED",
        "controlled_family_validation": "PASS",
        "o3_target_alignment": "PASS",
        "f2_construct_validation": "PASS",
        "resolved_dimension_invariance": "PASS",
        "data_contract_validation": "PASS",
        "reference_space_semantics": "PASS",
        "family_lifecycle_state": "RELEASE_ELIGIBLE",
    }
    if conditions is not None:
        from .experiment_scope import PRIMARY_CONDITIONS
        selected = set(conditions)
        if not selected or not selected <= set(PRIMARY_CONDITIONS):
            raise ValueError("release scope requires explicit supported conditions")
        if "O1-F2" not in selected:
            required.pop("f2_construct_validation")
        # Target fidelity still applies to every retained case, including a
        # fixed-O-only family. The target audit now compares that exact subset.
    blockers: list[str] = []
    if family.get("grounding_status") != required["grounding_status"]:
        blockers.append("grounding_status")
    if family.get("curator_status") != required["curator_status"]:
        blockers.append("curator_status")
    for key, expected in required.items():
        if key in {"grounding_status", "curator_status"}:
            continue
        value = audits.get(key)
        if value != expected:
            blockers.append(key)
    eligible = not blockers
    return {"release_eligible": eligible, "release_status": "ELIGIBLE" if eligible else "NOT_ELIGIBLE", "blockers": blockers, "predicate": required}


__all__ = [
    "audit_o3_target_alignment_from_cases",
    "canonicalize_operationalization_clause",
    "evaluate_family_release_eligibility",
    "normalize_resolved_clauses",
    "resolved_dimension_invariance",
    "validate_reference_space",
    "validate_reference_space_semantics",
    "validate_family_definition",
]

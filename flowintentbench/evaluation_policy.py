"""Compiled, explicit verification semantics for reference Findings.

This module is intentionally a thin sidecar compiler.  It does not add a new
GroundTruth ontology and it never inspects an evaluated-model response.  Its
only job is to turn authored ``VerificationSpec`` values plus construction
bindings into the exact deterministic rule that the evaluator is allowed to
execute.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .ground_truth import GroundTruth, ReferenceFinding, VerificationSpec


VERIFICATION_POLICY_VERSION = "finding-verification-policy-v1"
VERIFICATION_MODES = frozenset(
    {
        "semantic_only",
        "scalar_tolerance",
        "spatial_euclidean",
        "componentwise_vector",
        "exact_discrete_numeric",
        "exact_identity",
    }
)
TOLERANCE_AUTHORITIES = frozenset(
    {
        "DOMAIN_AUTHORED",
        "DATA_RESOLUTION_DERIVED",
        "REPORTING_PRECISION_DERIVED",
        "NOT_APPLICABLE",
    }
)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finding_role(finding: ReferenceFinding, locator: str | None) -> str:
    terminal = str(locator or "").strip().casefold().rsplit("/", 1)[-1]
    locator_roles = {
        "selected_region_size": "retained_points",
        "region_count": "region_count",
        "surface_area": "surface_area",
        "metric_value": "heterogeneity",
        "strength": "strength",
        "reported_location": "location",
        "area_weighted_centroid": "location",
        "coordinate_span": "extent",
        "level": "density",
        "field_name": "field",
        "q10": "q10",
        "q50": "q50",
        "q90": "q90",
    }
    if terminal in locator_roles:
        return locator_roles[terminal]
    return finding.category.value


def explicit_verification_mode(
    finding: ReferenceFinding,
    *,
    locator: str | None,
) -> str:
    """Return an authored/binding-derived mode without Python-type inference."""

    verification = finding.verification
    if verification is not None and verification.spatial_tolerance is not None:
        terminal = str(locator or "").strip().casefold().rsplit("/", 1)[-1]
        # Extents are axis-aligned bounds/lengths.  Their authored rule is a
        # component-wise tolerance, while point locations use Euclidean
        # distance.  The locator is the explicit representation authority;
        # Python list shape is never inspected here.
        return "componentwise_vector" if terminal in {"coordinate_span", "extent"} else "spatial_euclidean"
    if verification is not None and (
        verification.absolute_tolerance is not None
        or verification.relative_tolerance is not None
    ):
        return "scalar_tolerance"
    normalized_locator = str(locator or "").strip().casefold()
    if normalized_locator.endswith(("/selected_region_size", "/region_count")):
        return "exact_discrete_numeric"
    if normalized_locator.endswith("/field_name"):
        return "exact_identity"
    return "semantic_only"


def _finding_by_id(ground_truth: GroundTruth) -> dict[tuple[str, str], ReferenceFinding]:
    return {
        (branch.operationalization_id, finding.finding_id): finding
        for branch in ground_truth.findings_by_operationalization
        for finding in branch.findings
    }


def _operation_signature(ground_truth: GroundTruth) -> dict[str, str]:
    return {
        bundle.operationalization_id: _digest(
            sorted(
                (
                    decision.dimension.value,
                    " ".join(decision.statement.casefold().split()),
                )
                for decision in bundle.decisions
            )
        )
        for bundle in ground_truth.acceptable_operationalizations
    }


def compile_finding_verification_policy(
    *,
    ground_truth: GroundTruth,
    finding_execution_binding: Mapping[str, Any],
    scientific_target: str,
    semantic_contract_sha256: str,
    significant_figures: int = 3,
) -> dict[str, Any]:
    """Compile one case's complete evaluator verification sidecar.

    The stable policy key deliberately excludes condition and case ID so the
    same scientific Finding under the same Effective O receives the same
    policy in O1-F1 and O1-F2.
    """

    findings = _finding_by_id(ground_truth)
    operation_signatures = _operation_signature(ground_truth)
    rows: list[dict[str, Any]] = []
    for branch in finding_execution_binding.get("branches", ()):
        if not isinstance(branch, Mapping):
            continue
        operation_id = str(branch.get("operationalization_id", ""))
        for binding in branch.get("findings", ()):
            if not isinstance(binding, Mapping):
                continue
            finding_id = str(binding.get("finding_id", ""))
            finding = findings.get((operation_id, finding_id))
            if finding is None:
                raise ValueError(
                    f"verification binding references unknown Finding {operation_id}:{finding_id}"
                )
            locator = (
                str(binding.get("g_of_o_locator"))
                if binding.get("g_of_o_locator") is not None
                else None
            )
            mode = explicit_verification_mode(finding, locator=locator)
            verification = finding.verification or VerificationSpec()
            tolerance_authority = (
                "REPORTING_PRECISION_DERIVED"
                if mode == "scalar_tolerance"
                else "DOMAIN_AUTHORED"
                if mode == "spatial_euclidean"
                else "NOT_APPLICABLE"
            )
            formula_id = {
                "scalar_tolerance": "combined_absolute_plus_relative_v1",
                "spatial_euclidean": "euclidean_l2_v1",
                "componentwise_vector": "componentwise_abs_tolerance_v1",
                "exact_discrete_numeric": "exact_discrete_equality_v1",
                "exact_identity": "exact_identity_v1",
                "semantic_only": "semantic_judgment_v1",
            }[mode]
            role = _finding_role(finding, locator)
            # A null unit is not, by itself, evidence that a reader frame or
            # numeric scale is needed.  Only location/extent Findings consume
            # the case's explicit coordinate semantics; discrete counts and
            # descriptive stored-scale values remain unitless without asking
            # the runtime to infer a frame.
            locator_text = str(locator or "").casefold()
            frame_required = role in {"location", "extent"} or any(
                token in locator_text
                for token in ("coordinate", "centroid", "location", "extent")
            )
            # A coordinate/frame authority is meaningful only for spatial
            # representations.  Keep the rule explicit in the sidecar so a
            # runtime verifier can reject an unauthorized frame label without
            # guessing from a list-shaped value.
            coordinate_authority = (
                "CASE_READER_METADATA" if frame_required else "NOT_APPLICABLE"
            )
            policy_identity = {
                "dataset_id": ground_truth.dataset_id,
                "case_family_id": ground_truth.case_family_id,
                "scientific_target": " ".join(scientific_target.casefold().split()),
                "effective_o_semantic_sha256": operation_signatures.get(operation_id),
                "scientific_role": role,
                "g_of_o_locator": locator,
                "representation": finding.category.value,
                "canonical_unit": finding.unit,
            }
            policy_row = {
                    "operationalization_id": operation_id,
                    "finding_id": finding_id,
                    "scientific_finding_policy_key": _digest(policy_identity),
                    "policy_identity": policy_identity,
                    "verification_mode": mode,
                    "tolerance_authority": tolerance_authority,
                    "verification_parameters": verification.model_dump(mode="json"),
                    "formula_id": formula_id,
                    "formula_inputs": {
                        "reference_value": finding.value,
                        "resolved_verification_parameters": verification.model_dump(mode="json"),
                        "significant_figures": significant_figures if mode == "scalar_tolerance" else None,
                    },
                    "unit_frame_authority": {
                        "physical_unit": finding.unit,
                        "coordinate_or_numeric_scale": (
                            "REFERENCE_FINDING_UNIT"
                            if finding.unit is not None
                            else coordinate_authority
                        ),
                    },
                }
            policy_row["policy_digest"] = _digest(
                {
                    "scientific_finding_policy_key": policy_row[
                        "scientific_finding_policy_key"
                    ],
                    "verification_mode": mode,
                    "tolerance_authority": tolerance_authority,
                    "verification_parameters": policy_row["verification_parameters"],
                    "formula_id": formula_id,
                    "formula_inputs": policy_row["formula_inputs"],
                    "unit_frame_authority": policy_row["unit_frame_authority"],
                }
            )
            rows.append(policy_row)
    expected = set(findings)
    actual = {
        (row["operationalization_id"], row["finding_id"])
        for row in rows
    }
    if actual != expected:
        raise ValueError("verification policy does not cover every reference Finding")
    rows.sort(key=lambda row: (row["operationalization_id"], row["finding_id"]))
    payload = {
        "artifact_type": "FindingVerificationPolicy",
        "schema_version": VERIFICATION_POLICY_VERSION,
        "dataset_id": ground_truth.dataset_id,
        "case_id": ground_truth.case_id,
        "semantic_contract_sha256": semantic_contract_sha256,
        "runtime_inference_from_python_type_forbidden": True,
        "policies": rows,
    }
    payload["verification_policy_sha256"] = _digest(payload)
    return payload


def validate_finding_verification_policy(
    policy: Mapping[str, Any],
    *,
    ground_truth: GroundTruth | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    rows = policy.get("policies")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {"status": "FAIL", "errors": ["VERIFICATION_POLICIES_MISSING"]}
    keys: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("VERIFICATION_POLICY_ROW_INVALID")
            continue
        key = (str(row.get("operationalization_id", "")), str(row.get("finding_id", "")))
        if not all(key):
            errors.append("VERIFICATION_POLICY_IDENTITY_MISSING")
        elif key in keys:
            errors.append("VERIFICATION_POLICY_IDENTITY_DUPLICATE")
        keys.add(key)
        if row.get("verification_mode") not in VERIFICATION_MODES:
            errors.append("VERIFICATION_MODE_INVALID")
        if row.get("tolerance_authority") not in TOLERANCE_AUTHORITIES:
            errors.append("TOLERANCE_AUTHORITY_INVALID")
        expected_formula = {
            "scalar_tolerance": "combined_absolute_plus_relative_v1",
            "spatial_euclidean": "euclidean_l2_v1",
            "componentwise_vector": "componentwise_abs_tolerance_v1",
            "exact_discrete_numeric": "exact_discrete_equality_v1",
            "exact_identity": "exact_identity_v1",
            "semantic_only": "semantic_judgment_v1",
        }.get(str(row.get("verification_mode")))
        if row.get("formula_id") != expected_formula:
            errors.append("VERIFICATION_FORMULA_INVALID")
        if not isinstance(row.get("scientific_finding_policy_key"), str) or len(
            row.get("scientific_finding_policy_key", "")
        ) != 64:
            errors.append("SCIENTIFIC_FINDING_POLICY_KEY_INVALID")
        expected_policy_digest = _digest(
            {
                key: row.get(key)
                for key in (
                    "scientific_finding_policy_key",
                    "verification_mode",
                    "tolerance_authority",
                    "verification_parameters",
                    "formula_id",
                    "formula_inputs",
                    "unit_frame_authority",
                )
            }
        )
        if row.get("policy_digest") != expected_policy_digest:
            errors.append("VERIFICATION_POLICY_ROW_DIGEST_MISMATCH")
    if ground_truth is not None:
        expected = set(_finding_by_id(ground_truth))
        if keys != expected:
            errors.append("VERIFICATION_POLICY_COVERAGE_MISMATCH")
        if policy.get("case_id") != ground_truth.case_id:
            errors.append("VERIFICATION_POLICY_CASE_ID_MISMATCH")
    supplied_digest = policy.get("verification_policy_sha256")
    without_digest = dict(policy)
    without_digest.pop("verification_policy_sha256", None)
    if supplied_digest != _digest(without_digest):
        errors.append("VERIFICATION_POLICY_DIGEST_MISMATCH")
    return {"status": "PASS" if not errors else "FAIL", "errors": sorted(set(errors))}


def bind_policy_to_execution_binding(
    binding: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the execution binding with its explicit verifier authority attached."""

    index = policy_index(policy)
    enriched = json.loads(json.dumps(binding, ensure_ascii=False))
    for branch in enriched.get("branches", ()):
        operation_id = str(branch.get("operationalization_id", ""))
        for finding in branch.get("findings", ()):
            key = (operation_id, str(finding.get("finding_id", "")))
            row = index.get(key)
            if row is None:
                raise ValueError(f"verification policy missing execution binding {key}")
            finding.update(
                {
                    "scientific_finding_policy_key": row["scientific_finding_policy_key"],
                    "verification_policy_digest": row["policy_digest"],
                    "verification_mode": row["verification_mode"],
                    "verification_parameters": row["verification_parameters"],
                    "verification_formula_id": row["formula_id"],
                    "tolerance_authority": row["tolerance_authority"],
                    "unit_frame_authority": row["unit_frame_authority"],
                }
            )
    return enriched


def policy_index(policy: Mapping[str, Any] | None) -> dict[tuple[str, str], Mapping[str, Any]]:
    if not isinstance(policy, Mapping):
        return {}
    return {
        (str(row.get("operationalization_id", "")), str(row.get("finding_id", ""))): row
        for row in policy.get("policies", ())
        if isinstance(row, Mapping)
    }


__all__ = [
    "TOLERANCE_AUTHORITIES",
    "VERIFICATION_MODES",
    "VERIFICATION_POLICY_VERSION",
    "compile_finding_verification_policy",
    "explicit_verification_mode",
    "policy_index",
    "validate_finding_verification_policy",
]

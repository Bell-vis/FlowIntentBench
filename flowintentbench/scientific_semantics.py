"""Derived canonical scientific semantics for authored benchmark cases.

The projection in this module is read-only.  It deliberately derives from the
existing construction metadata and scientific-family records instead of
introducing another authored contract.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .case_design import CaseConstructionMetadata, FindingOpenness, normalize_case_text


SCIENTIFIC_SEMANTIC_PROJECTION_VERSION = "scientific-semantic-projection-v1"


class ScientificSemanticProjectionError(ValueError):
    """Raised when existing construction objects cannot form one projection."""


def _nonempty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ScientificSemanticProjectionError(f"{label} must be a non-empty string")
    return text


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificSemanticProjectionError(f"{label} must be an object")
    return value


def _json_value(value: Any) -> Any:
    """Return JSON-safe values without retaining model or enum wrappers."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ScientificSemanticProjectionError(
        f"scientific semantic value is not JSON serializable: {type(value).__name__}"
    )


def _source_objects(
    metadata_or_case: CaseConstructionMetadata | Mapping[str, Any],
    family_metadata: Mapping[str, Any] | None,
) -> tuple[CaseConstructionMetadata, Mapping[str, Any]]:
    if isinstance(metadata_or_case, CaseConstructionMetadata):
        raw_metadata: CaseConstructionMetadata | Mapping[str, Any] = metadata_or_case
        raw_family = family_metadata
    elif isinstance(metadata_or_case, Mapping) and isinstance(
        metadata_or_case.get("metadata"), Mapping
    ):
        raw_metadata = metadata_or_case["metadata"]
        raw_family = family_metadata or (
            metadata_or_case.get("family")
            if isinstance(metadata_or_case.get("family"), Mapping)
            else None
        )
    else:
        raw_metadata = metadata_or_case
        raw_family = family_metadata

    metadata = (
        raw_metadata
        if isinstance(raw_metadata, CaseConstructionMetadata)
        else CaseConstructionMetadata.model_validate(raw_metadata)
    )
    if raw_family is None:
        raise ScientificSemanticProjectionError(
            "family_metadata is required to derive family_id and concept_id"
        )
    return metadata, _mapping(raw_family, "family_metadata")


def _family_identity(
    metadata: CaseConstructionMetadata, family: Mapping[str, Any]
) -> tuple[str, str]:
    family_id = _nonempty(family.get("family_id"), "family_metadata.family_id")
    concept_id = _nonempty(family.get("concept_id"), "family_metadata.concept_id")
    family_dataset = str(family.get("dataset_id", "")).strip()
    if family_dataset and family_dataset != metadata.dataset_id:
        raise ScientificSemanticProjectionError(
            "family dataset_id does not match CaseConstructionMetadata.dataset_id"
        )
    if family_id != metadata.case_family_id:
        controlled = {
            str(item)
            for item in family.get("controlled_case_ids", ()) or ()
            if str(item).strip()
        }
        if metadata.case_id not in controlled:
            raise ScientificSemanticProjectionError(
                "family_id does not match case_family_id and the case is not bound by "
                "family_metadata.controlled_case_ids"
            )
    return family_id, concept_id


def _normalized_constraint_statements(items: Sequence[Any]) -> list[str]:
    return sorted(normalize_case_text(str(item.statement)) for item in items)


def _family_resolved_clauses(family: Mapping[str, Any]) -> Mapping[str, Any]:
    for field in ("resolved_operationalization_clauses", "baseline_resolved_clauses"):
        value = family.get(field)
        if isinstance(value, Mapping):
            return value
    return {}


def _resolved_operationalization(
    metadata: CaseConstructionMetadata,
    family: Mapping[str, Any],
    fixed_dimensions: set[str],
) -> list[dict[str, str]]:
    contract = metadata.responsibility_contract
    raw_contract_rows = (
        contract.get("resolved_operationalization_clauses")
        if isinstance(contract, Mapping)
        else None
    )
    rows_by_dimension: dict[str, Mapping[str, Any]] = {}
    if raw_contract_rows is not None:
        if not isinstance(raw_contract_rows, Sequence) or isinstance(
            raw_contract_rows, (str, bytes, bytearray)
        ):
            raise ScientificSemanticProjectionError(
                "responsibility_contract.resolved_operationalization_clauses must be a list"
            )
        for index, row in enumerate(raw_contract_rows):
            row = _mapping(row, f"resolved_operationalization_clauses[{index}]")
            dimension = _nonempty(row.get("dimension_id"), "resolved dimension_id")
            if dimension in rows_by_dimension:
                raise ScientificSemanticProjectionError(
                    f"duplicate resolved operationalization dimension: {dimension}"
                )
            rows_by_dimension[dimension] = row
        if set(rows_by_dimension) != fixed_dimensions:
            raise ScientificSemanticProjectionError(
                "resolved operationalization dimensions do not match the fixed dimension set"
            )
    else:
        family_clauses = _family_resolved_clauses(family)
        for dimension in fixed_dimensions:
            row = family_clauses.get(dimension)
            if not isinstance(row, Mapping):
                raise ScientificSemanticProjectionError(
                    f"no existing resolved clause for fixed dimension: {dimension}"
                )
            rows_by_dimension[dimension] = row

    result: list[dict[str, str]] = []
    for dimension in sorted(fixed_dimensions):
        row = rows_by_dimension[dimension]
        result.append(
            {
                "dimension_id": dimension,
                "canonical_id": _nonempty(
                    row.get("canonical_id"), f"resolved {dimension}.canonical_id"
                ),
                "normalized_meaning": normalize_case_text(
                    _nonempty(
                        row.get("normalized_meaning", row.get("description")),
                        f"resolved {dimension}.normalized_meaning",
                    )
                ),
            }
        )
    return result


def _unresolved_operationalization(
    metadata: CaseConstructionMetadata,
    unresolved_dimensions: set[str],
) -> list[dict[str, Any]]:
    contract = metadata.responsibility_contract
    raw_rows = (
        contract.get("unresolved_operationalization_dimensions")
        if isinstance(contract, Mapping)
        else None
    )
    if raw_rows is None:
        return [{"dimension_id": dimension} for dimension in sorted(unresolved_dimensions)]
    if not isinstance(raw_rows, Sequence) or isinstance(
        raw_rows, (str, bytes, bytearray)
    ):
        raise ScientificSemanticProjectionError(
            "responsibility_contract.unresolved_operationalization_dimensions must be a list"
        )
    rows_by_dimension: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(raw_rows):
        row = _mapping(row, f"unresolved_operationalization_dimensions[{index}]")
        dimension = _nonempty(row.get("dimension_id"), "unresolved dimension_id")
        if dimension in rows_by_dimension:
            raise ScientificSemanticProjectionError(
                f"duplicate unresolved operationalization dimension: {dimension}"
            )
        rows_by_dimension[dimension] = row
    if set(rows_by_dimension) != unresolved_dimensions:
        raise ScientificSemanticProjectionError(
            "responsibility contract unresolved dimensions do not match metadata"
        )
    result: list[dict[str, Any]] = []
    for dimension in sorted(unresolved_dimensions):
        row = rows_by_dimension[dimension]
        if row.get("must_remain_open") is not True:
            raise ScientificSemanticProjectionError(
                f"unresolved dimension must remain open: {dimension}"
            )
        result.append(
            {
                "dimension_id": dimension,
                "normalized_meaning": normalize_case_text(
                    _nonempty(
                        row.get("normalized_meaning"),
                        f"unresolved {dimension}.normalized_meaning",
                    )
                ),
                "must_remain_open": True,
            }
        )
    return result


def _finding_contract(family: Mapping[str, Any]) -> Mapping[str, Any]:
    for field in ("finding_requirement_contract", "finding_requirements"):
        value = family.get(field)
        if isinstance(value, Mapping):
            return value
    return {}


def _alternative_role_groups(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    raw_groups = contract.get("alternative_role_groups", ()) or ()
    if not isinstance(raw_groups, Sequence) or isinstance(
        raw_groups, (str, bytes, bytearray)
    ):
        raise ScientificSemanticProjectionError("alternative_role_groups must be a list")
    for index, raw in enumerate(raw_groups):
        group = _mapping(raw, f"alternative_role_groups[{index}]")
        roles = sorted(
            _nonempty(item, "alternative finding role")
            for item in group.get("roles", ()) or ()
        )
        minimum = group.get("min_required")
        if not isinstance(minimum, int) or isinstance(minimum, bool) or not roles:
            raise ScientificSemanticProjectionError(
                "alternative role group requires roles and integer min_required"
            )
        groups.append(
            {
                "group_id": _nonempty(group.get("group_id"), "alternative group_id"),
                "min_required": minimum,
                "roles": roles,
            }
        )
    return sorted(groups, key=lambda item: item["group_id"])


def _adequate_core_sets(contract: Mapping[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    raw_sets = contract.get("adequate_core_sets", ()) or ()
    if not isinstance(raw_sets, Sequence) or isinstance(
        raw_sets, (str, bytes, bytearray)
    ):
        raise ScientificSemanticProjectionError("adequate_core_sets must be a list")
    for raw in raw_sets:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ScientificSemanticProjectionError("each adequate core set must be a list")
        rows.append(sorted(_nonempty(role, "adequate-core role") for role in raw))
    return sorted(rows)


def _finding_semantics(
    metadata: CaseConstructionMetadata, family: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    mode = "F2" if metadata.finding_openness == FindingOpenness.OPEN else "F1"
    responsibility = (
        metadata.responsibility_contract.get("finding_responsibility")
        if isinstance(metadata.responsibility_contract, Mapping)
        else None
    )
    if isinstance(responsibility, Mapping):
        declared = _nonempty(responsibility.get("finding_mode"), "finding_mode")
        if declared != mode:
            raise ScientificSemanticProjectionError(
                "responsibility_contract finding_mode does not match finding_openness"
            )
        normalized_demand = normalize_case_text(
            _nonempty(responsibility.get("normalized_demand"), "normalized_demand")
        )
    else:
        normalized_demand = normalize_case_text(metadata.finding_goal)

    family_contract = _finding_contract(family)
    common = {
        "normalized_demand": normalized_demand,
        "finding_goal": normalize_case_text(metadata.finding_goal),
    }
    if mode == "F1":
        requirements = sorted(
            (
                item.category.value,
                normalize_case_text(item.statement),
            )
            for item in metadata.explicit_finding_requirements
        )
        role_by_category = family_contract.get("role_by_category")
        role_by_category = role_by_category if isinstance(role_by_category, Mapping) else {}
        required_roles = sorted(
            {
                str(role_by_category.get(category, f"category:{category}"))
                for category, _ in requirements
            }
        )
        return mode, {
            **common,
            "fixed_finding_requirements": [
                {"category": category, "normalized_meaning": meaning}
                for category, meaning in requirements
            ],
            "required_finding_roles": required_roles,
        }

    if not family_contract:
        raise ScientificSemanticProjectionError(
            "F2 projection requires the existing family finding requirement contract"
        )
    return mode, {
        **common,
        "adequate_core_semantics": {
            "scientific_entity_type": _nonempty(
                family_contract.get("scientific_entity_type"),
                "finding_requirement_contract.scientific_entity_type",
            ),
            "mandatory_roles": sorted(
                _nonempty(role, "mandatory finding role")
                for role in family_contract.get("mandatory_roles", ()) or ()
            ),
            "supporting_roles": sorted(
                _nonempty(role, "supporting finding role")
                for role in family_contract.get("supporting_roles", ()) or ()
            ),
            "adequate_core_sets": _adequate_core_sets(family_contract),
            "alternative_role_groups": _alternative_role_groups(family_contract),
            "role_by_category": {
                str(key): str(value)
                for key, value in sorted(
                    _mapping(
                        family_contract.get("role_by_category", {}),
                        "finding_requirement_contract.role_by_category",
                    ).items()
                )
            },
            "novel_role_policy": normalize_case_text(
                _nonempty(
                    family_contract.get("novel_role_policy"),
                    "finding_requirement_contract.novel_role_policy",
                )
            ),
        },
    }


def _verification_semantics(metadata: CaseConstructionMetadata) -> dict[str, Any]:
    contract = metadata.evaluation_representability_contract
    if not isinstance(contract, Mapping):
        return {
            "deterministic_verification_required_claim_types": [],
            "semantic_adjudication_required": False,
        }
    deterministic = contract.get("deterministic_verification")
    deterministic = deterministic if isinstance(deterministic, Mapping) else {}
    adjudication = contract.get("semantic_adjudication")
    adjudication = adjudication if isinstance(adjudication, Mapping) else {}
    return {
        "deterministic_verification_required_claim_types": sorted(
            _nonempty(item, "required deterministic claim type")
            for item in deterministic.get("required_claim_types", ()) or ()
        ),
        "semantic_adjudication_required": adjudication.get("required") is True,
    }


def build_scientific_semantic_projection(
    metadata_or_case: CaseConstructionMetadata | Mapping[str, Any],
    *,
    family_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive the canonical scientific meaning from current authored objects."""

    metadata, family = _source_objects(metadata_or_case, family_metadata)
    family_id, concept_id = _family_identity(metadata, family)
    principal = {item.value for item in metadata.principal_operationalization_dimensions}
    unresolved = {
        item.value for item in metadata.unresolved_operationalization_dimensions
    }
    fixed = principal - unresolved
    finding_mode, finding_semantics = _finding_semantics(metadata, family)
    required_capabilities = family.get("required_capabilities", {})
    semantic_capabilities = family.get("scientific_semantic_capabilities", {})
    projection = {
        "scientific_semantic_projection_version": SCIENTIFIC_SEMANTIC_PROJECTION_VERSION,
        "dataset_id": metadata.dataset_id,
        "family_id": family_id,
        "concept_id": concept_id,
        "scientific_target": normalize_case_text(metadata.scientific_target),
        "scientific_scope": {
            "scope_constraints": _normalized_constraint_statements(
                metadata.scope_constraints
            ),
            "condition_constraints": _normalized_constraint_statements(
                metadata.condition_constraints
            ),
            "selection_constraints": _normalized_constraint_statements(
                metadata.selection_constraints
            ),
        },
        "operationalization_responsibility": metadata.operationalization_responsibility.value,
        "principal_operationalization_dimensions": sorted(principal),
        "resolved_operationalization": _resolved_operationalization(
            metadata, family, fixed
        ),
        "unresolved_operationalization_dimensions": _unresolved_operationalization(
            metadata, unresolved
        ),
        "finding_responsibility": finding_mode,
        "finding_semantics": finding_semantics,
        "observable_scientific_semantics": {
            "required_capabilities": _json_value(
                _mapping(required_capabilities, "family_metadata.required_capabilities")
            ),
            "scientific_semantic_capabilities": _json_value(
                _mapping(
                    semantic_capabilities,
                    "family_metadata.scientific_semantic_capabilities",
                )
            ),
        },
        "verification_semantics": _verification_semantics(metadata),
    }
    return _json_value(projection)


def semantic_contract_sha256(
    metadata_or_projection: CaseConstructionMetadata | Mapping[str, Any],
    *,
    family_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Hash a derived projection, or derive and hash an existing case object."""

    if (
        isinstance(metadata_or_projection, Mapping)
        and str(
            metadata_or_projection.get(
                "scientific_semantic_projection_version", ""
            )
        ).startswith("scientific-semantic-projection-")
    ):
        projection = _json_value(metadata_or_projection)
    else:
        projection = build_scientific_semantic_projection(
            metadata_or_projection,
            family_metadata=family_metadata,
        )
    encoded = json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SCIENTIFIC_SEMANTIC_PROJECTION_VERSION",
    "ScientificSemanticProjectionError",
    "build_scientific_semantic_projection",
    "semantic_contract_sha256",
]

"""Human-facing question realization over a canonical semantic projection.

The functions in this module do not author scientific semantics.  They consume
an already-derived projection, render a small presentation set, and fail closed
when a rendered candidate changes target, scope, O responsibility, or F
responsibility.  Official SCQ remains responsible for qualifying the selected
final question.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable, Mapping, Sequence

from .scientific_run_snapshots import canonical_json_sha256


QUESTION_PRESENTATION_VERSION = "human-question-realization-v1"
QUESTION_PRESENTATION_AUTHORING_PACKET_VERSION = (
    "flow-expert-question-presentation-authoring-v1"
)
POST_REWRITE_SEMANTIC_FIDELITY_PACKET_VERSION = (
    "post-rewrite-semantic-fidelity-v1"
)
FINAL_FLOW_EXPERT_WORDING_REVIEW_PACKET_VERSION = (
    "final-flow-expert-wording-review-v1"
)
SEMANTIC_VISIBILITY_CLASSIFICATION_VERSION = "semantic-visibility-classification-v1"
SEMANTIC_VISIBILITY_CLASSES = frozenset(
    {"QUESTION_VISIBLE", "DATASET_CONTEXT_VISIBLE", "BACKEND_ONLY"}
)
QUESTION_REALIZATION_STYLES = (
    "DIRECT_SCIENTIFIC_QUESTION",
    "CONCISE_EXPERT_REQUEST",
    "INVESTIGATION_ORIENTED",
)

SemanticHashCallback = Callable[[Mapping[str, Any]], str]
QuestionPresentationAuthor = Callable[[Mapping[str, Any]], Mapping[str, Any]]
PostRewriteSemanticReviewer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
FinalFlowExpertWordingReviewer = Callable[[Mapping[str, Any]], Mapping[str, Any]]

FINAL_FLOW_EXPERT_WORDING_REVIEW_RECOMMENDATIONS = frozenset(
    {
        "KEEP",
        "REVISE_WORDING",
        "REVISE_TARGET",
        "REVISE_OPERATIONALIZATION_SPACE",
        "NOT_SCIENTIFICALLY_SUPPORTABLE",
    }
)


_PRESENTATION_FIREWALL_KEYS = frozenset(
    {
        "ground_truth",
        "reference_findings",
        "reference_answer",
        "accepted_o",
        "accepted_operationalization",
        "accepted_branch",
        "srac_adjudication",
        "curator_decision",
        "model_answer",
        "model_score",
        "benchmark_score",
        "evaluator_output",
    }
)


def _sequence(value: Any, *, field: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError(f"{field} must be a list")
    return list(value)


def _nonempty(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _semantic_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _display_dimension(value: str) -> str:
    return str(value).replace("_", " ").strip()


def _meaning(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in (
            "normalized_meaning",
            "description",
            "statement",
            "meaning",
            "canonical_id",
        ):
            if str(value.get(key, "")).strip():
                return str(value[key]).strip()
        return ""
    return str(value or "").strip()


def _target(projection: Mapping[str, Any]) -> str:
    return _nonempty(_meaning(projection.get("scientific_target")), field="scientific_target")


def _scope(projection: Mapping[str, Any]) -> str:
    value = projection.get("scientific_scope")
    if isinstance(value, Mapping):
        for key in ("normalized_meaning", "statement", "scope"):
            if str(value.get(key, "")).strip():
                return str(value[key]).strip()
        statements: list[str] = []
        for key in (
            "scope_constraints",
            "condition_constraints",
            "selection_constraints",
        ):
            raw = value.get(key, ()) or ()
            if isinstance(raw, Sequence) and not isinstance(
                raw, (str, bytes, bytearray)
            ):
                statements.extend(
                    str(item).strip() for item in raw if str(item).strip()
                )
        return "; ".join(statements)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _dimension_rows(projection: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = projection.get("resolved_operationalization")
    if value is None:
        value = projection.get("resolved_operationalization_clauses", ())
    rows: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        iterable = [
            ({**dict(item), "dimension_id": str(dimension)})
            if isinstance(item, Mapping)
            else {"dimension_id": str(dimension), "normalized_meaning": str(item)}
            for dimension, item in value.items()
        ]
    else:
        iterable = _sequence(value, field="resolved_operationalization")
    seen: set[str] = set()
    for item in iterable:
        if not isinstance(item, Mapping):
            raise ValueError("resolved operationalization clauses must be objects")
        dimension = _nonempty(
            item.get("dimension_id", item.get("dimension")),
            field="resolved dimension_id",
        )
        if dimension in seen:
            raise ValueError(f"duplicate resolved dimension: {dimension}")
        seen.add(dimension)
        statement = _nonempty(_meaning(item), field=f"resolved clause {dimension}")
        rows.append(
            {
                "dimension_id": dimension,
                "statement": statement,
                "canonical_id": str(item.get("canonical_id", "")),
                "required_semantic_anchors": [
                    str(anchor).strip()
                    for anchor in item.get("required_semantic_anchors", ()) or ()
                    if str(anchor).strip()
                ],
                "allowed_visible_realizations": [
                    str(realization).strip()
                    for realization in item.get("allowed_visible_realizations", ()) or ()
                    if str(realization).strip()
                ],
            }
        )
    return rows


def _unresolved_dimensions(projection: Mapping[str, Any]) -> list[str]:
    value = projection.get("unresolved_operationalization_dimensions", ())
    rows = _sequence(value, field="unresolved_operationalization_dimensions")
    result: list[str] = []
    for item in rows:
        dimension = (
            str(item.get("dimension_id", item.get("dimension", ""))).strip()
            if isinstance(item, Mapping)
            else str(item).strip()
        )
        if not dimension:
            raise ValueError("unresolved dimensions must be non-empty")
        result.append(dimension)
    if len(result) != len(set(result)):
        raise ValueError("unresolved dimensions must be unique")
    return result


def _finding_contract(projection: Mapping[str, Any]) -> dict[str, Any]:
    raw = projection.get("finding_responsibility")
    value = dict(raw) if isinstance(raw, Mapping) else {}
    semantic_value = projection.get("finding_semantics")
    semantic_value = dict(semantic_value) if isinstance(semantic_value, Mapping) else {}
    mode = str(
        value.get("finding_mode")
        or value.get("mode")
        or raw
        or projection.get("finding_mode")
        or ""
    ).upper()
    if mode not in {"F1", "F2"}:
        openness = str(projection.get("finding_openness", "")).casefold()
        mode = "F2" if openness == "open" else "F1" if openness == "bounded" else mode
    if mode not in {"F1", "F2"}:
        raise ValueError("finding_responsibility must resolve to F1 or F2")
    raw_requirements = projection.get("required_finding_roles")
    if raw_requirements is None:
        raw_requirements = projection.get("finding_requirements")
    if raw_requirements is None:
        raw_requirements = value.get("required_finding_roles", ())
    if not raw_requirements and mode == "F1":
        raw_requirements = semantic_value.get("fixed_finding_requirements", ())
    requirements: list[str] = []
    for item in _sequence(raw_requirements, field="required_finding_roles"):
        statement = _meaning(item)
        if statement:
            requirements.append(statement)
    adequate = semantic_value.get("adequate_core_semantics")
    adequate = dict(adequate) if isinstance(adequate, Mapping) else {}
    role_source = adequate or value
    mandatory = [
        str(item).strip()
        for item in role_source.get("mandatory_roles", ()) or ()
        if str(item).strip()
    ]
    alternatives: list[dict[str, Any]] = []
    for item in role_source.get("alternative_role_groups", ()) or ():
        if not isinstance(item, Mapping):
            raise ValueError("alternative_role_groups must contain objects")
        alternatives.append(
            {
                "group_id": str(item.get("group_id", "")).strip(),
                "min_required": item.get("min_required"),
                "roles": [
                    str(role).strip()
                    for role in item.get("roles", ()) or ()
                    if str(role).strip()
                ],
            }
        )
    adequate_core_sets = [
        [str(role).strip() for role in group if str(role).strip()]
        for group in role_source.get("adequate_core_sets", ()) or ()
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, bytearray))
    ]
    return {
        "finding_mode": mode,
        "fixed_requirements": requirements if mode == "F1" else [],
        "mandatory_roles": mandatory if mode == "F2" else [],
        "alternative_role_groups": alternatives if mode == "F2" else [],
        "adequate_core_sets": adequate_core_sets if mode == "F2" else [],
    }


def _projection_hash(
    projection: Mapping[str, Any],
    *,
    semantic_contract_sha256: str | None,
    semantic_hash_callback: SemanticHashCallback | None,
) -> str:
    if semantic_hash_callback is not None:
        value = str(semantic_hash_callback(projection)).strip()
    elif semantic_contract_sha256:
        value = str(semantic_contract_sha256).strip()
    elif str(projection.get("semantic_contract_sha256", "")).strip():
        value = str(projection["semantic_contract_sha256"]).strip()
    else:
        # The caller has already supplied the read-only scientific projection;
        # hashing it does not introduce another authored source of truth.
        value = canonical_json_sha256(projection)
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("semantic_contract_sha256 must be a lowercase SHA-256 digest")
    return value


def _assert_presentation_firewall(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            if normalized in _PRESENTATION_FIREWALL_KEYS:
                raise ValueError(f"presentation firewall rejects {path}.{key}")
            _assert_presentation_firewall(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _assert_presentation_firewall(item, path=f"{path}[{index}]")


def _semantic_contract_view(projection: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _dimension_rows(projection)
    unresolved = _unresolved_dimensions(projection)
    overlap = {row["dimension_id"] for row in resolved} & set(unresolved)
    if overlap:
        raise ValueError(
            f"resolved and unresolved O dimensions overlap: {sorted(overlap)}"
        )
    return {
        "scientific_target": _target(projection),
        "scientific_scope": _scope(projection),
        "fixed_operationalization": resolved,
        "unresolved_operationalization_dimensions": unresolved,
        "finding_responsibility": _finding_contract(projection),
    }


def build_semantic_visibility_classification(
    semantic_projection: Mapping[str, Any],
    *,
    dataset_context: Mapping[str, Any],
    placement_overrides: Mapping[str, str] | None = None,
    presentation_support_placements: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Classify where presentation information is visible without reauthoring it.

    Canonical scientific identity remains entirely in ``semantic_projection``.
    The returned sidecar controls placement only and is therefore excluded from
    ``semantic_contract_sha256``.
    """

    if not isinstance(dataset_context, Mapping):
        raise ValueError("dataset_context must be an object")
    contract = _semantic_contract_view(semantic_projection)
    rows: list[dict[str, str]] = [
        {
            "information_id": "scientific_target",
            "semantic_role": "SCIENTIFIC_TARGET",
            "visibility": "QUESTION_VISIBLE",
        },
        {
            "information_id": "scientific_scope",
            "semantic_role": "SCIENTIFIC_SCOPE",
            "visibility": "QUESTION_VISIBLE",
        },
    ]
    rows.extend(
        {
            "information_id": f"fixed_o:{item['dimension_id']}",
            "semantic_role": "FIXED_OPERATIONALIZATION",
            "visibility": "QUESTION_VISIBLE",
        }
        for item in contract["fixed_operationalization"]
    )
    rows.extend(
        {
            "information_id": f"unresolved_o:{dimension}",
            "semantic_role": "UNRESOLVED_OPERATIONALIZATION",
            "visibility": "QUESTION_VISIBLE",
        }
        for dimension in contract["unresolved_operationalization_dimensions"]
    )
    rows.append(
        {
            "information_id": "finding_responsibility",
            "semantic_role": "FINDING_RESPONSIBILITY",
            "visibility": "QUESTION_VISIBLE",
        }
    )
    by_id = {row["information_id"]: row for row in rows}
    for information_id, visibility in dict(placement_overrides or {}).items():
        if information_id not in by_id:
            raise ValueError(f"unknown semantic visibility information_id: {information_id}")
        by_id[information_id]["visibility"] = str(visibility)
    support_rows: list[dict[str, str]] = []
    for information_id, visibility in dict(
        presentation_support_placements or {}
    ).items():
        normalized_id = str(information_id).strip()
        if not normalized_id:
            raise ValueError("presentation support information_id must be non-empty")
        support_rows.append(
            {
                "information_id": f"presentation_support:{normalized_id}",
                "semantic_role": "PRESENTATION_SUPPORT",
                "visibility": str(visibility),
            }
        )
    classification = {
        "classification_version": SEMANTIC_VISIBILITY_CLASSIFICATION_VERSION,
        "items": [*rows, *support_rows],
        "semantic_contract_affected": False,
    }
    validation = validate_semantic_visibility_classification(
        classification,
        semantic_projection=semantic_projection,
        dataset_context=dataset_context,
    )
    if validation["status"] != "PASS":
        raise ValueError(
            "invalid semantic visibility classification: "
            + ", ".join(validation["errors"])
        )
    return classification


def validate_semantic_visibility_classification(
    classification: Mapping[str, Any],
    *,
    semantic_projection: Mapping[str, Any],
    dataset_context: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if classification.get("classification_version") != (
        SEMANTIC_VISIBILITY_CLASSIFICATION_VERSION
    ):
        errors.append("INVALID_SEMANTIC_VISIBILITY_VERSION")
    if classification.get("semantic_contract_affected") is not False:
        errors.append("VISIBILITY_MUST_NOT_AFFECT_SEMANTIC_CONTRACT")
    raw_items = classification.get("items")
    if not isinstance(raw_items, list):
        return {"status": "INVALID", "errors": [*errors, "VISIBILITY_ITEMS_NOT_LIST"]}
    contract = _semantic_contract_view(semantic_projection)
    expected_roles = {
        "scientific_target": "SCIENTIFIC_TARGET",
        "scientific_scope": "SCIENTIFIC_SCOPE",
        **{
            f"fixed_o:{item['dimension_id']}": "FIXED_OPERATIONALIZATION"
            for item in contract["fixed_operationalization"]
        },
        **{
            f"unresolved_o:{dimension}": "UNRESOLVED_OPERATIONALIZATION"
            for dimension in contract["unresolved_operationalization_dimensions"]
        },
        "finding_responsibility": "FINDING_RESPONSIBILITY",
    }
    seen: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping):
            errors.append(f"semantic_visibility.items[{index}]:OBJECT_REQUIRED")
            continue
        if set(raw) != {"information_id", "semantic_role", "visibility"}:
            errors.append(f"semantic_visibility.items[{index}]:INVALID_FIELDS")
        information_id = str(raw.get("information_id", "")).strip()
        role = str(raw.get("semantic_role", "")).strip()
        visibility = str(raw.get("visibility", "")).strip()
        if not information_id or information_id in seen:
            errors.append(f"semantic_visibility.items[{index}]:INVALID_INFORMATION_ID")
        if visibility not in SEMANTIC_VISIBILITY_CLASSES:
            errors.append(f"semantic_visibility.items[{index}]:INVALID_VISIBILITY")
        if information_id.startswith("presentation_support:"):
            if role != "PRESENTATION_SUPPORT":
                errors.append(f"semantic_visibility.items[{index}]:ROLE_MISMATCH")
        elif expected_roles.get(information_id) != role:
            errors.append(f"semantic_visibility.items[{index}]:ROLE_MISMATCH")
        if information_id:
            seen[information_id] = {
                "semantic_role": role,
                "visibility": visibility,
            }
    missing = sorted(set(expected_roles) - set(seen))
    unknown = sorted(
        information_id
        for information_id in set(seen) - set(expected_roles)
        if not information_id.startswith("presentation_support:")
    )
    if missing:
        errors.append("SEMANTIC_VISIBILITY_COVERAGE_MISSING:" + ",".join(missing))
    if unknown:
        errors.append("UNKNOWN_SEMANTIC_VISIBILITY_ITEMS:" + ",".join(unknown))
    for information_id, expected_role in expected_roles.items():
        row = seen.get(information_id)
        if row is None:
            continue
        visibility = row["visibility"]
        if expected_role in {
            "SCIENTIFIC_TARGET",
            "UNRESOLVED_OPERATIONALIZATION",
            "FINDING_RESPONSIBILITY",
        } and visibility != "QUESTION_VISIBLE":
            errors.append(f"{information_id}:QUESTION_VISIBILITY_REQUIRED")
        elif expected_role in {
            "SCIENTIFIC_SCOPE",
            "FIXED_OPERATIONALIZATION",
        } and visibility == "BACKEND_ONLY":
            errors.append(f"{information_id}:MODEL_VISIBILITY_REQUIRED")
        if visibility == "DATASET_CONTEXT_VISIBLE" and not dataset_context:
            errors.append(f"{information_id}:DATASET_CONTEXT_REQUIRED")
    return {"status": "PASS" if not errors else "INVALID", "errors": errors}


def build_question_presentation_authoring_packet(
    semantic_projection: Mapping[str, Any],
    *,
    dataset_context: Mapping[str, Any],
    candidate_count: int = 3,
    case_id: str | None = None,
    condition: str | None = None,
    semantic_visibility: Mapping[str, Any] | None = None,
    semantic_contract_sha256: str | None = None,
    semantic_hash_callback: SemanticHashCallback | None = None,
    repair_feedback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a GT-free wording task for the existing canonical Flow Expert."""

    if candidate_count not in {2, 3}:
        raise ValueError("candidate_count must be 2 or 3")
    if not isinstance(dataset_context, Mapping):
        raise ValueError("dataset_context must be an object")
    projection = copy.deepcopy(dict(semantic_projection))
    context = copy.deepcopy(dict(dataset_context))
    _assert_presentation_firewall(context, path="dataset_context")
    contract = _semantic_contract_view(projection)
    semantic_hash = _projection_hash(
        projection,
        semantic_contract_sha256=semantic_contract_sha256,
        semantic_hash_callback=semantic_hash_callback,
    )
    visibility = (
        build_semantic_visibility_classification(
            projection,
            dataset_context=context,
        )
        if semantic_visibility is None
        else copy.deepcopy(dict(semantic_visibility))
    )
    visibility_validation = validate_semantic_visibility_classification(
        visibility,
        semantic_projection=projection,
        dataset_context=context,
    )
    if visibility_validation["status"] != "PASS":
        raise ValueError(
            "invalid semantic visibility classification: "
            + ", ".join(visibility_validation["errors"])
        )
    packet = {
        "question_presentation_authoring_packet_version": (
            QUESTION_PRESENTATION_AUTHORING_PACKET_VERSION
        ),
        "task": "HUMAN_PRESENTATION_AUTHORING_ONLY",
        "case_identity": {
            "case_id": str(case_id or projection.get("case_id", "")),
            "condition": str(condition or projection.get("condition", "")),
        },
        "semantic_contract_sha256": semantic_hash,
        **contract,
        "dataset_context": context,
        "semantic_visibility": visibility,
        "candidate_count": candidate_count,
        "presentation_requirements": {
            "collaborator_natural_language": True,
            "condition_independent_style_choices": True,
            "backend_mechanical_language_only_when_scientifically_required": True,
            "selected_field_reported_as_which_field_not_identifier": True,
            "headings_optional": True,
            "scientific_semantics_must_not_change": True,
            "do_not_introduce_unlisted_operationalization_dimensions": True,
            "field_selection_language_only_when_contract_explicitly_requires_it": True,
        },
    }
    if repair_feedback is not None:
        if not isinstance(repair_feedback, Mapping):
            raise ValueError("repair_feedback must be an object")
        packet["repair_feedback"] = copy.deepcopy(dict(repair_feedback))
    validation = validate_question_presentation_authoring_packet(packet)
    if validation["status"] != "PASS":
        raise ValueError(
            "invalid question presentation authoring packet: "
            + ", ".join(validation["errors"])
        )
    return packet


def validate_question_presentation_authoring_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        _assert_presentation_firewall(packet)
    except ValueError as exc:
        errors.append(str(exc))
    if packet.get("question_presentation_authoring_packet_version") != (
        QUESTION_PRESENTATION_AUTHORING_PACKET_VERSION
    ):
        errors.append("INVALID_AUTHORING_PACKET_VERSION")
    if packet.get("task") != "HUMAN_PRESENTATION_AUTHORING_ONLY":
        errors.append("INVALID_AUTHORING_TASK")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(packet.get("semantic_contract_sha256", ""))
    ):
        errors.append("INVALID_SEMANTIC_CONTRACT_SHA256")
    if packet.get("candidate_count") not in {2, 3}:
        errors.append("INVALID_CANDIDATE_COUNT")
    for field in ("scientific_target", "scientific_scope"):
        if not str(packet.get(field, "")).strip():
            errors.append(f"MISSING_{field.upper()}")
    fixed = packet.get("fixed_operationalization")
    unresolved = packet.get("unresolved_operationalization_dimensions")
    finding = packet.get("finding_responsibility")
    if not isinstance(fixed, list):
        errors.append("FIXED_OPERATIONALIZATION_NOT_LIST")
        fixed = []
    if not isinstance(unresolved, list):
        errors.append("UNRESOLVED_OPERATIONALIZATION_NOT_LIST")
        unresolved = []
    fixed_ids = [
        str(item.get("dimension_id", "")).strip()
        for item in fixed
        if isinstance(item, Mapping)
    ]
    unresolved_ids = [str(item).strip() for item in unresolved]
    if any(not item for item in fixed_ids) or len(fixed_ids) != len(fixed):
        errors.append("INVALID_FIXED_OPERATIONALIZATION")
    if any(not item for item in unresolved_ids):
        errors.append("INVALID_UNRESOLVED_OPERATIONALIZATION")
    if len(fixed_ids) != len(set(fixed_ids)):
        errors.append("DUPLICATE_FIXED_DIMENSION")
    if len(unresolved_ids) != len(set(unresolved_ids)):
        errors.append("DUPLICATE_UNRESOLVED_DIMENSION")
    if set(fixed_ids) & set(unresolved_ids):
        errors.append("RESOLVED_UNRESOLVED_OVERLAP")
    if not isinstance(finding, Mapping) or finding.get("finding_mode") not in {
        "F1",
        "F2",
    }:
        errors.append("INVALID_FINDING_RESPONSIBILITY")
    if not isinstance(packet.get("dataset_context"), Mapping):
        errors.append("DATASET_CONTEXT_NOT_OBJECT")
    repair_feedback = packet.get("repair_feedback")
    if repair_feedback is not None:
        if not isinstance(repair_feedback, Mapping):
            errors.append("REPAIR_FEEDBACK_NOT_OBJECT")
        else:
            if repair_feedback.get("rewrite_only_identified_defects") is not True:
                errors.append("REPAIR_FEEDBACK_MUST_BE_DEFECT_SCOPED")
            repair_candidates = repair_feedback.get("candidates")
            if not isinstance(repair_candidates, list) or not repair_candidates:
                errors.append("REPAIR_FEEDBACK_CANDIDATES_INVALID")
            else:
                for index, repair_candidate in enumerate(repair_candidates):
                    if not isinstance(repair_candidate, Mapping):
                        errors.append(
                            f"REPAIR_FEEDBACK_CANDIDATE_{index}_NOT_OBJECT"
                        )
                        continue
                    if not str(repair_candidate.get("candidate_id", "")).strip():
                        errors.append(
                            f"REPAIR_FEEDBACK_CANDIDATE_{index}_ID_MISSING"
                        )
                    if not str(
                        repair_candidate.get("previous_model_visible_text", "")
                    ).strip():
                        errors.append(
                            f"REPAIR_FEEDBACK_CANDIDATE_{index}_TEXT_MISSING"
                        )
                    defects = repair_candidate.get("identified_defects")
                    if not isinstance(defects, list) or not defects or any(
                        not isinstance(item, str) or not item.strip()
                        for item in defects
                    ):
                        errors.append(
                            f"REPAIR_FEEDBACK_CANDIDATE_{index}_DEFECTS_INVALID"
                        )
    raw_visibility = packet.get("semantic_visibility")
    if not isinstance(raw_visibility, Mapping):
        errors.append("SEMANTIC_VISIBILITY_NOT_OBJECT")
    else:
        packet_projection = {
            "scientific_target": packet.get("scientific_target"),
            "scientific_scope": packet.get("scientific_scope"),
            "resolved_operationalization": packet.get("fixed_operationalization"),
            "unresolved_operationalization_dimensions": packet.get(
                "unresolved_operationalization_dimensions"
            ),
            "finding_responsibility": packet.get("finding_responsibility"),
        }
        try:
            visibility_validation = validate_semantic_visibility_classification(
                raw_visibility,
                semantic_projection=packet_projection,
                dataset_context=(
                    packet.get("dataset_context")
                    if isinstance(packet.get("dataset_context"), Mapping)
                    else {}
                ),
            )
            errors.extend(visibility_validation["errors"])
        except ValueError as exc:
            errors.append(f"INVALID_SEMANTIC_VISIBILITY_CONTRACT:{exc}")
    return {"status": "PASS" if not errors else "INVALID", "errors": errors}


def validate_question_presentation_authoring_output(
    output: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate wording output without accepting model-authored semantics."""

    errors: list[str] = []
    allowed_top = {"candidates"}
    unexpected_top = sorted(set(output) - allowed_top)
    if unexpected_top:
        errors.append("UNKNOWN_AUTHORING_OUTPUT_FIELDS:" + ",".join(unexpected_top))
    raw_candidates = output.get("candidates")
    if not isinstance(raw_candidates, list):
        return {
            "status": "INCOMPLETE",
            "errors": [*errors, "CANDIDATES_NOT_LIST"],
            "candidates": [],
        }
    if len(raw_candidates) != packet.get("candidate_count"):
        errors.append("CANDIDATE_COUNT_MISMATCH")
    candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_styles: set[str] = set()
    seen_texts: set[str] = set()
    allowed_candidate = {"candidate_id", "style", "model_visible_text"}
    for index, raw in enumerate(raw_candidates):
        field = f"candidates[{index}]"
        if not isinstance(raw, Mapping):
            errors.append(f"{field}:NOT_OBJECT")
            continue
        unexpected = sorted(set(raw) - allowed_candidate)
        if unexpected:
            errors.append(f"{field}:UNKNOWN_FIELDS:{','.join(unexpected)}")
        candidate_id = str(raw.get("candidate_id", "")).strip()
        style = str(raw.get("style", "")).strip()
        text = str(raw.get("model_visible_text", "")).strip()
        if not candidate_id:
            errors.append(f"{field}:MISSING_CANDIDATE_ID")
        elif candidate_id in seen_ids:
            errors.append(f"{field}:DUPLICATE_CANDIDATE_ID")
        seen_ids.add(candidate_id)
        if style not in QUESTION_REALIZATION_STYLES:
            errors.append(f"{field}:INVALID_STYLE")
        elif style in seen_styles:
            errors.append(f"{field}:DUPLICATE_STYLE")
        seen_styles.add(style)
        if not text:
            errors.append(f"{field}:EMPTY_MODEL_VISIBLE_TEXT")
        elif text in seen_texts:
            errors.append(f"{field}:DUPLICATE_MODEL_VISIBLE_TEXT")
        seen_texts.add(text)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "style": style,
                "model_visible_text": text,
            }
        )
    return {
        "status": "PASS" if not errors else "INCOMPLETE",
        "errors": errors,
        "candidates": candidates,
    }


def _parsed_model_output(raw: Mapping[str, Any]) -> Mapping[str, Any] | None:
    parsed = raw.get("parsed_result")
    if isinstance(parsed, Mapping):
        return parsed
    if isinstance(raw.get("candidates"), list):
        return raw
    if "scientific_target_preserved" in raw:
        return raw
    return None


def author_human_questions(
    semantic_projection: Mapping[str, Any],
    *,
    dataset_context: Mapping[str, Any],
    author: QuestionPresentationAuthor,
    candidate_count: int = 3,
    case_id: str | None = None,
    condition: str | None = None,
    semantic_visibility: Mapping[str, Any] | None = None,
    semantic_contract_sha256: str | None = None,
    semantic_hash_callback: SemanticHashCallback | None = None,
    repair_feedback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use an injected or live Flow Expert call to author wording only."""

    if not callable(author):
        raise ValueError("author must be callable")
    packet = build_question_presentation_authoring_packet(
        semantic_projection,
        dataset_context=dataset_context,
        candidate_count=candidate_count,
        case_id=case_id,
        condition=condition,
        semantic_visibility=semantic_visibility,
        semantic_contract_sha256=semantic_contract_sha256,
        semantic_hash_callback=semantic_hash_callback,
        repair_feedback=repair_feedback,
    )
    raw = author(copy.deepcopy(packet))
    if not isinstance(raw, Mapping):
        raise ValueError("author must return an object")
    parsed = _parsed_model_output(raw)
    if parsed is None:
        return {
            "status": "INFRA_INVALID",
            "semantic_contract_sha256": packet["semantic_contract_sha256"],
            "semantic_visibility": copy.deepcopy(packet["semantic_visibility"]),
            "candidate_count": 0,
            "candidates": [],
            "authoring_validation": {
                "status": "INCOMPLETE",
                "errors": ["NO_PARSED_AUTHORING_OUTPUT"],
                "candidates": [],
            },
            "authoring_call": copy.deepcopy(dict(raw)),
            "official_scq_required_for_selected_final_question": True,
            "scientific_qualification_performed": False,
            "core_membership_affected": False,
        }
    validation = validate_question_presentation_authoring_output(parsed, packet)
    candidates: list[dict[str, Any]] = []
    if validation["status"] == "PASS":
        for item in validation["candidates"]:
            candidate: dict[str, Any] = {
                **item,
                "presentation_version": QUESTION_PRESENTATION_VERSION,
                "authoring_mode": "FLOW_EXPERT_PRESENTATION_AUTHORING",
                "semantic_contract_sha256": packet["semantic_contract_sha256"],
            }
            candidate["presentation_sha256"] = canonical_json_sha256(candidate)
            candidates.append(candidate)
    return {
        "status": "PASS" if validation["status"] == "PASS" else "INCOMPLETE",
        "presentation_version": QUESTION_PRESENTATION_VERSION,
        "authoring_packet_version": QUESTION_PRESENTATION_AUTHORING_PACKET_VERSION,
        "case_id": packet["case_identity"]["case_id"],
        "condition": packet["case_identity"]["condition"],
        "semantic_contract_sha256": packet["semantic_contract_sha256"],
        "semantic_visibility": copy.deepcopy(packet["semantic_visibility"]),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "authoring_validation": validation,
        "authoring_call": copy.deepcopy(dict(raw)),
        "source_question_rewritten": True,
        "official_scq_required_for_selected_final_question": True,
        "scientific_qualification_performed": False,
        "core_membership_affected": False,
    }


def build_post_rewrite_semantic_fidelity_packet(
    semantic_projection: Mapping[str, Any],
    model_visible_text: str,
    *,
    dataset_context: Mapping[str, Any] | None = None,
    case_id: str | None = None,
    condition: str | None = None,
    semantic_visibility: Mapping[str, Any] | None = None,
    semantic_contract_sha256: str | None = None,
    semantic_hash_callback: SemanticHashCallback | None = None,
) -> dict[str, Any]:
    """Bind the actual final text to a read-only canonical semantic contract."""

    text = _nonempty(model_visible_text, field="model_visible_text")
    projection = copy.deepcopy(dict(semantic_projection))
    context = copy.deepcopy(dict(dataset_context or {}))
    _assert_presentation_firewall(context, path="dataset_context")
    semantic_hash = _projection_hash(
        projection,
        semantic_contract_sha256=semantic_contract_sha256,
        semantic_hash_callback=semantic_hash_callback,
    )
    visibility = (
        build_semantic_visibility_classification(
            projection,
            dataset_context=context,
        )
        if semantic_visibility is None
        else copy.deepcopy(dict(semantic_visibility))
    )
    visibility_validation = validate_semantic_visibility_classification(
        visibility,
        semantic_projection=projection,
        dataset_context=context,
    )
    if visibility_validation["status"] != "PASS":
        raise ValueError(
            "invalid semantic visibility classification: "
            + ", ".join(visibility_validation["errors"])
        )
    packet = {
        "post_rewrite_semantic_fidelity_packet_version": (
            POST_REWRITE_SEMANTIC_FIDELITY_PACKET_VERSION
        ),
        "task": "POST_REWRITE_SEMANTIC_FIDELITY_ONLY",
        "case_identity": {
            "case_id": str(case_id or projection.get("case_id", "")),
            "condition": str(condition or projection.get("condition", "")),
        },
        "semantic_contract_sha256": semantic_hash,
        **_semantic_contract_view(projection),
        "dataset_context": context,
        "semantic_visibility": visibility,
        "model_visible_text": text,
    }
    validation = validate_post_rewrite_semantic_fidelity_packet(packet)
    if validation["status"] != "PASS":
        raise ValueError(
            "invalid post-rewrite fidelity packet: "
            + ", ".join(validation["errors"])
        )
    return packet


def validate_post_rewrite_semantic_fidelity_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        _assert_presentation_firewall(packet)
    except ValueError as exc:
        errors.append(str(exc))
    if packet.get("post_rewrite_semantic_fidelity_packet_version") != (
        POST_REWRITE_SEMANTIC_FIDELITY_PACKET_VERSION
    ):
        errors.append("INVALID_FIDELITY_PACKET_VERSION")
    if packet.get("task") != "POST_REWRITE_SEMANTIC_FIDELITY_ONLY":
        errors.append("INVALID_FIDELITY_TASK")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(packet.get("semantic_contract_sha256", ""))
    ):
        errors.append("INVALID_SEMANTIC_CONTRACT_SHA256")
    if not str(packet.get("model_visible_text", "")).strip():
        errors.append("EMPTY_MODEL_VISIBLE_TEXT")
    authoring_shape = {
        "question_presentation_authoring_packet_version": (
            QUESTION_PRESENTATION_AUTHORING_PACKET_VERSION
        ),
        "task": "HUMAN_PRESENTATION_AUTHORING_ONLY",
        "semantic_contract_sha256": packet.get("semantic_contract_sha256"),
        "candidate_count": 2,
        "scientific_target": packet.get("scientific_target"),
        "scientific_scope": packet.get("scientific_scope"),
        "fixed_operationalization": packet.get("fixed_operationalization"),
        "unresolved_operationalization_dimensions": packet.get(
            "unresolved_operationalization_dimensions"
        ),
        "finding_responsibility": packet.get("finding_responsibility"),
        "dataset_context": packet.get("dataset_context"),
        "semantic_visibility": packet.get("semantic_visibility"),
    }
    errors.extend(validate_question_presentation_authoring_packet(authoring_shape)["errors"])
    return {"status": "PASS" if not errors else "INVALID", "errors": errors}


def build_final_flow_expert_wording_review_packet(
    semantic_projection: Mapping[str, Any],
    model_visible_text: str,
    *,
    dataset_context: Mapping[str, Any] | None = None,
    evidence_context: Mapping[str, Any] | None = None,
    case_id: str | None = None,
    condition: str | None = None,
    semantic_visibility: Mapping[str, Any] | None = None,
    semantic_contract_sha256: str | None = None,
    semantic_hash_callback: SemanticHashCallback | None = None,
) -> dict[str, Any]:
    """Bind the actual final question to an explicit Flow Expert review task.

    This packet is intentionally separate from the post-rewrite fidelity
    packet.  Fidelity asks whether semantic clauses survived a rewrite;
    this review asks whether the resulting question is meaningful,
    answerable, and appropriately scoped for a flow scientist.  Both packets
    are read-only projections of the same semantic contract and never expose
    Ground Truth or model-evaluation artifacts.
    """

    text = _nonempty(model_visible_text, field="model_visible_text")
    projection = copy.deepcopy(dict(semantic_projection))
    context = copy.deepcopy(dict(dataset_context or {}))
    evidence = copy.deepcopy(dict(evidence_context or {}))
    _assert_presentation_firewall(context, path="dataset_context")
    _assert_presentation_firewall(evidence, path="evidence_context")
    semantic_hash = _projection_hash(
        projection,
        semantic_contract_sha256=semantic_contract_sha256,
        semantic_hash_callback=semantic_hash_callback,
    )
    visibility = (
        build_semantic_visibility_classification(
            projection,
            dataset_context=context,
        )
        if semantic_visibility is None
        else copy.deepcopy(dict(semantic_visibility))
    )
    visibility_validation = validate_semantic_visibility_classification(
        visibility,
        semantic_projection=projection,
        dataset_context=context,
    )
    if visibility_validation["status"] != "PASS":
        raise ValueError(
            "invalid semantic visibility classification: "
            + ", ".join(visibility_validation["errors"])
        )
    packet = {
        "final_flow_expert_wording_review_packet_version": (
            FINAL_FLOW_EXPERT_WORDING_REVIEW_PACKET_VERSION
        ),
        "task": "FINAL_FLOW_EXPERT_WORDING_REVIEW",
        "case_identity": {
            "case_id": str(case_id or projection.get("case_id", "")),
            "condition": str(condition or projection.get("condition", "")),
        },
        "semantic_contract_sha256": semantic_hash,
        **_semantic_contract_view(projection),
        "dataset_context": context,
        "evidence_context": evidence,
        "semantic_visibility": visibility,
        "model_visible_text": text,
        "review_requirements": {
            "meaningfulness": True,
            "target_stability_and_fidelity": True,
            "scope_preservation": True,
            "fixed_o_exposure": True,
            "unresolved_o_openness": True,
            "unintended_o_responsibility_detection": True,
            "finding_responsibility": True,
            "answerability_in_principle": True,
            "evidence_limitations": True,
            "different_valid_o_may_produce_different_g_of_o": True,
            "human_clarity_and_actionability": True,
            "unsupported_scientific_assertion_detection": True,
            "mechanical_or_protocol_burden_detection": True,
        },
    }
    validation = validate_final_flow_expert_wording_review_packet(packet)
    if validation["status"] != "PASS":
        raise ValueError(
            "invalid final Flow Expert wording-review packet: "
            + ", ".join(validation["errors"])
        )
    return packet


def validate_final_flow_expert_wording_review_packet(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the final-wording review input and its presentation firewall."""

    errors: list[str] = []
    if not isinstance(packet, Mapping):
        return {"status": "INVALID", "errors": ["PACKET_NOT_OBJECT"]}
    try:
        _assert_presentation_firewall(packet)
    except ValueError as exc:
        errors.append(str(exc))
    if packet.get("final_flow_expert_wording_review_packet_version") != (
        FINAL_FLOW_EXPERT_WORDING_REVIEW_PACKET_VERSION
    ):
        errors.append("INVALID_FINAL_WORDING_REVIEW_PACKET_VERSION")
    if packet.get("task") != "FINAL_FLOW_EXPERT_WORDING_REVIEW":
        errors.append("INVALID_FINAL_WORDING_REVIEW_TASK")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(packet.get("semantic_contract_sha256", ""))
    ):
        errors.append("INVALID_SEMANTIC_CONTRACT_SHA256")
    if not str(packet.get("model_visible_text", "")).strip():
        errors.append("EMPTY_MODEL_VISIBLE_TEXT")
    identity = packet.get("case_identity")
    if not isinstance(identity, Mapping):
        errors.append("CASE_IDENTITY_NOT_OBJECT")
    else:
        if not str(identity.get("case_id", "")).strip():
            errors.append("CASE_IDENTITY_CASE_ID_MISSING")
        if not str(identity.get("condition", "")).strip():
            errors.append("CASE_IDENTITY_CONDITION_MISSING")
    if not isinstance(packet.get("dataset_context"), Mapping):
        errors.append("DATASET_CONTEXT_NOT_OBJECT")
    if not isinstance(packet.get("evidence_context"), Mapping):
        errors.append("EVIDENCE_CONTEXT_NOT_OBJECT")
    requirements = packet.get("review_requirements")
    if not isinstance(requirements, Mapping):
        errors.append("REVIEW_REQUIREMENTS_NOT_OBJECT")
    else:
        required_requirements = {
            "meaningfulness",
            "target_stability_and_fidelity",
            "scope_preservation",
            "fixed_o_exposure",
            "unresolved_o_openness",
            "unintended_o_responsibility_detection",
            "finding_responsibility",
            "answerability_in_principle",
            "evidence_limitations",
            "different_valid_o_may_produce_different_g_of_o",
            "human_clarity_and_actionability",
            "unsupported_scientific_assertion_detection",
            "mechanical_or_protocol_burden_detection",
        }
        if set(requirements) != required_requirements:
            errors.append("REVIEW_REQUIREMENTS_FIELDS_INVALID")
        elif any(value is not True for value in requirements.values()):
            errors.append("REVIEW_REQUIREMENTS_MUST_BE_TRUE")
    # Reuse the canonical contract/visibility validation.  This prevents a
    # caller from constructing a final-review packet with a subtly different
    # target or O/F shape.
    authoring_shape = {
        "question_presentation_authoring_packet_version": (
            QUESTION_PRESENTATION_AUTHORING_PACKET_VERSION
        ),
        "task": "HUMAN_PRESENTATION_AUTHORING_ONLY",
        "semantic_contract_sha256": packet.get("semantic_contract_sha256"),
        "candidate_count": 2,
        "scientific_target": packet.get("scientific_target"),
        "scientific_scope": packet.get("scientific_scope"),
        "fixed_operationalization": packet.get("fixed_operationalization"),
        "unresolved_operationalization_dimensions": packet.get(
            "unresolved_operationalization_dimensions"
        ),
        "finding_responsibility": packet.get("finding_responsibility"),
        "dataset_context": packet.get("dataset_context"),
        "semantic_visibility": packet.get("semantic_visibility"),
    }
    errors.extend(validate_question_presentation_authoring_packet(authoring_shape)["errors"])
    return {"status": "PASS" if not errors else "INVALID", "errors": sorted(set(errors))}


def validate_post_rewrite_semantic_fidelity_output(
    output: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[str, Any]:
    """Compile an independent text-level review into closed failure codes."""

    errors: list[str] = []
    failures: list[str] = []
    allowed_top = {
        "scientific_target_preserved",
        "scientific_scope_preserved",
        "fixed_o_checks",
        "unresolved_o_checks",
        "finding_responsibility_check",
        "no_additional_scientific_obligation",
        "additional_scientific_obligations",
        "rationale",
    }
    unexpected = sorted(set(output) - allowed_top)
    if unexpected:
        errors.append("UNKNOWN_FIDELITY_OUTPUT_FIELDS:" + ",".join(unexpected))

    def required_bool(field: str) -> bool | None:
        value = output.get(field)
        if not isinstance(value, bool):
            errors.append(f"{field}:BOOLEAN_REQUIRED")
            return None
        return value

    target_preserved = required_bool("scientific_target_preserved")
    scope_preserved = required_bool("scientific_scope_preserved")
    no_extra = required_bool("no_additional_scientific_obligation")
    if target_preserved is False:
        failures.append("TARGET_DRIFT")
    if scope_preserved is False:
        failures.append("SCOPE_DRIFT")

    expected_fixed = {
        str(item.get("dimension_id", "")).strip()
        for item in packet.get("fixed_operationalization", [])
        if isinstance(item, Mapping)
    }
    fixed_checks = output.get("fixed_o_checks")
    fixed_seen: dict[str, bool] = {}
    if not isinstance(fixed_checks, list):
        errors.append("fixed_o_checks:LIST_REQUIRED")
    else:
        for index, raw in enumerate(fixed_checks):
            if not isinstance(raw, Mapping):
                errors.append(f"fixed_o_checks[{index}]:OBJECT_REQUIRED")
                continue
            if set(raw) != {"dimension_id", "preserved"}:
                errors.append(f"fixed_o_checks[{index}]:INVALID_FIELDS")
            dimension = str(raw.get("dimension_id", "")).strip()
            preserved = raw.get("preserved")
            if not dimension or dimension in fixed_seen:
                errors.append(f"fixed_o_checks[{index}]:INVALID_DIMENSION")
            if not isinstance(preserved, bool):
                errors.append(f"fixed_o_checks[{index}]:BOOLEAN_REQUIRED")
            if dimension and isinstance(preserved, bool):
                fixed_seen[dimension] = preserved
    if set(fixed_seen) != expected_fixed:
        errors.append("FIXED_O_CHECK_COVERAGE_MISMATCH")
    if any(not value for value in fixed_seen.values()):
        failures.append("FIXED_O_DISAPPEARED_OR_CHANGED")

    expected_unresolved = {
        str(item).strip()
        for item in packet.get("unresolved_operationalization_dimensions", [])
    }
    unresolved_checks = output.get("unresolved_o_checks")
    unresolved_seen: dict[str, bool] = {}
    if not isinstance(unresolved_checks, list):
        errors.append("unresolved_o_checks:LIST_REQUIRED")
    else:
        for index, raw in enumerate(unresolved_checks):
            if not isinstance(raw, Mapping):
                errors.append(f"unresolved_o_checks[{index}]:OBJECT_REQUIRED")
                continue
            if set(raw) != {"dimension_id", "remains_open"}:
                errors.append(f"unresolved_o_checks[{index}]:INVALID_FIELDS")
            dimension = str(raw.get("dimension_id", "")).strip()
            remains_open = raw.get("remains_open")
            if not dimension or dimension in unresolved_seen:
                errors.append(f"unresolved_o_checks[{index}]:INVALID_DIMENSION")
            if not isinstance(remains_open, bool):
                errors.append(f"unresolved_o_checks[{index}]:BOOLEAN_REQUIRED")
            if dimension and isinstance(remains_open, bool):
                unresolved_seen[dimension] = remains_open
    if set(unresolved_seen) != expected_unresolved:
        errors.append("UNRESOLVED_O_CHECK_COVERAGE_MISMATCH")
    if any(not value for value in unresolved_seen.values()):
        failures.append("UNRESOLVED_DIMENSION_PREMATURELY_FIXED")

    finding = output.get("finding_responsibility_check")
    finding_mode = str(packet.get("finding_responsibility", {}).get("finding_mode", ""))
    finding_checks = {
        "finding_mode": None,
        "responsibility_preserved": None,
        "f1_remains_fixed": None,
        "f2_remains_open": None,
    }
    if not isinstance(finding, Mapping):
        errors.append("finding_responsibility_check:OBJECT_REQUIRED")
    else:
        if set(finding) != set(finding_checks):
            errors.append("finding_responsibility_check:INVALID_FIELDS")
        finding_checks.update({key: finding.get(key) for key in finding_checks})
        if finding_checks["finding_mode"] != finding_mode:
            errors.append("FINDING_MODE_CHECK_MISMATCH")
        for field in (
            "responsibility_preserved",
            "f1_remains_fixed",
            "f2_remains_open",
        ):
            if not isinstance(finding_checks[field], bool):
                errors.append(f"finding_responsibility_check.{field}:BOOLEAN_REQUIRED")
        if finding_checks["responsibility_preserved"] is False:
            failures.append("FINDING_RESPONSIBILITY_DRIFT")
        if finding_mode == "F1" and finding_checks["f1_remains_fixed"] is False:
            failures.append("F1_BECAME_OPEN")
        if finding_mode == "F2" and finding_checks["f2_remains_open"] is False:
            failures.append("F2_BECAME_HIDDEN_F1")

    obligations = output.get("additional_scientific_obligations")
    if not isinstance(obligations, list) or any(
        not isinstance(item, str) or not item.strip() for item in obligations or []
    ):
        errors.append("additional_scientific_obligations:STRING_LIST_REQUIRED")
        obligations = []
    if no_extra is False or obligations:
        failures.append("EXTRA_SCIENTIFIC_OBLIGATION")
    if not str(output.get("rationale", "")).strip():
        errors.append("rationale:NONEMPTY_STRING_REQUIRED")

    checks = {
        "target_preserved": target_preserved,
        "scope_preserved": scope_preserved,
        "fixed_o_preserved": bool(fixed_seen) and all(fixed_seen.values())
        if expected_fixed
        else set(fixed_seen) == expected_fixed,
        "unresolved_o_remains_open": bool(unresolved_seen)
        and all(unresolved_seen.values())
        if expected_unresolved
        else set(unresolved_seen) == expected_unresolved,
        "finding_responsibility_preserved": finding_checks[
            "responsibility_preserved"
        ],
        "no_additional_scientific_obligation": no_extra,
    }
    if errors:
        status = "INCOMPLETE"
    elif failures:
        status = "REVISE"
    else:
        status = "PASS"
    return {
        "status": status,
        "checks": checks,
        "failure_codes": sorted(set(failures)),
        "validation_errors": errors,
        "fixed_o_checks": fixed_seen,
        "unresolved_o_checks": unresolved_seen,
        "finding_responsibility_check": finding_checks,
        "additional_scientific_obligations": list(obligations),
        "rationale": str(output.get("rationale", "")),
    }


_FINAL_WORDING_REVIEW_FIELDS = frozenset(
    {
        "question_meaningful",
        "target_stable",
        "target_wording_faithful",
        "scope_preserved",
        "fixed_o_exposed",
        "unresolved_o_left_open",
        "no_unintended_o_responsibility",
        "finding_responsibility_correct",
        "answerable_in_principle",
        "evidence_limitations",
        "recommendation",
        "rationale",
        # Evidence identifiers are optional because a review can legitimately
        # report that the supplied evidence context is empty or inconclusive.
        "evidence_ids",
    }
)


def _final_review_value(
    output: Mapping[str, Any], canonical: str, aliases: Sequence[str] = ()
) -> Any:
    """Read a canonical final-review field while accepting stable aliases."""

    if canonical in output:
        return output.get(canonical)
    for alias in aliases:
        if alias in output:
            return output.get(alias)
    return None


def _final_review_parsed_output(raw: Mapping[str, Any]) -> Mapping[str, Any] | None:
    parsed = raw.get("parsed_result")
    if isinstance(parsed, Mapping):
        # A caller may pass through the result of ``audit_final...``.  In that
        # case consume only its structured model result, never the transport
        # metadata or derived status fields.
        nested = parsed.get("parsed_result")
        if isinstance(nested, Mapping):
            return nested
        nested = parsed.get("review")
        if isinstance(nested, Mapping) and any(
            key in nested for key in _FINAL_WORDING_REVIEW_FIELDS
        ):
            return nested
        return parsed
    if any(key in raw for key in _FINAL_WORDING_REVIEW_FIELDS):
        return raw
    return None


def validate_final_flow_expert_wording_review_output(
    output: Mapping[str, Any] | Any,
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an explicit final-wording Flow Expert judgment.

    The validator is intentionally independent from semantic-fidelity
    validation.  A successful result means the reviewer examined the actual
    final text and found it meaningful/answerable without introducing an
    undeclared responsibility; it does not qualify a benchmark case.
    """

    if not isinstance(output, Mapping):
        return {
            "status": "INCOMPLETE",
            "checks": {},
            "failure_codes": [],
            "validation_errors": ["OUTPUT_NOT_OBJECT"],
            "evidence_limitations": [],
            "recommendation": None,
            "rationale": "",
        }
    errors: list[str] = []
    failures: list[str] = []
    # Accept a small set of semantically identical field spellings from the
    # model, but normalize the stored result to the canonical names above.
    aliases = {
        "question_meaningful": ("meaningful", "scientific_question_meaningful"),
        "target_stable": ("scientific_target_stable",),
        "target_wording_faithful": ("target_faithful",),
        "scope_preserved": ("scientific_scope_preserved",),
        "fixed_o_exposed": ("fixed_operationalization_exposed", "fixed_o_correctly_exposed"),
        "unresolved_o_left_open": (
            "unresolved_operationalization_left_open",
            "unresolved_o_correctly_open",
        ),
        "no_unintended_o_responsibility": (
            "no_extra_o_responsibility",
            "unintended_o_responsibility_absent",
        ),
        "finding_responsibility_correct": (
            "finding_responsibility_preserved",
            "finding_responsibility_faithful",
        ),
        "answerable_in_principle": ("scientifically_answerable", "answerable"),
        "evidence_limitations": ("evidence_limitations_summary",),
        "evidence_ids": ("supporting_evidence_ids",),
    }
    recognized = set(_FINAL_WORDING_REVIEW_FIELDS)
    for values in aliases.values():
        recognized.update(values)
    unexpected = sorted(set(output) - recognized)
    if unexpected:
        errors.append("UNKNOWN_FINAL_WORDING_REVIEW_FIELDS:" + ",".join(unexpected))

    checks: dict[str, bool | None] = {}
    for field in (
        "question_meaningful",
        "target_stable",
        "target_wording_faithful",
        "scope_preserved",
        "fixed_o_exposed",
        "unresolved_o_left_open",
        "no_unintended_o_responsibility",
        "finding_responsibility_correct",
        "answerable_in_principle",
    ):
        value = _final_review_value(output, field, aliases[field])
        if not isinstance(value, bool):
            errors.append(f"{field}:BOOLEAN_REQUIRED")
            checks[field] = None
        else:
            checks[field] = value
            if value is False:
                failures.append(
                    {
                        "question_meaningful": "QUESTION_NOT_MEANINGFUL",
                        "target_stable": "TARGET_NOT_STABLE",
                        "target_wording_faithful": "TARGET_WORDING_DRIFT",
                        "scope_preserved": "SCOPE_DRIFT",
                        "fixed_o_exposed": "FIXED_O_NOT_EXPOSED",
                        "unresolved_o_left_open": "UNRESOLVED_O_CLOSED",
                        "no_unintended_o_responsibility": "UNDECLARED_O_RESPONSIBILITY",
                        "finding_responsibility_correct": "FINDING_RESPONSIBILITY_DRIFT",
                        "answerable_in_principle": "QUESTION_NOT_ANSWERABLE",
                    }[field]
                )

    limitations = _final_review_value(
        output, "evidence_limitations", aliases["evidence_limitations"]
    )
    if isinstance(limitations, str):
        limitations = [limitations.strip()] if limitations.strip() else []
    elif isinstance(limitations, list):
        if any(not isinstance(item, str) or not item.strip() for item in limitations):
            errors.append("evidence_limitations:STRING_LIST_REQUIRED")
            limitations = []
        else:
            limitations = [item.strip() for item in limitations]
    else:
        errors.append("evidence_limitations:STRING_OR_STRING_LIST_REQUIRED")
        limitations = []

    evidence_ids = _final_review_value(output, "evidence_ids", aliases["evidence_ids"])
    if evidence_ids is None:
        evidence_ids = []
    if isinstance(evidence_ids, str):
        evidence_ids = [evidence_ids] if evidence_ids.strip() else []
    if not isinstance(evidence_ids, list) or any(
        not isinstance(item, str) or not item.strip() for item in evidence_ids
    ):
        errors.append("evidence_ids:STRING_LIST_REQUIRED")
        evidence_ids = []
    else:
        evidence_ids = [item.strip() for item in evidence_ids]

    known_ids: set[str] = set()
    evidence_context = packet.get("evidence_context")
    if isinstance(evidence_context, Mapping):
        def collect(value: Any) -> None:
            if isinstance(value, Mapping):
                evidence_id = str(value.get("evidence_id", "")).strip()
                if evidence_id:
                    known_ids.add(evidence_id)
                for item in value.values():
                    collect(item)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for item in value:
                    collect(item)
        collect(evidence_context)
    if known_ids and set(evidence_ids) - known_ids:
        errors.append("UNKNOWN_EVIDENCE_ID")

    recommendation = _final_review_value(output, "recommendation")
    if recommendation not in FINAL_FLOW_EXPERT_WORDING_REVIEW_RECOMMENDATIONS:
        errors.append("recommendation:INVALID_ENUM")
    rationale = _final_review_value(output, "rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("rationale:NONEMPTY_STRING_REQUIRED")
        rationale = ""
    else:
        rationale = rationale.strip()
    if recommendation == "KEEP" and failures:
        errors.append("KEEP_CONTRADICTS_FAILED_CHECKS")
    if recommendation == "KEEP" and limitations is None:
        errors.append("KEEP_REQUIRES_EXPLICIT_EVIDENCE_LIMITATIONS")
    status = "INCOMPLETE" if errors else "REVISE" if failures else "PASS"
    return {
        "status": status,
        "checks": checks,
        "failure_codes": sorted(set(failures)),
        "validation_errors": sorted(set(errors)),
        "evidence_limitations": list(limitations),
        "evidence_ids": list(evidence_ids),
        "recommendation": recommendation,
        "rationale": rationale,
    }


def audit_final_flow_expert_wording(
    semantic_projection: Mapping[str, Any],
    model_visible_text: str,
    *,
    reviewer: FinalFlowExpertWordingReviewer,
    dataset_context: Mapping[str, Any] | None = None,
    evidence_context: Mapping[str, Any] | None = None,
    case_id: str | None = None,
    condition: str | None = None,
    semantic_visibility: Mapping[str, Any] | None = None,
    semantic_contract_sha256: str | None = None,
    semantic_hash_callback: SemanticHashCallback | None = None,
) -> dict[str, Any]:
    """Run and record the explicit final Flow Expert wording review."""

    if not callable(reviewer):
        raise ValueError("reviewer must be callable")
    packet = build_final_flow_expert_wording_review_packet(
        semantic_projection,
        model_visible_text,
        dataset_context=dataset_context,
        evidence_context=evidence_context,
        case_id=case_id,
        condition=condition,
        semantic_visibility=semantic_visibility,
        semantic_contract_sha256=semantic_contract_sha256,
        semantic_hash_callback=semantic_hash_callback,
    )
    raw = reviewer(copy.deepcopy(packet))
    if not isinstance(raw, Mapping):
        raw = {"raw_result": raw}
    parsed = _final_review_parsed_output(raw)
    if parsed is None:
        validation = {
            "status": "INCOMPLETE",
            "checks": {},
            "failure_codes": [],
            "validation_errors": ["NO_PARSED_FINAL_WORDING_REVIEW_OUTPUT"],
            "evidence_limitations": [],
            "evidence_ids": [],
            "recommendation": None,
            "rationale": "",
        }
    else:
        validation = validate_final_flow_expert_wording_review_output(parsed, packet)
    invocation_status = str(raw.get("invocation_status", "")).strip().upper()
    # Direct injected reviewers have no transport envelope; a valid structured
    # result is considered an injected successful review.  Live/provider
    # envelopes must explicitly report SUCCESS so a parsed error cannot count.
    if not invocation_status:
        invocation_status = "SUCCESS" if parsed is not None else "PARSE_FAILURE"
    invocation_success = invocation_status in {"SUCCESS", "PASS", "COMPLETED", "COMPLETE"}
    status = validation["status"] if invocation_success else "INCOMPLETE"
    if not invocation_success and "INVOCATION_NOT_SUCCESSFUL" not in validation["validation_errors"]:
        validation = {
            **validation,
            "validation_errors": [
                *validation.get("validation_errors", []),
                "INVOCATION_NOT_SUCCESSFUL",
            ],
        }
    text_hash = canonical_json_sha256({"model_visible_text": packet["model_visible_text"]})
    packet_hash = canonical_json_sha256(packet)
    provenance = {
        "review_type": "FINAL_FLOW_EXPERT_WORDING_REVIEW",
        "reviewer_agent_id": raw.get("agent_profile_id"),
        "reviewer_model_family": raw.get("model_family"),
        "role": raw.get("role", "scientific_reviewer"),
        "execution_mode": raw.get("execution_mode", "INJECTED_REVIEWER"),
        "invocation_status": invocation_status,
        "live_model_calls": raw.get("live_model_calls") is True,
        "semantic_contract_sha256": packet["semantic_contract_sha256"],
        "review_packet_sha256": packet_hash,
        "reviewed_model_visible_text_sha256": text_hash,
        "reviewed_model_visible_text": packet["model_visible_text"],
    }
    result = {
        **validation,
        "status": status,
        "final_flow_expert_review_status": status if invocation_success else "NOT_ESTABLISHED",
        "final_flow_expert_wording_review_version": (
            FINAL_FLOW_EXPERT_WORDING_REVIEW_PACKET_VERSION
        ),
        "semantic_contract_sha256": packet["semantic_contract_sha256"],
        "review_packet_sha256": packet_hash,
        "reviewed_model_visible_text_sha256": text_hash,
        "reviewed_model_visible_text": packet["model_visible_text"],
        "review_call": copy.deepcopy(dict(raw)),
        "invocation_status": invocation_status,
        "excluded_from_scientific_denominator": not (invocation_success and status == "PASS"),
        "official_scq_executed": False,
        "scientific_target_validity_judged": False,
        "core_membership_affected": False,
        "provenance": provenance,
    }
    return result


def audit_post_rewrite_semantic_fidelity(
    semantic_projection: Mapping[str, Any],
    model_visible_text: str,
    *,
    reviewer: PostRewriteSemanticReviewer,
    dataset_context: Mapping[str, Any] | None = None,
    case_id: str | None = None,
    condition: str | None = None,
    semantic_visibility: Mapping[str, Any] | None = None,
    semantic_contract_sha256: str | None = None,
    semantic_hash_callback: SemanticHashCallback | None = None,
) -> dict[str, Any]:
    """Audit final free text independently from its authoring/renderer path."""

    if not callable(reviewer):
        raise ValueError("reviewer must be callable")
    packet = build_post_rewrite_semantic_fidelity_packet(
        semantic_projection,
        model_visible_text,
        dataset_context=dataset_context,
        case_id=case_id,
        condition=condition,
        semantic_visibility=semantic_visibility,
        semantic_contract_sha256=semantic_contract_sha256,
        semantic_hash_callback=semantic_hash_callback,
    )
    raw = reviewer(copy.deepcopy(packet))
    if not isinstance(raw, Mapping):
        raise ValueError("reviewer must return an object")
    parsed = _parsed_model_output(raw)
    if parsed is None:
        result: dict[str, Any] = {
            "status": "INFRA_INVALID",
            "checks": {},
            "failure_codes": [],
            "validation_errors": ["NO_PARSED_FIDELITY_OUTPUT"],
            "excluded_from_semantic_fidelity_decision": True,
        }
    else:
        result = validate_post_rewrite_semantic_fidelity_output(parsed, packet)
        result["excluded_from_semantic_fidelity_decision"] = False
    if raw.get("invocation_status") and raw["invocation_status"] != "SUCCESS":
        result.update(status="INFRA_INVALID", excluded_from_semantic_fidelity_decision=True)
        result.setdefault("validation_errors", []).append("INVOCATION_NOT_SUCCESSFUL")
    result.update(
        {
            "post_rewrite_semantic_fidelity_version": (
                POST_REWRITE_SEMANTIC_FIDELITY_PACKET_VERSION
            ),
            "semantic_contract_sha256": packet["semantic_contract_sha256"],
            "semantic_visibility": copy.deepcopy(packet["semantic_visibility"]),
            "model_visible_text_sha256": canonical_json_sha256(
                {"model_visible_text": packet["model_visible_text"]}
            ),
            "review_call": copy.deepcopy(dict(raw)),
            "construction_time_semantic_realization_check": True,
            "official_scq_executed": False,
            "qualification_gate_created": False,
            "scientific_target_validity_judged": False,
            "core_membership_affected": False,
        }
    )
    return result


def validate_o1_f2_effective_o_semantics(
    o1_f1_projection: Mapping[str, Any],
    o1_f2_projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that O1-F2 changes Finding responsibility and nothing in O."""

    left = _semantic_contract_view(o1_f1_projection)
    right = _semantic_contract_view(o1_f2_projection)
    failures: list[str] = []
    left_condition = str(o1_f1_projection.get("condition", ""))
    right_condition = str(o1_f2_projection.get("condition", ""))
    if left_condition and left_condition != "O1-F1":
        failures.append("LEFT_CONDITION_NOT_O1_F1")
    if right_condition and right_condition != "O1-F2":
        failures.append("RIGHT_CONDITION_NOT_O1_F2")
    if left["finding_responsibility"]["finding_mode"] != "F1":
        failures.append("O1_F1_FINDING_MODE_INVALID")
    if right["finding_responsibility"]["finding_mode"] != "F2":
        failures.append("O1_F2_FINDING_MODE_INVALID")
    if left["scientific_target"] != right["scientific_target"]:
        failures.append("O1_F2_TARGET_DRIFT")
    if canonical_json_sha256(left["scientific_scope"]) != canonical_json_sha256(
        right["scientific_scope"]
    ):
        failures.append("O1_F2_SCOPE_DRIFT")
    if left["unresolved_operationalization_dimensions"] or right[
        "unresolved_operationalization_dimensions"
    ]:
        failures.append("O1_F2_MUST_REUSE_FULLY_FIXED_O1")
    left_o = sorted(
        left["fixed_operationalization"], key=lambda item: item["dimension_id"]
    )
    right_o = sorted(
        right["fixed_operationalization"], key=lambda item: item["dimension_id"]
    )
    if canonical_json_sha256(left_o) != canonical_json_sha256(right_o):
        failures.append("O1_F2_EFFECTIVE_O_DRIFT")
    return {
        "status": "PASS" if not failures else "REVISE",
        "failure_codes": sorted(set(failures)),
        "same_effective_o": "O1_F2_EFFECTIVE_O_DRIFT" not in failures
        and "O1_F2_MUST_REUSE_FULLY_FIXED_O1" not in failures,
        "finding_responsibility_changes_only": not failures,
    }


def nominate_primary_question_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    preferred_candidate_id: str | None = None,
) -> dict[str, Any]:
    """Nominate one audited candidate for inspection, never final selection."""

    rows = list(candidates)
    seen: set[str] = set()
    eligible: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, candidate in enumerate(rows):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidates[{index}] must be an object")
        candidate_id = _nonempty(
            candidate.get("candidate_id"), field=f"candidates[{index}].candidate_id"
        )
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        fidelity = candidate.get("post_rewrite_semantic_fidelity")
        hccq = candidate.get("hccq_presentation")
        if not isinstance(hccq, Mapping):
            hccq = candidate.get("hccq_presentation_audit")
        fidelity_status = (
            str(fidelity.get("status", "")) if isinstance(fidelity, Mapping) else ""
        )
        hccq_status = str(hccq.get("status", "")) if isinstance(hccq, Mapping) else ""
        if fidelity_status == "PASS" and hccq_status == "PASS":
            eligible.append(candidate)
        else:
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "semantic_fidelity_status": fidelity_status or "NOT_EVALUATED",
                    "hccq_status": hccq_status or "NOT_EVALUATED",
                }
            )
    preferred = str(preferred_candidate_id or "").strip()
    selected: Mapping[str, Any] | None = None
    basis = "NO_ELIGIBLE_CANDIDATE"
    if preferred:
        selected = next(
            (
                candidate
                for candidate in eligible
                if str(candidate.get("candidate_id", "")) == preferred
            ),
            None,
        )
        if selected is not None:
            basis = "PREFERRED_CANDIDATE_AFTER_AUDITS"
    if selected is None and eligible:
        selected = eligible[0]
        basis = "FIRST_ELIGIBLE_IN_AUTHORED_ORDER"
    return {
        "status": (
            "PRIMARY_CANDIDATE_NOMINATED"
            if selected is not None
            else "NO_ELIGIBLE_PRIMARY_CANDIDATE"
        ),
        "primary_candidate_id": (
            str(selected.get("candidate_id")) if selected is not None else None
        ),
        "primary_model_visible_text": (
            str(selected.get("model_visible_text", "")) if selected is not None else None
        ),
        "nomination_basis": basis,
        "eligible_candidate_ids": [
            str(candidate.get("candidate_id")) for candidate in eligible
        ],
        "rejected_candidates": rejected,
        "nomination_only": True,
        "human_final_selection_required": True,
        "human_question_selected": False,
    }


QUESTION_PRESENTATION_AUTHORING_INSTRUCTION = (
    "Act only as a scientific-question wording author. The payload contains a frozen canonical "
    "scientific target, scope, fixed Operationalization clauses, unresolved Operationalization "
    "dimensions, Finding responsibility, and relevant dataset context. Return exactly one JSON "
    "object with one field, candidates. candidates must contain exactly candidate_count objects; "
    "each object must contain only candidate_id, style, and model_visible_text. style must be "
    "DIRECT_SCIENTIFIC_QUESTION, CONCISE_EXPERT_REQUEST, or INVESTIGATION_ORIENTED, with no "
    "duplicate style or text. Write each complete model-visible question as a natural request "
    "from a scientific collaborator. Headings are optional and syntax should vary naturally. "
    "Do not use a condition-specific wording template. Preserve every fixed choice and make every "
    "unresolved choice understandable as a respondent-selected scientific choice without exposing "
    "internal dimension names. Preserve every F1 required Finding. For F2, do not select or require "
    "one particular optional characterization; ask for a scientifically relevant characterization "
    "chosen by the respondent while preserving the supplied adequate-core responsibility. Do not add "
    "scientific obligations. If repair_feedback is present, rewrite only the listed defects; do not "
    "broaden the target, Operationalization space, Finding responsibility, or scope while repairing. "
    "Translate storage/backend phrases into faithful domain language wherever "
    "the dataset context makes that possible: for example, say available gas-concentration fields, "
    "selected field, or location coordinates instead of c-prefixed scalar, stored field identifier, "
    "or exact stored grid-point coordinate. The examples 'selected field' and 'report which field was "
    "selected' apply only when the frozen contract explicitly contains observable/field selection as "
    "an Operationalization or Finding responsibility. Dataset context or backend field mappings alone "
    "never create a respondent duty to choose or report a field. When an F1 requirement explicitly "
    "identifies the selected field, "
    "express it as report which field was selected; never ask for a dataset field identifier, field "
    "identifier, field ID, or field name. This natural realization preserves the Finding responsibility. "
    "Do not require mechanical transcription. Keep a backend "
    "term only when removing it would genuinely change the frozen scientific meaning. Follow semantic_visibility: "
    "QUESTION_VISIBLE information must be conveyed in model_visible_text; DATASET_CONTEXT_VISIBLE information may "
    "be conveyed by the supplied dataset_context and need not be repeated literally; BACKEND_ONLY applies only to "
    "non-scientific presentation support details and must not be exposed as a respondent obligation. Do not change or critique the "
    "scientific semantics. Do not make a standalone how-many, number-of, count, or enumeration "
    "request the primary scientific target; a count may be included only as a supporting Finding "
    "for an already defined physical object or field. Do not make SCQ, SRAC, Ground Truth, curator, "
    "or benchmark decisions."
)


POST_REWRITE_SEMANTIC_FIDELITY_INSTRUCTION = (
    "Independently compare the actual model_visible_text with the frozen semantic contract. "
    "This is a post-rewrite construction audit, not question authoring and not official SCQ. "
    "Return exactly one JSON object containing scientific_target_preserved and "
    "scientific_scope_preserved booleans; fixed_o_checks with exactly one {dimension_id, "
    "preserved} object per fixed dimension; unresolved_o_checks with exactly one {dimension_id, "
    "remains_open} object per unresolved dimension; finding_responsibility_check with exactly "
    "finding_mode, responsibility_preserved, f1_remains_fixed, and f2_remains_open; "
    "no_additional_scientific_obligation; additional_scientific_obligations as a string list; "
    "and a concise non-empty rationale. Set non-applicable F1/F2 booleans true. A choice is not "
    "open if the wording selects a method, threshold, representation, or other value for it. "
    "F2 fails when wording creates a fixed or exhaustive Finding checklist. Judge preservation "
    "of meaning, not phrase overlap, so substantial paraphrase may pass. Apply semantic_visibility when deciding "
    "preservation: QUESTION_VISIBLE meaning must be present in model_visible_text; DATASET_CONTEXT_VISIBLE meaning "
    "may be preserved jointly by the final text and dataset_context without repeating backend field names; "
    "BACKEND_ONLY presentation support is not a scientific obligation. Never treat a scientific target, unresolved "
    "Operationalization choice, or Finding responsibility as hidden backend information. Do not judge whether "
    "the scientific target itself is valid and do not make benchmark membership decisions."
)


FINAL_FLOW_EXPERT_WORDING_REVIEW_INSTRUCTION = (
    "Act as the canonical FlowIntentBench Flow Expert reviewing the actual final "
    "model-visible scientific question text. This is a presentation-quality and "
    "answerability review only; do not rewrite the question, alter the semantic "
    "contract, select a target, or make SCQ/SRAC/Ground Truth decisions. Use only "
    "the supplied frozen target, scope, fixed and unresolved Operationalization "
    "responsibilities, Finding responsibility, dataset context, and evidence context. "
    "Return exactly one JSON object with these fields: question_meaningful, "
    "target_stable, target_wording_faithful, scope_preserved, fixed_o_exposed, "
    "unresolved_o_left_open, no_unintended_o_responsibility, "
    "finding_responsibility_correct, and answerable_in_principle (all booleans); "
    "evidence_limitations (a JSON list of concise strings, possibly empty); "
    "evidence_ids (a JSON list containing only cited evidence_id values, possibly "
    "empty); recommendation (exactly KEEP, REVISE_WORDING, REVISE_TARGET, "
    "REVISE_OPERATIONALIZATION_SPACE, or NOT_SCIENTIFICALLY_SUPPORTABLE); and a "
    "non-empty rationale. Check the wording itself, including whether the scientific "
    "question is meaningful in this physical context, whether target and scope are "
    "faithful, whether every fixed O is exposed and every unresolved O remains open, "
    "whether an undeclared O responsibility was introduced, whether F responsibility "
    "is correct, whether the request is clear and actionable for a scientific participant, "
    "For F2, finding selection remains the respondent responsibility. Adequate core sets are "
    "internal adequacy criteria, not a required literal checklist in the question. Do not "
    "require the question to name every reference property or convert F2 into bounded F1. "
    "whether it makes an unsupported physical assertion, whether it imposes unnecessary "
    "mechanical/schema/protocol burden, and whether the request is answerable in principle. "
    "Judge scientific meaning rather than literal backend phrasing. When dataset context "
    "already maps storage-level names to a scientific field class, human-facing wording such "
    "as 'documented gas-concentration fields' or 'every stored gas-concentration scalar' may "
    "faithfully expose that fixed scientific choice without repeating a c-prefix, array name, "
    "field identifier, or other backend mapping detail. Likewise, 'report which field was "
    "selected' satisfies a selected-field identity Finding without requiring identifier jargon. "
    "Do not mark such faithful natural-language realization as an omitted fixed O or an "
    "unintended respondent choice. "
    "Reflect any such failure through the existing booleans, recommendation, evidence "
    "limitations, and rationale; do not invent extra output fields. For O2/O3, "
    "different valid O choices may produce different G(O), rankings, or selected "
    "features; do not require result agreement. Report evidence limitations instead "
    "of inventing unsupported claims. A standalone how-many, number-of, count, or enumeration "
    "request is not a meaningful primary scientific target; recommend REVISE_TARGET for that case."
)


def _run_flow_expert_presentation_call(
    repository_root: str,
    packet: Mapping[str, Any],
    *,
    instruction: str,
    config_path: str | None,
    api_key: str | None,
    timeout: float,
    run_id: str | None,
    max_output_tokens: int,
) -> dict[str, Any]:
    from pathlib import Path

    from .live_agents import LiveModelCaller
    from .scientific_expert_review import FLOW_SCIENTIFIC_REVIEWER_PROFILE

    root = Path(repository_root).resolve()
    caller = LiveModelCaller(
        root,
        FLOW_SCIENTIFIC_REVIEWER_PROFILE,
        config_path=config_path,
        api_key=api_key,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
    )
    identity = {
        **dict(packet.get("case_identity", {})),
        "presentation_mode": str(packet.get("task", "")),
    }
    if run_id is not None:
        identity["presentation_run_id"] = run_id
    return caller.call(
        role="scientific_reviewer",
        visible_payload=packet,
        instruction=instruction,
        identity=identity,
        tool_free=True,
    )


def run_flow_expert_question_presentation_authoring(
    repository_root: str,
    packet: Mapping[str, Any],
    *,
    config_path: str | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Invoke the existing Flow Expert profile in wording-authoring mode."""

    validation = validate_question_presentation_authoring_packet(packet)
    if validation["status"] != "PASS":
        return {
            "status": "INPUT_REJECTED",
            "invocation_status": "NOT_RUN",
            "parsed_result": {},
            "input_validation": validation,
            "live_model_calls": False,
        }
    return _run_flow_expert_presentation_call(
        repository_root,
        packet,
        instruction=QUESTION_PRESENTATION_AUTHORING_INSTRUCTION,
        config_path=config_path,
        api_key=api_key,
        timeout=timeout,
        run_id=run_id,
        max_output_tokens=3000,
    )


def post_rewrite_semantic_fidelity_instruction(packet: Mapping[str, Any]) -> str:
    """Build the exact condition-specific fidelity instruction for calls and cache checks."""
    fixed_ids = [
        str(item.get("dimension_id", ""))
        for item in packet.get("fixed_operationalization", ())
        if isinstance(item, Mapping)
    ]
    unresolved_ids = [
        str(item)
        for item in packet.get("unresolved_operationalization_dimensions", ())
    ]
    instruction = (
        POST_REWRITE_SEMANTIC_FIDELITY_INSTRUCTION
        + " For this call, fixed_o_checks must contain exactly these dimension_id "
        "values (and must be [] when this list is empty): "
        + json.dumps(fixed_ids)
        + ". unresolved_o_checks must contain exactly these dimension_id values "
        "(and must be [] when this list is empty): "
        + json.dumps(unresolved_ids)
        + ". Finding roles belong only in finding_responsibility_check, never in "
        "fixed_o_checks or unresolved_o_checks."
    )
    return instruction


def run_flow_expert_post_rewrite_semantic_fidelity(
    repository_root: str,
    packet: Mapping[str, Any],
    *,
    config_path: str | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Invoke a separate Flow Expert call for actual-text fidelity."""

    validation = validate_post_rewrite_semantic_fidelity_packet(packet)
    if validation["status"] != "PASS":
        return {
            "status": "INPUT_REJECTED",
            "invocation_status": "NOT_RUN",
            "parsed_result": {},
            "input_validation": validation,
            "live_model_calls": False,
        }
    instruction = post_rewrite_semantic_fidelity_instruction(packet)
    return _run_flow_expert_presentation_call(
        repository_root,
        packet,
        instruction=instruction,
        config_path=config_path,
        api_key=api_key,
        timeout=timeout,
        run_id=run_id,
        max_output_tokens=2200,
    )


def run_flow_expert_final_flow_wording_review(
    repository_root: str,
    packet: Mapping[str, Any],
    *,
    config_path: str | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Invoke the existing Flow Expert on the actual final question text.

    This is a transport wrapper only.  The returned envelope is intentionally
    left uncompiled so :func:`audit_final_flow_expert_wording` can apply the
    same strict validator to injected and live calls.
    """

    validation = validate_final_flow_expert_wording_review_packet(packet)
    if validation["status"] != "PASS":
        return {
            "status": "INPUT_REJECTED",
            "invocation_status": "NOT_RUN",
            "parsed_result": {},
            "input_validation": validation,
            "live_model_calls": False,
        }
    return _run_flow_expert_presentation_call(
        repository_root,
        packet,
        instruction=FINAL_FLOW_EXPERT_WORDING_REVIEW_INSTRUCTION,
        config_path=config_path,
        api_key=api_key,
        timeout=timeout,
        run_id=run_id,
        max_output_tokens=1800,
    )


# Explicit aliases keep the API discoverable for callers that use the shorter
# ``final_wording`` terminology while retaining one implementation/profile.
run_flow_expert_final_wording_review = run_flow_expert_final_flow_wording_review
run_final_flow_expert_wording_review = run_flow_expert_final_flow_wording_review


def _question_for_style(style: str, target: str) -> str:
    target = target.rstrip(" .?")
    if style == "DIRECT_SCIENTIFIC_QUESTION":
        return f"What does the supplied dataset show about {target}?"
    if style == "CONCISE_EXPERT_REQUEST":
        return f"Determine {target} from the supplied dataset."
    if style == "INVESTIGATION_ORIENTED":
        return f"Using the supplied dataset, investigate {target}."
    raise ValueError(f"unsupported realization style: {style}")


def _finding_instruction(contract: Mapping[str, Any]) -> str:
    mode = str(contract["finding_mode"])
    if mode == "F1":
        requirements = [
            _human_visible_statement(str(item))
            for item in contract["fixed_requirements"]
        ]
        return "; ".join(requirements)
    mandatory = [
        _human_visible_statement(_display_dimension(item))
        for item in contract["mandatory_roles"]
    ]
    alternatives = []
    for group in contract["alternative_role_groups"]:
        roles = ", ".join(
            _human_visible_statement(_display_dimension(item))
            for item in group["roles"]
        )
        alternatives.append(
            f"at least {group['min_required']} of: {roles}"
        )
    core = "; ".join(
        [
            "Select and report the principal scientifically relevant findings supported by the data",
            f"include {', '.join(mandatory)}" if mandatory else "",
            *alternatives,
        ]
    )
    return "; ".join(part for part in core.split("; ") if part)


def _human_visible_scope(scope: str) -> str:
    """Render reader details in plain scientific language.

    Reader-critical names remain in dataset context and the canonical
    projection.  They are not repeated as backend schema prose in the
    human-facing question.  This is presentation-only and does not alter the
    semantic contract used by the evaluator.
    """

    replacements = (
        ("the supplied c1 and c11-c17 concentration arrays are point-associated",
         "the supplied gas-concentration fields are available for comparison"),
        ("speed is the Euclidean magnitude of the stored point-data vector array named `vectors`",
         "use the supplied velocity vectors to compute speed as their Euclidean magnitude"),
        ("speed is the Euclidean magnitude of the stored vectors array",
         "use the supplied velocity vectors to compute speed as their Euclidean magnitude"),
        ("speed is the Euclidean magnitude of the stored uvw array",
         "use the supplied velocity vectors to compute speed as their Euclidean magnitude"),
        ("speed is the magnitude of the canonical reader-derived Velocity array",
         "use the supplied velocity vectors to compute speed as their magnitude"),
    )
    result = scope
    for source, target in replacements:
        result = result.replace(source, target)
    return result


def _human_visible_statement(statement: str) -> str:
    """Remove implementation-only storage wording from a visible clause."""

    replacements = (
        ("stored point-data vector array named `vectors`", "supplied velocity vectors"),
        ("stored vectors array", "supplied velocity vectors"),
        ("stored uvw array", "supplied velocity vectors"),
        ("reader-exposed Velocity array", "supplied velocity vectors"),
        ("canonical reader-derived Velocity array", "supplied velocity vectors"),
        ("point-associated c1 or c11-c17 scalar array", "available gas-concentration field"),
        ("stored field identifier", "selected concentration field"),
        ("selected concentration-field identifier", "selected concentration field"),
        ("stored Density values", "density values"),
    )
    result = statement
    for source, target in replacements:
        result = result.replace(source, target)
    return result


def render_question_candidate(candidate: Mapping[str, Any]) -> str:
    """Render the final model-visible text from structured presentation fields."""

    lines = [
        "Scientific Question",
        _nonempty(candidate.get("scientific_question"), field="scientific_question"),
    ]
    scope = str(
        candidate.get("scientific_scope_visible", candidate.get("scientific_scope", ""))
    ).strip()
    if scope:
        lines.extend(["", "Scientific Scope", scope])
    constraints = _sequence(
        candidate.get("analysis_constraints"), field="analysis_constraints"
    )
    if constraints:
        lines.extend(["", "Analysis Constraints"])
        for item in constraints:
            if not isinstance(item, Mapping):
                raise ValueError("analysis_constraints must contain objects")
            lines.append(
                f"- {_display_dimension(str(item.get('dimension_id', '')))}: "
                f"{str(item.get('visible_statement', item.get('statement', ''))).strip()}"
            )
    choices = _sequence(
        candidate.get("open_analysis_choices"), field="open_analysis_choices"
    )
    if choices:
        lines.extend(["", "Open Analysis Choices"])
        lines.append(
            "- Select scientifically defensible choices for: "
            + ", ".join(_display_dimension(str(item)) for item in choices)
            + "."
        )
    findings = candidate.get("requested_findings")
    if not isinstance(findings, Mapping):
        raise ValueError("requested_findings must be an object")
    lines.extend(["", "Requested Findings", _finding_instruction(findings)])
    return "\n".join(lines).strip() + "\n"


def realize_human_questions(
    semantic_projection: Mapping[str, Any],
    *,
    case_id: str | None = None,
    condition: str | None = None,
    candidate_count: int = 3,
    semantic_contract_sha256: str | None = None,
    semantic_hash_callback: SemanticHashCallback | None = None,
) -> dict[str, Any]:
    """Build two or three independent, scientifically bound realizations."""

    if candidate_count not in {2, 3}:
        raise ValueError("candidate_count must be 2 or 3")
    projection = copy.deepcopy(dict(semantic_projection))
    target = _target(projection)
    scope = _scope(projection)
    resolved = _dimension_rows(projection)
    unresolved = _unresolved_dimensions(projection)
    overlap = {row["dimension_id"] for row in resolved} & set(unresolved)
    if overlap:
        raise ValueError(f"resolved and unresolved O dimensions overlap: {sorted(overlap)}")
    finding = _finding_contract(projection)
    semantic_hash = _projection_hash(
        projection,
        semantic_contract_sha256=semantic_contract_sha256,
        semantic_hash_callback=semantic_hash_callback,
    )
    candidates: list[dict[str, Any]] = []
    for index, style in enumerate(QUESTION_REALIZATION_STYLES[:candidate_count], start=1):
        candidate: dict[str, Any] = {
            "candidate_id": f"presentation-{index:02d}",
            "presentation_version": QUESTION_PRESENTATION_VERSION,
            "style": style,
            "semantic_contract_sha256": semantic_hash,
            "scientific_question": _question_for_style(style, target),
            "scientific_scope": scope,
            "scientific_scope_visible": _human_visible_scope(scope),
            "analysis_constraints": copy.deepcopy(resolved),
            "open_analysis_choices": list(unresolved),
            "requested_findings": copy.deepcopy(finding),
            "additional_scientific_obligations": [],
        }
        for constraint in candidate["analysis_constraints"]:
            constraint["visible_statement"] = _human_visible_statement(
                str(constraint.get("statement", ""))
            )
        candidate["model_visible_text"] = render_question_candidate(candidate)
        candidate["presentation_sha256"] = canonical_json_sha256(
            {key: value for key, value in candidate.items() if key != "presentation_sha256"}
        )
        candidates.append(candidate)
    return {
        "presentation_version": QUESTION_PRESENTATION_VERSION,
        "case_id": str(case_id or projection.get("case_id", "")),
        "condition": str(condition or projection.get("condition", "")),
        "semantic_contract_sha256": semantic_hash,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "source_question_rewritten": False,
        "official_scq_required_for_selected_final_question": True,
        "scientific_qualification_performed": False,
        "core_membership_affected": False,
    }


def _clause_matches(candidate: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    statement = _semantic_text(candidate.get("statement"))
    accepted = {
        _semantic_text(expected.get("statement")),
        *{
            _semantic_text(item)
            for item in expected.get("allowed_visible_realizations", ()) or ()
        },
    }
    accepted.discard("")
    if statement in accepted:
        return True
    anchors = [
        _semantic_text(item)
        for item in expected.get("required_semantic_anchors", ()) or ()
        if _semantic_text(item)
    ]
    return bool(anchors) and all(anchor in statement for anchor in anchors)


def _finding_contract_equal(
    candidate: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    if str(candidate.get("finding_mode", "")) != str(expected.get("finding_mode", "")):
        return False
    for field in (
        "fixed_requirements",
        "mandatory_roles",
        "alternative_role_groups",
        "adequate_core_sets",
    ):
        if canonical_json_sha256(candidate.get(field, [])) != canonical_json_sha256(
            expected.get(field, [])
        ):
            return False
    return True


def validate_question_semantic_fidelity(
    semantic_projection: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    semantic_contract_sha256: str | None = None,
    semantic_hash_callback: SemanticHashCallback | None = None,
) -> dict[str, Any]:
    """Fail closed when final presentation changes the supplied semantics."""

    projection = copy.deepcopy(dict(semantic_projection))
    target = _target(projection)
    scope = _scope(projection)
    resolved = _dimension_rows(projection)
    unresolved = _unresolved_dimensions(projection)
    finding = _finding_contract(projection)
    expected_hash = _projection_hash(
        projection,
        semantic_contract_sha256=semantic_contract_sha256,
        semantic_hash_callback=semantic_hash_callback,
    )
    checks: dict[str, str] = {}
    failures: list[str] = []

    if str(candidate.get("semantic_contract_sha256", "")) != expected_hash:
        failures.append("SEMANTIC_CONTRACT_BINDING_MISMATCH")
        checks["semantic_contract_binding"] = "FAIL"
    else:
        checks["semantic_contract_binding"] = "PASS"

    question = str(candidate.get("scientific_question", ""))
    if _semantic_text(target) not in _semantic_text(question):
        failures.append("TARGET_DRIFT")
        checks["target_preserved"] = "FAIL"
    else:
        checks["target_preserved"] = "PASS"
    candidate_scope = str(candidate.get("scientific_scope", ""))
    if _semantic_text(scope) != _semantic_text(candidate_scope):
        failures.append("SCOPE_DRIFT")
        checks["scope_preserved"] = "FAIL"
    else:
        checks["scope_preserved"] = "PASS"

    raw_constraints = candidate.get("analysis_constraints")
    constraints = (
        [dict(item) for item in raw_constraints if isinstance(item, Mapping)]
        if isinstance(raw_constraints, list)
        else []
    )
    by_dimension = {
        str(item.get("dimension_id", "")): item
        for item in constraints
        if str(item.get("dimension_id", "")).strip()
    }
    resolved_by_dimension = {row["dimension_id"]: row for row in resolved}
    missing_fixed = sorted(set(resolved_by_dimension) - set(by_dimension))
    changed_fixed = sorted(
        dimension
        for dimension in set(resolved_by_dimension) & set(by_dimension)
        if not _clause_matches(by_dimension[dimension], resolved_by_dimension[dimension])
    )
    if missing_fixed:
        failures.append("FIXED_O_DISAPPEARED")
    if changed_fixed:
        failures.append("FIXED_O_CHANGED")
    checks["resolved_o_preserved"] = (
        "PASS" if not missing_fixed and not changed_fixed else "FAIL"
    )

    principal = {
        str(item.get("dimension_id", item.get("dimension", ""))).strip()
        if isinstance(item, Mapping)
        else str(item).strip()
        for item in _sequence(
            projection.get("principal_operationalization_dimensions", ()),
            field="principal_operationalization_dimensions",
        )
    }
    principal.discard("")
    extra_constraints = sorted(set(by_dimension) - set(resolved_by_dimension))
    prematurely_fixed = sorted(set(extra_constraints) & set(unresolved))
    if prematurely_fixed:
        failures.append("UNRESOLVED_DIMENSION_PREMATURELY_FIXED")
    if set(extra_constraints) - set(unresolved):
        failures.append("EXTRA_SCIENTIFIC_OBLIGATION")
    choices = [
        str(item)
        for item in _sequence(
            candidate.get("open_analysis_choices"), field="open_analysis_choices"
        )
    ]
    if set(choices) != set(unresolved):
        failures.append("UNRESOLVED_O_SPACE_DRIFT")
    if principal and principal != set(resolved_by_dimension) | set(unresolved):
        failures.append("PROJECTION_O_COVERAGE_INVALID")
    checks["unresolved_o_remains_open"] = (
        "PASS"
        if not prematurely_fixed and set(choices) == set(unresolved)
        else "FAIL"
    )

    candidate_findings = candidate.get("requested_findings")
    finding_match = isinstance(candidate_findings, Mapping) and _finding_contract_equal(
        candidate_findings, finding
    )
    if not finding_match:
        failures.append("FINDING_RESPONSIBILITY_DRIFT")
    candidate_finding_mode = (
        str(candidate_findings.get("finding_mode", ""))
        if isinstance(candidate_findings, Mapping)
        else ""
    )
    if finding["finding_mode"] == "F1" and candidate_finding_mode == "F2":
        failures.append("F1_BECAME_OPEN")
    if finding["finding_mode"] == "F2" and (
        candidate_finding_mode == "F1"
        or bool(
            candidate_findings.get("fixed_requirements")
            if isinstance(candidate_findings, Mapping)
            else False
        )
    ):
        failures.append("F2_BECAME_HIDDEN_F1")
    finding_text = _semantic_text(
        _finding_instruction(candidate_findings)
        if isinstance(candidate_findings, Mapping)
        else ""
    )
    if finding["finding_mode"] == "F1" and any(
        marker in finding_text
        for marker in ("select principal", "whatever findings", "any other findings")
    ):
        failures.append("F1_BECAME_OPEN")
    if finding["finding_mode"] == "F2" and any(
        marker in finding_text
        for marker in ("report only", "exactly these", "no other findings")
    ):
        failures.append("F2_BECAME_HIDDEN_F1")
    checks["finding_responsibility_preserved"] = "PASS" if finding_match else "FAIL"

    additional = _sequence(
        candidate.get("additional_scientific_obligations"),
        field="additional_scientific_obligations",
    )
    style = str(candidate.get("style", ""))
    allowed_question = (
        _question_for_style(style, target)
        if style in QUESTION_REALIZATION_STYLES
        else ""
    )
    if additional or _semantic_text(question) != _semantic_text(allowed_question):
        failures.append("EXTRA_SCIENTIFIC_OBLIGATION")
    checks["no_extra_scientific_obligation"] = (
        "PASS"
        if "EXTRA_SCIENTIFIC_OBLIGATION" not in failures
        else "FAIL"
    )

    try:
        rendered = render_question_candidate(candidate)
    except ValueError:
        rendered = ""
    if rendered != str(candidate.get("model_visible_text", "")):
        failures.append("PRESENTATION_RENDER_MISMATCH")
        checks["final_model_visible_text_bound"] = "FAIL"
    else:
        checks["final_model_visible_text_bound"] = "PASS"
    expected_presentation_hash = canonical_json_sha256(
        {
            key: value
            for key, value in candidate.items()
            if key != "presentation_sha256"
        }
    )
    if str(candidate.get("presentation_sha256", "")) != expected_presentation_hash:
        failures.append("PRESENTATION_HASH_MISMATCH")
        checks["presentation_hash"] = "FAIL"
    else:
        checks["presentation_hash"] = "PASS"
    unique_failures = sorted(set(failures))
    return {
        "status": "PASS" if not unique_failures else "REVISE",
        "semantic_contract_sha256": expected_hash,
        "presentation_sha256": candidate.get("presentation_sha256"),
        "checks": checks,
        "failure_codes": unique_failures,
        "official_scq_required": True,
        "scientific_target_validity_judged": False,
        "core_membership_affected": False,
    }


__all__ = [
    "FINAL_FLOW_EXPERT_WORDING_REVIEW_INSTRUCTION",
    "FINAL_FLOW_EXPERT_WORDING_REVIEW_PACKET_VERSION",
    "FINAL_FLOW_EXPERT_WORDING_REVIEW_RECOMMENDATIONS",
    "POST_REWRITE_SEMANTIC_FIDELITY_INSTRUCTION",
    "POST_REWRITE_SEMANTIC_FIDELITY_PACKET_VERSION",
    "QUESTION_PRESENTATION_AUTHORING_INSTRUCTION",
    "QUESTION_PRESENTATION_AUTHORING_PACKET_VERSION",
    "QUESTION_PRESENTATION_VERSION",
    "QUESTION_REALIZATION_STYLES",
    "SEMANTIC_VISIBILITY_CLASSES",
    "SEMANTIC_VISIBILITY_CLASSIFICATION_VERSION",
    "audit_post_rewrite_semantic_fidelity",
    "audit_final_flow_expert_wording",
    "author_human_questions",
    "build_final_flow_expert_wording_review_packet",
    "build_post_rewrite_semantic_fidelity_packet",
    "build_question_presentation_authoring_packet",
    "build_semantic_visibility_classification",
    "nominate_primary_question_candidate",
    "realize_human_questions",
    "render_question_candidate",
    "run_flow_expert_post_rewrite_semantic_fidelity",
    "run_flow_expert_question_presentation_authoring",
    "run_flow_expert_final_flow_wording_review",
    "run_flow_expert_final_wording_review",
    "run_final_flow_expert_wording_review",
    "validate_final_flow_expert_wording_review_output",
    "validate_final_flow_expert_wording_review_packet",
    "validate_post_rewrite_semantic_fidelity_output",
    "validate_post_rewrite_semantic_fidelity_packet",
    "validate_question_presentation_authoring_output",
    "validate_question_presentation_authoring_packet",
    "validate_question_semantic_fidelity",
    "validate_o1_f2_effective_o_semantics",
    "validate_semantic_visibility_classification",
]


def review_constructed_case_manifest(
    repository_root, manifest_path, *, config_path=None, max_workers=4, timeout=180.0
):
    """Apply the existing final-wording audit to explicit, hash-bound case bytes.

    This batch adapter makes no curator or scientific-admission decision.
    Cached reviews are reusable only for an identical complete review packet.
    """
    import json
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .case_repository import load_case_record
    from .scientific_semantics import build_scientific_semantic_projection

    root = Path(repository_root).resolve()
    path = (root / manifest_path).resolve()
    manifest = json.loads(path.read_text())
    from .agent_profile import load_agent_profile

    profile_hash = load_agent_profile(
        "flow-scientific-reviewer-gpt-5.6-sol", repository_root=root
    ).profile_sha256
    import hashlib
    from .review_lineage import live_review_call_matches

    rows = manifest["cases"]
    identities = [row["case_id"] for row in rows]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("review manifest must have nonempty unique case identities")
    if not 1 <= max_workers <= 16:
        raise ValueError("max_workers must be in [1, 16]")

    def review(row):
        import hashlib

        case = (
            {
                "case_id": row["case_id"],
                "family_id": row["family_id"],
                "dataset_id": row["dataset_id"],
                "condition": row["condition"],
            }
            if str(row.get("origin", "")).startswith("FROZEN_REFERENCE")
            else load_case_record(root, row["case_id"])
        )
        for name, key in [
            ("case_input", "case_input"),
            ("case_construction_metadata", "metadata"),
            ("case_context", "case_context"),
        ]:
            artifact = root / row[f"{name}_path"]
            if (
                hashlib.sha256(artifact.read_bytes()).hexdigest()
                != row[f"{name}_sha256"]
            ):
                raise ValueError(
                    "case artifact changed since manifest: " + str(artifact)
                )
            case[key] = json.loads(artifact.read_text())
        if row.get("scientific_semantic_projection_path"):
            projection_path = root / row["scientific_semantic_projection_path"]
            if (
                hashlib.sha256(projection_path.read_bytes()).hexdigest()
                != row["scientific_semantic_projection_sha256"]
            ):
                raise ValueError("semantic projection changed since manifest")
            projection = json.loads(projection_path.read_text())
        else:
            projection = build_scientific_semantic_projection(case)
        question = case["case_input"]["scientific_question"]
        context = {
            "case_context": case["case_context"],
            "data_metadata": case["case_input"]["flow_data"]["data_metadata"],
        }
        packet = build_final_flow_expert_wording_review_packet(
            projection,
            question,
            dataset_context=context,
            case_id=row["case_id"],
            condition=row["condition"],
        )
        output = (root / row["case_input_path"]).parent / "final_wording_review.json"
        cached = json.loads(output.read_text()) if output.is_file() else {}
        if not (
            cached.get("review_packet_sha256") == canonical_json_sha256(packet)
            and cached.get("final_flow_expert_review_status") in {"PASS", "REVISE"}
            and cached.get("provenance", {}).get("live_model_calls") is True
            and cached.get("invocation_status") == "SUCCESS"
            and live_review_call_matches(
                cached.get("review_call", {}),
                packet,
                FINAL_FLOW_EXPERT_WORDING_REVIEW_INSTRUCTION,
                profile_hash,
            )
        ):
            cached = audit_final_flow_expert_wording(
                projection,
                question,
                dataset_context=context,
                case_id=row["case_id"],
                condition=row["condition"],
                reviewer=lambda packet: run_flow_expert_final_flow_wording_review(
                    str(root),
                    packet,
                    config_path=str(config_path) if config_path else None,
                    timeout=timeout,
                ),
            )
            output.write_text(json.dumps(cached, indent=2) + "\n")
        return {
            "case_id": row["case_id"],
            "status": cached.get("final_flow_expert_review_status", "NOT_ESTABLISHED"),
            "invocation_status": cached.get("invocation_status"),
            "rationale": cached.get("rationale"),
            "review_packet_sha256": cached.get("review_packet_sha256"),
            "review_path": str(output.relative_to(root)),
        }

    results = []
    output = path.parent / "wording_review_manifest.json"

    def persist():
        summary = {
            "case_count": len(rows),
            "reviewed_count": len(results),
            "pass_count": sum(r["status"] == "PASS" for r in results),
            "cases": sorted(results, key=lambda row: row["case_id"]),
            "scientific_qualification": "NOT_ESTABLISHED_BY_WORDING_REVIEW",
        }
        output.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(review, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "case_id": row["case_id"],
                    "status": "NOT_ESTABLISHED",
                    "error": str(exc),
                }
            results.append(result)
            persist()
            print(
                f"wording {len(results)}/{len(rows)} {row['case_id']}: {result['status']}",
                flush=True,
            )
    return persist()

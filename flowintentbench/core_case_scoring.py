"""Fast, model-agnostic scoring for one existing benchmark answer.

The reviewer is used for bounded extraction and semantic judgments.  The host
process owns all arithmetic, evidence validation, value verification and
one-to-one assignment.  This keeps the fast path compatible with the frozen
metric names while preventing a reviewer from writing the score by omission.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


CORE_SCHEMA_VERSION = "core-case-scoring-v2"
SCORER_IMPLEMENTATION_VERSION = "deterministic-host-v4-source-bound-20260923"
_DEFAULT_DIMENSION_NAMES = (
    "feature_definition",
    "property_measure",
    "aggregation_or_representation",
)
_DIMENSION_STATUSES = {"EXTRACTED", "MISSING", "AMBIGUOUS", "CONFLICTING"}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value.strip()


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _json_value(value: Any, field: str) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-compatible and finite") from exc
    return value


def _value_equal(
    predicted: Any, reference: Any, verification: Mapping[str, Any] | None
) -> bool | None:
    """Apply only the explicit GT tolerance; never infer one from shape."""
    if reference is None:
        return True
    if predicted is None:
        return False
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        if not isinstance(predicted, (int, float)) or isinstance(predicted, bool):
            return False
        policy = verification or {}
        absolute = policy.get("absolute_tolerance")
        relative = policy.get("relative_tolerance")
        delta = abs(float(predicted) - float(reference))
        if absolute is not None and delta <= float(absolute):
            return True
        if relative is not None and delta <= float(relative) * max(abs(float(reference)), 1e-15):
            return True
        return delta == 0.0 if absolute is None and relative is None else False
    if isinstance(reference, list) and isinstance(predicted, list):
        if len(reference) != len(predicted):
            return False
        spatial = (verification or {}).get("spatial_tolerance")
        if spatial is None:
            return predicted == reference
        try:
            return math.sqrt(
                sum((float(a) - float(b)) ** 2 for a, b in zip(predicted, reference))
            ) <= float(spatial)
        except (TypeError, ValueError):
            return False
    return predicted == reference


def _normal_unit(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return " ".join(value.strip().lower().replace("²", "2").split()) or None


def _host_numeric_match(prediction: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    """Verify numeric/value claims on the host, independent of GPT flags."""
    ref_value = reference.get("value")
    if ref_value is None:
        return True
    predicted_value = prediction.get("value")
    if predicted_value is None:
        return False
    ref_unit = _normal_unit(reference.get("unit"))
    pred_unit = _normal_unit(prediction.get("unit"))
    # The core path has no implicit unit conversion. A declared GT unit must
    # be present and equivalent; conversion remains an explicit deep-audit
    # responsibility rather than a reviewer guess.
    if ref_unit is not None and pred_unit != ref_unit:
        return False
    return bool(_value_equal(predicted_value, ref_value, reference.get("verification")))


def _valid_span(answer: str, value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return False
    start, end = value
    return 0 <= start < end <= len(answer) and bool(answer[start:end].strip())


def _evidence_valid(answer: str, item: Mapping[str, Any]) -> bool:
    if item.get("evidence_present") is not True:
        return False
    if _valid_span(answer, item.get("evidence_span")):
        return True
    evidence_text = item.get("evidence_text")
    return (
        isinstance(evidence_text, str)
        and bool(evidence_text.strip())
        and evidence_text.strip() in answer
    )


def _evidence_ledger(answer: str, dimensions: Sequence[Mapping[str, Any]], findings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Record auditable evidence material without allowing it to affect scores."""
    ledger: list[dict[str, Any]] = []
    for kind, items in (("operationalization", dimensions), ("finding", findings)):
        for index, item in enumerate(items):
            span = item.get("evidence_span")
            literal = answer[span[0]:span[1]] if _valid_span(answer, span) else ""
            if not literal and isinstance(item.get("evidence_text"), str):
                literal = item["evidence_text"].strip()
            ledger.append({
                "kind": kind,
                "index": index,
                "valid": _evidence_valid(answer, item),
                "sha256": hashlib.sha256(literal.encode("utf-8")).hexdigest() if literal else None,
            })
    return ledger


def _reference_packet(ground_truth: Any) -> dict[str, Any]:
    """Convert the project's Pydantic GT object or plain JSON to a packet."""
    if hasattr(ground_truth, "model_dump"):
        ground_truth = ground_truth.model_dump(mode="json")
    if not isinstance(ground_truth, Mapping):
        raise ValueError("ground_truth must be a mapping or model_dump-compatible object")
    branches: list[dict[str, Any]] = []
    for branch in ground_truth.get("findings_by_operationalization", ()):
        branches.append(
            {
                "branch_id": branch.get("operationalization_id"),
                "findings": [dict(item) for item in branch.get("findings", ())],
            }
        )
    operationalizations: list[dict[str, Any]] = []
    for bundle in ground_truth.get("acceptable_operationalizations", ()):
        decisions = []
        for decision in bundle.get("decisions", ()):
            dimension = decision.get("dimension")
            if hasattr(dimension, "value"):
                dimension = dimension.value
            decisions.append(
                {"dimension": dimension, "statement": decision.get("statement", "")}
            )
        operationalizations.append(
            {"branch_id": bundle.get("operationalization_id"), "decisions": decisions}
        )
    if not branches or not operationalizations:
        raise ValueError("ground_truth has no accepted operationalization branches")
    return {"operationalizations": operationalizations, "branches": branches}


def _dimensions(gt: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for branch in gt.get("operationalizations", ()):
        for decision in branch.get("decisions", ()):
            name = decision.get("dimension")
            if isinstance(name, str) and name not in values:
                values.append(name)
    return tuple(values) or _DEFAULT_DIMENSION_NAMES


def _prompt_packet(
    case_input: Mapping[str, Any], answer: str, gt: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": CORE_SCHEMA_VERSION,
        "case": {
            "case_id": case_input.get("case_id"),
            "scientific_question": case_input.get("scientific_question", ""),
            "case_context": case_input.get("case_context", {}),
            "required_operationalization_dimensions": list(_dimensions(gt)),
            "unresolved_operationalization_dimensions": list(
                case_input.get("unresolved_operationalization_dimensions", ())
            ),
        },
        "answer": answer,
        "reference": gt,
    }


def _span_schema() -> dict[str, Any]:
    return {
        "type": ["array", "null"],
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
    }


def judgment_schema(dimensions: tuple[str, ...] | list[str] | None = None) -> dict[str, Any]:
    """Strict schema for one bounded GPT reviewer call."""
    dimensions = tuple(dimensions or _DEFAULT_DIMENSION_NAMES)
    dimension_item = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "dimension", "status", "statement", "evidence_present",
            "evidence_span", "evidence_text", "branch_matches",
        ],
        "properties": {
            "dimension": {"type": "string", "enum": list(dimensions)},
            "status": {"type": "string", "enum": sorted(_DIMENSION_STATUSES)},
            "statement": {"type": "string"},
            "evidence_present": {"type": "boolean"},
            "evidence_span": _span_schema(),
            "evidence_text": {"type": ["string", "null"]},
            "branch_matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["branch_id", "semantic_match"],
                    "properties": {
                        "branch_id": {"type": "string"},
                        "semantic_match": {"type": "boolean"},
                    },
                },
            },
        },
    }
    match_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["branch_id", "gt_finding_id", "semantic_match", "numeric_match", "role_match", "consistent"],
        "properties": {
            "branch_id": {"type": "string"},
            "gt_finding_id": {"type": ["string", "null"]},
            "semantic_match": {"type": "boolean"},
            # Required on the strict wire schema, nullable as a diagnostic;
            # the host deliberately ignores it for numeric scoring.
            "numeric_match": {"type": ["boolean", "null"]},
            "role_match": {"type": "boolean"},
            "consistent": {"type": "boolean"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer_completed", "operationalization", "findings", "branch_consistency"],
        "properties": {
            "answer_completed": {"type": "boolean"},
            "operationalization": {
                "type": "object",
                "additionalProperties": False,
                "required": ["dimensions"],
                "properties": {
                    "dimensions": {"type": "array", "items": dimension_item}
                },
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "prediction_id", "statement", "eligible", "evidence_present",
                        "evidence_span", "evidence_text", "value", "unit", "matches",
                    ],
                    "properties": {
                        "prediction_id": {"type": "string"},
                        "statement": {"type": "string"},
                        "eligible": {"type": "boolean"},
                        "evidence_present": {"type": "boolean"},
                        "evidence_span": _span_schema(),
                        "evidence_text": {"type": ["string", "null"]},
                        # YiAPI's strict schema subset requires each branch
                        # to be typed and does not accept an unconstrained
                        # multi-type field. Benchmark values are scalars,
                        # numeric vectors, or null when no value is present.
                        "value": {
                            "anyOf": [
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "array", "items": {"type": "number"}},
                                {"type": "null"},
                            ]
                        },
                        "unit": {"type": ["string", "null"]},
                        "matches": {"type": "array", "items": match_item},
                    },
                },
            },
            "branch_consistency": {
                "type": "object",
                "additionalProperties": False,
                "required": ["auditable", "reason"],
                "properties": {
                    "auditable": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            },
        },
    }


def scoring_prompt(case_input: Mapping[str, Any], answer: str, ground_truth: Any) -> str:
    """Build one bounded GT-aware extraction/matching prompt."""
    gt = _reference_packet(ground_truth)
    packet = _prompt_packet(case_input, answer, gt)
    return (
        "You are a bounded benchmark reviewer. Return only JSON matching the supplied schema. "
        "Extract claims explicitly present in ANSWER; never invent a missing method, value, "
        "unit, evidence or finding. For every operationalization dimension return the exact "
        "answer statement (empty only when missing), its status, a literal evidence span or "
        "text, and a semantic_match for every reference branch. For every atomic finding retain "
        "it even when wrong or extra. Return its explicit value and unit, evidence span/text, "
        "and semantic/role/consistency judgments for reference candidates. Do not use the GT "
        "value to manufacture a claim. numeric_match is ignored by the host and may be null. "
        "Do not write any score or omit a finding to improve precision. The host will verify "
        "numbers, units, evidence, assignment, and all metrics deterministically.\n\n"
        "INPUT PACKET:\n" + json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    )


def _check_keys(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"{field} has unknown fields: {sorted(extra)}")


def _validate_judgment(
    value: Mapping[str, Any], dimensions: tuple[str, ...] | None = None
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("review judgment must be an object")
    _check_keys(
        value,
        {"answer_completed", "operationalization", "findings", "branch_consistency"},
        "judgment",
    )
    for key in ("answer_completed", "operationalization", "findings", "branch_consistency"):
        if key not in value:
            raise ValueError(f"review judgment missing {key}")
    _bool(value["answer_completed"], "answer_completed")
    operation = value["operationalization"]
    if not isinstance(operation, Mapping):
        raise ValueError("operationalization must be an object")
    _check_keys(operation, {"dimensions"}, "operationalization")
    dims = operation.get("dimensions")
    if not isinstance(dims, list):
        raise ValueError("operationalization.dimensions must be a list")
    seen: set[str] = set()
    allowed = set(dimensions or _DEFAULT_DIMENSION_NAMES)
    for item in dims:
        if not isinstance(item, Mapping):
            raise ValueError("operationalization dimension must be an object")
        _check_keys(
            item,
            {"dimension", "status", "statement", "evidence_present", "evidence_span", "evidence_text", "branch_matches"},
            "operationalization dimension",
        )
        dimension = _text(item.get("dimension"), "dimension")
        if dimension not in allowed or dimension in seen:
            raise ValueError("invalid or duplicate operationalization dimension")
        seen.add(dimension)
        if item.get("status") not in _DIMENSION_STATUSES:
            raise ValueError("invalid operationalization status")
        _optional_text(item.get("statement"), "dimension statement")
        _bool(item.get("evidence_present"), "dimension evidence_present")
        if item.get("evidence_span") is not None and not isinstance(item.get("evidence_span"), (list, tuple)):
            raise ValueError("dimension evidence_span must be a pair or null")
        if item.get("evidence_text") is not None:
            _optional_text(item.get("evidence_text"), "dimension evidence_text")
        if item.get("evidence_present") and item.get("evidence_span") is None and not item.get("evidence_text"):
            raise ValueError("dimension evidence requires evidence_span or evidence_text")
        branch_matches = item.get("branch_matches")
        if not isinstance(branch_matches, list):
            raise ValueError("dimension branch_matches must be a list")
        branch_seen: set[str] = set()
        for match in branch_matches:
            if not isinstance(match, Mapping):
                raise ValueError("dimension branch match must be an object")
            _check_keys(match, {"branch_id", "semantic_match"}, "dimension branch match")
            branch_id = _text(match.get("branch_id"), "dimension branch_id")
            if branch_id in branch_seen:
                raise ValueError("duplicate dimension branch_id")
            branch_seen.add(branch_id)
            _bool(match.get("semantic_match"), "dimension semantic_match")
    findings = value["findings"]
    if not isinstance(findings, list):
        raise ValueError("findings must be a list")
    prediction_ids: set[str] = set()
    for finding in findings:
        if not isinstance(finding, Mapping):
            raise ValueError("finding must be an object")
        _check_keys(
            finding,
            {"prediction_id", "statement", "eligible", "evidence_present", "evidence_span", "evidence_text", "value", "unit", "matches"},
            "finding",
        )
        prediction_id = _text(finding.get("prediction_id"), "prediction_id")
        if prediction_id in prediction_ids:
            raise ValueError("prediction_id values must be unique")
        prediction_ids.add(prediction_id)
        _text(finding.get("statement"), "finding statement")
        _bool(finding.get("eligible"), "finding eligible")
        _bool(finding.get("evidence_present"), "finding evidence_present")
        _json_value(finding.get("value"), "finding value")
        if finding.get("unit") is not None:
            _text(finding.get("unit"), "finding unit")
        if finding.get("evidence_span") is not None and not isinstance(finding.get("evidence_span"), (list, tuple)):
            raise ValueError("finding evidence_span must be a pair or null")
        if finding.get("evidence_text") is not None:
            _optional_text(finding.get("evidence_text"), "finding evidence_text")
        if finding.get("evidence_present") and finding.get("evidence_span") is None and not finding.get("evidence_text"):
            raise ValueError("finding evidence requires evidence_span or evidence_text")
        matches = finding.get("matches")
        if not isinstance(matches, list):
            raise ValueError("finding.matches must be a list")
        match_seen: set[tuple[str, str | None]] = set()
        for match in matches:
            if not isinstance(match, Mapping):
                raise ValueError("finding match must be an object")
            _check_keys(match, {"branch_id", "gt_finding_id", "semantic_match", "numeric_match", "role_match", "consistent"}, "finding match")
            branch_id = _text(match.get("branch_id"), "match branch_id")
            gt_id = match.get("gt_finding_id")
            if gt_id is not None:
                gt_id = _text(gt_id, "match gt_finding_id")
            key = (branch_id, gt_id)
            if key in match_seen:
                raise ValueError("duplicate finding match")
            match_seen.add(key)
            for field in ("semantic_match", "role_match", "consistent"):
                _bool(match.get(field), f"match {field}")
            if match.get("numeric_match") is not None:
                _bool(match.get("numeric_match"), "match numeric_match")
    consistency = value["branch_consistency"]
    if not isinstance(consistency, Mapping):
        raise ValueError("branch_consistency must be an object")
    _check_keys(consistency, {"auditable", "reason"}, "branch_consistency")
    _bool(consistency.get("auditable"), "branch_consistency.auditable")
    _text(consistency.get("reason"), "branch_consistency.reason")


def _dimension_branch_match(item: Mapping[str, Any], branch_id: str) -> bool:
    if item.get("status") != "EXTRACTED":
        return False
    for match in item.get("branch_matches", ()):
        if match.get("branch_id") == branch_id:
            return bool(match.get("semantic_match"))
    return False


def _maximum_matching(
    predictions: Sequence[str],
    edges: Mapping[str, Sequence[tuple[str, bool, bool]]],
) -> list[tuple[str, str, bool, bool]]:
    """Maximum-cardinality deterministic bipartite assignment."""
    match_gt: dict[str, str] = {}

    def visit(prediction_id: str, seen: set[str]) -> bool:
        candidates = sorted(
            edges.get(prediction_id, ()),
            key=lambda item: (-int(item[1]), -int(item[2]), item[0]),
        )
        for gt_id, numeric_ok, consistent in candidates:
            if gt_id in seen:
                continue
            seen.add(gt_id)
            previous = match_gt.get(gt_id)
            if previous is None or visit(previous, seen):
                match_gt[gt_id] = prediction_id
                return True
        return False

    for prediction_id in predictions:
        visit(prediction_id, set())
    assigned: list[tuple[str, str, bool, bool]] = []
    for gt_id, prediction_id in match_gt.items():
        edge = next(
            (item for item in edges.get(prediction_id, ()) if item[0] == gt_id),
            None,
        )
        if edge is not None:
            assigned.append((prediction_id, gt_id, edge[1], edge[2]))
    return assigned


def _metric_block(
    branch_id: str,
    findings: list[Mapping[str, Any]],
    refs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    ref_by_id = {str(ref.get("finding_id")): ref for ref in refs}
    core_refs = [ref for ref in refs if ref.get("importance") == "core"]
    # Eligibility is retained as a diagnostic field, but it cannot be used by
    # the same reviewer to remove claims from the precision denominator.  All
    # extracted claims therefore remain in the assignment population.
    eligible = list(findings)
    edges: dict[str, list[tuple[str, bool, bool]]] = {}
    for item in eligible:
        prediction_id = str(item["prediction_id"])
        candidates: list[tuple[str, bool, bool]] = []
        for match in item.get("matches", ()):
            if (
                match.get("branch_id") != branch_id
                or not match.get("semantic_match")
                or not match.get("role_match")
            ):
                continue
            gt_id = match.get("gt_finding_id")
            if gt_id is None or str(gt_id) not in ref_by_id:
                continue
            ref = ref_by_id[str(gt_id)]
            candidates.append(
                (
                    str(gt_id),
                    _host_numeric_match(item, ref),
                    bool(match.get("consistent")),
                )
            )
        edges[prediction_id] = candidates
    semantic_assignment = _maximum_matching(
        [str(item["prediction_id"]) for item in eligible], edges
    )
    valid_assignment = [item for item in semantic_assignment if item[2]]
    assigned_core = sum(
        ref_by_id[gt_id].get("importance") == "core"
        for _, gt_id, _, _ in valid_assignment
    )
    numeric_pairs = [
        item[2]
        for item in semantic_assignment
        if ref_by_id[item[1]].get("value") is not None
    ]
    consistent_pairs = [item[3] for item in valid_assignment]
    best_refs = {ref_by_id[gt_id].get("finding_id") for _, gt_id, _, _ in valid_assignment}
    return {
        "branch_id": branch_id,
        "finding_precision": len(valid_assignment) / len(eligible) if eligible else 0.0,
        "core_finding_recall": assigned_core / len(core_refs) if core_refs else None,
        "finding_requirement_recall": assigned_core / len(core_refs) if core_refs else None,
        "numeric_accuracy": sum(numeric_pairs) / len(numeric_pairs) if numeric_pairs else None,
        "c_score": sum(consistent_pairs) / len(consistent_pairs) if consistent_pairs else None,
        "valid_predicted_finding_count": len(valid_assignment),
        "matched_finding_count": len(valid_assignment),
        "matched_core_finding_count": assigned_core,
        "raw_predicted_finding_count": len(findings),
        "excluded_finding_count": sum(item.get("eligible") is not True for item in findings),
        "semantic_assignment_count": len(semantic_assignment),
        "numeric_verified_count": sum(item[2] for item in semantic_assignment),
        "matched_reference_ids": sorted(best_refs),
    }


def score_judgment(
    case_input: Mapping[str, Any],
    answer: str,
    ground_truth: Any,
    judgment: Mapping[str, Any],
    *,
    efficiency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one reviewer judgment and calculate metrics on the host."""
    gt = _reference_packet(ground_truth)
    dimension_names = _dimensions(gt)
    _validate_judgment(judgment, dimension_names)
    branch_ids = {str(branch["branch_id"]) for branch in gt["operationalizations"]}
    finding_ids_by_branch = {
        str(branch["branch_id"]): {str(ref["finding_id"]) for ref in branch["findings"]}
        for branch in gt["branches"]
    }
    dims = {
        item["dimension"]: item
        for item in judgment["operationalization"]["dimensions"]
    }
    for item in judgment["operationalization"]["dimensions"]:
        for match in item.get("branch_matches", ()):
            if match["branch_id"] not in branch_ids:
                raise ValueError(f"unknown operationalization branch_id: {match['branch_id']}")
    for finding in judgment["findings"]:
        for match in finding.get("matches", ()):
            branch_id = str(match["branch_id"])
            if branch_id not in finding_ids_by_branch:
                raise ValueError(f"unknown finding branch_id: {branch_id}")
            if (
                match.get("gt_finding_id") is not None
                and str(match["gt_finding_id"]) not in finding_ids_by_branch[branch_id]
            ):
                raise ValueError(f"unknown gt_finding_id: {match['gt_finding_id']}")

    operation_scores: list[tuple[float, str]] = []
    for branch in gt["operationalizations"]:
        required = [item["dimension"] for item in branch["decisions"]]
        matched = [
            _dimension_branch_match(
                dims.get(dim, {"status": "MISSING"}), str(branch["branch_id"])
            )
            for dim in required
        ]
        operation_scores.append(
            (
                sum(matched) / len(matched) if matched else 0.0,
                str(branch["branch_id"]),
            )
        )
    best_o = max(score for score, _ in operation_scores)
    best_o_ids = {branch_id for score, branch_id in operation_scores if score == best_o}

    unresolved = tuple(
        str(item) for item in case_input.get("unresolved_operationalization_dimensions", ())
    )
    if unresolved:
        urs_scores = []
        for branch in gt["operationalizations"]:
            matches = [
                _dimension_branch_match(
                    dims.get(dim, {"status": "MISSING"}), str(branch["branch_id"])
                )
                for dim in unresolved
            ]
            urs_scores.append(sum(matches) / len(matches))
        urs = max(urs_scores) if urs_scores else None
    else:
        urs = None

    dimension_items = [
        dims.get(
            name,
            {"dimension": name, "status": "MISSING", "evidence_present": False},
        )
        for name in dimension_names
    ]
    branch_metrics = [
        _metric_block(str(branch["branch_id"]), judgment["findings"], branch["findings"])
        for branch in gt["branches"]
    ]
    best_pair = (
        max(
            (
                item["finding_requirement_recall"]
                if item["finding_requirement_recall"] is not None
                else -1.0,
                item["finding_precision"],
            )
            for item in branch_metrics
        )
        if branch_metrics
        else None
    )
    best_finding_branches = {
        item["branch_id"]
        for item in branch_metrics
        if best_pair is not None
        and (
            item["finding_requirement_recall"]
            if item["finding_requirement_recall"] is not None
            else -1.0,
            item["finding_precision"],
        )
        == best_pair
    }
    best_finding = next(
        (item for item in branch_metrics if item["branch_id"] in best_finding_branches),
        None,
    )
    evidence_items = dimension_items + list(judgment["findings"])
    evidence_have = sum(_evidence_valid(answer, item) for item in evidence_items)
    evidence_total = len(evidence_items)
    evidence_ledger = _evidence_ledger(answer, dimension_items, judgment["findings"])
    audit_pending = (
        not judgment["branch_consistency"].get("auditable")
        or any(item.get("status") != "EXTRACTED" for item in dimension_items)
        or any(not _evidence_valid(answer, item) for item in evidence_items)
    )
    core = {
        "o_score": best_o,
        "urs": urs,
        "resolved_o_compliance": None,
        "finding_precision": best_finding["finding_precision"] if best_finding else None,
        "core_finding_recall": best_finding["core_finding_recall"] if best_finding else None,
        "finding_requirement_recall": best_finding["finding_requirement_recall"] if best_finding else None,
        "numeric_accuracy": best_finding["numeric_accuracy"] if best_finding else None,
        "c_score": best_finding["c_score"] if best_finding else None,
        "branch_alignment": (
            int(bool(best_o_ids & best_finding_branches))
            if best_finding is not None
            and best_finding["matched_finding_count"] > 0
            and best_o > 0.0
            and judgment["branch_consistency"].get("auditable")
            else None
        ),
        "evidence_coverage": evidence_have / evidence_total if evidence_total else None,
        "efficiency": dict(efficiency or {}),
    }
    # This compatibility scorer cannot award point credit from an ungrounded
    # extraction. The trusted entrypoint additionally retains item-level bounds.
    from .answer_evidence import bind_value
    invalid_dimensions = {item["dimension"] for item in dimension_items
                          if item.get("status") == "EXTRACTED" and not _evidence_valid(answer, item)}
    invalid_values = []
    for item in judgment["findings"]:
        span = item.get("evidence_span")
        quote = answer[span[0]:span[1]] if _valid_span(answer, span) else item.get("evidence_text")
        if not _evidence_valid(answer, item) or bind_value(answer, quote, item.get("value"))["status"] != "BOUND":
            invalid_values.append(item["prediction_id"])
    unknown_metrics = set()
    if invalid_dimensions:
        unknown_metrics.update(("o_score", "c_score", "branch_alignment"))
        if invalid_dimensions & set(unresolved):
            unknown_metrics.add("urs")
    if invalid_values:
        unknown_metrics.update(("finding_precision", "core_finding_recall", "finding_requirement_recall",
                                "numeric_accuracy", "c_score", "branch_alignment"))
    for name in unknown_metrics:
        core[name] = None
    audit_pending = audit_pending or bool(unknown_metrics)
    if not judgment["answer_completed"]:
        core = {
            key: (dict(efficiency or {}) if key == "efficiency" else None)
            for key in core
        }
        branch_metrics = []
        audit_pending = True
    return {
        "schema_version": CORE_SCHEMA_VERSION,
        "scorer_implementation_version": SCORER_IMPLEMENTATION_VERSION,
        "case_id": case_input.get("case_id"),
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "status": "CORE_SCORED" if judgment["answer_completed"] else "MODEL_NONCOMPLETION",
        "audit_status": "PARTIAL" if audit_pending else "COMPLETE",
        "metrics": core,
        "source_binding_failures": {"dimensions": sorted(invalid_dimensions), "findings": invalid_values},
        "metric_intervals": {name: {"lower": 0.0, "upper": 1.0, "reason": "UNBOUND_EXTRACTION"}
                             for name in sorted(unknown_metrics)},
        "branch_metrics": branch_metrics,
        "coverage": {
            "core_metrics": "SCORABLE",
            "audit": "PENDING" if audit_pending else "COMPLETE",
            "evidence_valid_n": evidence_have,
            "evidence_total_n": evidence_total,
            "raw_finding_n": len(judgment["findings"]),
            "eligible_finding_n": sum(
                item.get("eligible") is True for item in judgment["findings"]
            ),
            "evidence_ledger_sha256": _sha(evidence_ledger),
        },
        "evidence_ledger": evidence_ledger,
        "judgment_sha256": _sha(judgment),
    }


@dataclass(frozen=True)
class CoreCaseScorer:
    """One-call scorer. ``completion`` returns a decoded JSON judgment."""

    completion: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None

    def score(
        self,
        case_input: Mapping[str, Any],
        answer: str,
        ground_truth: Any,
        *,
        efficiency: Mapping[str, Any] | None = None,
        judgment: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if judgment is None:
            if self.completion is None:
                raise ValueError("completion or judgment is required")
            gt = _reference_packet(ground_truth)
            judgment = self.completion(
                scoring_prompt(case_input, answer, ground_truth),
                judgment_schema(_dimensions(gt)),
            )
        return score_judgment(case_input, answer, ground_truth, judgment, efficiency=efficiency)


__all__ = [
    "CORE_SCHEMA_VERSION",
    "SCORER_IMPLEMENTATION_VERSION",
    "CoreCaseScorer",
    "judgment_schema",
    "score_judgment",
    "scoring_prompt",
]

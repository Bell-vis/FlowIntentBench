"""Finding-responsibility sensitivity challenges for the canonical Flow Expert.

This module mutates only the already-sanitized Finding contract carried by a
scientific-review packet.  It neither defines new scientific concepts nor
assigns family-specific scientific truth.  A mutation is callable only when
the changed packet still satisfies the canonical packet contract and its exact
payload hash can be bound to the model invocation.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .scientific_expert_review import (
    validate_scientific_expert_output,
    validate_scientific_review_packet,
)
from .scientific_run_snapshots import canonical_json_sha256


RESPONSIBILITY_MUTATION_TYPES = (
    "F1_PRESENTED_AS_OPEN_FINDING_RESPONSIBILITY",
    "F2_PRESENTED_AS_FIXED_F1_ROLES",
    "REQUIRED_F1_ROLE_REMOVED",
    "IRRELEVANT_FINDING_ROLE_SUBSTITUTED",
    "F2_ADEQUATE_CORE_REQUIREMENT_REMOVED",
)

_INFRASTRUCTURE_STATUSES = frozenset(
    {
        "INFRASTRUCTURE_INVALID",
        "NOT_RUN",
        "PROVIDER_ERROR",
        "PROVIDER_TIMEOUT",
        "TOOL_ERROR",
    }
)


def _condition_id(value: Mapping[str, Any]) -> str:
    return str(value.get("condition") or value.get("condition_id") or "").strip()


def _conditions(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = packet.get("conditions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("review packet conditions must be a list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("every review packet condition must be an object")
        condition = _condition_id(item)
        if not condition:
            raise ValueError("every review packet condition requires condition")
        if condition in seen:
            raise ValueError(f"duplicate review packet condition: {condition}")
        seen.add(condition)
        rows.append(copy.deepcopy(dict(item)))
    if not rows:
        raise ValueError("review packet requires at least one condition")
    return rows


def _finding_mode(condition: Mapping[str, Any]) -> str:
    explicit = str(condition.get("finding_responsibility", "")).strip().upper()
    if explicit in {"F1", "F2"}:
        return explicit
    openness = str(condition.get("finding_openness", "")).strip().casefold()
    if openness == "bounded":
        return "F1"
    if openness == "open":
        return "F2"
    suffix = _condition_id(condition).upper()
    if suffix.endswith("F1"):
        return "F1"
    if suffix.endswith("F2"):
        return "F2"
    raise ValueError(f"condition has no recognizable Finding responsibility: {suffix}")


def _condition_index(packet: Mapping[str, Any], mode: str) -> int:
    wanted = mode.upper()
    for index, row in enumerate(_conditions(packet)):
        if _finding_mode(row) == wanted:
            return index
    raise ValueError(f"review packet has no {wanted} condition")


def _condition_by_id(packet: Mapping[str, Any], condition_id: str) -> Mapping[str, Any]:
    for row in _conditions(packet):
        if _condition_id(row) == condition_id:
            return row
    raise ValueError(f"review packet has no condition {condition_id}")


def project_finding_contract(
    packet: Mapping[str, Any], condition_id: str
) -> dict[str, Any]:
    """Project the existing model-visible Finding contract for one condition."""

    condition = _condition_by_id(packet, condition_id)
    return {
        "condition": _condition_id(condition),
        "case_id": str(condition.get("case_id", "")),
        "question": str(condition.get("question", "")),
        "finding_goal": str(condition.get("finding_goal", "")),
        "finding_responsibility": str(condition.get("finding_responsibility", "")),
        "finding_openness": str(condition.get("finding_openness", "")),
        "finding_requirements": copy.deepcopy(
            list(condition.get("finding_requirements", ()) or ())
        ),
    }


def recursive_json_diff(
    before: Any, after: Any, *, path: str = "$"
) -> list[dict[str, Any]]:
    """Return a deterministic, value-preserving recursive JSON diff."""

    if isinstance(before, Mapping) and isinstance(after, Mapping):
        rows: list[dict[str, Any]] = []
        keys = sorted(set(before) | set(after), key=str)
        for key in keys:
            child = f"{path}.{key}"
            if key not in before:
                rows.append(
                    {
                        "path": child,
                        "change": "ADDED",
                        "before": None,
                        "after": copy.deepcopy(after[key]),
                    }
                )
            elif key not in after:
                rows.append(
                    {
                        "path": child,
                        "change": "REMOVED",
                        "before": copy.deepcopy(before[key]),
                        "after": None,
                    }
                )
            else:
                rows.extend(recursive_json_diff(before[key], after[key], path=child))
        return rows
    if (
        isinstance(before, Sequence)
        and not isinstance(before, (str, bytes, bytearray))
        and isinstance(after, Sequence)
        and not isinstance(after, (str, bytes, bytearray))
    ):
        rows = []
        common = min(len(before), len(after))
        for index in range(common):
            rows.extend(
                recursive_json_diff(before[index], after[index], path=f"{path}[{index}]")
            )
        for index in range(common, len(before)):
            rows.append(
                {
                    "path": f"{path}[{index}]",
                    "change": "REMOVED",
                    "before": copy.deepcopy(before[index]),
                    "after": None,
                }
            )
        for index in range(common, len(after)):
            rows.append(
                {
                    "path": f"{path}[{index}]",
                    "change": "ADDED",
                    "before": None,
                    "after": copy.deepcopy(after[index]),
                }
            )
        return rows
    if before != after or type(before) is not type(after):
        return [
            {
                "path": path,
                "change": "CHANGED",
                "before": copy.deepcopy(before),
                "after": copy.deepcopy(after),
            }
        ]
    return []


def _plain_f1_requirement_indices(condition: Mapping[str, Any]) -> list[int]:
    raw = condition.get("finding_requirements")
    if not isinstance(raw, list):
        raise ValueError("Finding requirements must be a list")
    return [
        index
        for index, item in enumerate(raw)
        if isinstance(item, Mapping)
        and item.get("requirement_type") != "FAMILY_ROLE_CONTRACT"
    ]


def _family_role_requirement_indices(condition: Mapping[str, Any]) -> list[int]:
    raw = condition.get("finding_requirements")
    if not isinstance(raw, list):
        raise ValueError("Finding requirements must be a list")
    return [
        index
        for index, item in enumerate(raw)
        if isinstance(item, Mapping)
        and item.get("requirement_type") == "FAMILY_ROLE_CONTRACT"
    ]


def _mutate_f1_as_open(packet: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    rows = _conditions(packet)
    index = _condition_index(packet, "F1")
    row = rows[index]
    row["finding_responsibility"] = "F2"
    row["finding_openness"] = "open"
    packet["conditions"] = rows
    prefix = f"$.conditions[{index}]"
    return _condition_id(row), (
        f"{prefix}.finding_responsibility",
        f"{prefix}.finding_openness",
    )


def _mutate_f2_as_fixed(packet: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    rows = _conditions(packet)
    index = _condition_index(packet, "F2")
    row = rows[index]
    row["finding_responsibility"] = "F1"
    row["finding_openness"] = "bounded"
    packet["conditions"] = rows
    prefix = f"$.conditions[{index}]"
    return _condition_id(row), (
        f"{prefix}.finding_responsibility",
        f"{prefix}.finding_openness",
    )


def _mutate_remove_f1_role(packet: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    rows = _conditions(packet)
    index = _condition_index(packet, "F1")
    row = rows[index]
    requirement_indices = _plain_f1_requirement_indices(row)
    if not requirement_indices:
        raise ValueError("F1 role-removal mutation requires an explicit Finding role")
    remove_index = requirement_indices[-1]
    requirements = copy.deepcopy(list(row["finding_requirements"]))
    requirements.pop(remove_index)
    row["finding_requirements"] = requirements
    packet["conditions"] = rows
    return _condition_id(row), (
        f"$.conditions[{index}].finding_requirements[{remove_index}]",
    )


def _mutate_substitute_irrelevant_role(
    packet: dict[str, Any],
) -> tuple[str, tuple[str, ...]]:
    rows = _conditions(packet)
    index = _condition_index(packet, "F1")
    row = rows[index]
    requirement_indices = _plain_f1_requirement_indices(row)
    if not requirement_indices:
        raise ValueError("irrelevant-role mutation requires an explicit F1 Finding role")
    replace_index = requirement_indices[-1]
    requirements = copy.deepcopy(list(row["finding_requirements"]))
    replacement = dict(requirements[replace_index])
    replacement["statement"] = (
        "report the implementation wall-clock duration as a scientific Finding"
    )
    replacement["question_fragment"] = "report wall-clock duration"
    requirements[replace_index] = replacement
    row["finding_requirements"] = requirements
    packet["conditions"] = rows
    prefix = f"$.conditions[{index}].finding_requirements[{replace_index}]"
    return _condition_id(row), (
        f"{prefix}.statement",
        f"{prefix}.question_fragment",
    )


def _mutate_remove_f2_adequate_core(
    packet: dict[str, Any],
) -> tuple[str, tuple[str, ...]]:
    rows = _conditions(packet)
    index = _condition_index(packet, "F2")
    row = rows[index]
    requirement_indices = _family_role_requirement_indices(row)
    if not requirement_indices:
        raise ValueError(
            "F2 adequate-core mutation requires a visible FAMILY_ROLE_CONTRACT"
        )
    requirements = copy.deepcopy(list(row["finding_requirements"]))
    for remove_index in reversed(requirement_indices):
        requirements.pop(remove_index)
    row["finding_requirements"] = requirements
    packet["conditions"] = rows
    return _condition_id(row), tuple(
        f"$.conditions[{index}].finding_requirements[{remove_index}]"
        for remove_index in requirement_indices
    )


_MUTATORS = {
    "F1_PRESENTED_AS_OPEN_FINDING_RESPONSIBILITY": _mutate_f1_as_open,
    "F2_PRESENTED_AS_FIXED_F1_ROLES": _mutate_f2_as_fixed,
    "REQUIRED_F1_ROLE_REMOVED": _mutate_remove_f1_role,
    "IRRELEVANT_FINDING_ROLE_SUBSTITUTED": _mutate_substitute_irrelevant_role,
    "F2_ADEQUATE_CORE_REQUIREMENT_REMOVED": _mutate_remove_f2_adequate_core,
}


def _diff_verification(
    source: Mapping[str, Any],
    mutated: Mapping[str, Any],
    *,
    expected_paths: Sequence[str],
) -> dict[str, Any]:
    diff = recursive_json_diff(source, mutated)
    changed_paths = [str(item["path"]) for item in diff]
    expected = [str(item) for item in expected_paths]

    def covered(expected_path: str) -> bool:
        return any(
            path == expected_path
            or path.startswith(f"{expected_path}.")
            or path.startswith(f"{expected_path}[")
            for path in changed_paths
        )

    def allowed(path: str) -> bool:
        return any(
            path == expected_path
            or path.startswith(f"{expected_path}.")
            or path.startswith(f"{expected_path}[")
            for expected_path in expected
        )

    missing = [path for path in expected if not covered(path)]
    unexpected = [path for path in changed_paths if not allowed(path)]
    return {
        "status": "PASS" if diff and not missing and not unexpected else "FAIL",
        "changed_paths": changed_paths,
        "expected_changed_paths": expected,
        "missing_expected_paths": missing,
        "unexpected_changed_paths": unexpected,
        "diff": diff,
    }


def _validated_source(packet: Mapping[str, Any], label: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(packet))
    validation = validate_scientific_review_packet(value)
    if validation.get("status") != "PASS":
        raise ValueError(
            f"{label} is not a valid canonical scientific-review packet: "
            + ", ".join(validation.get("reason_codes", ()))
        )
    return value


def build_responsibility_mutation_suite(
    f1_packet: Mapping[str, Any],
    *,
    f2_packet: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build five generic F1/F2 challenges from validated canonical packets.

    ``f2_packet`` may be omitted when ``f1_packet`` contains both an F1 and an
    F2 condition.  Scientific evidence, O contracts, and materialization views
    are never changed by these mutations.
    """

    f1_source = _validated_source(f1_packet, "f1_packet")
    _condition_index(f1_source, "F1")
    f2_source = (
        f1_source
        if f2_packet is None
        else _validated_source(f2_packet, "f2_packet")
    )
    _condition_index(f2_source, "F2")
    source_by_type = {
        "F1_PRESENTED_AS_OPEN_FINDING_RESPONSIBILITY": f1_source,
        "F2_PRESENTED_AS_FIXED_F1_ROLES": f2_source,
        "REQUIRED_F1_ROLE_REMOVED": f1_source,
        "IRRELEVANT_FINDING_ROLE_SUBSTITUTED": f1_source,
        "F2_ADEQUATE_CORE_REQUIREMENT_REMOVED": f2_source,
    }
    mutations: list[dict[str, Any]] = []
    for index, mutation_type in enumerate(RESPONSIBILITY_MUTATION_TYPES, start=1):
        source = source_by_type[mutation_type]
        packet = copy.deepcopy(source)
        condition, expected_paths = _MUTATORS[mutation_type](packet)
        original_contract = project_finding_contract(source, condition)
        mutated_contract = project_finding_contract(packet, condition)
        packet_validation = validate_scientific_review_packet(packet)
        packet_diff = _diff_verification(
            source, packet, expected_paths=expected_paths
        )
        reaches_packet = (
            packet_validation.get("status") == "PASS"
            and packet_diff["status"] == "PASS"
            and original_contract != mutated_contract
        )
        if not reaches_packet:
            raise ValueError(
                f"responsibility mutation {mutation_type} did not reach a valid canonical packet"
            )
        mutations.append(
            {
                "mutation_id": f"responsibility-mutation-{index:02d}",
                "mutation_type": mutation_type,
                "condition": condition,
                "expected_contract_check": "FINDING_RESPONSIBILITY_DRIFT_DETECTED",
                "expected_atomic_signal": {
                    "finding_contract_supported": False
                },
                "source_packet_sha256": canonical_json_sha256(source),
                "mutated_packet_sha256": canonical_json_sha256(packet),
                "original_finding_contract": original_contract,
                "mutated_finding_contract": mutated_contract,
                "packet_diff_verification": packet_diff,
                "canonical_packet_validation": packet_validation,
                "mutation_reaches_validated_canonical_packet": True,
                "scientific_truth_assigned": False,
                "packet": packet,
            }
        )
    return {
        "suite_version": "flow-expert-responsibility-sensitivity-v1",
        "mutation_count": len(mutations),
        "mutation_types": list(RESPONSIBILITY_MUTATION_TYPES),
        "mutations": mutations,
        "scientific_truth_assigned": False,
    }


def validate_responsibility_mutation_delivery(
    mutation: Mapping[str, Any], submitted_packet: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove that the exact mutated contract reaches a callable packet."""

    errors: list[str] = []
    expected_hash = str(mutation.get("mutated_packet_sha256", ""))
    actual_hash = canonical_json_sha256(submitted_packet)
    if not expected_hash or actual_hash != expected_hash:
        errors.append("MUTATED_PACKET_HASH_MISMATCH")
    validation = validate_scientific_review_packet(submitted_packet)
    if validation.get("status") != "PASS":
        errors.append("CANONICAL_PACKET_INVALID")
    condition = str(mutation.get("condition", ""))
    try:
        delivered_contract = project_finding_contract(submitted_packet, condition)
    except ValueError:
        delivered_contract = None
        errors.append("MUTATED_CONDITION_MISSING")
    if delivered_contract != mutation.get("mutated_finding_contract"):
        errors.append("MUTATED_FINDING_CONTRACT_MISMATCH")
    diff_verification = mutation.get("packet_diff_verification")
    if not isinstance(diff_verification, Mapping) or diff_verification.get("status") != "PASS":
        errors.append("SOURCE_TO_MUTATED_PACKET_DIFF_UNVERIFIED")
    return {
        "status": "PASS" if not errors else "FAIL",
        "mutation_id": mutation.get("mutation_id"),
        "mutated_packet_sha256": actual_hash,
        "canonical_packet_validation_status": validation.get("status"),
        "mutation_reaches_validated_canonical_packet": not errors,
        "errors": errors,
    }


def _attempts_by_id(
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("every responsibility review attempt must be an object")
        mutation_id = str(attempt.get("mutation_id", "")).strip()
        if not mutation_id:
            raise ValueError("every responsibility review attempt requires mutation_id")
        if mutation_id in result:
            raise ValueError(f"duplicate responsibility review attempt: {mutation_id}")
        result[mutation_id] = attempt
    return result


def audit_responsibility_mutation_reviews(
    mutation_suite: Mapping[str, Any],
    review_attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit F-responsibility sensitivity without majority vote or case truth."""

    raw_mutations = mutation_suite.get("mutations")
    if not isinstance(raw_mutations, Sequence) or isinstance(
        raw_mutations, (str, bytes, bytearray)
    ):
        raise ValueError("responsibility mutation suite requires mutations")
    attempts = _attempts_by_id(review_attempts)
    rows: list[dict[str, Any]] = []
    for raw in raw_mutations:
        if not isinstance(raw, Mapping):
            raise ValueError("every responsibility mutation must be an object")
        mutation_id = str(raw.get("mutation_id", "")).strip()
        mutation_type = str(raw.get("mutation_type", "")).strip()
        if mutation_type not in RESPONSIBILITY_MUTATION_TYPES:
            raise ValueError(f"unknown responsibility mutation type: {mutation_type}")
        base = {
            "mutation_id": mutation_id,
            "mutation_type": mutation_type,
            "packet_diff_verification": str(
                (raw.get("packet_diff_verification") or {}).get("status", "FAIL")
            ),
        }
        attempt = attempts.get(mutation_id)
        if attempt is None:
            rows.append(
                {
                    **base,
                    "status": "NOT_RUN",
                    "detected": None,
                    "finding_contract_supported": None,
                }
            )
            continue
        invocation_status = str(
            attempt.get("invocation_status") or attempt.get("status") or ""
        ).upper()
        if invocation_status in _INFRASTRUCTURE_STATUSES:
            rows.append(
                {
                    **base,
                    "status": "INFRASTRUCTURE_INVALID",
                    "detected": None,
                    "finding_contract_supported": None,
                }
            )
            continue
        submitted_hash = str(
            attempt.get("submitted_packet_sha256")
            or attempt.get("packet_sha256")
            or ""
        )
        if submitted_hash != str(raw.get("mutated_packet_sha256", "")):
            rows.append(
                {
                    **base,
                    "status": "PACKET_DELIVERY_UNVERIFIED",
                    "detected": False,
                    "finding_contract_supported": None,
                }
            )
            continue
        review = attempt.get("review")
        if not isinstance(review, Mapping):
            review = attempt.get("parsed_review")
        packet = raw.get("packet")
        if not isinstance(review, Mapping) or not isinstance(packet, Mapping):
            rows.append(
                {
                    **base,
                    "status": "REVIEW_INCOMPLETE",
                    "detected": False,
                    "finding_contract_supported": None,
                }
            )
            continue
        output_validation = validate_scientific_expert_output(review, packet)
        if output_validation.get("status") != "PASS":
            rows.append(
                {
                    **base,
                    "status": "REVIEW_INCOMPLETE",
                    "detected": False,
                    "finding_contract_supported": None,
                    "reason_codes": list(output_validation.get("reason_codes", ())),
                }
            )
            continue
        condition = str(raw.get("condition", ""))
        reviewed = next(
            (
                row
                for row in output_validation.get("condition_reviews", ())
                if isinstance(row, Mapping) and _condition_id(row) == condition
            ),
            None,
        )
        if reviewed is None:
            rows.append(
                {
                    **base,
                    "status": "REVIEW_INCOMPLETE",
                    "detected": False,
                    "finding_contract_supported": None,
                    "reason_codes": ["MUTATED_CONDITION_REVIEW_MISSING"],
                }
            )
            continue
        signal = reviewed.get("finding_contract_supported")
        detected = signal is False
        rows.append(
            {
                **base,
                "status": "DETECTED" if detected else "NOT_DETECTED",
                "detected": detected,
                "finding_contract_supported": signal,
                "eligibility_recommendation": reviewed.get(
                    "eligibility_recommendation"
                ),
            }
        )
    evaluable = [row for row in rows if isinstance(row["detected"], bool)]
    any_attempt = any(row["status"] != "NOT_RUN" for row in rows)
    all_detected = (
        len(evaluable) == len(RESPONSIBILITY_MUTATION_TYPES)
        and all(row["detected"] is True for row in evaluable)
    )
    remaining = [
        row["mutation_type"]
        for row in rows
        if row["status"] in {
            "NOT_DETECTED",
            "PACKET_DELIVERY_UNVERIFIED",
            "REVIEW_INCOMPLETE",
        }
    ]
    return {
        "status": (
            "PASS"
            if all_detected
            else "NOT_RUN"
            if not any_attempt
            else "PARTIAL"
        ),
        "mutation_count": len(rows),
        "detected_count": sum(row["detected"] is True for row in rows),
        "not_detected_count": sum(row["status"] == "NOT_DETECTED" for row in rows),
        "infrastructure_invalid_count": sum(
            row["status"] == "INFRASTRUCTURE_INVALID" for row in rows
        ),
        "packet_delivery_unverified_count": sum(
            row["status"] == "PACKET_DELIVERY_UNVERIFIED" for row in rows
        ),
        "review_incomplete_count": sum(
            row["status"] == "REVIEW_INCOMPLETE" for row in rows
        ),
        "packet_diff_verified_count": sum(
            row["packet_diff_verification"] == "PASS" for row in rows
        ),
        "remaining_ambiguity": remaining,
        "rows": rows,
        "scientific_truth_assigned": False,
        "majority_vote_used": False,
    }


__all__ = [
    "RESPONSIBILITY_MUTATION_TYPES",
    "audit_responsibility_mutation_reviews",
    "build_responsibility_mutation_suite",
    "project_finding_contract",
    "recursive_json_diff",
    "validate_responsibility_mutation_delivery",
]

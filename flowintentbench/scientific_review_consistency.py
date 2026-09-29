"""Cross-condition consistency checks for atomic scientific observations."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


FINDING_CONTRACT_INCONSISTENCY = "CROSS_CONDITION_FINDING_CONTRACT_INCONSISTENCY"
SCIENTIFIC_TARGET_INCONSISTENCY = "CROSS_CONDITION_SCIENTIFIC_TARGET_INCONSISTENCY"
SCIENTIFIC_REVIEW_INCONSISTENT = "SCIENTIFIC_REVIEW_INCONSISTENT"


def _contract_fingerprint(contract: Mapping[str, Any]) -> str:
    value = contract.get(
        "finding_requirement_contract",
        contract.get("finding_contract", contract.get("finding_requirements")),
    )
    if value is None:
        return "__UNSPECIFIED_F1_CONTRACT__"
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _condition_specific_finding_rationale(review: Mapping[str, Any]) -> str:
    for key in (
        "finding_contract_condition_specific_rationale",
        "condition_specific_scientific_rationale",
    ):
        value = review.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            for nested_key in ("finding_contract_supported", "finding_contract", "F1"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return ""


def audit_cross_condition_scientific_consistency(
    condition_reviews: Sequence[Mapping[str, Any]],
    condition_contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit shared target support and identical canonical F1 contracts.

    An explicit condition-specific rationale can explain an F1 verdict that
    differs from O1-F1 (or the first condition when O1-F1 is absent). The
    audit records the explanation but never selects a scientifically correct
    verdict. Family-level target-support disagreements cannot be resolved by
    condition-specific rationale.
    """

    reviews = {
        str(item.get("condition")): item
        for item in condition_reviews
        if isinstance(item, Mapping) and item.get("condition") is not None
    }
    failure_codes: list[str] = []
    invalidated: dict[str, set[str]] = {}

    target_verdicts = {
        condition: review.get("scientific_target_supported")
        for condition, review in reviews.items()
        if isinstance(review.get("scientific_target_supported"), bool)
    }
    canonical_targets = {
        condition: str(condition_contracts.get(condition, {}).get("scientific_target", "")).strip()
        for condition in reviews
        if str(condition_contracts.get(condition, {}).get("scientific_target", "")).strip()
    }
    target_inconsistent = (
        len(set(target_verdicts.values())) > 1
        or len({target.casefold() for target in canonical_targets.values()}) > 1
    )
    if target_inconsistent:
        failure_codes.append(SCIENTIFIC_TARGET_INCONSISTENCY)
        for condition in reviews:
            invalidated.setdefault(condition, set()).add("scientific_target_supported")

    f1_groups: dict[str, list[str]] = {}
    for condition in reviews:
        if not condition.endswith("-F1"):
            continue
        contract = condition_contracts.get(condition, {})
        f1_groups.setdefault(_contract_fingerprint(contract), []).append(condition)

    finding_groups: list[dict[str, Any]] = []
    for fingerprint, conditions in sorted(f1_groups.items()):
        conditions = sorted(conditions)
        verdicts = {
            condition: reviews[condition].get("finding_contract_supported")
            for condition in conditions
            if isinstance(reviews[condition].get("finding_contract_supported"), bool)
        }
        rationales = {
            condition: _condition_specific_finding_rationale(reviews[condition])
            for condition in conditions
        }
        incompatible = len(set(verdicts.values())) > 1
        unexplained: list[str] = []
        if incompatible:
            baseline = "O1-F1" if "O1-F1" in verdicts else sorted(verdicts)[0]
            baseline_verdict = verdicts[baseline]
            unexplained = sorted(
                condition
                for condition, verdict in verdicts.items()
                if verdict != baseline_verdict and not rationales.get(condition)
            )
            if unexplained:
                failure_codes.append(FINDING_CONTRACT_INCONSISTENCY)
                for condition in conditions:
                    invalidated.setdefault(condition, set()).add(
                        "finding_contract_supported"
                    )
        finding_groups.append(
            {
                "contract_fingerprint": fingerprint,
                "conditions": conditions,
                "verdicts": verdicts,
                "condition_specific_rationales": {
                    key: value for key, value in rationales.items() if value
                },
                "status": (
                    "INCONSISTENT"
                    if unexplained
                    else "EXPLAINED_BY_CONDITION_SPECIFIC_RATIONALE"
                    if incompatible
                    else "CONSISTENT"
                ),
                "unexplained_conditions": unexplained,
            }
        )

    failure_codes = sorted(set(failure_codes))
    invalidated_rows = {
        condition: sorted(fields) for condition, fields in sorted(invalidated.items())
    }
    return {
        "status": "FAIL" if failure_codes else "PASS",
        "scientific_review_status": (
            SCIENTIFIC_REVIEW_INCONSISTENT if failure_codes else "PASS"
        ),
        "failure_codes": failure_codes,
        "scientific_target_support": {
            "status": "INCONSISTENT" if target_inconsistent else "CONSISTENT",
            "verdicts": target_verdicts,
            "canonical_targets": canonical_targets,
        },
        "finding_contract_groups": finding_groups,
        "invalidated_observations_by_condition": invalidated_rows,
        "inconsistent_conditions": sorted(invalidated_rows),
        "router_auto_resolution": False,
    }


def apply_scientific_consistency_guard(
    observations: Mapping[str, Any],
    condition: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Make unresolved shared-observation conflicts unavailable to policy."""

    guarded = dict(observations)
    invalidated = audit.get("invalidated_observations_by_condition", {})
    fields = invalidated.get(condition, ()) if isinstance(invalidated, Mapping) else ()
    for field in fields:
        guarded[str(field)] = None
    if fields:
        guarded["scientific_review_consistent"] = False
    return guarded


__all__ = [
    "FINDING_CONTRACT_INCONSISTENCY",
    "SCIENTIFIC_TARGET_INCONSISTENCY",
    "SCIENTIFIC_REVIEW_INCONSISTENT",
    "apply_scientific_consistency_guard",
    "audit_cross_condition_scientific_consistency",
]

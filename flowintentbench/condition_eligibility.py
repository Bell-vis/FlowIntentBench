"""Deterministic policy for O1/O2/O3 and F1/F2 condition eligibility.

Flow experts supply atomic scientific observations.  This module applies the
already-frozen condition definitions and never treats a free-form expert
recommendation as the policy result.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


CONDITIONS = frozenset({"O1-F1", "O2-F1", "O3-F1", "O1-F2"})
ELIGIBILITY_RESULTS = frozenset({"ELIGIBLE", "NOT_ELIGIBLE", "MORE_EVIDENCE_REQUIRED", "REVISE_QUESTION"})

REASON_CODES = frozenset({
    "TARGET_UNSUPPORTED",
    "TARGET_DRIFT",
    "FIXED_O_UNSUPPORTED",
    "FIXED_O_QUESTION_DRIFT",
    "UNRESOLVED_O_NOT_LEGITIMATE",
    "UNRESOLVED_O_ACCIDENTALLY_FIXED",
    "F1_CONTRACT_UNSUPPORTED",
    "F2_CONTRACT_UNSUPPORTED",
    "G_OF_O_NOT_EXECUTABLE",
    "MATERIALIZATION_SCIENTIFIC_MEANING_UNSUPPORTED",
    "SCIENTIFIC_REVIEW_INCONSISTENT",
    "VALID_UNENUMERATED_ROUTE_MISSING",
    "O1_PRINCIPAL_O_NOT_FIXED",
    "O2_NO_UNRESOLVED_O",
    "O3_PRINCIPAL_O_NOT_OPEN",
    "SCIENTIFIC_SCOPE_DRIFT",
    "O1_INHERITANCE_MISMATCH",
    "SCIENTIFIC_OBSERVATION_MISSING",
})

_REVISE_CODES = frozenset({
    "TARGET_DRIFT",
    "FIXED_O_QUESTION_DRIFT",
    "UNRESOLVED_O_ACCIDENTALLY_FIXED",
    "SCIENTIFIC_SCOPE_DRIFT",
})
_MORE_EVIDENCE_CODES = frozenset({"SCIENTIFIC_REVIEW_INCONSISTENT"})
_NOT_ELIGIBLE_CODES = (
    REASON_CODES
    - _REVISE_CODES
    - _MORE_EVIDENCE_CODES
    - {"SCIENTIFIC_OBSERVATION_MISSING"}
)
_MISSING = object()


def _normalize_text(value: Any) -> str:
    text = str(value or "").casefold().replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def audit_target_fidelity(
    question: str,
    canonical_target: str,
    *,
    required_target_anchors: Sequence[str],
) -> dict[str, Any]:
    """Check explicit target anchors without inventing scientific synonyms."""

    normalized_question = _normalize_text(question)
    anchors = [str(item) for item in required_target_anchors if str(item).strip()]
    missing = [anchor for anchor in anchors if _normalize_text(anchor) not in normalized_question]
    return {
        "status": "PASS" if not missing else "FAIL",
        "canonical_target": str(canonical_target),
        "required_target_anchors": anchors,
        "missing_target_anchors": missing,
        "reason_codes": [] if not missing else ["TARGET_DRIFT"],
    }


def _observation(value: Mapping[str, Any], key: str) -> bool | None:
    observed = value.get(key, _MISSING)
    return observed if isinstance(observed, bool) else None


def _dimension_observation(
    value: Mapping[str, Any],
    key: str,
    dimensions: Sequence[str],
) -> tuple[bool | None, list[str]]:
    """Resolve an atomic all-dimension observation and failing dimension IDs."""

    observed = value.get(key, _MISSING)
    dimension_ids = [str(item) for item in dimensions]
    if not dimension_ids:
        return True, []
    if isinstance(observed, bool):
        return observed, [] if observed else dimension_ids
    if isinstance(observed, Mapping):
        missing = [dimension for dimension in dimension_ids if not isinstance(observed.get(dimension), bool)]
        if missing:
            return None, missing
        unsupported = [dimension for dimension in dimension_ids if observed.get(dimension) is False]
        return not unsupported, unsupported
    return None, dimension_ids


def _status(reason_codes: Sequence[str], missing_observations: Sequence[str]) -> str:
    if any(code in _NOT_ELIGIBLE_CODES for code in reason_codes):
        return "NOT_ELIGIBLE"
    if any(code in _REVISE_CODES for code in reason_codes):
        return "REVISE_QUESTION"
    if missing_observations or any(code in _MORE_EVIDENCE_CODES for code in reason_codes):
        return "MORE_EVIDENCE_REQUIRED"
    if reason_codes:
        return "NOT_ELIGIBLE"
    return "ELIGIBLE"


def evaluate_condition_eligibility(
    condition: str,
    case_contract: Mapping[str, Any],
    expert_observations: Mapping[str, Any],
    execution_facts: Mapping[str, Any],
    *,
    o1_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen condition predicates to explicit scientific facts.

    ``expert_observations`` contains atomic scientific judgments only.  Its
    optional ``eligibility_recommendation`` is persisted for audit, but is not
    consulted when deriving ``eligibility``.
    """

    normalized_condition = str(condition).upper()
    if normalized_condition not in CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")

    principal = [str(item) for item in case_contract.get("principal_dimensions", ())]
    unresolved = [str(item) for item in case_contract.get("unresolved_dimensions", ())]
    fixed = [str(item) for item in case_contract.get("fixed_dimensions", ())]
    if not fixed and principal:
        fixed = [item for item in principal if item not in set(unresolved)]

    reason_codes: list[str] = []
    missing_observations: list[str] = []
    details: dict[str, Any] = {}

    target_supported = _observation(expert_observations, "scientific_target_supported")
    if target_supported is None:
        missing_observations.append("scientific_target_supported")
    elif not target_supported:
        reason_codes.append("TARGET_UNSUPPORTED")

    fidelity = audit_target_fidelity(
        str(case_contract.get("question", "")),
        str(case_contract.get("scientific_target", "")),
        required_target_anchors=tuple(case_contract.get("required_target_anchors", ())),
    )
    question_target_preserved = _observation(expert_observations, "question_target_preserved")
    if fidelity["status"] != "PASS" or question_target_preserved is False:
        reason_codes.append("TARGET_DRIFT")
    elif question_target_preserved is None and not fidelity["required_target_anchors"]:
        missing_observations.append("question_target_preserved")

    finding_supported = _observation(expert_observations, "finding_contract_supported")
    finding_code = "F2_CONTRACT_UNSUPPORTED" if normalized_condition.endswith("F2") else "F1_CONTRACT_UNSUPPORTED"
    if finding_supported is None:
        missing_observations.append("finding_contract_supported")
    elif not finding_supported:
        reason_codes.append(finding_code)

    g_executable = execution_facts.get("g_of_o_executable", _MISSING)
    execution_materialization_status = execution_facts.get(
        "execution_materialization_status", _MISSING
    )
    if not isinstance(execution_materialization_status, str) or not execution_materialization_status:
        execution_materialization_status = (
            "MATERIALIZED"
            if g_executable is True
            else "MATERIALIZATION_UNSUPPORTED"
            if g_executable is False
            else None
        )
    meaningful = _observation(expert_observations, "materialization_scientifically_meaningful")
    if not isinstance(g_executable, bool):
        missing_observations.append("g_of_o_executable")
    elif not g_executable:
        reason_codes.append("G_OF_O_NOT_EXECUTABLE")
    elif meaningful is None:
        missing_observations.append("materialization_scientifically_meaningful")
    elif not meaningful:
        reason_codes.append("MATERIALIZATION_SCIENTIFIC_MEANING_UNSUPPORTED")

    scientific_review_consistent = expert_observations.get(
        "scientific_review_consistent", True
    )
    if scientific_review_consistent is False:
        reason_codes.append("SCIENTIFIC_REVIEW_INCONSISTENT")

    fixed_supported, unsupported_fixed = _dimension_observation(
        expert_observations, "fixed_o_supported", fixed
    )
    # O1-F2 changes only Finding responsibility.  Its fixed-O support is
    # inherited from the deterministic O1-F1 result, never re-decided by a
    # second advisory review.
    if normalized_condition == "O1-F2" and isinstance(o1_result, Mapping):
        inherited_support = o1_result.get("fixed_o_support", _MISSING)
        if isinstance(inherited_support, bool):
            fixed_supported = inherited_support
            unsupported_fixed = list(fixed) if not inherited_support else []
    details["unsupported_fixed_dimensions"] = unsupported_fixed
    if fixed_supported is None:
        missing_observations.append("fixed_o_supported")
    elif not fixed_supported:
        reason_codes.append("FIXED_O_UNSUPPORTED")

    if normalized_condition in {"O1-F1", "O1-F2"}:
        if set(fixed) != set(principal) or unresolved:
            reason_codes.append("O1_PRINCIPAL_O_NOT_FIXED")
        wording = _observation(expert_observations, "fixed_o_question_faithful")
        if wording is None:
            missing_observations.append("fixed_o_question_faithful")
        elif not wording:
            reason_codes.append("FIXED_O_QUESTION_DRIFT")

    if normalized_condition == "O2-F1":
        if not unresolved:
            reason_codes.append("O2_NO_UNRESOLVED_O")
        legitimate, illegitimate = _dimension_observation(
            expert_observations,
            "unresolved_o_is_legitimate_scientific_choice",
            unresolved,
        )
        details["illegitimate_unresolved_dimensions"] = illegitimate
        if legitimate is None:
            missing_observations.append("unresolved_o_is_legitimate_scientific_choice")
        elif not legitimate:
            reason_codes.append("UNRESOLVED_O_NOT_LEGITIMATE")
        exposed = _observation(expert_observations, "unresolved_o_exposed_by_question")
        if exposed is None:
            missing_observations.append("unresolved_o_exposed_by_question")
        elif not exposed:
            reason_codes.append("UNRESOLVED_O_ACCIDENTALLY_FIXED")
        anchor_count = execution_facts.get("executable_authored_o_anchor_count", _MISSING)
        if not isinstance(anchor_count, int):
            missing_observations.append("executable_authored_o_anchor_count")
        elif anchor_count < 1:
            reason_codes.append("G_OF_O_NOT_EXECUTABLE")
        route = execution_facts.get("valid_unenumerated_route_exists", _MISSING)
        if not isinstance(route, bool):
            missing_observations.append("valid_unenumerated_route_exists")
        elif not route:
            reason_codes.append("VALID_UNENUMERATED_ROUTE_MISSING")

    if normalized_condition == "O3-F1":
        if set(unresolved) != set(principal) or not principal:
            reason_codes.append("O3_PRINCIPAL_O_NOT_OPEN")
        scope = _observation(expert_observations, "question_scope_preserved")
        if scope is None:
            missing_observations.append("question_scope_preserved")
        elif not scope:
            reason_codes.append("SCIENTIFIC_SCOPE_DRIFT")
        route = execution_facts.get("valid_unenumerated_route_exists", _MISSING)
        if not isinstance(route, bool):
            missing_observations.append("valid_unenumerated_route_exists")
        elif not route:
            reason_codes.append("VALID_UNENUMERATED_ROUTE_MISSING")

    if normalized_condition == "O1-F2":
        inherited_equal = case_contract.get("inherits_o1_fixed_o_exactly", _MISSING)
        if not isinstance(inherited_equal, bool):
            missing_observations.append("inherits_o1_fixed_o_exactly")
        elif not inherited_equal:
            reason_codes.append("O1_INHERITANCE_MISMATCH")
        if o1_result is None:
            missing_observations.append("o1_result")
        else:
            o1_reasons = set(str(item) for item in o1_result.get("reason_codes", ()))
            if "FIXED_O_UNSUPPORTED" in o1_reasons or o1_result.get("fixed_o_support") is False:
                reason_codes.append("FIXED_O_UNSUPPORTED")

    reason_codes = sorted(set(reason_codes))
    missing_observations = sorted(set(missing_observations))
    if missing_observations:
        reason_codes = sorted(set(reason_codes) | {"SCIENTIFIC_OBSERVATION_MISSING"})
    eligibility = _status(reason_codes, missing_observations)
    advisory = str(expert_observations.get("eligibility_recommendation", "")).upper() or None
    distinct_results = execution_facts.get("distinct_valid_g_of_o_count")

    return {
        "condition": normalized_condition,
        "eligibility": eligibility,
        "reason_codes": reason_codes,
        "missing_observations": missing_observations,
        "target_fidelity": fidelity,
        "fixed_o_support": fixed_supported,
        "unresolved_o_legitimacy": (
            legitimate if normalized_condition == "O2-F1" else None
        ),
        "finding_contract_support": finding_supported,
        "materialization_support": g_executable is True and meaningful is True,
        "execution_materialization_status": execution_materialization_status,
        "scientific_materialization_support": meaningful,
        "expert_recommendation_advisory": advisory,
        "expert_recommendation_used_as_policy": False,
        "advisory_recommendation_overridden": advisory is not None and advisory != eligibility,
        "non_unique_valid_o_outcomes_allowed": normalized_condition == "O2-F1",
        "non_unique_valid_o_outcomes_observed": (
            normalized_condition == "O2-F1"
            and isinstance(distinct_results, int)
            and distinct_results > 1
        ),
        "non_unique_valid_o_outcomes_used_as_failure": False,
        "predicate_details": details,
        "scientific_ambiguities": [
            str(item) for item in expert_observations.get("scientific_ambiguities", ())
        ],
    }


__all__ = [
    "CONDITIONS",
    "ELIGIBILITY_RESULTS",
    "REASON_CODES",
    "audit_target_fidelity",
    "evaluate_condition_eligibility",
]

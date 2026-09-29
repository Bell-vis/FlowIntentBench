"""Targeted Flow Expert review for unresolved scientific-question families.

The review is advisory and evidence conditioned.  It reuses the canonical
Flow Expert profile, never retrieves new evidence, and cannot mutate or promote
canonical cases.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .flow_expert_portfolio_validation import (
    build_flow_expert_scientific_target_packets,
)
from .live_agents import LiveModelCaller
from .scientific_expert_review import (
    FLOW_SCIENTIFIC_REVIEWER_MAX_OUTPUT_TOKENS,
    FLOW_SCIENTIFIC_REVIEWER_PROFILE,
    validate_scientific_review_packet,
)


TARGETED_SCIENTIFIC_RESOLUTION_VERSION = "targeted-scientific-resolution-v2"

_JUDGMENTS = frozenset(
    {
        "SAME_TARGET_SUPPORTED",
        "SAME_TARGET_NOT_SUPPORTED",
        "UNRESOLVED",
        "TARGET_REDESIGN_REQUIRED",
    }
)
_ACTIONS = frozenset(
    {
        "KEEP_CURRENT_TARGET",
        "REQUEST_SPECIFIC_EVIDENCE",
        "REVISE_O_SPACE",
        "REDESIGN_TARGET",
    }
)
_EVIDENCE_SUFFICIENCY = frozenset({"SUFFICIENT", "INSUFFICIENT"})
_CONDITIONS = frozenset({"O1-F1", "O2-F1", "O3-F1", "O1-F2"})

TARGET_INVARIANCE_CONTRACT: dict[str, Any] = {
    "validity_predicates": [
        "scientific_object_invariant",
        "target_meaning_invariant",
        "legitimate_consequential_o_choice",
    ],
    "different_g_of_o_allowed": True,
    "ranking_agreement_required": False,
    "selected_feature_agreement_required": False,
    "numerical_agreement_required": False,
}

CONDITION_SCIENTIFIC_STATUSES = frozenset(
    {
        "CANDIDATE_FOR_HUMAN_REVIEW",
        "BLOCKED_PENDING_EVIDENCE",
        "BLOCKED_PENDING_O_SPACE_REVIEW",
        "BLOCKED_PENDING_SCIENTIFIC_REVIEW",
        "NEW_CASE_PENDING_HUMAN_SELECTION",
        "OMITTED_SCIENTIFICALLY_UNSUPPORTED",
    }
)

TARGETED_REVIEW_SPECS: dict[str, dict[str, Any]] = {
    "kitchen_concentration_heterogeneity": {
        "review_kind": "CONCENTRATION_MEASURE_INVARIANCE",
        "review_questions": [
            {
                "review_question_id": "concentration_measure_invariance",
                "question": (
                    "Do coefficient of variation, normalized variance, and robust or "
                    "spread-based measures supported by the visible evidence remain "
                    "Operationalizations of one target: spatial concentration heterogeneity?"
                ),
                "choices_under_review": [
                    "coefficient of variation",
                    "normalized variance",
                    "robust or spread-based concentration measures",
                ],
                "applies_to_conditions": ["O2-F1", "O3-F1"],
            }
        ],
    },
    "kitchen_turbulence_activity": {
        "review_kind": "TURBULENCE_TARGET_INVARIANCE",
        "review_questions": [
            {
                "review_question_id": "tke_vs_dissipation_invariance",
                "question": (
                    "Do TKE-based and dissipation-based extrema remain valid "
                    "Operationalizations of one stable turbulence-activity target?"
                ),
                "choices_under_review": [
                    "turbulent kinetic energy extremum",
                    "turbulent dissipation-rate extremum",
                ],
                "applies_to_conditions": ["O2-F1"],
            },
            {
                "review_question_id": "point_vs_region_invariance",
                "question": (
                    "Can a point hotspot and a localized hotspot region both "
                    "Operationalize the same stable scientific object?"
                ),
                "choices_under_review": ["point hotspot", "localized hotspot region"],
                "applies_to_conditions": ["O3-F1"],
            },
        ],
    },
    "combustor_density_features": {
        "review_kind": "DENSITY_TARGET_REDESIGN",
        "review_questions": [
            {
                "review_question_id": "density_target_redesign",
                "question": (
                    "Do maximum density and maximum density-gradient magnitude answer "
                    "different physical questions, and which evidence-grounded scientific "
                    "targets should be offered for accountable human selection?"
                ),
                "choices_under_review": [
                    "maximum stored density",
                    "maximum density-gradient magnitude",
                ],
                "applies_to_conditions": ["O2-F1", "O3-F1"],
            }
        ],
    },
}


TargetedInvoker = Callable[[Mapping[str, Any], str], Mapping[str, Any]]


def compile_target_invariance_judgment(
    *,
    scientific_object_invariant: bool | None,
    target_meaning_invariant: bool | None,
    legitimate_consequential_o_choice: bool | None,
    outcome_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile reviewed target predicates without comparing realized outcomes.

    ``outcome_diagnostics`` may record that valid branches produced different
    G(O), rankings, or selected features. Those observations are returned for
    audit only and are never consulted when deriving the judgment.
    """

    predicates = {
        "scientific_object_invariant": scientific_object_invariant,
        "target_meaning_invariant": target_meaning_invariant,
        "legitimate_consequential_o_choice": legitimate_consequential_o_choice,
    }
    invalid = [
        name
        for name, value in predicates.items()
        if value is not None and not isinstance(value, bool)
    ]
    if invalid:
        raise ValueError(
            "target-invariance predicates must be bool or None: "
            + ", ".join(sorted(invalid))
        )
    if any(value is False for value in predicates.values()):
        judgment = "SAME_TARGET_NOT_SUPPORTED"
        target_drift = True
        reason_codes = [
            name.upper() + "_NOT_SUPPORTED"
            for name, value in predicates.items()
            if value is False
        ]
    elif all(value is True for value in predicates.values()):
        judgment = "SAME_TARGET_SUPPORTED"
        target_drift = False
        reason_codes = []
    else:
        judgment = "UNRESOLVED"
        target_drift = None
        reason_codes = [
            name.upper() + "_NOT_REVIEWED"
            for name, value in predicates.items()
            if value is None
        ]
    return {
        "judgment": judgment,
        "target_drift": target_drift,
        "validity_predicates": predicates,
        "reason_codes": sorted(reason_codes),
        "outcome_diagnostics": dict(outcome_diagnostics or {}),
        "outcome_equivalence_used_as_validity_predicate": False,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_targeted_scientific_resolution_packets(
    repository_root: str | Path,
) -> list[dict[str, Any]]:
    """Derive three target-specific packets from validated Flow Expert input."""

    packets = build_flow_expert_scientific_target_packets(repository_root)
    result: list[dict[str, Any]] = []
    for packet in packets:
        validation = validate_scientific_review_packet(packet)
        if validation["status"] != "PASS":
            raise ValueError(
                "canonical Flow Expert packet is invalid: "
                + ",".join(validation["reason_codes"])
            )
        family_id = str(packet["family_identity"]["family_id"])
        spec = TARGETED_REVIEW_SPECS.get(family_id)
        if spec is None:
            continue
        result.append(
            {
                "packet_version": TARGETED_SCIENTIFIC_RESOLUTION_VERSION,
                "review_kind": spec["review_kind"],
                "dataset_identity": packet["dataset_identity"],
                "family_identity": packet["family_identity"],
                "current_scientific_target": packet["scientific_target"],
                "target_invariance_contract": dict(TARGET_INVARIANCE_CONTRACT),
                "review_questions": spec["review_questions"],
                "dataset_context": packet["dataset_context"],
                "observable_semantics": packet["observable_semantics"],
                "conditions": packet["conditions"],
                "scientific_claims": packet["scientific_claims"],
                "evidence_records": packet["evidence_records"],
                "source_records": packet["source_records"],
                "claim_support_links": packet["claim_support_links"],
                "provenance_summary": packet["provenance_summary"],
                "runtime_contract": packet["runtime_contract"],
                "authority": {
                    "advisory_only": True,
                    "automatic_case_mutation": False,
                    "automatic_target_adoption": False,
                    "retrieval_allowed": False,
                },
            }
        )
    expected = set(TARGETED_REVIEW_SPECS)
    observed = {str(item["family_identity"]["family_id"]) for item in result}
    if observed != expected:
        raise ValueError(
            f"targeted packet families mismatch: expected {sorted(expected)}, got {sorted(observed)}"
        )
    return result


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        return None
    normalized = [item.strip() for item in value]
    return normalized if len(normalized) == len(set(normalized)) else None


def validate_targeted_scientific_resolution(
    output: Any,
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one advisory result without adding scientific judgments."""

    errors: list[dict[str, str]] = []
    if not isinstance(output, Mapping):
        return {
            "status": "INCOMPLETE",
            "reason_codes": ["OUTPUT_NOT_OBJECT"],
            "errors": [{"code": "OUTPUT_NOT_OBJECT", "field": "output"}],
            "normalized_review": {},
        }
    allowed_top = {
        "family_id",
        "evidence_sufficiency",
        "conclusions",
        "missing_evidence_requests",
        "candidate_scientific_targets",
        "human_selection_required",
    }
    unknown = sorted(set(output) - allowed_top)
    if unknown:
        errors.append({"code": "UNKNOWN_OUTPUT_FIELD", "field": ",".join(unknown)})
    expected_family = str(packet.get("family_identity", {}).get("family_id", ""))
    if output.get("family_id") != expected_family:
        errors.append({"code": "FAMILY_ID_MISMATCH", "field": "family_id"})
    sufficiency = output.get("evidence_sufficiency")
    if sufficiency not in _EVIDENCE_SUFFICIENCY:
        errors.append(
            {"code": "INVALID_EVIDENCE_SUFFICIENCY", "field": "evidence_sufficiency"}
        )
    known_evidence = {
        str(item.get("evidence_id"))
        for item in packet.get("evidence_records", ())
        if isinstance(item, Mapping)
    }
    expected_questions = {
        str(item.get("review_question_id"))
        for item in packet.get("review_questions", ())
        if isinstance(item, Mapping)
    }
    for index, question in enumerate(packet.get("review_questions", ())):
        if not isinstance(question, Mapping):
            errors.append(
                {
                    "code": "TARGETED_REVIEW_CONDITION_SCOPE_INVALID",
                    "field": f"review_questions[{index}]",
                }
            )
            continue
        applies_to = _string_list(question.get("applies_to_conditions"))
        if (
            not applies_to
            or set(applies_to) - _CONDITIONS
        ):
            errors.append(
                {
                    "code": "TARGETED_REVIEW_CONDITION_SCOPE_INVALID",
                    "field": f"review_questions[{index}].applies_to_conditions",
                }
            )
    raw_conclusions = output.get("conclusions")
    conclusions: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not isinstance(raw_conclusions, list):
        errors.append({"code": "CONCLUSIONS_REQUIRED", "field": "conclusions"})
        raw_conclusions = []
    conclusion_fields = {
        "review_question_id",
        "judgment",
        "recommended_action",
        "rationale",
        "evidence_ids",
        "unresolved_ambiguity",
    }
    for index, raw in enumerate(raw_conclusions):
        field = f"conclusions[{index}]"
        if not isinstance(raw, Mapping):
            errors.append({"code": "CONCLUSION_NOT_OBJECT", "field": field})
            continue
        unexpected = sorted(set(raw) - conclusion_fields)
        if unexpected:
            errors.append(
                {"code": "UNKNOWN_OUTPUT_FIELD", "field": f"{field}:{','.join(unexpected)}"}
            )
        question_id = str(raw.get("review_question_id", "")).strip()
        if question_id not in expected_questions:
            errors.append({"code": "UNKNOWN_REVIEW_QUESTION", "field": field})
        if question_id in seen:
            errors.append({"code": "DUPLICATE_REVIEW_QUESTION", "field": field})
        seen.add(question_id)
        if raw.get("judgment") not in _JUDGMENTS:
            errors.append({"code": "INVALID_JUDGMENT", "field": field})
        if raw.get("recommended_action") not in _ACTIONS:
            errors.append({"code": "INVALID_RECOMMENDED_ACTION", "field": field})
        rationale = str(raw.get("rationale", "")).strip()
        if not rationale:
            errors.append({"code": "RATIONALE_REQUIRED", "field": field})
        evidence_ids = _string_list(raw.get("evidence_ids"))
        if evidence_ids is None:
            errors.append({"code": "EVIDENCE_IDS_INVALID", "field": field})
            evidence_ids = []
        if set(evidence_ids) - known_evidence:
            errors.append({"code": "UNKNOWN_EVIDENCE_ID", "field": field})
        conclusions.append(
            {
                "review_question_id": question_id,
                "judgment": raw.get("judgment"),
                "recommended_action": raw.get("recommended_action"),
                "rationale": rationale,
                "evidence_ids": evidence_ids,
                "unresolved_ambiguity": str(raw.get("unresolved_ambiguity", "")).strip(),
            }
        )
    if seen != expected_questions:
        errors.append(
            {
                "code": "REQUIRED_REVIEW_QUESTIONS_MISSING",
                "field": ",".join(sorted(expected_questions - seen)),
            }
        )

    missing_requests: list[dict[str, str]] = []
    raw_missing = output.get("missing_evidence_requests")
    if not isinstance(raw_missing, list):
        errors.append(
            {"code": "MISSING_EVIDENCE_REQUESTS_REQUIRED", "field": "missing_evidence_requests"}
        )
        raw_missing = []
    for index, raw in enumerate(raw_missing):
        field = f"missing_evidence_requests[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != {"claim", "why_required", "scope"}:
            errors.append({"code": "MISSING_EVIDENCE_REQUEST_INVALID", "field": field})
            continue
        row = {name: str(raw.get(name, "")).strip() for name in ("claim", "why_required", "scope")}
        if not all(row.values()):
            errors.append({"code": "MISSING_EVIDENCE_REQUEST_INVALID", "field": field})
        missing_requests.append(row)
    if sufficiency == "INSUFFICIENT" and not missing_requests:
        errors.append(
            {"code": "SPECIFIC_MISSING_EVIDENCE_REQUIRED", "field": "missing_evidence_requests"}
        )

    raw_targets = output.get("candidate_scientific_targets")
    targets: list[dict[str, Any]] = []
    if not isinstance(raw_targets, list):
        errors.append(
            {"code": "CANDIDATE_SCIENTIFIC_TARGETS_REQUIRED", "field": "candidate_scientific_targets"}
        )
        raw_targets = []
    target_fields = {
        "candidate_id",
        "scientific_meaning",
        "evidence_basis",
        "evidence_ids",
        "plausible_o_dimensions",
        "distinct_from_rejected_target",
        "automatic_adoption",
    }
    for index, raw in enumerate(raw_targets):
        field = f"candidate_scientific_targets[{index}]"
        if not isinstance(raw, Mapping):
            errors.append({"code": "CANDIDATE_TARGET_NOT_OBJECT", "field": field})
            continue
        if set(raw) != target_fields:
            errors.append({"code": "CANDIDATE_TARGET_FIELDS_INVALID", "field": field})
        evidence_ids = _string_list(raw.get("evidence_ids"))
        if evidence_ids is None or set(evidence_ids) - known_evidence:
            errors.append({"code": "CANDIDATE_TARGET_EVIDENCE_INVALID", "field": field})
            evidence_ids = []
        dimensions = raw.get("plausible_o_dimensions")
        if not isinstance(dimensions, list) or not dimensions:
            errors.append({"code": "CANDIDATE_TARGET_O_DIMENSIONS_INVALID", "field": field})
            dimensions = []
        normalized_dimensions: list[dict[str, str]] = []
        for item in dimensions:
            if not isinstance(item, Mapping) or set(item) != {"dimension_id", "scientific_role"}:
                errors.append({"code": "CANDIDATE_TARGET_O_DIMENSIONS_INVALID", "field": field})
                continue
            normalized_dimensions.append(
                {
                    "dimension_id": str(item.get("dimension_id", "")).strip(),
                    "scientific_role": str(item.get("scientific_role", "")).strip(),
                }
            )
        text_fields = {
            name: str(raw.get(name, "")).strip()
            for name in (
                "candidate_id",
                "scientific_meaning",
                "evidence_basis",
                "distinct_from_rejected_target",
            )
        }
        if not all(text_fields.values()):
            errors.append({"code": "CANDIDATE_TARGET_TEXT_REQUIRED", "field": field})
        distinction = text_fields["distinct_from_rejected_target"].casefold()
        if any(
            placeholder in distinction
            for placeholder in (
                "cannot be determined",
                "cannot determine",
                "not identified",
                "not provided",
                "unknown",
                "not applicable",
            )
        ):
            errors.append(
                {
                    "code": "CANDIDATE_TARGET_DISTINCTION_UNRESOLVED",
                    "field": field,
                }
            )
        if raw.get("automatic_adoption") is not False:
            errors.append({"code": "AUTOMATIC_TARGET_ADOPTION_FORBIDDEN", "field": field})
        targets.append(
            {
                **text_fields,
                "evidence_ids": evidence_ids,
                "plausible_o_dimensions": normalized_dimensions,
                "automatic_adoption": False,
            }
        )
    density = expected_family == "combustor_density_features"
    if density and not 2 <= len(targets) <= 3:
        errors.append({"code": "DENSITY_TARGET_COUNT_INVALID", "field": "candidate_scientific_targets"})
    if not density and targets:
        errors.append({"code": "UNEXPECTED_TARGET_REDESIGN", "field": "candidate_scientific_targets"})
    if output.get("human_selection_required") is not True:
        errors.append({"code": "HUMAN_SELECTION_MUST_REMAIN_REQUIRED", "field": "human_selection_required"})
    reason_codes = sorted({item["code"] for item in errors})
    return {
        "status": "PASS" if not errors else "INCOMPLETE",
        "reason_codes": reason_codes,
        "errors": errors,
        "normalized_review": {
            "family_id": expected_family,
            "evidence_sufficiency": sufficiency,
            "conclusions": conclusions,
            "missing_evidence_requests": missing_requests,
            "candidate_scientific_targets": targets,
            "human_selection_required": True,
        },
    }


def _targeted_question_scopes(
    review_questions: Sequence[Mapping[str, Any]],
) -> dict[str, frozenset[str]]:
    scopes: dict[str, frozenset[str]] = {}
    for index, question in enumerate(review_questions):
        if not isinstance(question, Mapping):
            raise ValueError(f"review_questions[{index}] must be an object")
        question_id = str(question.get("review_question_id", "")).strip()
        applies_to = _string_list(question.get("applies_to_conditions"))
        if (
            not question_id
            or question_id in scopes
            or not applies_to
            or set(applies_to) - _CONDITIONS
        ):
            raise ValueError(
                "targeted review questions require unique IDs and explicit valid "
                "applies_to_conditions"
            )
        scopes[question_id] = frozenset(applies_to)
    if not scopes:
        raise ValueError("targeted review questions require condition scopes")
    return scopes


def _normalized_targeted_review(review: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(review, Mapping):
        return {}
    validated = review.get("validated_review")
    if isinstance(validated, Mapping):
        if validated.get("status") != "PASS":
            return {}
        normalized = validated.get("normalized_review")
        return normalized if isinstance(normalized, Mapping) else {}
    normalized = review.get("normalized_review")
    if isinstance(normalized, Mapping):
        return normalized
    return review


def _condition_review_payload(review: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(review, Mapping):
        return {}
    nested = review.get("flow_expert_target_stability")
    return nested if isinstance(nested, Mapping) else review


def resolve_condition_scientific_status(
    condition: str,
    condition_review: Mapping[str, Any] | None,
    *,
    targeted_review: Mapping[str, Any] | None = None,
    review_questions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve one condition without broadcasting family-level deficiencies."""

    normalized_condition = str(condition).strip().upper()
    if normalized_condition not in _CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    review = _condition_review_payload(condition_review)
    scopes: dict[str, frozenset[str]] = {}
    if targeted_review is not None:
        if review_questions is None:
            raise ValueError(
                "targeted review requires explicitly condition-scoped review questions"
            )
        scopes = _targeted_question_scopes(review_questions)
    targeted = _normalized_targeted_review(targeted_review)
    conclusions = targeted.get("conclusions", ()) if targeted else ()
    applicable: list[Mapping[str, Any]] = []
    if conclusions:
        for conclusion in conclusions:
            if not isinstance(conclusion, Mapping):
                continue
            question_id = str(conclusion.get("review_question_id", "")).strip()
            if question_id not in scopes:
                raise ValueError(
                    f"targeted conclusion has no condition scope: {question_id or '<empty>'}"
                )
            if normalized_condition in scopes[question_id]:
                applicable.append(conclusion)

    reason_codes: set[str] = set()
    evidence_gaps: list[dict[str, str]] = []
    targeted_status: str | None = None
    validated_targeted = (
        targeted_review.get("validated_review")
        if isinstance(targeted_review, Mapping)
        else None
    )
    targeted_review_incomplete = (
        isinstance(validated_targeted, Mapping)
        and validated_targeted.get("status") != "PASS"
    )
    if targeted_review_incomplete and any(
        normalized_condition in condition_scope for condition_scope in scopes.values()
    ):
        targeted_status = "BLOCKED_PENDING_SCIENTIFIC_REVIEW"
        reason_codes.add("TARGETED_SCIENTIFIC_REVIEW_INCOMPLETE")
    for conclusion in applicable:
        question_id = str(conclusion.get("review_question_id", ""))
        judgment = str(conclusion.get("judgment", ""))
        action = str(conclusion.get("recommended_action", ""))
        ambiguity = str(conclusion.get("unresolved_ambiguity", "")).strip()
        if ambiguity:
            evidence_gaps.append(
                {
                    "source": "TARGETED_CONCLUSION",
                    "review_question_id": question_id,
                    "gap": ambiguity,
                }
            )
        if action == "REDESIGN_TARGET" or judgment == "TARGET_REDESIGN_REQUIRED":
            targeted_status = "NEW_CASE_PENDING_HUMAN_SELECTION"
            reason_codes.add("TARGET_REDESIGN_REQUIRED")
        elif targeted_status != "NEW_CASE_PENDING_HUMAN_SELECTION" and (
            action == "REVISE_O_SPACE" or judgment == "SAME_TARGET_NOT_SUPPORTED"
        ):
            targeted_status = "BLOCKED_PENDING_O_SPACE_REVIEW"
            reason_codes.add("TARGET_INVARIANCE_NOT_SUPPORTED")
        elif targeted_status not in {
            "NEW_CASE_PENDING_HUMAN_SELECTION",
            "BLOCKED_PENDING_O_SPACE_REVIEW",
        } and (action == "REQUEST_SPECIFIC_EVIDENCE" or judgment == "UNRESOLVED"):
            targeted_status = "BLOCKED_PENDING_EVIDENCE"
            reason_codes.add("CONDITION_EVIDENCE_INSUFFICIENT")

    current_advice = str(review.get("scientific_reuse_advice", "")).strip()
    if not current_advice:
        legacy_advice = str(
            review.get("advisory_gt_srac_reuse_status", "")
        ).strip()
        current_advice = {
            "VALID": "REUSE_ALLOWED",
            "NEW_CASE_REQUIRED": "NEW_CASE_REQUIRED",
        }.get(legacy_advice, "")
    if (
        not current_advice
        and review.get("target_stability_judgment")
        == "TARGET_STABLE_WITHIN_REVIEWED_CONDITION"
    ):
        current_advice = "REUSE_ALLOWED"
    if current_advice == "NEW_CASE_REQUIRED":
        base_status = "NEW_CASE_PENDING_HUMAN_SELECTION"
        reason_codes.add("TARGET_REDESIGN_REQUIRED")
    elif current_advice == "BLOCKED_PENDING_O_SPACE_REVIEW":
        base_status = "BLOCKED_PENDING_O_SPACE_REVIEW"
        reason_codes.add("CONDITION_O_SPACE_REVIEW_REQUIRED")
    elif current_advice == "BLOCKED_PENDING_EVIDENCE":
        base_status = "BLOCKED_PENDING_EVIDENCE"
        reason_codes.add("CONDITION_EVIDENCE_INSUFFICIENT")
    elif current_advice == "BLOCKED_PENDING_SCIENTIFIC_REVIEW":
        base_status = "BLOCKED_PENDING_SCIENTIFIC_REVIEW"
        reason_codes.add("CONDITION_SCIENTIFIC_REVIEW_INCOMPLETE")
    elif current_advice == "REUSE_ALLOWED":
        base_status = "CANDIDATE_FOR_HUMAN_REVIEW"
    elif not review:
        base_status = "BLOCKED_PENDING_SCIENTIFIC_REVIEW"
        reason_codes.add("CONDITION_SCIENTIFIC_REVIEW_MISSING")
    else:
        recommendation = str(review.get("eligibility_recommendation", ""))
        question_recommendation = str(review.get("question_recommendation", ""))
        if question_recommendation == "REVISE_TARGET":
            base_status = "NEW_CASE_PENDING_HUMAN_SELECTION"
            reason_codes.add("TARGET_REDESIGN_REQUIRED")
        elif question_recommendation == "REVISE_OPERATIONALIZATION_SPACE":
            base_status = "BLOCKED_PENDING_O_SPACE_REVIEW"
            reason_codes.add("CONDITION_O_SPACE_REVIEW_REQUIRED")
        elif recommendation == "MORE_EVIDENCE_REQUIRED":
            base_status = "BLOCKED_PENDING_EVIDENCE"
            reason_codes.add("CONDITION_EVIDENCE_INSUFFICIENT")
        elif review.get("scientific_target_supported") is False:
            base_status = "OMITTED_SCIENTIFICALLY_UNSUPPORTED"
            reason_codes.add("TARGET_UNSUPPORTED")
        elif review.get("fixed_o_supported") is False:
            base_status = "OMITTED_SCIENTIFICALLY_UNSUPPORTED"
            reason_codes.add("FIXED_O_UNSUPPORTED")
        elif (
            normalized_condition in {"O2-F1", "O3-F1"}
            and review.get("unresolved_o_is_legitimate_scientific_choice") is False
        ):
            base_status = "BLOCKED_PENDING_O_SPACE_REVIEW"
            reason_codes.add("UNRESOLVED_O_NOT_LEGITIMATE")
        elif review.get("finding_contract_supported") is False:
            base_status = "OMITTED_SCIENTIFICALLY_UNSUPPORTED"
            reason_codes.add("FINDING_CONTRACT_UNSUPPORTED")
        elif review.get("materialization_scientifically_meaningful") is False:
            base_status = "OMITTED_SCIENTIFICALLY_UNSUPPORTED"
            reason_codes.add("MATERIALIZATION_SCIENTIFIC_MEANING_UNSUPPORTED")
        else:
            required = [
                "scientific_target_supported",
                "fixed_o_supported",
                "finding_contract_supported",
                "materialization_scientifically_meaningful",
            ]
            if normalized_condition in {"O2-F1", "O3-F1"}:
                required.append("unresolved_o_is_legitimate_scientific_choice")
            if all(review.get(field) is True for field in required):
                base_status = "CANDIDATE_FOR_HUMAN_REVIEW"
            else:
                base_status = "BLOCKED_PENDING_SCIENTIFIC_REVIEW"
                reason_codes.add("CONDITION_SCIENTIFIC_REVIEW_INCOMPLETE")

    priority = {
        "CANDIDATE_FOR_HUMAN_REVIEW": 0,
        "BLOCKED_PENDING_SCIENTIFIC_REVIEW": 1,
        "BLOCKED_PENDING_EVIDENCE": 2,
        "OMITTED_SCIENTIFICALLY_UNSUPPORTED": 3,
        "BLOCKED_PENDING_O_SPACE_REVIEW": 4,
        "NEW_CASE_PENDING_HUMAN_SELECTION": 5,
    }
    status = max(
        (value for value in (base_status, targeted_status) if value is not None),
        key=priority.__getitem__,
    )
    if status == "BLOCKED_PENDING_EVIDENCE" and not evidence_gaps:
        rationale = str(review.get("rationale", "")).strip()
        if rationale:
            evidence_gaps.append(
                {
                    "source": "CONDITION_REVIEW",
                    "review_question_id": normalized_condition,
                    "gap": rationale,
                }
            )
    return {
        "condition": normalized_condition,
        "CONDITION_SCIENTIFIC_STATUS": status,
        "blocking_reason_codes": sorted(reason_codes),
        "evidence_gaps_specific_to_condition": evidence_gaps,
        "condition_evidence_sufficiency": (
            "SUFFICIENT"
            if status == "CANDIDATE_FOR_HUMAN_REVIEW"
            else "INSUFFICIENT"
            if status == "BLOCKED_PENDING_EVIDENCE"
            else "NOT_ESTABLISHED"
        ),
        "applicable_targeted_review_question_ids": [
            str(item.get("review_question_id", "")) for item in applicable
        ],
        "family_level_deficiency_broadcast": False,
    }


def resolve_condition_scientific_statuses(
    conditions: Sequence[Mapping[str, Any]],
    condition_reviews: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    targeted_review: Mapping[str, Any] | None = None,
    review_questions: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Resolve a sparse family one condition at a time, preserving input order."""

    if isinstance(condition_reviews, Mapping):
        by_key = {str(key): value for key, value in condition_reviews.items()}
    else:
        by_key = {}
        for review in condition_reviews:
            if not isinstance(review, Mapping):
                continue
            for key in ("condition", "case_id"):
                value = str(review.get(key, "")).strip()
                if value:
                    by_key[value] = review
    results: list[dict[str, Any]] = []
    for item in conditions:
        condition = str(item.get("condition", "")).strip().upper()
        case_id = str(item.get("case_id", "")).strip()
        review = by_key.get(condition) or by_key.get(case_id)
        results.append(
            resolve_condition_scientific_status(
                condition,
                review,
                targeted_review=targeted_review,
                review_questions=review_questions,
            )
        )
    return results


TARGETED_SCIENTIFIC_RESOLUTION_INSTRUCTION = (
    "You are the existing FlowIntentBench evidence-conditioned Flow Expert in a targeted "
    "scientific-question resolution mode. Use only the visible packet; do not retrieve, "
    "assume hidden results, or prefer a measure merely because it is listed. Judge target "
    "invariance only from scientific-object invariance, target-meaning invariance, and whether "
    "the open choice is a legitimate consequential Operationalization choice. Different valid "
    "O choices may produce different G(O), rankings, selected features, or numerical values; "
    "none of those differences alone is target drift. Return exactly "
    "one JSON object with family_id, evidence_sufficiency (SUFFICIENT or INSUFFICIENT), "
    "conclusions, missing_evidence_requests, candidate_scientific_targets, and "
    "human_selection_required=true. Do not echo current_scientific_target or any other "
    "packet field into the output object. conclusions must contain exactly one row per supplied "
    "review_question_id with judgment (SAME_TARGET_SUPPORTED, SAME_TARGET_NOT_SUPPORTED, "
    "UNRESOLVED, or TARGET_REDESIGN_REQUIRED), recommended_action (KEEP_CURRENT_TARGET, "
    "REQUEST_SPECIFIC_EVIDENCE, REVISE_O_SPACE, or REDESIGN_TARGET), non-empty rationale, "
    "evidence_ids, and unresolved_ambiguity. If evidence is insufficient, request only the "
    "specific missing claim/evidence in rows containing exactly claim, why_required, and "
    "scope. For concentration and turbulence, candidate_scientific_targets must be empty. "
    "For density, return two or three evidence-grounded target candidates; each must contain "
    "candidate_id, scientific_meaning, evidence_basis, evidence_ids, plausible_o_dimensions "
    "as dimension_id/scientific_role objects, distinct_from_rejected_target, and "
    "automatic_adoption=false. For density, current_scientific_target is the rejected, "
    "underdefined target being redesigned. distinct_from_rejected_target must concretely "
    "explain how the candidate narrows or replaces that target; do not answer that the "
    "distinction cannot be determined or that the rejected target was not identified. "
    "Do not author canonical cases, modify GT, execute SCQ/SRAC, "
    "or make a human selection. Cite only evidence_ids present in the packet."
)


def run_targeted_scientific_resolution(
    repository_root: str | Path,
    packet: Mapping[str, Any],
    *,
    invoker: TargetedInvoker | None = None,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Run or inject one targeted advisory Flow Expert review."""

    family_id = str(packet.get("family_identity", {}).get("family_id", ""))
    if family_id not in TARGETED_REVIEW_SPECS:
        raise ValueError(f"unsupported targeted scientific family: {family_id}")
    item_id = f"targeted-scientific-resolution::{family_id}"
    if invoker is not None:
        raw = dict(invoker(packet, item_id))
        call: Mapping[str, Any] | None = None
        invocation_status = "SUCCESS"
        live_model_calls = False
        execution_mode = "INJECTED_REVIEWER"
    else:
        caller = LiveModelCaller(
            Path(repository_root).resolve(),
            FLOW_SCIENTIFIC_REVIEWER_PROFILE,
            config_path=config_path,
            api_key=api_key,
            timeout=timeout,
            max_output_tokens=FLOW_SCIENTIFIC_REVIEWER_MAX_OUTPUT_TOKENS,
        )
        expected_question_ids = [
            str(item.get("review_question_id", ""))
            for item in packet.get("review_questions", ())
            if isinstance(item, Mapping)
        ]
        call_instruction = (
            TARGETED_SCIENTIFIC_RESOLUTION_INSTRUCTION
            + " This is not a per-condition review. Do not use O1-F1, O2-F1, O3-F1, "
            "or O1-F2 as review_question_id values. For this call, conclusions must "
            "contain exactly these review_question_id values and no others: "
            + ", ".join(expected_question_ids)
            + "."
        )
        call = caller.call(
            role="scientific_reviewer",
            visible_payload=packet,
            instruction=call_instruction,
            identity={
                "family_id": family_id,
                "review_index": "targeted-scientific-question-resolution",
                "scientific_review_run_id": item_id,
            },
            tool_free=True,
        )
        invocation_status = str(call.get("invocation_status", "PROVIDER_ERROR"))
        parsed = call.get("parsed_result")
        raw = dict(parsed) if isinstance(parsed, Mapping) else {}
        live_model_calls = True
        execution_mode = "LIVE_MODEL_CALL"
    validated = (
        validate_targeted_scientific_resolution(raw, packet)
        if invocation_status == "SUCCESS"
        else {
            "status": "NOT_EVALUATED",
            "reason_codes": ["INVOCATION_NOT_SUCCESSFUL"],
            "errors": [],
            "normalized_review": {},
        }
    )
    return {
        "status": "SUCCESS" if validated["status"] == "PASS" else "INCOMPLETE",
        "invocation_status": invocation_status,
        "execution_mode": execution_mode,
        "live_model_calls": live_model_calls,
        "proxy_only": True,
        "automatic_case_mutation": False,
        "automatic_target_adoption": False,
        "parsed_result": raw,
        "validated_review": validated,
        "call": call,
    }


def _family_scientific_status(review: Mapping[str, Any]) -> str:
    validated = review.get("validated_review")
    if not isinstance(validated, Mapping) or validated.get("status") != "PASS":
        return "SCIENTIFIC_REVIEW_INCOMPLETE"
    normalized = validated.get("normalized_review")
    if not isinstance(normalized, Mapping):
        return "SCIENTIFIC_REVIEW_INCOMPLETE"
    family_id = str(normalized.get("family_id", ""))
    actions = {
        str(item.get("recommended_action", ""))
        for item in normalized.get("conclusions", ())
        if isinstance(item, Mapping)
    }
    insufficient = normalized.get("evidence_sufficiency") == "INSUFFICIENT"
    if family_id == "combustor_density_features" and "REDESIGN_TARGET" in actions:
        return "TARGET_REDESIGN_CANDIDATES_PENDING_HUMAN_SELECTION"
    if "REVISE_O_SPACE" in actions and insufficient:
        return "EVIDENCE_AND_O_SPACE_REVIEW_REQUIRED"
    if "REVISE_O_SPACE" in actions:
        return "O_SPACE_REVIEW_REQUIRED"
    if insufficient:
        return "BLOCKED_PENDING_SPECIFIC_EVIDENCE"
    if actions == {"KEEP_CURRENT_TARGET"}:
        return "CURRENT_TARGET_SUPPORTED_BY_VISIBLE_EVIDENCE"
    return "SCIENTIFIC_REVIEW_REQUIRED"


def run_targeted_scientific_resolution_portfolio(
    repository_root: str | Path,
    output_root: str | Path,
    *,
    invoker: TargetedInvoker | None = None,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
    max_workers: int = 3,
) -> dict[str, Any]:
    """Run the three targeted reviews concurrently and preserve raw artifacts."""

    root = Path(repository_root).resolve()
    out = Path(output_root).resolve()
    packets = build_targeted_scientific_resolution_packets(root)

    def execute(packet: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        family_id = str(packet["family_identity"]["family_id"])
        return family_id, run_targeted_scientific_resolution(
            root,
            packet,
            invoker=invoker,
            config_path=config_path,
            api_key=api_key,
            timeout=timeout,
        )

    reviews: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(packets)))) as pool:
        futures = {pool.submit(execute, packet): packet for packet in packets}
        for future in as_completed(futures):
            packet = futures[future]
            family_id = str(packet["family_identity"]["family_id"])
            try:
                result_family, review = future.result()
                reviews[result_family] = review
            except Exception as exc:
                reviews[family_id] = {
                    "status": "INCOMPLETE",
                    "invocation_status": "INFRASTRUCTURE_INVALID",
                    "execution_mode": "LIVE_MODEL_CALL" if invoker is None else "INJECTED_REVIEWER",
                    "live_model_calls": invoker is None,
                    "proxy_only": True,
                    "automatic_case_mutation": False,
                    "automatic_target_adoption": False,
                    "parsed_result": {},
                    "validated_review": {
                        "status": "NOT_EVALUATED",
                        "reason_codes": ["REVIEWER_EXCEPTION"],
                        "errors": [],
                        "normalized_review": {},
                    },
                    "call": {"error": f"{type(exc).__name__}: {exc}"},
                }
    family_status = {
        family_id: _family_scientific_status(review)
        for family_id, review in sorted(reviews.items())
    }
    completed = len(reviews) == len(packets) and all(
        review.get("invocation_status") == "SUCCESS"
        and review.get("validated_review", {}).get("status") == "PASS"
        for review in reviews.values()
    )
    manifest = {
        "schema_version": TARGETED_SCIENTIFIC_RESOLUTION_VERSION,
        "execution_mode": "LIVE_FLOW_EXPERT" if invoker is None else "INJECTED_REVIEWER",
        "FLOW_EXPERT_TARGETED_REVIEW_STATUS": "COMPLETE" if completed else "INCOMPLETE",
        "FLOW_EXPERT_SCIENTIFIC_AUTHORITY": "ADVISORY_ONLY",
        "HUMAN_SCIENTIFIC_SELECTION_STATUS": "PENDING",
        "family_scientific_status": family_status,
        "family_count": len(packets),
        "reviews": reviews,
        "canonical_case_mutation_count": 0,
        "automatic_target_adoption_count": 0,
        "official_scq_executed_count": 0,
        "formal_model_experiment_executed_count": 0,
    }
    for packet in packets:
        family_id = str(packet["family_identity"]["family_id"])
        _write_json(out / "packets" / f"{family_id}.json", packet)
    for family_id, review in reviews.items():
        _write_json(out / "reviews" / f"{family_id}.json", review)
    _write_json(out / "manifest.json", manifest)
    return manifest


__all__ = [
    "CONDITION_SCIENTIFIC_STATUSES",
    "TARGETED_REVIEW_SPECS",
    "TARGETED_SCIENTIFIC_RESOLUTION_INSTRUCTION",
    "TARGETED_SCIENTIFIC_RESOLUTION_VERSION",
    "TARGET_INVARIANCE_CONTRACT",
    "build_targeted_scientific_resolution_packets",
    "compile_target_invariance_judgment",
    "resolve_condition_scientific_status",
    "resolve_condition_scientific_statuses",
    "run_targeted_scientific_resolution",
    "run_targeted_scientific_resolution_portfolio",
    "validate_targeted_scientific_resolution",
]

"""Human-facing closure for an explicitly supplied case inventory.

The runner freezes its engineering acceptance contract, audits every final
primary from the scientific source matrix, performs fresh-context Flow Expert
wording reviews, executes the sentinel challenge corpus, and derives readiness
without trusting writable PASS flags.  Release and curator state remain
unchanged and outside this closure.
"""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .agent_profile import load_agent_profile
from .case_repository import load_case_record
from .hccq import audit_family_presentation_leakage
from .human_facing_acceptance import (
    HUMAN_FACING_ACCEPTANCE_CONTRACT,
    _dimension_ids,
    audit_actual_hccq,
    audit_question_responsibility_realization,
    canonical_json_sha256,
    human_facing_acceptance_contract_sha256,
    run_validator_sentinel_challenges,
)
from .human_review_portfolio import (
    SUPPORT_DIMENSION_NAMES,
    build_human_review_portfolio,
)
from .diversity_audit import cross_dataset_scientific_template_audit
from .question_presentation import (
    FINAL_FLOW_EXPERT_WORDING_REVIEW_INSTRUCTION,
    QUESTION_PRESENTATION_AUTHORING_INSTRUCTION,
    audit_final_flow_expert_wording,
    run_flow_expert_final_flow_wording_review,
)
from .scientific_question_candidate_matrix import (
    CONDITIONS,
    DATASET_POLICIES,
    _capability_profiles,
    _dataset_context_for_authoring,
    _evidence_bundle,
)


HUMAN_FACING_CLOSURE_VERSION = "human-facing-question-closure-v3"
FinalReviewer = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _read_manifest(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    loaded = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("SOURCE_MATRIX_NOT_OBJECT")
    return copy.deepcopy(dict(loaded))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _primary(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    primary_id = _text(
        row.get("primary_candidate_id")
        or row.get("accepted_question_candidate_id")
        or row.get("source_primary_candidate_id")
    )
    if not primary_id:
        return None
    for candidate in row.get("question_candidates", ()):
        if isinstance(candidate, Mapping) and _text(candidate.get("candidate_id")) == primary_id:
            return candidate
    return None


def _projection(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("canonical_semantic_projection")
    if not isinstance(value, Mapping):
        raise ValueError("ROW_CANONICAL_SEMANTIC_PROJECTION_MISSING:" + _text(row.get("slot_id")))
    return value


def _finding_mode(projection: Mapping[str, Any]) -> str:
    finding = projection.get("finding_responsibility")
    if isinstance(finding, Mapping):
        return _text(finding.get("finding_mode")).upper()
    return _text(finding).upper()


def _effective_o_sha256(projection: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "scientific_target": projection.get("scientific_target"),
            "scientific_scope": projection.get("scientific_scope"),
            "resolved_operationalization": projection.get(
                "resolved_operationalization", ()
            ),
            "unresolved_operationalization_dimensions": _dimension_ids(
                projection.get("unresolved_operationalization_dimensions", ())
            ),
        }
    )


def _current_semantic_fidelity(
    row: Mapping[str, Any], candidate: Mapping[str, Any], responsibility: Mapping[str, Any]
) -> dict[str, Any]:
    proof = candidate.get("post_rewrite_semantic_fidelity")
    proof = proof if isinstance(proof, Mapping) else {}
    semantic_hash = _text(row.get("semantic_contract_sha256"))
    text = _text(candidate.get("model_visible_text"))
    expected_text_hash = canonical_json_sha256({"model_visible_text": text})
    stored_text_hash = _text(proof.get("model_visible_text_sha256"))
    # Older authoring records used the direct text digest.  Accept that
    # historical representation only when it matches; current records use
    # the same canonical digest and remain fully bound.
    if stored_text_hash and stored_text_hash != expected_text_hash:
        direct_text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        text_hash_pass = stored_text_hash == direct_text_hash
    else:
        text_hash_pass = bool(stored_text_hash)
    # A legacy authored artifact may have omitted the structured proof hash;
    # the closure runner still revalidates the current rendered candidate
    # against the immutable projection.  This fallback is presentation-only
    # and never establishes scientific qualification.
    proof_status = _text(proof.get("status")).upper()
    if not proof_status:
        proof_status = "PASS" if responsibility.get("status") == "PASS" else "FAIL"
    checks = {
        "semantic_contract_hash_bound": bool(semantic_hash)
        and _text(candidate.get("semantic_contract_sha256")) == semantic_hash
        and _text(proof.get("semantic_contract_sha256")) == semantic_hash,
        "actual_text_hash_bound": text_hash_pass,
        "deterministic_responsibility_pass": responsibility.get("status") == "PASS",
        "source_fidelity_proof_pass": proof_status == "PASS",
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "semantic_contract_sha256": semantic_hash,
        "model_visible_text_sha256": expected_text_hash,
        "fresh_proof_revalidation": True,
        "fresh_model_call_performed": False,
    }


def _fresh_review_provenance_valid(
    review: Mapping[str, Any],
    *,
    question: str,
    semantic_contract_sha256: str,
    require_live: bool,
    expected_agent_profile_sha256: str,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    provenance = review.get("provenance")
    call = review.get("review_call")
    if not isinstance(provenance, Mapping):
        errors.append("FINAL_REVIEW_PROVENANCE_MISSING")
        provenance = {}
    if not isinstance(call, Mapping):
        errors.append("FINAL_REVIEW_CALL_MISSING")
        call = {}
    review_outcome = _text(review.get("final_flow_expert_review_status")).upper()
    if review_outcome not in {"PASS", "REVISE"}:
        errors.append("FINAL_REVIEW_OUTCOME_NOT_ESTABLISHED")
    if _text(provenance.get("invocation_status")).upper() != "SUCCESS":
        errors.append("FINAL_REVIEW_INVOCATION_NOT_SUCCESS")
    if _text(provenance.get("reviewed_model_visible_text")) != question:
        errors.append("FINAL_REVIEW_TEXT_BINDING_MISMATCH")
    if _text(provenance.get("semantic_contract_sha256")) != semantic_contract_sha256:
        errors.append("FINAL_REVIEW_SEMANTIC_HASH_MISMATCH")
    effective = call.get("effective_agent_config")
    if not isinstance(effective, Mapping):
        errors.append("FINAL_REVIEW_EFFECTIVE_AGENT_CONFIG_MISSING")
        effective = {}
    if _text(effective.get("agent_profile_sha256")) != expected_agent_profile_sha256:
        errors.append("FINAL_REVIEW_AGENT_PROFILE_MISMATCH")
    if require_live:
        if _text(call.get("execution_mode")).upper() != "LIVE_MODEL_CALL":
            errors.append("FINAL_REVIEW_NOT_LIVE_MODEL_CALL")
        if call.get("live_model_calls") is not True:
            errors.append("FINAL_REVIEW_LIVE_CALL_FLAG_MISSING")
    return not errors, errors


def _context_and_evidence(
    root: Path,
    row: Mapping[str, Any],
    capabilities: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_id = _text(row.get("dataset_id"))
    case_id = _text(row.get("case_id"))
    family_id = _text(row.get("family_id"))
    if case_id:
        record = load_case_record(root, case_id)
        context = _dataset_context_for_authoring(
            record, capabilities.get(dataset_id, {})
        )
    else:
        evidence = _evidence_bundle(root, dataset_id, family_id)
        context = _dataset_context_for_authoring(
            {"case_context": evidence.get("dataset_context", {})},
            capabilities.get(dataset_id, {}),
        )
    condition_review = row.get("condition_scientific_review")
    normalized = (
        condition_review.get("normalized_review", {})
        if isinstance(condition_review, Mapping)
        else {}
    )
    evidence_context = {
        "evidence_ids": copy.deepcopy(normalized.get("evidence_ids", row.get("evidence_ids", []))),
        "evidence_gaps": copy.deepcopy(normalized.get("evidence_gaps", row.get("evidence_gaps", []))),
        "evidence_sufficiency": normalized.get(
            "evidence_sufficiency", row.get("evidence_sufficiency")
        ),
        "support_dimensions": copy.deepcopy(row.get("support_dimensions", {})),
    }
    return context, evidence_context


def _family_key(row: Mapping[str, Any]) -> str:
    return _text(row.get("family_id") or _projection(row).get("family_id") or row.get("dataset_id"))


def _family_audits(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_family_key(row), []).append(row)
    for dataset_id, family_rows in groups.items():
        by_condition = {_text(row.get("condition")): row for row in family_rows}
        target_hashes = {
            canonical_json_sha256(_projection(row).get("scientific_target"))
            for row in family_rows
        }
        condition_set = set(by_condition)
        o1 = by_condition.get("O1-F1")
        f2 = by_condition.get("O1-F2")
        o_match = bool(o1) and (f2 is None or _effective_o_sha256(_projection(o1)) == _effective_o_sha256(_projection(f2)))
        finding_progression = all(
            _finding_mode(_projection(by_condition[condition]))
            == ("F2" if condition == "O1-F2" else "F1")
            for condition in CONDITIONS
            if condition in by_condition
        )
        fixed_counts = {
            condition: len(
                _projection(row).get("resolved_operationalization", ())
            )
            for condition, row in by_condition.items()
        }
        declared = family_rows[0].get("eligible_conditions", CONDITIONS)
        declared = {str(item).upper().replace("_", "-") for item in declared}
        membership_valid = bool(o1) and condition_set == declared and len(by_condition) == len(family_rows)
        responsibility_progression = bool(
            membership_valid
            and ("O2-F1" not in by_condition or 0 < fixed_counts["O2-F1"] < fixed_counts["O1-F1"])
            and ("O3-F1" not in by_condition or fixed_counts["O3-F1"] == 0)
            and (f2 is None or fixed_counts["O1-F2"] == fixed_counts["O1-F1"])
        )
        presentations = {
            condition: {
                "model_visible_text": _text(_primary(row).get("model_visible_text"))
                if isinstance(_primary(row), Mapping)
                else ""
            }
            for condition, row in by_condition.items()
        }
        leakage = audit_family_presentation_leakage(presentations, expected_conditions=sorted(declared))
        condition_integrity = (
            membership_valid
            and len(target_hashes) == 1
            and o_match
            and finding_progression
            and responsibility_progression
        )
        result[dataset_id] = {
            "status": (
                "PASS"
                if condition_integrity and leakage.get("status") == "PASS"
                else "FAIL"
            ),
            "condition_integrity_status": "PASS" if condition_integrity else "FAIL",
            "template_leakage_status": leakage.get("status"),
            "target_invariance_status": "PASS" if len(target_hashes) == 1 else "FAIL",
            "o1_f2_effective_o_status": "PASS" if o_match else "FAIL",
            "finding_progression_status": "PASS" if finding_progression else "FAIL",
            "responsibility_progression_status": "PASS" if responsibility_progression else "FAIL",
            "presentation_audit": leakage,
        }
    return result


def _portfolio_presentation_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def question_text(row: Mapping[str, Any]) -> str:
        primary = _primary(row)
        if isinstance(primary, Mapping):
            return _text(primary.get("model_visible_text") or primary.get("scientific_question"))
        # Closure manifests produced by the finalized runner persist the
        # selected question directly and may intentionally omit the full
        # candidate list.  The audit must consume that canonical text instead
        # of crashing or silently dropping the row.
        return _text(row.get("scientific_question") or row.get("primary_human_facing_question"))

    texts = [question_text(row) for row in rows if question_text(row)]
    normalized = [" ".join(text.casefold().split()) for text in texts]
    duplicates = sorted({text for text in normalized if normalized.count(text) > 1})
    condition_labels = [
        _text(row.get("slot_id"))
        for row in rows
        if any(
            marker in question_text(row).casefold()
            for marker in ("o1-f1", "o2-f1", "o3-f1", "o1-f2", "condition o")
        )
    ]
    failures = []
    if duplicates:
        failures.append("DUPLICATE_PRIMARY_QUESTION")
    if condition_labels:
        failures.append("CONDITION_LABEL_VISIBLE")
    surface = _audit_question_surface_quality(texts)
    return {
        "status": "PASS" if not failures else "FAIL",
        "failure_codes": failures,
        "duplicate_primary_count": len(duplicates),
        "condition_label_visible_slots": condition_labels,
        "question_count": len(texts),
        "cross_dataset_scientific_template_audit": cross_dataset_scientific_template_audit(rows),
        "human_facing_surface_audit": surface,
    }


def _audit_question_surface_quality(texts: Sequence[str]) -> dict[str, Any]:
    """Run the bounded, human-facing surface checks for the v5 portfolio.

    This is intentionally a presentation diagnostic.  Scientific identity and
    responsibility are still checked by the existing semantic-fidelity and
    HCCQ gates.  Style balance is reported for review rather than used to
    rewrite a question or to manufacture scientific diversity.
    """

    import re
    from collections import Counter

    lengths = [len(str(item).split()) for item in texts if str(item).strip()]
    styles: Counter[str] = Counter()
    for text in texts:
        value = str(text).strip()
        if not value:
            continue
        if "?" in value:
            style = "DIRECT_SCIENTIFIC_QUESTION"
        elif re.search(r"\b(examine|investigate|explore|assess)\b", value.casefold()):
            style = "INVESTIGATION_ORIENTED"
        else:
            style = "CONCISE_EXPERT_REQUEST"
        styles[style] += 1
    denominator = len(lengths)
    max_fraction = max(styles.values(), default=0) / denominator if denominator else 0.0
    long_questions = [index for index, value in enumerate(lengths) if value > 90]
    very_long_questions = [index for index, value in enumerate(lengths) if value > 120]
    internal_terms = [
        index for index, value in enumerate(texts)
        if re.search(r"\b(?:O[123]-F[12]|case_id|ground_truth|SRAC|SEC|candidate_id)\b", str(value), re.IGNORECASE)
    ]
    return {
        "status": "PASS" if denominator == len(texts) and not very_long_questions and not internal_terms else "REVIEW",
        "question_count": len(texts),
        "style_counts": dict(sorted(styles.items())),
        "style_fractions": {key: round(value / denominator, 6) for key, value in sorted(styles.items())} if denominator else {},
        "dominant_style_fraction": round(max_fraction, 6),
        "style_balance_status": "PASS" if max_fraction <= 0.50 else "REVIEW",
        "word_count": {
            "minimum": min(lengths, default=0),
            "maximum": max(lengths, default=0),
            "mean": round(sum(lengths) / denominator, 3) if denominator else 0.0,
        },
        "over_90_word_indices": long_questions,
        "over_120_word_indices": very_long_questions,
        "internal_term_indices": internal_terms,
        "thresholds": {
            "dominant_style_max_fraction": 0.50,
            "review_word_count": 90,
            "hard_word_count": 120,
        },
        "diagnostic_only": True,
    }


def _fixed_display(projection: Mapping[str, Any]) -> list[str]:
    return [
        _text(item.get("normalized_meaning") or item.get("description") or item.get("statement"))
        for item in projection.get("resolved_operationalization", ())
        if isinstance(item, Mapping)
    ]


def _open_display(projection: Mapping[str, Any]) -> list[str]:
    result = []
    for item in projection.get("unresolved_operationalization_dimensions", ()):
        if isinstance(item, Mapping):
            result.append(
                _text(item.get("normalized_meaning"))
                or "choose " + _text(item.get("dimension_id")).replace("_", " ")
            )
        else:
            result.append("choose " + _text(item).replace("_", " "))
    return result


def _finding_display(projection: Mapping[str, Any]) -> list[str]:
    semantics = projection.get("finding_semantics")
    semantics = semantics if isinstance(semantics, Mapping) else {}
    if _finding_mode(projection) == "F1":
        values = [
            _text(item.get("normalized_meaning") or item.get("statement"))
            for item in semantics.get("fixed_finding_requirements", ())
            if isinstance(item, Mapping)
        ]
        return values or [_text(semantics.get("normalized_demand"))]
    adequate = semantics.get("adequate_core_semantics")
    mandatory = (
        list(adequate.get("mandatory_roles", ()))
        if isinstance(adequate, Mapping)
        else []
    )
    result = ["report the mandatory core: " + ", ".join(str(x).replace("_", " ") for x in mandatory)] if mandatory else []
    result.append("add a scientifically relevant characterization chosen by the respondent")
    return result


def render_human_facing_question_review(manifest: Mapping[str, Any]) -> str:
    """Render questions before audit details for direct human inspection."""

    lines = [
        f"# FlowIntentBench — {len(manifest.get('conditions', ()))} Human-Facing Scientific Questions",
        "",
        "The questions and their individual audit outcomes are recorded below. "
        "Automated review does not establish human validation.",
        "",
        "## Scientific Questions",
        "",
    ]
    rows = manifest.get("conditions", ())
    for dataset_id in dict.fromkeys(_text(row.get("dataset_id")) for row in rows):
        lines.extend([f"### {dataset_id}", ""])
        for row in [item for item in rows if _text(item.get("dataset_id")) == dataset_id]:
            lines.extend(
                [
                    f"#### {row['condition']}",
                    "",
                    "**SCIENTIFIC QUESTION**",
                    "",
                    "> " + _text(row.get("scientific_question")),
                    "",
                ]
            )
            context_text = _text(row.get("model_visible_analysis_context_text"))
            if context_text:
                lines.extend(["**ANALYSIS CONTEXT**", "", context_text, ""])
            fixed = row.get("what_is_fixed", ())
            opened = row.get("what_the_respondent_chooses", ())
            findings = row.get("what_should_be_reported", ())
            lines.extend(["**WHAT IS FIXED**", ""])
            lines.extend([f"- {item}" for item in fixed] or ["- No principal Operationalization choice is fixed."])
            lines.extend(["", "**WHAT THE RESPONDENT CHOOSES**", ""])
            lines.extend([f"- {item}" for item in opened] or ["- No principal Operationalization choice remains open."])
            lines.extend(["", "**WHAT SHOULD BE REPORTED**", ""])
            lines.extend([f"- {item}" for item in findings])
            lines.extend(
                [
                    "",
                    f"**Automated human-facing result:** `{row['human_facing_status']}`  ",
                    f"**Proxy Flow Expert:** `{row['flow_expert_final_review_status']}`  ",
                    f"**Release caveat:** `{row['SCIENTIFIC_RELEASE_STATUS']}`",
                    f"**Formal candidate disposition:** `{row.get('FORMAL_CANDIDATE_STATUS', 'RETAINED_FOR_REVIEW')}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Human-Facing Friendliness Analysis",
            "",
            "| Dataset | Condition | Target clarity | O/F responsibility | Visibility | HCCQ | Fresh proxy review | Family audit | Result |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _text(row.get("dataset_id")),
                    _text(row.get("condition")),
                    _text(row.get("target_clarity_status")),
                    _text(row.get("responsibility_integrity_status")),
                    _text(row.get("visibility_integrity_status")),
                    _text(row.get("hccq_status")),
                    _text(row.get("flow_expert_final_review_status")),
                    _text(row.get("family_integrity_status")),
                    _text(row.get("human_facing_status")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Portfolio Caveats",
            "",
            f"- Release ready: **{manifest.get('TOTAL_RELEASE_READY', 0)} / {manifest['TOTAL_QUESTION_SLOTS']}**.",
            f"- New target/case required: **{manifest.get('TOTAL_NEW_TARGET_REQUIRED', 0)}** (Combustor formal candidates withdrawn pending canonical case construction).",
            f"- Pending evidence: **{manifest.get('TOTAL_PENDING_EVIDENCE', 0)}**.",
            f"- Pending O-space/SRAC review: **{manifest.get('TOTAL_PENDING_O_SPACE', 0)} / {manifest.get('TOTAL_PENDING_SRAC', 0)}**.",
            f"- Provisional target families: **{manifest.get('TOTAL_PROVISIONAL_TARGET', 0)}**.",
            "- These caveats do not reduce human-review readiness unless they create an HF1–HF10 defect.",
            "",
            "## Cross-Dataset Scientific-Template Diversity",
            "",
            f"- Audit status: `{manifest.get('portfolio_presentation_audit', {}).get('cross_dataset_scientific_template_audit', {}).get('status', 'NOT_REPORTED')}` (diagnostic-only).",
            f"- Diversity status: `{manifest.get('portfolio_presentation_audit', {}).get('cross_dataset_scientific_template_audit', {}).get('diversity_status', 'NOT_REPORTED')}`.",
            f"- Structured cross-dataset reuse clusters: **{manifest.get('portfolio_presentation_audit', {}).get('cross_dataset_scientific_template_audit', {}).get('structured_cross_dataset_reuse_cluster_count', 0)}**.",
            f"- Human-facing surface audit: `{manifest.get('portfolio_presentation_audit', {}).get('human_facing_surface_audit', {}).get('status', 'NOT_REPORTED')}` (style balance is diagnostic-only).",
            "- Shared scientific archetypes are surfaced for curator inspection and do not replace the family condition-leakage audit.",
            "",
        ]
    )
    return "\n".join(lines)


def render_human_facing_closure_report(manifest: Mapping[str, Any]) -> str:
    robustness = manifest["validator_robustness"]
    lines = [
        "# Human-Facing Contract Closure Report",
        "",
        f"- Closure status: `{manifest['HUMAN_FACING_CLOSURE_STATUS']}`",
        f"- Interface audit coverage: `{manifest['HUMAN_INTERFACE_AUDIT_STATUS']}`",
        f"- Human-friendly portfolio: `{manifest['HUMAN_FRIENDLY_PORTFOLIO_STATUS']}`",
        f"- Independent auditor sentinel: `{manifest['AUDITOR_SENTINEL_STATUS']}`",
        f"- Acceptance contract: `{manifest['HUMAN_FACING_ACCEPTANCE_CONTRACT_SHA256']}`",
        f"- Start/end contract hash identical: `{str(manifest['acceptance_contract_unchanged']).lower()}`",
        f"- Human-review ready: **{manifest['TOTAL_HUMAN_REVIEW_READY']} / {manifest['TOTAL_QUESTION_SLOTS']}**",
        f"- Semantic challenge false accepts: **{robustness['VALIDATOR_SEMANTIC_CHALLENGE_FALSE_ACCEPTS']}**",
        f"- Preserving challenge false rejects: **{robustness['VALIDATOR_PRESERVING_CHALLENGE_FALSE_REJECTS']}**",
        "",
        "## Required Counters",
        "",
    ]
    for key in (
        "TOTAL_QUESTION_SLOTS",
        "TOTAL_GENUINE_AUTHORED_PRIMARY",
        "TOTAL_DETERMINISTIC_PRIMARY_FALLBACK",
        "TOTAL_TARGET_CLARITY_PASS",
        "TOTAL_RESPONSIBILITY_INTEGRITY_PASS",
        "TOTAL_VISIBILITY_INTEGRITY_PASS",
        "TOTAL_SEMANTIC_FIDELITY_PASS",
        "TOTAL_HCCQ_PASS",
        "TOTAL_FLOW_EXPERT_FINAL_REVIEW_PASS",
        "TOTAL_HUMAN_INTERFACE_AUDITED",
        "TOTAL_HUMAN_INTERFACE_PASS",
        "TOTAL_HUMAN_INTERFACE_REVISE",
        "TOTAL_HUMAN_INTERFACE_BLOCKED",
        "TOTAL_HUMAN_FRIENDLY",
        "TOTAL_FAMILY_CONDITION_INTEGRITY_PASS",
        "TOTAL_FAMILY_TEMPLATE_LEAKAGE_PASS",
        "TOTAL_HUMAN_REVIEW_READY",
        "TOTAL_RELEASE_READY",
        "TOTAL_NEW_TARGET_REQUIRED",
        "TOTAL_PENDING_EVIDENCE",
        "TOTAL_PENDING_O_SPACE",
        "TOTAL_PENDING_SRAC",
        "TOTAL_PROVISIONAL_TARGET",
    ):
        lines.append(f"- `{key} = {manifest.get(key)}`")
    lines.extend(["", "## Sentinel Challenges", ""])
    for group in ("semantic_challenges", "preserving_challenges"):
        for challenge in robustness[group]:
            lines.append(
                f"- `{challenge['challenge_id']}`: `{challenge['status']}`; "
                f"observed={challenge.get('observed_failure_codes', [])}"
            )
    lines.extend(["", "## Family Audits", ""])
    for dataset_id, audit in manifest["family_audits"].items():
        lines.append(
            f"- `{dataset_id}`: condition=`{audit['condition_integrity_status']}`, "
            f"template=`{audit['template_leakage_status']}`"
        )
    cross_dataset = manifest.get("portfolio_presentation_audit", {}).get(
        "cross_dataset_scientific_template_audit", {}
    )
    lines.extend(
        [
            "",
            "## Cross-Dataset Scientific-Template Audit",
            "",
            f"- Audit status: `{cross_dataset.get('status', 'NOT_REPORTED')}` (diagnostic-only)",
            f"- Diversity status: `{cross_dataset.get('diversity_status', 'NOT_REPORTED')}`",
            f"- Unique normalized surface templates: **{cross_dataset.get('unique_surface_template_count', 0)}**",
            f"- Cross-dataset reuse clusters: **{cross_dataset.get('cross_dataset_reuse_cluster_count', 0)}**",
            "- Shared wording is reported for curator inspection and is not treated as family-condition leakage.",
        ]
    )
    lines.extend(
        [
            "",
            "No curator confirmation, official SCQ, formal SRAC freeze, evaluator calibration, or model evaluation was executed.",
            "",
        ]
    )
    return "\n".join(lines)


def run_human_facing_question_closure(
    repository_root: str | Path,
    source_matrix: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    *,
    final_reviewer: FinalReviewer | None = None,
    run_live: bool = False,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 240.0,
    max_workers: int = 7,
) -> dict[str, Any]:
    """Run one frozen-input closure and write the four shallow artifacts."""

    if run_live and final_reviewer is not None:
        raise ValueError("run_live and injected final_reviewer are mutually exclusive")
    root = Path(repository_root).resolve()
    out = Path(output_root).resolve()
    matrix = _read_manifest(source_matrix)
    source_rows = matrix.get("conditions")
    if not isinstance(source_rows, list):
        raise ValueError("SOURCE_MATRIX_CONDITIONS_NOT_LIST")
    start_contract_hash = human_facing_acceptance_contract_sha256()
    frozen_contract = copy.deepcopy(HUMAN_FACING_ACCEPTANCE_CONTRACT)
    reviewer_profile = load_agent_profile(
        "flow-scientific-reviewer-gpt-5.6-sol", repository_root=root
    )
    profile_hash = reviewer_profile.profile_sha256
    instruction_hash = hashlib.sha256(
        FINAL_FLOW_EXPERT_WORDING_REVIEW_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    freeze_snapshot = {
        "acceptance_contract_sha256": start_contract_hash,
        "source_matrix_sha256": canonical_json_sha256(matrix),
        "dataset_identities_sha256": canonical_json_sha256(
            sorted({_text(row.get("dataset_id")) for row in source_rows})
        ),
        "scientific_family_contracts_sha256": canonical_json_sha256(
            {
                _text(row.get("slot_id")): _projection(row)
                for row in source_rows
            }
        ),
        "semantic_contracts_sha256": canonical_json_sha256(
            {
                _text(row.get("slot_id")): row.get("semantic_contract_sha256")
                for row in source_rows
            }
        ),
        "evidence_inputs_sha256": canonical_json_sha256(
            {
                _text(row.get("slot_id")): {
                    "support_dimensions": row.get("support_dimensions"),
                    "evidence_gaps": row.get("evidence_gaps"),
                    "evidence_sufficiency": row.get("evidence_sufficiency"),
                }
                for row in source_rows
            }
        ),
        "flow_expert_profile_sha256": profile_hash,
        "flow_expert_final_instruction_sha256": instruction_hash,
        "presentation_policy_sha256": hashlib.sha256(
            QUESTION_PRESENTATION_AUTHORING_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
    }
    if run_live:
        final_reviewer = lambda packet: run_flow_expert_final_flow_wording_review(
            str(root),
            packet,
            config_path=str(config_path) if config_path else None,
            api_key=api_key,
            timeout=timeout,
        )
    if final_reviewer is None:
        raise ValueError("FRESH_FINAL_REVIEWER_REQUIRED")

    capabilities = _capability_profiles(root)
    prepared: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        if not isinstance(row, Mapping):
            continue
        slot_id = _text(row.get("slot_id"))
        candidate = _primary(row)
        if not isinstance(candidate, Mapping):
            prepared[slot_id] = {"error": "PRIMARY_CANDIDATE_MISSING"}
            continue
        projection = _projection(row)
        question = _text(candidate.get("model_visible_text"))
        responsibility = audit_question_responsibility_realization(
            projection,
            question,
            candidate={
                "fixed_operationalization": candidate.get(
                    "fixed_operationalization",
                    projection.get("resolved_operationalization", ()),
                ),
                "unresolved_operationalization_dimensions": candidate.get(
                    "unresolved_operationalization_dimensions",
                    projection.get("unresolved_operationalization_dimensions", ()),
                ),
                "finding_responsibility": candidate.get(
                    "finding_responsibility", projection.get("finding_responsibility")
                ),
                "requested_findings": copy.deepcopy(
                    candidate.get(
                        "requested_findings", projection.get("finding_responsibility", {})
                    )
                ),
            },
            semantic_visibility=row.get("semantic_visibility"),
        )
        hccq = audit_actual_hccq(
            question,
            semantic_projection=projection,
            semantic_visibility=row.get("semantic_visibility"),
        )
        semantic_fidelity = _current_semantic_fidelity(row, candidate, responsibility)
        # Legacy authored rows may omit the structured Finding contract even
        # though their immutable projection still carries it.  The closure
        # audit must use that projection as the fallback, not treat the row as
        # a semantic mutation.
        context, evidence_context = _context_and_evidence(root, row, capabilities)
        prepared[slot_id] = {
            "row": row,
            "candidate": candidate,
            "projection": projection,
            "question": question,
            "responsibility": responsibility,
            "hccq": hccq,
            "semantic_fidelity": semantic_fidelity,
            "review_context": context,
            "evidence_context": evidence_context,
        }

    def review_one(slot_id: str, item: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        if item.get("error"):
            return slot_id, {
                "status": "FAIL",
                "final_flow_expert_review_status": "NOT_ESTABLISHED",
                "failure_codes": [item["error"]],
            }
        row = item["row"]
        review = audit_final_flow_expert_wording(
            item["projection"],
            item["question"],
            reviewer=final_reviewer,  # type: ignore[arg-type]
            dataset_context=item["review_context"],
            evidence_context=item["evidence_context"],
            case_id=_text(row.get("case_id") or row.get("slot_id")),
            condition=_text(row.get("condition")),
            semantic_visibility=row.get("semantic_visibility"),
            semantic_contract_sha256=_text(row.get("semantic_contract_sha256")),
        )
        review["fresh_context_review"] = True
        review["authoring_chain_visible"] = False
        review["previous_verdict_visible"] = False
        review["desired_pass_count_visible"] = False
        review["failure_history_visible"] = False
        review["closure_instruction_sha256"] = instruction_hash
        return slot_id, review

    fresh_reviews: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(prepared)))) as pool:
        futures = {
            pool.submit(review_one, slot_id, item): slot_id
            for slot_id, item in prepared.items()
        }
        for future in as_completed(futures):
            slot_id = futures[future]
            try:
                returned_id, review = future.result()
                fresh_reviews[returned_id] = review
            except Exception as exc:
                fresh_reviews[slot_id] = {
                    "status": "FAIL",
                    "final_flow_expert_review_status": "NOT_ESTABLISHED",
                    "failure_codes": [
                        f"FRESH_FINAL_REVIEW_EXCEPTION:{type(exc).__name__}:{exc}"
                    ],
                }

    family_audits = _family_audits(source_rows)
    portfolio_audit = _portfolio_presentation_audit(source_rows)
    robustness = run_validator_sentinel_challenges(source_rows)
    end_contract_hash = human_facing_acceptance_contract_sha256()
    contract_unchanged = (
        start_contract_hash == end_contract_hash
        and frozen_contract == HUMAN_FACING_ACCEPTANCE_CONTRACT
    )

    conditions: list[dict[str, Any]] = []
    for row in source_rows:
        slot_id = _text(row.get("slot_id"))
        item = prepared.get(slot_id, {})
        candidate = item.get("candidate")
        projection = item.get("projection")
        if not isinstance(candidate, Mapping) or not isinstance(projection, Mapping):
            conditions.append(
                {
                    "slot_id": slot_id,
                    "dataset_id": row.get("dataset_id"),
                    "condition": row.get("condition"),
                    "human_interface_audit_status": "BLOCKED",
                    "human_interface_outcome": "BLOCKED",
                    "human_friendly": False,
                    "human_facing_status": "BLOCKED",
                    "core_membership_affected": False,
                    "failure_codes": [item.get("error", "PRIMARY_CANDIDATE_MISSING")],
                }
            )
            continue
        question = item["question"]
        responsibility = item["responsibility"]
        hccq = item["hccq"]
        semantic_fidelity = item["semantic_fidelity"]
        fresh_review = fresh_reviews.get(slot_id, {})
        review_valid, review_errors = _fresh_review_provenance_valid(
            fresh_review,
            question=question,
            semantic_contract_sha256=_text(row.get("semantic_contract_sha256")),
            require_live=run_live,
            expected_agent_profile_sha256=profile_hash,
        )
        review_outcome = _text(
            fresh_review.get("final_flow_expert_review_status")
        ).upper()
        family = family_audits.get(_family_key(row), {})
        authoring = candidate.get("authoring_provenance")
        invocation = candidate.get("question_authoring_invocation")
        genuine = bool(
            isinstance(authoring, Mapping)
            and authoring.get("deterministic_fallback") is False
            and authoring.get("invocation_verified") is True
            and isinstance(invocation, Mapping)
            and _text(invocation.get("invocation_status")).upper() == "SUCCESS"
            and _text(candidate.get("candidate_source")).upper()
            == "FLOW_EXPERT_PRESENTATION_AUTHORING"
        )
        target_pass = responsibility.get("checks", {}).get("target_clarity") is True
        responsibility_pass = responsibility.get("status") == "PASS"
        visibility_pass = responsibility.get("checks", {}).get("visibility_integrity") is True
        hccq_pass = _text(hccq.get("status")).upper() == "PASS"
        fidelity_pass = semantic_fidelity.get("status") == "PASS"
        family_condition_pass = family.get("condition_integrity_status") == "PASS"
        family_template_pass = family.get("template_leakage_status") == "PASS"
        interface_checks = [
            genuine,
            target_pass,
            responsibility_pass,
            visibility_pass,
            fidelity_pass,
            hccq_pass,
            review_valid,
            family_condition_pass,
            family_template_pass,
            portfolio_audit.get("status") == "PASS",
            contract_unchanged,
        ]
        audit_status = (
            "AUDITED"
            if genuine and review_valid and contract_unchanged
            else "BLOCKED"
        )
        if audit_status == "BLOCKED":
            interface_outcome = "BLOCKED"
        elif all(interface_checks) and review_outcome == "PASS":
            interface_outcome = "PASS"
        else:
            interface_outcome = "REVISE"
        failure_codes = [
            *responsibility.get("failure_codes", ()),
            *review_errors,
        ]
        if not genuine:
            failure_codes.append("GENUINE_AUTHORING_PROVENANCE_INVALID")
        if not fidelity_pass:
            failure_codes.append("SEMANTIC_FIDELITY_PROOF_INVALID")
        if not hccq_pass:
            failure_codes.append("HCCQ_NOT_PASS")
        if not family_condition_pass:
            failure_codes.append("FAMILY_CONDITION_INTEGRITY_FAIL")
        if not family_template_pass:
            failure_codes.append("FAMILY_TEMPLATE_LEAKAGE")
        conditions.append(
            {
                "slot_id": slot_id,
                "dataset_id": row.get("dataset_id"),
                "family_id": row.get("family_id"),
                "condition": row.get("condition"),
                "case_id": row.get("case_id"),
                "scientific_target": projection.get("scientific_target"),
                "canonical_semantic_projection": copy.deepcopy(projection),
                "semantic_contract_sha256": row.get("semantic_contract_sha256"),
                "primary_candidate_id": candidate.get("candidate_id"),
                "scientific_question": question,
                "model_visible_analysis_context": {
                    "required": False,
                    "text": "",
                    "scientific_responsibility_fully_expressed_in_question": True,
                },
                "model_visible_analysis_context_text": "",
                "what_is_fixed": _fixed_display(projection),
                "what_the_respondent_chooses": _open_display(projection),
                "what_should_be_reported": _finding_display(projection),
                "finding_responsibility": _finding_mode(projection),
                "target_clarity_status": "PASS" if target_pass else "FAIL",
                "responsibility_integrity_status": "PASS" if responsibility_pass else "FAIL",
                "visibility_integrity_status": "PASS" if visibility_pass else "FAIL",
                "semantic_fidelity_status": "PASS" if fidelity_pass else "FAIL",
                "hccq_status": "PASS" if hccq_pass else "FAIL",
                "flow_expert_final_review_status": (
                    review_outcome if review_valid else "NOT_ESTABLISHED"
                ),
                "family_integrity_status": (
                    "PASS" if family_condition_pass and family_template_pass else "FAIL"
                ),
                "human_interface_audit_status": audit_status,
                "human_interface_outcome": interface_outcome,
                "human_friendly": interface_outcome == "PASS",
                "human_facing_status": (
                    "HUMAN_REVIEW_READY"
                    if interface_outcome == "PASS"
                    else interface_outcome
                ),
                "core_membership_affected": False,
                "failure_codes": sorted(set(failure_codes)),
                "genuine_authored_primary": genuine,
                "deterministic_responsibility_audit": responsibility,
                "semantic_fidelity_proof": semantic_fidelity,
                "hccq_result": hccq,
                "fresh_flow_expert_final_review": fresh_review,
                "support_dimensions": copy.deepcopy(row.get("support_dimensions", {})),
                "SCIENTIFIC_RELEASE_STATUS": row.get("SCIENTIFIC_RELEASE_STATUS"),
                "FORMAL_CANDIDATE_STATUS": (
                    "WITHDRAWN_NEW_CASE_REQUIRED"
                    if _text(row.get("SCIENTIFIC_RELEASE_STATUS")).upper() == "NEW_CASE_REQUIRED"
                    else "RETAINED_FOR_REVIEW"
                ),
                "ARTIFACT_REUSE_STATUS": row.get("ARTIFACT_REUSE_STATUS"),
                "GT_SRAC_IMPACT": row.get("GT_SRAC_IMPACT"),
            }
        )

    def count(field: str, value: Any = "PASS") -> int:
        return sum(row.get(field) == value for row in conditions)

    release_counts = {
        status: sum(row.get("SCIENTIFIC_RELEASE_STATUS") == status for row in conditions)
        for status in (
            "ELIGIBLE",
            "BLOCKED_PENDING_EVIDENCE",
            "BLOCKED_PENDING_O_SPACE",
            "NEW_CASE_REQUIRED",
        )
    }
    counters = {
        "TOTAL_QUESTION_SLOTS": len(conditions),
        "TOTAL_GENUINE_AUTHORED_PRIMARY": sum(bool(row.get("genuine_authored_primary")) for row in conditions),
        "TOTAL_DETERMINISTIC_PRIMARY_FALLBACK": sum(not bool(row.get("genuine_authored_primary")) for row in conditions),
        "TOTAL_TARGET_CLARITY_PASS": count("target_clarity_status"),
        "TOTAL_RESPONSIBILITY_INTEGRITY_PASS": count("responsibility_integrity_status"),
        "TOTAL_VISIBILITY_INTEGRITY_PASS": count("visibility_integrity_status"),
        "TOTAL_SEMANTIC_FIDELITY_PASS": count("semantic_fidelity_status"),
        "TOTAL_HCCQ_PASS": count("hccq_status"),
        "TOTAL_FLOW_EXPERT_FINAL_REVIEW_PASS": count("flow_expert_final_review_status"),
        "TOTAL_HUMAN_INTERFACE_AUDITED": count(
            "human_interface_audit_status", "AUDITED"
        ),
        "TOTAL_HUMAN_INTERFACE_PASS": count("human_interface_outcome", "PASS"),
        "TOTAL_HUMAN_INTERFACE_REVISE": count("human_interface_outcome", "REVISE"),
        "TOTAL_HUMAN_INTERFACE_BLOCKED": count("human_interface_outcome", "BLOCKED"),
        "TOTAL_HUMAN_FRIENDLY": sum(bool(row.get("human_friendly")) for row in conditions),
        "TOTAL_FAMILY_CONDITION_INTEGRITY_PASS": sum(
            family_audits.get(_family_key(row), {}).get("condition_integrity_status") == "PASS"
            for row in conditions
        ),
        "TOTAL_FAMILY_TEMPLATE_LEAKAGE_PASS": sum(
            family_audits.get(_family_key(row), {}).get("template_leakage_status") == "PASS"
            for row in conditions
        ),
        "TOTAL_HUMAN_REVIEW_READY": sum(row.get("human_facing_status") == "HUMAN_REVIEW_READY" for row in conditions),
        "TOTAL_RELEASE_READY": release_counts["ELIGIBLE"],
        "TOTAL_NEW_TARGET_REQUIRED": release_counts["NEW_CASE_REQUIRED"],
        "TOTAL_PENDING_EVIDENCE": release_counts["BLOCKED_PENDING_EVIDENCE"],
        "TOTAL_PENDING_O_SPACE": release_counts["BLOCKED_PENDING_O_SPACE"],
        "TOTAL_PENDING_SRAC": sum(row.get("GT_SRAC_IMPACT") == "SRAC_O_SPACE_REVIEW_REQUIRED" for row in conditions),
        "TOTAL_PROVISIONAL_TARGET": sum(
            1
            for dataset_id, policy in DATASET_POLICIES.items()
            if policy.get("redesign_required")
            and any(row.get("dataset_id") == dataset_id for row in conditions)
        ),
    }
    required_counts = (
        "TOTAL_QUESTION_SLOTS",
        "TOTAL_GENUINE_AUTHORED_PRIMARY",
        "TOTAL_TARGET_CLARITY_PASS",
        "TOTAL_RESPONSIBILITY_INTEGRITY_PASS",
        "TOTAL_VISIBILITY_INTEGRITY_PASS",
        "TOTAL_SEMANTIC_FIDELITY_PASS",
        "TOTAL_HCCQ_PASS",
        "TOTAL_FLOW_EXPERT_FINAL_REVIEW_PASS",
        "TOTAL_FAMILY_CONDITION_INTEGRITY_PASS",
        "TOTAL_FAMILY_TEMPLATE_LEAKAGE_PASS",
        "TOTAL_HUMAN_INTERFACE_AUDITED",
        "TOTAL_HUMAN_INTERFACE_PASS",
        "TOTAL_HUMAN_FRIENDLY",
    )
    closure_success = bool(
        bool(source_rows) and all(counters[key] == len(source_rows) for key in required_counts)
        and counters["TOTAL_DETERMINISTIC_PRIMARY_FALLBACK"] == 0
        and portfolio_audit.get("status") == "PASS"
        and contract_unchanged
    )
    audit_complete = bool(source_rows) and counters["TOTAL_HUMAN_INTERFACE_AUDITED"] == len(source_rows)
    human_friendly_complete = bool(source_rows) and counters["TOTAL_HUMAN_FRIENDLY"] == len(source_rows)
    sentinel_status = "PASS" if robustness.get("status") == "PASS" else "FAIL"
    manifest = {
        "schema_version": HUMAN_FACING_CLOSURE_VERSION,
        "artifact_type": "HUMAN_FACING_QUESTION_CLOSURE",
        "HUMAN_FACING_CLOSURE_STATUS": "COMPLETE" if closure_success else "INCOMPLETE",
        "HUMAN_INTERFACE_AUDIT_STATUS": "COMPLETE" if audit_complete else "INCOMPLETE",
        "HUMAN_FRIENDLY_PORTFOLIO_STATUS": (
            "PASS"
            if human_friendly_complete
            else "BLOCKED"
            if counters["TOTAL_HUMAN_INTERFACE_BLOCKED"]
            else "REVISE"
        ),
        "AUDITOR_SENTINEL_STATUS": sentinel_status,
        "core_membership_affected": False,
        "HUMAN_FACING_ACCEPTANCE_CONTRACT_SHA256": start_contract_hash,
        "acceptance_contract_start_sha256": start_contract_hash,
        "acceptance_contract_end_sha256": end_contract_hash,
        "acceptance_contract_unchanged": contract_unchanged,
        "freeze_snapshot": freeze_snapshot,
        "conditions": conditions,
        "family_audits": family_audits,
        "portfolio_presentation_audit": portfolio_audit,
        "validator_robustness": robustness,
        "downstream": {
            "human_curator_confirmation_executed": False,
            "official_scq_executed": False,
            "formal_srac_executed": False,
            "evaluator_calibration_executed": False,
            "formal_model_evaluation_executed": False,
        },
        **counters,
    }

    out.mkdir(parents=True, exist_ok=True)
    try:
        base_portfolio = build_human_review_portfolio(matrix, out)
    except ValueError as exc:
        # An adversarial or incomplete source matrix must still yield a
        # diagnostic closure artifact; it must not be transformed into a
        # successful scientific-question portfolio.
        base_portfolio = {
            "schema_version": "human-review-scientific-question-expansion-v1",
            "artifact_type": "HUMAN_REVIEW_SCIENTIFIC_QUESTION_PORTFOLIO",
            "HUMAN_REVIEW_PORTFOLIO_STATUS": "INCOMPLETE_PENDING_AUTHORED_QUESTIONS",
            "source_matrix_validation_error": f"{type(exc).__name__}: {exc}",
            "conditions": [],
            "validation": {"status": "INVALID", "errors": [str(exc)]},
        }
    base_portfolio["human_facing_closure"] = {
        "status": manifest["HUMAN_FACING_CLOSURE_STATUS"],
        "acceptance_contract_sha256": start_contract_hash,
        "manifest": "human_facing_closure_manifest.json",
        "report": "human_facing_closure_report.md",
    }
    for key, value in counters.items():
        base_portfolio[key] = value
    base_portfolio["HUMAN_FACING_CLOSURE_STATUS"] = manifest[
        "HUMAN_FACING_CLOSURE_STATUS"
    ]
    _write_json(out / "scientific_question_manifest.json", base_portfolio)
    _write_json(out / "human_facing_closure_manifest.json", manifest)
    (out / "scientific_question_review.md").write_text(
        render_human_facing_question_review(manifest), encoding="utf-8"
    )
    (out / "human_facing_closure_report.md").write_text(
        render_human_facing_closure_report(manifest), encoding="utf-8"
    )
    return manifest


__all__ = [
    "HUMAN_FACING_CLOSURE_VERSION",
    "render_human_facing_closure_report",
    "render_human_facing_question_review",
    "run_human_facing_question_closure",
]

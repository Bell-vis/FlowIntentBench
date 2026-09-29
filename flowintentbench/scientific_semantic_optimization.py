"""Scientific-semantics, presentation, and target-audit integration.

This module is a construction/audit harness.  It derives semantics from the
existing case records, proposes human-facing realizations, and optionally asks
the canonical Flow Expert for an advisory target-stability review.  It never
promotes a case, executes official SCQ, or mutates Ground Truth.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .case_repository import condition_from_case_id, load_case_record
from .flow_expert_portfolio_validation import (
    FLOW_EXPERT_SCIENTIFIC_TARGET_CASE_MATRIX,
    build_flow_expert_scientific_target_packets,
)
from .hccq import audit_family_presentation_leakage, audit_human_presentation
from .question_presentation import (
    realize_human_questions,
    validate_question_semantic_fidelity,
)
from .scientific_artifact_binding import (
    WORDING_ONLY,
    artifact_sha256,
    evaluate_scientific_dependency_invalidation,
)
from .scientific_expert_review import (
    run_scientific_expert_review,
    validate_scientific_expert_output,
)
from .scientific_semantics import (
    build_scientific_semantic_projection,
    semantic_contract_sha256,
)


SCIENTIFIC_SEMANTIC_OPTIMIZATION_VERSION = (
    "scientific-semantics-presentation-gt-binding-v1"
)

ReviewInvoker = Callable[[Mapping[str, Any], str], Mapping[str, Any]]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _current_question(record: Mapping[str, Any]) -> str:
    case_input = record.get("case_input")
    if isinstance(case_input, Mapping):
        return str(case_input.get("scientific_question", ""))
    return ""


def _candidate_rows(
    record: Mapping[str, Any],
    projection: Mapping[str, Any],
    semantic_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = str(record.get("case_id", ""))
    condition = str(record.get("condition") or condition_from_case_id(case_id) or "")
    realization = realize_human_questions(
        projection,
        case_id=case_id,
        condition=condition,
        candidate_count=3,
        semantic_contract_sha256=semantic_sha256,
    )
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate in realization["candidates"]:
        fidelity = validate_question_semantic_fidelity(
            projection,
            candidate,
            semantic_contract_sha256=semantic_sha256,
        )
        hccq = audit_human_presentation(candidate)
        if fidelity["status"] == "PASS" and selected is None:
            selected = dict(candidate)
        if fidelity["status"] != "PASS":
            candidate_status = "SEMANTICALLY_INVALID_REALIZATION"
        elif hccq["status"] == "PASS":
            candidate_status = "READY_FOR_HUMAN_SELECTION"
        else:
            candidate_status = "REQUIRES_PRESENTATION_REVISION"
        rows.append(
            {
                "candidate": candidate,
                "semantic_fidelity": fidelity,
                "hccq_presentation_audit": hccq,
                "candidate_status": candidate_status,
            }
        )
    if selected is None:
        raise ValueError(f"no semantically faithful realization for {case_id}")
    return (
        {
            **realization,
            "candidates": rows,
            "selected_for_advisory_review_candidate_id": selected["candidate_id"],
            "selected_for_advisory_review_is_canonical": False,
        },
        selected,
    )


def build_semantic_presentation_portfolio(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Derive the 12-case semantic and presentation portfolio without agents."""

    root = Path(repository_root).resolve()
    case_rows: list[dict[str, Any]] = []
    question_overrides: dict[str, str] = {}
    selected_by_family: dict[str, dict[str, Any]] = {}
    all_fidelity_pass = True
    for family_id, case_ids in FLOW_EXPERT_SCIENTIFIC_TARGET_CASE_MATRIX.items():
        selected_by_family[family_id] = {}
        for case_id in case_ids:
            record = load_case_record(root, case_id)
            projection = build_scientific_semantic_projection(record)
            semantic_sha256 = semantic_contract_sha256(projection)
            realization, selected = _candidate_rows(
                record, projection, semantic_sha256
            )
            condition = str(record.get("condition") or condition_from_case_id(case_id) or "")
            question_overrides[case_id] = str(selected["model_visible_text"])
            selected_by_family[family_id][condition] = selected
            fidelity_rows = [
                row["semantic_fidelity"] for row in realization["candidates"]
            ]
            all_fidelity_pass = all_fidelity_pass and all(
                row["status"] == "PASS" for row in fidelity_rows
            )
            dependency = evaluate_scientific_dependency_invalidation(WORDING_ONLY)
            case_dir = Path(str(record.get("case_dir", "")))
            contract_path = case_dir / "scientific_evaluation_contract.json"
            binding_path = case_dir / "scientific_artifact_binding.json"
            ground_truth = record.get("ground_truth")
            has_ground_truth = isinstance(ground_truth, Mapping) and bool(ground_truth)
            case_rows.append(
                {
                    "case_id": case_id,
                    "dataset_id": str(record.get("dataset_id", "")),
                    "family_id": family_id,
                    "condition": condition,
                    "scientific_target": projection["scientific_target"],
                    "scientific_scope": projection["scientific_scope"],
                    "fixed_operationalization": projection[
                        "resolved_operationalization"
                    ],
                    "unresolved_operationalization": projection[
                        "unresolved_operationalization_dimensions"
                    ],
                    "finding_responsibility": projection[
                        "finding_responsibility"
                    ],
                    "finding_semantics": projection["finding_semantics"],
                    "canonical_semantic_projection": projection,
                    "semantic_contract_sha256": semantic_sha256,
                    "source_human_facing_question": _current_question(record),
                    "human_facing_question_realizations": realization,
                    "wording_dependency_result": dependency,
                    "artifact_binding_state": dependency[
                        "artifact_binding_state"
                    ],
                    "scientific_artifact_status": dependency[
                        "scientific_artifact_status"
                    ],
                    "presentation_binding_status": dependency[
                        "presentation_status"
                    ],
                    "ground_truth_present": has_ground_truth,
                    "ground_truth_sha256": (
                        artifact_sha256(ground_truth) if has_ground_truth else None
                    ),
                    "scientific_evaluation_contract_present": contract_path.is_file(),
                    "scientific_artifact_binding_present": binding_path.is_file(),
                    "formal_binding_readiness": (
                        "BOUND"
                        if contract_path.is_file() and binding_path.is_file()
                        else "PENDING_FROZEN_SCIENTIFIC_EVALUATION_CONTRACT"
                    ),
                    "canonical_case_mutated": False,
                    "official_scq_executed": False,
                }
            )
    family_leakage = {
        family_id: audit_family_presentation_leakage(presentations)
        for family_id, presentations in selected_by_family.items()
    }
    return {
        "schema_version": SCIENTIFIC_SEMANTIC_OPTIMIZATION_VERSION,
        "case_count": len(case_rows),
        "family_count": len(selected_by_family),
        "cases": case_rows,
        "question_overrides_for_advisory_review": question_overrides,
        "family_presentation_leakage_audits": family_leakage,
        "SEMANTIC_PROJECTION_STATUS": "PASS",
        "PRESENTATION_SEMANTIC_FIDELITY_STATUS": (
            "PASS" if all_fidelity_pass else "REVISE"
        ),
        "HCCQ_CORE_MEMBERSHIP_EFFECT_COUNT": 0,
        "CANONICAL_CASE_MUTATION_COUNT": 0,
        "OFFICIAL_SCQ_EXECUTED_COUNT": 0,
    }


def _normalize_review_artifact(
    raw: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[str, Any]:
    if isinstance(raw.get("validated_review"), Mapping):
        return dict(raw)
    parsed = raw.get("parsed_result")
    if not isinstance(parsed, Mapping) and isinstance(raw.get("condition_reviews"), list):
        parsed = raw
    if not isinstance(parsed, Mapping):
        return {
            "status": "FAILED",
            "invocation_status": str(
                raw.get("invocation_status", "INFRASTRUCTURE_INVALID")
            ),
            "parsed_result": {},
            "validated_review": {
                "status": "NOT_EVALUATED",
                "reason_codes": ["INVOCATION_NOT_SUCCESSFUL"],
                "condition_reviews": [],
                "question_reviews": [],
            },
        }
    validated = validate_scientific_expert_output(parsed, packet)
    return {
        "status": "SUCCESS" if validated["status"] == "PASS" else "INCOMPLETE",
        "invocation_status": str(raw.get("invocation_status", "SUCCESS")),
        "parsed_result": dict(parsed),
        "validated_review": validated,
        "call": raw.get("call"),
        "live_model_calls": raw.get("live_model_calls", False),
    }


def _target_stability_row(
    review: Mapping[str, Any] | None,
    question_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(review, Mapping):
        return {
            "target_stability_judgment": "NOT_EVALUATED",
            "scientific_reuse_advice": "BLOCKED_PENDING_SCIENTIFIC_REVIEW",
            "human_scientific_selection_required": True,
            "reason_codes": ["FLOW_EXPERT_NOT_RUN"],
        }
    reasons: list[str] = []
    if review.get("scientific_target_supported") is False:
        reasons.append("SCIENTIFIC_TARGET_NOT_SUPPORTED")
    if review.get("unresolved_o_is_legitimate_scientific_choice") is False:
        reasons.append("TARGET_INVARIANCE_ACROSS_OPEN_O_NOT_ESTABLISHED")
    if review.get("fixed_o_supported") is False:
        reasons.append("FIXED_O_NOT_SUPPORTED")
    if review.get("materialization_scientifically_meaningful") is False:
        reasons.append("MATERIALIZATION_NOT_SCIENTIFICALLY_MEANINGFUL")
    recommendation = str(review.get("eligibility_recommendation", ""))
    question_recommendation = (
        str(question_review.get("recommendation", ""))
        if isinstance(question_review, Mapping)
        else ""
    )
    if recommendation == "MORE_EVIDENCE_REQUIRED":
        reasons.append("MORE_EVIDENCE_REQUIRED")
    if question_recommendation:
        reasons.append(f"QUESTION_RECOMMENDATION_{question_recommendation}")

    # Review readiness is advisory and orthogonal to artifact identity. A new
    # case is warranted only when the reviewer explicitly changes the target.
    # Evidence and O-space deficiencies block scientific reuse without claiming
    # that any semantic hash or bound artifact changed.
    if question_recommendation == "REVISE_TARGET":
        judgment = "SCIENTIFIC_TARGET_REVISION_RECOMMENDED"
        reuse_advice = "NEW_CASE_REQUIRED"
    elif question_recommendation == "REVISE_OPERATIONALIZATION_SPACE":
        judgment = "OPERATIONALIZATION_SPACE_REQUIRES_REVIEW"
        reuse_advice = "BLOCKED_PENDING_O_SPACE_REVIEW"
    elif recommendation == "MORE_EVIDENCE_REQUIRED":
        judgment = "TARGET_SUPPORT_NOT_ESTABLISHED"
        reuse_advice = "BLOCKED_PENDING_EVIDENCE"
    elif review.get("unresolved_o_is_legitimate_scientific_choice") is False:
        judgment = "OPERATIONALIZATION_SPACE_REQUIRES_REVIEW"
        reuse_advice = "BLOCKED_PENDING_O_SPACE_REVIEW"
    elif (
        review.get("scientific_target_supported") is False
        or review.get("fixed_o_supported") is False
        or review.get("materialization_scientifically_meaningful") is False
    ):
        judgment = "SCIENTIFIC_TARGET_OR_O_SPACE_REQUIRES_HUMAN_REVIEW"
        reuse_advice = "BLOCKED_PENDING_SCIENTIFIC_REVIEW"
    elif all(
        review.get(field) is True
        for field in (
            "scientific_target_supported",
            "fixed_o_supported",
            "unresolved_o_is_legitimate_scientific_choice",
            "materialization_scientifically_meaningful",
        )
    ):
        judgment = "TARGET_STABLE_WITHIN_REVIEWED_CONDITION"
        reuse_advice = "REUSE_ALLOWED"
    else:
        judgment = "INDETERMINATE"
        reuse_advice = "BLOCKED_PENDING_SCIENTIFIC_REVIEW"
        reasons.append("INCOMPLETE_ATOMIC_TARGET_REVIEW")
    return {
        "target_stability_judgment": judgment,
        "scientific_reuse_advice": reuse_advice,
        "human_scientific_selection_required": reuse_advice != "REUSE_ALLOWED",
        "reason_codes": sorted(set(reasons)),
        "eligibility_recommendation": recommendation or None,
        "question_recommendation": question_recommendation or None,
        "question_review_rationale": (
            str(question_review.get("rationale", ""))
            if isinstance(question_review, Mapping)
            else ""
        ),
        "scientific_ambiguities": list(
            review.get("scientific_ambiguities", ()) or ()
        ),
        "rationale": str(review.get("rationale", "")),
        "evidence_ids": list(review.get("evidence_ids", ()) or ()),
        "source_ids": list(review.get("source_ids", ()) or ()),
    }


def _run_reviews(
    root: Path,
    packets: Sequence[Mapping[str, Any]],
    *,
    reviewer: ReviewInvoker | None,
    run_live: bool,
    config_path: str | Path | None,
    api_key: str | None,
    timeout: float,
    max_workers: int,
) -> dict[str, dict[str, Any]]:
    if run_live and reviewer is not None:
        raise ValueError("run_live and reviewer are mutually exclusive")
    if not run_live and reviewer is None:
        return {}

    def execute(packet: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        family = str(packet.get("family_identity", {}).get("family_id", ""))
        item_id = f"scientific-target-audit::{family}"
        if reviewer is not None:
            raw = reviewer(packet, item_id)
        else:
            raw = run_scientific_expert_review(
                root,
                packet,
                config_path=config_path,
                api_key=api_key,
                timeout=timeout,
                run_id=item_id,
            )
        return family, _normalize_review_artifact(dict(raw), packet)

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(packets)))) as pool:
        futures = {pool.submit(execute, packet): packet for packet in packets}
        for future in as_completed(futures):
            packet = futures[future]
            family = str(packet.get("family_identity", {}).get("family_id", ""))
            try:
                result_family, artifact = future.result()
                results[result_family] = artifact
            except Exception as exc:
                results[family] = {
                    "status": "FAILED",
                    "invocation_status": "INFRASTRUCTURE_INVALID",
                    "parsed_result": {},
                    "validated_review": {
                        "status": "NOT_EVALUATED",
                        "reason_codes": ["REVIEWER_EXCEPTION"],
                        "condition_reviews": [],
                        "question_reviews": [],
                    },
                    "call": {"error": f"{type(exc).__name__}: {exc}"},
                }
    return results


def _markdown_report(result: Mapping[str, Any]) -> str:
    status = result["status"]
    lines = [
        "# Scientific semantics / presentation / GT binding audit",
        "",
        "This is a construction and advisory scientific-review exercise. It does not promote a case, execute official SCQ, or rewrite Ground Truth.",
        "",
        "## Status",
        "",
    ]
    for key, value in status.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Condition results",
            "",
            "| Family | Condition | Semantic hash | Fidelity | HCCQ | Flow Expert target audit | Scientific reuse advice |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in result["conditions"]:
        lines.append(
            "| `{family_id}` | `{condition}` | `{semantic_contract_sha256}` | "
            "`{fidelity}` | `{hccq}` | `{target}` | `{reuse}` |".format(
                family_id=row["family_id"],
                condition=row["condition"],
                semantic_contract_sha256=row["semantic_contract_sha256"][:12],
                fidelity=row["semantic_fidelity_status"],
                hccq=row["hccq_presentation_status"],
                target=row["flow_expert_target_stability"][
                    "target_stability_judgment"
                ],
                reuse=row["flow_expert_target_stability"]["scientific_reuse_advice"],
            )
        )
    lines.extend(
        [
            "",
            "Artifact identity and scientific reuse advice are independent: evidence or O-space review may block reuse without making an unchanged GT/SRAC artifact hash-stale.",
            "",
            f"Candidate scientific redesign requests: `{len(result['candidate_scientific_cases'])}` (human decision required; no automatic promotion).",
            "",
            "## Condition contracts and presentation candidates",
            "",
        ]
    )
    for row in result["conditions"]:
        audit = row["flow_expert_target_stability"]
        realization = row["human_facing_question_realizations"]
        lines.extend(
            [
                f"### {row['family_id']} / {row['condition']}",
                "",
                f"- Scientific target: {row['scientific_target']}",
                f"- Scientific scope: {row['scientific_scope']}",
                "- Fixed O: `" + json.dumps(
                    row["fixed_operationalization"],
                    ensure_ascii=False,
                    sort_keys=True,
                ) + "`",
                "- Unresolved O: `" + json.dumps(
                    row["unresolved_operationalization"],
                    ensure_ascii=False,
                    sort_keys=True,
                ) + "`",
                "- Finding responsibility: `" + json.dumps(
                    row["finding_responsibility"],
                    ensure_ascii=False,
                    sort_keys=True,
                ) + "`",
                f"- Semantic hash: `{row['semantic_contract_sha256']}`",
                f"- Fidelity / HCCQ: `{row['semantic_fidelity_status']}` / `{row['hccq_presentation_status']}`",
                f"- Artifact binding: `{row['artifact_binding_state']}` (science `{row['scientific_artifact_status']}`, presentation `{row['presentation_binding_status']}`)",
                f"- Flow Expert: `{audit['target_stability_judgment']}`; reuse advice `{audit['scientific_reuse_advice']}`; question recommendation `{audit.get('question_recommendation') or 'NONE'}`",
                "- Human-facing candidates:",
            ]
        )
        for candidate_row in realization["candidates"]:
            candidate = candidate_row["candidate"]
            lines.append(
                f"  - `{candidate['candidate_id']}`: {candidate['model_visible_text'].replace(chr(10), ' ')}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_scientific_semantic_optimization(
    repository_root: str | Path,
    output_root: str | Path,
    *,
    reviewer: ReviewInvoker | None = None,
    run_live: bool = False,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    max_workers: int = 3,
) -> dict[str, Any]:
    """Build all outputs and optionally execute three parallel live reviews."""

    root = Path(repository_root).resolve()
    out = Path(output_root).resolve()
    portfolio = build_semantic_presentation_portfolio(root)
    packets = build_flow_expert_scientific_target_packets(
        root,
        question_overrides=portfolio["question_overrides_for_advisory_review"],
    )
    reviews = _run_reviews(
        root,
        packets,
        reviewer=reviewer,
        run_live=run_live,
        config_path=config_path,
        api_key=api_key,
        timeout=timeout,
        max_workers=max_workers,
    )
    by_case = {row["case_id"]: row for row in portfolio["cases"]}
    condition_rows: list[dict[str, Any]] = []
    candidate_cases: list[dict[str, Any]] = []
    for packet in packets:
        family = str(packet["family_identity"]["family_id"])
        artifact = reviews.get(family)
        validated = (
            artifact.get("validated_review", {})
            if isinstance(artifact, Mapping)
            else {}
        )
        review_rows = {
            str(item.get("condition")): item
            for item in validated.get("condition_reviews", ())
            if isinstance(item, Mapping)
        }
        question_review_rows = {
            str(item.get("condition")): item
            for item in validated.get("question_reviews", ())
            if isinstance(item, Mapping)
        }
        for condition in packet["conditions"]:
            case_id = str(condition["case_id"])
            base = by_case[case_id]
            candidate_rows = base["human_facing_question_realizations"]["candidates"]
            selected_id = base["human_facing_question_realizations"][
                "selected_for_advisory_review_candidate_id"
            ]
            selected_row = next(
                row
                for row in candidate_rows
                if row["candidate"]["candidate_id"] == selected_id
            )
            target_row = _target_stability_row(
                review_rows.get(str(condition["condition"])),
                question_review_rows.get(str(condition["condition"])),
            )
            row = {
                **base,
                "semantic_fidelity_status": selected_row["semantic_fidelity"][
                    "status"
                ],
                "hccq_presentation_status": selected_row[
                    "hccq_presentation_audit"
                ]["status"],
                "flow_expert_target_stability": target_row,
                "flow_expert_review_validation_status": validated.get(
                    "status", "NOT_RUN"
                ),
            }
            condition_rows.append(row)
            if target_row["scientific_reuse_advice"] == "NEW_CASE_REQUIRED":
                candidate_cases.append(
                    {
                        "candidate_case_id": f"{case_id}__target_redesign_candidate",
                        "source_case_id": case_id,
                        "family_id": family,
                        "status": "CANDIDATE_REQUIRES_HUMAN_SCIENTIFIC_AUTHORING",
                        "proposed_scientific_target": None,
                        "flow_expert_rationale": target_row["rationale"],
                        "scientific_ambiguities": target_row[
                            "scientific_ambiguities"
                        ],
                        "automatic_promotion_performed": False,
                    }
                )

    requested_review = run_live or reviewer is not None
    completed = len(reviews) == len(packets) and all(
        item.get("invocation_status") == "SUCCESS"
        and item.get("validated_review", {}).get("status") == "PASS"
        for item in reviews.values()
    )
    hccq_revisions = sum(
        row["hccq_presentation_status"] != "PASS" for row in condition_rows
    )
    leakage_observed = sum(
        audit.get("status") != "PASS"
        for audit in portfolio["family_presentation_leakage_audits"].values()
    )
    status = {
        "STATIC_IMPLEMENTATION_STATUS": "PASS",
        "SEMANTIC_PROJECTION_STATUS": portfolio["SEMANTIC_PROJECTION_STATUS"],
        "PRESENTATION_SEMANTIC_FIDELITY_STATUS": portfolio[
            "PRESENTATION_SEMANTIC_FIDELITY_STATUS"
        ],
        "HCCQ_PRESENTATION_STATUS": (
            "REVISIONS_OBSERVED" if hccq_revisions else "PASS"
        ),
        "FAMILY_PRESENTATION_LEAKAGE_STATUS": (
            "LEAKAGE_OBSERVED" if leakage_observed else "PASS"
        ),
        "FLOW_EXPERT_LIVE_EXECUTION_STATUS": (
            "COMPLETE"
            if requested_review and completed
            else "INCOMPLETE"
            if requested_review
            else "NOT_RUN"
        ),
        "FLOW_EXPERT_SCIENTIFIC_AUTHORITY": "ADVISORY_ONLY",
        "GT_SRAC_DEPENDENCY_IMPLEMENTATION_STATUS": "PASS",
        "FORMAL_ARTIFACT_BINDING_STATUS": (
            "PENDING_FROZEN_SCIENTIFIC_EVALUATION_CONTRACT"
        ),
        "CANONICAL_CASE_MUTATION_STATUS": "NOT_PERFORMED",
        "OFFICIAL_SCQ_EXECUTED_COUNT": 0,
        "FORMAL_MODEL_EXPERIMENT_EXECUTED_COUNT": 0,
        "HUMAN_SCIENTIFIC_SELECTION_STATUS": "PENDING",
    }
    result = {
        "schema_version": SCIENTIFIC_SEMANTIC_OPTIMIZATION_VERSION,
        "execution_mode": (
            "LIVE_FLOW_EXPERT"
            if run_live
            else "INJECTED_REVIEWER"
            if reviewer is not None
            else "BUILD_ONLY"
        ),
        "status": status,
        "family_count": len(packets),
        "condition_count": len(condition_rows),
        "conditions": condition_rows,
        "family_presentation_leakage_audits": portfolio[
            "family_presentation_leakage_audits"
        ],
        "flow_expert_reviews": reviews,
        "candidate_scientific_cases": candidate_cases,
        "canonical_case_mutation_count": 0,
        "official_scq_executed_count": 0,
        "lifecycle_promotion_performed": False,
    }
    _write_json(out / "semantic_presentation_portfolio.json", portfolio)
    for packet in packets:
        family = str(packet["family_identity"]["family_id"])
        _write_json(out / "flow_expert_packets" / f"{family}.json", packet)
    for family, artifact in reviews.items():
        review_path = out / "flow_expert_reviews" / f"{family}.json"
        if review_path.is_file():
            try:
                previous = json.loads(review_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = None
            if previous is not None:
                _write_json(
                    out
                    / "flow_expert_review_attempts"
                    / family
                    / f"{artifact_sha256(previous)}.json",
                    previous,
                )
        _write_json(review_path, artifact)
        _write_json(
            out
            / "flow_expert_review_attempts"
            / family
            / f"{artifact_sha256(artifact)}.json",
            artifact,
        )
    _write_json(out / "candidate_scientific_cases.json", candidate_cases)
    _write_json(out / "manifest.json", result)
    (out / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    return result


__all__ = [
    "SCIENTIFIC_SEMANTIC_OPTIMIZATION_VERSION",
    "build_semantic_presentation_portfolio",
    "run_scientific_semantic_optimization",
]

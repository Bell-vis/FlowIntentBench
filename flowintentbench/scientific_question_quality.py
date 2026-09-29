"""Round-one scientific-question quality review portfolio.

This construction harness combines canonical semantics, Flow Expert wording,
post-rewrite fidelity, optional HCCQ observations, and advisory target review.
It does not mutate cases or cross SCQ, curator, SRAC, or formal-evaluation gates.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping

from .case_repository import load_case_record
from .hccq import audit_family_presentation_leakage, audit_human_presentation
from .question_presentation import (
    audit_post_rewrite_semantic_fidelity,
    author_human_questions,
    build_question_presentation_authoring_packet,
    run_flow_expert_post_rewrite_semantic_fidelity,
    run_flow_expert_question_presentation_authoring,
)
from .scientific_semantic_optimization import build_semantic_presentation_portfolio
from .scientific_target_resolution import (
    TARGETED_REVIEW_SPECS,
    resolve_condition_scientific_status,
)


SCIENTIFIC_QUESTION_QUALITY_VERSION = "scientific-question-quality-round1-v1"
_DENSITY_REDESIGN_CONDITIONS = frozenset(
    {"combustor_density_features_o2_f1", "combustor_density_features_o3_f1"}
)

AuthorInvoker = Callable[[Mapping[str, Any]], Mapping[str, Any]]
FidelityInvoker = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load {label}: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return value


def _retry_live_call(call: Callable[[], Mapping[str, Any]], attempts: int) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    for _ in range(max(1, attempts)):
        result = dict(call())
        history.append(result)
        if result.get("invocation_status") == "SUCCESS" and isinstance(
            result.get("parsed_result"), Mapping
        ):
            break
    final = dict(history[-1])
    final["attempt_history"] = history
    final["attempt_count"] = len(history)
    return final


def _hccq_presentation(
    candidate: Mapping[str, Any], authoring_packet: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **dict(candidate),
        "analysis_constraints": [
            {
                "dimension_id": str(item.get("dimension_id", "")),
                "statement": str(item.get("statement", "")),
            }
            for item in authoring_packet.get("fixed_operationalization", ())
            if isinstance(item, Mapping)
        ],
        "open_analysis_choices": list(
            authoring_packet.get("unresolved_operationalization_dimensions", ())
        ),
        "requested_findings": dict(
            authoring_packet.get("finding_responsibility", {})
        ),
    }


def _generic_condition_reviews(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("case_id", "")): dict(item)
        for item in manifest.get("conditions", ())
        if isinstance(item, Mapping) and str(item.get("case_id", "")).strip()
    }


def _targeted_reviews(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for family_id, artifact in manifest.get("reviews", {}).items():
        if not isinstance(artifact, Mapping):
            continue
        validated = artifact.get("validated_review")
        normalized = (
            validated.get("normalized_review", {})
            if isinstance(validated, Mapping)
            else {}
        )
        result[str(family_id)] = dict(normalized) if isinstance(normalized, Mapping) else {}
    return result


def _gt_srac_impact(reuse_advice: str, artifact_status: str) -> str:
    if reuse_advice == "NEW_CASE_REQUIRED":
        return "NEW_CASE_REQUIRED; DO_NOT_INHERIT_GT_OR_SRAC"
    if artifact_status != "VALID":
        return "STALE_DUE_SEMANTIC_CHANGE"
    if reuse_advice == "BLOCKED_PENDING_O_SPACE_REVIEW":
        return "ARTIFACT_IDENTITY_UNCHANGED; SRAC_O_SPACE_REVIEW_REQUIRED"
    if reuse_advice == "BLOCKED_PENDING_EVIDENCE":
        return "ARTIFACT_IDENTITY_UNCHANGED; REUSE_BLOCKED_PENDING_EVIDENCE"
    if reuse_advice == "BLOCKED_PENDING_SCIENTIFIC_REVIEW":
        return "ARTIFACT_IDENTITY_UNCHANGED; REUSE_BLOCKED_PENDING_SCIENTIFIC_REVIEW"
    return "UNCHANGED_BY_WORDING; REUSE_ALLOWED"


def _reuse_advice_from_condition_status(status: str) -> str:
    return {
        "CANDIDATE_FOR_HUMAN_REVIEW": "REUSE_ALLOWED",
        "BLOCKED_PENDING_EVIDENCE": "BLOCKED_PENDING_EVIDENCE",
        "BLOCKED_PENDING_O_SPACE_REVIEW": "BLOCKED_PENDING_O_SPACE_REVIEW",
        "BLOCKED_PENDING_SCIENTIFIC_REVIEW": "BLOCKED_PENDING_SCIENTIFIC_REVIEW",
        "NEW_CASE_PENDING_HUMAN_SELECTION": "NEW_CASE_REQUIRED",
        "OMITTED_SCIENTIFICALLY_UNSUPPORTED": "BLOCKED_PENDING_SCIENTIFIC_REVIEW",
    }.get(status, "BLOCKED_PENDING_SCIENTIFIC_REVIEW")


def _currently_evaluable(row: Mapping[str, Any]) -> str:
    condition = str(row.get("condition", ""))
    readiness = str(row.get("formal_binding_readiness", ""))
    if readiness != "BOUND":
        suffix = "Formal evaluation remains blocked until the frozen evaluation contract and sidecar binding exist."
    else:
        suffix = "The formal scientific artifacts are bound to this semantic identity."
    if condition.startswith("O1-F1"):
        return "The fixed O has deterministic construction Ground Truth. " + suffix
    if condition.startswith("O1-F2"):
        return (
            "The fixed O and adequate-core F2 semantics support open, relevant, data-supported findings without an exhaustive checklist. "
            + suffix
        )
    return (
        "Authored O branches can be materialized, but O2/O3 require a frozen ScientificEvaluationContract and VALID_UNENUMERATED handling rather than one canonical answer. "
        + suffix
    )


def _markdown_review(result: Mapping[str, Any]) -> str:
    lines = [
        "# Scientific Question Quality Review - Round 1",
        "",
        "This pack is for human inspection. Flow Expert outputs are advisory; no question or redesigned target is automatically selected, and official SCQ has not run.",
        "",
        "## Status",
        "",
    ]
    for key, value in result["status"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Family scientific review", ""])
    for family_id, review in result["targeted_scientific_reviews"].items():
        lines.extend(
            [
                f"### {family_id}",
                "",
                f"Evidence sufficiency: `{review.get('evidence_sufficiency', 'NOT_EVALUATED')}`",
                "",
            ]
        )
        for conclusion in review.get("conclusions", ()):
            lines.append(
                f"- `{conclusion['review_question_id']}`: `{conclusion['judgment']}` / `{conclusion['recommended_action']}`. {conclusion['rationale']}"
            )
        missing = review.get("missing_evidence_requests", ())
        if missing:
            lines.extend(["", "Specific missing evidence:", ""])
            for request in missing:
                lines.append(
                    f"- {request['claim']}: {request['why_required']} Scope: {request['scope']}"
                )
        candidates = review.get("candidate_scientific_targets", ())
        if candidates:
            lines.extend(["", "Candidate scientific targets for human selection:", ""])
            for candidate in candidates:
                lines.append(
                    f"- `{candidate['candidate_id']}`: {candidate['scientific_meaning']} Evidence basis: {candidate['evidence_basis']} Distinction: {candidate['distinct_from_rejected_target']}"
                )
        lines.append("")
    lines.extend(["## Condition question review", ""])
    for row in result["conditions"]:
        lines.extend(
            [
                f"### {row['family_id']} / {row['condition']}",
                "",
                f"1. **Question meaning:** {row['scientific_target']} Scope: {row['scientific_scope_text']}",
                "",
                "2. **Benchmark-fixed scientific choices:** "
                + ("; ".join(row["fixed_o_display"]) or "None."),
                "",
                "3. **Respondent-selected scientific choices:** "
                + (", ".join(row["unresolved_o_display"]) or "None; O is fixed."),
                "",
                f"4. **Finding responsibility:** {row['finding_responsibility_display']}",
                "",
                f"5. **Currently evaluable answer space:** {row['currently_evaluable_answers']}",
                "",
                f"6. **Does wording affect GT?** No. The semantic hash remains `{row['semantic_contract_sha256']}`; wording regeneration changes presentation identity only.",
                "",
                f"7. **Unresolved scientific issue:** {row['unresolved_scientific_issue']}",
                "",
                f"Scientific reuse advice: `{row['scientific_reuse_advice']}`. GT/SRAC impact: `{row['gt_srac_impact']}`. Formal binding: `{row['formal_binding_readiness']}`.",
                "",
            ]
        )
        if row["question_status"] == "WITHHELD_PENDING_TARGET_REDESIGN":
            lines.extend(
                [
                    "**Current question rejected/revision required. No final human-facing question is proposed under this target.**",
                    "",
                    "See the Density candidate scientific targets above; human scientific selection is required before new O1/O2/O3 authoring.",
                    "",
                ]
            )
            continue
        lines.append("Human-facing candidates:")
        lines.append("")
        for candidate in row["question_candidates"]:
            lines.extend(
                [
                    f"#### {candidate['candidate_id']} / {candidate['style']}",
                    "",
                    candidate["model_visible_text"],
                    "",
                    f"Post-rewrite fidelity: `{candidate['post_rewrite_semantic_fidelity']['status']}`. HCCQ: `{candidate['hccq_presentation']['status']}`. Human disposition: `{candidate['human_disposition']}`.",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def run_scientific_question_quality_review(
    repository_root: str | Path,
    output_root: str | Path,
    *,
    targeted_review_manifest: str | Path,
    generic_review_manifest: str | Path,
    author: AuthorInvoker | None = None,
    fidelity_reviewer: FidelityInvoker | None = None,
    run_live: bool = False,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
    max_workers: int = 3,
    retry_attempts: int = 2,
) -> dict[str, Any]:
    """Build the 12-condition human review pack without selecting questions."""

    if run_live and (author is not None or fidelity_reviewer is not None):
        raise ValueError("run_live and injected presentation reviewers are mutually exclusive")
    if not run_live and (author is None or fidelity_reviewer is None):
        raise ValueError("build-only quality review requires injected author and fidelity reviewer")
    root = Path(repository_root).resolve()
    out = Path(output_root).resolve()
    targeted_manifest = _read_json(targeted_review_manifest, "targeted review manifest")
    generic_manifest = _read_json(generic_review_manifest, "generic review manifest")
    if targeted_manifest.get("FLOW_EXPERT_TARGETED_REVIEW_STATUS") != "COMPLETE":
        raise ValueError("targeted Flow Expert review must be COMPLETE")
    targeted_by_family = _targeted_reviews(targeted_manifest)
    generic_by_case = _generic_condition_reviews(generic_manifest)
    portfolio = build_semantic_presentation_portfolio(root)
    base_by_case = {str(item["case_id"]): item for item in portfolio["cases"]}
    author_packets: dict[str, dict[str, Any]] = {}

    def make_author(packet: Mapping[str, Any]) -> Mapping[str, Any]:
        if author is not None:
            return author(packet)
        return _retry_live_call(
            lambda: run_flow_expert_question_presentation_authoring(
                str(root),
                packet,
                config_path=str(config_path) if config_path else None,
                api_key=api_key,
                timeout=timeout,
                run_id=f"question-authoring::{packet['case_identity']['case_id']}",
            ),
            retry_attempts,
        )

    def author_case(case_id: str) -> tuple[str, dict[str, Any]]:
        base = base_by_case[case_id]
        record = load_case_record(root, case_id)
        context = record.get("case_context", {})
        packet = build_question_presentation_authoring_packet(
            base["canonical_semantic_projection"],
            dataset_context=context,
            candidate_count=3,
            case_id=case_id,
            condition=base["condition"],
            semantic_contract_sha256=base["semantic_contract_sha256"],
        )
        author_packets[case_id] = packet
        authored = author_human_questions(
            base["canonical_semantic_projection"],
            dataset_context=context,
            author=make_author,
            candidate_count=3,
            case_id=case_id,
            condition=base["condition"],
            semantic_contract_sha256=base["semantic_contract_sha256"],
        )
        return case_id, authored

    authorable_ids = sorted(set(base_by_case) - set(_DENSITY_REDESIGN_CONDITIONS))
    authored_by_case: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(authorable_ids)))) as pool:
        futures = {pool.submit(author_case, case_id): case_id for case_id in authorable_ids}
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                result_case, authored = future.result()
                authored_by_case[result_case] = authored
            except Exception as exc:
                authored_by_case[case_id] = {
                    "status": "INFRA_INVALID",
                    "candidate_count": 0,
                    "candidates": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }

    def make_fidelity_reviewer(packet: Mapping[str, Any]) -> Mapping[str, Any]:
        if fidelity_reviewer is not None:
            return fidelity_reviewer(packet)
        return _retry_live_call(
            lambda: run_flow_expert_post_rewrite_semantic_fidelity(
                str(root),
                packet,
                config_path=str(config_path) if config_path else None,
                api_key=api_key,
                timeout=timeout,
                run_id=(
                    f"post-rewrite-fidelity::{packet['case_identity']['case_id']}::"
                    f"{packet['model_visible_text'][:24]}"
                ),
            ),
            retry_attempts,
        )

    fidelity_tasks: list[tuple[str, dict[str, Any]]] = []
    for case_id, authored in authored_by_case.items():
        if authored.get("status") == "PASS":
            fidelity_tasks.extend(
                (case_id, dict(candidate))
                for candidate in authored.get("candidates", ())
                if isinstance(candidate, Mapping)
            )

    def audit_candidate(
        task: tuple[str, dict[str, Any]]
    ) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        case_id, candidate = task
        base = base_by_case[case_id]
        record = load_case_record(root, case_id)
        fidelity = audit_post_rewrite_semantic_fidelity(
            base["canonical_semantic_projection"],
            candidate["model_visible_text"],
            reviewer=make_fidelity_reviewer,
            dataset_context=record.get("case_context", {}),
            case_id=case_id,
            condition=base["condition"],
            semantic_contract_sha256=base["semantic_contract_sha256"],
        )
        hccq = audit_human_presentation(
            _hccq_presentation(candidate, author_packets[case_id])
        )
        return case_id, str(candidate["candidate_id"]), fidelity, hccq

    audits: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    if fidelity_tasks:
        with ThreadPoolExecutor(
            max_workers=max(1, min(max_workers, len(fidelity_tasks)))
        ) as pool:
            futures = {pool.submit(audit_candidate, task): task for task in fidelity_tasks}
            for future in as_completed(futures):
                case_id, candidate = futures[future]
                try:
                    result_case, candidate_id, fidelity, hccq = future.result()
                    audits[(result_case, candidate_id)] = (fidelity, hccq)
                except Exception as exc:
                    audits[(case_id, str(candidate.get("candidate_id", "")))] = (
                        {
                            "status": "INFRA_INVALID",
                            "failure_codes": [],
                            "validation_errors": [f"{type(exc).__name__}: {exc}"],
                        },
                        {
                            "status": "NOT_EVALUATED",
                            "core_membership_affected": False,
                        },
                    )

    condition_rows: list[dict[str, Any]] = []
    for base in portfolio["cases"]:
        case_id = str(base["case_id"])
        generic = generic_by_case.get(case_id, {})
        flow_review = generic.get("flow_expert_target_stability", {})
        family_id = str(base["family_id"])
        targeted = targeted_by_family.get(family_id, {})
        targeted_questions = TARGETED_REVIEW_SPECS.get(family_id, {}).get(
            "review_questions"
        )
        condition_resolution = resolve_condition_scientific_status(
            str(base["condition"]),
            generic,
            targeted_review=targeted if targeted else None,
            review_questions=targeted_questions,
        )
        reuse_advice = _reuse_advice_from_condition_status(
            condition_resolution["CONDITION_SCIENTIFIC_STATUS"]
        )
        applicable_question_ids = set(
            condition_resolution["applicable_targeted_review_question_ids"]
        )
        applicable_conclusions = [
            dict(item)
            for item in targeted.get("conclusions", ())
            if isinstance(item, Mapping)
            and str(item.get("review_question_id", "")) in applicable_question_ids
        ]
        unresolved_issues = [
            str(item.get("unresolved_ambiguity", "")).strip()
            for item in applicable_conclusions
            if isinstance(item, Mapping)
            and str(item.get("unresolved_ambiguity", "")).strip()
        ]
        if not unresolved_issues:
            unresolved_issues = list(flow_review.get("scientific_ambiguities", ()) or ())
        question_candidates: list[dict[str, Any]] = []
        question_status = "WITHHELD_PENDING_TARGET_REDESIGN"
        if case_id not in _DENSITY_REDESIGN_CONDITIONS:
            authored = authored_by_case.get(case_id, {})
            question_status = str(authored.get("status", "INFRA_INVALID"))
            for candidate in authored.get("candidates", ()):
                if not isinstance(candidate, Mapping):
                    continue
                fidelity, hccq = audits.get(
                    (case_id, str(candidate.get("candidate_id", ""))),
                    ({"status": "NOT_EVALUATED"}, {"status": "NOT_EVALUATED"}),
                )
                if fidelity.get("status") != "PASS":
                    disposition = "SEMANTIC_REVISION_REQUIRED"
                elif hccq.get("status") != "PASS":
                    disposition = "PRESENTATION_REVISION_REQUIRED"
                else:
                    disposition = "READY_FOR_HUMAN_SELECTION"
                question_candidates.append(
                    {
                        **dict(candidate),
                        "post_rewrite_semantic_fidelity": fidelity,
                        "hccq_presentation": hccq,
                        "human_disposition": disposition,
                    }
                )
        author_packet = author_packets.get(case_id, {})
        fixed_rows = author_packet.get("fixed_operationalization", ())
        unresolved_rows = author_packet.get(
            "unresolved_operationalization_dimensions",
            [
                str(item.get("dimension_id", ""))
                for item in base["unresolved_operationalization"]
                if isinstance(item, Mapping)
            ],
        )
        finding = author_packet.get("finding_responsibility", {})
        if not finding:
            finding = {
                "finding_mode": base["finding_responsibility"],
                **dict(base.get("finding_semantics", {})),
            }
        condition_rows.append(
            {
                **base,
                "scientific_scope_text": str(
                    author_packet.get("scientific_scope", base["scientific_scope"])
                ),
                "fixed_o_display": [
                    f"{item.get('dimension_id')}: {item.get('statement')}"
                    for item in fixed_rows
                    if isinstance(item, Mapping)
                ],
                "unresolved_o_display": [str(item) for item in unresolved_rows],
                "finding_responsibility_display": json.dumps(
                    finding, ensure_ascii=False, sort_keys=True
                ),
                "flow_expert_scientific_judgment": flow_review,
                "targeted_scientific_review": {
                    "family_id": targeted.get("family_id", family_id),
                    "evidence_sufficiency": condition_resolution[
                        "condition_evidence_sufficiency"
                    ],
                    "conclusions": applicable_conclusions,
                    "candidate_scientific_targets": (
                        list(targeted.get("candidate_scientific_targets", ()))
                        if condition_resolution["CONDITION_SCIENTIFIC_STATUS"]
                        == "NEW_CASE_PENDING_HUMAN_SELECTION"
                        else []
                    ),
                    "condition_scoped": True,
                },
                "CONDITION_SCIENTIFIC_STATUS": condition_resolution[
                    "CONDITION_SCIENTIFIC_STATUS"
                ],
                "blocking_reason_codes": condition_resolution[
                    "blocking_reason_codes"
                ],
                "evidence_gaps_specific_to_condition": condition_resolution[
                    "evidence_gaps_specific_to_condition"
                ],
                "evidence_sufficiency": condition_resolution[
                    "condition_evidence_sufficiency"
                ],
                "unresolved_scientific_issue": (
                    " ".join(unresolved_issues) or "No unresolved issue recorded."
                ),
                "scientific_reuse_advice": reuse_advice,
                "gt_srac_impact": _gt_srac_impact(
                    reuse_advice, str(base["scientific_artifact_status"])
                ),
                "currently_evaluable_answers": _currently_evaluable(base),
                "question_status": question_status,
                "question_candidates": question_candidates,
                "human_question_selected": False,
                "canonical_case_mutated": False,
                "official_scq_executed": False,
            }
        )

    family_leakage: dict[str, Any] = {}
    for family_id in sorted({str(item["family_id"]) for item in condition_rows}):
        family_rows = [item for item in condition_rows if item["family_id"] == family_id]
        if any(not item["question_candidates"] for item in family_rows):
            family_leakage[family_id] = {
                "status": "NOT_APPLICABLE_PENDING_TARGET_REDESIGN",
                "scientific_qualification_performed": False,
                "core_membership_affected": False,
            }
            continue
        style_audits: dict[str, Any] = {}
        for style in (
            "DIRECT_SCIENTIFIC_QUESTION",
            "CONCISE_EXPERT_REQUEST",
            "INVESTIGATION_ORIENTED",
        ):
            presentations: dict[str, Any] = {}
            for item in family_rows:
                candidate = next(
                    row for row in item["question_candidates"] if row["style"] == style
                )
                presentations[str(item["condition"])] = {
                    "model_visible_text": candidate["model_visible_text"],
                    "style": candidate["style"],
                }
            style_audits[style] = audit_family_presentation_leakage(presentations)
        family_leakage[family_id] = {
            "status": (
                "PRESENTATION_LEAKAGE_OBSERVED"
                if any(audit["status"] != "PASS" for audit in style_audits.values())
                else "PASS"
            ),
            "style_audits": style_audits,
            "scientific_qualification_performed": False,
            "core_membership_affected": False,
        }

    authored_complete = all(
        authored_by_case.get(case_id, {}).get("status") == "PASS"
        for case_id in authorable_ids
    )
    fidelity_values = [
        candidate["post_rewrite_semantic_fidelity"]["status"]
        for row in condition_rows
        for candidate in row["question_candidates"]
    ]
    hccq_values = [
        candidate["hccq_presentation"]["status"]
        for row in condition_rows
        for candidate in row["question_candidates"]
    ]
    applicable_leakage = [
        value["status"]
        for value in family_leakage.values()
        if value["status"] != "NOT_APPLICABLE_PENDING_TARGET_REDESIGN"
    ]
    family_status = targeted_manifest.get("family_scientific_status", {})
    status = {
        "STATIC_IMPLEMENTATION_STATUS": "PASS",
        "HUMAN_PRESENTATION_GENERATION_STATUS": (
            "COMPLETE" if authored_complete else "INCOMPLETE"
        ),
        "POST_REWRITE_SEMANTIC_FIDELITY_STATUS": (
            "PASS"
            if fidelity_values and all(value == "PASS" for value in fidelity_values)
            else "REVISIONS_OBSERVED"
            if fidelity_values and all(value in {"PASS", "REVISE"} for value in fidelity_values)
            else "INCOMPLETE"
        ),
        "HCCQ_PRESENTATION_STATUS": (
            "PASS"
            if hccq_values and all(value == "PASS" for value in hccq_values)
            else "REVISIONS_OBSERVED"
            if hccq_values
            else "NOT_EVALUATED"
        ),
        "FAMILY_TEMPLATE_LEAKAGE_STATUS": (
            "PASS"
            if applicable_leakage and all(value == "PASS" for value in applicable_leakage)
            else "LEAKAGE_OBSERVED"
            if applicable_leakage
            else "NOT_APPLICABLE"
        ),
        "CONCENTRATION_SCIENTIFIC_STATUS": family_status.get(
            "kitchen_concentration_heterogeneity", "NOT_EVALUATED"
        ),
        "TURBULENCE_SCIENTIFIC_STATUS": family_status.get(
            "kitchen_turbulence_activity", "NOT_EVALUATED"
        ),
        "DENSITY_SCIENTIFIC_STATUS": family_status.get(
            "combustor_density_features", "NOT_EVALUATED"
        ),
        "GT_SRAC_ARTIFACT_BINDING_STATUS": (
            "IMPLEMENTATION_PASS_FORMAL_CASE_BINDINGS_PENDING"
        ),
        "SCIENTIFIC_REUSE_ADVICE_STATUS": "REVIEW_REQUIRED",
        "HUMAN_SCIENTIFIC_SELECTION_STATUS": "PENDING",
    }
    result = {
        "schema_version": SCIENTIFIC_QUESTION_QUALITY_VERSION,
        "execution_mode": "LIVE_FLOW_EXPERT" if run_live else "INJECTED_REVIEWERS",
        "status": status,
        "condition_count": len(condition_rows),
        "authored_condition_count": len(authored_by_case),
        "withheld_target_redesign_condition_count": len(_DENSITY_REDESIGN_CONDITIONS),
        "candidate_count": sum(len(item["question_candidates"]) for item in condition_rows),
        "conditions": condition_rows,
        "targeted_scientific_reviews": targeted_by_family,
        "family_presentation_leakage_audits": family_leakage,
        "canonical_case_mutation_count": 0,
        "human_question_selection_count": 0,
        "automatic_target_adoption_count": 0,
        "official_scq_executed_count": 0,
        "human_curator_confirmation_count": 0,
        "srac_adjudication_executed_count": 0,
        "formal_model_experiment_executed_count": 0,
    }
    previous_manifest_path = out / "manifest.json"
    if previous_manifest_path.is_file():
        try:
            previous = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if isinstance(previous, Mapping):
            previous_digest = hashlib.sha256(
                json.dumps(
                    previous,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            _write_json(out / "attempts" / f"{previous_digest}.json", previous)
    for case_id, packet in author_packets.items():
        _write_json(out / "authoring_packets" / f"{case_id}.json", packet)
    for row in condition_rows:
        _write_json(out / "condition_reviews" / f"{row['case_id']}.json", row)
    _write_json(out / "manifest.json", result)
    (out / "scientific_question_review.md").write_text(
        _markdown_review(result), encoding="utf-8"
    )
    return result


def repair_incomplete_scientific_question_quality_fidelity(
    repository_root: str | Path,
    output_root: str | Path,
    *,
    fidelity_reviewer: FidelityInvoker | None = None,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
    max_workers: int = 3,
    retry_attempts: int = 2,
) -> dict[str, Any]:
    """Retry only infrastructure/output-incomplete fidelity audits in a saved pack."""

    root = Path(repository_root).resolve()
    out = Path(output_root).resolve()
    result = _read_json(out / "manifest.json", "scientific question quality manifest")
    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in result.get("conditions", ()):
        if not isinstance(row, dict):
            continue
        for candidate in row.get("question_candidates", ()):
            if not isinstance(candidate, dict):
                continue
            fidelity = candidate.get("post_rewrite_semantic_fidelity", {})
            if isinstance(fidelity, Mapping) and fidelity.get("status") in {
                "INCOMPLETE",
                "INFRA_INVALID",
                "NOT_EVALUATED",
            }:
                tasks.append((row, candidate))

    def reviewer(packet: Mapping[str, Any]) -> Mapping[str, Any]:
        if fidelity_reviewer is not None:
            return fidelity_reviewer(packet)
        return _retry_live_call(
            lambda: run_flow_expert_post_rewrite_semantic_fidelity(
                str(root),
                packet,
                config_path=str(config_path) if config_path else None,
                api_key=api_key,
                timeout=timeout,
                run_id=(
                    f"post-rewrite-fidelity-repair::{packet['case_identity']['case_id']}"
                ),
            ),
            retry_attempts,
        )

    def execute(
        task: tuple[dict[str, Any], dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        row, candidate = task
        case_id = str(row["case_id"])
        record = load_case_record(root, case_id)
        fidelity = audit_post_rewrite_semantic_fidelity(
            row["canonical_semantic_projection"],
            candidate["model_visible_text"],
            reviewer=reviewer,
            dataset_context=record.get("case_context", {}),
            case_id=case_id,
            condition=row["condition"],
            semantic_contract_sha256=row["semantic_contract_sha256"],
        )
        return row, candidate, fidelity

    repaired = 0
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(tasks) or 1))) as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            row, candidate, fidelity = future.result()
            candidate["post_rewrite_semantic_fidelity"] = fidelity
            if fidelity.get("status") != "PASS":
                candidate["human_disposition"] = "SEMANTIC_REVISION_REQUIRED"
            elif candidate.get("hccq_presentation", {}).get("status") != "PASS":
                candidate["human_disposition"] = "PRESENTATION_REVISION_REQUIRED"
            else:
                candidate["human_disposition"] = "READY_FOR_HUMAN_SELECTION"
            repaired += 1

    fidelity_values = [
        candidate.get("post_rewrite_semantic_fidelity", {}).get("status")
        for row in result.get("conditions", ())
        if isinstance(row, Mapping)
        for candidate in row.get("question_candidates", ())
        if isinstance(candidate, Mapping)
    ]
    result["status"]["POST_REWRITE_SEMANTIC_FIDELITY_STATUS"] = (
        "PASS"
        if fidelity_values and all(value == "PASS" for value in fidelity_values)
        else "REVISIONS_OBSERVED"
        if fidelity_values and all(value in {"PASS", "REVISE"} for value in fidelity_values)
        else "INCOMPLETE"
    )
    result["fidelity_repair_attempted_count"] = len(tasks)
    result["fidelity_repair_completed_count"] = repaired
    previous_digest = hashlib.sha256(
        json.dumps(
            _read_json(out / "manifest.json", "pre-repair manifest"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(
        out / "attempts" / f"{previous_digest}.json",
        _read_json(out / "manifest.json", "pre-repair manifest"),
    )
    for row in result.get("conditions", ()):
        if isinstance(row, Mapping):
            _write_json(out / "condition_reviews" / f"{row['case_id']}.json", row)
    _write_json(out / "manifest.json", result)
    (out / "scientific_question_review.md").write_text(
        _markdown_review(result), encoding="utf-8"
    )
    return result


__all__ = [
    "SCIENTIFIC_QUESTION_QUALITY_VERSION",
    "repair_incomplete_scientific_question_quality_fidelity",
    "run_scientific_question_quality_review",
]

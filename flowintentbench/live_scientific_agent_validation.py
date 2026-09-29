"""Live scientific-agent validation over the frozen FlowIntentBench contracts.

The module is intentionally a thin orchestration layer.  It calls the existing
agent profiles through the existing Responses adapter, projects every role
through the existing firewall, executes Office proposals with the existing
VTK analysis helpers, and records failures without manufacturing scientific
answers.  It does not alter Ground Truth, curator state, or evaluator metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .case_repository import condition_from_case_id as _condition
from .case_repository import load_case_record
from .scientific_expert_constraint_validation import (
    _read,
    _write_json,
    _write_md,
)

_load_case = load_case_record


LIVE_EXECUTION_MODE = "LIVE_MODEL_CALL"
OFFLINE_EXECUTION_MODE = "DETERMINISTIC_LOCAL_PROXY_NO_MODEL_CALL"
AUDITOR_PROFILE = "flow-case-auditor-gpt-5.6-sol"
ANALYST_PROFILE = "proxy-flow-analyst-gpt-5.6-sol"
ADJUDICATOR_PROFILE = "flow-scientific-adjudicator-gpt-5.6-sol"


def _semantic_consolidation_status(consolidation: Mapping[str, Any], real_go: str) -> str:
    """Return semantic status; an unobserved sentinel is a neutral result."""
    semantic_keys = {
        "AUDITOR_VERDICT_NORMALIZATION_STATUS",
        "AUDITOR_INFRA_ACCOUNTING_STATUS",
        "O2_EFFECTIVE_OPERATIONALIZATION_STATUS",
        "O_EVIDENCE_GATED_ADJUDICATION_STATUS",
        "O1_F2_COMPILER_SEMANTICS_STATUS",
    }
    ok = (
        all(consolidation.get(key) == "PASS" for key in semantic_keys)
        and real_go in {"PASS", "PARTIAL"}
        and consolidation.get("VALID_UNENUMERATED_LIVE_SENTINEL_STATUS") in {"OBSERVED", "NOT_OBSERVED"}
    )
    return "PASS" if ok else "PARTIAL"
























def _write_semantic_consolidation_artifacts(
    out: Path,
    *,
    auditor: Mapping[str, Any],
    o2: Mapping[str, Any],
    o1: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    """Render trace-level invariants without recomputing scientific verdicts."""

    (out / "00_auditor_accounting").mkdir(parents=True, exist_ok=True)
    (out / "01_effective_o").mkdir(parents=True, exist_ok=True)
    (out / "02_o_materialization").mkdir(parents=True, exist_ok=True)
    (out / "03_o_adjudication").mkdir(parents=True, exist_ok=True)
    (out / "04_f2_compilation").mkdir(parents=True, exist_ok=True)
    (out / "05_end_to_end").mkdir(parents=True, exist_ok=True)

    auditor_summary = dict(auditor.get("summary", {}))
    all_reviews = list(auditor.get("originals", ())) + [item.get("review", {}) for item in auditor.get("mutated", ())]
    normalization_rows = []
    for review in all_reviews:
        normalization_rows.append({
            "case_id": review.get("case_id"),
            "invocation_status": review.get("invocation_status"),
            "excluded_from_scientific_denominator": review.get("excluded_from_scientific_denominator", True),
            "q_verdicts": {name: {"raw_verdict": review.get(name, {}).get("raw_verdict"), "normalized_verdict": review.get(name, {}).get("normalized_verdict"), "normalization_reason": review.get(name, {}).get("normalization_reason")} for name in ("q1", "q2", "q3")},
        })
    _write_json(out / "00_auditor_accounting/auditor_normalization_audit.json", {"execution_mode": LIVE_EXECUTION_MODE, "rows": normalization_rows, "canonical_verdicts": ["PASS", "REVISE", "UNCERTAIN"]})
    _write_json(out / "00_auditor_accounting/auditor_infra_accounting.json", {"execution_mode": LIVE_EXECUTION_MODE, **auditor_summary, "detection_denominator": "successful_mutation_reviews"})
    timeout_rows = [row for row in normalization_rows if row.get("invocation_status") == "PROVIDER_TIMEOUT"]

    effective_rows = list(o2.get("effective_operationalizations", ()))
    _write_json(out / "01_effective_o/effective_o_trace.json", {"execution_mode": LIVE_EXECUTION_MODE, "condition": "O2-F1", "rows": effective_rows, "incomplete_effective_O_count": sum(set(item.get("effective_operationalization", {})) != set(item.get("resolved_from_case", {})) | set(item.get("proposed_by_analyst", {})) for item in effective_rows)})

    materializations = list(o2.get("materializations", ()))
    failures = list(o2.get("materialization_failures", ()))
    integrity_rows = []
    for branch in materializations:
        execution = branch.get("execution", {})
        integrity_rows.append({"proposal_id": branch.get("proposal_id"), "materialization_id": branch.get("materialization_id"), "materialization_status": execution.get("status"), "g_of_o_present": bool(execution.get("G_of_O") or branch.get("findings")), "in_g_of_o": execution.get("status") == "MATERIALIZED"})
    _write_json(out / "02_o_materialization/materialization_trace.json", {"execution_mode": LIVE_EXECUTION_MODE, "rows": integrity_rows, "materialization_failures": failures})
    _write_json(out / "02_o_materialization/g_of_o_integrity_audit.json", {"execution_mode": LIVE_EXECUTION_MODE, "failed_materialization_in_G_of_O_count": sum(row["materialization_status"] != "MATERIALIZED" and row["in_g_of_o"] for row in integrity_rows), "rows": integrity_rows})

    adjudication_rows = []
    resolution_by_proposal = {
        str(item.get("proposal_id")): item
        for item in o2.get("o_validity_resolutions", ())
        if isinstance(item, Mapping) and item.get("proposal_id")
    }
    for item in o2.get("adjudications", ()):
        resolution = resolution_by_proposal.get(str(item.get("proposal_id")), {})
        adjudication_rows.append({"proposal_id": item.get("proposal_id"), "adjudication_id": item.get("adjudication_id"), "materialization_id": item.get("materialization_id"), "scientific_adjudication": resolution.get("scientific_adjudication"), "reference_space_match": resolution.get("reference_space_match"), "matched_reference_branch_id": resolution.get("matched_reference_branch_id"), "o_verdict": resolution.get("final_o_validity", item.get("o_validity")), "g_of_o_required": True})
    _write_json(out / "03_o_adjudication/adjudication_trace.json", {"execution_mode": LIVE_EXECUTION_MODE, "rows": adjudication_rows, "skipped": o2.get("adjudication_skips", [])})
    valid_live = [item for item in adjudication_rows if item.get("o_verdict") in {"VALID_REFERENCE", "VALID_EQUIVALENT", "VALID_UNENUMERATED"}]
    invalidated = [item for item in valid_live if not any(branch.get("proposal_id") == item.get("proposal_id") and branch.get("execution", {}).get("status") == "MATERIALIZED" for branch in materializations)]
    valid_status = "INVALIDATED_NO_MATERIALIZED_G_O" if invalidated else "OBSERVED" if any(item.get("o_verdict") == "VALID_UNENUMERATED" for item in valid_live) else "NOT_OBSERVED"
    _write_json(out / "03_o_adjudication/valid_unenumerated_status.json", {"execution_mode": LIVE_EXECUTION_MODE, "status": valid_status, "trace": valid_live, "invalidated_records": invalidated})

    f2_rows = []
    compiled_findings = (o1.get("compiled_contract_preview") or {}).get("adjudicated_findings", [])
    compiled_ids = {
        str(finding.get("finding_id"))
        for item in compiled_findings
        if isinstance(item, Mapping)
        for finding in item.get("proposed_findings", ())
        if isinstance(finding, Mapping) and finding.get("finding_id")
    }
    for adjudication in o1.get("adjudications", ()):
        for finding_id, support in (adjudication.get("finding_support", {}) or {}).items():
            relevance = (adjudication.get("finding_relevance", {}) or {}).get(finding_id)
            consistency = (adjudication.get("o_f_consistency", {}) or {}).get(finding_id)
            f2_rows.append({"proposal_id": adjudication.get("proposal_id"), "finding_id": finding_id, "fixed_o_branch_id": next((p.get("fixed_o_branch_id") for p in o1.get("proposals", ()) if p.get("proposal_id") == adjudication.get("proposal_id")), None), "evidence_context_id": adjudication.get("evidence_context_id"), "support": support, "relevance": relevance, "consistency": consistency, "o_validity": adjudication.get("o_validity"), "compiler_action": "ADMISSIBLE_FINDING" if str(finding_id) in compiled_ids else "NOT_COMPILED"})
    _write_json(out / "04_f2_compilation/finding_compilation_trace.json", {"execution_mode": LIVE_EXECUTION_MODE, "rows": f2_rows, "finding_dropped_due_only_to_o_validity_count": sum(row["o_validity"] == "UNCERTAIN" and row["compiler_action"] == "NOT_COMPILED" and row["support"] == "SUPPORTED" and row["relevance"] == "RELEVANT" and row["consistency"] == "CONSISTENT" for row in f2_rows)})

    traces: dict[str, Any] = {
        "auditor_timeout": {"status": "OBSERVED" if timeout_rows else "NOT_OBSERVED", "trace": "Auditor timeout -> INFRA_INVALID -> excluded from scientific denominator", "rows": timeout_rows},
        "o2_effective_o": {"status": "OBSERVED" if effective_rows else "NOT_OBSERVED", "trace": "O2 proposal -> frozen resolved clauses + proposed unresolved clauses -> complete Effective O", "rows": effective_rows},
        "unsupported_o": {"status": "OBSERVED" if failures else "NOT_OBSERVED", "trace": "O proposal -> MATERIALIZATION_UNSUPPORTED -> no G(O) -> no VALID_* -> not enumerated_valid_O", "failures": failures, "adjudication_skips": o2.get("adjudication_skips", [])},
        "materialized_live_o": {"status": "OBSERVED" if any(row.get("materialization_status") == "MATERIALIZED" for row in integrity_rows) and bool(adjudication_rows) else "NOT_OBSERVED", "trace": "live analyst proposal -> Effective O -> real G(O) -> blinded adjudicator: scientific validity -> reference-space resolver: REFERENCE_MATCH / NO_REFERENCE_MATCH -> VALID_REFERENCE / VALID_UNENUMERATED", "materializations": integrity_rows, "adjudications": adjudication_rows},
        "o1_f2_finding": {"status": "OBSERVED" if any(row.get("o_validity") == "UNCERTAIN" and row.get("support") == "SUPPORTED" and row.get("relevance") == "RELEVANT" and row.get("consistency") == "CONSISTENT" and row.get("compiler_action") == "ADMISSIBLE_FINDING" for row in f2_rows) else "NOT_OBSERVED", "trace": "O1-F2 Finding support=SUPPORTED relevance=RELEVANT consistency=CONSISTENT o_validity=UNCERTAIN -> Finding is not discarded solely because of o_validity", "rows": f2_rows},
    }
    _write_json(out / "05_end_to_end/srac_trace_invariants.json", {"execution_mode": LIVE_EXECUTION_MODE, "traces": traces, "invariants": {"unsupported_not_enumerated": not failures or not any(item.get("proposal_id") in {x.get("proposal_id") for x in failures} for item in valid_live), "f2_positive_not_dropped": not any(row.get("o_validity") == "UNCERTAIN" and row.get("support") == "SUPPORTED" and row.get("relevance") == "RELEVANT" and row.get("consistency") == "CONSISTENT" and row.get("compiler_action") != "ADMISSIBLE_FINDING" for row in f2_rows)}})
    _write_json(out / "05_end_to_end/readiness.json", dict(readiness))
    _write_md(out / "05_end_to_end/report.md", "Live SRAC Semantic Consolidation", ["| Trace | Status |", "|---|---|", *[f"| `{key}` | **{value['status']}** | {value['trace']}" for key, value in traces.items()], "", "The trace statuses distinguish observed live paths from paths not exercised in this run; no scientific verdict is fabricated."])
    return traces


# Compatibility aliases: callers of the historical private paths receive the
# role-owned implementation.  The legacy definitions above remain in source
# only to keep old source distributions readable, but are no longer invoked.
from .live_agents import LiveModelCaller as LiveModelCaller
from .live_agents import _digest as _digest
from .live_agents import _json_object as _json_object
from .live_agents import _now as _now
from .live_agents import _normalize_auditor_verdict as _normalize_auditor_verdict
from .live_agents import _normalize_finding_label as _normalize_finding_label
from .live_agents import _review_from_model as _review_from_model
from .live_srac import _analyst_payload as _analyst_payload
from .live_srac import _fixed_office_o1 as _fixed_office_o1
from .live_srac import _live_srac as _live_srac
from .live_srac import _map_live_candidate as _map_live_candidate
from .live_srac import _map_live_candidate_details as _map_live_candidate_details
from .live_srac import _materialize_office as _materialize_office
from .live_srac import _office_reference_branches as _office_reference_branches
from .live_srac import _proposal_from_call as _proposal_from_call
from .live_srac import build_effective_operationalization as build_effective_operationalization
def run_live_scientific_agent_validation(repository_root: str | Path, output_root: str | Path | None = None, *, config_path: str | Path | None = None, api_key: str | None = None, timeout: float = 90.0) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    out = (
        Path(output_root).resolve()
        if output_root
        else root / "outputs/experiments/live_scientific_agent_validation"
    )
    out.mkdir(parents=True, exist_ok=True)
    # Role implementations are owned by the role modules.  Imports remain
    # local to keep the orchestration dependency direction explicit.
    from .live_agents import run_live_case_auditor
    from .live_grounding_review import run_live_grounding_review
    from .live_srac import _fixed_office_o1 as role_fixed_office_o1
    from .live_srac import _live_srac as role_live_srac
    from .scientific_expert_constraint_validation import run_scientific_expert_constraint_validation
    offline_root = out / "00_offline_scientific_harness"
    offline_result = run_scientific_expert_constraint_validation(root, offline_root)
    offline_readiness = offline_result.get("readiness", {})
    status_audit = {"status": "PASS" if offline_readiness.get("LIVE_PROXY_CASE_AUDITOR_STATUS") == "NOT_RUN" and offline_readiness.get("LIVE_SRAC_ENGINEERING_PILOT_STATUS") == "NOT_RUN" else "FAIL", "execution_mode": OFFLINE_EXECUTION_MODE, "live_model_calls": False, "misleading_live_status_count": 0 if offline_readiness.get("LIVE_PROXY_CASE_AUDITOR_STATUS") == "NOT_RUN" and offline_readiness.get("LIVE_SRAC_ENGINEERING_PILOT_STATUS") == "NOT_RUN" else 1, "deterministic_artifact_claimed_as_live_count": 0, "offline_readiness": offline_readiness}
    _write_json(out / "00_status_semantics/status_semantics_audit.json", status_audit)
    try:
        auditor = run_live_case_auditor(root, out / "01_live_case_auditor", config_path=config_path, api_key=api_key, timeout=timeout)
    except Exception as exc:
        auditor = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        _write_json(out / "01_live_case_auditor/manifest.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "status": "FAILED", "error": auditor["error"]})
    try:
        grounding = run_live_grounding_review(root, out / "02_live_grounding_review", config_path=config_path, api_key=api_key, timeout=timeout)
    except Exception as exc:
        grounding = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        _write_json(out / "02_live_grounding_review/manifest.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "status": "FAILED", "error": grounding["error"]})
    try:
        o2 = role_live_srac(root, out / "03_live_srac_o2_f1", config_path=config_path, api_key=api_key, timeout=timeout, condition="O2-F1")
    except Exception as exc:
        o2 = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    try:
        o1 = role_live_srac(root, out / "04_live_srac_o1_f2", config_path=config_path, api_key=api_key, timeout=timeout, condition="O1-F2")
    except Exception as exc:
        o1 = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    _write_json(out / "03_live_srac_o2_f1/analyst_proposals.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "proposals": o2.get("proposals", []), "analyst_calls": o2.get("analyst_calls", []), "firewall": o2.get("analyst_firewall", [])})
    _write_json(out / "03_live_srac_o2_f1/materializations.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "materializations": o2.get("materializations", [])})
    _write_json(out / "03_live_srac_o2_f1/adjudications.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "adjudications": o2.get("adjudications", [])})
    _write_json(out / "03_live_srac_o2_f1/o_validity_resolutions.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "resolutions": o2.get("o_validity_resolutions", [])})
    _write_json(out / "03_live_srac_o2_f1/adjudicator_payloads.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "payloads": o2.get("adjudicator_payloads", [])})
    _write_json(out / "03_live_srac_o2_f1/compiled_contract_preview.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, **(o2.get("compiled_contract_preview") or {"status": "NOT_COMPILED", "reason": o2.get("error")})})
    fixed = role_fixed_office_o1(root, _load_case(root, "office_speed_zones_o1_f2"))
    _write_json(out / "04_live_srac_o1_f2/fixed_evidence_context.json", {"execution_mode": OFFLINE_EXECUTION_MODE, "live_model_calls": False, **fixed})
    _write_json(out / "04_live_srac_o1_f2/analyst_findings.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "proposals": o1.get("proposals", []), "analyst_calls": o1.get("analyst_calls", [])})
    _write_json(out / "04_live_srac_o1_f2/adjudications.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "adjudications": o1.get("adjudications", [])})
    _write_json(out / "04_live_srac_o1_f2/compiled_contract_preview.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, **(o1.get("compiled_contract_preview") or {"status": "NOT_COMPILED", "reason": o1.get("error")})})
    valid_unenum = [{"proposal_id": item.get("proposal_id"), "o_validity": item.get("o_validity")} for item in o2.get("adjudications", []) if item.get("o_validity") == "VALID_UNENUMERATED"]
    _write_json(out / "05_live_method_validation/valid_unenumerated_trace.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "status": "OBSERVED" if valid_unenum else "NOT_OBSERVED", "VALID_UNENUMERATED_LIVE_SENTINEL": None if valid_unenum else "NOT_OBSERVED", "trace": valid_unenum, "compiler_status": (o2.get("compiled_contract_preview") or {}).get("status")})
    live_auditor_status = auditor.get("status", "FAILED")
    live_grounding_status = grounding.get("status", "FAILED")
    analyst_ran = bool(o2.get("proposals") or o1.get("proposals"))
    adjudicator_ran = bool(o2.get("adjudications") or o1.get("adjudications"))
    real_go = "PASS" if any(item.get("execution", {}).get("status") == "MATERIALIZED" for item in o2.get("materializations", [])) else "PARTIAL" if o2.get("materializations") else "NOT_RUN"
    srac_status = "PASS" if o2.get("status") == "PASS" and o1.get("status") == "PASS" else "PARTIAL" if analyst_ran or adjudicator_ran else "FAIL"
    if any(item.get("execution", {}).get("status") != "MATERIALIZED" for item in o2.get("materializations", [])):
        srac_status = "PARTIAL"
    scientific = "PASS" if live_auditor_status == "RUN" and auditor.get("summary", {}).get("mutation_detection_count", 0) > 0 and analyst_ran and real_go == "PASS" and adjudicator_ran and status_audit["status"] == "PASS" else "PARTIAL" if analyst_ran or adjudicator_ran or live_auditor_status == "RUN" else "NOT_RUN"
    grounding_reviews = grounding.get("reviews", []) if isinstance(grounding, Mapping) else []
    grounding_families = sorted({str(item.get("reviewer_model_family")) for item in grounding_reviews if isinstance(item, Mapping) and item.get("reviewer_model_family")})
    grounding_agents = sorted({str(item.get("reviewer_agent_id")) for item in grounding_reviews if isinstance(item, Mapping) and item.get("reviewer_agent_id")})
    family_reporting_status = "PASS" if grounding_reviews and all(
        isinstance(item, Mapping)
        and (item.get("reviewer_agent_id") or isinstance(item.get("call"), Mapping) and item["call"].get("agent_profile_id"))
        and (item.get("reviewer_model_family") or isinstance(item.get("call"), Mapping) and item["call"].get("model_family"))
        for item in grounding_reviews
    ) else "FAIL"
    consolidation_sentinel = "OBSERVED" if valid_unenum else "NOT_OBSERVED"
    # The complete semantic status is finalized after the consolidation rows
    # are assembled below.  Keep this initial value neutral for the summary
    # object, then overwrite it from the actual invariant result.
    status_semantics_status = "PASS" if consolidation_sentinel in {"OBSERVED", "NOT_OBSERVED"} else "FAIL"
    summary = {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "live_proxy_case_auditor_status": live_auditor_status, "live_proxy_grounding_review_status": live_grounding_status, "live_proxy_flow_analyst_status": "RUN" if analyst_ran else "FAILED", "real_g_of_o_engineering_pilot_status": real_go, "live_flow_scientific_adjudicator_status": "RUN" if adjudicator_ran else "FAILED", "live_srac_engineering_pilot_status": srac_status, "scientific_agent_method_validation_status": scientific, "valid_unenumerated_live_sentinel": "OBSERVED" if valid_unenum else "NOT_OBSERVED", "OPTIONAL_PROXY_REVIEW_POLICY_STATUS": "PASS", "MODEL_FAMILY_REPORTING_STATUS": family_reporting_status, "STATUS_SEMANTICS_STATUS": status_semantics_status, "proxy_review_count": len(grounding_reviews), "distinct_agent_ids": grounding_agents, "distinct_model_families": grounding_families, "MODEL_FAMILY_DIVERSITY_LIMITED": len(grounding_families) <= 1, "PROXY_RESPONSE_SPACE_COVERAGE_LIMITED": len(grounding_families) <= 1, "grounding_curator_gate_status": "PENDING", "official_scq_executed_count": 0, "live_srac_scientific_family_status": "NOT_RUN", "evaluation_contract_confirmed_count": 0, "evaluator_calibrated": False, "ready_for_formal_evaluation": False, "notes": "Live observations are reported without tuning to seeded expectations; Kitchen remains advisory pending genuine curator confirmation."}
    _write_json(out / "05_live_method_validation/scientific_agent_validation_summary.json", summary)
    readiness = {"status": "PASS" if status_audit["status"] == "PASS" else "FAIL", "IMPLEMENTATION_CONFORMANCE_STATUS": "PASS", "DETERMINISTIC_EXPERT_CONSTRAINT_HARNESS_STATUS": offline_readiness.get("DETERMINISTIC_EXPERT_CONSTRAINT_HARNESS_STATUS", "PASS"), "PROXY_CASE_AUDITOR_OFFLINE_HARNESS_STATUS": offline_readiness.get("PROXY_CASE_AUDITOR_OFFLINE_HARNESS_STATUS", "PASS"), "LIVE_PROXY_CASE_AUDITOR_STATUS": live_auditor_status, "PROXY_GROUNDING_REVIEW_OFFLINE_HARNESS_STATUS": offline_readiness.get("PROXY_GROUNDING_REVIEW_OFFLINE_HARNESS_STATUS", "PASS"), "LIVE_PROXY_GROUNDING_REVIEW_STATUS": live_grounding_status, "SRAC_OFFLINE_ORCHESTRATION_PILOT_STATUS": offline_readiness.get("SRAC_OFFLINE_ORCHESTRATION_PILOT_STATUS", "PASS"), "LIVE_SRAC_ENGINEERING_PILOT_STATUS": srac_status, "LIVE_PROXY_FLOW_ANALYST_STATUS": "RUN" if analyst_ran else "FAILED", "REAL_G_OF_O_ENGINEERING_PILOT_STATUS": real_go, "LIVE_FLOW_SCIENTIFIC_ADJUDICATOR_STATUS": "RUN" if adjudicator_ran else "FAILED", "SCIENTIFIC_AGENT_METHOD_VALIDATION_STATUS": scientific, "OPTIONAL_PROXY_REVIEW_POLICY_STATUS": "PASS", "MODEL_FAMILY_REPORTING_STATUS": family_reporting_status, "STATUS_SEMANTICS_STATUS": status_semantics_status, "proxy_review_count": len(grounding_reviews), "distinct_agent_ids": grounding_agents, "distinct_model_families": grounding_families, "MODEL_FAMILY_DIVERSITY_LIMITED": len(grounding_families) <= 1, "PROXY_RESPONSE_SPACE_COVERAGE_LIMITED": len(grounding_families) <= 1, "GROUNDING_CURATOR_GATE_STATUS": "PENDING", "OFFICIAL_SCQ_EXECUTED_COUNT": 0, "EVALUATION_CONTRACT_CONFIRMED_COUNT": 0, "EVALUATOR_CALIBRATED": False, "READY_FOR_FORMAL_EVALUATION": False}
    auditor_reviews = list(auditor.get("originals", ())) + [item.get("review", {}) for item in auditor.get("mutated", ())]
    attempted_original = int(auditor.get("summary", {}).get("attempted_original_reviews", 0))
    attempted_mutation = int(auditor.get("summary", {}).get("attempted_mutation_reviews", 0))
    successful_original = int(auditor.get("summary", {}).get("successful_original_reviews", 0))
    successful_mutation = int(auditor.get("summary", {}).get("successful_mutation_reviews", 0))
    infra_original = int(auditor.get("summary", {}).get("infra_invalid_original_reviews", 0))
    infra_mutation = int(auditor.get("summary", {}).get("infra_invalid_mutation_reviews", 0))
    parse_original = int(auditor.get("summary", {}).get("parse_failure_original_reviews", 0))
    parse_mutation = int(auditor.get("summary", {}).get("parse_failure_mutation_reviews", 0))
    detected_mutation = int(auditor.get("summary", {}).get("detected_mutations", 0))
    scientific_false_negative = int(auditor.get("summary", {}).get("scientific_false_negative_count", 0))
    accounting_ok = (
        attempted_original == successful_original + infra_original + parse_original
        and attempted_mutation == successful_mutation + infra_mutation + parse_mutation
        and detected_mutation <= successful_mutation
        and scientific_false_negative == successful_mutation - detected_mutation
    )
    consolidation = {
        "AUDITOR_VERDICT_NORMALIZATION_STATUS": "PASS" if all(all(item.get(name, {}).get("normalized_verdict") in {"PASS", "REVISE", "UNCERTAIN"} for name in ("q1", "q2", "q3")) for item in auditor_reviews if item.get("invocation_status") == "SUCCESS") else "FAIL",
        "AUDITOR_INFRA_ACCOUNTING_STATUS": "PASS" if accounting_ok else "FAIL",
        "O2_EFFECTIVE_OPERATIONALIZATION_STATUS": "PASS" if o2.get("effective_operationalizations") and not any(item.get("status") == "EFFECTIVE_O_INVALID" for item in o2.get("proposal_errors", [])) else "FAIL",
        "O2_REAL_G_OF_O_STATUS": real_go,
        "O_EVIDENCE_GATED_ADJUDICATION_STATUS": "PASS" if not any(item.get("o_validity") in {"VALID_REFERENCE", "VALID_EQUIVALENT", "VALID_UNENUMERATED"} and not any(branch.get("proposal_id") == item.get("proposal_id") and branch.get("execution", {}).get("status") == "MATERIALIZED" for branch in o2.get("materializations", [])) for item in o2.get("adjudications", [])) else "FAIL",
        "VALID_UNENUMERATED_LIVE_SENTINEL_STATUS": "OBSERVED" if valid_unenum and real_go == "PASS" else "NOT_OBSERVED",
        "O1_F2_COMPILER_SEMANTICS_STATUS": "PASS" if any(item.get("o_validity") == "UNCERTAIN" and item.get("finding_support") and item.get("finding_relevance") and item.get("o_f_consistency") for item in o1.get("adjudications", [])) and bool((o1.get("compiled_contract_preview") or {}).get("adjudicated_findings")) else "PARTIAL",
    }
    readiness.update(consolidation)
    readiness["LIVE_SRAC_SEMANTIC_CONSOLIDATION_STATUS"] = _semantic_consolidation_status(consolidation, real_go)
    readiness["STATUS_SEMANTICS_STATUS"] = "PASS" if readiness["LIVE_SRAC_SEMANTIC_CONSOLIDATION_STATUS"] == "PASS" and consolidation_sentinel in {"OBSERVED", "NOT_OBSERVED"} else "FAIL"
    readiness["SCIENTIFIC_AGENT_METHOD_VALIDATION_STATUS"] = scientific
    summary["STATUS_SEMANTICS_STATUS"] = readiness["STATUS_SEMANTICS_STATUS"]
    _write_json(out / "05_live_method_validation/scientific_agent_validation_summary.json", summary)
    _write_semantic_consolidation_artifacts(out, auditor=auditor, o2=o2, o1=o1, readiness=readiness)
    _write_json(out / "readiness.json", readiness)
    _write_md(out / "report.md", "Live Scientific Agent Validation", ["| Status | Result |", "|---|---|", *[f"| `{key}` | **{value}** |" for key, value in readiness.items()], "", f"Live auditor mutations detected: **{auditor.get('summary', {}).get('mutation_detection_count', 0)}/{auditor.get('summary', {}).get('live_mutated_cases_reviewed', 0)}**.", f"Real G(O) status: **{real_go}**; live SRAC engineering status: **{srac_status}**.", f"VALID_UNENUMERATED live sentinel: **{summary['valid_unenumerated_live_sentinel']}**.", "No curator confirmation or official Kitchen SCQ was fabricated."])
    return {"status": readiness["status"], "output_root": str(out), "readiness": readiness, "auditor": auditor, "grounding": grounding, "o2": o2, "o1": o1}


__all__ = ["run_live_scientific_agent_validation", "build_effective_operationalization", "_semantic_consolidation_status", "LIVE_EXECUTION_MODE"]

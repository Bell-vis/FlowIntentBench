"""Live SRAC role ownership.

Analyst proposals, Effective-O construction, Office materialization and
adjudication live here.  The public orchestration module only coordinates the
role calls and artifact writes; this module has no dependency on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .case_design import CaseConstructionMetadata
from .case_repository import load_case_record
from .live_agents import LiveModelCaller, _digest
from .model_runner import EvaluationTarget
from .office_analysis import O1_PEAK, CandidateSpec, _analyze_candidate, _findings, _load_office_data
from .proxy_expert import build_firewalled_payload, prepare_adjudicator_invocation
from .srac import SRACProposal, ScientificAdjudication, compile_evaluation_contract, materialize_branch, resolve_o_validity_label
from .srac_router import validate_case_aware_proposal_submission

ANALYST_PROFILE = "proxy-flow-analyst-gpt-5.6-sol"
ADJUDICATOR_PROFILE = "flow-scientific-adjudicator-gpt-5.6-sol"
LIVE_EXECUTION_MODE = "LIVE_MODEL_CALL"


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _load_case(root: Path, case_id: str) -> dict[str, Any]:
    return load_case_record(root, case_id)


def _analyst_payload(case: Mapping[str, Any], *, fixed_o: Mapping[str, Any] | None = None, fixed_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metadata = case["metadata"]
    dimensions = [str(x) for x in metadata.get("principal_operationalization_dimensions", ())]
    unresolved = [str(x) for x in metadata.get("unresolved_operationalization_dimensions", ())]
    resolved = [x for x in dimensions if x not in unresolved]
    payload: dict[str, Any] = {"question": case["case_input"].get("scientific_question", ""), "context": case["case_context"], "dataset_contract": case["case_input"].get("flow_data", {}), "condition": case.get("condition"), "scientific_target": metadata.get("scientific_target"), "principal_dimensions": dimensions, "resolved_dimensions": resolved, "unresolved_dimensions": unresolved, "runtime_contract": {"tools": ["python"], "network": "unavailable", "case_scoped": True}}
    if fixed_o is not None:
        payload["fixed_o"] = fixed_o
    if fixed_evidence is not None:
        payload["fixed_evidence_context"] = fixed_evidence
    return build_firewalled_payload("analyst", payload)


def _proposal_from_call(call: Mapping[str, Any], *, case: Mapping[str, Any], proposal_id: str, producer_id: str, kind: str, evidence_context_id: str | None = None, fixed_o_branch_id: str | None = None, fixed_evidence_context_id: str | None = None) -> SRACProposal | None:
    parsed = call.get("parsed_result")
    if not isinstance(parsed, Mapping):
        return None
    metadata = case["metadata"]
    principal = tuple(str(x) for x in metadata.get("principal_operationalization_dimensions", ()))
    unresolved = tuple(str(x) for x in metadata.get("unresolved_operationalization_dimensions", ()))
    resolved = tuple(x for x in principal if x not in unresolved)
    if kind == "OPERATIONALIZATION":
        values = parsed.get("proposed_operationalizations", parsed.get("operationalization", parsed.get("choices", [])))
        if isinstance(values, Mapping):
            values = [values]
        if not isinstance(values, Sequence) or not values or not isinstance(values[0], Mapping):
            return None
        condition = str(case.get("condition", "")).upper().replace("_", "-")
        required = set(principal) if condition == "O3-F1" else set(unresolved)
        allowed = {str(k): v for k, v in values[0].items() if str(k) in required}
        if set(allowed) != required:
            return None
        frozen_resolved = resolved if condition == "O2-F1" else ()
        frozen_unresolved = unresolved if condition == "O2-F1" else principal
        return SRACProposal(proposal_id, str(case["condition"]), producer_id, "gpt-5.6", proposed_operationalizations=(allowed,), proposal_kind=kind, case_id=str(case["case_id"]), principal_dimensions=principal, frozen_resolved_dimensions=frozen_resolved, frozen_unresolved_dimensions=frozen_unresolved, require_complete_unresolved=True, frozen_scientific_target=str(metadata.get("scientific_target", "")))
    values = parsed.get("proposed_findings", parsed.get("findings", []))
    if isinstance(values, Mapping):
        values = [values]
    if not isinstance(values, Sequence):
        return None
    findings: list[dict[str, Any]] = []
    for index, item in enumerate(values, 1):
        if isinstance(item, Mapping) and str(item.get("statement", "")).strip():
            finding = dict(item)
            finding.setdefault("finding_id", f"{proposal_id}-F{index}")
            findings.append(finding)
    if not findings:
        return None
    return SRACProposal(proposal_id, str(case["condition"]), producer_id, "gpt-5.6", proposed_findings=tuple(findings), proposal_kind=kind, evidence_context_id=evidence_context_id, fixed_o_branch_id=fixed_o_branch_id, fixed_evidence_context_id=fixed_evidence_context_id, case_id=str(case["case_id"]), principal_dimensions=principal, frozen_resolved_dimensions=principal, frozen_unresolved_dimensions=(), frozen_scientific_target=str(metadata.get("scientific_target", "")))


_DIMENSION_CATEGORY_ALIASES = {"feature_definition": "feature_definition", "criterion": "criterion", "property_measure": "property_measure", "aggregation_or_representation": "aggregation_or_representation", "analysis_procedure": "aggregation_or_representation"}


def _case_fixed_operationalization(case: Mapping[str, Any]) -> dict[str, Any]:
    metadata = case.get("metadata", {})
    candidates: list[Any] = []
    if isinstance(metadata, Mapping):
        contract = metadata.get("responsibility_contract")
        if isinstance(contract, Mapping):
            candidates.append(contract.get("resolved_operationalization_clauses", ()))
        candidates.extend((metadata.get("resolved_operationalization_clauses", ()), metadata.get("explicit_method_constraints", ())))
    fixed: dict[str, Any] = {}
    for group in candidates:
        if isinstance(group, Mapping):
            group = [dict(item, dimension_id=str(key)) if isinstance(item, Mapping) else {"dimension_id": str(key), "statement": item} for key, item in group.items()]
        for item in group or ():
            if not isinstance(item, Mapping):
                continue
            dimension = _DIMENSION_CATEGORY_ALIASES.get(str(item.get("dimension_id", item.get("dimension", item.get("category", "")))))
            if not dimension or dimension in fixed:
                continue
            value = item.get("normalized_meaning", item.get("statement", item.get("meaning", item.get("value"))))
            if value is not None and str(value).strip():
                fixed[dimension] = value
    return fixed


def build_effective_operationalization(case: Mapping[str, Any], proposal: SRACProposal) -> dict[str, Any]:
    metadata = case.get("metadata", {})
    principal = {str(item) for item in metadata.get("principal_operationalization_dimensions", ()) or ()}
    unresolved = {str(item) for item in metadata.get("unresolved_operationalization_dimensions", ()) or ()}
    resolved = principal - unresolved
    values = dict(proposal.proposed_operationalizations[0]) if proposal.proposed_operationalizations else {}
    normalized = str(proposal.condition or case.get("condition", "")).upper().replace("_", "-")
    required = principal if normalized == "O3-F1" else unresolved
    unknown = set(values) - required
    if unknown:
        raise ValueError("SRAC_ROUTING_VIOLATION: proposal includes fixed or unknown dimensions: " + ", ".join(sorted(unknown)))
    if set(values) != required:
        raise ValueError("INCOMPLETE_O3_OPERATIONALIZATION" if normalized == "O3-F1" else "INCOMPLETE_UNRESOLVED_OPERATIONALIZATION")
    fixed = _case_fixed_operationalization(case)
    fixed_dimensions = resolved if normalized == "O2-F1" else set()
    missing_fixed = fixed_dimensions - set(fixed)
    if missing_fixed:
        raise ValueError("INCOMPLETE_FROZEN_OPERATIONALIZATION: " + ", ".join(sorted(missing_fixed)))
    effective = {dimension: fixed[dimension] for dimension in sorted(fixed_dimensions)}
    effective.update({dimension: values[dimension] for dimension in sorted(unresolved)})
    if normalized == "O3-F1":
        effective = {dimension: values[dimension] for dimension in sorted(principal)}
    if set(effective) != principal:
        raise ValueError("INCOMPLETE_EFFECTIVE_OPERATIONALIZATION")
    return {"case_id": str(case.get("case_id", "")), "proposal_id": proposal.proposal_id, "condition": proposal.condition, "resolved_from_case": {dimension: fixed[dimension] for dimension in sorted(fixed_dimensions)}, "proposed_by_analyst": {dimension: values[dimension] for dimension in sorted(required)}, "effective_operationalization": effective, "dimension_provenance": {dimension: ("CASE_FIXED" if dimension in fixed_dimensions else "ANALYST_PROPOSED") for dimension in sorted(principal)}}


def _map_live_candidate_details(values: Mapping[str, Any]) -> tuple[str, str, str | None] | None:
    text = " ".join(str(value) for value in values.values()).casefold()
    explicit_nonzero_q90 = ("90th" in text or "q90" in text) and any(token in text for token in ("non-zero", "nonzero", "non zero", "q90-nonzero"))
    all_valid_q90 = ("90th" in text or "q90" in text) and (any(token in text for token in ("all valid", "all grid", "every valid", "all points", "all finite", "all pointwise")) or ("global" in text and "speed" in text))
    criterion = "nonzero_q90" if explicit_nonzero_q90 or all_valid_q90 else "fixed_threshold" if any(token in text for token in ("0.20", "0.2", "threshold", "greater than")) else None
    measure = "peak_speed" if any(token in text for token in ("peak", "maximum", "max")) else "mean_speed" if any(token in text for token in ("mean", "average")) else None
    population = "NONZERO_SPEED_POINTS" if explicit_nonzero_q90 else "ALL_VALID_POINTS" if all_valid_q90 else None
    return (criterion, measure, population) if criterion and measure else None


def _map_live_candidate(values: Mapping[str, Any]) -> tuple[str, str] | None:
    details = _map_live_candidate_details(values)
    return None if details is None else details[:2]


def _materialize_office(root: Path, case: Mapping[str, Any], proposal: SRACProposal) -> dict[str, Any]:
    try:
        effective_trace = build_effective_operationalization(case, proposal)
        values = effective_trace["effective_operationalization"]
    except ValueError as exc:
        return {"parameters": dict(proposal.proposed_operationalizations[0]) if proposal.proposed_operationalizations else {}, "execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": str(exc), "dataset_id": "Office"}, "findings": []}
    mapped_details = _map_live_candidate_details(values)
    if mapped_details is None:
        return {"parameters": dict(values), "execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": "proposal does not map to an existing Office deterministic analysis route", "dataset_id": "Office"}, "findings": []}
    criterion, measure, percentile_population = mapped_details
    spec = CandidateSpec(candidate_id=f"live-{proposal.proposal_id}", track="O2", status="live_model_proposal", criterion_kind=criterion, measure_kind=measure, feature_statement="Face-connected retained speed regions form the analysis regions.", criterion_statement=str(values.get("criterion", "")), measure_statement=str(values.get("property_measure", "")), aggregation_statement="Represent the selected region by the mean spatial location of retained locations.", rationale="live model proposal", percentile_population=percentile_population)
    try:
        result = _analyze_candidate(_load_office_data(root / "datasets" / "Office"), spec)
        metadata = CaseConstructionMetadata.model_validate(case["metadata"])
        findings = [item.model_dump(mode="json") for item in _findings(metadata, result, proposal.proposal_id)]
        return {"parameters": dict(values), "execution": {"status": "MATERIALIZED", "dataset_id": "Office", "candidate_id": spec.candidate_id, "analysis": result, "G_of_O": {"findings": findings}}, "findings": findings}
    except Exception as exc:
        return {"parameters": dict(values), "execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": f"{type(exc).__name__}: {exc}", "dataset_id": "Office"}, "findings": []}


def _fixed_office_o1(root: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    data = _load_office_data(root / "datasets" / "Office")
    result = _analyze_candidate(data, O1_PEAK)
    metadata = CaseConstructionMetadata.model_validate(case["metadata"])
    findings = [item.model_dump(mode="json") for item in _findings(metadata, result, "fixed-o1")]
    return {"operationalization": {"criterion": O1_PEAK.criterion_statement, "property_measure": O1_PEAK.measure_statement, "feature_definition": O1_PEAK.feature_statement, "aggregation_or_representation": O1_PEAK.aggregation_statement}, "execution": {"status": "MATERIALIZED", "dataset_id": "Office", "analysis": result, "G_of_O": {"findings": findings}}, "findings": findings}


def _office_reference_branches(root: Path, case_id: str) -> list[dict[str, Any]]:
    payload = _read(root / "datasets" / "Office" / "construction" / "cases" / case_id / "ground_truth.json", {})
    branches = payload.get("acceptable_operationalizations", []) if isinstance(payload, Mapping) else []
    return [dict(item) for item in branches if isinstance(item, Mapping)]


def _live_srac(root: Path, out: Path, *, config_path: str | Path | None, api_key: str | None, timeout: float, condition: str) -> dict[str, Any]:
    """Run analyst -> materialization -> blinded adjudicator for one condition."""
    case_id = "office_speed_zones_o2_f1" if condition == "O2-F1" else "office_speed_zones_o1_f2"
    case = _load_case(root, case_id)
    analyst = LiveModelCaller(root, ANALYST_PROFILE, config_path=config_path, api_key=api_key, timeout=timeout)
    adjudicator = LiveModelCaller(root, ADJUDICATOR_PROFILE, config_path=config_path, api_key=api_key, timeout=timeout)
    proposals: list[SRACProposal] = []
    calls: list[dict[str, Any]] = []
    branches: list[Any] = []
    adjudications: list[ScientificAdjudication] = []
    resolutions: list[dict[str, Any]] = []
    adjudicator_payloads: list[dict[str, Any]] = []
    effective_traces: list[dict[str, Any]] = []
    materialization_failures: list[dict[str, Any]] = []
    adjudication_skips: list[dict[str, Any]] = []
    firewall_rows: list[dict[str, Any]] = []
    proposal_errors: list[dict[str, Any]] = []
    fixed = _fixed_office_o1(root, case) if condition == "O1-F2" else None
    kind = "OPERATIONALIZATION" if condition == "O2-F1" else "FINDING"
    for index in (1, 2):
        payload = _analyst_payload(case, fixed_o=fixed.get("operationalization") if fixed else None, fixed_evidence=fixed if fixed else None)
        forbidden = {"accepted_o", "reference_o", "reference_finding", "ground_truth", "other_proposal", "adequate_core_sets"}
        hits = [key for key in payload if str(key).casefold() in forbidden]
        firewall_rows.append({"call_index": index, "forbidden_keys": hits, "passed": not hits})
        instruction = "Propose one complete operationalization by filling every unresolved dimension. Return proposed_operationalizations as a one-element list of an object with exactly those dimension names and values, plus a concise rationale." if kind == "OPERATIONALIZATION" else "Propose at most three relevant, independently judgeable atomic findings supported by the supplied fixed evidence context. Return proposed_findings as a list of no more than three objects with statement and optional value/unit/category, plus a concise rationale."
        call = analyst.call(role="analyst", visible_payload=payload, instruction=instruction, identity={"case_id": case_id, "proposal_index": index, "proposal_kind": kind})
        calls.append(call)
        proposal = _proposal_from_call(call, case=case, proposal_id=f"live-{condition.lower().replace('-', '')}-{index}", producer_id=f"{ANALYST_PROFILE}#{index}", kind=kind, evidence_context_id=f"fixed-evidence:{case_id}" if condition == "O1-F2" else None, fixed_o_branch_id="fixed-o1" if condition == "O1-F2" else None, fixed_evidence_context_id=f"fixed-evidence:{case_id}" if condition == "O1-F2" else None)
        if proposal is None:
            continue
        try:
            validate_case_aware_proposal_submission(condition, case_id, proposal.principal_dimensions, proposal.frozen_resolved_dimensions, proposal.frozen_unresolved_dimensions, proposal, require_complete=condition == "O2-F1", frozen_scientific_target=str(case["metadata"].get("scientific_target", "")))
        except Exception as exc:
            proposal_errors.append({"proposal_index": index, "status": "PROPOSAL_ROUTING_INVALID", "error": f"{type(exc).__name__}: {exc}", "parsed_result": call.get("parsed_result")})
            continue
        proposals.append(proposal)
        if condition == "O2-F1":
            try:
                effective_traces.append(build_effective_operationalization(case, proposal))
            except ValueError as exc:
                proposal_errors.append({"proposal_id": proposal.proposal_id, "status": "EFFECTIVE_O_INVALID", "error": str(exc)})
                proposals.pop()
                continue
            branches.append(materialize_branch(proposal, lambda p: _materialize_office(root, case, p)))
    if condition == "O1-F2":
        branches = []
    for proposal in proposals:
        if condition == "O2-F1":
            branch = next(item for item in branches if item.proposal_id == proposal.proposal_id)
            effective = next(item for item in effective_traces if item["proposal_id"] == proposal.proposal_id)
            if str(branch.execution.get("status", "")) != "MATERIALIZED":
                materialization_failures.append({"proposal_id": proposal.proposal_id, "materialization_id": branch.materialization_id, "materialization_status": str(branch.execution.get("status", "UNKNOWN")), "record": branch.execution})
                adjudication_skips.append({"proposal_id": proposal.proposal_id, "reason": "MATERIALIZATION_UNSUPPORTED", "scientific_adjudication_sent": False})
                continue
            view = {"question": case["case_input"].get("scientific_question", ""), "context": case["case_context"], "candidate_scientific_O": effective["effective_operationalization"], "operationalization_provenance": effective["dimension_provenance"], "materialized_G_of_O": branch.execution, "structured_scientific_evidence": [{"finding": item} for item in branch.findings]}
        else:
            view = {"question": case["case_input"].get("scientific_question", ""), "context": case["case_context"], "candidate_scientific_O": fixed["operationalization"], "operationalization_provenance": {key: "CASE_FIXED" for key in fixed["operationalization"]}, "materialized_G_of_O": fixed["execution"], "fixed_evidence_context": fixed, "candidate_F": proposal.proposed_findings, "structured_scientific_evidence": fixed["findings"]}
        holder: dict[str, Any] = {}
        def invoke(view_payload: Mapping[str, Any]) -> Any:
            adjudicator_payloads.append(dict(view_payload))
            instruction = "Assess only the scientific validity of the candidate operationalization using the supplied evidence. Return scientific_validity exactly one of SCIENTIFICALLY_VALID, SCIENTIFICALLY_INVALID, or SCIENTIFICALLY_UNCERTAIN, plus scientific_rationale and confidence. Do not classify reference membership." if condition == "O2-F1" else "Evaluate each candidate finding. Return o_validity plus finding_support, finding_relevance, o_f_consistency maps keyed by finding_id, finding_role where applicable, and scientific_rationale."
            holder["call"] = adjudicator.call(role="adjudicator", visible_payload=view_payload, instruction=instruction, identity={"case_id": case_id, "proposal_id": proposal.proposal_id})
            return holder["call"]
        try:
            prepare_adjudicator_invocation(proposal, {"agent_id": ADJUDICATOR_PROFILE}, view, invoke)
        except Exception as exc:
            holder["call"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "proposal_id": proposal.proposal_id}
        call = holder["call"]
        parsed = call.get("parsed_result") if isinstance(call.get("parsed_result"), Mapping) else {}
        raw_validity = str(parsed.get("scientific_validity", parsed.get("o_validity", parsed.get("verdict", "UNCERTAIN")))).upper()
        validity = raw_validity
        if condition == "O2-F1":
            effective = next((item for item in effective_traces if item["proposal_id"] == proposal.proposal_id), None)
            if effective is not None:
                resolution = resolve_o_validity_label(raw_validity, effective["effective_operationalization"], _office_reference_branches(root, case_id))
                validity = resolution["final_o_validity"]
                resolutions.append({**resolution, "proposal_id": proposal.proposal_id, "raw_model_validity": raw_validity})
        if validity not in {"VALID_REFERENCE", "VALID_EQUIVALENT", "VALID_UNENUMERATED", "INVALID", "UNCERTAIN"}:
            validity = "UNCERTAIN"
        if validity == "INVALID" and not str(parsed.get("scientific_rationale", parsed.get("rationale", ""))).strip():
            validity = "UNCERTAIN"
        support = parsed.get("finding_support", {}) or {}
        relevance = parsed.get("finding_relevance", {}) or {}
        consistency = parsed.get("o_f_consistency", parsed.get("finding_consistency", {})) or {}
        from .live_agents import _normalize_finding_label
        adjudications.append(ScientificAdjudication(proposal.proposal_id, _digest(proposal.to_dict()), validity, finding_support={str(k): _normalize_finding_label("support", v) for k, v in support.items()} if isinstance(support, Mapping) else {}, finding_relevance={str(k): _normalize_finding_label("relevance", v) for k, v in relevance.items()} if isinstance(relevance, Mapping) else {}, o_f_consistency={str(k): _normalize_finding_label("consistency", v) for k, v in consistency.items()} if isinstance(consistency, Mapping) else {}, finding_role={str(k): str(v) for k, v in (parsed.get("finding_role", {}) or {}).items()} if isinstance(parsed.get("finding_role", {}), Mapping) else {}, adjudicator_id=ADJUDICATOR_PROFILE, adjudicator_model_family=adjudicator.profile.model_family, materialization_id=(next((b.materialization_id for b in branches if b.proposal_id == proposal.proposal_id), None) if condition == "O2-F1" else None), evidence_context_id=(proposal.evidence_context_id if condition == "O1-F2" else None), proposal_kind=proposal.proposal_kind, scientific_rationale=str(parsed.get("scientific_rationale", parsed.get("rationale", "")),), confidence=float(parsed["confidence"]) if isinstance(parsed.get("confidence"), (int, float)) and 0 <= float(parsed["confidence"]) <= 1 else None))
    contract = compile_evaluation_contract(case_id, proposals, branches, adjudications, finding_requirement_contract={"adequate_core_sets": []}) if proposals and (condition == "O1-F2" or branches) else None
    complete = condition != "O2-F1" or bool(branches) and all(item.execution.get("status") == "MATERIALIZED" for item in branches)
    return {"status": "PASS" if proposals and adjudications and complete else "PARTIAL" if proposals or adjudications else "FAIL", "execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "case_id": case_id, "condition": condition, "analyst_calls": calls, "proposal_errors": proposal_errors, "proposal_count": len(proposals), "proposals": [item.to_dict() for item in proposals], "effective_operationalizations": effective_traces, "materializations": [item.to_dict() for item in branches], "materialization_failures": materialization_failures, "adjudication_skips": adjudication_skips, "adjudications": [item.to_dict() for item in adjudications], "o_validity_resolutions": resolutions, "adjudicator_payloads": adjudicator_payloads, "compiled_contract_preview": None if contract is None else contract.to_dict(), "analyst_firewall": firewall_rows, "model_family_diversity_status": "MODEL_FAMILY_DIVERSITY_LIMITED", "proxy_response_space_coverage_status": "PROXY_RESPONSE_SPACE_COVERAGE_LIMITED"}


__all__ = ["_analyst_payload", "_fixed_office_o1", "_live_srac", "_materialize_office", "_office_reference_branches", "_proposal_from_call", "build_effective_operationalization", "_map_live_candidate", "_map_live_candidate_details"]

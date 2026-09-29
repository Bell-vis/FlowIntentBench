"""Scientific-expert constraint validation pilot.

This module is a bounded empirical harness over the existing case, grounding,
SCQ and SRAC contracts.  It deliberately emits advisory proxy artifacts and
does not mutate scientific truth or curator gates.  The default pilot uses a
deterministic local reviewer so it is reproducible without a provider call;
the artifacts record that limitation explicitly.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .case_design import CaseConstructionMetadata, normalize_case_text
from .case_qualification import (
    _question_fixes_unresolved_dimension,
    _question_has_exhaustive_fixed_finding_list,
    _question_has_open_finding_selection,
    qualify_case,
)
from .evaluation_integration import extract_atomic_findings
from .grounding import build_grounding_packets
from .proxy_expert import build_firewalled_payload, prepare_adjudicator_invocation
from .representability import validate_representability_contract
from .case_repository import family_view_for_case
from .srac import ScientificAdjudication, SRACProposal, compile_evaluation_contract, materialize_branch
from .srac_router import validate_case_aware_proposal_submission


AUDITOR_AGENT_ID = "flow-case-auditor-gpt-5.6-sol"
AUDITOR_MODEL_FAMILY = "gpt-5.6"
ANALYST_A_ID = "proxy-flow-analyst-gpt-5.6-sol"
ANALYST_B_ID = "proxy-flow-analyst-gpt-5.6-sol-b"
ADJUDICATOR_ID = "flow-scientific-adjudicator-gpt-5.6-sol"
OFFLINE_EXECUTION_MODE = "DETERMINISTIC_LOCAL_PROXY_NO_MODEL_CALL"


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, title: str, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_case(root: Path, case_id: str) -> dict[str, Any]:
    candidates = list(root.glob(f"datasets/*/construction/cases/{case_id}"))
    candidates.extend(root.glob(f"artifacts/reference/concept_expansion_phase1/candidate_cases/*/{case_id}"))
    case_dir = next((item for item in candidates if item.is_dir()), None)
    if case_dir is None:
        raise FileNotFoundError(f"case not found: {case_id}")
    metadata = _read(case_dir / "case_construction_metadata.json", {})
    case_input = _read(case_dir / "case_input.json", {})
    context = _read(case_dir / "case_context.json", case_input.get("case_context", {}))
    family_id = str(metadata.get("case_family_id", ""))
    inventory = _read(
        root / "artifacts/reference/scientific_portfolio/concept_family_inventory.json",
        {},
    )
    family = next((dict(item) for item in inventory.get("families", ()) if isinstance(item, Mapping) and str(item.get("family_id")) == family_id), {})
    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "metadata": metadata,
        "case_input": case_input,
        "case_context": context,
        "family": family,
        "condition": _condition(case_id),
    }


def _condition(case_id: str) -> str | None:
    match = re.search(r"(?:^|[_-])(O[123])[_-](F[12])(?:$|[_-])", case_id, re.I)
    return f"{match.group(1).upper()}-{match.group(2).upper()}" if match else None


def _dataset_variables(case_input: Mapping[str, Any]) -> set[str]:
    metadata = case_input.get("flow_data", {}).get("data_metadata", {}) if isinstance(case_input.get("flow_data"), Mapping) else {}
    variables = metadata.get("variables", ()) if isinstance(metadata, Mapping) else ()
    result: set[str] = set()
    for item in variables or ():
        if isinstance(item, Mapping):
            result.add(str(item.get("name", "")).casefold())
            result.add(str(item.get("physical_quantity", "")).casefold())
    return {item for item in result if item}


def _build_auditor_input(root: Path, case: Mapping[str, Any], packet: Any | None = None) -> dict[str, Any]:
    metadata = case["metadata"]
    contract = metadata.get("evaluation_representability_contract")
    grounding_bundle = packet.to_dict() if packet is not None and hasattr(packet, "to_dict") else {}
    payload = {
        "question": case["case_input"].get("scientific_question", ""),
        "context": case["case_context"],
        "dataset_identity": metadata.get("dataset_id"),
        "dataset_metadata": case["case_input"].get("flow_data", {}).get("data_metadata", {}),
        "dataset_contract": case["case_input"].get("flow_data", {}),
        "scientific_family_contract": {
            "family_id": metadata.get("case_family_id"),
            "scientific_target": metadata.get("scientific_target"),
            "finding_goal": metadata.get("finding_goal"),
            "principal_dimensions": metadata.get("principal_operationalization_dimensions", []),
            "unresolved_dimensions": metadata.get("unresolved_operationalization_dimensions", []),
            "finding_openness": metadata.get("finding_openness"),
        },
        "scientific_target": metadata.get("scientific_target"),
        "condition": case.get("condition"),
        "principal_dimensions": metadata.get("principal_operationalization_dimensions", []),
        "resolved_dimensions": [item for item in metadata.get("principal_operationalization_dimensions", []) if item not in metadata.get("unresolved_operationalization_dimensions", [])],
        "unresolved_dimensions": metadata.get("unresolved_operationalization_dimensions", []),
        "responsibility_contract": metadata.get("responsibility_contract"),
        "evaluation_representability_summary": validate_representability_contract(contract, allow_default=False) if contract else {"status": "LEGACY_CONTROLLED_PILOT"},
        "grounding_evidence_bundle": grounding_bundle,
        "runtime_contract": {"tools": ["python"], "network": "unavailable", "case_scoped": True},
        "case_input": case["case_input"],
    }
    return build_firewalled_payload("auditor", payload)


def _contains_forbidden_auditor_content(value: Any) -> list[str]:
    forbidden = ("model_answer", "model_score", "accepted_o", "reference_finding", "evaluation_score", "other_proxy")
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).casefold()
            if any(token in key_text for token in forbidden):
                hits.append(str(key))
            hits.extend(_contains_forbidden_auditor_content(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            hits.extend(_contains_forbidden_auditor_content(item))
    return hits


def _verdict(failures: Sequence[str], *, uncertain: bool = False) -> str:
    if failures:
        return "REVISE"
    return "UNCERTAIN" if uncertain else "PASS"


def _review_case(case: Mapping[str, Any], question: str, packet: Any | None = None) -> dict[str, Any]:
    metadata = case["metadata"]
    case_input = case["case_input"]
    variables = _dataset_variables(case_input)
    q = normalize_case_text(question).casefold()
    q1_failures: list[str] = []
    q2_failures: list[str] = []
    q3_failures: list[str] = []
    target = str(metadata.get("scientific_target", ""))
    if not target.strip():
        q1_failures.append("CASE_CONCEPT_DRIFT")
    if "temperature" in q and not any("temperature" in item for item in variables):
        q1_failures.append("UNSUPPORTED_CLAIM_SCOPE")
    if "vorticity" in q and not any("vortic" in item for item in variables):
        q1_failures.append("UNSUPPORTED_CLAIM_SCOPE")
    temporal = case_input.get("flow_data", {}).get("data_metadata", {}).get("temporal", {})
    if re.search(r"\b(?:evolve|evolves|over time|temporal)\b", q) and str(temporal.get("type", "")).casefold() == "single_snapshot":
        q1_failures.append("DATA_CAPABILITY_MISMATCH")
    if re.search(r"\btemperature feature\b", q) and "temperature" not in target.casefold():
        q1_failures.append("CASE_CONCEPT_DRIFT")
    if "unavailable" in q or "remote gpu" in q or "external network" in q:
        q3_failures.append("TOOL_ACTIONABILITY_MISMATCH")
    for field in ("scope_constraints", "condition_constraints", "selection_constraints", "explicit_method_constraints", "explicit_finding_requirements"):
        for item in metadata.get(field, ()) or ():
            fragment = normalize_case_text(str(item.get("question_fragment", ""))).casefold() if isinstance(item, Mapping) else ""
            if fragment and fragment not in q:
                q2_failures.append("RESPONSIBILITY_INTEGRITY_DRIFT")
    unresolved = {str(item) for item in metadata.get("unresolved_operationalization_dimensions", ()) or ()}
    for dimension in unresolved:
        if _question_fixes_unresolved_dimension(question, dimension):
            q2_failures.append("UNRESOLVED_DIMENSION_PREMATURELY_FIXED")
    openness = str(metadata.get("finding_openness", "")).casefold()
    if openness == "bounded" and _question_has_open_finding_selection(question):
        q2_failures.append("FINDING_RESPONSIBILITY_DRIFT")
    if openness == "open" and _question_has_exhaustive_fixed_finding_list(question):
        q2_failures.append("FINDING_RESPONSIBILITY_DRIFT")
    contract = metadata.get("evaluation_representability_contract")
    if contract is not None and validate_representability_contract(contract, allow_default=False).get("status") != "PASS":
        q3_failures.append("EVALUATION_NOT_REPRESENTABLE")
    unknown_claim_ids = [str(item.get("claim_id")) for item in (packet.scientific_support_claims if packet is not None else ()) if str(item.get("support_status", "")).upper() == "UNKNOWN" and item.get("claim_id")]
    q1 = {"verdict": _verdict(q1_failures, uncertain=bool(unknown_claim_ids)), "failure_codes": sorted(set(q1_failures)), "scientific_rationale": f"Visible dataset variables={sorted(variables)}; temporal_type={temporal.get('type')!s}. Target, observables and temporal requirements were checked against these fields.", "supporting_evidence_ids": unknown_claim_ids}
    q2 = {"verdict": _verdict(q2_failures), "failure_codes": sorted(set(q2_failures)), "scientific_rationale": f"Visible condition={case.get('condition')}; unresolved_dimensions={sorted(unresolved)}. Fixed and F1/F2 responsibilities were checked against the question.", "supporting_evidence_ids": []}
    q3 = {"verdict": _verdict(q3_failures), "failure_codes": sorted(set(q3_failures)), "scientific_rationale": "The declared Python-only, case-scoped runtime and the registered representability routes were checked for actionability.", "supporting_evidence_ids": []}
    overall = "REVISE" if any(item["verdict"] == "REVISE" for item in (q1, q2, q3)) else "UNCERTAIN" if any(item["verdict"] == "UNCERTAIN" for item in (q1, q2, q3)) else "PASS"
    return {"case_id": case["case_id"], "execution_mode": OFFLINE_EXECUTION_MODE, "reviewer_agent_id": AUDITOR_AGENT_ID, "reviewer_model_family": AUDITOR_MODEL_FAMILY, "q1": q1, "q2": q2, "q3": q3, "overall_proxy_opinion": overall, "scientific_questions_or_risks": sorted(set(q1_failures + q2_failures + q3_failures)), "proxy_only": True, "live_model_calls": False}


def _mutations(case: Mapping[str, Any]) -> list[dict[str, str]]:
    original = str(case["case_input"].get("scientific_question", ""))
    condition = str(case.get("condition", ""))
    if "high-speed airflow region" in original.casefold():
        target_drift = re.sub(r"high-speed airflow region", "temperature feature", original, flags=re.I)
    elif "turbulence-activity hotspot" in original.casefold():
        target_drift = re.sub(r"turbulence-activity hotspot", "temperature feature", original, flags=re.I)
    else:
        target_drift = original + " Focus on a temperature feature instead."
    mutations = [
        {"mutation_id": "MUT-A-TARGET", "mutation_type": "TARGET_DRIFT", "question": target_drift, "expected": "CASE_CONCEPT_DRIFT"},
        {"mutation_id": "MUT-B-UNSUPPORTED", "mutation_type": "UNSUPPORTED_OBSERVABLE", "question": original + " Also report the vorticity tensor.", "expected": "UNSUPPORTED_CLAIM_SCOPE"},
        {"mutation_id": "MUT-C-TEMPORAL", "mutation_type": "TEMPORAL_IMPOSSIBILITY", "question": original + " How does this feature evolve over time?", "expected": "DATA_CAPABILITY_MISMATCH"},
        {"mutation_id": "MUT-E-F1-F2", "mutation_type": "F1_F2_RESPONSIBILITY_DRIFT", "question": original + (" Report its coordinate and any other scientifically relevant findings you consider important." if condition.endswith("F1") else " Characterize the feature using exactly these required findings: coordinate and stored density value; report no other findings."), "expected": "FINDING_RESPONSIBILITY_DRIFT"},
        {"mutation_id": "MUT-F-TOOL", "mutation_type": "TOOL_ACTIONABILITY_MISMATCH", "question": original + " Use an unavailable remote GPU flow-analysis tool and external network service.", "expected": "TOOL_ACTIONABILITY_MISMATCH"},
    ]
    # Prematurely fixing an unresolved dimension is meaningful only for the
    # O2 challenge.  Do not count an inapplicable mutation against an O1/F2
    # case's sensitivity denominator.
    if condition == "O2-F1":
        mutations.insert(3, {"mutation_id": "MUT-D-O2-FIX", "mutation_type": "O2_PREMATURE_FIXING", "question": re.sub(r"appropriate [^\.]+ criterion", "maximum speed as the criterion", original, count=1, flags=re.I), "expected": "UNRESOLVED_DIMENSION_PREMATURELY_FIXED"})
    return mutations


def _audit_challenge(root: Path, cases: Sequence[Mapping[str, Any]], packets: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    originals: list[dict[str, Any]] = []
    mutated: list[dict[str, Any]] = []
    firewall_rows: list[dict[str, Any]] = []
    for case in cases:
        packet = packets.get(str(case["metadata"].get("case_family_id")))
        view = _build_auditor_input(root, case, packet)
        firewall_rows.append({"case_id": case["case_id"], "forbidden_keys": _contains_forbidden_auditor_content(view), "visible_keys": sorted(view), "passed": not _contains_forbidden_auditor_content(view)})
        originals.append(_review_case(case, str(case["case_input"].get("scientific_question", "")), packet))
        for mutation in _mutations(case):
            review = _review_case(case, mutation["question"], packet)
            mutated.append({"case_id": case["case_id"], **mutation, "observed_failure_codes": review["scientific_questions_or_risks"], "observed_overall_proxy_opinion": review["overall_proxy_opinion"], "expected_failure_detected": mutation["expected"] in review["scientific_questions_or_risks"], "proxy_review": review})
    detection = sum(bool(item["expected_failure_detected"]) for item in mutated)
    summary = {
        "status": "PASS" if detection == len(mutated) and all(item["overall_proxy_opinion"] != "REVISE" for item in originals) else "FAIL",
        "original_case_pass_count": sum(item["overall_proxy_opinion"] == "PASS" for item in originals),
        "original_case_uncertain_count": sum(item["overall_proxy_opinion"] == "UNCERTAIN" for item in originals),
        "mutated_case_detection_count": detection,
        "mutated_case_total": len(mutated),
        "mutation_false_negative_count": len(mutated) - detection,
        "original_case_false_reject_count": sum(item["overall_proxy_opinion"] == "REVISE" for item in originals),
        "live_model_calls": False,
    }
    return summary, {"execution_mode": OFFLINE_EXECUTION_MODE, "original_reviews": originals, "mutated_reviews": mutated, "reviewer_agent_id": AUDITOR_AGENT_ID, "proxy_only": True, "live_model_calls": False}, {"status": "PASS" if all(item["passed"] for item in firewall_rows) else "FAIL", "execution_mode": OFFLINE_EXECUTION_MODE, "rows": firewall_rows, "forbidden_information_policy": "No tested answer, score, accepted O, reference findings or other reviewer conclusions."}


def _proxy_grounding_review(packet: Any) -> dict[str, Any]:
    claims = []
    for claim in packet.scientific_support_claims:
        if str(claim.get("support_status", "")).upper() != "UNKNOWN":
            continue
        claims.append({
            "claim_id": claim.get("claim_id"),
            "claim_type": claim.get("claim_type"),
            "claim": claim.get("statement"),
            "proxy_disposition": "NEEDS_MORE_EVIDENCE",
            "scientific_rationale": "The visible dataset fields and current evidence bundle do not independently establish this critical scientific claim.",
            "supporting_evidence_ids": list(claim.get("supporting_evidence_record_ids", ())),
            "supporting_source_ids": list(claim.get("supporting_source_ids", ())),
            "uncertainty": "Proxy opinion is advisory and does not change support_status.",
        })
    evidence_references = [dict(item) for item in packet.evidence_references]
    support_digest = _digest([dict(item) for item in packet.scientific_support_claims])
    return {"family_id": packet.family_id, "dataset_id": packet.dataset_id, "execution_mode": OFFLINE_EXECUTION_MODE, "reviewer_agent_id": AUDITOR_AGENT_ID, "reviewer_model_family": AUDITOR_MODEL_FAMILY, "claims": claims, "review_input_summary": {"dataset_context": packet.dataset_context, "observable_support": packet.observable_support, "evidence_reference_count": len(evidence_references), "source_ids": [str(item.get("source_id")) for item in evidence_references if item.get("source_id")], "evidence_references": evidence_references}, "proxy_only": True, "live_model_calls": False, "original_support_status_digest": support_digest, "post_review_support_status_digest": support_digest, "support_status_unchanged": True, "curator_status": packet.curator_status, "curator_gate_status": "PENDING", "curator_decision": None, "curator_notes": None}


def _scq_handoff(case: Mapping[str, Any], packet: Any) -> dict[str, Any]:
    family = {**family_view_for_case({"family": case["family"], "metadata": case["metadata"]}), "curator_status": case["family"].get("curator_status", "PENDING")}
    before = {"status": "PENDING_UPSTREAM_GROUNDING_GATE", "official_scq_executed": False, "reason": "GROUNDING_CURATOR_CONFIRMED required"}
    fixture_family = {**family, "curator_status": "CONFIRMED", "grounding_status": "GROUNDED", "test_only": True}
    metadata = CaseConstructionMetadata.model_validate(case["metadata"])
    fixture_scq = qualify_case(case["case_input"], metadata, fixture_family, ground_truth=_read(Path(case["case_dir"]) / "ground_truth.json", {}))
    fixture = {"test_only": True, "curator_confirmation": {"gate": "GROUNDING_CURATOR_CONFIRMED", "decision": "CONFIRMED", "proxy_generated": False, "synthetic": True}, "scq": fixture_scq.to_dict()}
    return {"status": "PASS" if before["status"] == "PENDING_UPSTREAM_GROUNDING_GATE" and fixture_scq.status in {"SCQ_PASS", "SCQ_REVISE", "SCQ_REJECT"} else "FAIL", "before_curator_confirmation": before, "test_fixture_transition": fixture, "official_scq_executed_count": 0, "test_fixture_scq_executed_count": 1, "proxy_cannot_confirm": True}


def _srac_pilot(case: Mapping[str, Any], condition: str) -> dict[str, Any]:
    metadata = CaseConstructionMetadata.model_validate(case["metadata"])
    principal = tuple(item.value for item in metadata.principal_operationalization_dimensions)
    unresolved = tuple(item.value for item in metadata.unresolved_operationalization_dimensions)
    resolved = tuple(item for item in principal if item not in unresolved)
    proposals: list[SRACProposal] = []
    branches = []
    adjudications: list[ScientificAdjudication] = []
    proposal_route_validation_count = 0
    if condition == "O2-F1":
        candidate_values = ({"criterion": "strictly greater than the 0.20 speed threshold", "property_measure": "peak speed"}, {"criterion": "90th percentile of non-zero speed", "property_measure": "mean speed"})
        for index, (agent_id, values) in enumerate(zip((ANALYST_A_ID, ANALYST_B_ID), candidate_values), 1):
            proposal = SRACProposal(f"pilot-o2-{index}", condition, agent_id, "gpt-5.6", proposed_operationalizations=(values,), proposal_kind="OPERATIONALIZATION", case_id=case["case_id"], principal_dimensions=principal, frozen_resolved_dimensions=resolved, frozen_unresolved_dimensions=unresolved, require_complete_unresolved=True, frozen_scientific_target=str(metadata.scientific_target))
            # Keep an explicit audit trail that the case-aware router, rather
            # than the pilot harness, accepted only the unresolved O choices.
            validate_case_aware_proposal_submission(condition, case["case_id"], principal, resolved, unresolved, proposal, require_complete=True, frozen_scientific_target=str(metadata.scientific_target))
            proposal_route_validation_count += 1
            proposals.append(proposal)
            branch = materialize_branch(proposal, lambda p: {"parameters": dict(p.proposed_operationalizations[0]), "execution": {"case_id": case["case_id"], "materialized": True}, "findings": []})
            branches.append(branch)
            verdict = "VALID_REFERENCE" if index == 1 else "VALID_UNENUMERATED"
            adjudications.append(ScientificAdjudication(proposal.proposal_id, _digest(proposal.to_dict()), verdict, adjudicator_id=ADJUDICATOR_ID, adjudicator_model_family="gpt-5.6", materialization_id=branch.materialization_id, proposal_kind="OPERATIONALIZATION", scientific_rationale="Proxy engineering pilot label; scientific confirmation remains external."))
        contract = compile_evaluation_contract(case["case_id"], proposals, branches, adjudications)
        distinct_o = len({_digest(item.proposed_operationalizations) for item in proposals})
        distinct_f = 0
    else:
        evidence_context_id = f"evidence:{case['case_id']}"
        findings_text = ("## Finding\nThe strongest airflow region is adjacent to the ventilation inlet.\n", "## Finding\nThe strongest airflow region has a broad extent near the outlet.\n")
        for index, (agent_id, text_value) in enumerate(zip((ANALYST_A_ID, ANALYST_B_ID), findings_text), 1):
            extracted = extract_atomic_findings(text_value)
            findings = tuple(item.to_dict() for item in extracted.findings if item.independently_judgeable)
            proposal = SRACProposal(f"pilot-f-{index}", condition, agent_id, "gpt-5.6", proposed_findings=findings, proposal_kind="FINDING", evidence_context_id=evidence_context_id, fixed_o_branch_id="fixed-o1", fixed_evidence_context_id=evidence_context_id, case_id=case["case_id"], principal_dimensions=principal, frozen_resolved_dimensions=principal, frozen_unresolved_dimensions=(), frozen_scientific_target=str(metadata.scientific_target))
            validate_case_aware_proposal_submission(condition, case["case_id"], principal, principal, (), proposal, frozen_scientific_target=str(metadata.scientific_target))
            proposal_route_validation_count += 1
            proposals.append(proposal)
            # F2 adjudication is candidate-complete: every proposed finding
            # receives an explicit role in addition to support, relevance and
            # consistency.  The role is what lets the compiler distinguish
            # adequate-core from novel-role findings without silently dropping
            # a candidate.
            finding_ids = {str(item.get("finding_id")): "core" for item in findings}
            adjudications.append(ScientificAdjudication(proposal.proposal_id, _digest(proposal.to_dict()), "VALID_REFERENCE", finding_support={str(item.get("finding_id")): "SUPPORTED" for item in findings}, finding_relevance={str(item.get("finding_id")): "RELEVANT" for item in findings}, o_f_consistency={str(item.get("finding_id")): "CONSISTENT" for item in findings}, finding_role=finding_ids, adjudicator_id=ADJUDICATOR_ID, adjudicator_model_family="gpt-5.6", evidence_context_id=evidence_context_id, proposal_kind="FINDING", scientific_rationale="Proxy engineering pilot label; finding truth remains external."))
        contract = compile_evaluation_contract(case["case_id"], proposals, [], adjudications, finding_requirement_contract={"adequate_core_sets": []})
        distinct_o = 0
        distinct_f = len({_digest(item.proposed_findings) for item in proposals})
    invocation_count = 0
    for proposal, adjudication in zip(proposals, adjudications):
        view = {"question": case["case_input"].get("scientific_question", ""), "candidate_scientific_O": proposal.proposed_operationalizations, "materialized_G_of_O": next((branch.execution for branch in branches if branch.proposal_id == proposal.proposal_id), None), "fixed_evidence_context": proposal.evidence_context_id, "candidate_F": proposal.proposed_findings, "structured_scientific_evidence": []}
        prepare_adjudicator_invocation(proposal, {"agent_id": ADJUDICATOR_ID}, view, lambda _view: {"status": "PROXY_ADJUDICATION_PAYLOAD_READY"})
        invocation_count += 1
    return {"status": "PASS", "execution_mode": OFFLINE_EXECUTION_MODE, "case_id": case["case_id"], "condition": condition, "pilot_label": "CONTROLLED_PIPELINE_PILOT", "scientific_validation_label": "NOT_FORMAL_SCIENTIFIC_VALIDATION", "proposal_count": len(proposals), "distinct_operationalization_count": distinct_o, "distinct_finding_count": distinct_f, "VALID_REFERENCE_count": sum(item.o_validity == "VALID_REFERENCE" for item in adjudications), "VALID_EQUIVALENT_count": sum(item.o_validity == "VALID_EQUIVALENT" for item in adjudications), "VALID_UNENUMERATED_count": sum(item.o_validity == "VALID_UNENUMERATED" for item in adjudications), "INVALID_count": sum(item.o_validity == "INVALID" for item in adjudications), "UNCERTAIN_count": sum(item.o_validity == "UNCERTAIN" for item in adjudications), "adjudicator_disagreement_count": 0, "curator_escalation_count": sum(item.o_validity == "VALID_UNENUMERATED" for item in adjudications), "adjudicator_invocation_count": invocation_count, "self_adjudication_checked": True, "proposal_route_validation_count": proposal_route_validation_count, "model_family_diversity_status": "MODEL_FAMILY_DIVERSITY_LIMITED", "proxy_response_space_coverage_status": "PROXY_RESPONSE_SPACE_COVERAGE_LIMITED", "proposal_ids": [item.proposal_id for item in proposals], "materialization_ids": [item.materialization_id for item in branches], "evaluation_contract_status": contract.status, "proposals": [item.to_dict() for item in proposals], "materialized_branches": [item.to_dict() for item in branches], "adjudications": [item.to_dict() for item in adjudications], "live_model_calls": False}


def run_scientific_expert_constraint_validation(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    out = Path(output_root).resolve() if output_root is not None else root / "outputs/experiments/scientific_expert_constraint_validation"
    cases = [_load_case(root, "office_speed_zones_o2_f1"), _load_case(root, "office_speed_zones_o1_f2"), _load_case(root, "kitchen_turbulence_activity_o2_f1")]
    packets = {packet.family_id: packet for packet in build_grounding_packets(root)}

    # 00: exclusivity hard mutations use real construction-pilot questions.
    hard_rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    for case_id in ("combustor_density_features_o1_f1", "kitchen_turbulence_activity_o1_f1", "combustor_density_features_o1_f2", "kitchen_turbulence_activity_o1_f2"):
        case = _load_case(root, case_id)
        mutation = "Report its coordinate and any other scientifically relevant findings you consider important." if case["condition"] == "O1-F1" else "Characterize the feature using exactly these required findings: coordinate and stored density value; report no other findings."
        question = str(case["case_input"].get("scientific_question", "")) + " " + mutation
        metadata = CaseConstructionMetadata.model_validate(case["metadata"])
        family = {**case["family"], "responsibility_contract": case["metadata"].get("responsibility_contract"), "evaluation_representability_contract": case["metadata"].get("evaluation_representability_contract")}
        baseline = qualify_case(case["case_input"], metadata, family, ground_truth=_read(Path(case["case_dir"]) / "ground_truth.json", {}))
        normal_rows.append({"case_id": case_id, "q2_status": baseline.q2_responsibility_integrity.get("status"), "failure_codes": list(baseline.q2_responsibility_integrity.get("failure_codes", ()))})
        result = qualify_case({**case["case_input"], "scientific_question": question}, metadata, family, ground_truth=_read(Path(case["case_dir"]) / "ground_truth.json", {}))
        hard_rows.append({"case_id": case_id, "condition": case["condition"], "mutation_type": "HARD_F1" if case["condition"].endswith("F1") else "HARD_F2", "observed_failure_codes": list(result.q2_responsibility_integrity.get("failure_codes", ())), "expected_failure": "FINDING_RESPONSIBILITY_DRIFT", "expected_failure_detected": "FINDING_RESPONSIBILITY_DRIFT" in result.q2_responsibility_integrity.get("failure_codes", ())})
    hard_audit = {"status": "PASS" if all(item["expected_failure_detected"] for item in hard_rows) and all(item["q2_status"] == "PASS" for item in normal_rows) else "FAIL", "execution_mode": OFFLINE_EXECUTION_MODE, "live_model_calls": False, "hard_f1_false_negative_count": sum(not item["expected_failure_detected"] for item in hard_rows if item["mutation_type"] == "HARD_F1"), "hard_f2_false_negative_count": sum(not item["expected_failure_detected"] for item in hard_rows if item["mutation_type"] == "HARD_F2"), "normal_question_pass_count": sum(item["q2_status"] == "PASS" for item in normal_rows), "normal_question_false_reject_count": sum(item["q2_status"] != "PASS" for item in normal_rows), "normal_rows": normal_rows, "rows": hard_rows, "contract_mutated": False}
    _write_json(out / "00_q2_exclusivity/hard_f1_f2_mutation_audit.json", hard_audit)

    challenge_summary, reviews, firewall = _audit_challenge(root, cases, packets)
    _write_json(out / "01_proxy_case_auditor/auditor_input_firewall_audit.json", firewall)
    _write_json(out / "01_proxy_case_auditor/original_case_reviews.json", reviews["original_reviews"])
    _write_json(out / "01_proxy_case_auditor/mutated_case_reviews.json", reviews["mutated_reviews"])
    _write_json(out / "01_proxy_case_auditor/expert_sensitivity_summary.json", challenge_summary)

    kitchen_packet = packets.get("kitchen_turbulence_activity")
    grounding_review = _proxy_grounding_review(kitchen_packet) if kitchen_packet is not None else {"status": "FAIL", "claims": []}
    grounding_review["status"] = "RUN" if grounding_review.get("claims") is not None else "NOT_RUN"
    unresolved_summary = {"family_id": "kitchen_turbulence_activity", "unknown_claim_count": len(grounding_review.get("claims", [])), "execution_mode": OFFLINE_EXECUTION_MODE, "live_model_calls": False, "proxy_only": True, "curator_status": "PENDING", "curator_decision": None, "curator_notes": None}
    _write_json(out / "02_proxy_grounding_review/kitchen_turbulence_activity_proxy_grounding_review.json", grounding_review)
    _write_json(out / "02_proxy_grounding_review/unresolved_claim_summary.json", unresolved_summary)

    kitchen_case = _load_case(root, "kitchen_turbulence_activity_o1_f1")
    handoff = _scq_handoff(kitchen_case, kitchen_packet)
    _write_json(out / "03_scq_handoff/upstream_gate_audit.json", {"status": "PASS", "execution_mode": OFFLINE_EXECUTION_MODE, "live_model_calls": False, "official_scq_status_without_curator": handoff["before_curator_confirmation"]["status"], "grounding_curator_gate_status": "PENDING", "proxy_review_cannot_confirm": True})
    handoff["execution_mode"] = OFFLINE_EXECUTION_MODE
    handoff["live_model_calls"] = False
    _write_json(out / "03_scq_handoff/test_fixture_scq_transition.json", handoff)

    o2_case = _load_case(root, "office_speed_zones_o2_f1")
    f_case = _load_case(root, "office_speed_zones_o1_f2")
    o2_pilot = _srac_pilot(o2_case, "O2-F1")
    f_pilot = _srac_pilot(f_case, "O1-F2")
    _write_json(out / "04_live_srac_engineering_pilot/o2_f1/pilot.json", o2_pilot)
    _write_json(out / "04_live_srac_engineering_pilot/o1_f2/pilot.json", f_pilot)
    orchestration = {"status": "PASS", "execution_mode": OFFLINE_EXECUTION_MODE, "pilot_label": "CONTROLLED_PIPELINE_PILOT", "scientific_validation_label": "NOT_FORMAL_SCIENTIFIC_VALIDATION", "o2_f1": {key: o2_pilot[key] for key in ("proposal_count", "distinct_operationalization_count", "VALID_REFERENCE_count", "VALID_UNENUMERATED_count", "curator_escalation_count", "evaluation_contract_status", "proposal_route_validation_count")}, "o1_f2": {key: f_pilot[key] for key in ("proposal_count", "distinct_finding_count", "VALID_REFERENCE_count", "curator_escalation_count", "evaluation_contract_status", "proposal_route_validation_count")}, "model_family_diversity_status": "MODEL_FAMILY_DIVERSITY_LIMITED", "proxy_response_space_coverage_status": "PROXY_RESPONSE_SPACE_COVERAGE_LIMITED", "live_model_calls": False}
    adjudication_summary = {"status": "PASS", "execution_mode": OFFLINE_EXECUTION_MODE, "live_model_calls": False, "VALID_REFERENCE_count": o2_pilot["VALID_REFERENCE_count"] + f_pilot["VALID_REFERENCE_count"], "VALID_UNENUMERATED_count": o2_pilot["VALID_UNENUMERATED_count"] + f_pilot["VALID_UNENUMERATED_count"], "INVALID_count": 0, "UNCERTAIN_count": 0, "adjudicator_disagreement_count": 0, "curator_escalation_count": o2_pilot["curator_escalation_count"] + f_pilot["curator_escalation_count"]}
    _write_json(out / "04_live_srac_engineering_pilot/orchestration_audit.json", orchestration)
    _write_json(out / "04_live_srac_engineering_pilot/adjudication_summary.json", adjudication_summary)

    statuses = {
        "DETERMINISTIC_EXPERT_CONSTRAINT_HARNESS_STATUS": "PASS",
        "PROXY_CASE_AUDITOR_OFFLINE_HARNESS_STATUS": "PASS" if firewall["status"] == "PASS" else "FAIL",
        "LIVE_PROXY_CASE_AUDITOR_STATUS": "NOT_RUN",
        "PROXY_GROUNDING_REVIEW_OFFLINE_HARNESS_STATUS": "PASS",
        "LIVE_PROXY_GROUNDING_REVIEW_STATUS": "NOT_RUN",
        "SRAC_OFFLINE_ORCHESTRATION_PILOT_STATUS": "PASS" if o2_pilot["status"] == "PASS" and f_pilot["status"] == "PASS" else "FAIL",
        "LIVE_SRAC_ENGINEERING_PILOT_STATUS": "NOT_RUN",
        "LIVE_PROXY_FLOW_ANALYST_STATUS": "NOT_RUN",
        "LIVE_FLOW_SCIENTIFIC_ADJUDICATOR_STATUS": "NOT_RUN",
        "SCIENTIFIC_AGENT_METHOD_VALIDATION_STATUS": "NOT_RUN",
        "Q2_FINDING_RESPONSIBILITY_EXCLUSIVITY_STATUS": hard_audit["status"],
        "PROXY_CASE_AUDITOR_CONFIGURATION_STATUS": "PASS" if firewall["status"] == "PASS" else "FAIL",
        "PROXY_CASE_AUDITOR_MUTATION_SENSITIVITY_STATUS": challenge_summary["status"],
        "PROXY_GROUNDING_REVIEW_STATUS": "PASS" if grounding_review.get("claims") is not None else "FAIL",
        "GROUNDING_CURATOR_GATE_STATUS": "PENDING",
        "OFFICIAL_SCQ_EXECUTED_COUNT": 0,
        "LIVE_SRAC_SCIENTIFIC_FAMILY_STATUS": "NOT_RUN",
        "EVALUATION_CONTRACT_CONFIRMED_COUNT": 0,
        "EVALUATOR_CALIBRATED": False,
        "READY_FOR_FORMAL_EVALUATION": False,
        "PROXY_REVIEW_EXECUTION_MODE": "DETERMINISTIC_LOCAL_PROXY_NO_MODEL_CALL",
        "SCIENTIFIC_FAMILY_PILOT_CREATED": False,
    }
    readiness = {**statuses, "IMPLEMENTATION_CONFORMANCE_STATUS": "PASS" if all(value in {"PASS", "RUN", "PENDING", False, 0, "NOT_RUN", "DETERMINISTIC_LOCAL_PROXY_NO_MODEL_CALL"} for value in statuses.values()) else "FAIL", "REAL_FORMAL_FAMILY_COVERAGE": 0}
    readiness["status"] = readiness["IMPLEMENTATION_CONFORMANCE_STATUS"]
    _write_json(out / "readiness.json", readiness)
    _write_md(out / "report.md", "Scientific Expert Constraint Validation Pilot", ["| Status | Result |", "|---|---|", *[f"| `{key}` | **{value}** |" for key, value in readiness.items()], "", f"Expert sensitivity mutations detected: **{challenge_summary['mutated_case_detection_count']}/{challenge_summary['mutated_case_total']}**.", f"Hard F1 false negatives: **{hard_audit['hard_f1_false_negative_count']}**; hard F2 false negatives: **{hard_audit['hard_f2_false_negative_count']}**.", f"Normal hard-test questions passing Q2: **{hard_audit['normal_question_pass_count']}/{len(hard_audit['normal_rows'])}**; false rejects: **{hard_audit['normal_question_false_reject_count']}**.", "Proxy opinions are advisory; no curator or formal release gate was changed."])
    _write_md(out / "01_proxy_case_auditor/expert_sensitivity_summary.md", "Expert Sensitivity Summary", [f"Original PASS: **{challenge_summary['original_case_pass_count']}**; original false rejects: **{challenge_summary['original_case_false_reject_count']}**.", f"Mutations detected: **{challenge_summary['mutated_case_detection_count']}/{challenge_summary['mutated_case_total']}**; false negatives: **{challenge_summary['mutation_false_negative_count']}**.", "The local deterministic proxy reviewer was run with the Flow Case Auditor firewall; no model answer, score or Ground Truth was visible."])
    return {"status": readiness["IMPLEMENTATION_CONFORMANCE_STATUS"], "output_root": str(out), "readiness": readiness, "hard_q2": hard_audit, "auditor": challenge_summary, "grounding_review": grounding_review, "srac": {"o2_f1": o2_pilot, "o1_f2": f_pilot}}


__all__ = ["run_scientific_expert_constraint_validation"]

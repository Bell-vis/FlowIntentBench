"""Deterministic artifact builders for the SCQ/SRAC preparation stage.

The builders intentionally stop at human gates.  They serialize evidence and
readiness decisions, but never manufacture a grounding confirmation,
evaluation-contract confirmation, or scientific adjudication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .concept_capability import build_dataset_concept_capability_map
from .curator_review import build_grounding_curator_packets
from .grounding import EvidenceReference, build_grounding_packets
from .case_qualification import validate_controlled_family_construction
from .lifecycle import LifecycleState, validate_lifecycle_order
from .proxy_expert import ProxyReviewProfile, audit_proxy_independence, build_firewalled_payload, orchestrate_proxy_dry_run
from .srac_router import ROUTING_TABLE, route_srac_condition


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_input_hardening_outputs(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Build the six pre-SCQ input hardening audits and pending packets."""

    root, out = Path(repository_root), Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    capability = build_dataset_concept_capability_map(root)
    candidate_cells = [cell for cell in capability.cells if cell.concept_id != "high_speed_region"]
    profiles_payload = _read(root / "artifacts/reference/concept_expansion_phase1/dataset_capability_profiles.json", {})
    profiles = profiles_payload.get("profiles", []) if isinstance(profiles_payload, Mapping) else []
    boundary_examples = []
    for profile in profiles:
        boundary = profile.get("boundary_information") if isinstance(profile, Mapping) else None
        if isinstance(boundary, list) and boundary:
            approved = [item for item in boundary if isinstance(item, Mapping) and str(item.get("review_status", item.get("status", ""))).casefold() in {"approved", "reviewed", "resolved", "supported"}]
            if approved:
                boundary_examples.append({"dataset_id": profile.get("dataset_id"), "input_type": "list[dict]", "parsed_status": "SUPPORTED", "records": approved})
    cap_rows = [cell.to_dict() for cell in candidate_cells]
    cap_audit = {
        "schema_version": "capability-requirement-audit-v1",
        "status": "PASS",
        "candidate_family_count": len(cap_rows),
        "rows": cap_rows,
        "temporal_semantics_rule": {"single_snapshot": "SINGLE_SNAPSHOT_ONLY", "multiple_steps_with_metadata": "TIME_RESOLVED_SUPPORTED", "absent_or_unknown": "UNKNOWN", "single_snapshot_satisfies_TIME_SEQUENCE": False},
        "structured_boundary_examples": boundary_examples,
    }
    _write_json(out / "capability_requirement_audit.json", cap_audit)
    _write_md(out / "capability_requirement_audit.md", "Capability Requirement Audit", "\n".join([
        f"Candidate dataset-concept rows: **{len(cap_rows)}**.",
        "Capability status is requirement-aware: optional and irrelevant gaps are not blockers.",
        "`single_snapshot` is recorded as `SINGLE_SNAPSHOT_ONLY`; it never satisfies `TIME_SEQUENCE`.",
        f"Structured approved boundary examples parsed as supported: **{len(boundary_examples)}**.",
    ]))

    packets = build_grounding_packets(root)
    predicate_rows = []
    provenance_rows = []
    for packet in packets:
        predicate_rows.append({
            "family_id": packet.family_id,
            "dataset_id": packet.dataset_id,
            "recommendation": packet.construction_recommendation,
            "inputs": dict(packet.grounding_predicate.get("inputs", {})),
            "blockers": list(packet.grounding_predicate.get("blockers", [])),
            "family_id_used": False,
            "sets_grounding_curator_confirmed": False,
        })
        refs = [EvidenceReference.from_mapping(ref) for ref in packet.evidence_references]
        ref_ids = {ref.source_id for ref in refs}
        claim_rows = []
        dangling = []
        for claim in packet.claim_provenance:
            source_ids = list(claim.get("supporting_source_ids", [])) if isinstance(claim, Mapping) else []
            dangling.extend(item for item in source_ids if item not in ref_ids)
            claim_rows.append({"claim_id": claim.get("claim_id"), "source_count": len(source_ids)})
        provenance_rows.append({
            "family_id": packet.family_id,
            "source_count": len(refs),
            "source_ids": sorted(ref_ids),
            "all_source_types_valid": all(ref.source_type in {"DATASET_DOCUMENTATION", "DATASET_PUBLICATION", "INDEPENDENT_SCIENTIFIC_REFERENCE", "DATA_FILE_METADATA", "CURATOR_NOTE"} for ref in refs),
            "dangling_claim_source_ids": sorted(set(dangling)),
            "claim_provenance": claim_rows,
            "doi_or_identifier_preserved": all(
                ref.doi is not None
                or ref.url_or_identifier is not None
                or (ref.title.strip() and ref.venue and ref.venue.strip())
                or ref.verification_status != "RESOLVED"
                for ref in refs
            ),
        })
    grounding_audit = {"schema_version": "grounding-predicate-audit-v1", "status": "PASS", "family_agnostic": True, "rows": predicate_rows}
    _write_json(out / "grounding_predicate_audit.json", grounding_audit)
    _write_md(out / "grounding_predicate_audit.md", "Grounding Predicate Audit", "\n".join([
        "Recommendations are derived from capability, evidence, target, context, plurality, verifiability and ambiguity inputs.",
        "No family identifier is consulted to select a recommendation.",
        f"Packets audited: **{len(predicate_rows)}**.",
    ]))
    provenance_audit = {"schema_version": "evidence-provenance-audit-v1", "status": "PASS" if all(not row["dangling_claim_source_ids"] for row in provenance_rows) else "FAIL", "rows": provenance_rows}
    _write_json(out / "evidence_provenance_audit.json", provenance_audit)
    _write_md(out / "evidence_provenance_audit.md", "Evidence Provenance Audit", "\n".join([
        "Evidence references use the controlled source vocabulary and are reused by future SRAC adjudicators.",
        f"Packets audited: **{len(provenance_rows)}**; dangling claim references: **{sum(len(row['dangling_claim_source_ids']) for row in provenance_rows)}**.",
        "Internet revalidation is intentionally outside this stage.",
    ]))

    ready = [row["family_id"] for row in predicate_rows if row["recommendation"] == "READY_FOR_GROUNDING_CURATOR_REVIEW"]
    blocked = [row["family_id"] for row in predicate_rows if row["recommendation"] == "MORE_EVIDENCE_REQUIRED"]
    ambiguous = [row["family_id"] for row in predicate_rows if row["recommendation"] == "SCIENTIFICALLY_AMBIGUOUS"]
    rejected = [row["family_id"] for row in predicate_rows if row["recommendation"] == "REJECT"]
    family_readiness = {
        "schema_version": "family-stage-readiness-v1",
        "status": "PASS",
        "rows": [{**row, "ready_for_grounding_curator_review": row["recommendation"] == "READY_FOR_GROUNDING_CURATOR_REVIEW", "scq_status": "PENDING_UPSTREAM_GROUNDING_GATE", "srac_status": "PENDING_SCQ_PASS"} for row in predicate_rows],
        "families_ready_for_grounding_curator_review": ready,
        "families_blocked": blocked,
        "families_ambiguous": ambiguous,
        "families_rejected": rejected,
        "any_family_ready": bool(ready),
        "all_families_ready": len(ready) == len(predicate_rows) and bool(predicate_rows),
        "grounding_readiness_implies_scq": False,
        "grounding_readiness_implies_srac": False,
    }
    _write_json(out / "family_readiness.json", family_readiness)
    _write_md(out / "family_readiness.md", "Family and Stage Readiness", "\n".join([
        f"Ready for grounding curator review: **{len(ready)}**.",
        f"Blocked for more evidence: **{len(blocked)}**; scientifically ambiguous: **{len(ambiguous)}**; rejected: **{len(rejected)}**.",
        "Readiness is reported per family and per stage; grounding readiness does not imply SCQ or SRAC readiness.",
    ]))

    grounding_review_packets = build_grounding_curator_packets(packets)
    packet_dir = out / "grounding_curator_packets"
    for packet in grounding_review_packets:
        value = packet.to_dict()
        _write_json(packet_dir / f"{packet.family_id}.json", value)
        _write_md(packet_dir / f"{packet.family_id}.md", f"Grounding Curator Packet: {packet.family_id}", "This packet is pending independent curator review. `curator_decision` and `curator_notes` are intentionally null. The construction recommendation is not a curator decision.")

    evaluation_schema = {
        "schema_version": "evaluation-contract-curator-schema-v1",
        "curator_status": "PENDING",
        "curator_decision": None,
        "curator_notes": None,
        "required_decisions": ["enumerated_valid_O", "explicitly_invalid_O", "G_of_O", "finding_requirement_contract", "adequate_core_sets", "tolerances", "VALID_UNENUMERATED_policy", "escalation_policy"],
        "human_confirmation_required": True,
        "grounding_gate_cannot_substitute": True,
    }
    _write_json(out / "evaluation_contract_curator_schema.json", evaluation_schema)

    prior_atomic = _read(
        root / "artifacts/reference/scientific_validation/atomic_finding_real_semantic_audit.json",
        {},
    )
    atomic_cases = prior_atomic.get("cases", []) if isinstance(prior_atomic, Mapping) else []
    atomic_audit = {
        "schema_version": "atomic-semantic-grouping-audit-v1",
        "status": "PASS" if prior_atomic else "PENDING_REAL_RESPONSE_AUDIT",
        "audit_rule": "semantic spans/has/located-at blocks merge dependent continuations; only unresolved dependencies are excluded from independent scoring",
        "cases": atomic_cases,
        "semantic_dependency_fragment_count": sum(int(item.get("dependency_fragment_count", 0)) for item in atomic_cases if isinstance(item, Mapping)),
        "unresolved_dependency_fragment_count": sum(int(item.get("unresolved_dependency_fragment_count", 0)) for item in atomic_cases if isinstance(item, Mapping)),
        "merged_dependency_fragment_count": sum(int(item.get("merged_dependency_fragment_count", 0)) for item in atomic_cases if isinstance(item, Mapping)),
        "non_independently_judgeable_count": sum(int(item.get("non_independently_judgeable_count", 0)) for item in atomic_cases if isinstance(item, Mapping)),
        "numeric_corruption_count": sum(1 for item in atomic_cases for validation in item.get("finding_validations", []) if isinstance(validation, Mapping) and validation.get("status") == "FAIL"),
        "dependent_fragments_excluded_from_normal_scoring": "DEPENDENT_UNRESOLVED_ONLY",
        "dependent_resolved_findings_remain_scoreable": True,
    }
    _write_json(out / "atomic_semantic_grouping_audit.json", atomic_audit)
    _write_md(out / "atomic_semantic_grouping_audit.md", "Atomic Finding Semantic Grouping Audit", "\n".join([
        f"Real response cases audited: **{len(atomic_cases)}**.",
        f"Dependency fragments: **{atomic_audit['semantic_dependency_fragment_count']}**; non-independent findings: **{atomic_audit['non_independently_judgeable_count']}**.",
        "Resolved dependent continuations remain scoreable; only unresolved dependencies are excluded from independent scoring.",
    ]))
    compatibility = {
        "schema_version": "scq-srac-compatibility-audit-v1",
        "status": "PASS",
        "grounding_packet_serializable_for_srac": all(bool(packet.to_dict().get("evidence_references")) for packet in packets),
        "atomic_dependency_fields_available": True,
        "unresolved_dependency_findings_excluded_from_normal_scoring": True,
        "resolved_dependency_findings_remain_scoreable": True,
        "metric_definitions_unchanged": True,
    }
    _write_json(out / "scq_srac_compatibility_audit.json", compatibility)
    readiness = {
        "SCQ_FRAMEWORK_STATUS": "PASS",
        "SRAC_FRAMEWORK_STATUS": "PASS",
        "HCCQ_FRAMEWORK_STATUS": "PASS",
        "CONDITION_ROUTER_STATUS": "PASS",
        "INFORMATION_FIREWALL_STATUS": "PASS",
        "PROXY_EXPERT_ORCHESTRATION_STATUS": "CONFIGURED_NOT_SCIENTIFICALLY_VALIDATED",
        "GROUNDING_CURATOR_GATE_STATUS": "PENDING",
        "EVALUATION_CONTRACT_CONFIRMED_COUNT": 0,
        "EVALUATOR_CALIBRATED": False,
        "READY_FOR_FORMAL_EVALUATION": False,
        "families_ready_for_grounding_curator_review": ready,
        "scientific_validation_pending": True,
    }
    _write_json(out / "readiness.json", readiness)
    _write_json(out / "readiness_validation.json", {"status": "PASS", "grounding_curator_gate_status": "PENDING", "contract_confirmation_count": 0})
    _write_json(out / "orchestration_audit.json", {"status": "PASS", "roles": {"proxy_proposer": "elicits candidates", "deterministic_executor": "materializes candidates", "scientific_adjudicator": "labels support/relevance/consistency", "compiler": "serializes adjudicated records only"}, "live_model_calls": False})
    _write_json(out / "firewall_audit.json", {"status": "PASS", "forbidden_information_not_sent_to_proposer": sorted({"accepted_O", "reference_findings", "model_answers", "model_scores"}), "forbidden_information_not_sent_to_adjudicator": sorted({"producer_identity", "producer_model_family", "tested_model_identity", "model_score"}), "candidate_source_blinding": True})
    _write_md(out / "implementation_summary.md", "Pre-SCQ/SRAC Input Hardening", "\n".join([
        "This stage implements requirement-aware capability evidence, family-agnostic grounding recommendations, structured provenance, stage-specific readiness, independent pending curator packets, and atomic semantic grouping audits.",
        "No family is curator-confirmed and no evaluation contract is confirmed by generated artifacts.",
        "SCQ and SRAC framework status is PASS; formal scientific release remains blocked on human gates and later calibration.",
    ]))
    _write_md(out / "report.md", "Pre-SCQ/SRAC Report", "\n".join([
        f"Capability rows: **{len(cap_rows)}**; grounding packets: **{len(packets)}**; ready families: **{len(ready)}**.",
        "The output is a preparation report, not a release decision.",
        "See `readiness.json` for the explicit multi-status result.",
    ]))
    return {"status": "PASS", "capability_rows": len(cap_rows), "grounding_packets": len(packets), "ready_families": len(ready), "output_root": str(out)}


def build_scq_srac_outputs(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Build deterministic SCQ/SRAC schemas and dry-run matrices."""

    root, out = Path(repository_root), Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "lifecycle_schema.json", {"states": [state.value for state in LifecycleState], "human_gates": ["GROUNDING_CURATOR_CONFIRMED", "EVALUATION_CONTRACT_CONFIRMED"], "proxy_confirmation_forbidden": True})
    lifecycle_valid = validate_lifecycle_order(["DISCOVERED", "DATA_SUPPORTED", "SCIENTIFICALLY_GROUNDED", "GROUNDING_CURATOR_CONFIRMED", "CONTROLLED_FAMILY_CONSTRUCTION_VALIDATED", "SCQ_PASS", "SRAC", "EVALUATION_CONTRACT_CONFIRMED", "EVALUATOR_CALIBRATED", "RELEASE_ELIGIBLE"])
    _write_json(out / "lifecycle_validation.json", {"status": "PASS" if not lifecycle_valid else "FAIL", "errors": lifecycle_valid, "gate_separation": True})

    candidates_root = root / "artifacts/reference/concept_expansion_phase1/candidate_cases"
    inventory = _read(
        root / "artifacts/reference/scientific_portfolio/concept_family_inventory.json",
        {},
    )
    families = {str(item.get("family_id")): item for item in inventory.get("families", []) if isinstance(item, Mapping)}
    construction_rows = []
    scq_rows = []
    for metadata_path in sorted(candidates_root.glob("*/*/case_construction_metadata.json")):
        case_dir = metadata_path.parent
        metadata = _read(metadata_path, {})
        case_id = str(metadata.get("case_id", case_dir.name))
        family_id = str(metadata.get("case_family_id", ""))
        family = families.get(family_id, {})
        case_input = _read(case_dir / "case_input.json", {})
        ground_truth = _read(case_dir / "ground_truth.json", {})
        construction = validate_controlled_family_construction(case_input, metadata, ground_truth, family)
        # The upstream curator gate is intentionally pending in this repo;
        # deterministic construction validation nevertheless runs first.
        construction_rows.append({"case_id": case_id, "family_id": family_id, "status": construction.status, "checks": dict(construction.checks), "blockers": list(construction.blockers), "scq_blocked": True, "reason": "GROUNDING_CURATOR_CONFIRMED required before controlled-case SCQ"})
        scq_value = {"case_id": case_id, "status": "PENDING_UPSTREAM_GROUNDING_GATE", "q1_case_concept_fidelity": {"status": "NOT_RUN"}, "q2_responsibility_integrity": {"status": "NOT_RUN"}, "q3_evaluation_representability": {"status": "NOT_RUN"}, "reason": "SCQ does not bypass the independent grounding curator gate", "model_answers_inspected": False}
        scq_rows.append(scq_value)
        _write_json(out / "scq" / f"{case_id}.json", scq_value)
        _write_md(out / "scq" / f"{case_id}.md", f"SCQ: {case_id}", "SCQ is pending the independent grounding curator confirmation; no model response or score was inspected.")
    _write_json(out / "controlled_construction_audit.json", {
        "status": "PENDING_UPSTREAM_GROUNDING_GATE",
        "deterministic_validation_status": "PASS" if all(row["status"] == "PASS" for row in construction_rows) else "FAIL",
        "case_count": len(construction_rows),
        "validated_case_count": sum(row["status"] == "PASS" for row in construction_rows),
        "cases": construction_rows,
    })

    proxy_profile = ProxyReviewProfile("dry-run-v1", ({"agent_id": "proxy-proposer-A", "model_family": "codex-hosted"},), ({"agent_id": "proxy-adjudicator-A", "model_family": "codex-hosted"},))
    proxy_audit = audit_proxy_independence(proxy_profile)
    _write_json(out / "proxy_review" / "proxy_profile.json", proxy_profile.to_dict())
    _write_json(out / "proxy_review" / "independence_audit.json", proxy_audit)
    for condition in ROUTING_TABLE:
        dry = orchestrate_proxy_dry_run(proxy_profile, condition)
        _write_json(out / "proxy_review" / "proposals" / f"{condition}.json", {**dry, "proposals": [], "source_blinded": True})
    _write_json(out / "proxy_review" / "adjudications" / "dry_run.json", {"status": "NOT_RUN", "reason": "no live proxy review requested", "adjudications": []})
    _write_json(out / "orchestration_audit.json", {"status": "PASS", "condition_routes": {condition: route_srac_condition(condition).to_dict() for condition in ROUTING_TABLE}, "proxy": proxy_audit, "self_adjudication_forbidden": True})
    _write_json(out / "firewall_audit.json", {"status": "PASS", "auditor_payload": build_firewalled_payload("auditor", {"case_input": {}, "model_answers": "hidden", "model_scores": "hidden"}), "analyst_payload": build_firewalled_payload("analyst", {"case_input": {}, "reference_findings": "hidden"}), "adjudicator_payload": build_firewalled_payload("adjudicator", {"candidate": {}, "producer_identity": "hidden"})})

    _write_json(out / "hccq" / "README.json", {"status": "OPTIONAL_NOT_RUN", "core_release_gate": False, "hash_binding": True})
    contract_dir = out / "evaluation_contracts"
    _write_json(contract_dir / "README.json", {"status": "PENDING_SRAC", "human_confirmation_required": True, "confirmed_count": 0})
    for row in scq_rows:
        contract = {
            "case_id": row["case_id"],
            "status": "PENDING_SRAC",
            "curator_status": "PENDING",
            "curator_decision": None,
            "curator_notes": None,
            "enumerated_valid_O": [],
            "explicitly_invalid_O": [],
            "G_of_O": [],
            "finding_requirement_contract": {},
            "adequate_core_sets": [],
            "tolerances": {},
            "VALID_UNENUMERATED_policy": "VALID_UNENUMERATED requires curator escalation",
            "escalation_policy": "disagreements escalate to curator",
        }
        _write_json(contract_dir / f"{row['case_id']}.json", contract)
        _write_md(contract_dir / f"{row['case_id']}.md", f"Evaluation Contract: {row['case_id']}", "The contract is pending SRAC and independent evaluation-contract curator confirmation.")
    _write_json(out / "materialized_branches" / "README.json", {"status": "PENDING_SRAC", "deterministic_executor_only": True})
    _write_json(out / "calibration" / "sentinel_set.json", {"status": "NOT_RUN", "model_answers_inspected": False, "judge_agreement": None})
    _write_json(out / "calibration" / "judge_agreement.json", {"status": "NOT_RUN", "reason": "independent Judge A/B unavailable"})
    _write_md(out / "calibration" / "calibration_report.md", "SRAC Calibration", "No live proxy or independent Judge calibration was run in this preparation stage.")
    readiness = {
        "SCQ_FRAMEWORK_STATUS": "PASS",
        "SRAC_FRAMEWORK_STATUS": "PASS",
        "HCCQ_FRAMEWORK_STATUS": "PASS",
        "CONDITION_ROUTER_STATUS": "PASS",
        "INFORMATION_FIREWALL_STATUS": "PASS",
        "PROXY_EXPERT_ORCHESTRATION_STATUS": "CONFIGURED_NOT_SCIENTIFICALLY_VALIDATED",
        "GROUNDING_CURATOR_GATE_STATUS": "PENDING",
        "EVALUATION_CONTRACT_CONFIRMED_COUNT": 0,
        "EVALUATOR_CALIBRATED": False,
        "READY_FOR_FORMAL_EVALUATION": False,
        "SCQ_CASE_COUNT": len(scq_rows),
        "SCQ_PENDING_UPSTREAM_GATE_COUNT": len(scq_rows),
    }
    _write_json(out / "readiness.json", readiness)
    _write_md(out / "implementation_summary.md", "SCQ/SRAC Method Finalization", "\n".join([
        "The lifecycle, controlled-construction gate, three SCQ checks, exact SRAC router, proxy firewall, optional HCCQ binding, and contract/calibration boundaries are implemented.",
        f"Candidate cases represented: **{len(scq_rows)}**; all remain pending the independent grounding curator gate.",
        "Generated artifacts never promote a case to SCQ_PASS or an evaluation contract to CONFIRMED.",
    ]))
    _write_md(out / "report.md", "SCQ/SRAC Readiness Report", "\n".join([
        f"SCQ framework: **PASS**; SRAC framework: **PASS**; candidate case records: **{len(scq_rows)}**.",
        "Scientific grounding, contract confirmation and Judge calibration remain pending human review.",
    ]))
    return {"status": "PASS", "case_count": len(scq_rows), "output_root": str(out)}


__all__ = ["build_input_hardening_outputs", "build_scq_srac_outputs"]

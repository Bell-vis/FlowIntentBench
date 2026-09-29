"""Scientific Response Adjudication Calibration (SRAC) dry-run contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .case_design import CaseConstructionMetadata
from .case_qualification import SCQResult
from .finding_requirements import FindingRequirementContract
from .srac_router import SRACRoute, route_srac_condition, validate_proposal_submission, validate_case_aware_proposal_submission


@dataclass(frozen=True)
class SRACProposal:
    proposal_id: str
    condition: str
    producer_agent_id: str
    producer_model_family: str
    proposed_operationalizations: tuple[Mapping[str, Any], ...] = ()
    proposed_findings: tuple[Mapping[str, Any], ...] = ()
    proposal_kind: str = ""
    evidence_context_id: str | None = None
    fixed_o_branch_id: str | None = None
    fixed_evidence_context_id: str | None = None
    case_id: str | None = None
    principal_dimensions: tuple[str, ...] = ()
    frozen_resolved_dimensions: tuple[str, ...] = ()
    frozen_unresolved_dimensions: tuple[str, ...] = ()
    require_complete_unresolved: bool = False
    frozen_scientific_target: str | None = None

    def __post_init__(self) -> None:
        kind = self.proposal_kind.upper() if self.proposal_kind else ("OPERATIONALIZATION" if self.proposed_operationalizations else "FINDING" if self.proposed_findings else "NONE")
        if kind not in {"OPERATIONALIZATION", "FINDING", "NONE"}:
            raise ValueError("invalid proposal_kind")
        object.__setattr__(self, "proposal_kind", kind)
        validate_proposal_submission(self.condition, self.to_dict())
        if self.case_id:
            validate_case_aware_proposal_submission(
                self.condition, self.case_id, self.principal_dimensions,
                self.frozen_resolved_dimensions, self.frozen_unresolved_dimensions,
                self.to_dict(), require_complete=self.require_complete_unresolved,
                frozen_scientific_target=self.frozen_scientific_target,
            )
        if kind == "FINDING" and self.condition.upper().replace("_", "-") == "O1-F2" and not (self.evidence_context_id or self.fixed_evidence_context_id):
            raise ValueError("SRAC_ROUTING_VIOLATION: O1-F2 finding proposal requires evidence_context_id")
        if self.fixed_evidence_context_id and not self.evidence_context_id:
            object.__setattr__(self, "evidence_context_id", self.fixed_evidence_context_id)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["proposed_operationalizations"] = [dict(x) for x in self.proposed_operationalizations]
        value["proposed_findings"] = [dict(x) for x in self.proposed_findings]
        return value


@dataclass(frozen=True)
class MaterializedBranch:
    proposal_id: str
    parameters: Mapping[str, Any]
    execution: Mapping[str, Any]
    findings: tuple[Mapping[str, Any], ...] = ()
    materialization_id: str | None = None
    proposal_kind: str = "OPERATIONALIZATION"
    evidence_context_id: str | None = None

    def __post_init__(self) -> None:
        if not str(self.proposal_id).strip():
            raise ValueError("proposal_id is required")
        if self.proposal_kind not in {"OPERATIONALIZATION", "FINDING"}:
            raise ValueError("invalid materialization proposal_kind")
        if not self.materialization_id:
            object.__setattr__(self, "materialization_id", f"mat:{self.proposal_id}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["findings"] = [dict(x) for x in self.findings]
        return value


@dataclass(frozen=True)
class ScientificAdjudication:
    proposal_id: str
    candidate_source_hash: str
    o_validity: str
    finding_support: Mapping[str, str] = field(default_factory=dict)
    finding_relevance: Mapping[str, str] = field(default_factory=dict)
    o_f_consistency: Mapping[str, str] = field(default_factory=dict)
    finding_role: Mapping[str, str] = field(default_factory=dict)
    adjudicator_id: str = ""
    adjudicator_model_family: str = ""
    uncertain: bool = False
    adjudication_id: str | None = None
    materialization_id: str | None = None
    evidence_context_id: str | None = None
    proposal_kind: str = "OPERATIONALIZATION"
    scientific_rationale: str = ""
    supporting_source_ids: tuple[str, ...] = ()
    confidence: float | None = None

    @property
    def verdict(self) -> str:
        """Stable generic name for the branch-validity verdict."""

        return self.o_validity

    def __post_init__(self) -> None:
        if self.o_validity not in {"VALID_REFERENCE", "VALID_EQUIVALENT", "VALID_UNENUMERATED", "INVALID", "UNCERTAIN"}:
            raise ValueError("invalid O validity")
        if self.proposal_kind not in {"OPERATIONALIZATION", "FINDING"}:
            raise ValueError("invalid adjudication proposal_kind")
        if self.confidence is not None and not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not self.adjudication_id:
            object.__setattr__(self, "adjudication_id", f"adj:{self.proposal_id}:{self.adjudicator_id or 'unknown'}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supporting_source_ids"] = list(self.supporting_source_ids)
        return value


@dataclass(frozen=True)
class ScientificEvaluationContract:
    case_id: str
    enumerated_valid_O: tuple[Mapping[str, Any], ...]
    explicitly_invalid_O: tuple[Mapping[str, Any], ...]
    unenumerated_policy: str
    G_of_O: tuple[Mapping[str, Any], ...]
    finding_requirement_contract: Mapping[str, Any]
    adequate_core_sets: tuple[tuple[str, ...], ...]
    tolerances: Mapping[str, Any]
    adjudication_provenance: tuple[Mapping[str, Any], ...]
    status: str = "DRAFT_REQUIRES_HUMAN_CONFIRMATION"
    adjudicated_findings: tuple[Mapping[str, Any], ...] = ()
    source_semantic_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status == "CONFIRMED":
            raise ValueError("compiler cannot confirm a scientific evaluation contract")
        if self.source_semantic_contract_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.source_semantic_contract_sha256
        ):
            raise ValueError(
                "source_semantic_contract_sha256 must be a lowercase SHA-256 digest"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["enumerated_valid_O"] = [dict(x) for x in self.enumerated_valid_O]
        value["explicitly_invalid_O"] = [dict(x) for x in self.explicitly_invalid_O]
        value["G_of_O"] = [dict(x) for x in self.G_of_O]
        value["adequate_core_sets"] = [list(x) for x in self.adequate_core_sets]
        value["adjudication_provenance"] = [dict(x) for x in self.adjudication_provenance]
        value["adjudicated_findings"] = [dict(x) for x in self.adjudicated_findings]
        # Preserve the historical draft serialization for legacy callers.
        # Formal frozen validation rejects an absent binding separately.
        if self.source_semantic_contract_sha256 is None:
            value.pop("source_semantic_contract_sha256", None)
        return value


def _canonical_reference_value(value: Any) -> Any:
    """Normalize only representation noise for exact reference-space joins.

    This helper deliberately does not infer scientific equivalence.  It only
    makes JSON key order and harmless whitespace/case differences stable.
    """

    if isinstance(value, Mapping):
        return {str(key): _canonical_reference_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_reference_value(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip().casefold()
    return value


def _reference_branch_operationalization(branch: Mapping[str, Any]) -> dict[str, Any]:
    """Project one accepted-reference branch to its dimension/value mapping."""

    for key in ("effective_operationalization", "operationalization", "dimensions"):
        value = branch.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    decisions = branch.get("decisions")
    if isinstance(decisions, Sequence) and not isinstance(decisions, (str, bytes)):
        projected: dict[str, Any] = {}
        for decision in decisions:
            if not isinstance(decision, Mapping):
                continue
            dimension = decision.get("dimension", decision.get("dimension_id"))
            statement = decision.get("statement", decision.get("value", decision.get("normalized_meaning")))
            if dimension is not None and statement is not None:
                projected[str(dimension)] = statement
        return projected
    # A direct dimension mapping is also accepted, excluding branch metadata.
    metadata_keys = {"operationalization_id", "branch_id", "evidence_ids", "status"}
    return {str(key): value for key, value in branch.items() if str(key) not in metadata_keys}


def resolve_o_validity_label(
    scientific_validity: str,
    effective_o: Mapping[str, Any],
    accepted_reference_branches: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve scientific validity and reference membership as separate steps.

    The adjudicator supplies only scientific validity.  This post-adjudication
    resolver performs an exact canonical join against accepted reference
    branches and then derives the legacy final ``o_validity`` label.
    """

    raw = str(scientific_validity or "").strip().upper()
    compatibility = {
        "VALID_REFERENCE": "SCIENTIFICALLY_VALID",
        "VALID_EQUIVALENT": "SCIENTIFICALLY_VALID",
        "VALID_UNENUMERATED": "SCIENTIFICALLY_VALID",
        "VALID": "SCIENTIFICALLY_VALID",
        "INVALID": "SCIENTIFICALLY_INVALID",
        "SCIENTIFICALLY_INVALID": "SCIENTIFICALLY_INVALID",
        "UNCERTAIN": "SCIENTIFICALLY_UNCERTAIN",
        "SCIENTIFICALLY_UNCERTAIN": "SCIENTIFICALLY_UNCERTAIN",
        "SCIENTIFICALLY_VALID": "SCIENTIFICALLY_VALID",
    }
    scientific = compatibility.get(raw, "SCIENTIFICALLY_UNCERTAIN")
    match = "NOT_APPLICABLE"
    matched_id: str | None = None
    final = "UNCERTAIN"
    if scientific == "SCIENTIFICALLY_VALID":
        target = _canonical_reference_value(dict(effective_o))
        for branch in accepted_reference_branches:
            if not isinstance(branch, Mapping):
                continue
            candidate = _canonical_reference_value(_reference_branch_operationalization(branch))
            if candidate == target:
                match = "REFERENCE_MATCH"
                matched_id = str(branch.get("operationalization_id", branch.get("branch_id", ""))) or None
                break
        else:
            match = "NO_REFERENCE_MATCH"
        final = "VALID_REFERENCE" if match == "REFERENCE_MATCH" else "VALID_UNENUMERATED"
    elif scientific == "SCIENTIFICALLY_INVALID":
        final = "INVALID"
    return {
        "scientific_adjudication": scientific,
        "reference_space_match": match,
        "matched_reference_branch_id": matched_id,
        "final_o_validity": final,
    }


def materialize_branch(proposal: SRACProposal, deterministic_executor: Any) -> MaterializedBranch:
    """Execute a proposal with deterministic code; executor performs no adjudication."""

    if not callable(deterministic_executor):
        raise ValueError("deterministic_executor must be callable")
    result = deterministic_executor(proposal)
    if not isinstance(result, Mapping):
        raise ValueError("deterministic executor must return an object")
    return MaterializedBranch(
        proposal_id=proposal.proposal_id,
        parameters=dict(result.get("parameters", {})),
        execution=dict(result.get("execution", result.get("G_of_O", {}))),
        findings=tuple(item for item in result.get("findings", ()) if isinstance(item, Mapping)),
        materialization_id=str(result.get("materialization_id")) if result.get("materialization_id") else None,
        proposal_kind=proposal.proposal_kind if proposal.proposal_kind != "NONE" else "OPERATIONALIZATION",
        evidence_context_id=proposal.evidence_context_id,
    )


def compile_evaluation_contract(
    case_id: str,
    proposals: Sequence[SRACProposal],
    branches: Sequence[MaterializedBranch],
    adjudications: Sequence[ScientificAdjudication],
    finding_requirement_contract: Mapping[str, Any] | None = None,
    *,
    source_semantic_contract_sha256: str | None = None,
) -> ScientificEvaluationContract:
    """Compile deterministic ID-linked records; conflicts are escalated."""

    def unique_index(items: Sequence[Any], label: str, key: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in items:
            value = str(getattr(item, key, "") or "")
            if not value:
                raise ValueError(f"{label} missing {key}")
            if value in result:
                raise ValueError(f"duplicate {label} {key}: {value}")
            result[value] = item
        return result

    finding_contract = FindingRequirementContract.from_mapping(
        finding_requirement_contract
    )
    recognized_roles: set[str] = set()
    if finding_contract is not None:
        recognized_roles.update(finding_contract.mandatory_roles)
        recognized_roles.update(finding_contract.supporting_roles)
        recognized_roles.update(finding_contract.role_by_category.values())
        for group in finding_contract.alternative_role_groups:
            recognized_roles.update(group.roles)
        for core in finding_contract.adequate_core_sets:
            recognized_roles.update(core)

    proposal_map = unique_index(proposals, "proposal", "proposal_id")
    branch_map = unique_index(branches, "materialization", "materialization_id")
    adjudication_ids = unique_index(adjudications, "adjudication", "adjudication_id")
    del adjudication_ids
    by_proposal: dict[str, list[ScientificAdjudication]] = {}
    for adjudication in adjudications:
        proposal = proposal_map.get(adjudication.proposal_id)
        if proposal is None:
            raise ValueError(f"dangling adjudication proposal_id: {adjudication.proposal_id}")
        expected_kind = "OPERATIONALIZATION" if proposal.proposal_kind == "NONE" else proposal.proposal_kind
        if adjudication.proposal_kind != expected_kind:
            raise ValueError("WRONG_PROPOSAL_KIND")
        if proposal.producer_agent_id and adjudication.adjudicator_id and proposal.producer_agent_id == adjudication.adjudicator_id:
            raise ValueError("SELF_ADJUDICATION_FORBIDDEN")
        requires_materialization = proposal.proposal_kind == "OPERATIONALIZATION" and proposal.condition.upper().replace("_", "-") in {"O2-F1", "O3-F1"}
        if requires_materialization and not adjudication.materialization_id:
            raise ValueError("MISSING_MATERIALIZATION_ID")
        if adjudication.materialization_id and adjudication.materialization_id not in branch_map:
            raise ValueError(f"dangling materialization_id: {adjudication.materialization_id}")
        if adjudication.materialization_id:
            branch = branch_map[adjudication.materialization_id]
            if branch.proposal_id != adjudication.proposal_id:
                raise ValueError("WRONG_MATERIALIZATION_PROPOSAL_ID")
            # An operationalization validity verdict is evidence-gated.  A
            # failed materialization is an infrastructure/representability
            # outcome, never a scientific INVALID verdict and never a source
            # of G(O).  Keep a compatibility path for old synthetic records
            # that predate the explicit execution status field.
            if proposal.proposal_kind != "FINDING" and adjudication.o_validity in {
                "VALID_REFERENCE", "VALID_EQUIVALENT", "VALID_UNENUMERATED"
            }:
                execution_status = branch.execution.get("status") if isinstance(branch.execution, Mapping) else None
                if execution_status in {"MATERIALIZATION_UNSUPPORTED", "MATERIALIZATION_FAILED", "DATA_READER_ERROR", "TOOL_ERROR"}:
                    raise ValueError("VALID_O_WITHOUT_MATERIALIZED_EVIDENCE")
                if execution_status == "MATERIALIZED" and not isinstance(branch.execution.get("G_of_O"), Mapping):
                    raise ValueError("VALID_O_WITHOUT_MATERIALIZED_EVIDENCE")
        if proposal.evidence_context_id and not adjudication.evidence_context_id:
            raise ValueError("MISSING_EVIDENCE_CONTEXT_ID")
        if proposal.evidence_context_id and adjudication.evidence_context_id != proposal.evidence_context_id:
            raise ValueError("WRONG_EVIDENCE_CONTEXT_ID")
        if proposal.proposal_kind == "FINDING" and proposal.condition.upper().replace("_", "-") == "O1-F2" and not adjudication.evidence_context_id:
            raise ValueError("MISSING_EVIDENCE_CONTEXT_ID")
        if proposal.proposal_kind == "FINDING":
            finding_ids = [
                str(item.get("finding_id", "")).strip()
                for item in proposal.proposed_findings
                if isinstance(item, Mapping)
            ]
            if not finding_ids or any(not item for item in finding_ids):
                raise ValueError("F2_FINDING_ID_REQUIRED")
            if len(finding_ids) != len(set(finding_ids)):
                raise ValueError("F2_FINDING_ID_DUPLICATE")
            expected_ids = set(finding_ids)
            label_contracts = (
                ("finding_support", {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}),
                ("finding_relevance", {"RELEVANT", "IRRELEVANT", "UNCERTAIN"}),
                ("o_f_consistency", {"CONSISTENT", "INCONSISTENT", "UNCERTAIN"}),
            )
            for field_name, allowed_labels in label_contracts:
                labels = getattr(adjudication, field_name)
                if set(labels) != expected_ids:
                    raise ValueError(f"F2_ADJUDICATION_CANDIDATE_COVERAGE:{field_name}")
                if any(str(value).strip().upper() not in allowed_labels for value in labels.values()):
                    raise ValueError(f"F2_ADJUDICATION_LABEL_INVALID:{field_name}")
            role_ids = set(adjudication.finding_role)
            if not role_ids <= expected_ids:
                raise ValueError("F2_ADJUDICATION_CANDIDATE_COVERAGE:finding_role")
            if any(not str(value).strip() for value in adjudication.finding_role.values()):
                raise ValueError("F2_ADJUDICATION_LABEL_INVALID:finding_role")
            for finding_id in expected_ids:
                eligible = (
                    str(adjudication.finding_support[finding_id]).upper() == "SUPPORTED"
                    and str(adjudication.finding_relevance[finding_id]).upper() == "RELEVANT"
                    and str(adjudication.o_f_consistency[finding_id]).upper() == "CONSISTENT"
                )
                if eligible and finding_id not in role_ids:
                    raise ValueError("F2_ADJUDICATION_CANDIDATE_COVERAGE:finding_role_required_for_credit")
                if (
                    eligible
                    and recognized_roles
                    and adjudication.finding_role[finding_id]
                    not in recognized_roles
                ):
                    raise ValueError("F2_ADJUDICATION_LABEL_INVALID:finding_role")
        by_proposal.setdefault(adjudication.proposal_id, []).append(adjudication)

    conflicts: list[str] = []
    accepted: list[tuple[SRACProposal, ScientificAdjudication]] = []
    accepted_finding_proposals: list[
        tuple[SRACProposal, ScientificAdjudication, tuple[Mapping[str, Any], ...]]
    ] = []
    invalid: list[dict[str, Any]] = []
    unenumerated_requires_escalation = False
    for proposal_id in sorted(by_proposal):
        records = by_proposal[proposal_id]
        proposal = proposal_map[proposal_id]

        # O1-F2 keeps O fixed and opens F.  Finding admissibility therefore
        # comes from the finding-level labels, not from a fresh O verdict.
        # This branch intentionally precedes the generic O verdict handling.
        if proposal.proposal_kind == "FINDING":
            accepted_findings: list[Mapping[str, Any]] = []
            proposal_has_conflict = False
            for finding in proposal.proposed_findings:
                finding_id = str(finding["finding_id"])
                dimension_labels = {
                    field_name: {
                        str(getattr(record, field_name)[finding_id]).strip().upper()
                        for record in records
                    }
                    for field_name in (
                        "finding_support",
                        "finding_relevance",
                        "o_f_consistency",
                    )
                }
                role_labels = {
                    str(record.finding_role[finding_id]).strip()
                    for record in records
                    if finding_id in record.finding_role
                }
                if any(len(labels) != 1 for labels in dimension_labels.values()):
                    proposal_has_conflict = True
                    continue
                support = next(iter(dimension_labels["finding_support"]))
                relevance = next(iter(dimension_labels["finding_relevance"]))
                consistency = next(iter(dimension_labels["o_f_consistency"]))
                if "UNCERTAIN" in {support, relevance, consistency}:
                    proposal_has_conflict = True
                    continue
                if (support, relevance, consistency) == (
                    "SUPPORTED",
                    "RELEVANT",
                    "CONSISTENT",
                ):
                    if len(role_labels) != 1:
                        proposal_has_conflict = True
                        continue
                    compiled_finding = dict(finding)
                    compiled_finding["role"] = next(iter(role_labels))
                    accepted_findings.append(compiled_finding)
            if proposal_has_conflict:
                conflicts.append(proposal_id)
            if accepted_findings:
                accepted_finding_proposals.append(
                    (proposal, records[0], tuple(accepted_findings))
                )
            continue

        verdicts = {item.o_validity for item in records}
        if len(verdicts - {"VALID_REFERENCE", "VALID_EQUIVALENT"}) > 0 and len(verdicts) > 1:
            conflicts.append(proposal_id)
            continue
        if "UNCERTAIN" in verdicts or any(item.uncertain for item in records):
            conflicts.append(proposal_id)
            continue
        selected = records[0]
        if len(verdicts) > 1 and verdicts <= {"VALID_REFERENCE", "VALID_EQUIVALENT"}:
            # Frozen equivalence policy: both labels are valid and collapse to
            # one proposal record; the complete adjudication provenance remains.
            selected = next(item for item in records if item.o_validity == "VALID_REFERENCE")
        if selected.o_validity in {"VALID_REFERENCE", "VALID_EQUIVALENT", "VALID_UNENUMERATED"}:
            accepted.append((proposal_map[proposal_id], selected))
            if selected.o_validity == "VALID_UNENUMERATED":
                unenumerated_requires_escalation = True
        elif selected.o_validity == "INVALID":
            if not selected.scientific_rationale.strip():
                raise ValueError("INVALID_O_RATIONALE_REQUIRED")
            invalid.append(proposal_map[proposal_id].to_dict() | {"rationale": selected.scientific_rationale, "supporting_source_ids": list(selected.supporting_source_ids), "confidence": selected.confidence})
    status = "ESCALATE_TO_CURATOR" if conflicts or unenumerated_requires_escalation else "DRAFT_REQUIRES_HUMAN_CONFIRMATION"
    valid = tuple(proposal.to_dict() for proposal, _ in accepted if proposal.proposal_kind != "FINDING")
    adjudicated_findings_list: list[dict[str, Any]] = []
    for proposal, adjudication, accepted_findings in accepted_finding_proposals:
        compiled = proposal.to_dict()
        compiled["proposed_findings"] = [dict(item) for item in accepted_findings]
        compiled["evidence_context_id"] = (
            proposal.evidence_context_id or adjudication.evidence_context_id
        )
        adjudicated_findings_list.append(compiled)
    adjudicated_findings = tuple(adjudicated_findings_list)
    adjudicated_branches_list = []
    for proposal, adjudication in accepted:
        if proposal.proposal_kind == "FINDING":
            continue
        materialization_id = adjudication.materialization_id or f"mat:{proposal.proposal_id}"
        branch = next((item for item in branches if item.materialization_id == materialization_id or (item.proposal_id == proposal.proposal_id and not adjudication.materialization_id)), None)
        if branch is not None:
            adjudicated_branches_list.append(branch.to_dict())
    adjudicated_branches = tuple(adjudicated_branches_list)
    return ScientificEvaluationContract(
        case_id=case_id, enumerated_valid_O=valid, explicitly_invalid_O=tuple(invalid),
        unenumerated_policy="VALID_UNENUMERATED requires curator escalation",
        G_of_O=adjudicated_branches,
        finding_requirement_contract=dict(finding_requirement_contract or {}),
        adequate_core_sets=tuple(tuple(str(x) for x in item) for item in (finding_requirement_contract or {}).get("adequate_core_sets", []) if isinstance(item, Sequence) and not isinstance(item, (str, bytes))),
        tolerances={}, adjudication_provenance=tuple(asdict(item) for item in sorted(adjudications, key=lambda item: (item.proposal_id, item.adjudication_id or ""))), status=status,
        adjudicated_findings=adjudicated_findings,
        source_semantic_contract_sha256=source_semantic_contract_sha256,
    )


def build_srac_session(case_id: str, condition: str, scq: SCQResult) -> dict[str, Any]:
    scq_status = scq.status if hasattr(scq, "status") else str(scq.get("status", "")) if isinstance(scq, Mapping) else ""
    if scq_status != "SCQ_PASS":
        raise ValueError("SRAC requires SCQ_PASS")
    route = route_srac_condition(condition)
    return {"case_id": case_id, "route": route.to_dict(), "stage": "S1_PROPOSE", "status": "READY_FOR_PROPOSAL"}


__all__ = ["SRACProposal", "MaterializedBranch", "ScientificAdjudication", "ScientificEvaluationContract", "materialize_branch", "compile_evaluation_contract", "build_srac_session", "resolve_o_validity_label"]

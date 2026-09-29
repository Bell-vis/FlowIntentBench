"""Frozen condition-specific SRAC elicitation router."""

from __future__ import annotations

from typing import Sequence

from dataclasses import dataclass
from typing import Any, Mapping


ROUTING_TABLE = {
    "O1-F1": {"independent_O_elicitation": False, "independent_F_elicitation": False, "allowed_proposal": "none"},
    "O2-F1": {"independent_O_elicitation": True, "independent_F_elicitation": False, "allowed_proposal": "unresolved_O_only"},
    "O3-F1": {"independent_O_elicitation": True, "independent_F_elicitation": False, "allowed_proposal": "full_principal_O"},
    "O1-F2": {"independent_O_elicitation": False, "independent_F_elicitation": True, "allowed_proposal": "up_to_3_principal_findings"},
}


class SRACRoutingError(ValueError):
    pass


@dataclass(frozen=True)
class SRACRoute:
    condition: str
    independent_O_elicitation: bool
    independent_F_elicitation: bool
    allowed_proposal: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "independent_O_elicitation": self.independent_O_elicitation,
            "independent_F_elicitation": self.independent_F_elicitation,
            "allowed_proposal": self.allowed_proposal,
        }


def route_srac_condition(condition: str) -> SRACRoute:
    normalized = str(condition).upper().replace("_", "-")
    if normalized not in ROUTING_TABLE:
        raise SRACRoutingError(f"unsupported SRAC condition: {condition}")
    return SRACRoute(normalized, **ROUTING_TABLE[normalized])


def validate_route_payload(route: SRACRoute, payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if route.condition == "O1-F1" and (payload.get("proposed_operationalizations") or payload.get("proposed_findings")):
        errors.append("O1-F1 forbids independent elicitation")
    if not route.independent_O_elicitation and payload.get("proposed_operationalizations"):
        errors.append("route forbids proposed operationalizations")
    if not route.independent_F_elicitation and payload.get("proposed_findings"):
        errors.append("route forbids proposed findings")
    if route.condition == "O1-F2" and len(payload.get("proposed_findings", []) or []) > 3:
        errors.append("O1-F2 allows at most three principal findings")
    return errors


def validate_proposal_submission(condition: str, payload: Mapping[str, Any]) -> None:
    """Enforce the frozen SRAC route at proposal submission time."""

    if payload.get("case_id") is not None:
        validate_case_aware_proposal_submission(
            condition,
            str(payload.get("case_id")),
            payload.get("principal_dimensions", ()),
            payload.get("frozen_resolved_dimensions", ()),
            payload.get("frozen_unresolved_dimensions", ()),
            payload,
            require_complete=bool(payload.get("require_complete_unresolved", False)),
            frozen_scientific_target=payload.get("frozen_scientific_target"),
        )
        return

    route = route_srac_condition(condition)
    errors = validate_route_payload(route, payload)
    if errors:
        raise SRACRoutingError("SRAC_ROUTING_VIOLATION: " + "; ".join(errors))


def _proposal_dimensions(proposal: Mapping[str, Any]) -> set[str]:
    dimensions: set[str] = set()
    records = proposal.get("proposed_operationalizations", proposal.get("operationalizations", ()))
    if isinstance(records, Mapping):
        records = (records,)
    for record in records or ():
        if isinstance(record, Mapping):
            explicit = record.get("dimension_id", record.get("dimension", record.get("name", record.get("id"))))
            if explicit:
                dimensions.add(str(explicit))
            decisions = record.get("decisions")
            if isinstance(decisions, Mapping):
                dimensions.update(str(key) for key in decisions)
            elif isinstance(decisions, (list, tuple)):
                for item in decisions:
                    if isinstance(item, Mapping):
                        value = item.get("dimension_id", item.get("dimension", item.get("name")))
                        if value:
                            dimensions.add(str(value))
            # Compact proposals commonly encode ``{"criterion": ...}``.
            if not explicit and not decisions:
                dimensions.update(str(key) for key in record if str(key) not in {"statement", "value", "description"})
    return dimensions


def validate_case_aware_proposal_submission(
    condition: str,
    case_id: str,
    principal_dimensions: Sequence[str],
    frozen_resolved_dimensions: Sequence[str],
    frozen_unresolved_dimensions: Sequence[str],
    proposal: Mapping[str, Any] | Any,
    *,
    require_complete: bool = False,
    frozen_scientific_target: str | None = None,
) -> None:
    """Validate an SRAC proposal against one case's frozen dimensions."""

    if not str(case_id).strip():
        raise SRACRoutingError("SRAC_ROUTING_VIOLATION: case_id is required")
    payload = proposal.to_dict() if hasattr(proposal, "to_dict") else dict(proposal)
    base_payload = {key: item for key, item in payload.items() if key not in {"case_id", "principal_dimensions", "frozen_resolved_dimensions", "frozen_unresolved_dimensions", "require_complete_unresolved"}}
    # Validate the ordinary condition route first, without recursively
    # re-entering the case-aware dispatch above.
    route = route_srac_condition(condition)
    route_errors = validate_route_payload(route, base_payload)
    if route_errors:
        raise SRACRoutingError("SRAC_ROUTING_VIOLATION: " + "; ".join(route_errors))
    if frozen_scientific_target is not None:
        proposed_target = payload.get("scientific_target", payload.get("target"))
        if proposed_target is not None and str(proposed_target).strip().casefold() != str(frozen_scientific_target).strip().casefold():
            raise SRACRoutingError("SRAC_ROUTING_VIOLATION: scientific target drift")
    normalized = str(condition).upper().replace("_", "-")
    principal = {str(item) for item in principal_dimensions}
    resolved = {str(item) for item in frozen_resolved_dimensions}
    unresolved = {str(item) for item in frozen_unresolved_dimensions}
    if not resolved.issubset(principal) or not unresolved.issubset(principal) or resolved & unresolved:
        raise SRACRoutingError("SRAC_ROUTING_VIOLATION: inconsistent frozen dimension partition")
    proposed = _proposal_dimensions(payload)
    if normalized == "O2-F1":
        invalid = proposed - unresolved
        if invalid:
            raise SRACRoutingError("SRAC_ROUTING_VIOLATION: O2 proposal includes fixed or unknown dimensions: " + ", ".join(sorted(invalid)))
        if require_complete and proposed != unresolved:
            raise SRACRoutingError("SRAC_ROUTING_VIOLATION: incomplete O2 proposal; all unresolved dimensions are required")
    elif normalized == "O3-F1":
        invalid = proposed - principal
        if invalid:
            raise SRACRoutingError("SRAC_ROUTING_VIOLATION: O3 proposal includes dimensions outside principal space: " + ", ".join(sorted(invalid)))
        if require_complete and proposed != principal:
            raise SRACRoutingError("SRAC_ROUTING_VIOLATION: incomplete O3 proposal; all principal dimensions are required")
    elif normalized == "O1-F2":
        if payload.get("proposed_operationalizations"):
            raise SRACRoutingError("SRAC_ROUTING_VIOLATION: O proposal is forbidden for O1-F2")
        if not payload.get("fixed_o_branch_id") or not (payload.get("fixed_evidence_context_id") or payload.get("evidence_context_id")):
            raise SRACRoutingError("SRAC_ROUTING_VIOLATION: O1-F2 finding proposal requires fixed_o_branch_id and fixed_evidence_context_id")



__all__ = ["ROUTING_TABLE", "SRACRoute", "SRACRoutingError", "route_srac_condition", "validate_route_payload", "validate_proposal_submission", "validate_case_aware_proposal_submission"]

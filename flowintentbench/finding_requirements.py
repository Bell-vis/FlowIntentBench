"""Explicit, family-scoped Finding requirement contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class AlternativeRoleGroup:
    group_id: str
    min_required: int
    roles: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"group_id": self.group_id, "min_required": self.min_required, "roles": list(self.roles)}


@dataclass(frozen=True)
class FindingRequirementContract:
    mandatory_roles: tuple[str, ...] = ()
    alternative_role_groups: tuple[AlternativeRoleGroup, ...] = ()
    adequate_core_sets: tuple[tuple[str, ...], ...] = ()
    supporting_roles: tuple[str, ...] = ()
    novel_role_policy: str = "VALID_UNENUMERATED requires blinded adjudication"
    scientific_entity_type: str = "OTHER"
    contract_source: str = "CONSTRUCTION_DEFINED"
    role_by_category: Mapping[str, str] = field(default_factory=dict)
    # Category-level roles are a compatibility fallback. Formal F2 scoring
    # needs a case-specific role for each enumerated Reference Finding because
    # one broad category such as ``characterization`` can represent extent,
    # distribution spread, or another scientific role.
    reference_finding_role_map: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "FindingRequirementContract | None":
        if not isinstance(value, Mapping):
            return None
        groups = tuple(
            AlternativeRoleGroup(
                group_id=str(item.get("group_id", "")),
                min_required=int(item.get("min_required", 1)),
                roles=tuple(str(x) for x in item.get("roles", []) or []),
            )
            for item in value.get("alternative_role_groups", []) or []
            if isinstance(item, Mapping)
        )
        sets = tuple(
            tuple(str(role) for role in item)
            for item in value.get("adequate_core_sets", []) or []
            if isinstance(item, (list, tuple))
        )
        return cls(
            mandatory_roles=tuple(str(x) for x in value.get("mandatory_roles", []) or []),
            alternative_role_groups=groups,
            adequate_core_sets=sets,
            supporting_roles=tuple(str(x) for x in value.get("supporting_roles", []) or []),
            novel_role_policy=str(value.get("novel_role_policy", "VALID_UNENUMERATED requires blinded adjudication")),
            scientific_entity_type=str(value.get("scientific_entity_type", "OTHER")),
            contract_source=str(value.get("contract_source", "CONSTRUCTION_DEFINED")),
            role_by_category={str(k): str(v) for k, v in (value.get("role_by_category", {}) or {}).items()} if isinstance(value.get("role_by_category", {}), Mapping) else {},
            reference_finding_role_map={
                str(k): str(v)
                for k, v in (value.get("reference_finding_role_map", {}) or {}).items()
            }
            if isinstance(value.get("reference_finding_role_map", {}), Mapping)
            else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mandatory_roles": list(self.mandatory_roles),
            "alternative_role_groups": [item.to_dict() for item in self.alternative_role_groups],
            "adequate_core_sets": [list(item) for item in self.adequate_core_sets],
            "supporting_roles": list(self.supporting_roles),
            "novel_role_policy": self.novel_role_policy,
            "scientific_entity_type": self.scientific_entity_type,
            "contract_source": self.contract_source,
            "role_by_category": dict(self.role_by_category or {}),
            "reference_finding_role_map": dict(
                self.reference_finding_role_map or {}
            ),
        }

    @property
    def allowed_roles(self) -> frozenset[str]:
        """All authored roles, shared by validation, adjudication and scoring."""
        return frozenset(self.mandatory_roles) | frozenset(self.supporting_roles) | frozenset(
            role for core in self.adequate_core_sets for role in core
        ) | frozenset(
            role for group in self.alternative_role_groups for role in group.roles
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.adequate_core_sets:
            errors.append("adequate_core_sets is empty")
        mandatory = set(self.mandatory_roles)
        for index, core in enumerate(self.adequate_core_sets):
            if not core or any(not role.strip() for role in core):
                errors.append(f"adequate_core_sets[{index}] must contain non-empty roles")
            if len(core) != len(set(core)):
                errors.append(f"adequate_core_sets[{index}] has duplicate roles")
            if not mandatory <= set(core):
                errors.append(f"adequate_core_sets[{index}] omits mandatory roles")
        for group in self.alternative_role_groups:
            if group.min_required < 1 or group.min_required > len(group.roles):
                errors.append(f"invalid min_required for alternative group {group.group_id!r}")
        allowed_roles = self.allowed_roles
        for finding_id, role in self.reference_finding_role_map.items():
            if not str(finding_id).strip() or not str(role).strip():
                errors.append("reference_finding_role_map requires non-empty IDs and roles")
            elif role not in allowed_roles:
                errors.append(
                    f"reference Finding {finding_id!r} uses ineligible role {role!r}"
                )
        # Scientific obligations belong to the authored family, not to broad
        # entity labels. A fixed FIELD can require a statistic without selection,
        # and a fixed region can require flux without rediscovering its location.
        return errors


def evaluate_adequate_core_sets(
    matched_roles: Iterable[str],
    contract: FindingRequirementContract | Mapping[str, Any],
) -> dict[str, Any]:
    normalized = contract if isinstance(contract, FindingRequirementContract) else FindingRequirementContract.from_mapping(contract)
    if normalized is None or not normalized.adequate_core_sets:
        return {"status": "NOT_EVALUABLE", "best_recall": None, "matched_roles": sorted(set(matched_roles))}
    errors = normalized.validate()
    if errors:
        return {"status": "NOT_EVALUABLE", "best_recall": None,
                "matched_roles": sorted(set(matched_roles)), "errors": errors}
    matched = set(str(x) for x in matched_roles)
    sets = []
    for index, core in enumerate(normalized.adequate_core_sets, start=1):
        required = set(normalized.mandatory_roles) | set(core)
        sets.append({
            "set_id": f"S{index}",
            "core_set": list(core),
            "required_roles": sorted(required),
            "matched_role_count": len(required & matched),
            "required_role_count": len(required),
            "recall": len(required & matched) / len(required) if required else 0.0,
        })
    best_recall = max(item["recall"] for item in sets)
    best = next(item for item in sets if item["recall"] == best_recall)
    mandatory_ok = set(normalized.mandatory_roles) <= matched
    return {
        "status": "PASS" if mandatory_ok and best["recall"] >= 1.0 else "INCOMPLETE",
        "mandatory_roles_satisfied": mandatory_ok,
        "best_recall": best["recall"],
        "best_set_ids": [item["set_id"] for item in sets if item["recall"] == best["recall"]],
        "matched_roles": sorted(matched),
        "sets": sets,
        "contract_entity_type": normalized.scientific_entity_type,
        "contract_source": normalized.contract_source,
    }


def audit_f2_requirements(family: Mapping[str, Any], *, cases: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Check that F2 is described as an adequate-set task, not a hidden list."""

    schema = family.get("finding_schema", {}) if isinstance(family, Mapping) else {}
    goal = " ".join(str(schema.get(key) or "") for key in ("f1_goal", "f2_goal"))
    contract_value = family.get("finding_requirement_contract") or family.get("f2_requirement_contract")
    contract = FindingRequirementContract.from_mapping(contract_value)
    has_sets = bool(contract and contract.adequate_core_sets)
    differs = bool(schema.get("f2_goal")) and schema.get("f2_goal") != schema.get("f1_goal")
    blockers = []
    if contract is None:
        blockers.append("missing finding requirement contract")
    else:
        blockers.extend(contract.validate())
    if not differs:
        blockers.append("F2_CONSTRUCT_COLLAPSE")
    return {
        "family_id": family.get("family_id"),
        "f2_goal_present": bool(schema.get("f2_goal")),
        "adequate_sets_explicit": has_sets,
        "f1_f2_noncollapse": differs,
        "f2_differs_from_f1": differs,
        "adequate_core_sets": [list(item) for item in contract.adequate_core_sets] if contract else [],
        "finding_requirement_contract": contract.to_dict() if contract else None,
        "novel_finding_policy": contract.novel_role_policy if contract else None,
        "blockers": blockers,
        "status": "PASS" if not blockers and bool(schema.get("f2_goal")) else ("NOT_EVALUABLE" if contract is None else "REVIEW"),
        "note": goal,
        "scientific_entity_type": contract.scientific_entity_type if contract else None,
        "contract_source": contract.contract_source if contract else None,
    }


__all__ = ["AlternativeRoleGroup", "FindingRequirementContract", "audit_f2_requirements", "evaluate_adequate_core_sets"]

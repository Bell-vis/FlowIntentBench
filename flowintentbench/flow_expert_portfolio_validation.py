"""Cross-family validation harness for the canonical tool-free Flow Expert.

This module assembles existing family-scoped construction artifacts into the
canonical scientific-review packet.  It is a validation harness only: it does
not change evidence support, curator state, SCQ eligibility, or benchmark
membership.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .case_repository import condition_from_case_id, load_case_record
from .grounding import (
    FamilyScientificEvidenceResolution,
    resolve_family_scientific_evidence,
)
from .scientific_expert_review import (
    FLOW_SCIENTIFIC_REVIEWER_PROFILE,
    build_scientific_review_packet,
    run_scientific_expert_review,
    validate_scientific_expert_output,
)
from .scientific_expert_validation import write_immutable_scientific_expert_run
from .scientific_run_snapshots import canonical_json_sha256


@dataclass(frozen=True)
class FlowExpertPortfolioTarget:
    dataset_id: str
    family_id: str
    concept_id: str
    case_id: str
    resource_family_id: str
    evidence_kind: str


FLOW_EXPERT_PORTFOLIO_TARGETS = (
    FlowExpertPortfolioTarget(
        "Kitchen",
        "kitchen_flow_regions",
        "high_speed_region",
        "kitchen_o2_f1",
        "kitchen__high_speed_region",
        "authoritative_kitchen_review",
    ),
    FlowExpertPortfolioTarget(
        "Kitchen",
        "kitchen_turbulence_activity",
        "kitchen_turbulence_activity",
        "kitchen_turbulence_activity_o2_f1",
        "kitchen_turbulence_activity",
        "concept_card",
    ),
    FlowExpertPortfolioTarget(
        "Kitchen",
        "kitchen_concentration_heterogeneity",
        "kitchen_concentration_heterogeneity",
        "kitchen_concentration_heterogeneity_o2_f1",
        "kitchen_concentration_heterogeneity",
        "concept_card",
    ),
    FlowExpertPortfolioTarget(
        "Combustor",
        "combustor_density_features",
        "combustor_density_features",
        "combustor_density_features_o2_f1",
        "combustor_density_features",
        "concept_card",
    ),
)


# Four family-scoped packets cover the five responsibility semantics without
# exhaustively reviewing all 16 authored cases.  Keeping one packet per family
# also preserves the existing family-level evidence and firewall boundary.
FLOW_EXPERT_RESPONSIBILITY_CASE_MATRIX = {
    "kitchen_flow_regions": (
        "kitchen_o2_f1",
        "kitchen_o1_f1",
        "kitchen_o1_f2",
    ),
    "kitchen_turbulence_activity": (
        "kitchen_turbulence_activity_o2_f1",
        "kitchen_turbulence_activity_o3_f1",
    ),
    "kitchen_concentration_heterogeneity": (
        "kitchen_concentration_heterogeneity_o3_f1",
        "kitchen_concentration_heterogeneity_o1_f2",
    ),
    "combustor_density_features": (
        "combustor_density_features_o1_f1",
    ),
}


# Full controlled families used by the semantic/presentation optimization
# audit.  These are reviewed as one family-scoped packet so the reviewer can
# compare the scientific object induced by O1/O2/O3 instead of judging each
# condition in isolation.
FLOW_EXPERT_SCIENTIFIC_TARGET_CASE_MATRIX = {
    "kitchen_concentration_heterogeneity": (
        "kitchen_concentration_heterogeneity_o1_f1",
        "kitchen_concentration_heterogeneity_o2_f1",
        "kitchen_concentration_heterogeneity_o3_f1",
        "kitchen_concentration_heterogeneity_o1_f2",
    ),
    "kitchen_turbulence_activity": (
        "kitchen_turbulence_activity_o1_f1",
        "kitchen_turbulence_activity_o2_f1",
        "kitchen_turbulence_activity_o3_f1",
        "kitchen_turbulence_activity_o1_f2",
    ),
    "combustor_density_features": (
        "combustor_density_features_o1_f1",
        "combustor_density_features_o2_f1",
        "combustor_density_features_o3_f1",
        "combustor_density_features_o1_f2",
    ),
}

_FORBIDDEN_LIFECYCLE_KEYS = frozenset(
    {
        "curator_confirmation",
        "curator_decision",
        "formal_benchmark_membership",
        "formal_release_eligible",
        "official_scq_eligibility",
        "official_scq_executed",
        "scq_result",
        "srac_adjudication",
    }
)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _all_dataset_evidence(root: Path, dataset_id: str) -> list[dict[str, Any]]:
    construction = root / "datasets" / dataset_id / "construction"
    rows: list[dict[str, Any]] = []
    for filename in (
        "context_evidence.json",
        "operationalization_evidence.json",
        "finding_evidence.json",
    ):
        payload = _read_json(construction / filename, [])
        if isinstance(payload, list):
            rows.extend(dict(item) for item in payload if isinstance(item, Mapping))
    return rows


def _concept_card(root: Path, concept_id: str) -> dict[str, Any]:
    path = (
        root
        / "artifacts/reference/concept_expansion_phase1/concept_design_cards"
        / f"{concept_id}.json"
    )
    value = _read_json(path, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid concept card: {path}")
    return dict(value)


def _evidence_resolution(
    root: Path, target: FlowExpertPortfolioTarget
) -> FamilyScientificEvidenceResolution:
    if target.evidence_kind == "authoritative_kitchen_review":
        path = root / "artifacts/reference/kitchen_scientific_review/evidence_binding.json"
        value = _read_json(path, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"invalid authoritative evidence binding: {path}")
        return FamilyScientificEvidenceResolution.from_mapping(value)

    card = _concept_card(root, target.concept_id)
    wanted_evidence = {str(item) for item in card.get("evidence_ids", ()) or ()}
    evidence_records = [
        item
        for item in _all_dataset_evidence(root, target.dataset_id)
        if str(item.get("evidence_id", "")) in wanted_evidence
    ]
    evidence_source_ids = {
        str(item.get("source_id", ""))
        for item in evidence_records
        if str(item.get("source_id", "")).strip()
    }
    construction = root / "datasets" / target.dataset_id / "construction"
    source_payload = _read_json(construction / "sources.json", {})
    all_sources = (
        source_payload.get("sources", ()) if isinstance(source_payload, Mapping) else ()
    )
    source_records = [
        dict(item)
        for item in all_sources
        if isinstance(item, Mapping)
        and str(item.get("source_id", "")) in evidence_source_ids
    ]
    return resolve_family_scientific_evidence(
        target.dataset_id,
        target.family_id,
        target.concept_id,
        source_records,
        evidence_records,
        card.get("scientific_support_claims", ()),
        card.get("claim_support_links", ()),
        evidence_family_id=target.family_id,
        evidence_concept_id=target.concept_id,
    )


def _family_resources(
    root: Path,
    target: FlowExpertPortfolioTarget,
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    directory = (
        root
        / "artifacts/reference/scientific_portfolio/families"
        / target.resource_family_id
    )
    materialization_execution: dict[str, Any] = {}
    if target.evidence_kind == "authoritative_kitchen_review":
        selected_case_id = case_id or target.case_id
        condition = condition_from_case_id(selected_case_id)
        if condition is None:
            raise ValueError(
                f"cannot resolve materialization condition from case: {selected_case_id}"
            )
        value = _read_json(
            root
            / "artifacts/reference/kitchen_scientific_review"
            / f"materialized_g_of_o__{condition}.json",
            {},
        )
        if not isinstance(value, Mapping) or not value:
            raise ValueError(
                f"missing authoritative materialization for {selected_case_id}"
            )
        materialization_execution = dict(value)
    return {
        "definition": _read_json(directory / "family_definition.json", {}),
        "finding_requirements": _read_json(
            directory / "finding_requirements.json", {}
        ),
        "operationalizations": _read_json(
            directory / "reference_operationalizations.json", {}
        ),
        "materialization_execution": materialization_execution,
    }


def _observable_semantics(
    case_input: Mapping[str, Any], concept_id: str
) -> dict[str, Any]:
    flow_data = case_input.get("flow_data")
    flow_data = flow_data if isinstance(flow_data, Mapping) else {}
    metadata = flow_data.get("data_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    variables = metadata.get("variables", ())
    variables = (
        [item for item in variables if isinstance(item, Mapping)]
        if isinstance(variables, Sequence) and not isinstance(variables, (str, bytes))
        else []
    )

    def relevant(variable: Mapping[str, Any]) -> bool:
        name = str(variable.get("name", "")).casefold()
        quantity = str(variable.get("physical_quantity", "")).casefold()
        if concept_id == "high_speed_region":
            return name == "velocity" or quantity == "velocity"
        if concept_id == "kitchen_turbulence_activity":
            return name in {"ke", "ep"}
        if concept_id == "kitchen_concentration_heterogeneity":
            return name.startswith("c") and "concentration" in quantity
        if concept_id == "combustor_density_features":
            return name == "density" or quantity == "density"
        return False

    return {
        "format": dict(metadata.get("format", {}))
        if isinstance(metadata.get("format"), Mapping)
        else {},
        "grid": dict(metadata.get("grid", {}))
        if isinstance(metadata.get("grid"), Mapping)
        else {},
        "variables": [dict(item) for item in variables if relevant(item)],
        "coordinate_system": dict(metadata.get("coordinate_system", {}))
        if isinstance(metadata.get("coordinate_system"), Mapping)
        else {},
        "temporal": dict(metadata.get("temporal", {}))
        if isinstance(metadata.get("temporal"), Mapping)
        else {},
    }


def _condition_view(
    record: Mapping[str, Any], resources: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    case_input = record.get("case_input")
    case_input = case_input if isinstance(case_input, Mapping) else {}
    principal = [
        str(item)
        for item in metadata.get("principal_operationalization_dimensions", ()) or ()
    ]
    unresolved = [
        str(item)
        for item in metadata.get("unresolved_operationalization_dimensions", ()) or ()
    ]
    fixed = [item for item in principal if item not in set(unresolved)]
    definition = resources.get("definition")
    definition = definition if isinstance(definition, Mapping) else {}
    clauses = definition.get("resolved_operationalization_clauses")
    clauses = clauses if isinstance(clauses, Mapping) else {}
    # The construction resource contains branch IDs, validity labels, findings,
    # and result values.  The reviewer needs only complete authored O clauses.
    # Project and deduplicate those descriptions without copying any branch or
    # Ground Truth identity into the model-visible packet.
    operationalizations = resources.get("operationalizations")
    operationalizations = (
        operationalizations if isinstance(operationalizations, Mapping) else {}
    )
    raw_options = operationalizations.get("accepted", ()) or ()
    authored: list[dict[str, Any]] = []
    seen_options: set[str] = set()
    current_case_id = str(record.get("case_id", ""))
    for item in raw_options if isinstance(raw_options, Sequence) else ():
        if not isinstance(item, Mapping):
            continue
        if str(item.get("source_case_id", "")) != current_case_id:
            continue
        option_clauses = item.get("resolved_operationalization_clauses")
        if not isinstance(option_clauses, Mapping):
            continue
        decisions = []
        for dimension in principal:
            clause = option_clauses.get(dimension)
            if not isinstance(clause, Mapping):
                continue
            decisions.append(
                {
                    "dimension_id": dimension,
                    "canonical_id": str(clause.get("canonical_id", "")),
                    "description": str(clause.get("description", "")),
                }
            )
        if {row["dimension_id"] for row in decisions} != set(principal):
            continue
        signature = json.dumps(decisions, sort_keys=True, separators=(",", ":"))
        if signature in seen_options:
            continue
        seen_options.add(signature)
        authored.append(
            {
                "case_id": current_case_id,
                "candidate_index": len(authored) + 1,
                "decision_status": "AUTHORED_EXECUTABLE_CANDIDATE",
                "decisions": decisions,
                "scientific_result_values_visible": False,
            }
        )
    if not authored:
        fixed_decisions = []
        for dimension in fixed:
            clause = clauses.get(dimension)
            clause = clause if isinstance(clause, Mapping) else {}
            fixed_decisions.append(
                {
                    "dimension_id": dimension,
                    "canonical_id": str(clause.get("canonical_id", "")),
                    "description": str(clause.get("description", "")),
                }
            )
        authored.append(
            {
                "case_id": current_case_id,
                "candidate_index": 1,
                "decision_status": "PARTIAL_AUTHORED_O_ONLY",
                "decisions": fixed_decisions,
                "scientific_result_values_visible": False,
            }
        )

    explicit_findings = metadata.get("explicit_finding_requirements", ()) or ()
    finding_rows = [
        dict(item) for item in explicit_findings if isinstance(item, Mapping)
    ]
    family_finding = resources.get("finding_requirements")
    if (
        str(metadata.get("finding_openness", "")).casefold() == "open"
        and isinstance(family_finding, Mapping)
    ):
        finding_rows.append(
            {
                "requirement_type": "FAMILY_ROLE_CONTRACT",
                "mandatory_roles": list(family_finding.get("mandatory_roles", ()) or ()),
                "alternative_role_groups": list(
                    family_finding.get("alternative_role_groups", ()) or ()
                ),
                "scientific_entity_type": family_finding.get(
                    "scientific_entity_type"
                ),
            }
        )

    representability = metadata.get("evaluation_representability_contract")
    representability = (
        representability if isinstance(representability, Mapping) else {}
    )
    materialization = representability.get("materialization")
    materialization = (
        materialization if isinstance(materialization, Mapping) else {}
    )
    execution = resources.get("materialization_execution")
    execution = execution if isinstance(execution, Mapping) else {}
    execution_branches = execution.get("branches")
    execution_branches = (
        [item for item in execution_branches if isinstance(item, Mapping)]
        if isinstance(execution_branches, Sequence)
        and not isinstance(execution_branches, (str, bytes))
        else []
    )
    authored_signatures = sorted(
        tuple(
            sorted(
                (
                    str(decision.get("dimension_id", "")),
                    str(decision.get("description", "")),
                )
                for decision in candidate.get("decisions", ()) or ()
                if isinstance(decision, Mapping)
            )
        )
        for candidate in authored
    )
    execution_signatures = sorted(
        tuple(
            sorted(
                (str(dimension), str(description))
                for dimension, description in effective_o.items()
            )
        )
        for item in execution_branches
        for effective_o in [item.get("effective_operationalization")]
        if isinstance(effective_o, Mapping)
    )

    def has_result(value: Any) -> bool:
        if isinstance(value, Mapping):
            return bool(value)
        return isinstance(value, Sequence) and not isinstance(
            value, (str, bytes)
        ) and bool(value)

    condition_id = str(record.get("condition", ""))
    effective_o_condition = "O1-F1" if condition_id == "O1-F2" else condition_id
    execution_recorded = (
        execution.get("status") == "MATERIALIZED"
        and str(execution.get("case_id", "")) == current_case_id
        and str(execution.get("effective_o_condition", ""))
        == effective_o_condition
        and bool(str(execution.get("handler_id", "")).strip())
        and (
            not str(materialization.get("handler_id", "")).strip()
            or str(execution.get("handler_id", ""))
            == str(materialization.get("handler_id", ""))
        )
        and bool(execution_branches)
        and all(
            str(item.get("status", "")) == "MATERIALIZED"
            and str(item.get("effective_o_condition", ""))
            == effective_o_condition
            and isinstance(item.get("execution"), Mapping)
            and str(item["execution"].get("status", "")) == "MATERIALIZED"
            and str(item["execution"].get("dataset_id", ""))
            == str(record.get("dataset_id", ""))
            and has_result(item.get("G_of_O"))
            and has_result(item["execution"].get("G_of_O"))
            for item in execution_branches
        )
        and authored_signatures == execution_signatures
    )
    if execution and not execution_recorded:
        raise ValueError(
            f"materialization execution integrity mismatch for {current_case_id}"
        )
    if execution_recorded and len(execution_branches) != len(authored):
        raise ValueError(
            f"materialization/authored-O count mismatch for {current_case_id}: "
            f"{len(execution_branches)} != {len(authored)}"
        )
    route_declared = bool(materialization) and materialization.get("available") is True
    handler_id = (
        str(execution.get("handler_id", ""))
        if execution_recorded
        else str(materialization.get("handler_id", ""))
        if route_declared
        else ""
    )
    return {
        "condition": str(record.get("condition", "")),
        "case_id": str(record.get("case_id", "")),
        "question": str(case_input.get("scientific_question", "")),
        "scientific_target": str(metadata.get("scientific_target", "")),
        "finding_goal": str(metadata.get("finding_goal", "")),
        "operationalization_responsibility": str(
            metadata.get("operationalization_responsibility", "")
        ),
        "finding_openness": str(metadata.get("finding_openness", "")),
        "finding_responsibility": (
            "F2"
            if str(metadata.get("finding_openness", "")).casefold() == "open"
            else "F1"
        ),
        "fixed_o_dimensions": fixed,
        "unresolved_o_dimensions": unresolved,
        "finding_requirements": finding_rows,
        "authored_operationalizations": authored,
        "materialization_route": {
            "execution_status": (
                "MATERIALIZED_RESULTS_WITHHELD"
                if execution_recorded
                else "DETERMINISTIC_ROUTE_DECLARED_AVAILABLE"
                if route_declared
                else "DETERMINISTIC_ROUTE_NOT_DECLARED"
            ),
            "handler_id": handler_id,
            "effective_o_condition": effective_o_condition,
            "branch_count": len(authored),
            "branch_statuses": [
                (
                    "MATERIALIZED_RESULT_WITHHELD"
                    if execution_recorded
                    else "AUTHORED_O_ROUTE_DECLARED"
                    if route_declared
                    else "AUTHORED_O_ROUTE_NOT_DECLARED"
                )
                for _ in authored
            ],
            "scientific_result_values_visible": False,
        },
    }


def _forbidden_lifecycle_paths(value: Any, path: str = "result") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key).strip().casefold() in _FORBIDDEN_LIFECYCLE_KEYS:
                hits.append(child)
            hits.extend(_forbidden_lifecycle_paths(item, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            hits.extend(_forbidden_lifecycle_paths(item, f"{path}[{index}]"))
    return hits


def build_flow_expert_portfolio_packets(
    repository_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Build one canonical O2-F1 packet for each frozen validation family."""

    root = Path(repository_root).resolve()
    packets: list[dict[str, Any]] = []
    for target in FLOW_EXPERT_PORTFOLIO_TARGETS:
        record = load_case_record(root, target.case_id)
        resources = _family_resources(root, target)
        resolution = _evidence_resolution(root, target)
        condition = _condition_view(record, resources)
        case_context = record.get("case_context")
        case_context = case_context if isinstance(case_context, Mapping) else {}
        case_input = record.get("case_input")
        case_input = case_input if isinstance(case_input, Mapping) else {}
        packet = build_scientific_review_packet(
            dataset_id=target.dataset_id,
            family_id=target.family_id,
            concept_id=target.concept_id,
            scientific_target=str(record.get("metadata", {}).get("scientific_target", "")),
            dataset_context=case_context,
            observable_semantics=_observable_semantics(case_input, target.concept_id),
            conditions=[condition],
            evidence_resolution=resolution,
        )
        packets.append(packet)
    return tuple(packets)


def build_flow_expert_responsibility_matrix_packets(
    repository_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Build the compact four-family, eight-condition responsibility matrix."""

    root = Path(repository_root).resolve()
    packets: list[dict[str, Any]] = []
    for target in FLOW_EXPERT_PORTFOLIO_TARGETS:
        case_ids = FLOW_EXPERT_RESPONSIBILITY_CASE_MATRIX.get(target.family_id)
        if not case_ids:
            raise ValueError(
                f"responsibility matrix has no cases for family: {target.family_id}"
            )
        records = [load_case_record(root, case_id) for case_id in case_ids]
        for record in records:
            if (
                str(record.get("dataset_id", "")) != target.dataset_id
                or str(record.get("family_id", "")) != target.family_id
            ):
                raise ValueError(
                    "responsibility matrix case identity mismatch: "
                    f"{record.get('case_id')}"
                )
        conditions = [
            _condition_view(
                record,
                _family_resources(
                    root,
                    target,
                    case_id=str(record.get("case_id", "")),
                ),
            )
            for record in records
        ]
        first_record = records[0]
        case_context = first_record.get("case_context")
        case_context = case_context if isinstance(case_context, Mapping) else {}
        case_input = first_record.get("case_input")
        case_input = case_input if isinstance(case_input, Mapping) else {}
        packet = build_scientific_review_packet(
            dataset_id=target.dataset_id,
            family_id=target.family_id,
            concept_id=target.concept_id,
            scientific_target=str(
                first_record.get("metadata", {}).get("scientific_target", "")
            ),
            dataset_context=case_context,
            observable_semantics=_observable_semantics(case_input, target.concept_id),
            conditions=conditions,
            evidence_resolution=_evidence_resolution(root, target),
        )
        packets.append(packet)
    return tuple(packets)


def build_flow_expert_scientific_target_packets(
    repository_root: str | Path,
    *,
    question_overrides: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build the three complete family packets used for target-stability audit.

    ``question_overrides`` changes only the presentation supplied to the
    reviewer.  Scientific metadata, authored Operationalizations, evidence,
    and materialization routes continue to come from the canonical case
    records.  The helper therefore cannot promote a realization or mutate a
    scientific case.
    """

    root = Path(repository_root).resolve()
    overrides = dict(question_overrides or {})
    packets: list[dict[str, Any]] = []
    targets = {
        target.family_id: target
        for target in FLOW_EXPERT_PORTFOLIO_TARGETS
        if target.family_id in FLOW_EXPERT_SCIENTIFIC_TARGET_CASE_MATRIX
    }
    for family_id, case_ids in FLOW_EXPERT_SCIENTIFIC_TARGET_CASE_MATRIX.items():
        target = targets.get(family_id)
        if target is None:
            raise ValueError(f"scientific target audit has no portfolio target: {family_id}")
        records = [load_case_record(root, case_id) for case_id in case_ids]
        conditions: list[dict[str, Any]] = []
        for record in records:
            if (
                str(record.get("dataset_id", "")) != target.dataset_id
                or str(record.get("family_id", "")) != family_id
            ):
                raise ValueError(
                    "scientific target audit case identity mismatch: "
                    f"{record.get('case_id')}"
                )
            condition = _condition_view(
                record,
                _family_resources(
                    root,
                    target,
                    case_id=str(record.get("case_id", "")),
                ),
            )
            case_id = str(record.get("case_id", ""))
            if case_id in overrides:
                question = str(overrides[case_id]).strip()
                if not question:
                    raise ValueError(f"empty question override for {case_id}")
                condition["question"] = question
            conditions.append(condition)

        first_record = records[0]
        case_context = first_record.get("case_context")
        case_context = case_context if isinstance(case_context, Mapping) else {}
        case_input = first_record.get("case_input")
        case_input = case_input if isinstance(case_input, Mapping) else {}
        packets.append(
            build_scientific_review_packet(
                dataset_id=target.dataset_id,
                family_id=family_id,
                concept_id=target.concept_id,
                scientific_target=str(
                    first_record.get("metadata", {}).get("scientific_target", "")
                ),
                dataset_context=case_context,
                observable_semantics=_observable_semantics(
                    case_input, target.concept_id
                ),
                conditions=conditions,
                evidence_resolution=_evidence_resolution(root, target),
            )
        )
    return tuple(packets)


def _markdown_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Flow Expert cross-family validation",
        "",
        "This is a scientific-review validation exercise. It performs no curator, SCQ, SRAC, or release transition.",
        "",
        "| Family / concept | Invocation | Family binding | Provenance | Claim evidence | Review validation |",
        "|---|---|---|---|---|---|",
    ]
    for row in result.get("families", ()):
        lines.append(
            "| `{family_id}` / `{concept_id}` | `{invocation_status}` | "
            "`{family_binding_status}` | `{evidence_provenance_closure}` | "
            "`{claim_evidence_sufficiency}` | `{review_validation_status}` |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "`review_permitted` means only that family identity and every declared citation close through visible evidence and sources. It does not mean that a scientific claim is supported.",
            "",
            f"- Cross-family status: `{result.get('cross_family_validation_status')}`",
            f"- Lifecycle promotion performed: `{str(result.get('lifecycle_promotion_performed')).lower()}`",
            "- Human calibration status: `PENDING`",
            "- Scientifically validated Flow Expert: `false`",
        ]
    )
    return "\n".join(lines) + "\n"


ReviewCallable = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def run_flow_expert_portfolio_validation(
    repository_root: str | Path,
    output_root: str | Path,
    *,
    run_live: bool = False,
    reviewer: ReviewCallable | None = None,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Build, optionally run, validate, and report the four-family exercise."""

    if run_live and reviewer is not None:
        raise ValueError("run_live and reviewer are mutually exclusive")
    root = Path(repository_root).resolve()
    out = Path(output_root).resolve()
    packets = build_flow_expert_portfolio_packets(root)
    rows: list[dict[str, Any]] = []
    packet_hashes: dict[str, str] = {}
    review_hashes: dict[str, str] = {}
    for packet in packets:
        identity = packet["family_identity"]
        family_id = str(identity["family_id"])
        concept_id = str(identity["concept_id"])
        packet_hash = canonical_json_sha256(packet)
        packet_hashes[family_id] = packet_hash
        _write_json(out / "packets" / f"{family_id}.json", packet)

        artifact: dict[str, Any] | None = None
        if reviewer is not None:
            raw = dict(reviewer(packet))
            validated = validate_scientific_expert_output(raw, packet)
            artifact = {
                "status": "SUCCESS" if validated["status"] == "PASS" else "INCOMPLETE",
                "invocation_status": "SUCCESS",
                "execution_mode": "INJECTED_VALIDATION_REVIEWER",
                "live_model_calls": False,
                "proxy_only": True,
                "parsed_result": raw,
                "validated_review": validated,
            }
        elif run_live:
            artifact = run_scientific_expert_review(
                root,
                packet,
                config_path=config_path,
                api_key=api_key,
                timeout=timeout,
                run_id=f"cross-family::{family_id}",
            )

        provenance = packet["provenance_summary"]
        validated_review = (
            artifact.get("validated_review", {}) if isinstance(artifact, Mapping) else {}
        )
        condition_reviews = (
            validated_review.get("condition_reviews", ())
            if isinstance(validated_review, Mapping)
            else ()
        )
        lifecycle_hits = _forbidden_lifecycle_paths(artifact or {})
        immutable_run: dict[str, Any] | None = None
        if run_live and isinstance(artifact, Mapping):
            call = artifact.get("call")
            call = call if isinstance(call, Mapping) else {}
            parsed_review = artifact.get("parsed_result")
            parsed_review = (
                parsed_review if isinstance(parsed_review, Mapping) else None
            )
            immutable_run = write_immutable_scientific_expert_run(
                out / "families" / family_id,
                review_packet=packet,
                review=parsed_review,
                invocation_status=str(
                    artifact.get("invocation_status", "PROVIDER_ERROR")
                ),
                model_identity={
                    "provider": str(call.get("provider", "UNKNOWN")),
                    "model_id": str(call.get("model_id", "UNKNOWN")),
                    "model_family": str(call.get("model_family", "UNKNOWN")),
                },
                profile_identity={
                    "agent_profile_id": str(
                        call.get("agent_profile_id", FLOW_SCIENTIFIC_REVIEWER_PROFILE)
                    ),
                    "agent_profile_sha256": str(
                        call.get("agent_profile_sha256", "UNKNOWN")
                    ),
                    "runtime_profile_id": str(
                        call.get("runtime_profile_id", "flow-tool-free-v1")
                    ),
                    "runtime_profile_sha256": str(
                        call.get("runtime_profile_sha256", "UNKNOWN")
                    ),
                },
                invocation_metadata={
                    "dataset_id": packet["dataset_identity"]["dataset_id"],
                    "family_id": family_id,
                    "concept_id": concept_id,
                    "validation_role": "CROSS_FAMILY_FLOW_EXPERT",
                },
            )
        if artifact is not None:
            _write_json(out / "reviews" / f"{family_id}.json", artifact)
            review_hashes[family_id] = canonical_json_sha256(artifact)
        rows.append(
            {
                "dataset_id": packet["dataset_identity"]["dataset_id"],
                "family_id": family_id,
                "concept_id": concept_id,
                "condition": packet["conditions"][0]["condition"],
                "case_id": packet["conditions"][0]["case_id"],
                "profile": FLOW_SCIENTIFIC_REVIEWER_PROFILE,
                "invocation_status": (
                    str(artifact.get("invocation_status", "NOT_RUN"))
                    if isinstance(artifact, Mapping)
                    else "NOT_RUN"
                ),
                "review_validation_status": (
                    str(validated_review.get("status", "NOT_RUN"))
                    if isinstance(validated_review, Mapping)
                    else "NOT_RUN"
                ),
                "scientific_observations": list(condition_reviews),
                "family_binding_status": provenance["family_binding_status"],
                "evidence_provenance_closure": provenance[
                    "evidence_provenance_closure"
                ],
                "claim_evidence_sufficiency": provenance[
                    "claim_evidence_sufficiency"
                ],
                "unresolved_claim_ids": list(provenance["unresolved_claim_ids"]),
                "review_permitted": True,
                "review_permitted_semantics": (
                    "FAMILY_AND_DECLARED_CITATION_PROVENANCE_ONLY; "
                    "NOT_SCIENTIFIC_SUPPORT"
                ),
                "visible_claim_count": provenance["visible_claim_count"],
                "evidence_count": provenance["evidence_count"],
                "source_count": provenance["source_count"],
                "deterministic_condition_result": "NOT_APPLICABLE_VALIDATION_ONLY",
                "lifecycle_promotion_performed": False,
                "forbidden_lifecycle_paths": lifecycle_hits,
                "packet_sha256": packet_hash,
                "review_sha256": review_hashes.get(family_id),
                "immutable_run_id": (
                    immutable_run.get("run_id") if immutable_run else None
                ),
                "immutable_run_packet_sha256": (
                    immutable_run.get("packet_sha256") if immutable_run else None
                ),
                "immutable_run_review_sha256": (
                    immutable_run.get("review_sha256") if immutable_run else None
                ),
            }
        )

    requested_run = run_live or reviewer is not None
    valid_runs = all(
        row["review_validation_status"] == "PASS"
        and not row["forbidden_lifecycle_paths"]
        for row in rows
    )
    result = {
        "schema_version": "flow-expert-cross-family-validation-v1",
        "profile": FLOW_SCIENTIFIC_REVIEWER_PROFILE,
        "execution_mode": (
            "LIVE_MODEL_CALL"
            if run_live
            else "INJECTED_VALIDATION_REVIEWER"
            if reviewer is not None
            else "BUILD_ONLY"
        ),
        "family_count": len(rows),
        "families": rows,
        "packet_hashes": packet_hashes,
        "review_hashes": review_hashes,
        "cross_family_validation_status": (
            "PASS" if requested_run and valid_runs else "INCOMPLETE" if requested_run else "NOT_RUN"
        ),
        "packet_construction_status": "PASS",
        "lifecycle_promotion_performed": False,
        "official_scq_executed": False,
        "srac_executed": False,
        "human_calibration_status": "PENDING",
        "flow_expert_scientifically_validated": False,
    }
    _write_json(out / "manifest.json", result)
    (out / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    return result


__all__ = [
    "FLOW_EXPERT_PORTFOLIO_TARGETS",
    "FLOW_EXPERT_RESPONSIBILITY_CASE_MATRIX",
    "FLOW_EXPERT_SCIENTIFIC_TARGET_CASE_MATRIX",
    "FlowExpertPortfolioTarget",
    "build_flow_expert_portfolio_packets",
    "build_flow_expert_responsibility_matrix_packets",
    "build_flow_expert_scientific_target_packets",
    "run_flow_expert_portfolio_validation",
]

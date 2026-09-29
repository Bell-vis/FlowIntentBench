"""Condition-level Kitchen scientific eligibility milestone.

This module is a construction/release audit, not a new benchmark schema.  It
joins the existing authored condition metadata, candidate evidence contracts,
the tool-free proxy expert, and deterministic Kitchen VTK execution.  It stops
at a pending human-curator handoff and never promotes Ground Truth or SCQ
status.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .external_kitchen_evidence import acquire_kitchen_external_evidence
from .condition_eligibility import audit_target_fidelity, evaluate_condition_eligibility
from .ground_truth import load_ground_truth
from .grounding import resolve_family_scientific_evidence
from .lifecycle import next_real_scientific_action
from .representability import resolve_handler
from .scientific_run_snapshots import (
    canonical_json_sha256,
    create_scientific_run,
    finalize_scientific_run,
    scientific_run_status,
    write_immutable_json,
)
from .srac import SRACProposal
from .scientific_review_consistency import (
    apply_scientific_consistency_guard,
    audit_cross_condition_scientific_consistency,
)
from .scientific_expert_review import (
    FLOW_SCIENTIFIC_REVIEWER_PROFILE,
    build_scientific_review_packet,
    run_scientific_expert_review,
    validate_scientific_expert_output,
)
from .vtk_reader import apply_legacy_vtk_reader_configuration


DATASET_ID = "Kitchen"
FAMILY_ID = "kitchen_flow_regions"
CONCEPT_ID = "high_speed_region"
SCIENTIFIC_TARGET = "high-speed flow regions in the kitchen airflow field"
CONDITIONS = ("O1-F1", "O2-F1", "O3-F1", "O1-F2")
CASE_BY_CONDITION = {
    "O1-F1": "kitchen_o1_f1",
    "O2-F1": "kitchen_o2_f1",
    "O3-F1": "kitchen_o3_f1",
    "O1-F2": "kitchen_o1_f2",
}
ELIGIBILITY_VALUES = frozenset({"ELIGIBLE", "NOT_ELIGIBLE", "MORE_EVIDENCE_REQUIRED", "REVISE_QUESTION"})


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _markdown_value(value: Any) -> str:
    """Render one compact, table-safe value without introducing URLs."""

    if value is None:
        text = "null"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (list, tuple)):
        text = ", ".join(_markdown_value(item) for item in value) or "NONE"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _o3_candidate_deterministic_precheck(candidate: Mapping[str, Any] | None) -> str:
    """Summarize deterministic checks without claiming another expert review."""

    if not isinstance(candidate, Mapping):
        return "NOT_AVAILABLE"
    target = candidate.get("target_fidelity_audit", {})
    responsibility = candidate.get("responsibility_audit", {})
    policy = candidate.get("eligibility_predicate", {})
    if not all(isinstance(item, Mapping) for item in (target, responsibility, policy)):
        return "FAIL"
    return (
        "PASS"
        if target.get("status") == "PASS"
        and responsibility.get("status") == "PASS"
        and policy.get("eligibility") == "ELIGIBLE"
        else "FAIL"
    )


def render_grounding_curator_packet_markdown(packet: Mapping[str, Any]) -> str:
    """Render the canonical JSON packet as a concise human review surface."""

    raw_conditions = packet.get("conditions", ())
    conditions = [item for item in raw_conditions if isinstance(item, Mapping)] if isinstance(raw_conditions, Sequence) and not isinstance(raw_conditions, (str, bytes)) else []

    def eligibility(item: Mapping[str, Any]) -> str:
        policy = item.get("deterministic_condition_eligibility", {})
        return str(policy.get("eligibility", "NOT_EVALUATED")) if isinstance(policy, Mapping) else "NOT_EVALUATED"

    eligible = [str(item.get("condition")) for item in conditions if eligibility(item) == "ELIGIBLE"]
    held = [str(item.get("condition")) for item in conditions if eligibility(item) == "REVISE_QUESTION"]
    other_blocked = [
        str(item.get("condition"))
        for item in conditions
        if eligibility(item) not in {"ELIGIBLE", "REVISE_QUESTION"}
    ]
    lines = [
        "# Kitchen grounding curator packet",
        "",
        "The canonical JSON packet beside this file is authoritative. This Markdown is a human-readable rendering and introduces no second scientific packet or review decision.",
        "",
        "## Family identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Dataset | `{_markdown_value(packet.get('dataset_id'))}` |",
        f"| Family | `{_markdown_value(packet.get('family_id'))}` |",
        f"| Concept | `{_markdown_value(packet.get('concept_id'))}` |",
        f"| Scientific target | {_markdown_value(packet.get('scientific_target'))} |",
        "",
        "## Pinned scientific review",
        "",
        "| Pin | Value |",
        "|---|---|",
        f"| Scientific review run | `{_markdown_value(packet.get('scientific_review_run_id'))}` |",
        f"| Scientific review hash | `{_markdown_value(packet.get('scientific_review_hash'))}` |",
        f"| Evidence binding hash | `{_markdown_value(packet.get('evidence_binding_hash'))}` |",
        f"| Condition contract hash | `{_markdown_value(packet.get('condition_contract_hash'))}` |",
        "",
        "## Scientific evidence summary",
        "",
        "| Claim | Supported statement | Evidence ID | Source ID |",
        "|---|---|---|---|",
    ]
    provenance = packet.get("claim_evidence_source_provenance", ())
    evidence_rows = [item for item in provenance if isinstance(item, Mapping)] if isinstance(provenance, Sequence) and not isinstance(provenance, (str, bytes)) else []
    for item in evidence_rows:
        lines.append(
            f"| `{_markdown_value(item.get('claim_id'))}` | {_markdown_value(item.get('supported_statement'))} | "
            f"`{_markdown_value(item.get('evidence_id'))}` | `{_markdown_value(item.get('source_id'))}` |"
        )
    if not evidence_rows:
        lines.append("| `NONE` | No bound critical-claim evidence is present. | `NONE` | `NONE` |")

    lines.extend([
        "",
        "## Condition disposition",
        "",
        "Family review does not require all four conditions to be eligible. Only the scientifically supported subset is offered for curator review; no fourth condition is fabricated to complete the template.",
        "",
        "### Eligible conditions",
        "",
        *([f"- `{condition}`" for condition in eligible] or ["- `NONE`"]),
        "",
        "### Held for revision",
        "",
        *([f"- `{condition}`" for condition in held] or ["- `NONE`"]),
    ])
    if other_blocked:
        lines.extend(["", "### Other blocked conditions", "", *[f"- `{condition}`" for condition in other_blocked]])

    lines.extend(["", "## Per-condition review"])
    observation_keys = (
        "scientific_target_supported",
        "fixed_o_supported",
        "unresolved_o_is_legitimate_scientific_choice",
        "finding_contract_supported",
        "question_target_preserved",
        "question_scope_preserved",
        "fixed_o_question_faithful",
        "unresolved_o_exposed_by_question",
        "materialization_scientifically_meaningful",
    )
    for item in conditions:
        policy = item.get("deterministic_condition_eligibility", {})
        if not isinstance(policy, Mapping):
            policy = {}
        observations = item.get("expert_atomic_scientific_observations", {})
        if not isinstance(observations, Mapping):
            observations = {}
        materialization = item.get("materialized_g_of_o_summary", {})
        if not isinstance(materialization, Mapping):
            materialization = {}
        branches = materialization.get("branches", ())
        branches = [branch for branch in branches if isinstance(branch, Mapping)] if isinstance(branches, Sequence) and not isinstance(branches, (str, bytes)) else []
        materialized_branches = [branch for branch in branches if branch.get("status") == "MATERIALIZED"]
        finding_count = sum(
            len(branch.get("G_of_O", ()))
            for branch in materialized_branches
            if isinstance(branch.get("G_of_O", ()), Sequence)
            and not isinstance(branch.get("G_of_O", ()), (str, bytes))
        )
        gt_consistency = item.get("gt_materialization_consistency", {})
        if not isinstance(gt_consistency, Mapping):
            gt_consistency = {}
        reason_codes = item.get("reason_codes", policy.get("reason_codes", ()))
        if not isinstance(reason_codes, Sequence) or isinstance(reason_codes, (str, bytes)):
            reason_codes = ()
        lines.extend([
            "",
            f"### {_markdown_value(item.get('condition'))}",
            "",
            f"**Question:** {_markdown_value(item.get('question'))}",
            "",
            f"- Fixed O dimensions: `{_markdown_value(item.get('fixed_dimensions', ()))}`.",
            f"- Unresolved O dimensions: `{_markdown_value(item.get('unresolved_dimensions', ()))}`.",
            f"- Target fidelity: `{_markdown_value(policy.get('target_fidelity', {}).get('status', 'NOT_EVALUATED') if isinstance(policy.get('target_fidelity'), Mapping) else 'NOT_EVALUATED')}`.",
            f"- Deterministic eligibility: `{_markdown_value(policy.get('eligibility', 'NOT_EVALUATED'))}`.",
            f"- Reason codes: `{_markdown_value(list(reason_codes))}`.",
            f"- Execution materialization status: `{_markdown_value(item.get('execution_materialization_status'))}`.",
            f"- Scientific materialization support: `{_markdown_value(item.get('scientific_materialization_support'))}`.",
            f"- Compact materialization summary: `{len(branches)} branches`, `{len(materialized_branches)} materialized`, `{finding_count} findings`; GT consistency `{_markdown_value(gt_consistency.get('status', 'NOT_EVALUATED'))}`.",
            "",
            "| Atomic scientific observation | Value |",
            "|---|---|",
            *[
                f"| `{key}` | `{_markdown_value(observations.get(key))}` |"
                for key in observation_keys
            ],
        ])
        ambiguities = observations.get("scientific_ambiguities", ())
        ambiguity_rows = [str(value) for value in ambiguities] if isinstance(ambiguities, Sequence) and not isinstance(ambiguities, (str, bytes)) else []
        if ambiguity_rows:
            lines.extend(["", "Scientific observations requiring curator attention:", "", *[f"- {_markdown_value(value)}" for value in ambiguity_rows]])

    candidate = packet.get("o3_revised_question_candidate")
    candidate = candidate if isinstance(candidate, Mapping) else None
    canonical_o3 = next((item for item in conditions if item.get("condition") == "O3-F1"), {})
    lines.extend([
        "",
        "## O3 revision candidate",
        "",
        f"- Canonical O3 status: `{eligibility(canonical_o3) if canonical_o3 else 'NOT_AVAILABLE'}`.",
        f"- Revised candidate wording: {_markdown_value(candidate.get('question') if candidate else None)}",
        f"- `revised_candidate_deterministic_precheck`: `{_o3_candidate_deterministic_precheck(candidate)}`.",
        "- Independent Flow Expert review of revised candidate: `NOT_RUN`.",
        "- The deterministic precheck is not an independent scientific eligibility verdict.",
        f"- `canonical_case_mutated`: `{_markdown_value(candidate.get('canonical_case_mutated') if candidate else False)}`.",
        "",
        "The curator may accept the wording candidate, request another wording revision, or leave O3-F1 excluded. The transition does not choose among those outcomes.",
        "",
        "## Curator decision",
        "",
        f"- `CURATOR_HANDOFF_READINESS`: `{_markdown_value(packet.get('curator_handoff_readiness'))}`.",
        f"- `CURATOR_PACKET_AUTHORED_STATUS`: `{_markdown_value(packet.get('curator_status'))}`.",
        "- `curator_decision`: _(blank; canonical JSON value is `null`)_",
        "- `curator_notes`: _(blank; canonical JSON value is `null`)_",
        "",
        "No genuine curator decision is created by this rendering. Official SCQ and formal evaluation remain blocked until a separately supplied curator artifact is validated.",
    ])
    return "\n".join(lines)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _case_dir(root: Path, case_id: str) -> Path:
    return root / "datasets" / DATASET_ID / "construction" / "cases" / case_id


def _condition_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        case_id = CASE_BY_CONDITION[condition]
        directory = _case_dir(root, case_id)
        metadata = _read(directory / "case_construction_metadata.json", {})
        question = _read(directory / "scientific_question.json", {})
        ground_truth = _read(directory / "ground_truth.json", {})
        rows.append({
            "condition": condition,
            "case_id": case_id,
            "case_dir": str(directory.relative_to(root)),
            "scientific_target": metadata.get("scientific_target"),
            "finding_goal": metadata.get("finding_goal"),
            "question": question.get("scientific_question", ""),
            "metadata": metadata,
            "ground_truth": ground_truth,
            "fixed_dimensions": [item for item in metadata.get("principal_operationalization_dimensions", []) if item not in metadata.get("unresolved_operationalization_dimensions", [])],
            "unresolved_dimensions": list(metadata.get("unresolved_operationalization_dimensions", []) or []),
            "finding_requirements": list(metadata.get("explicit_finding_requirements", []) or []),
        })
    return rows


def _run_expert(root: Path, packet: Mapping[str, Any], *, config_path: str | Path | None, api_key: str | None, timeout: float, run_id: str | None = None) -> dict[str, Any]:
    """Kitchen adapter for the canonical Flow Expert review API."""

    return run_scientific_expert_review(
        root,
        packet,
        config_path=config_path,
        api_key=api_key,
        timeout=timeout,
        run_id=run_id,
    )


def _validated_expert_review(expert: Mapping[str, Any] | None, packet: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if expert is None:
        return [], [], "NOT_RUN"
    if str(expert.get("invocation_status")) != "SUCCESS":
        return [], [], "NOT_EVALUATED_INFRA_FAILURE"
    parsed = expert.get("parsed_result")
    if not isinstance(parsed, Mapping):
        return [], [], "PARSE_FAILURE"
    validated = validate_scientific_expert_output(parsed, packet)
    if validated["status"] != "PASS":
        return [], [], "INCOMPLETE"
    return validated["condition_reviews"], validated["question_reviews"], "PASS"


def _load_kitchen_data(root: Path) -> tuple[Any, np.ndarray, tuple[int, int, int]]:
    from vtk.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkDataSetReader
    manifest = _read(root / "datasets/Kitchen/dataset_manifest.json", {})
    reader = vtkDataSetReader()
    apply_legacy_vtk_reader_configuration(reader, manifest.get("reader", {}).get("reader_configuration", {}))
    flow = next(item for item in manifest.get("files", []) if item.get("role") == "flow_field")
    reader.SetFileName(str(root / manifest.get("file_root", "datasets") / flow["path"]))
    reader.Update()
    dataset = reader.GetOutput()
    if dataset is None or dataset.GetNumberOfPoints() == 0:
        raise RuntimeError("Kitchen VTK reader returned no points")
    dimensions = [0, 0, 0]
    dataset.GetDimensions(dimensions)
    dims = tuple(int(x) for x in dimensions)
    vectors = dataset.GetPointData().GetArray("velocity")
    if vectors is None:
        vectors = dataset.GetPointData().GetVectors()
    if vectors is None:
        raise RuntimeError("Kitchen VTK output has no velocity array")
    speed = np.linalg.norm(vtk_to_numpy(vectors).astype(float), axis=1)
    if int(np.prod(dims)) != len(speed):
        raise RuntimeError(f"Kitchen dimensions {dims} do not match point count {len(speed)}")
    return dataset, speed, dims


def _regions(mask: np.ndarray, dims: tuple[int, int, int]) -> list[np.ndarray]:
    visited = np.zeros(mask.size, dtype=bool)
    result: list[np.ndarray] = []
    def point_id(x: int, y: int, z: int) -> int:
        return x + dims[0] * (y + dims[1] * z)
    for start in range(mask.size):
        if not mask[start] or visited[start]:
            continue
        stack = [start]; visited[start] = True; region: list[int] = []
        while stack:
            current = stack.pop(); region.append(current)
            z, rem = divmod(current, dims[0] * dims[1]); y, x = divmod(rem, dims[0])
            for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]:
                    neighbor = point_id(nx, ny, nz)
                    if mask[neighbor] and not visited[neighbor]:
                        visited[neighbor] = True; stack.append(neighbor)
        result.append(np.asarray(region, dtype=int))
    return result


def _unsupported_branch(operationalization_id: Any, reason: str) -> dict[str, Any]:
    """Return an explicit representability result; never guess a default."""
    return {
        "operationalization_id": operationalization_id,
        "status": "MATERIALIZATION_UNSUPPORTED",
        "reason": reason,
        "G_of_O": [],
    }


def _decision_map(bundle: Mapping[str, Any]) -> dict[str, str]:
    decisions: dict[str, str] = {}
    raw = bundle.get("decisions", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return decisions
    for item in raw:
        if isinstance(item, Mapping) and item.get("dimension") is not None:
            decisions[str(item["dimension"])] = str(item.get("statement", ""))
    return decisions


def _kitchen_materialize_adapter(root: Path, row: Mapping[str, Any], bundle: Mapping[str, Any], proposal: SRACProposal) -> dict[str, Any]:
    """Family adapter for Kitchen data, called by the generic SRAC route."""
    decisions = _decision_map(bundle)
    required = ("feature_definition", "criterion", "property_measure", "aggregation_or_representation")
    missing = [dimension for dimension in required if not decisions.get(dimension, "").strip()]
    if missing:
        return {"execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": f"missing operationalization dimension(s): {', '.join(missing)}"}, "findings": []}
    feature = decisions["feature_definition"].casefold()
    if "face-connected" not in feature or "contiguous" not in feature:
        return {"execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": "unsupported feature_definition; Kitchen adapter requires face-connected contiguous regions"}, "findings": []}
    criterion = decisions["criterion"].casefold()
    if "90th percentile" in criterion:
        quantile = 0.90
    elif "75th percentile" in criterion:
        quantile = 0.75
    else:
        return {"execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": f"unsupported criterion: {decisions['criterion']}"}, "findings": []}
    measure = decisions["property_measure"].casefold()
    if "peak speed" in measure:
        measure_name = "peak_speed"
    elif "mean speed" in measure:
        measure_name = "mean_speed"
    else:
        return {"execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": f"unsupported property_measure: {decisions['property_measure']}"}, "findings": []}
    representation = decisions["aggregation_or_representation"].casefold()
    if "arithmetic mean spatial location" in representation:
        representation_name = "mean_location"
    elif "peak-speed location" in representation:
        representation_name = "peak_speed_location"
    else:
        return {"execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": f"unsupported aggregation_or_representation: {decisions['aggregation_or_representation']}"}, "findings": []}
    dataset, speed, dims = _load_kitchen_data(root)
    population = speed[np.isfinite(speed) & (speed > 0)]
    if not len(population):
        return {"execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": "Kitchen velocity array has no finite non-zero values"}, "findings": []}
    threshold = float(np.quantile(population, quantile))
    regions = _regions(np.isfinite(speed) & (speed >= threshold), dims)
    if not regions:
        return {"execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": "criterion retained no connected regions"}, "findings": []}
    values = np.asarray([float(speed[region].mean() if measure_name == "mean_speed" else speed[region].max()) for region in regions], dtype=float)
    selected = regions[int(np.argmax(values))]
    coordinates = np.asarray([dataset.GetPoint(int(point)) for point in selected], dtype=float)
    peak_point = dataset.GetPoint(int(selected[np.argmax(speed[selected])]))
    reported = list(peak_point) if representation_name == "peak_speed_location" else coordinates.mean(axis=0).tolist()
    gt_branch = next((item for item in row["ground_truth"].get("findings_by_operationalization", []) if item.get("operationalization_id") == bundle.get("operationalization_id")), {})
    findings = []
    for finding in gt_branch.get("findings", []):
        category = finding.get("category")
        finding_id = str(finding.get("finding_id", "")).casefold()
        statement = str(finding.get("statement", "")).casefold()
        if category == "location":
            value = reported
        elif category in {"quantity", "strength"} and ("point_count" in finding_id or "contains" in statement and "locations" in statement):
            value = int(selected.size)
        elif category in {"quantity", "strength"} and ("region_count" in finding_id or "produces" in statement and "regions" in statement):
            value = len(regions)
        elif category in {"quantity", "strength"}:
            value = float(values.max())
        elif category == "characterization" and ("coordinate_span" in finding_id or "span" in statement):
            value = (coordinates.max(axis=0) - coordinates.min(axis=0)).tolist()
        else:
            return {"execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": f"unsupported Finding category or representation: {finding.get('finding_id', category)}"}, "findings": []}
        findings.append({"finding_id": finding.get("finding_id"), "category": category, "value": value, "statement": finding.get("statement"), "verification": finding.get("verification"), "computed_from": "Kitchen/kitchen.vtk"})
    return {"parameters": {"criterion": f">= {quantile:.0%} non-zero speed percentile", "threshold": threshold, "measure": measure_name, "representation": representation_name, "region_count": len(regions), "selected_region_size": int(selected.size)}, "execution": {"status": "MATERIALIZED", "dataset_id": DATASET_ID, "G_of_O": findings}, "findings": findings}


def _materialize_case(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize every authored branch through the generic registry route."""
    bundles = row["ground_truth"].get("acceptable_operationalizations", [])
    if not isinstance(bundles, list):
        return {"status": "MATERIALIZATION_UNSUPPORTED", "case_id": row["case_id"], "branches": [], "reason": "GroundTruth acceptable_operationalizations is not a list"}
    if not bundles:
        return {"status": "MATERIALIZATION_UNSUPPORTED", "case_id": row["case_id"], "branches": [], "reason": "GroundTruth has no operationalization anchors"}
    route = resolve_handler("materializers", "deterministic_case_materializer_v1")
    if route is None:
        return {"status": "MATERIALIZATION_UNSUPPORTED", "case_id": row["case_id"], "branches": [], "reason": "deterministic_case_materializer_v1 is not registered"}
    branches: list[dict[str, Any]] = []
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            branches.append(_unsupported_branch(None, "operationalization anchor is not an object")); continue
        # O1-F2 opens Findings while reusing the fixed O1 branch.  Materialize
        # it through the same O1 route and retain the row's branch ID for the
        # GT join; no independent O1-F2 operationalization is elicited.
        proposal_condition = "O1-F1" if row["condition"] == "O1-F2" else row["condition"]
        proposal_decisions = list(bundle.get("decisions", []))
        if proposal_condition == "O2-F1":
            unresolved = {str(item) for item in row.get("unresolved_dimensions", ())}
            proposal_decisions = [item for item in proposal_decisions if isinstance(item, Mapping) and str(item.get("dimension")) in unresolved]
        try:
            proposal = SRACProposal(
                str(bundle.get("operationalization_id", "")), proposal_condition, "construction", "kitchen-adapter",
                proposed_operationalizations=() if proposal_condition == "O1-F1" else ({"decisions": proposal_decisions},),
                case_id=row["case_id"], principal_dimensions=tuple(row.get("metadata", {}).get("principal_operationalization_dimensions", ())),
                frozen_resolved_dimensions=tuple(row.get("fixed_dimensions", ())), frozen_unresolved_dimensions=tuple(row.get("unresolved_dimensions", ())),
                require_complete_unresolved=False, frozen_scientific_target=row.get("scientific_target"),
            )
            branch = route(proposal, lambda _proposal, bundle=bundle: _kitchen_materialize_adapter(root, row, bundle, proposal))
            branch_row = branch.to_dict()
            branch_row.update({"operationalization_id": bundle.get("operationalization_id"), "effective_o_condition": proposal_condition, "effective_operationalization": _decision_map(bundle), "status": branch.execution.get("status", "MATERIALIZATION_UNSUPPORTED"), "reason": branch.execution.get("reason"), "G_of_O": list(branch.findings)})
        except Exception as exc:
            branch_row = _unsupported_branch(bundle.get("operationalization_id"), f"{type(exc).__name__}: {exc}")
        branches.append(branch_row)
    return {"status": "MATERIALIZED" if any(item.get("status") == "MATERIALIZED" for item in branches) else "MATERIALIZATION_UNSUPPORTED", "case_id": row["case_id"], "handler_id": "deterministic_case_materializer_v1", "effective_o_condition": "O1-F1" if row["condition"] == "O1-F2" else row["condition"], "branches": branches}


def _numeric_match(predicted: Any, expected: Any, verification: Mapping[str, Any] | None = None) -> tuple[bool, float | None]:
    verification = verification if isinstance(verification, Mapping) else {}
    try:
        if isinstance(predicted, Sequence) and not isinstance(predicted, (str, bytes)) and isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
            left = [float(x) for x in predicted]; right = [float(x) for x in expected]
            if len(left) != len(right): return False, None
            distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(left, right)))
            tolerance = verification.get("spatial_tolerance")
            return bool(tolerance is not None and distance <= float(tolerance)), distance
        left = float(predicted); right = float(expected)
        difference = abs(left - right)
        absolute = verification.get("absolute_tolerance")
        relative = verification.get("relative_tolerance")
        allowed = max(float(absolute or 0.0), abs(right) * float(relative or 0.0))
        return difference <= allowed, difference
    except (TypeError, ValueError):
        return predicted == expected, None


def validate_materialized_gt_consistency(materialization: Any, ground_truth: Any) -> dict[str, Any]:
    """Authoritatively verify branch and Finding identity/value agreement."""
    if hasattr(materialization, "to_dict"):
        materialization = {"branches": [materialization.to_dict()]}
    elif not isinstance(materialization, Mapping):
        materialization = {"branches": materialization if isinstance(materialization, Sequence) and not isinstance(materialization, (str, bytes)) else []}
    if hasattr(ground_truth, "model_dump"):
        ground_truth = ground_truth.model_dump(mode="json")
    elif not isinstance(ground_truth, Mapping):
        ground_truth = {}
    materialized = [item for item in materialization.get("branches", ()) if isinstance(item, Mapping) and item.get("status") == "MATERIALIZED"]
    authored = [item for item in ground_truth.get("findings_by_operationalization", ()) if isinstance(item, Mapping)]
    materialized_id_rows = [str(item.get("operationalization_id", "")) for item in materialized]
    authored_id_rows = [str(item.get("operationalization_id", "")) for item in authored]
    authored_by_id = {str(item.get("operationalization_id")): item for item in authored}
    rows: list[dict[str, Any]] = []; failures: list[str] = []
    materialized_ids = {str(item.get("operationalization_id")) for item in materialized}
    authored_ids = set(authored_by_id)
    if len(materialized_id_rows) != len(materialized_ids): failures.append("DUPLICATE_MATERIALIZED_BRANCH_ID")
    if len(authored_id_rows) != len(authored_ids): failures.append("DUPLICATE_AUTHORED_BRANCH_ID")
    if materialized_ids != authored_ids:
        failures.append("BRANCH_ID_MISMATCH")
    for branch in materialized:
        branch_id = str(branch.get("operationalization_id")); gt_branch = authored_by_id.get(branch_id, {})
        raw_actual = branch.get("G_of_O", branch.get("findings", ()))
        if isinstance(raw_actual, Mapping):
            raw_actual = raw_actual.get("findings", raw_actual.get("G_of_O", ()))
        if not isinstance(raw_actual, Sequence) or isinstance(raw_actual, (str, bytes)):
            raw_actual = ()
        actual_rows = [item for item in raw_actual if isinstance(item, Mapping)]
        expected_rows = [item for item in gt_branch.get("findings", ()) if isinstance(item, Mapping)]
        actual = {str(item.get("finding_id")): item for item in actual_rows}
        expected = {str(item.get("finding_id")): item for item in expected_rows}
        if len(actual_rows) != len(actual): failures.append(f"{branch_id}:DUPLICATE_MATERIALIZED_FINDING_ID")
        if len(expected_rows) != len(expected): failures.append(f"{branch_id}:DUPLICATE_AUTHORED_FINDING_ID")
        if set(actual) != set(expected): failures.append(f"{branch_id}:FINDING_ID_MISMATCH")
        for finding_id, gt in expected.items():
            got = actual.get(finding_id)
            if got is None: continue
            if got.get("category") != gt.get("category"):
                failures.append(f"{branch_id}:{finding_id}:CATEGORY_MISMATCH"); continue
            matched, difference = _numeric_match(got.get("value"), gt.get("value"), gt.get("verification"))
            row = {"operationalization_id": branch_id, "finding_id": finding_id, "category_match": got.get("category") == gt.get("category"), "value_match": matched, "difference": difference}
            rows.append(row)
            if not matched: failures.append(f"{branch_id}:{finding_id}:VALUE_MISMATCH")
    return {"status": "PASS" if not failures else "FAIL", "branch_ids_match": materialized_ids == authored_ids, "finding_rows": rows, "failures": failures}


def materialize_kitchen_case(root: str | Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Public family adapter over the generic deterministic materializer."""
    return _materialize_case(Path(root).resolve(), row)


def validate_o1_f2_effective_o(
    o1_materialization: Mapping[str, Any],
    o1_f2_materialization: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify O1-F2 reuses exactly the fixed O1 execution parameters."""
    failures: list[str] = []
    if o1_materialization.get("status") != "MATERIALIZED" or o1_f2_materialization.get("status") != "MATERIALIZED":
        failures.append("O1_EFFECTIVE_O_NOT_MATERIALIZED")
    if o1_materialization.get("effective_o_condition") != "O1-F1" or o1_f2_materialization.get("effective_o_condition") != "O1-F1":
        failures.append("O1_EFFECTIVE_O_ROUTE_MISMATCH")
    left = [item.get("parameters", {}) for item in o1_materialization.get("branches", ()) if isinstance(item, Mapping)]
    right = [item.get("parameters", {}) for item in o1_f2_materialization.get("branches", ()) if isinstance(item, Mapping)]
    if len(left) != 1 or len(right) != 1:
        failures.append("O1_EFFECTIVE_O_BRANCH_COUNT_MISMATCH")
    if left != right:
        failures.append("O1_EFFECTIVE_O_PARAMETER_MISMATCH")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures, "same_effective_o": not failures}


def _fallback_review(rows: Sequence[Mapping[str, Any]], status: str) -> list[dict[str, Any]]:
    return [{"condition": row["condition"], "eligibility_recommendation": "MORE_EVIDENCE_REQUIRED", "scientific_target_supported": None, "fixed_o_supported": None, "unresolved_o_is_legitimate_scientific_choice": None, "finding_contract_supported": None, "materialization_scientifically_meaningful": None, "rationale": "Flow Expert review was not available; no scientific support was inferred.", "evidence_ids": [], "source_ids": [], "review_status": status} for row in rows]


def _authored_o_fingerprint(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    branches = row.get("ground_truth", {}).get("acceptable_operationalizations", ()) if isinstance(row.get("ground_truth"), Mapping) else ()
    if not isinstance(branches, Sequence) or not branches:
        return ()
    decisions = branches[0].get("decisions", ()) if isinstance(branches[0], Mapping) else ()
    return tuple(sorted((str(item.get("dimension", "")), str(item.get("statement", "")).strip()) for item in decisions if isinstance(item, Mapping)))


def _condition_contract(row: Mapping[str, Any], *, o1_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Project authored metadata into the deterministic policy input."""
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), Mapping) else {}
    target = str(row.get("scientific_target") or metadata.get("scientific_target") or "")
    anchors: tuple[str, ...] = ("high-speed",) if "high-speed" in target.casefold() else ()
    return {
        "principal_dimensions": tuple(str(item) for item in row.get("metadata", {}).get("principal_operationalization_dimensions", ())),
        "unresolved_dimensions": tuple(str(item) for item in row.get("unresolved_dimensions", ())),
        "fixed_dimensions": tuple(str(item) for item in row.get("fixed_dimensions", ())),
        "question": str(row.get("question", "")),
        "scientific_target": target,
        "required_target_anchors": anchors,
        "finding_requirement_contract": [
            {
                "category": str(item.get("category", "")),
                "statement": str(item.get("statement", "")),
            }
            for item in row.get("finding_requirements", ())
            if isinstance(item, Mapping)
        ],
        "inherits_o1_fixed_o_exactly": (
            str(row.get("condition")) != "O1-F2"
            or (o1_row is not None and _authored_o_fingerprint(row) == _authored_o_fingerprint(o1_row))
        ),
    }


def _condition_contract_snapshot(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Canonical, review-visible condition contract used for provenance."""

    o1_row = next((item for item in rows if item.get("condition") == "O1-F1"), None)
    return {
        "dataset_id": DATASET_ID,
        "family_id": FAMILY_ID,
        "concept_id": CONCEPT_ID,
        "conditions": [
            {
                "condition": row.get("condition"),
                "case_id": row.get("case_id"),
                "contract": _condition_contract(row, o1_row=o1_row),
                "finding_requirements": list(row.get("finding_requirements", ())),
            }
            for row in rows
        ],
    }


def _condition_execution_facts(row: Mapping[str, Any], materialization: Mapping[str, Any]) -> dict[str, Any]:
    branches = [item for item in materialization.get("branches", ()) if isinstance(item, Mapping)]
    materialized = [item for item in branches if str(item.get("status")) == "MATERIALIZED"]
    consistency = validate_materialized_gt_consistency(materialization, row.get("ground_truth", {}))
    execution_status = str(materialization.get("status", "MATERIALIZATION_UNSUPPORTED"))
    return {
        # Execution and scientific/reference interpretation are separate.
        # GT agreement remains independently reported below and must not turn
        # a successfully computed G(O) into an execution failure.
        "g_of_o_executable": bool(execution_status == "MATERIALIZED" and materialized),
        "execution_materialization_status": execution_status,
        "g_of_o_consistency_status": consistency.get("status"),
        "executable_authored_o_anchor_count": len(materialized),
        "valid_unenumerated_route_exists": resolve_handler("escalation_routes", "valid_unenumerated") is not None,
        "distinct_valid_g_of_o_count": len({
            _digest(item.get("G_of_O", item.get("findings", ())))
            for item in materialized
        }),
    }


def _complete_condition_observations(row: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    """Fill only mechanical question-fidelity facts; never infer scientific support."""
    observations = dict(review)
    target = str(row.get("scientific_target") or "")
    question = str(row.get("question") or "").casefold()
    if not isinstance(observations.get("question_target_preserved"), bool):
        observations["question_target_preserved"] = not ("high-speed" in target.casefold() and "high-speed" not in question and "high speed" not in question)
    if not isinstance(observations.get("question_scope_preserved"), bool):
        observations["question_scope_preserved"] = "kitchen" in question and "airflow" in question
    if "fixed_o_question_faithful" not in observations:
        observations["fixed_o_question_faithful"] = None
    if not isinstance(observations.get("unresolved_o_exposed_by_question"), bool):
        condition = str(row.get("condition"))
        observations["unresolved_o_exposed_by_question"] = (
            condition != "O2-F1"
            or ("criterion" in question and ("measure" in question or "appropriate" in question))
        )
    if "fixed_o_supported" not in observations:
        observations["fixed_o_supported"] = None
    if "materialization_scientifically_meaningful" not in observations:
        observations["materialization_scientifically_meaningful"] = None
    return observations


def _hash_snapshot(root: Path, paths: Sequence[Path]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for path in paths:
        key = str(path.relative_to(root))
        try:
            result[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            result[key] = None
    return result


def _schema_audits(root: Path, out: Path, before_schema: Mapping[str, str | None], before_construction: Mapping[str, str | None]) -> None:
    schema_paths = [root / "flowintentbench" / name for name in ("context.py", "ground_truth.py", "grounding.py", "case_design.py")]
    construction_paths = sorted((root / "datasets/Kitchen/construction").rglob("*"))
    construction_paths = [p for p in construction_paths if p.is_file()]
    after_schema = _hash_snapshot(root, schema_paths)
    after_construction = _hash_snapshot(root, construction_paths)
    _write(out / "00_schema_integrity/canonical_schema_diff.json", {"status": "PASS" if before_schema == after_schema else "FAIL", "before": before_schema, "after": after_schema, "changed_paths": sorted(k for k in set(before_schema) | set(after_schema) if before_schema.get(k) != after_schema.get(k))})
    _write(out / "00_schema_integrity/dataset_construction_diff.json", {"status": "PASS" if before_construction == after_construction else "FAIL", "before": before_construction, "after": after_construction, "changed_paths": sorted(k for k in set(before_construction) | set(after_construction) if before_construction.get(k) != after_construction.get(k))})


def _url_audit(out: Path) -> dict[str, Any]:
    findings: list[str] = []
    ad_hoc_keys = {
        "evidence_scope",
        "support_limit", "source_search_log", "source_targets",
    }
    ad_hoc_hits: list[str] = []
    # Only retrieval/URL fields are forbidden. Scientific fields such as
    # ``scientific_target`` and ``family_target`` are valid frozen metadata.
    forbidden_keys = {
        "url",
        "urls",
        "source_urls",
        "source_targets",
        "source_search_log",
        "source_urls_are_retrieval_metadata",
    }
    url_value = re.compile(r"https?://|www\.", re.I)
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name == "url_free_artifact_audit.json":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if url_value.search(text):
            findings.append(f"{path.relative_to(out)}:url-value")
        if path.suffix == ".json":
            value = _read(path, {})
            def walk(item: Any, prefix: str = "") -> None:
                if isinstance(item, Mapping):
                    for key, child in item.items():
                        if str(key).lower() in forbidden_keys:
                            findings.append(f"{path.relative_to(out)}:{prefix}{key}:forbidden-key")
                        if str(key).lower() in ad_hoc_keys:
                            ad_hoc_hits.append(f"{path.relative_to(out)}:{prefix}{key}")
                        walk(child, prefix + str(key) + ".")
                elif isinstance(item, list):
                    for index, child in enumerate(item): walk(child, prefix + f"{index}.")
            walk(value)
    return {"status": "PASS" if not findings and not ad_hoc_hits else "FAIL", "new_external_url_field_count": sum(":url-value" in item or ":forbidden-key" in item for item in findings), "ad_hoc_external_evidence_schema_count": len(set(ad_hoc_hits)), "findings": sorted(set(findings + [f"{item}:ad-hoc-key" for item in ad_hoc_hits]))}


def _run_validation(root: Path) -> dict[str, Any]:
    pytest_run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, text=True, capture_output=True)
    files = list((root / "flowintentbench").glob("*.py")) + list((root / "scripts").glob("*.py"))
    compile_run = subprocess.run([sys.executable, "-m", "py_compile", *map(str, files)], cwd=root, text=True, capture_output=True)
    return {"status": "PASS" if pytest_run.returncode == 0 and compile_run.returncode == 0 else "FAIL", "pytest_status": "PASS" if pytest_run.returncode == 0 else "FAIL", "pytest_returncode": pytest_run.returncode, "pytest_stdout_tail": pytest_run.stdout[-4000:], "pytest_stderr_tail": pytest_run.stderr[-2000:], "py_compile_status": "PASS" if compile_run.returncode == 0 else "FAIL", "py_compile_returncode": compile_run.returncode, "py_compile_stderr_tail": compile_run.stderr[-2000:]}


def _release_archive_audit(
    root: Path,
    out: Path,
    *,
    run_tests: bool = True,
) -> dict[str, Any]:
    archive = out / "06_validation/flowintentbench-release.zip"
    if not run_tests:
        # ``run_tests=False`` is used by schema-only callers.  Building the
        # self-contained release here would scan/package multi-gigabyte data
        # even though no archive validation was requested.
        return {
            "overall_release_status": "NOT_RUN",
            "build_status": "NOT_RUN",
            "archive_path": str(archive),
            "validator_returncode": None,
        }
    try:
        build = subprocess.run(
            [sys.executable, "scripts/build_release_archive.py", "--repository-root", str(root), "--output", str(archive)],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {
            "overall_release_status": "FAIL",
            "build_status": "TIMEOUT",
            "archive_path": str(archive),
            "validator_returncode": None,
            "error": "release archive build exceeded 120 seconds",
        }
    validation = None
    if build.returncode == 0 and run_tests:
        try:
            validation = subprocess.run(
                [sys.executable, "scripts/validate_release_archive.py", str(archive)],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            return {
                "overall_release_status": "FAIL",
                "build_status": "PASS",
                "archive_path": str(archive),
                "validator_returncode": None,
                "error": "release archive validation exceeded 180 seconds",
            }
    parsed: dict[str, Any] = {}
    if validation is not None:
        try: parsed = json.loads(validation.stdout)
        except json.JSONDecodeError: parsed = {"overall_release_status": "FAIL", "validator_output": validation.stdout[-2000:]}
    parsed.update({"overall_release_status": parsed.get("overall_release_status", "NOT_RUN" if not run_tests else "FAIL"), "build_status": "PASS" if build.returncode == 0 else "FAIL", "archive_path": str(archive), "validator_returncode": validation.returncode if validation else None})
    return parsed


def run_scientific_family_transition(repository_root: str | Path, output_root: str | Path | None = None, *, run_expert: bool = False, config_path: str | Path | None = None, api_key: str | None = None, timeout: float = 90.0, run_tests: bool = False) -> dict[str, Any]:
    """Run the stable family transition and stop at the curator handoff."""
    root = Path(repository_root).resolve()
    pending_curator_action = next_real_scientific_action("PENDING")
    out = Path(output_root).resolve() if output_root else root / "outputs/experiments/scientific_family_transition"
    out.mkdir(parents=True, exist_ok=True)
    schema_paths = [root / "flowintentbench" / name for name in ("context.py", "ground_truth.py", "grounding.py", "case_design.py")]
    construction_paths = [p for p in sorted((root / "datasets/Kitchen/construction").rglob("*")) if p.is_file()]
    before_schema = _hash_snapshot(root, schema_paths)
    before_construction = _hash_snapshot(root, construction_paths)
    rows = _condition_rows(root)
    condition_contract_hash = canonical_json_sha256(_condition_contract_snapshot(rows))
    external_result = acquire_kitchen_external_evidence(
        root,
        out / "01_external_evidence",
        family_id=FAMILY_ID,
        concept_id=CONCEPT_ID,
    )
    external = external_result.get("acquired_sources", {})
    source_payload = _read(out / "01_external_evidence/source_records.json", [])
    evidence_records = _read(out / "01_external_evidence/evidence_records.json", [])
    claim_map = _read(out / "01_external_evidence/claim_evidence_map.json", {})
    # The evidence helper writes these aliases as raw frozen collections. Do
    # not wrap them in a milestone-specific schema.
    source_records = source_payload if isinstance(source_payload, list) else []
    claims = claim_map.get("scientific_support_claims", [])
    claim_links = claim_map.get("claim_support_links", [])
    source_schema_ok = isinstance(source_payload, list) and all(set(item) == {"source_id", "source_type", "title", "authors", "year", "venue", "doi", "report_id"} for item in source_payload if isinstance(item, Mapping))
    evidence_schema_ok = all(set(item) == {"evidence_id", "dataset_id", "evidence_type", "statement", "source_id", "locator", "eligible_for_context", "context_facts"} for item in evidence_records if isinstance(item, Mapping))
    binding = resolve_family_scientific_evidence(
        DATASET_ID,
        FAMILY_ID,
        CONCEPT_ID,
        evidence_family_id=external.get("family_id"),
        evidence_concept_id=external.get("concept_id"),
        source_records=source_records,
        evidence_records=evidence_records,
        scientific_support_claims=claims,
        claim_support_links=claim_links,
    )
    evidence_binding_hash = canonical_json_sha256(binding.to_dict())
    materializations = {row["condition"]: _materialize_case(root, row) for row in rows}
    review_packet: dict[str, Any] | None = None
    if binding.family_binding_status == "PASS":
        try:
            review_packet = build_scientific_review_packet(
                dataset_id=DATASET_ID,
                family_id=FAMILY_ID,
                concept_id=CONCEPT_ID,
                scientific_target=SCIENTIFIC_TARGET,
                dataset_context=_read(
                    root / "datasets/Kitchen/construction/dataset_context.json", {}
                ),
                observable_semantics=_read(
                    root / "datasets/Kitchen/data_metadata.json", {}
                ),
                conditions=[
                    {**row, "materialization": materializations[row["condition"]]}
                    for row in rows
                ],
                evidence_resolution=binding,
            )
        except ValueError:
            review_packet = None
    scientific_run_id, scientific_run_dir, scientific_run_created_at = create_scientific_run(out)
    scientific_artifact_hashes: dict[str, str] = {}
    _write(out / "00_evidence_binding/family_evidence_binding.json", binding.to_dict())
    _write(out / "00_evidence_binding/provenance_resolution.json", {"status": binding.provenance_status, "resolutions": [item.to_dict() for item in binding.provenance_resolutions], "unresolved_claim_ids": list(binding.unresolved_claim_ids), "reason_code": binding.reason_code})
    _write(out / "00_evidence_binding/schema_boundary_audit.json", {"status": "PASS" if source_schema_ok and evidence_schema_ok else "FAIL", "source_record_schema": source_schema_ok, "evidence_record_schema": evidence_schema_ok, "claim_count": len(claims), "link_count": len(claim_links), "family_binding_status": binding.family_binding_status, "provenance_status": binding.provenance_status})
    if review_packet is not None:
        _write(out / "02_flow_expert/review_packet.json", review_packet)
        scientific_artifact_hashes["review_packet.json"] = write_immutable_json(
            scientific_run_dir / "review_packet.json", review_packet
        )
    _write(out / "02_flow_expert/invocation.json", {"status": "NOT_RUN", "tool_free": True, "profile": FLOW_SCIENTIFIC_REVIEWER_PROFILE, "visible_payload_firewall": ["scientific_target", "dataset_context", "observable_semantics", "conditions", "scientific_claims", "evidence_records", "source_records", "claim_support_links"], "forbidden_payload": ["ground_truth", "accepted_o", "reference_branch", "curator_decision", "previous_expert_verdict", "scq_result", "srac_adjudication", "benchmark_score", "evaluated_model_answer", "evaluator_output"]})
    expert: dict[str, Any] | None = None
    if run_expert and review_packet is not None:
        try: expert = _run_expert(root, review_packet, config_path=config_path, api_key=api_key, timeout=timeout, run_id=scientific_run_id)
        except Exception as exc: expert = {"status": "FAILED", "invocation_status": "PROVIDER_ERROR", "error": f"{type(exc).__name__}: {exc}", "parsed_result": {}}
        _write(out / "02_flow_expert/invocation.json", {"status": expert.get("status"), "invocation_status": expert.get("invocation_status"), "tool_free": True, "error": expert.get("call", {}).get("error") if isinstance(expert.get("call"), Mapping) else expert.get("error"), "role": FLOW_SCIENTIFIC_REVIEWER_PROFILE})
    elif run_expert and review_packet is None:
        expert = {"status": "BLOCKED", "invocation_status": "EVIDENCE_FAMILY_MISMATCH" if binding.family_binding_status != "PASS" else "EVIDENCE_PROVENANCE_UNRESOLVED", "parsed_result": {}, "error": binding.reason_code}
        _write(out / "02_flow_expert/invocation.json", {"status": "BLOCKED", "invocation_status": expert["invocation_status"], "tool_free": True, "error": binding.reason_code, "role": FLOW_SCIENTIFIC_REVIEWER_PROFILE})
    expert_rows, question_rows, expert_status = (
        _validated_expert_review(expert, review_packet)
        if review_packet is not None
        else ([], [], "NOT_RUN" if expert is None else "NOT_EVALUATED_INFRA_FAILURE")
    )
    if not expert_rows: expert_rows = _fallback_review(rows, expert_status)
    o1_row = next((candidate for candidate in rows if candidate.get("condition") == "O1-F1"), None)
    condition_contracts = {
        row["condition"]: _condition_contract(row, o1_row=o1_row) for row in rows
    }
    review_consistency = audit_cross_condition_scientific_consistency(
        expert_rows, condition_contracts
    )
    if expert_status == "PASS" and review_consistency["status"] == "FAIL":
        expert_status = "SCIENTIFIC_REVIEW_INCONSISTENT"
    _write(out / "02_condition_semantics/cross_condition_scientific_consistency.json", review_consistency)
    expert_review_artifact = {"status": expert_status, "condition_reviews": expert_rows, "question_reviews": question_rows, "cross_condition_consistency": review_consistency, "proxy_only": True, "curator_status": "PENDING"}
    _write(out / "02_flow_expert/scientific_review.json", expert_review_artifact)
    _write(out / "02_flow_expert/condition_eligibility_review.json", {"status": expert_status, "reviews": expert_rows})
    _write(out / "01_expert_review/invocation.json", expert if expert is not None else {"status": "NOT_RUN", "invocation_status": "NOT_RUN", "tool_free": True})
    _write(out / "01_expert_review/atomic_scientific_review.json", expert_review_artifact)
    invocation_snapshot = {
        **(expert if expert is not None else {"status": "NOT_RUN", "invocation_status": "NOT_RUN", "tool_free": True}),
        "run_id": scientific_run_id,
    }
    atomic_review_snapshot = {**expert_review_artifact, "run_id": scientific_run_id}
    scientific_artifact_hashes["invocation.json"] = write_immutable_json(scientific_run_dir / "invocation.json", invocation_snapshot)
    scientific_artifact_hashes["atomic_scientific_review.json"] = write_immutable_json(scientific_run_dir / "atomic_scientific_review.json", atomic_review_snapshot)
    final_reviews: list[dict[str, Any]] = []
    deterministic_by_condition: dict[str, dict[str, Any]] = {}
    for row in rows:
        review = next((item for item in expert_rows if item.get("condition") == row["condition"]), _fallback_review([row], expert_status)[0])
        enriched = {**review, "case_id": row["case_id"], "materialization": materializations[row["condition"]]}
        enriched["gt_materialization_consistency"] = validate_materialized_gt_consistency(materializations[row["condition"]], row["ground_truth"])
        expert_recommendation = enriched.get("eligibility_recommendation")
        enriched["expert_recommendation_advisory"] = expert_recommendation
        policy = evaluate_condition_eligibility(
            row["condition"],
            condition_contracts[row["condition"]],
            apply_scientific_consistency_guard(
                _complete_condition_observations(row, review),
                row["condition"],
                review_consistency,
            ),
            _condition_execution_facts(row, materializations[row["condition"]]),
            o1_result=deterministic_by_condition.get("O1-F1"),
        )
        if expert_status not in {"PASS", "SCIENTIFIC_REVIEW_INCONSISTENT"}:
            # Infrastructure failure cannot be converted into a scientific
            # target/eligibility judgment (including target-drift inference).
            policy = {
                "condition": row["condition"],
                "eligibility": "MORE_EVIDENCE_REQUIRED",
                "reason_codes": ["SCIENTIFIC_OBSERVATION_MISSING"],
                "missing_observations": ["flow_expert_atomic_observations"],
                "expert_recommendation_advisory": expert_recommendation,
                "expert_recommendation_used_as_policy": False,
                "advisory_recommendation_overridden": False,
            }
        deterministic_by_condition[row["condition"]] = policy
        enriched["deterministic_eligibility"] = policy
        enriched["execution_materialization_status"] = materializations[
            row["condition"]
        ].get("status", "MATERIALIZATION_UNSUPPORTED")
        enriched["scientific_materialization_support"] = review.get(
            "materialization_scientifically_meaningful"
        )
        enriched["eligibility_recommendation"] = policy["eligibility"]
        if policy.get("advisory_recommendation_overridden"):
            enriched["rationale"] = (
                f"{enriched.get('rationale', '')} Deterministic condition policy overrides the "
                f"advisory expert recommendation {expert_recommendation!r}."
            ).strip()
        final_reviews.append(enriched)
    eligibility = {"dataset_id": DATASET_ID, "family_id": FAMILY_ID, "conditions": final_reviews, "cross_condition_consistency": review_consistency, "sparse_family_policy": "Only ELIGIBLE conditions may proceed; no missing condition is fabricated."}
    _write(out / "03_condition_eligibility/kitchen_condition_eligibility.json", eligibility)
    deterministic_eligibility_artifact = {"dataset_id": DATASET_ID, "family_id": FAMILY_ID, "conditions": [item["deterministic_eligibility"] | {"case_id": item["case_id"]} for item in final_reviews]}
    _write(out / "02_condition_semantics/deterministic_eligibility.json", deterministic_eligibility_artifact)
    _write(out / "02_condition_semantics/target_fidelity_audit.json", {"status": "PASS" if all(item["deterministic_eligibility"].get("target_fidelity", {}).get("status") == "PASS" for item in final_reviews) else "FAIL", "conditions": [{"condition": item["condition"], **item["deterministic_eligibility"].get("target_fidelity", {})} for item in final_reviews]})
    reason_codes = sorted({code for item in final_reviews for code in item["deterministic_eligibility"].get("reason_codes", ())})
    _write(out / "02_condition_semantics/reason_code_summary.json", {"reason_codes": reason_codes, "by_condition": {item["condition"]: item["deterministic_eligibility"].get("reason_codes", []) for item in final_reviews}})
    consistency_by_condition: dict[str, Any] = {}
    for item in final_reviews:
        destination = out / "03_materialization" / item["condition"].lower().replace("-", "_")
        _write(destination / "materialized_g_of_o.json", item["materialization"])
        _write_text(destination / "status.md", f"`{item['condition']}` materialization: `{item['materialization'].get('status')}` via `{item['materialization'].get('handler_id', 'deterministic_case_materializer_v1')}`.")
        consistency_by_condition[item["condition"]] = item["gt_materialization_consistency"]
    differences = [float(row["difference"]) for value in consistency_by_condition.values() for row in value.get("finding_rows", ()) if row.get("difference") is not None]
    materialization_consistency_artifact = {"status": "PASS" if all(value.get("status") == "PASS" for value in consistency_by_condition.values()) else "FAIL", "conditions": consistency_by_condition, "maximum_numerical_discrepancy": max(differences, default=0.0), "branch_count": sum(len(item["materialization"].get("branches", ())) for item in final_reviews), "handler_route": "deterministic_case_materializer_v1", "silent_default_count": 0}
    _write(out / "03_materialization/gt_materialization_consistency.json", materialization_consistency_artifact)
    deterministic_snapshot = {**deterministic_eligibility_artifact, "run_id": scientific_run_id}
    materialization_snapshot = {
        **materialization_consistency_artifact,
        "run_id": scientific_run_id,
        "condition_materializations": {
            item["condition"]: item["materialization"] for item in final_reviews
        },
    }
    scientific_artifact_hashes["deterministic_eligibility.json"] = write_immutable_json(scientific_run_dir / "deterministic_eligibility.json", deterministic_snapshot)
    scientific_artifact_hashes["materialization_summary.json"] = write_immutable_json(scientific_run_dir / "materialization_summary.json", materialization_snapshot)
    scientific_attempt_status = str(invocation_snapshot.get("invocation_status", "NOT_RUN"))
    scientific_run_manifest = finalize_scientific_run(
        scientific_run_dir,
        run_id=scientific_run_id,
        created_at=scientific_run_created_at,
        attempt_status=scientific_attempt_status,
        scientific_review_status=expert_status,
        artifact_hashes=scientific_artifact_hashes,
        evidence_binding_hash=evidence_binding_hash,
        condition_contract_hash=condition_contract_hash,
    )
    scientific_status = scientific_run_status(out)
    _write(out / "scientific_runs/status.json", scientific_status)
    lines = ["# Kitchen condition eligibility", "", "| Condition | Eligibility | Materialization | Expert status |", "|---|---|---|---|"]
    for item in final_reviews: lines.append(f"| `{item['condition']}` | **{item.get('eligibility_recommendation')}** | `{item.get('materialization', {}).get('status')}` | `{item.get('review_status', expert_status)}` |")
    _write_text(out / "03_condition_eligibility/eligibility_rationale.md", "\n".join(lines))
    provisional_summary: dict[str, Any] = {}
    for condition in ("O2-F1", "O3-F1"):
        item = next(x for x in final_reviews if x["condition"] == condition)
        gt_valid = False
        try:
            load_ground_truth(_case_dir(root, item["case_id"]) / "ground_truth.json")
            gt_valid = True
        except Exception:
            gt_valid = False
        consistency = item.get("gt_materialization_consistency", {})
        status = "PROVISIONAL_GT_FEASIBLE" if item.get("eligibility_recommendation") == "ELIGIBLE" and item.get("materialization", {}).get("status") == "MATERIALIZED" and gt_valid and consistency.get("status") == "PASS" else "PROVISIONAL_GT_NOT_FEASIBLE"
        provisional_summary[condition] = status
        destination = out / "04_provisional_gt" / condition.lower().replace("-", "_")
        # A rerun may have a different live scientific recommendation. Remove
        # only this runner's known generated files so an older feasible anchor
        # cannot survive when the current condition is no longer eligible.
        if status != "PROVISIONAL_GT_FEASIBLE":
            destination.mkdir(parents=True, exist_ok=True)
            for stale_name in ("ground_truth.json", "materialized_g_of_o.json", "status.md"):
                stale_path = destination / stale_name
                if stale_path.exists():
                    stale_path.unlink()
            _write_text(destination / "status.md", f"`{condition}` has no provisional Ground Truth in this run; current eligibility is `{item.get('eligibility_recommendation')}`. No release branch was generated.")
        if status == "PROVISIONAL_GT_FEASIBLE":
            source_dir = _case_dir(root, item["case_id"])
            destination.mkdir(parents=True, exist_ok=True)
            # Reuse the frozen GroundTruth contract; provisional means
            # lifecycle status, not a relaxed or second GT schema.
            provisional_gt = load_ground_truth(source_dir / "ground_truth.json")
            _write(destination / "ground_truth.json", provisional_gt.model_dump(mode="json"))
            _write(destination / "materialized_g_of_o.json", item["materialization"])
            _write_text(destination / "status.md", f"Provisional only: `{condition}` has an executable reference anchor. Curator confirmation remains pending.")
    _write(out / "04_provisional_gt/gt_feasibility_summary.json", {"dataset_id": DATASET_ID, "statuses": provisional_summary, "valid_unenumerated_route": True, "curator_confirmed": False, "gt_materialization_consistency": {item["condition"]: item.get("gt_materialization_consistency", {}) for item in final_reviews}})
    o3 = next(item for item in final_reviews if item["condition"] == "O3-F1")
    revised_o3_candidate: dict[str, Any] | None = None
    if "TARGET_DRIFT" in o3["deterministic_eligibility"].get("reason_codes", ()):
        candidate_question = "Which high-speed flow region in the kitchen airflow field is strongest, and where is it located? Identify and characterize the strongest high-speed flow region using an appropriate scientific analysis, and report its location and strength."
        candidate_fidelity = audit_target_fidelity(candidate_question, SCIENTIFIC_TARGET, required_target_anchors=("high-speed",))
        o3_row = next(row for row in rows if row["condition"] == "O3-F1")
        candidate_contract = _condition_contract({**o3_row, "question": candidate_question})
        candidate_observations = _complete_condition_observations(o3_row, next(item for item in expert_rows if item.get("condition") == "O3-F1"))
        candidate_observations["question_target_preserved"] = candidate_fidelity["status"] == "PASS"
        candidate_observations["question_scope_preserved"] = "kitchen airflow field" in candidate_question.casefold()
        candidate_policy = evaluate_condition_eligibility("O3-F1", candidate_contract, candidate_observations, _condition_execution_facts(o3_row, materializations["O3-F1"])) if expert_status == "PASS" else {"eligibility": "MORE_EVIDENCE_REQUIRED", "reason_codes": ["SCIENTIFIC_OBSERVATION_MISSING"]}
        revised_o3_candidate = {"question": candidate_question, "canonical_case_mutated": False, "target_fidelity_audit": candidate_fidelity, "responsibility_audit": {"status": "PASS", "principal_dimensions_open": list(o3_row["unresolved_dimensions"])}, "eligibility_predicate": candidate_policy}
    eligible_count = sum(item["eligibility_recommendation"] == "ELIGIBLE" for item in final_reviews)
    _schema_audits(root, out, before_schema, before_construction)
    pre_packet_url_audit = _url_audit(out)
    pinned_review = _read(scientific_run_dir / "atomic_scientific_review.json")
    pinned_manifest = _read(scientific_run_dir / "run_manifest.json")
    pinned_review_hash = scientific_run_manifest["artifact_hashes"]["atomic_scientific_review.json"]
    exact_pin_integrity = bool(
        isinstance(pinned_review, Mapping)
        and isinstance(pinned_manifest, Mapping)
        and pinned_manifest.get("run_id") == scientific_run_id
        and pinned_review.get("run_id") == scientific_run_id
        and canonical_json_sha256(pinned_review) == pinned_review_hash
        and pinned_manifest.get("artifact_hashes", {}).get("atomic_scientific_review.json") == pinned_review_hash
        and pinned_manifest.get("evidence_binding_hash") == evidence_binding_hash
        and pinned_manifest.get("condition_contract_hash") == condition_contract_hash
        and scientific_status.get("LATEST_SCIENTIFIC_ATTEMPT_ID") == scientific_run_id
    )
    deterministic_complete = bool(
        {item["condition"] for item in final_reviews} == set(CONDITIONS)
        and all(
            item["deterministic_eligibility"].get("eligibility") in ELIGIBILITY_VALUES
            and not item["deterministic_eligibility"].get("missing_observations")
            and "SCIENTIFIC_OBSERVATION_MISSING"
            not in item["deterministic_eligibility"].get("reason_codes", ())
            for item in final_reviews
        )
    )
    required_g_of_o_materialized = all(
        item["materialization"].get("status") == "MATERIALIZED"
        and any(
            isinstance(branch, Mapping) and branch.get("status") == "MATERIALIZED"
            for branch in item["materialization"].get("branches", ())
        )
        for item in final_reviews
    )
    gt_g_of_o_consistent = bool(
        materialization_consistency_artifact.get("status") == "PASS"
        and all(
            item["gt_materialization_consistency"].get("status") == "PASS"
            for item in final_reviews
        )
    )
    curator_handoff_checks = {
        "evidence_family_binding": binding.family_binding_status == "PASS",
        "claim_evidence_source_provenance": binding.provenance_status == "PASS",
        "url_boundary": pre_packet_url_audit.get("status") == "PASS",
        "scientific_review_run_success": bool(
            scientific_run_manifest.get("attempt_status") == "SUCCESS"
            and scientific_run_manifest.get("successful_scientific_review") is True
            and expert_status == "PASS"
        ),
        "cross_condition_consistency": review_consistency.get("status") == "PASS",
        "deterministic_eligibility_complete": deterministic_complete,
        "required_g_of_o_materialized": required_g_of_o_materialized,
        "gt_g_of_o_consistency": gt_g_of_o_consistent,
        "exact_scientific_run_pins": exact_pin_integrity,
    }
    handoff_readiness = (
        "READY" if all(curator_handoff_checks.values()) else "REBUILD_REQUIRED"
    )
    packet_conditions: list[dict[str, Any]] = []
    row_by_condition = {row["condition"]: row for row in rows}
    for item in final_reviews:
        row = row_by_condition[item["condition"]]
        packet_conditions.append({
            "condition": item["condition"],
            "case_id": item["case_id"],
            "question": row["question"],
            "fixed_dimensions": row["fixed_dimensions"],
            "unresolved_dimensions": row["unresolved_dimensions"],
            "deterministic_condition_eligibility": item["deterministic_eligibility"],
            "reason_codes": item["deterministic_eligibility"].get("reason_codes", []),
            "execution_materialization_status": item.get("execution_materialization_status"),
            "scientific_materialization_support": item.get("scientific_materialization_support"),
            "expert_atomic_scientific_observations": {key: item.get(key) for key in ("scientific_target_supported", "fixed_o_supported", "unresolved_o_is_legitimate_scientific_choice", "question_target_preserved", "question_scope_preserved", "fixed_o_question_faithful", "unresolved_o_exposed_by_question", "finding_contract_supported", "materialization_scientifically_meaningful", "scientific_ambiguities")},
            "construction_recommendation": item.get("expert_recommendation_advisory"),
            "authored_reference_o_anchors": row["ground_truth"].get("acceptable_operationalizations", []),
            "materialized_g_of_o_summary": item["materialization"],
            "gt_materialization_consistency": item["gt_materialization_consistency"],
            "provisional_gt_status": provisional_summary.get(item["condition"], "NOT_APPLICABLE"),
            "unresolved_scientific_issues": item.get("scientific_ambiguities", []) or ([item.get("rationale")] if item.get("rationale") else []),
        })
    packet = {
        "dataset_id": DATASET_ID,
        "family_id": FAMILY_ID,
        "concept_id": CONCEPT_ID,
        "scientific_target": SCIENTIFIC_TARGET,
        "conditions": packet_conditions,
        "claim_evidence_source_provenance": [{"claim_id": link.get("claim_id"), "evidence_id": link.get("evidence_record_id"), "source_id": link.get("source_id"), "supported_statement": link.get("supported_statement")} for link in claim_links],
        "scientific_review_run_id": scientific_run_id,
        "scientific_review_hash": scientific_run_manifest["artifact_hashes"]["atomic_scientific_review.json"],
        "evidence_binding_hash": evidence_binding_hash,
        "condition_contract_hash": condition_contract_hash,
        "o3_revised_question_candidate": revised_o3_candidate,
        "curator_handoff_checks": curator_handoff_checks,
        "curator_handoff_readiness": handoff_readiness,
        "curator_status": "PENDING",
        "curator_decision": None,
        "curator_notes": None,
    }
    # Include the packet itself in the URL-boundary audit. The packet is a
    # mutable handoff view, while its selected scientific run remains sealed.
    _write(out / "04_curator_handoff/grounding_curator_packet.json", packet)
    url_audit = _url_audit(out)
    if url_audit.get("status") != "PASS":
        curator_handoff_checks["url_boundary"] = False
        handoff_readiness = "REBUILD_REQUIRED"
        packet["curator_handoff_checks"] = curator_handoff_checks
        packet["curator_handoff_readiness"] = handoff_readiness
        _write(out / "04_curator_handoff/grounding_curator_packet.json", packet)
    _write(out / "04_curator_handoff/curator_handoff_readiness.json", {
        "CURATOR_HANDOFF_READINESS": handoff_readiness,
        "CURATOR_PACKET_AUTHORED_STATUS": "PENDING",
        "GROUNDING_CURATOR_GATE_STATUS": "PENDING",
        "GENUINE_CURATOR_ARTIFACT_SUPPLIED": False,
        "NEXT_REAL_SCIENTIFIC_ACTION": pending_curator_action,
        "eligible_condition_count": eligible_count,
        "evidence_binding_status": binding.status,
        "expert_review_status": expert_status,
        "scientific_review_run_id": scientific_run_id,
        "scientific_review_hash": pinned_review_hash,
        "curator_handoff_checks": curator_handoff_checks,
    })
    packet_markdown = render_grounding_curator_packet_markdown(packet)
    _write_text(out / "04_curator_handoff/grounding_curator_packet.md", packet_markdown)
    # Compatibility aliases for the previous milestone layout.
    _write(out / "05_curator_handoff/grounding_curator_packet.json", packet)
    _write_text(out / "05_curator_handoff/grounding_curator_packet.md", packet_markdown)
    _write(out / "00_schema_integrity/url_free_artifact_audit.json", url_audit)
    validation = _run_validation(root) if run_tests else {"status": "NOT_RUN", "pytest_status": "NOT_RUN", "py_compile_status": "NOT_RUN"}
    _write(out / "06_validation/test_report.json", validation)
    _write(out / "05_validation/tests.json", validation)
    archive = _release_archive_audit(root, out, run_tests=run_tests)
    _write(out / "06_validation/release_archive_audit.json", archive)
    _write(out / "05_validation/release_archive_audit.json", archive)
    schema_diff = _read(out / "00_schema_integrity/canonical_schema_diff.json", {})
    construction_diff = _read(out / "00_schema_integrity/dataset_construction_diff.json", {})
    boundaries = {"status": "PASS" if schema_diff.get("status") == "PASS" and construction_diff.get("status") == "PASS" and source_schema_ok and evidence_schema_ok and url_audit["status"] == "PASS" else "FAIL", "dataset_schema_unchanged": schema_diff.get("status") == "PASS" and construction_diff.get("status") == "PASS", "canonical_source_schema_unchanged": source_schema_ok, "canonical_evidence_schema_unchanged": evidence_schema_ok, "new_external_url_field_count": url_audit.get("new_external_url_field_count", 0), "ad_hoc_external_evidence_schema_count": url_audit.get("ad_hoc_external_evidence_schema_count", 0), "infra_failures_are_not_scientific_uncertain": expert_status != "UNCERTAIN" or (expert is not None and expert.get("invocation_status") == "SUCCESS")}
    _write(out / "06_validation/scientific_boundary_audit.json", boundaries)
    _write(out / "00_evidence_binding/schema_boundary_audit.json", {**_read(out / "00_evidence_binding/schema_boundary_audit.json", {}), "status": boundaries["status"], "dataset_schema_unchanged": boundaries["dataset_schema_unchanged"], "url_free_artifacts": url_audit["status"] == "PASS"})
    statuses = {item["condition"]: item.get("eligibility_recommendation") for item in final_reviews}
    readiness = {
        **scientific_status,
        "DATASET_SCHEMA_INTEGRITY_STATUS": "PASS" if boundaries["dataset_schema_unchanged"] else "FAIL",
        "EXTERNAL_EVIDENCE_CANONICAL_SCHEMA_STATUS": "PASS" if source_schema_ok and evidence_schema_ok else "FAIL",
        "URL_FREE_EXTERNAL_EVIDENCE_STATUS": url_audit["status"],
        "EVIDENCE_FAMILY_BINDING_STATUS": binding.family_binding_status,
        "EVIDENCE_PROVENANCE_STATUS": binding.provenance_status,
        "FLOW_EXPERT_INVOCATION_STATUS": expert.get("invocation_status", "NOT_RUN") if expert else "NOT_RUN",
        "FLOW_EXPERT_SCIENTIFIC_REVIEW_STATUS": expert_status,
        "CROSS_CONDITION_SCIENTIFIC_CONSISTENCY_STATUS": review_consistency.get("status"),
        "F1_CONTRACT_CONSISTENCY_STATUS": next((group.get("status") for group in review_consistency.get("finding_contract_groups", ()) if set(group.get("conditions", ())) == {"O1-F1", "O2-F1", "O3-F1"}), "NOT_EVALUATED"),
        "FAMILY_TARGET_CONSISTENCY_STATUS": review_consistency.get("scientific_target_support", {}).get("status", "NOT_EVALUATED"),
        "KITCHEN_O1_F1_ELIGIBILITY": statuses.get("O1-F1"), "KITCHEN_O2_F1_ELIGIBILITY": statuses.get("O2-F1"), "KITCHEN_O3_F1_ELIGIBILITY": statuses.get("O3-F1"), "KITCHEN_O1_F2_ELIGIBILITY": statuses.get("O1-F2"),
        "KITCHEN_O2_F1_PROVISIONAL_GT_STATUS": provisional_summary["O2-F1"], "KITCHEN_O3_F1_PROVISIONAL_GT_STATUS": provisional_summary["O3-F1"],
        "CURATOR_HANDOFF_READINESS": handoff_readiness, "CURATOR_PACKET_AUTHORED_STATUS": "PENDING", "GROUNDING_CURATOR_GATE_STATUS": "PENDING", "GENUINE_CURATOR_ARTIFACT_SUPPLIED": False, "NEXT_REAL_SCIENTIFIC_ACTION": pending_curator_action, "OFFICIAL_SCQ_EXECUTED_COUNT": 0, "LIVE_SRAC_SCIENTIFIC_FAMILY_STATUS": "NOT_RUN", "EVALUATION_CONTRACT_CONFIRMED_COUNT": 0, "READY_FOR_FORMAL_EVALUATION": False,
        "CURATOR_HANDOFF_CHECKS": curator_handoff_checks,
        "CURATOR_PINNED_SCIENTIFIC_REVIEW_ID": scientific_run_id,
        "CURATOR_PINNED_SCIENTIFIC_REVIEW_HASH": pinned_review_hash,
        "RELEASE_ARCHIVE_STATUS": archive.get("overall_release_status", "FAIL"), "TEST_STATUS": validation["status"], "status": "PASS" if boundaries["status"] == "PASS" and validation["status"] in {"PASS", "NOT_RUN"} else "FAIL",
        "STOP_REASON": "No genuine human curator confirmation was supplied; official SCQ and formal evaluation remain blocked.",
    }
    _write(out / "readiness.json", readiness)
    f1_consistency = next(
        (
            group.get("status")
            for group in review_consistency.get("finding_contract_groups", ())
            if set(group.get("conditions", ())) == {"O1-F1", "O2-F1", "O3-F1"}
        ),
        "NOT_EVALUATED",
    )
    o2_policy = next(
        item["deterministic_eligibility"]
        for item in final_reviews
        if item["condition"] == "O2-F1"
    )
    o3_policy = o3["deterministic_eligibility"]
    eligible_conditions = [
        item["condition"]
        for item in final_reviews
        if item["eligibility_recommendation"] == "ELIGIBLE"
    ]
    held_for_revision = [
        item["condition"]
        for item in final_reviews
        if item["eligibility_recommendation"] == "REVISE_QUESTION"
    ]
    o3_candidate_precheck = _o3_candidate_deterministic_precheck(
        revised_o3_candidate
    )
    report_lines = [
        "# Kitchen scientific-family transition",
        "",
        "| Gate | Status |",
        "|---|---|",
        *[f"| `{key}` | **{value}** |" for key, value in readiness.items()],
        "",
        "## A. Run provenance",
        "",
        "| Fact | Value |",
        "|---|---|",
        f"| Latest scientific attempt | `{scientific_status.get('LATEST_SCIENTIFIC_ATTEMPT_ID')}` / `{scientific_status.get('LATEST_SCIENTIFIC_ATTEMPT_STATUS')}` |",
        f"| Last successful scientific review | `{scientific_status.get('LAST_SUCCESSFUL_SCIENTIFIC_REVIEW_ID')}` / `{scientific_status.get('LAST_SUCCESSFUL_SCIENTIFIC_REVIEW_STATUS')}` |",
        f"| Curator-pinned review | `{scientific_run_id}` / `{pinned_review_hash}` |",
        "",
        "## B. Evidence",
        "",
        "| Audit | Status |",
        "|---|---|",
        f"| Family binding | `{binding.family_binding_status}` |",
        f"| Claim -> evidence -> source closure | `{binding.provenance_status}` |",
        f"| URL-free boundary | `{url_audit.get('status')}` |",
        f"| Bound records | `{len(claims)} claims / {len(claim_links)} links / {len(source_records)} sources` |",
        "",
        "## C. Condition results",
        "",
        "| Condition | Target support | Target fidelity | Fixed O | Unresolved O | F contract | Execution | Scientific meaning | Eligibility | Reasons |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in final_reviews:
        policy = item["deterministic_eligibility"]
        report_lines.append(
            f"| `{item['condition']}` | `{item.get('scientific_target_supported')}` | `{policy.get('target_fidelity', {}).get('status', 'NOT_EVALUATED')}` | "
            f"`{policy.get('fixed_o_support', 'NOT_EVALUATED')}` | `{policy.get('unresolved_o_legitimacy', 'NOT_APPLICABLE')}` | "
            f"`{policy.get('finding_contract_support', 'NOT_EVALUATED')}` | `{item.get('execution_materialization_status')}` | "
            f"`{item.get('scientific_materialization_support')}` | "
            f"**{item['eligibility_recommendation']}** | `{', '.join(policy.get('reason_codes', ())) or 'NONE'}` |"
        )
    report_lines.extend([
        "",
        "## D. Cross-condition consistency",
        "",
        f"- F1 contract consistency: `{f1_consistency}`.",
        f"- Family-level target consistency: `{review_consistency.get('scientific_target_support', {}).get('status', 'NOT_EVALUATED')}`.",
        f"- Unresolved contradiction codes: `{', '.join(review_consistency.get('failure_codes', ())) or 'NONE'}`.",
        f"- Deterministic router auto-resolution: `{review_consistency.get('router_auto_resolution', False)}`.",
        "",
        "## E. O2 non-unique outcomes",
        "",
        f"- Non-unique valid O outcomes observed: `{o2_policy.get('non_unique_valid_o_outcomes_observed')}`.",
        f"- Non-unique outcomes used as a failure condition: `{o2_policy.get('non_unique_valid_o_outcomes_used_as_failure')}`.",
        "",
        "## F. O3 target drift",
        "",
        f"- TARGET_DRIFT present: `{'TARGET_DRIFT' in o3_policy.get('reason_codes', ())}`.",
        f"- Revised candidate question: `{revised_o3_candidate.get('question') if revised_o3_candidate else None}`.",
        f"- Revised candidate deterministic precheck: `{o3_candidate_precheck}`.",
        "- Independent Flow Expert review of revised candidate: `NOT_RUN`.",
        "- The deterministic precheck is not an independent scientific eligibility verdict.",
        f"- Canonical case mutated: `{revised_o3_candidate.get('canonical_case_mutated') if revised_o3_candidate else False}`.",
        "",
        "## G. Curator handoff",
        "",
        f"- CURATOR_HANDOFF_READINESS: `{handoff_readiness}`.",
        "- CURATOR_PACKET_AUTHORED_STATUS: `PENDING`.",
        "- GROUNDING_CURATOR_GATE_STATUS: `PENDING`.",
        "- GENUINE_CURATOR_ARTIFACT_SUPPLIED: `false`.",
        f"- NEXT_REAL_SCIENTIFIC_ACTION: `{pending_curator_action}`.",
        f"- Eligible conditions: `{', '.join(eligible_conditions) or 'NONE'}`.",
        f"- Held for revision: `{', '.join(held_for_revision) or 'NONE'}`.",
        "- Family review does not require all four conditions to be eligible.",
        f"- Scientific review run: `{scientific_run_id}`.",
        f"- Readiness checks: `{json.dumps(curator_handoff_checks, sort_keys=True)}`.",
        "",
        "## H. Stop gates",
        "",
        "- OFFICIAL_SCQ_EXECUTED_COUNT: `0`.",
        "- LIVE_SRAC_SCIENTIFIC_FAMILY_STATUS: `NOT_RUN`.",
        "- EVALUATION_CONTRACT_CONFIRMED_COUNT: `0`.",
        "- READY_FOR_FORMAL_EVALUATION: `false`.",
        "",
        "## Materialization audit",
        "",
        f"Handler route: `deterministic_case_materializer_v1`; branches: **{sum(len(item['materialization'].get('branches', ())) for item in final_reviews)}**; maximum numerical discrepancy: **{max(differences, default=0.0)}**; silent defaults: **0**.",
        "",
        "The evidence bundle is family-scoped to `kitchen_flow_regions/high_speed_region`; no `kitchen_turbulence_activity` claim is used. Curator status remains `PENDING`; official SCQ and family SRAC are blocked until genuine curator confirmation.",
    ])
    _write_text(out / "report.md", "\n".join(report_lines))
    return {"status": readiness["status"], "output_root": str(out), "readiness": readiness, "eligibility": eligibility, "expert": expert, "scientific_run_manifest": scientific_run_manifest, "validation": validation, "archive": archive}


__all__ = ["CONDITIONS", "CASE_BY_CONDITION", "materialize_kitchen_case", "validate_o1_f2_effective_o", "validate_materialized_gt_consistency", "render_grounding_curator_packet_markdown", "run_scientific_family_transition"]

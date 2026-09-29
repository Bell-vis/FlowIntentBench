"""Reusable Office reference-analysis computation.

The module owns the dataset-specific numerical analysis and Ground Truth
assembly used by the Office pilot.  CLI/context-builder concerns stay in
``scripts.build_office_formal_pilot``; callers provide the context builder at
the boundary when materializing a complete case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from flowintentbench import (  # noqa: E402
    CaseConstructionMetadata,
    FindingImportance,
    FindingRequirementCategory,
    GroundTruth,
    OperationalizationDecision,
    OperationalizationDimension,
    OperationalizationFindingBranch,
    OperationalizationBundle,
    ReferenceFinding,
    VerificationSpec,
    primary_case_type,
    save_ground_truth,
)
from flowintentbench.construction_tolerances import (  # noqa: E402
    absolute_tolerance_for_significant_figures,
)
from flowintentbench.vtk_reader import apply_legacy_vtk_reader_configuration  # noqa: E402


CASE_IDS = (
    "office_speed_zones_o1_f1",
    "office_speed_zones_o2_f1",
    "office_speed_zones_o3_f1",
    "office_speed_zones_o1_f2",
)

# Reporting precision is a construction decision for the scientific quantity,
# rather than a verifier default. Pointwise peaks retain five significant
# figures; mesh-dependent regional aggregates use four, which is the precision
# their authored pilot prose claims.
STRENGTH_REPORTING_SIGNIFICANT_FIGURES_BY_MEASURE = {
    "peak_speed": 5,
    "mean_speed": 4,
    "integrated_excess": 4,
}


def _strength_reporting_significant_figures(measure_kind: str) -> int:
    try:
        return STRENGTH_REPORTING_SIGNIFICANT_FIGURES_BY_MEASURE[measure_kind]
    except KeyError as exc:  # pragma: no cover - candidate table is closed
        raise ValueError(f"unsupported measure kind: {measure_kind}") from exc


@dataclass(frozen=True)
class CandidateSpec:
    """Construction-only candidate description; never serialized as benchmark schema."""

    candidate_id: str
    track: str
    status: str
    criterion_kind: str
    measure_kind: str
    feature_statement: str
    criterion_statement: str
    measure_statement: str
    aggregation_statement: str
    rationale: str
    # Explicit only for percentile candidates.  None preserves the existing
    # construction candidates' non-zero population semantics.
    percentile_population: str | None = None


@dataclass(frozen=True)
class OfficeData:
    dataset: Any
    speed: np.ndarray
    dimensions: tuple[int, int, int]
    point_count: int


O1_PEAK = CandidateSpec(
    candidate_id="o1_peak_speed",
    track="O1",
    status="provisionally_accepted",
    criterion_kind="fixed_threshold",
    measure_kind="peak_speed",
    feature_statement=(
        "Face-connected (6-neighbor) regions of retained high-speed locations "
        "form the analysis regions."
    ),
    criterion_statement="Retain locations whose speed magnitude is strictly greater than 0.20.",
    measure_statement="Use peak speed as the strength of each retained region.",
    aggregation_statement=(
        "Represent each selected region by the arithmetic mean spatial location "
        "of its retained high-speed locations."
    ),
    rationale="The O1 question fixes this complete operationalization.",
)

O2_CANDIDATES = (
    CandidateSpec(
        candidate_id="o2_peak_speed",
        track="O2",
        status="provisionally_accepted",
        criterion_kind="fixed_threshold",
        measure_kind="peak_speed",
        feature_statement=O1_PEAK.feature_statement,
        criterion_statement=O1_PEAK.criterion_statement,
        measure_statement="Use peak speed as the strength of each retained region.",
        aggregation_statement=O1_PEAK.aggregation_statement,
        rationale=(
            "Accepted: it directly represents the most intense local airflow "
            "within each thresholded region."
        ),
    ),
    CandidateSpec(
        candidate_id="o2_mean_speed",
        track="O2",
        status="provisionally_accepted",
        criterion_kind="fixed_threshold",
        measure_kind="mean_speed",
        feature_statement=O1_PEAK.feature_statement,
        criterion_statement=O1_PEAK.criterion_statement,
        measure_statement=(
            "Use the mean speed across retained locations as the strength of each region."
        ),
        aggregation_statement=O1_PEAK.aggregation_statement,
        rationale=(
            "Accepted: it represents sustained regional airflow intensity and is "
            "computable from the supplied velocity field."
        ),
    ),
    CandidateSpec(
        candidate_id="o2_q90_mean_speed",
        track="O2",
        status="provisionally_accepted",
        criterion_kind="nonzero_q90",
        measure_kind="mean_speed",
        feature_statement=O1_PEAK.feature_statement,
        criterion_statement=(
            "Retain locations whose speed is at or above the 90th percentile of "
            "non-zero speed values as a distribution-informed high-speed criterion."
        ),
        measure_statement=(
            "Use the mean speed across retained locations as the strength of each region."
        ),
        aggregation_statement=O1_PEAK.aggregation_statement,
        rationale=(
            "Provisionally retained: a complete, executable distribution-informed "
            "speed criterion that is materially different from the fixed threshold; "
            "its scientific grounding remains pending under the same gate as 0.20."
        ),
    ),
    CandidateSpec(
        candidate_id="o2_integrated_speed_excess",
        track="O2",
        status="rejected",
        criterion_kind="fixed_threshold",
        measure_kind="integrated_excess",
        feature_statement=O1_PEAK.feature_statement,
        criterion_statement=O1_PEAK.criterion_statement,
        measure_statement=(
            "Use the point-summed speed excess above 0.20 as the strength of each region."
        ),
        aggregation_statement=O1_PEAK.aggregation_statement,
        rationale=(
            "Rejected: a point sum is sampling-density dependent and no cell-volume "
            "integration contract is available for this point-based task."
        ),
    ),
)

O3_CANDIDATES = (
    CandidateSpec(
        candidate_id="o3_fixed_peak_speed",
        track="O3",
        status="provisionally_accepted",
        criterion_kind="fixed_threshold",
        measure_kind="peak_speed",
        feature_statement=O1_PEAK.feature_statement,
        criterion_statement=O1_PEAK.criterion_statement,
        measure_statement="Use peak speed as the strength of each retained region.",
        aggregation_statement=O1_PEAK.aggregation_statement,
        rationale=(
            "Accepted: a complete, executable interpretation of the strongest "
            "region in the Office velocity field."
        ),
    ),
    CandidateSpec(
        candidate_id="o3_fixed_mean_speed",
        track="O3",
        status="provisionally_accepted",
        criterion_kind="fixed_threshold",
        measure_kind="mean_speed",
        feature_statement=O1_PEAK.feature_statement,
        criterion_statement=O1_PEAK.criterion_statement,
        measure_statement=(
            "Use the mean speed across retained locations as the strength of each region."
        ),
        aggregation_statement=O1_PEAK.aggregation_statement,
        rationale=(
            "Accepted: a complete, meaningfully distinct sustained-intensity "
            "interpretation that is executable on the supplied data."
        ),
    ),
    CandidateSpec(
        candidate_id="o3_q90_mean_speed",
        track="O3",
        status="provisionally_accepted",
        criterion_kind="nonzero_q90",
        measure_kind="mean_speed",
        feature_statement=O1_PEAK.feature_statement,
        criterion_statement=(
            "Retain locations whose speed is at or above the 90th percentile of "
            "non-zero speed values as a distribution-informed high-speed criterion."
        ),
        measure_statement=(
            "Use the mean speed across retained locations as the strength of each region."
        ),
        aggregation_statement=O1_PEAK.aggregation_statement,
        rationale=(
            "Provisionally retained: it is a complete, executable criterion/measure "
            "combination and receives the same pending grounding status as the 0.20 "
            "criterion rather than being rejected by an asymmetric standard."
        ),
    ),
    CandidateSpec(
        candidate_id="o3_peak_point_speed",
        track="O3",
        status="provisionally_accepted",
        criterion_kind="fixed_threshold",
        measure_kind="peak_speed",
        feature_statement=O1_PEAK.feature_statement,
        criterion_statement=O1_PEAK.criterion_statement,
        measure_statement="Use peak speed as the strength of each retained region.",
        aggregation_statement=(
            "Represent the selected region by the spatial location of its peak-speed "
            "location."
        ),
        rationale=(
            "Provisionally retained: peak-point location is a complete and interpretable "
            "alternative representation of the strongest region."
        ),
    ),
    CandidateSpec(
        candidate_id="o3_local_peak_features",
        track="O3",
        status="rejected",
        criterion_kind="local_peak",
        measure_kind="peak_speed",
        feature_statement=(
            "Treat isolated local speed maxima above the selected cutoff as the airflow "
            "features rather than contiguous regions."
        ),
        criterion_statement=(
            "Retain speed locations above 0.20 that are strictly greater than their "
            "face-connected neighbors."
        ),
        measure_statement="Use the speed at each local maximum as feature strength.",
        aggregation_statement=(
            "Represent each feature by the spatial location of its local maximum."
        ),
        rationale=(
            "Rejected: execution is possible, but isolated point features do not address "
            "the family target of characterizing a contiguous airflow region."
        ),
    ),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_office_data(dataset_dir: Path) -> OfficeData:
    from vtk.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkDataSetReader

    manifest = _read_json(dataset_dir / "dataset_manifest.json")
    reader = vtkDataSetReader()
    apply_legacy_vtk_reader_configuration(
        reader, manifest["reader"]["reader_configuration"]
    )
    flow_file = next(item for item in manifest["files"] if item["role"] == "flow_field")
    reader.SetFileName(str(ROOT / manifest["file_root"] / flow_file["path"]))
    reader.Update()
    dataset = reader.GetOutput()
    if dataset is None:
        raise RuntimeError("Office canonical VTK reader returned no dataset")
    dimensions = [0, 0, 0]
    dataset.GetDimensions(dimensions)
    dims = tuple(int(value) for value in dimensions)
    point_count = dataset.GetNumberOfPoints()
    if int(np.prod(dims)) != point_count:
        raise RuntimeError(f"Office dimensions {dims} do not match {point_count} points")
    vectors = dataset.GetPointData().GetVectors()
    if vectors is None:
        raise RuntimeError("Office VTK output has no active velocity vector array")
    speed = np.linalg.norm(vtk_to_numpy(vectors).astype(np.float64), axis=1)
    return OfficeData(dataset, speed, dims, point_count)


def _regions_for_mask(mask: np.ndarray, dimensions: tuple[int, int, int]) -> list[np.ndarray]:
    visited = np.zeros(mask.size, dtype=bool)
    regions: list[np.ndarray] = []

    def point_id(x: int, y: int, z: int) -> int:
        return x + dimensions[0] * (y + dimensions[1] * z)

    for start in range(mask.size):
        if not mask[start] or visited[start]:
            continue
        stack = [start]
        visited[start] = True
        region: list[int] = []
        while stack:
            current = stack.pop()
            region.append(current)
            z, remainder = divmod(current, dimensions[0] * dimensions[1])
            y, x = divmod(remainder, dimensions[0])
            for dx, dy, dz in (
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ):
                nx, ny, nz = x + dx, y + dy, z + dz
                if not (0 <= nx < dimensions[0] and 0 <= ny < dimensions[1] and 0 <= nz < dimensions[2]):
                    continue
                neighbor = point_id(nx, ny, nz)
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        regions.append(np.asarray(region, dtype=int))
    return regions


def _analyze_candidate(data: OfficeData, spec: CandidateSpec) -> dict[str, Any]:
    if spec.criterion_kind == "fixed_threshold":
        threshold = 0.20
        mask = data.speed > threshold
        criterion_value = "> 0.20"
    elif spec.criterion_kind == "nonzero_q90":
        population = spec.percentile_population or "NONZERO_SPEED_POINTS"
        if population == "ALL_VALID_POINTS":
            values = data.speed[np.isfinite(data.speed)]
        elif population == "NONZERO_SPEED_POINTS":
            values = data.speed[data.speed > 0]
        else:
            raise ValueError(f"unsupported percentile population: {population}")
        if values.size == 0:
            raise ValueError("percentile population is empty")
        threshold = float(np.quantile(values, 0.90))
        mask = data.speed >= threshold
        criterion_value = f">= {population} 90th percentile ({threshold:.12f})"
    elif spec.criterion_kind == "local_peak":
        threshold = 0.20
        mask = data.speed > threshold
        criterion_value = "> 0.20 and strictly greater than face-connected neighbors"
    else:  # pragma: no cover - candidate table is closed in this script
        raise ValueError(f"unsupported criterion kind: {spec.criterion_kind}")

    if spec.criterion_kind == "local_peak":
        local_peak_mask = np.zeros(data.speed.size, dtype=bool)
        for point in np.flatnonzero(mask):
            z, remainder = divmod(int(point), data.dimensions[0] * data.dimensions[1])
            y, x = divmod(remainder, data.dimensions[0])
            neighbors: list[int] = []
            for dx, dy, dz in (
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ):
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < data.dimensions[0] and 0 <= ny < data.dimensions[1] and 0 <= nz < data.dimensions[2]:
                    neighbors.append(nx + data.dimensions[0] * (ny + data.dimensions[1] * nz))
            if all(data.speed[int(point)] > data.speed[neighbor] for neighbor in neighbors):
                local_peak_mask[int(point)] = True
        mask = local_peak_mask

    regions = _regions_for_mask(mask, data.dimensions)
    if not regions:
        raise RuntimeError(f"candidate {spec.candidate_id} retained no regions")
    if spec.measure_kind == "peak_speed":
        values = np.asarray([data.speed[region].max() for region in regions])
    elif spec.measure_kind == "mean_speed":
        values = np.asarray([data.speed[region].mean() for region in regions])
    elif spec.measure_kind == "integrated_excess":
        values = np.asarray([
            np.maximum(data.speed[region] - threshold, 0).sum() for region in regions
        ])
    else:  # pragma: no cover - candidate table is closed in this script
        raise ValueError(f"unsupported measure kind: {spec.measure_kind}")

    selected_index = int(np.argmax(values))
    selected = regions[selected_index]
    coordinates = np.asarray(
        [data.dataset.GetPoint(int(point)) for point in selected], dtype=float
    )
    peak_point = data.dataset.GetPoint(
        int(selected[np.argmax(data.speed[selected])])
    )
    if "peak-speed location" in spec.aggregation_statement or "local maximum" in spec.aggregation_statement:
        reported_location = list(peak_point)
        location_kind = "peak_speed_location"
    else:
        reported_location = coordinates.mean(axis=0).tolist()
        location_kind = "mean_spatial_location"
    return {
        "candidate_id": spec.candidate_id,
        "status": spec.status,
        "measure_kind": spec.measure_kind,
        "criterion": criterion_value,
        "retained_points": int(mask.sum()),
        "regions": len(regions),
        "region_sizes": [int(len(region)) for region in regions],
        "selected_region_size": int(len(selected)),
        "mean_spatial_location": coordinates.mean(axis=0).tolist(),
        "reported_location": reported_location,
        "location_kind": location_kind,
        "coordinate_span": (coordinates.max(axis=0) - coordinates.min(axis=0)).tolist(),
        "strength": float(values[selected_index]),
        "peak_speed": float(data.speed[selected].max()),
        "mean_speed": float(data.speed[selected].mean()),
        "peak_point": list(peak_point),
    }


def _candidate_specs_for_case(case_id: str) -> tuple[CandidateSpec, ...]:
    if case_id == "office_speed_zones_o1_f1":
        return (O1_PEAK,)
    if case_id == "office_speed_zones_o2_f1":
        return O2_CANDIDATES
    if case_id == "office_speed_zones_o3_f1":
        return O3_CANDIDATES
    if case_id == "office_speed_zones_o1_f2":
        return (O1_PEAK,)
    raise ValueError(f"unsupported Office pilot case: {case_id}")


def _accepted_specs_for_case(case_id: str) -> tuple[CandidateSpec, ...]:
    return tuple(
        spec
        for spec in _candidate_specs_for_case(case_id)
        if spec.status == "provisionally_accepted"
    )


def _operationalization_id(case_id: str, candidate_id: str) -> str:
    if case_id == "office_speed_zones_o1_f1":
        return "o1_speed_zones"
    if case_id == "office_speed_zones_o1_f2":
        return "o1_speed_zones_open"
    return candidate_id


def _decisions(spec: CandidateSpec) -> list[OperationalizationDecision]:
    return [
        OperationalizationDecision(
            dimension=OperationalizationDimension.FEATURE_DEFINITION,
            statement=spec.feature_statement,
        ),
        OperationalizationDecision(
            dimension=OperationalizationDimension.CRITERION,
            statement=spec.criterion_statement,
        ),
        OperationalizationDecision(
            dimension=OperationalizationDimension.PROPERTY_MEASURE,
            statement=spec.measure_statement,
        ),
        OperationalizationDecision(
            dimension=OperationalizationDimension.AGGREGATION_OR_REPRESENTATION,
            statement=spec.aggregation_statement,
        ),
    ]


def _findings(
    metadata: CaseConstructionMetadata,
    result: dict[str, Any],
    operationalization_id: str,
) -> list[ReferenceFinding]:
    location = [float(value) for value in result["reported_location"]]
    strength = float(result["strength"])
    reporting_significant_figures = _strength_reporting_significant_figures(
        result["measure_kind"]
    )
    strength_tolerance = absolute_tolerance_for_significant_figures(
        strength,
        significant_figures=reporting_significant_figures,
    )
    measure_label = {
        "peak_speed": "peak speed",
        "mean_speed": "mean speed",
        "integrated_excess": "point-summed speed excess",
    }[result["measure_kind"]]
    location_label = (
        "peak-speed location"
        if result["location_kind"] == "peak_speed_location"
        else "mean spatial location"
    )
    findings = [
        ReferenceFinding(
            finding_id=f"{operationalization_id}_mean_location",
            category=FindingRequirementCategory.LOCATION,
            statement=(
                f"The selected strongest region has {location_label} approximately "
                f"{location} in the dataset coordinate frame."
            ),
            importance=FindingImportance.CORE,
            value=location,
            verification=VerificationSpec(spatial_tolerance=0.05),
        ),
        ReferenceFinding(
            finding_id=f"{operationalization_id}_strength",
            category=FindingRequirementCategory.QUANTITY,
            statement=(
                f"The selected strongest region has {measure_label} strength approximately "
                f"{strength:.{reporting_significant_figures}g} in the stored velocity scale."
            ),
            importance=FindingImportance.CORE,
            value=strength,
            verification=VerificationSpec(absolute_tolerance=strength_tolerance),
        ),
    ]
    if metadata.case_id == "office_speed_zones_o1_f2":
        findings.append(
            ReferenceFinding(
                finding_id=f"{operationalization_id}_coordinate_span",
                category=FindingRequirementCategory.CHARACTERIZATION,
                statement=(
                    "The strongest region has coordinate-space spatial span approximately "
                    f"{result['coordinate_span']} along the dataset axes; no physical unit "
                    "is inferred."
                ),
                importance=FindingImportance.CORE,
                value=[float(value) for value in result["coordinate_span"]],
                verification=VerificationSpec(spatial_tolerance=0.05),
            )
        )
        findings.extend(
            [
                ReferenceFinding(
                    finding_id=f"{operationalization_id}_retained_point_count",
                    category=FindingRequirementCategory.QUANTITY,
                    statement=(
                        "The selected region contains "
                        f"{result['selected_region_size']} retained high-speed locations."
                    ),
                    importance=FindingImportance.SUPPORTING,
                    value=int(result["selected_region_size"]),
                ),
                ReferenceFinding(
                    finding_id=f"{operationalization_id}_thresholded_region_count",
                    category=FindingRequirementCategory.QUANTITY,
                    statement=(
                        "The criterion produces "
                        f"{result['regions']} face-connected thresholded regions in the supplied grid."
                    ),
                    importance=FindingImportance.SUPPORTING,
                    value=int(result["regions"]),
                ),
            ]
        )
    return findings


def _ground_truth(
    metadata: CaseConstructionMetadata,
    results: dict[str, dict[str, Any]],
) -> GroundTruth:
    accepted = _accepted_specs_for_case(metadata.case_id)
    bundles = []
    branches = []
    for spec in accepted:
        operationalization_id = _operationalization_id(metadata.case_id, spec.candidate_id)
        bundles.append(
            OperationalizationBundle(
                operationalization_id=operationalization_id,
                decisions=_decisions(spec),
            )
        )
        branches.append(
            OperationalizationFindingBranch(
                operationalization_id=operationalization_id,
                findings=_findings(
                    metadata,
                    results[spec.candidate_id],
                    operationalization_id,
                ),
            )
        )
    return GroundTruth(
        dataset_id=metadata.dataset_id,
        case_id=metadata.case_id,
        case_family_id=metadata.case_family_id,
        acceptable_operationalizations=bundles,
        findings_by_operationalization=branches,
    )


def _result_lines(result: dict[str, Any]) -> list[str]:
    status = result["status"]
    status_label = {
        "provisionally_accepted": "provisionally accepted",
        "accepted": "accepted",
        "rejected": "rejected",
    }.get(status, status)
    return [
        f"- Candidate: `{result['candidate_id']}` ({status_label})",
        f"- Criterion: `{result['criterion']}`; retained points: `{result['retained_points']}`",
        f"- Regions: `{result['regions']}`; sizes: `{result['region_sizes']}`",
        f"- Selected region size: `{result['selected_region_size']}`",
        f"- Mean spatial location: `{result['mean_spatial_location']}`",
        f"- Reported location ({result['location_kind']}): `{result['reported_location']}`",
        f"- Candidate strength: `{result['strength']}`",
        f"- Selected-region peak speed: `{result['peak_speed']}`",
        f"- Selected-region mean speed: `{result['mean_speed']}`",
        f"- Peak point: `{result['peak_point']}`",
        f"- Measure kind: `{result['measure_kind']}`",
    ]


def _write_reference_analysis(
    path: Path,
    question: str,
    metadata: CaseConstructionMetadata,
    results: dict[str, dict[str, Any]],
) -> None:
    accepted = _accepted_specs_for_case(metadata.case_id)
    baseline = results[accepted[0].candidate_id]
    baseline_spec = accepted[0]
    lines = [
        f"# Office {metadata.case_id} Reference Analysis",
        "",
        "This construction artifact records executed candidate analyses and curator adjudication.",
        "It is not model-facing input and does not add a tool or scientific-operation schema.",
        "",
        "## Authored question",
        "",
        question,
        "",
        "## Reference execution baseline",
        "",
        "The Office legacy VTK file was read with `vtkDataSetReader` and the existing canonical",
        "read-all-scalars/read-all-vectors configuration. Mean locations are arithmetic means",
        "of retained locations, while peak-point candidates report the location of the selected",
        "region's peak-speed location. Candidate",
        "analyses below use only the supplied velocity field and face-connected structured-grid",
        "regions; no units are inferred for the stored velocity scale.",
        "",
        "The provisionally accepted O space is the smallest candidate set judged defensible for this pilot.",
        "Every provisionally accepted candidate has an independent G(O) branch in ground_truth.json.",
        "",
        "## Baseline candidate result",
        "",
        *_result_lines(baseline),
        "",
    ]
    if metadata.case_id in {"office_speed_zones_o2_f1", "office_speed_zones_o3_f1"}:
        lines.extend(["## Candidate-O exploration and adjudication", ""])
        for spec in _candidate_specs_for_case(metadata.case_id):
            result = results[spec.candidate_id]
            changed_dimensions = [
                dimension
                for dimension, left, right in (
                    ("feature_definition", spec.feature_statement, baseline_spec.feature_statement),
                    ("criterion", spec.criterion_statement, baseline_spec.criterion_statement),
                    ("property_measure", spec.measure_statement, baseline_spec.measure_statement),
                    ("aggregation_or_representation", spec.aggregation_statement, baseline_spec.aggregation_statement),
                )
                if left != right
            ]
            lines.extend(
                [
                    f"### `{spec.candidate_id}`",
                    "",
                    f"- Dimensions differing from baseline candidate: `{changed_dimensions or ['none']}`",
                    f"- Feature: {spec.feature_statement}",
                    f"- Criterion: {spec.criterion_statement}",
                    f"- Measure: {spec.measure_statement}",
                    f"- Representation: {spec.aggregation_statement}",
                    f"- Adjudication: **{spec.status.upper()}** — {spec.rationale}",
                    *_result_lines(result),
                    "",
                ]
            )
    if metadata.case_id == "office_speed_zones_o1_f2":
        lines.extend(
            [
                "## F2 finding adequacy review",
                "",
                "The narrowed open finding goal is answered by a minimum core consisting of:",
                "(1) the strongest region's mean spatial location, (2) its prominence/strength,",
                "and (3) its coordinate-space spatial span. The selected-region point count and",
                "total thresholded-region count are retained only as supporting construction",
                "statistics because they depend on grid sampling and do not define physical extent.",
                "",
            ]
        )
    if metadata.case_id == "office_speed_zones_o2_f1":
        peak = results["o2_peak_speed"]
        mean = results["o2_mean_speed"]
        q90 = results["o2_q90_mean_speed"]
        lines.extend(
            [
                "## Unresolved criterion and measure sensitivity",
                "",
                "The unresolved `property_measure` is consequential on this dataset: changing",
                "from peak speed to mean speed changes both the selected strongest region and",
                f"the resulting reference strength/location (peak: {peak['strength']:.10f} at "
                f"{peak['reported_location']}; mean: {mean['strength']:.10f} at "
                f"{mean['reported_location']}).",
                "The unresolved `criterion` was also genuinely explored. The distribution-informed",
                "non-zero 90th-percentile criterion retains a different region and is provisionally",
                f"retained for comparison (strength: {q90['strength']:.10f} at {q90['reported_location']}).",
                "Both the fixed 0.20 and q90 criteria remain subject to the same pending Scientific",
                "Grounding Gate; q90 is not rejected merely because grounding is pending.",
                "",
            ]
        )
    if metadata.case_id == "office_speed_zones_o3_f1":
        lines.extend(
            [
                "## Full-openness breadth review",
                "",
                "The exploration includes complete candidates that vary criterion, property measure,",
                "and location representation, plus a feature-definition candidate based on isolated",
                "local maxima. The local-peak feature was executed and rejected because it does not",
                "preserve the family target of a contiguous airflow region. Accepted candidates are",
                "therefore provisional and remain the smallest defensible space found in this pilot;",
                "they are not a pre-frozen O3 registry.",
                "",
            ]
        )
    lines.extend(
        [
            "The numerical results are reference-analysis outputs from the supplied flow data.",
            "They do not establish publication-level scientific grounding for the candidate",
            "speed criteria or the connected-region interpretation; that release gate remains pending.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_pilot_review(
    path: Path,
    metadata: CaseConstructionMetadata,
    results: dict[str, dict[str, Any]],
) -> None:
    case_type = primary_case_type(metadata)
    accepted_count = len(_accepted_specs_for_case(metadata.case_id))
    candidate_count = len(_candidate_specs_for_case(metadata.case_id))
    if case_type == "O1-F1":
        semantic_status = "O1 semantic completeness: PASS"
        adjudication_status = "Reference analysis baseline: PASS"
    elif case_type == "O2-F1":
        semantic_status = "Two-dimensional partial openness: IMPLEMENTED (criterion + property measure)"
        adjudication_status = (
            "Criterion exploration: PROVISIONAL; property-measure exploration: PASS; "
            "reference-space adjudication: PROVISIONAL"
        )
    elif case_type == "O3-F1":
        semantic_status = "Full principal openness: PASS"
        adjudication_status = (
            "Reference-space breadth: PROVISIONAL (criterion, representation, and "
            "feature candidate explored; local-peak feature rejected)"
        )
    else:
        semantic_status = "F2 reference adequacy: PROVISIONAL (three-finding core; two supporting statistics)"
        adjudication_status = "Reference analysis baseline: PASS"
    path.write_text(
        f"""# Office {metadata.case_id} Pilot Review

Primary type: `{case_type}`

## Pipeline result

- Natural scientific question, metadata, and context selection: **PASS**.
- Structural metadata and matched-family validation: **PASS**.
- Dataset Context, Case Context, and model-facing case input: **PASS**.
- Candidate analyses executed: `{candidate_count}`; provisionally accepted operationalizations in GT: `{accepted_count}`.
- Ground Truth branches validate one-to-one with accepted operationalizations: **PASS**.

## Case-specific review

- {semantic_status}
- {adjudication_status}

## Scientific release status

The Office evidence currently documents the official velocity/streamline example but does not
directly ground the authored candidate speed criteria or connected-region interpretation. No velocity
unit is inferred, and `finding_evidence` remains empty. Scientific Grounding is therefore
**PENDING**, and Formal Release remains **HOLD / CONDITIONAL**.

The operationalizations listed in Ground Truth are **pilot-adjudicated / provisional** until
the Scientific Grounding Gate passes; the Ground Truth schema intentionally carries no status
field for this construction state.

This pilot review records construction evidence and curator adjudication; it does not claim
that structural validators prove scientific semantic validity.
""",
        encoding="utf-8",
    )


def build_case(case_id: str, *, dataset_builder: Callable[..., Any] | None = None) -> None:
    """Build one Office pilot case using a caller-supplied context builder.

    The analysis library deliberately does not import ``scripts``.  The CLI
    supplies the repository's construction builder at the boundary.
    """
    if dataset_builder is None:
        raise RuntimeError("dataset_builder is required at the CLI boundary")
    dataset_dir = ROOT / "datasets" / "Office"
    case_dir = dataset_dir / "construction" / "cases" / case_id
    question = _read_json(case_dir / "scientific_question.json")["scientific_question"]
    metadata = CaseConstructionMetadata.model_validate(
        _read_json(case_dir / "case_construction_metadata.json")
    )
    dataset_builder(dataset_dir, case_id=case_id, require_formal_case=True)
    data = _load_office_data(dataset_dir)
    results = {
        spec.candidate_id: _analyze_candidate(data, spec)
        for spec in _candidate_specs_for_case(case_id)
    }
    _write_reference_analysis(case_dir / "reference_analysis.md", question, metadata, results)
    save_ground_truth(
        _ground_truth(metadata, results),
        case_dir / "ground_truth.json",
        case_metadata=metadata,
    )
    _write_pilot_review(case_dir / "pilot_review.md", metadata, results)
    baseline = _accepted_specs_for_case(case_id)[0]
    print(
        f"built {case_id}: candidates={len(results)}, "
        f"provisionally_accepted={len(_accepted_specs_for_case(case_id))}, "
        f"baseline_candidate={baseline.candidate_id}, strength={results[baseline.candidate_id]['strength']}"
    )


# Public library names used by runtime and audit code.  The underscored
# aliases above remain available to the historical CLI/tests unchanged.
def load_office_data(dataset_dir: Path) -> OfficeData:
    return _load_office_data(Path(dataset_dir))


def analyze_candidate(data: OfficeData, spec: CandidateSpec) -> dict[str, Any]:
    return _analyze_candidate(data, spec)


def build_findings(
    metadata: CaseConstructionMetadata,
    result: dict[str, Any],
    operationalization_id: str,
) -> list[ReferenceFinding]:
    return _findings(metadata, result, operationalization_id)


def build_ground_truth(
    metadata: CaseConstructionMetadata,
    results: dict[str, dict[str, Any]],
) -> GroundTruth:
    return _ground_truth(metadata, results)


__all__ = [
    "CASE_IDS", "CandidateSpec", "OfficeData", "O1_PEAK", "O2_CANDIDATES",
    "O3_CANDIDATES", "_accepted_specs_for_case", "_analyze_candidate", "_read_json",
    "_candidate_specs_for_case", "_decisions", "_findings", "_ground_truth",
    "_load_office_data", "_operationalization_id", "_regions_for_mask",
    "_strength_reporting_significant_figures", "_write_pilot_review",
    "_write_reference_analysis", "build_case", "load_office_data",
    "analyze_candidate", "build_findings", "build_ground_truth",
]

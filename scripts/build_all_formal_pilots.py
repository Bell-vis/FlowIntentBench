"""Materialize primary formal-pilot cases for all seven VTK-family datasets.

The Office family keeps its dataset-specific construction in
``build_office_formal_pilot.py``.  This module handles the six other datasets
with authored, dataset-specific question/context specifications and a small
executed candidate space.  It is construction infrastructure only: every
accepted O is provisional until the dataset-specific Scientific Grounding
Gate is reviewed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (  # noqa: E402
    CaseConstructionMetadata,
    FindingImportance,
    FindingRequirementCategory,
    GroundTruth,
    OperationalizationBundle,
    OperationalizationDecision,
    OperationalizationDimension,
    OperationalizationFindingBranch,
    ReferenceFinding,
    VerificationSpec,
    save_ground_truth,
)
from flowintentbench.construction_tolerances import (  # noqa: E402
    absolute_tolerance_for_significant_figures,
)
from flowintentbench.vtk_reader import apply_legacy_vtk_reader_configuration  # noqa: E402
from scripts.build_context_construction import build_dataset  # noqa: E402


DATASET_IDS = (
    "Blunt_Fin",
    "Carotid",
    "Combustor",
    "FireFlow",
    "Kitchen",
    "NASA_LOx_Post",
    "Office",
)
NON_OFFICE_DATASETS = DATASET_IDS[:-1]
CASE_SUFFIXES = ("o1_f1", "o2_f1", "o3_f1", "o1_f2")
# Construction-specific scientific reporting precision. Pointwise peaks use
# five significant figures; regional mean and point-summed quantities use four.
# This is authored into each GT finding and never inferred by the evaluator.
STRENGTH_REPORTING_SIGNIFICANT_FIGURES_BY_MEASURE = {
    "peak": 5,
    "mean": 4,
    "integrated_excess": 4,
}


def _strength_reporting_significant_figures(measure_kind: str) -> int:
    try:
        return STRENGTH_REPORTING_SIGNIFICANT_FIGURES_BY_MEASURE[measure_kind]
    except KeyError as exc:  # pragma: no cover - candidate table is closed
        raise ValueError(f"unsupported measure kind: {measure_kind}") from exc
PRINCIPAL_DIMENSIONS = [
    OperationalizationDimension.FEATURE_DEFINITION,
    OperationalizationDimension.CRITERION,
    OperationalizationDimension.PROPERTY_MEASURE,
    OperationalizationDimension.AGGREGATION_OR_REPRESENTATION,
]


@dataclass(frozen=True)
class DatasetPilotConfig:
    dataset_id: str
    family_id: str
    setting: str
    vector_name: str
    grouping_phrase: str
    grouping_statement: str
    grouping_mode: str


@dataclass(frozen=True)
class CandidateSpec:
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


@dataclass
class FlowData:
    dataset: Any
    speed: np.ndarray
    valid: np.ndarray
    coordinates: np.ndarray
    dimensions: tuple[int, int, int] | None
    adjacency: list[list[int]] | None = None


CONFIGS = {
    "Blunt_Fin": DatasetPilotConfig(
        "Blunt_Fin", "blunt_fin_flow_regions", "the blunt-fin flow field", "Velocity",
        "treat locations connected through shared grid faces as one contiguous region",
        "Face-connected locations form one contiguous high-speed flow region.", "structured",
    ),
    "Carotid": DatasetPilotConfig(
        "Carotid", "carotid_flow_regions", "the carotid artery flow field", "vectors",
        "treat locations connected through shared grid faces as one contiguous region",
        "Face-connected locations form one contiguous high-speed flow region.", "structured",
    ),
    "Combustor": DatasetPilotConfig(
        "Combustor", "combustor_flow_regions", "the annular-combustor flow field", "Velocity",
        "treat locations connected through shared grid faces as one contiguous region",
        "Face-connected locations form one contiguous high-speed flow region.", "structured",
    ),
    "FireFlow": DatasetPilotConfig(
        "FireFlow", "fireflow_flow_regions", "the FireFlow room flow field", "uvw",
        "treat locations belonging to the same connected mesh neighborhood as one region",
        "Mesh-connected locations form one contiguous high-speed flow region.", "unstructured",
    ),
    "Kitchen": DatasetPilotConfig(
        "Kitchen", "kitchen_flow_regions", "the kitchen airflow field", "velocity",
        "treat locations connected through shared grid faces as one contiguous region",
        "Face-connected locations form one contiguous high-speed flow region.", "structured",
    ),
    "NASA_LOx_Post": DatasetPilotConfig(
        "NASA_LOx_Post", "lox_post_flow_regions", "the LOx-post flow field", "Velocity",
        "treat locations connected through shared grid faces as one contiguous region",
        "Face-connected locations form one contiguous high-speed flow region.", "structured",
    ),
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_flow(config: DatasetPilotConfig) -> FlowData:
    from vtk.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOParallel import vtkMultiBlockPLOT3DReader
    from vtkmodules.vtkIOLegacy import vtkDataSetReader
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    dataset_dir = ROOT / "datasets" / config.dataset_id
    manifest = _read_json(dataset_dir / "dataset_manifest.json")
    fmt = manifest["reader"]["format"]
    if fmt == "VTK":
        reader = vtkDataSetReader()
        apply_legacy_vtk_reader_configuration(reader, manifest["reader"]["reader_configuration"])
        flow_file = next(item for item in manifest["files"] if item["role"] == "flow_field")
        reader.SetFileName(str(ROOT / manifest["file_root"] / flow_file["path"]))
        reader.Update()
        dataset = reader.GetOutput()
    elif fmt == "VTU":
        reader = vtkXMLUnstructuredGridReader()
        flow_file = next(item for item in manifest["files"] if item["role"] == "flow_field")
        reader.SetFileName(str(ROOT / manifest["file_root"] / flow_file["path"]))
        reader.Update()
        dataset = reader.GetOutput()
    elif fmt == "PLOT3D":
        reader = vtkMultiBlockPLOT3DReader()
        configuration = manifest["reader"]["reader_configuration"]
        grid_file = next(item for item in manifest["files"] if item["role"] == "grid")
        solution_file = next(item for item in manifest["files"] if item["role"] == "solution")
        reader.SetXYZFileName(str(ROOT / manifest["file_root"] / grid_file["path"]))
        reader.SetQFileName(str(ROOT / manifest["file_root"] / solution_file["path"]))
        reader.SetAutoDetectFormat(bool(configuration.get("auto_detect_format", True)))
        reader.SetForceRead(True)
        details = _read_json(dataset_dir / "data_metadata.json")["format"]["details"]
        reader.SetBinaryFile(details.get("encoding") == "binary")
        if details.get("endianness") == "big":
            reader.SetByteOrderToBigEndian()
        else:
            reader.SetByteOrderToLittleEndian()
        reader.SetHasByteCount(bool(details.get("has_byte_count", False)))
        reader.SetIBlanking(bool(manifest["reader"].get("grid_blanking")))
        reader.SetScalarFunctionNumber(int(configuration["scalar_function_number"]))
        reader.SetVectorFunctionNumber(int(configuration["vector_function_number"]))
        reader.Update()
        output = reader.GetOutput()
        dataset = output.GetBlock(0) if output is not None else None
    else:  # pragma: no cover - manifest formats are closed by the dataset bundle
        raise ValueError(f"unsupported dataset format: {fmt}")

    if dataset is None:
        raise RuntimeError(f"{config.dataset_id} reader returned no dataset")
    point_data = dataset.GetPointData()
    vectors = point_data.GetArray(config.vector_name) or point_data.GetVectors()
    if vectors is None:
        vectors = next(
            point_data.GetArray(index)
            for index in range(point_data.GetNumberOfArrays())
            if point_data.GetArray(index).GetNumberOfComponents() == 3
        )
    vector_values = vtk_to_numpy(vectors).astype(np.float64)
    speed = np.linalg.norm(vector_values, axis=1)
    valid = np.isfinite(speed)
    iblank = point_data.GetArray("IBlank")
    if iblank is not None:
        valid &= vtk_to_numpy(iblank) > 0
    coordinates = np.asarray(
        [dataset.GetPoint(index) for index in range(dataset.GetNumberOfPoints())],
        dtype=np.float64,
    )
    dimensions = [0, 0, 0]
    if config.grouping_mode == "structured":
        dataset.GetDimensions(dimensions)
        dimensions_tuple = tuple(int(value) for value in dimensions)
        if int(np.prod(dimensions_tuple)) != len(speed):
            raise RuntimeError(f"{config.dataset_id} dimensions do not match point count")
    else:
        dimensions_tuple = None
    return FlowData(dataset, speed, valid, coordinates, dimensions_tuple)


def _ensure_adjacency(data: FlowData) -> list[list[int]]:
    if data.adjacency is not None:
        return data.adjacency
    import vtk

    adjacency = [[] for _ in range(len(data.speed))]
    point_ids = vtk.vtkIdList()
    for cell_index in range(data.dataset.GetNumberOfCells()):
        data.dataset.GetCellPoints(cell_index, point_ids)
        points = [point_ids.GetId(index) for index in range(point_ids.GetNumberOfIds())]
        for left_index, left in enumerate(points):
            for right in points[left_index + 1 :]:
                adjacency[left].append(right)
                adjacency[right].append(left)
    data.adjacency = adjacency
    return adjacency


def _regions_for_mask(data: FlowData, mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(mask.size, dtype=bool)
    regions: list[np.ndarray] = []
    if data.dimensions is not None:
        nx, ny, nz = data.dimensions

        def point_id(x: int, y: int, z: int) -> int:
            return x + nx * (y + ny * z)

        def neighbors(current: int):
            z, remainder = divmod(current, nx * ny)
            y, x = divmod(remainder, nx)
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                xx, yy, zz = x + dx, y + dy, z + dz
                if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                    yield point_id(xx, yy, zz)
    else:
        adjacency = _ensure_adjacency(data)

        def neighbors(current: int):
            return adjacency[current]

    for start in np.flatnonzero(mask):
        start = int(start)
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        region: list[int] = []
        while stack:
            current = stack.pop()
            region.append(current)
            for neighbor in neighbors(current):
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        regions.append(np.asarray(region, dtype=int))
    return regions


def _candidate_specs(config: DatasetPilotConfig, case_suffix: str) -> tuple[CandidateSpec, ...]:
    feature = config.grouping_statement
    mean_representation = "Represent the selected region by the arithmetic mean spatial location of its retained high-speed locations."
    peak_representation = "Represent the selected region by the spatial location of its peak-speed location."
    speed_values = (
        "speed values computed as the Euclidean magnitude of the stored point-data "
        "vector array named `vectors`"
        if config.dataset_id == "Carotid"
        else "recorded speed values"
    )
    q90 = f"Retain locations at or above the 90th percentile of non-zero {speed_values}."
    q75 = f"Retain locations at or above the 75th percentile of non-zero {speed_values}."
    peak = "Use peak speed as the strength of each retained region."
    mean = "Use mean speed across retained locations as the strength of each retained region."
    integrated = "Use the point-summed speed excess above the selected criterion as region strength."
    if case_suffix == "o1_f1" or case_suffix == "o1_f2":
        suffix = "o1_high_speed_regions_open" if case_suffix == "o1_f2" else "o1_high_speed_regions"
        return (CandidateSpec(suffix, "O1", "provisionally_accepted", "q90", "peak", feature, q90, peak, mean_representation, "The authored O1 specification fixes this complete operationalization."),)
    if case_suffix == "o2_f1":
        return (
            CandidateSpec("o2_q90_peak", "O2", "provisionally_accepted", "q90", "peak", feature, q90, peak, mean_representation, "Provisional: fixed distribution-informed criterion with peak strength."),
            CandidateSpec("o2_q90_mean", "O2", "provisionally_accepted", "q90", "mean", feature, q90, mean, mean_representation, "Provisional: same criterion with sustained regional strength."),
            CandidateSpec("o2_q75_mean", "O2", "provisionally_accepted", "q75", "mean", feature, q75, mean, mean_representation, "Provisional: a distinct distribution-informed criterion/measure bundle; grounding remains pending."),
            CandidateSpec("o2_q90_integrated_excess", "O2", "rejected", "q90", "integrated_excess", feature, q90, integrated, mean_representation, "Rejected: point-summed strength is sampling-density dependent without a cell-volume contract."),
        )
    if case_suffix == "o3_f1":
        local_feature = "Treat isolated local speed maxima above the selected cutoff as point features rather than connected flow regions."
        local_criterion = "Retain speed locations above the selected cutoff that are strictly greater than their mesh neighbors."
        return (
            CandidateSpec("o3_q90_peak", "O3", "provisionally_accepted", "q90", "peak", feature, q90, peak, mean_representation, "Provisional complete O with distribution-informed criterion and peak strength."),
            CandidateSpec("o3_q90_mean", "O3", "provisionally_accepted", "q90", "mean", feature, q90, mean, mean_representation, "Provisional complete O with sustained regional strength."),
            CandidateSpec("o3_q75_mean", "O3", "provisionally_accepted", "q75", "mean", feature, q75, mean, mean_representation, "Provisional complete O with a distinct criterion and measure."),
            CandidateSpec("o3_q90_peak_point", "O3", "provisionally_accepted", "q90", "peak", feature, q90, peak, peak_representation, "Provisional complete O with peak-point representation."),
            CandidateSpec("o3_local_peak_features", "O3", "rejected", "local_peak", "peak", local_feature, local_criterion, peak, "Represent each feature by the spatial location of its local maximum.", "Rejected: isolated point features do not satisfy the question's target entity of a flow region."),
        )
    raise ValueError(f"unsupported case suffix: {case_suffix}")


def _threshold(data: FlowData, kind: str) -> tuple[float, str]:
    values = data.speed[data.valid & (data.speed > 0)]
    if kind == "q90":
        return float(np.quantile(values, 0.90)), "90th percentile"
    if kind == "q75":
        return float(np.quantile(values, 0.75)), "75th percentile"
    if kind == "local_peak":
        return 0.0, "selected cutoff"
    raise ValueError(f"unsupported criterion: {kind}")


def _analyze(data: FlowData, spec: CandidateSpec) -> dict[str, Any]:
    threshold, threshold_label = _threshold(data, spec.criterion_kind)
    mask = data.valid & (data.speed >= threshold)
    if spec.criterion_kind == "local_peak":
        adjacency = _ensure_adjacency(data)
        local = np.zeros(mask.size, dtype=bool)
        for point in np.flatnonzero(mask):
            point = int(point)
            neighbors = adjacency[point] if data.dimensions is None else list(_structured_neighbors(data, point))
            if all(data.speed[point] > data.speed[neighbor] for neighbor in neighbors):
                local[point] = True
        mask = local
    regions = _regions_for_mask(data, mask)
    if not regions:
        raise RuntimeError(f"candidate {spec.candidate_id} retained no region")
    if spec.measure_kind == "peak":
        values = np.asarray([data.speed[region].max() for region in regions])
    elif spec.measure_kind == "mean":
        values = np.asarray([data.speed[region].mean() for region in regions])
    elif spec.measure_kind == "integrated_excess":
        values = np.asarray([np.maximum(data.speed[region] - threshold, 0).sum() for region in regions])
    else:  # pragma: no cover
        raise ValueError(f"unsupported measure: {spec.measure_kind}")
    selected = regions[int(np.argmax(values))]
    selected_peak_index = int(selected[np.argmax(data.speed[selected])])
    mean_location = data.coordinates[selected].mean(axis=0).tolist()
    peak_location = data.coordinates[selected_peak_index].tolist()
    reported_location = peak_location if "peak-speed location" in spec.aggregation_statement or "local maximum" in spec.aggregation_statement else mean_location
    return {
        "candidate_id": spec.candidate_id,
        "status": spec.status,
        "measure_kind": spec.measure_kind,
        "criterion": f">= {threshold_label} ({threshold:.12f})",
        "threshold": threshold,
        "retained_points": int(mask.sum()),
        "regions": len(regions),
        "region_sizes": [int(len(region)) for region in regions],
        "selected_region_size": int(len(selected)),
        "mean_spatial_location": mean_location,
        "reported_location": reported_location,
        "location_kind": "peak_speed_location" if reported_location == peak_location and reported_location != mean_location else "mean_spatial_location",
        "coordinate_span": (data.coordinates[selected].max(axis=0) - data.coordinates[selected].min(axis=0)).tolist(),
        "strength": float(values[np.argmax(values)]),
        "peak_speed": float(data.speed[selected].max()),
        "mean_speed": float(data.speed[selected].mean()),
    }


def _structured_neighbors(data: FlowData, point: int):
    assert data.dimensions is not None
    nx, ny, nz = data.dimensions
    z, remainder = divmod(point, nx * ny)
    y, x = divmod(remainder, nx)
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        xx, yy, zz = x + dx, y + dy, z + dz
        if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
            yield xx + nx * (yy + ny * zz)


def _metadata(config: DatasetPilotConfig, case_id: str, suffix: str) -> tuple[CaseConstructionMetadata, str]:
    setting_fragment = f"in {config.setting}"
    selection = {"statement": "select the strongest region", "question_fragment": "strongest"}
    finding_goal = f"characterize the strongest high-speed flow region in {config.setting}"
    target = f"high-speed flow regions in {config.setting}"
    if suffix in {"o1_f1", "o1_f2"}:
        question_fragment = (
            "compute speed as the Euclidean magnitude of the stored point-data vector array named `vectors`, "
            "then define high-speed locations as those at or above the 90th percentile of its non-zero values"
            if config.dataset_id == "Carotid"
            else (
                "define high-speed locations as those at or above the 90th percentile of non-zero recorded speed "
                "in the stored velocity scale"
            )
        )
        if suffix == "o1_f1":
            question = (
                f"Which high-speed flow region is strongest in {config.setting}, and where is it located? For this analysis, "
                f"{question_fragment}, {config.grouping_phrase}, use peak speed as the measure of strength, and represent "
                "the strongest region by the mean spatial location of its retained high-speed locations. "
                "Identify the strongest region and report its location and strength."
            )
        else:
            question = (
                f"Which high-speed flow region is strongest in {config.setting}? For this analysis, "
                f"{question_fragment}, {config.grouping_phrase}, use peak speed as the measure of strength, and represent "
                "the strongest region by the mean spatial location of its retained high-speed locations. "
                "Characterize the strongest high-speed flow region."
            )
        constraints = [
            {"category": "feature_definition", "statement": config.grouping_statement, "question_fragment": config.grouping_phrase},
            {"category": "criterion", "statement": question_fragment, "question_fragment": question_fragment},
            {"category": "property_measure", "statement": "use peak speed as the measure of strength", "question_fragment": "use peak speed as the measure of strength"},
            {"category": "analysis_procedure", "statement": "represent the strongest region by the mean spatial location of its retained high-speed locations", "question_fragment": "represent the strongest region by the mean spatial location of its retained high-speed locations"},
        ]
        unresolved: list[str] = []
        responsibility = "user_specified"
        openness = "bounded" if suffix == "o1_f1" else "open"
    elif suffix == "o2_f1":
        feature_fragment = config.grouping_phrase.capitalize()
        representation_fragment = "represent the selected region by the mean spatial location of its high-speed locations"
        question = (
            f"Which high-speed flow region is strongest in {config.setting}, and where is it located? Identify contiguous "
            f"high-speed regions using an appropriate speed-based criterion and use an appropriate measure to compare their "
            f"strength. {feature_fragment}, represent the selected region by the mean spatial location of its high-speed "
            "locations, and report its location and strength."
        )
        constraints = [
            {"category": "feature_definition", "statement": config.grouping_statement, "question_fragment": feature_fragment},
            {"category": "analysis_procedure", "statement": representation_fragment, "question_fragment": representation_fragment},
        ]
        unresolved = ["criterion", "property_measure"]
        responsibility = "partially_specified"
        openness = "bounded"
    else:
        question = (
            f"Which flow region in {config.setting} is strongest, and where is it located? Identify and characterize the "
            "strongest region using an appropriate scientific analysis, and report its location and strength."
        )
        constraints = []
        unresolved = [item.value for item in PRINCIPAL_DIMENSIONS]
        responsibility = "model_selected"
        openness = "bounded"
    requirements = [] if openness == "open" else [
        {"category": "location", "statement": "report the location of the strongest region", "question_fragment": "report its location"},
        {"category": "quantity", "statement": "report the strength of the strongest region", "question_fragment": "strength"},
    ]
    payload = {
        "case_id": case_id,
        "case_family_id": config.family_id,
        "dataset_id": config.dataset_id,
        "scientific_target": target,
        "finding_goal": finding_goal,
        "operationalization_responsibility": responsibility,
        "finding_openness": openness,
        "principal_operationalization_dimensions": [item.value for item in PRINCIPAL_DIMENSIONS],
        "unresolved_operationalization_dimensions": unresolved,
        "scope_constraints": [{"statement": setting_fragment, "question_fragment": setting_fragment}],
        "condition_constraints": [],
        "selection_constraints": [selection],
        "explicit_method_constraints": constraints,
        "explicit_finding_requirements": requirements,
    }
    return CaseConstructionMetadata.model_validate(payload), question


def _findings(metadata: CaseConstructionMetadata, result: dict[str, Any], op_id: str) -> list[ReferenceFinding]:
    location = [float(value) for value in result["reported_location"]]
    strength = float(result["strength"])
    reporting_significant_figures = _strength_reporting_significant_figures(
        result["measure_kind"]
    )
    strength_tolerance = absolute_tolerance_for_significant_figures(
        strength,
        significant_figures=reporting_significant_figures,
    )
    label = "peak-speed location" if result["location_kind"] == "peak_speed_location" else "mean spatial location"
    measure = {
        "peak": "peak speed",
        "mean": "mean speed",
        "integrated_excess": "point-summed speed excess",
    }[result["measure_kind"]]
    findings = [
        ReferenceFinding(
            finding_id=f"{op_id}_location",
            category=FindingRequirementCategory.LOCATION,
            statement=f"The selected strongest region has {label} approximately {location} in the dataset coordinate frame.",
            importance=FindingImportance.CORE,
            value=location,
            verification=VerificationSpec(spatial_tolerance=0.1),
        ),
        ReferenceFinding(
            finding_id=f"{op_id}_strength",
            category=FindingRequirementCategory.QUANTITY,
            statement=(
                f"The selected strongest region has {measure} strength approximately "
                f"{strength:.{reporting_significant_figures}g} in the stored velocity scale."
            ),
            importance=FindingImportance.CORE,
            value=strength,
            verification=VerificationSpec(absolute_tolerance=strength_tolerance),
        ),
    ]
    if metadata.finding_openness == "open":
        findings.append(ReferenceFinding(
            finding_id=f"{op_id}_coordinate_span",
            category=FindingRequirementCategory.CHARACTERIZATION,
            statement=f"The strongest region has coordinate-space span approximately {result['coordinate_span']} along the dataset axes; no physical unit is inferred.",
            importance=FindingImportance.CORE,
            value=[float(value) for value in result["coordinate_span"]],
            verification=VerificationSpec(spatial_tolerance=0.1),
        ))
        findings.extend([
            ReferenceFinding(finding_id=f"{op_id}_point_count", category=FindingRequirementCategory.QUANTITY, statement=f"The selected region contains {result['selected_region_size']} retained locations.", importance=FindingImportance.SUPPORTING, value=int(result["selected_region_size"])),
            ReferenceFinding(finding_id=f"{op_id}_region_count", category=FindingRequirementCategory.QUANTITY, statement=f"The criterion produces {result['regions']} connected regions in the supplied mesh or grid.", importance=FindingImportance.SUPPORTING, value=int(result["regions"])),
        ])
    return findings


def _ground_truth(metadata: CaseConstructionMetadata, specs: tuple[CandidateSpec, ...], results: dict[str, dict[str, Any]]) -> GroundTruth:
    accepted = [spec for spec in specs if spec.status == "provisionally_accepted"]
    bundles = []
    branches = []
    for spec in accepted:
        op_id = f"{metadata.case_id}_{spec.candidate_id}"
        decisions = [
            OperationalizationDecision(dimension=OperationalizationDimension.FEATURE_DEFINITION, statement=spec.feature_statement),
            OperationalizationDecision(dimension=OperationalizationDimension.CRITERION, statement=spec.criterion_statement),
            OperationalizationDecision(dimension=OperationalizationDimension.PROPERTY_MEASURE, statement=spec.measure_statement),
            OperationalizationDecision(dimension=OperationalizationDimension.AGGREGATION_OR_REPRESENTATION, statement=spec.aggregation_statement),
        ]
        bundles.append(OperationalizationBundle(operationalization_id=op_id, decisions=decisions))
        branches.append(OperationalizationFindingBranch(operationalization_id=op_id, findings=_findings(metadata, results[spec.candidate_id], op_id)))
    return GroundTruth(
        dataset_id=metadata.dataset_id,
        case_id=metadata.case_id,
        case_family_id=metadata.case_family_id,
        acceptable_operationalizations=bundles,
        findings_by_operationalization=branches,
    )


def _result_lines(result: dict[str, Any]) -> list[str]:
    return [
        f"- Candidate: `{result['candidate_id']}` ({result['status']})",
        f"- Criterion: `{result['criterion']}`; retained points: `{result['retained_points']}`",
        f"- Connected regions: `{result['regions']}`; sizes: `{result['region_sizes']}`",
        f"- Selected region size: `{result['selected_region_size']}`",
        f"- Reported location ({result['location_kind']}): `{result['reported_location']}`",
        f"- Measure kind: `{result['measure_kind']}`",
        f"- Strength: `{result['strength']}`",
    ]


def _write_reference_analysis(path: Path, config: DatasetPilotConfig, metadata: CaseConstructionMetadata, question: str, specs: tuple[CandidateSpec, ...], results: dict[str, dict[str, Any]]) -> None:
    baseline = specs[0]
    lines = [
        f"# {config.dataset_id} {metadata.case_id} Reference Analysis",
        "",
        "This is a dataset-specific construction artifact, not model-facing input.",
        "All candidates were executed on the supplied reader-visible flow field; no physical units are inferred.",
        "",
        "## Authored question",
        "",
        question,
        "",
        "## Candidate-O exploration and adjudication",
        "",
    ]
    for spec in specs:
        result = results[spec.candidate_id]
        changes = []
        for name, left, right in (("feature_definition", spec.feature_statement, baseline.feature_statement), ("criterion", spec.criterion_statement, baseline.criterion_statement), ("property_measure", spec.measure_statement, baseline.measure_statement), ("aggregation_or_representation", spec.aggregation_statement, baseline.aggregation_statement)):
            if left != right:
                changes.append(name)
        lines.extend([
            f"### `{spec.candidate_id}`",
            "",
            f"- Dimensions differing from baseline candidate: `{changes or ['none']}`",
            f"- Feature: {spec.feature_statement}",
            f"- Criterion: {spec.criterion_statement}",
            f"- Measure: {spec.measure_statement}",
            f"- Representation: {spec.aggregation_statement}",
            f"- Adjudication: **{spec.status.upper()}** — {spec.rationale}",
            *_result_lines(result),
            "",
        ])
    if metadata.case_id.endswith("o2_f1"):
        lines.extend(["## O2 unresolved-dimension review", "", "Both criterion and property measure were explored using complete candidate bundles; no Cartesian product was generated. The q90 and q75 criteria remain provisional under the same grounding standard.", ""])
    if metadata.case_id.endswith("o3_f1"):
        lines.extend(["## O3 breadth review", "", "Criterion, measure, representation and feature-definition candidates were considered. The isolated local-peak candidate was rejected because it represents point features rather than the flow region requested by the question.", ""])
    lines.extend(["Scientific Grounding for this dataset-specific operationalization remains PENDING; Formal Release is HOLD / CONDITIONAL.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_pilot_review(path: Path, config: DatasetPilotConfig, metadata: CaseConstructionMetadata, specs: tuple[CandidateSpec, ...], results: dict[str, dict[str, Any]]) -> None:
    case_type = metadata.operationalization_responsibility.value + "/" + metadata.finding_openness.value
    path.write_text(
        f"""# {config.dataset_id} {metadata.case_id} Pilot Review

Primary type: `{case_type}`

## Pipeline result

- Authored scientific question and metadata: **PASS**.
- Dataset Context, Case Context and model-facing case input: **PASS**.
- Candidate analyses executed: `{len(specs)}`; provisionally accepted O: `{sum(spec.status == 'provisionally_accepted' for spec in specs)}`.
- Complete O-to-G(O) branches: **PASS**.

## Scientific status

This is a dataset-specific infrastructure pilot. Candidate adjudication is provisional and is not a final benchmark acceptance decision.
Scientific Grounding Gate: **PENDING**.
Formal Release: **HOLD / CONDITIONAL**.

The current sources and executed data analysis do not by themselves establish that the selected speed-region criterion is scientifically grounded for this dataset. No velocity unit is inferred, and no finding evidence is added automatically.
""",
        encoding="utf-8",
    )


def _materialize_case(config: DatasetPilotConfig, suffix: str) -> dict[str, Any]:
    case_id = f"{config.dataset_id.lower()}_{suffix}"
    case_dir = ROOT / "datasets" / config.dataset_id / "construction" / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    metadata, question = _metadata(config, case_id, suffix)
    initial_selection = _read_json(ROOT / "datasets" / config.dataset_id / "construction" / "cases" / "initial_case" / "context_selection.json")
    (case_dir / "scientific_question.json").write_text(json.dumps({"scientific_question": question}, indent=2) + "\n", encoding="utf-8")
    (case_dir / "case_construction_metadata.json").write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (case_dir / "context_selection.json").write_text(json.dumps({"case_id": case_id, "dataset_id": config.dataset_id, "include_fact_ids": initial_selection["include_fact_ids"]}, indent=2) + "\n", encoding="utf-8")
    build_dataset(ROOT / "datasets" / config.dataset_id, case_id=case_id, require_formal_case=True)
    data = _load_flow(config)
    specs = _candidate_specs(config, suffix)
    results = {spec.candidate_id: _analyze(data, spec) for spec in specs}
    ground_truth = _ground_truth(metadata, specs, results)
    save_ground_truth(ground_truth, case_dir / "ground_truth.json", case_metadata=metadata)
    _write_reference_analysis(case_dir / "reference_analysis.md", config, metadata, question, specs, results)
    _write_pilot_review(case_dir / "pilot_review.md", config, metadata, specs, results)
    return {
        "dataset_id": config.dataset_id,
        "case_id": case_id,
        "case_type": case_type_from_suffix(suffix),
        "scientific_question": question,
        "candidate_count": len(specs),
        "provisionally_accepted": sum(spec.status == "provisionally_accepted" for spec in specs),
        "accepted_operationalization_ids": [item.operationalization_id for item in ground_truth.acceptable_operationalizations],
        "finding_counts": {item.operationalization_id: len(item.findings) for item in ground_truth.findings_by_operationalization},
        "ground_truth": str(case_dir / "ground_truth.json"),
    }


def case_type_from_suffix(suffix: str) -> str:
    return {"o1_f1": "O1-F1", "o2_f1": "O2-F1", "o3_f1": "O3-F1", "o1_f2": "O1-F2"}[suffix]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_ids", nargs="*", choices=DATASET_IDS, default=None)
    args = parser.parse_args()
    from scripts.build_office_formal_pilot import (
        CASE_IDS as OFFICE_CASE_IDS,
        _accepted_specs_for_case,
        _candidate_specs_for_case,
        build_case as build_office_case,
    )

    selected = tuple(args.dataset_ids or DATASET_IDS)
    summaries: list[dict[str, Any]] = []
    for dataset_id in selected:
        if dataset_id == "Office":
            for case_id in OFFICE_CASE_IDS:
                build_office_case(case_id)
                case_dir = ROOT / "datasets" / "Office" / "construction" / "cases" / case_id
                ground_truth = _read_json(case_dir / "ground_truth.json")
                summaries.append(
                    {
                        "dataset_id": "Office",
                        "case_id": case_id,
                        "case_type": "Office refined family",
                        "scientific_question": _read_json(case_dir / "scientific_question.json")["scientific_question"],
                        "candidate_count": len(_candidate_specs_for_case(case_id)),
                        "provisionally_accepted": len(_accepted_specs_for_case(case_id)),
                        "accepted_operationalization_ids": [item["operationalization_id"] for item in ground_truth["acceptable_operationalizations"]],
                        "finding_counts": {item["operationalization_id"]: len(item["findings"]) for item in ground_truth["findings_by_operationalization"]},
                        "ground_truth": str(case_dir / "ground_truth.json"),
                    }
                )
            continue
        config = CONFIGS[dataset_id]
        for suffix in CASE_SUFFIXES:
            summary = _materialize_case(config, suffix)
            summaries.append(summary)
            print(f"built {summary['case_id']}: candidates={summary['candidate_count']}, provisionally_accepted={summary['provisionally_accepted']}")
    summary = {
        "datasets": selected,
        "cases": summaries,
        "scientific_grounding": "PENDING",
        "formal_release": "HOLD / CONDITIONAL",
    }
    (ROOT / "datasets" / "formal_pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Seven-dataset construction/full-pipeline pilot summary",
        "",
        "These artifacts are construction pilots. Scientific Grounding remains **PENDING** and Formal Release remains **HOLD / CONDITIONAL**.",
        "",
        "| Dataset | Case | Family | Candidates | Provisional O | Ground truth |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            f"| `{item['dataset_id']}` | `{item['case_id']}` | `{item['case_type']}` | "
            f"{item.get('candidate_count', 'n/a')} | {item.get('provisionally_accepted', 'n/a')} | "
            f"`{Path(item['ground_truth']).relative_to(ROOT)}` |"
        )
    (ROOT / "datasets" / "formal_pilot_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

"""Execute the Phase 1 Dataset -> Evidence -> Concept construction pilot.

This script deliberately materializes *construction-pilot* artifacts outside
the dataset release tree.  It does not invoke an evaluated model, mutate the
evaluator, or claim a formal benchmark release.  Candidate families are
authored from the reviewed variables and sources, then each accepted complete
operationalization is executed to produce its G(O) branch; screening status,
not code membership, controls readiness.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (  # noqa: E402
    BenchmarkCaseInput,
    CaseConstructionMetadata,
    CaseConstructionMetadataValidator,
    CaseContextSelector,
    ContextSelection,
    ControlledCaseFamilyValidator,
    DataFile,
    DatasetContextBuilder,
    DatasetManifest,
    EvidenceRecord,
    FindingImportance,
    FindingOpenness,
    FindingRequirementCategory,
    GroundTruth,
    GroundTruthValidator,
    OperationalizationBundle,
    OperationalizationDecision,
    OperationalizationDimension,
    OperationalizationFindingBranch,
    OperationalizationResponsibility,
    ReferenceFinding,
    VerificationSpec,
    load_source_collection,
    save_ground_truth,
)
from flowintentbench.construction_tolerances import (  # noqa: E402
    absolute_tolerance_for_significant_figures,
)
from flowintentbench.schema import DataMetadata, FlowData  # noqa: E402
from flowintentbench.case_design import normalize_case_text  # noqa: E402
from flowintentbench.vtk_reader import apply_legacy_vtk_reader_configuration  # noqa: E402


DATASET_IDS = (
    "Blunt_Fin",
    "Carotid",
    "Combustor",
    "FireFlow",
    "Kitchen",
    "NASA_LOx_Post",
    "Office",
)
PILOT_STATUS = "CONSTRUCTION PILOT / NOT FORMAL RELEASE"


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    dataset_id: str
    title: str
    concept: str
    evidence_ids: tuple[str, ...]
    principal_dimensions: tuple[OperationalizationDimension, ...]
    o2_unresolved: tuple[OperationalizationDimension, ...]
    eligible_o2_masks: tuple[tuple[OperationalizationDimension, ...], ...]
    scientific_coherence: str
    o_role_coherence: str
    ambiguity_quality: str
    grounding_assessment: str
    context_fact_ids: tuple[str, ...]
    analyze: Callable[[Any], dict[str, dict[str, Any]]]
    definition: dict[str, Any] | None = None


@dataclass(frozen=True)
class F1FindingProperty:
    """One authored scientific property shared by F1 text and F1 GT."""

    property_id: str
    category: FindingRequirementCategory
    request_phrase: str
    requirement_statement: str


def _f1_finding_properties(
    family: FamilySpec,
    *,
    o1_resolved: bool,
) -> tuple[F1FindingProperty, ...]:
    if family.definition is not None:
        return tuple(F1FindingProperty(
            item["property_id"], FindingRequirementCategory(item["category"]),
            item["request_phrase"], item["requirement_statement"],
        ) for item in family.definition["finding_properties"])
    if family.family_id == "kitchen_turbulence_activity":
        return (
            F1FindingProperty(
                "selected_hotspot_location",
                FindingRequirementCategory.LOCATION,
                "its coordinate",
                "report the selected hotspot coordinate",
            ),
            F1FindingProperty(
                "turbulent_kinetic_energy",
                FindingRequirementCategory.QUANTITY,
                "turbulent kinetic energy at the selected location",
                "report turbulent kinetic energy at the selected location",
            ),
            F1FindingProperty(
                "turbulent_dissipation_rate",
                FindingRequirementCategory.QUANTITY,
                "turbulent dissipation rate at the selected location",
                "report turbulent dissipation rate at the selected location",
            ),
        )
    if family.family_id == "kitchen_concentration_heterogeneity":
        measure_phrase = (
            "the cell-volume-weighted coefficient-of-variation value"
            if o1_resolved
            else "the numerical value of the heterogeneity measure used to select it"
        )
        return (
            F1FindingProperty(
                "selected_concentration_field",
                FindingRequirementCategory.EXISTENCE_OR_IDENTITY,
                "the selected field identifier",
                "report the selected concentration-field identifier",
            ),
            F1FindingProperty(
                "heterogeneity_value",
                FindingRequirementCategory.QUANTITY,
                measure_phrase,
                "report the selected heterogeneity value",
            ),
        )
    measure_phrase = "the stored density value" if o1_resolved else "the numerical value of the density criterion used to select it"
    return (
        F1FindingProperty(
            "selected_density_feature_location",
            FindingRequirementCategory.LOCATION,
            "its coordinate",
            "report the selected density-feature coordinate",
        ),
        F1FindingProperty(
            "density_feature_value",
            FindingRequirementCategory.QUANTITY,
            measure_phrase,
            "report the selected density-feature value",
        ),
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


CONSTRUCTION_FLOAT_DIGITS = 14


def _canonicalize_json(value: Any) -> Any:
    """Canonicalize persisted construction floats without changing decisions.

    Branch selection always happens on the raw computation.  This function is
    applied only when materializing artifacts, using 14 significant digits—far
    more precision than the five-digit scientific reporting convention used by
    the pilot—so harmless cross-platform last-bit differences do not alter
    artifact identity.
    """

    if isinstance(value, float):
        return float(format(value, f".{CONSTRUCTION_FLOAT_DIGITS}g"))
    if isinstance(value, dict):
        return {key: _canonicalize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize_json(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_canonicalize_json(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _records(construction_dir: Path) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for name in ("context_evidence.json", "operationalization_evidence.json", "finding_evidence.json"):
        records.extend(EvidenceRecord.model_validate(item) for item in _read_json(construction_dir / name))
    return records


def _flow_data(dataset_dir: Path) -> FlowData:
    metadata = DataMetadata.model_validate(_read_json(dataset_dir / "data_metadata.json"))
    manifest = DatasetManifest.model_validate(_read_json(dataset_dir / "dataset_manifest.json"))
    return FlowData(
        data_files=[DataFile(path=item.path, role=item.role) for item in manifest.files],
        data_metadata=metadata,
    )


def _load_dataset(dataset_dir: Path) -> Any:
    """Load only the reviewed reader-visible dataset with manifest settings."""

    manifest = _read_json(dataset_dir / "dataset_manifest.json")
    reader_config = manifest["reader"]["reader_configuration"]
    format_name = manifest["reader"]["format"]
    file_root = dataset_dir.parent if manifest["file_root"] == "datasets" else ROOT / manifest["file_root"]
    files = manifest["files"]
    if format_name == "VTK":
        from vtkmodules.vtkIOLegacy import vtkDataSetReader

        reader = vtkDataSetReader()
        apply_legacy_vtk_reader_configuration(reader, reader_config)
        flow = next(item for item in files if item["role"] == "flow_field")
        reader.SetFileName(str(file_root / flow["path"]))
        reader.Update()
        dataset = reader.GetOutput()
    elif format_name == "VTU":
        from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

        reader = vtkXMLUnstructuredGridReader()
        flow = next(item for item in files if item["role"] == "flow_field")
        reader.SetFileName(str(file_root / flow["path"]))
        reader.Update()
        dataset = reader.GetOutput()
    elif format_name == "PLOT3D":
        from vtkmodules.vtkIOParallel import vtkMultiBlockPLOT3DReader

        details = _read_json(dataset_dir / "data_metadata.json")["format"]["details"]
        grid = next(item for item in files if item["role"] == "grid")
        solution = next(item for item in files if item["role"] == "solution")
        reader = vtkMultiBlockPLOT3DReader()
        reader.SetXYZFileName(str(file_root / grid["path"]))
        reader.SetQFileName(str(file_root / solution["path"]))
        reader.SetAutoDetectFormat(bool(reader_config.get("auto_detect_format", True)))
        reader.SetForceRead(True)
        reader.SetBinaryFile(details.get("encoding") == "binary")
        if details.get("endianness") == "big":
            reader.SetByteOrderToBigEndian()
        else:
            reader.SetByteOrderToLittleEndian()
        reader.SetHasByteCount(bool(details.get("has_byte_count", False)))
        reader.SetIBlanking(bool(manifest["reader"].get("grid_blanking")))
        reader.SetScalarFunctionNumber(int(reader_config["scalar_function_number"]))
        reader.SetVectorFunctionNumber(int(reader_config["vector_function_number"]))
        reader.Update()
        output = reader.GetOutput()
        dataset = output.GetBlock(0) if output is not None else None
    else:
        raise ValueError(f"Phase 1 does not execute unsupported reader format {format_name!r}")
    if dataset is None or dataset.GetNumberOfPoints() == 0:
        raise RuntimeError(f"reader produced no points for {dataset_dir.name}")
    return dataset


def _coordinates(dataset: Any) -> np.ndarray:
    return np.asarray([dataset.GetPoint(index) for index in range(dataset.GetNumberOfPoints())], dtype=float)


def _point_scalars(dataset: Any, name: str) -> np.ndarray:
    from vtk.util.numpy_support import vtk_to_numpy

    array = dataset.GetPointData().GetArray(name)
    if array is None:
        raise RuntimeError(f"expected point scalar {name!r} is unavailable")
    return vtk_to_numpy(array).astype(float)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(np.interp(quantile, cumulative, ordered_values))


def _cell_volume_and_point_means(dataset: Any, scalar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use VTK cell volumes and explicit point-to-cell arithmetic means.

    The calculation is authored for this Kitchen structured-grid pilot; it is
    intentionally not a generic concentration-analysis framework.
    """

    from vtk.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    size_filter = vtkCellSizeFilter()
    size_filter.SetInputData(dataset)
    size_filter.SetComputeVolume(True)
    size_filter.Update()
    volumes = vtk_to_numpy(size_filter.GetOutput().GetCellData().GetArray("Volume")).astype(float)
    means = np.empty(dataset.GetNumberOfCells(), dtype=float)
    for cell_id in range(dataset.GetNumberOfCells()):
        point_ids = dataset.GetCell(cell_id).GetPointIds()
        means[cell_id] = float(np.mean([scalar[point_ids.GetId(index)] for index in range(point_ids.GetNumberOfIds())]))
    valid = np.isfinite(volumes) & (volumes > 0) & np.isfinite(means)
    return volumes[valid], means[valid]


def _analyze_kitchen_turbulence(dataset: Any) -> dict[str, dict[str, Any]]:
    coordinates = _coordinates(dataset)
    tke = _point_scalars(dataset, "ke")
    dissipation = _point_scalars(dataset, "ep")

    def extremum(field: np.ndarray, companion: np.ndarray, *, name: str) -> dict[str, Any]:
        valid = np.isfinite(field)
        index = int(np.flatnonzero(valid)[np.argmax(field[valid])])
        return {
            "operationalization_id": name,
            "feature_value": float(field[index]),
            "companion_value": float(companion[index]),
            "location": [float(value) for value in coordinates[index]],
            "point_index": index,
        }

    return {
        "maximum_turbulent_kinetic_energy": extremum(tke, dissipation, name="maximum_turbulent_kinetic_energy"),
        "maximum_turbulent_dissipation": extremum(dissipation, tke, name="maximum_turbulent_dissipation"),
    }


def _analyze_kitchen_concentrations(dataset: Any) -> dict[str, dict[str, Any]]:
    fields = ("c1", "c11", "c12", "c13", "c14", "c15", "c16", "c17")
    metrics: list[dict[str, Any]] = []
    for field_name in fields:
        volumes, values = _cell_volume_and_point_means(dataset, _point_scalars(dataset, field_name))
        mean = float(np.average(values, weights=volumes))
        standard_deviation = float(np.sqrt(np.average((values - mean) ** 2, weights=volumes)))
        q10 = _weighted_quantile(values, volumes, 0.10)
        q50 = _weighted_quantile(values, volumes, 0.50)
        q90 = _weighted_quantile(values, volumes, 0.90)
        metrics.append(
            {
                "field_name": field_name,
                "mean": mean,
                "standard_deviation": standard_deviation,
                "coefficient_of_variation": standard_deviation / max(abs(mean), 1e-30),
                # The weighted mean stays defined for the reviewed nonnegative
                # concentration fields.  Do not normalize by q50: several
                # legitimate fields have a zero median, which would create an
                # artificial, non-adjudicable blow-up.
                "relative_interdecile_spread": (q90 - q10) / max(abs(mean), 1e-30),
                "q10": q10,
                "q50": q50,
                "q90": q90,
                "cell_count": int(len(values)),
            }
        )
    return {
        "coefficient_of_variation": max(metrics, key=lambda item: item["coefficient_of_variation"]),
        "relative_interdecile_spread": max(metrics, key=lambda item: item["relative_interdecile_spread"]),
        "all_fields": {item["field_name"]: item for item in metrics},
    }


def _analyze_combustor_density(dataset: Any) -> dict[str, dict[str, Any]]:
    from vtk.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersGeneral import vtkGradientFilter

    coordinates = _coordinates(dataset)
    density = _point_scalars(dataset, "Density")
    max_density_index = int(np.nanargmax(density))
    gradient_filter = vtkGradientFilter()
    gradient_filter.SetInputData(dataset)
    gradient_filter.SetInputScalars(0, "Density")
    gradient_filter.SetResultArrayName("phase1_density_gradient")
    gradient_filter.Update()
    gradient = vtk_to_numpy(
        gradient_filter.GetOutput().GetPointData().GetArray("phase1_density_gradient")
    ).astype(float)
    gradient_magnitude = np.linalg.norm(gradient, axis=1)
    max_gradient_index = int(np.nanargmax(gradient_magnitude))
    return {
        "maximum_density": {
            "operationalization_id": "maximum_density",
            "feature_value": float(density[max_density_index]),
            "companion_value": float(gradient_magnitude[max_density_index]),
            "location": [float(value) for value in coordinates[max_density_index]],
            "point_index": max_density_index,
        },
        "maximum_density_gradient": {
            "operationalization_id": "maximum_density_gradient",
            "feature_value": float(gradient_magnitude[max_gradient_index]),
            "companion_value": float(density[max_gradient_index]),
            "location": [float(value) for value in coordinates[max_gradient_index]],
            "point_index": max_gradient_index,
        },
    }


FAMILIES = (
    FamilySpec(
        family_id="kitchen_turbulence_activity",
        dataset_id="Kitchen",
        title="Kitchen turbulence-activity hotspots",
        concept="Local turbulence-activity hotspots defined through distinct standard turbulence fields.",
        evidence_ids=("op_002", "op_003"),
        principal_dimensions=(
            OperationalizationDimension.FEATURE_DEFINITION,
            OperationalizationDimension.CRITERION,
            OperationalizationDimension.AGGREGATION_OR_REPRESENTATION,
        ),
        o2_unresolved=(OperationalizationDimension.CRITERION,),
        eligible_o2_masks=((OperationalizationDimension.CRITERION,),),
        scientific_coherence="Both documented standard turbulence fields can define a local activity hotspot; their common-concept interpretation remains subject to domain review.",
        o_role_coherence="feature_definition is the local grid-point hotspot; criterion selects it by maximizing ke or ep; representation is the stored coordinate.",
        ambiguity_quality="CONSEQUENTIAL scientific criterion choice: maximum ke and maximum ep select different points.",
        grounding_assessment="Sufficient for construction pilot through Kitchen variable documentation and cited k-epsilon literature; formal expert review remains pending.",
        context_fact_ids=("kitchen_physical_setting", "kitchen_room", "kitchen_hot_stove"),
        analyze=_analyze_kitchen_turbulence,
    ),
    FamilySpec(
        family_id="kitchen_concentration_heterogeneity",
        dataset_id="Kitchen",
        title="Kitchen gas-concentration heterogeneity",
        concept="Spatial heterogeneity among documented stored gas-concentration fields.",
        evidence_ids=("op_002", "op_004"),
        principal_dimensions=(
            OperationalizationDimension.FEATURE_DEFINITION,
            OperationalizationDimension.PROPERTY_MEASURE,
            OperationalizationDimension.AGGREGATION_OR_REPRESENTATION,
        ),
        o2_unresolved=(OperationalizationDimension.PROPERTY_MEASURE,),
        eligible_o2_masks=((OperationalizationDimension.PROPERTY_MEASURE,),),
        scientific_coherence="The documented concentration fields support a common heterogeneity question; measure equivalence remains subject to domain review.",
        o_role_coherence="feature_definition identifies the documented c-prefixed fields; property_measure compares heterogeneity; representation is the field identifier after volume weighting.",
        ambiguity_quality="CONSEQUENTIAL scientific measure choice: weighted CV and mean-normalized weighted interdecile spread select different fields.",
        grounding_assessment="Sufficient for construction pilot through Kitchen variable documentation and mixture-characteristics literature; species labels and units remain review items.",
        context_fact_ids=("kitchen_physical_setting", "kitchen_room", "kitchen_hot_stove"),
        analyze=_analyze_kitchen_concentrations,
    ),
    FamilySpec(
        family_id="combustor_density_features",
        dataset_id="Combustor",
        title="Combustor density features",
        concept="Prominent stored-density features defined by density magnitude or density-gradient magnitude.",
        evidence_ids=("op_002", "op_003"),
        principal_dimensions=(
            OperationalizationDimension.FEATURE_DEFINITION,
            OperationalizationDimension.CRITERION,
            OperationalizationDimension.AGGREGATION_OR_REPRESENTATION,
        ),
        o2_unresolved=(OperationalizationDimension.CRITERION,),
        eligible_o2_masks=((OperationalizationDimension.CRITERION,),),
        scientific_coherence="Stored density is documented, but the common-concept status of density magnitude versus density-gradient prominence needs scientific review.",
        o_role_coherence="feature_definition is a local density feature; criterion selects by density or density-gradient magnitude; representation is the stored coordinate.",
        ambiguity_quality="CONSEQUENTIAL candidate distinction, but the gradient branch requires stronger domain grounding before formal use.",
        grounding_assessment="Only density-feature visualization is directly documented; gradient-based operationalization is not yet independently grounded.",
        context_fact_ids=("combustor_physical_setting", "annular_combustor"),
        analyze=_analyze_combustor_density,
    ),
)


def _capability_profiles(datasets_root: Path, dataset_ids=None) -> list[dict[str, Any]]:
    limitations = {
        "Blunt_Fin": "No reviewed surface, reference direction, or boundary mapping supports a separation/wake concept.",
        "Carotid": "Velocity exists, but reviewed context supplies no centreline, wall, or anatomical direction for profile or shear claims.",
        "Combustor": "Stored Density has direct example/documentation support; no direction-dependent or combustion-state claim is made.",
        "FireFlow": "The snapshot and velocity are available, but reviewed sources do not establish physical semantics for t or mfrac.",
        "Kitchen": "Documented ke, ep, and c-prefixed gas concentrations support two non-speed candidate concepts; units remain unspecified.",
        "NASA_LOx_Post": "Velocity is derived, but reviewed resources do not establish reference direction, wall mapping, or a post-wake criterion.",
        "Office": "Ventilation context supports the existing strongest-region family, but no new independently grounded scalar concept is available.",
    }
    profiles: list[dict[str, Any]] = []
    for dataset_id in (DATASET_IDS if dataset_ids is None else dataset_ids):
        dataset_dir = datasets_root / dataset_id
        metadata = _read_json(datasets_root / dataset_id / "data_metadata.json")
        manifest = _read_json(dataset_dir / "dataset_manifest.json")
        construction = dataset_dir / "construction"
        sources = _read_json(construction / "sources.json")["sources"]
        evidence = _records(construction)
        context_records = [item for item in evidence if item.evidence_type == "context"]
        context_facts = [fact for item in context_records for fact in item.context_facts]
        variables = [
            {"name": variable["name"], "physical_quantity": variable["physical_quantity"], "source": variable["source"]}
            for variable in metadata["variables"]
        ]
        profiles.append(
            {
                "dataset_id": dataset_id,
                "grid_type": metadata["grid"]["type"],
                "temporal_support": metadata["temporal"]["type"],
                "available_variables": variables,
                "derived_quantities_that_are_physically_supported": [
                    item["name"] for item in variables
                    if item["source"] == "reader_derived" and item["physical_quantity"] is not None
                ],
                "geometry_information": [
                    item["path"] for item in manifest["files"] if item["role"] == "geometry"
                ] or "No separately reviewed geometry asset.",
                "boundary_information": [fact.model_dump(mode="json") for fact in context_facts if fact.field == "boundaries"] or "No approved boundary fact.",
                "reference_direction_information": [fact.model_dump(mode="json") for fact in context_facts if fact.field == "reference_directions"] or "No approved reference-direction fact.",
                "auxiliary_parameters": [
                    item["name"] for item in variables if item["physical_quantity"] is None
                ],
                "source_evidence": [
                    {"source_id": item["source_id"], "source_type": item["source_type"], "title": item["title"]}
                    for item in sources
                ],
                "known_scientific_use_context": [item.statement for item in context_records],
                "limitation_or_basis": limitations.get(dataset_id, "Only documented stored fields and this supplied snapshot are available; scientific task adequacy is reviewed separately."),
            }
        )
    return profiles


def _screening_matrix() -> list[dict[str, str]]:
    """Return the authored, sparse screening decisions for this pilot.

    The matrix is deliberately explicit rather than inferred from the number
    of ``FamilySpec`` objects.  In particular, the Combustor gradient branch
    remains construction material until its scientific grounding is reviewed.
    """

    return [
        {
            "dataset_id": "Kitchen",
            "concept_id": "kitchen_turbulence_activity",
            "data_feasibility": "PASS: stored ke and ep point arrays",
            "context_feasibility": "PASS: documented kitchen convection setting",
            "scientific_grounding": "PASS for construction pilot: KTH variable documentation and k-epsilon literature",
            "operationalization_feasibility": "PASS: maximum ke and maximum ep are reproducible criteria",
            "status": "STRONG_CANDIDATE",
            "data_evidence": "ke and ep point arrays with stored coordinates",
            "scientific_evidence": "Kitchen op_002, op_003",
            "major_ambiguity": "Which documented turbulence quantity should select the local activity hotspot?",
            "expected_o_choices": "criterion=max ke; criterion=max ep",
            "expected_findings": "coordinate, defining field value, local companion value",
            "unresolved_scientific_questions": "Stored units and formal domain review remain open.",
            "scientific_coherence": "KEEP provisionally for the construction pilot.",
            "o_role_coherence": "Hotspot is the object; field extremum is the criterion; coordinate is representation.",
        },
        {
            "dataset_id": "Kitchen",
            "concept_id": "kitchen_concentration_heterogeneity",
            "data_feasibility": "PASS: stored c-prefixed concentration fields and cell volumes",
            "context_feasibility": "PASS: documented room convection setting",
            "scientific_grounding": "PASS for construction pilot: KTH variable documentation and mixture-characteristics literature",
            "operationalization_feasibility": "PASS: two stable volume-weighted spread measures",
            "status": "STRONG_CANDIDATE",
            "data_evidence": "c1, c11-c17 point arrays and VTK cell volumes",
            "scientific_evidence": "Kitchen op_002, op_004",
            "major_ambiguity": "Which normalized dispersion summary defines greatest heterogeneity?",
            "expected_o_choices": "weighted CV; mean-normalized weighted interdecile spread",
            "expected_findings": "field identity, heterogeneity value, optional distribution characterization",
            "unresolved_scientific_questions": "Species labels and units require formal release review.",
            "scientific_coherence": "KEEP provisionally for the construction pilot.",
            "o_role_coherence": "Concentration field is the object; spread statistic is its property measure; field identifier is representation.",
        },
        {
            "dataset_id": "Combustor",
            "concept_id": "combustor_density_features",
            "data_feasibility": "PASS: stored Density point field and structured coordinates",
            "context_feasibility": "PASS: documented annular-combustor dataset",
            "scientific_grounding": "PARTIAL: density visualization is documented, gradient prominence is not independently grounded",
            "operationalization_feasibility": "PASS computationally: density and density-gradient extrema are reproducible",
            "status": "PROVISIONAL",
            "data_evidence": "Density point array and structured coordinates",
            "scientific_evidence": "Combustor op_002, op_003",
            "major_ambiguity": "Does prominence mean high density or strongest density change?",
            "expected_o_choices": "criterion=max density; criterion=max density-gradient magnitude",
            "expected_findings": "coordinate, defining value, companion density/gradient value",
            "unresolved_scientific_questions": "Whether the two criteria define one concept requires domain review; density units remain open.",
            "scientific_coherence": "PROVISIONAL pending review; not eligible for real-model readiness.",
            "o_role_coherence": "A local density feature is the object; density/gradient extremum is the criterion; coordinate is representation.",
        },
        {
            "dataset_id": "Office",
            "concept_id": "new_scalar_concept",
            "data_feasibility": "PARTIAL",
            "context_feasibility": "PARTIAL",
            "scientific_grounding": "FAIL for a new family",
            "operationalization_feasibility": "NOT ASSESSED",
            "status": "NOT_SUPPORTED",
            "data_evidence": "No independently reviewed new observable",
            "scientific_evidence": "No accepted new-concept evidence",
            "major_ambiguity": "No candidate ambiguity can be justified",
            "expected_o_choices": "None",
            "expected_findings": "None",
            "unresolved_scientific_questions": "Acquire new evidence before proposing a family.",
            "scientific_coherence": "REJECTED for Phase 1",
            "o_role_coherence": "Not applicable",
        },
        {
            "dataset_id": "FireFlow",
            "concept_id": "thermal_or_mixture_concept",
            "data_feasibility": "PARTIAL",
            "context_feasibility": "FAIL",
            "scientific_grounding": "FAIL",
            "operationalization_feasibility": "NOT ASSESSED",
            "status": "NOT_SUPPORTED",
            "data_evidence": "t and mfrac arrays are present but semantics are unreviewed",
            "scientific_evidence": "No accepted dataset-specific grounding",
            "major_ambiguity": "Variable meaning is unresolved",
            "expected_o_choices": "None",
            "expected_findings": "None",
            "unresolved_scientific_questions": "Review source documentation first.",
            "scientific_coherence": "REJECTED for Phase 1",
            "o_role_coherence": "Not applicable",
        },
        {
            "dataset_id": "Carotid",
            "concept_id": "profile_or_wall_shear_concept",
            "data_feasibility": "PARTIAL",
            "context_feasibility": "FAIL",
            "scientific_grounding": "PARTIAL",
            "operationalization_feasibility": "NOT ASSESSED",
            "status": "NOT_SUPPORTED",
            "data_evidence": "Velocity is present without reviewed wall/centreline metadata",
            "scientific_evidence": "No accepted dataset-specific grounding",
            "major_ambiguity": "Reference direction and boundary are unresolved",
            "expected_o_choices": "None",
            "expected_findings": "None",
            "unresolved_scientific_questions": "Acquire anatomical/boundary evidence first.",
            "scientific_coherence": "REJECTED for Phase 1",
            "o_role_coherence": "Not applicable",
        },
        {
            "dataset_id": "Blunt_Fin",
            "concept_id": "separation_or_wake_concept",
            "data_feasibility": "PARTIAL",
            "context_feasibility": "FAIL",
            "scientific_grounding": "PARTIAL",
            "operationalization_feasibility": "NOT ASSESSED",
            "status": "NOT_SUPPORTED",
            "data_evidence": "Flow data exist without reviewed fin surface or direction mapping",
            "scientific_evidence": "No accepted dataset-specific grounding",
            "major_ambiguity": "Boundary and streamwise reference are unresolved",
            "expected_o_choices": "None",
            "expected_findings": "None",
            "unresolved_scientific_questions": "Acquire geometry and direction evidence first.",
            "scientific_coherence": "REJECTED for Phase 1",
            "o_role_coherence": "Not applicable",
        },
        {
            "dataset_id": "NASA_LOx_Post",
            "concept_id": "post_wake_concept",
            "data_feasibility": "PARTIAL",
            "context_feasibility": "FAIL",
            "scientific_grounding": "PARTIAL",
            "operationalization_feasibility": "NOT ASSESSED",
            "status": "NOT_SUPPORTED",
            "data_evidence": "Velocity is derived but post-wake boundaries are unreviewed",
            "scientific_evidence": "No accepted dataset-specific grounding",
            "major_ambiguity": "Reference direction and boundary mapping are unresolved",
            "expected_o_choices": "None",
            "expected_findings": "None",
            "unresolved_scientific_questions": "Acquire geometry and direction evidence first.",
            "scientific_coherence": "REJECTED for Phase 1",
            "o_role_coherence": "Not applicable",
        },
    ]


def _screening_status_by_family(screening: list[dict[str, str]]) -> dict[str, str]:
    """Use the authored screening matrix as the sole readiness source."""

    result: dict[str, str] = {}
    for family in FAMILIES:
        matches = [item for item in screening if item["concept_id"] == family.family_id]
        if len(matches) != 1:
            raise RuntimeError(
                f"screening matrix must contain exactly one status for {family.family_id}"
            )
        result[family.family_id] = matches[0]["status"]
    return result


def _grounding_labels(screening_status: str) -> dict[str, str]:
    """Return status-derived wording for generated cards and case artifacts."""

    labels = {
        "STRONG_CANDIDATE": {
            "construction_grounding": "CONSTRUCTION-PILOT-GROUNDING-PASS",
            "formal_review": "FORMAL-SCIENTIFIC-REVIEW-PENDING",
        },
        "PROVISIONAL": {
            "construction_grounding": "PROVISIONAL-SCIENTIFIC-GROUNDING",
            "formal_review": "INSUFFICIENT-FOR-REAL-MODEL-READINESS",
        },
        "NOT_SUPPORTED": {
            "construction_grounding": "NOT-SUPPORTED",
            "formal_review": "NOT-SUPPORTED",
        },
    }
    try:
        return labels[screening_status]
    except KeyError as exc:
        raise RuntimeError(f"unknown screening status: {screening_status!r}") from exc


def _classify_branch_divergence(*, same_outcome: bool) -> str:
    """Return a construction diagnostic, never a scientific validity gate."""

    return "SAME_OUTCOME" if same_outcome else "CONSEQUENTIAL_DIFFERENCE"


def _questions(family: FamilySpec, suffix: str) -> tuple[str, dict[str, Any]]:
    if family.definition is not None:
        from flowintentbench.construction_recipes import authored_question
        return authored_question(family, suffix)
    if family.family_id == "kitchen_turbulence_activity":
        scope = "the Kitchen airflow field"
        target = "the most prominent local turbulence-activity hotspot"
        finding_goal = "identify the most prominent local turbulence-activity hotspot and report its coordinate and requested local quantities"
        fixed_feature = "treat a turbulence-activity hotspot as a stored grid point"
        criterion = "select the hotspot by maximizing turbulent kinetic energy"
        representation = "represent its location as its exact stored grid-point coordinate"
        quantity = "Report its coordinate, turbulent kinetic energy at the selected location, and turbulent dissipation rate at the selected location."
        o2 = "Using an appropriate local turbulence-activity criterion, identify the most prominent hotspot."
        o3 = "identify the most prominent local turbulence-activity hotspot using a scientifically defensible operationalization."
    elif family.family_id == "kitchen_concentration_heterogeneity":
        scope = "the Kitchen airflow field"
        target = "the stored gas-concentration field with the greatest spatial heterogeneity"
        finding_goal = "identify the stored gas-concentration field with the greatest spatial heterogeneity and report its identifier and requested heterogeneity quantities"
        fixed_feature = "treat every stored c-prefixed scalar as a gas-concentration field"
        criterion = "compare the fields using the cell-volume-weighted coefficient of variation"
        representation = "represent the selected result by its stored field identifier"
        quantity = "Report the selected field identifier and the cell-volume-weighted coefficient-of-variation value."
        o2 = "Using an appropriate quantitative concentration-heterogeneity measure, identify the field with greatest heterogeneity."
        o3 = "identify the stored gas-concentration field with greatest spatial heterogeneity using a scientifically defensible operationalization."
    else:
        scope = "the annular-combustor flow field"
        target = "the most prominent density feature"
        finding_goal = "identify the most prominent density feature and report its coordinate and requested density quantities"
        fixed_feature = "treat a density feature as a stored grid point"
        criterion = "select the feature by maximizing stored density"
        representation = "represent its location as its exact stored grid-point coordinate"
        quantity = "Report its coordinate and the stored density value."
        o2 = "Using an appropriate density-feature criterion, identify the most prominent density feature."
        o3 = "identify the most prominent density feature using a scientifically defensible operationalization."
    if suffix in {"o1_f1", "o1_f2"}:
        if family.family_id == "kitchen_concentration_heterogeneity":
            o_clause = f"Within {scope}, {fixed_feature}, {criterion}, and {representation}."
            methods = [
                {"category": "feature_definition", "statement": fixed_feature, "question_fragment": fixed_feature},
                {"category": "property_measure", "statement": criterion, "question_fragment": criterion},
                {"category": "analysis_procedure", "statement": representation, "question_fragment": representation},
            ]
            dimension_clauses = {
                OperationalizationDimension.FEATURE_DEFINITION: fixed_feature,
                OperationalizationDimension.PROPERTY_MEASURE: criterion,
                OperationalizationDimension.AGGREGATION_OR_REPRESENTATION: representation,
            }
        else:
            o_clause = f"Within {scope}, {fixed_feature}, {criterion}, and {representation}."
            methods = [
                {"category": "feature_definition", "statement": fixed_feature, "question_fragment": fixed_feature},
                {"category": "criterion", "statement": criterion, "question_fragment": criterion},
                {"category": "analysis_procedure", "statement": representation, "question_fragment": representation},
            ]
            dimension_clauses = {
                OperationalizationDimension.FEATURE_DEFINITION: fixed_feature,
                OperationalizationDimension.CRITERION: criterion,
                OperationalizationDimension.AGGREGATION_OR_REPRESENTATION: representation,
            }
        responsibility = OperationalizationResponsibility.USER_SPECIFIED
        unresolved: tuple[OperationalizationDimension, ...] = ()
        openness = FindingOpenness.OPEN if suffix == "o1_f2" else FindingOpenness.BOUNDED
    elif suffix == "o2_f1":
        o_clause = f"Within {scope}, {fixed_feature}, and {representation}."
        if family.family_id == "kitchen_concentration_heterogeneity":
            quantity = "Report the selected field identifier and the numerical value of the heterogeneity measure used to select it."
        elif family.family_id == "combustor_density_features":
            quantity = "Report its coordinate and the numerical value of the density criterion used to select it."
        question = f"{o_clause} {o2} {quantity}"
        methods = [
            {"category": "feature_definition", "statement": fixed_feature, "question_fragment": fixed_feature},
            {"category": "analysis_procedure", "statement": representation, "question_fragment": representation},
        ]
        dimension_clauses = {
            OperationalizationDimension.FEATURE_DEFINITION: fixed_feature,
            OperationalizationDimension.AGGREGATION_OR_REPRESENTATION: representation,
        }
        responsibility = OperationalizationResponsibility.PARTIALLY_SPECIFIED
        unresolved = family.o2_unresolved
        openness = FindingOpenness.BOUNDED
    elif suffix == "o3_f1":
        o_clause = f"Within {scope},"
        if family.family_id == "kitchen_concentration_heterogeneity":
            quantity = "Report the selected field identifier and the numerical value of the heterogeneity measure used to select it."
        elif family.family_id == "combustor_density_features":
            quantity = "Report its coordinate and the numerical value of the density criterion used to select it."
        question = f"{o_clause} {o3} {quantity}"
        methods = []
        dimension_clauses = {}
        responsibility = OperationalizationResponsibility.MODEL_SELECTED
        unresolved = family.principal_dimensions
        openness = FindingOpenness.BOUNDED
    else:
        raise ValueError(f"unknown case suffix {suffix}")
    if suffix in {"o1_f1", "o1_f2"}:
        f1_properties = (
            _f1_finding_properties(family, o1_resolved=True)
            if suffix == "o1_f1"
            else ()
        )
        finding_request = (
            f"Identify {target}. Report "
            + ", ".join(property_.request_phrase for property_ in f1_properties)
            + "."
            if suffix == "o1_f1"
            else f"Characterize {target} and report the scientifically relevant findings supported by the data."
        )
        question = f"{o_clause} {finding_request}"
    elif openness == FindingOpenness.BOUNDED:
        f1_properties = _f1_finding_properties(
            family,
            o1_resolved=False,
        )
    else:
        f1_properties = ()
    requirements = [
        {
            "category": property_.category.value,
            "statement": property_.requirement_statement,
            "question_fragment": property_.request_phrase,
        }
        for property_ in f1_properties
    ]
    return question, {
        "scientific_target": target,
        "finding_goal": finding_goal,
        "responsibility": responsibility,
        "unresolved": unresolved,
        "openness": openness,
        "methods": methods,
        "dimension_clauses": dimension_clauses,
        "requirements": requirements,
        "f1_properties": [property_.property_id for property_ in f1_properties],
        "scope": scope,
        # This is the controlled selection instruction shared by O1/O2/O3/F2;
        # the scientific target wording itself may be intentionally less specific
        # in O2/O3 questions.
        "selection": "identify the requested target",
        "selection_fragment": "Characterize" if suffix == "o1_f2" else "Identify",
    }


def _metadata(family: FamilySpec, case_id: str, suffix: str) -> tuple[CaseConstructionMetadata, str, dict[str, Any]]:
    question, details = _questions(family, suffix)
    responsibility_contract = _build_responsibility_contract(family, suffix, details)
    representability_contract = _build_representability_contract()
    metadata = CaseConstructionMetadata.model_validate(
        {
            "case_id": case_id,
            "case_family_id": family.family_id,
            "dataset_id": family.dataset_id,
            "scientific_target": details["scientific_target"],
            "finding_goal": details["finding_goal"],
            "operationalization_responsibility": details["responsibility"].value,
            "finding_openness": details["openness"].value,
            "principal_operationalization_dimensions": [item.value for item in family.principal_dimensions],
            "unresolved_operationalization_dimensions": [item.value for item in details["unresolved"]],
            "scope_constraints": [{"statement": details["scope"], "question_fragment": details["scope"]}],
            "condition_constraints": [],
            "selection_constraints": [{"statement": details["selection"], "question_fragment": details["selection_fragment"]}],
            "explicit_method_constraints": details["methods"],
            "explicit_finding_requirements": details["requirements"],
            "responsibility_contract": responsibility_contract,
            "evaluation_representability_contract": representability_contract,
        }
    )
    return metadata, question, details


def _build_responsibility_contract(
    family: FamilySpec, suffix: str, details: dict[str, Any]
) -> dict[str, Any]:
    """Materialize the frozen Q2 responsibility contract from authored case data."""

    target = normalize_case_text(details["scientific_target"])
    if "density feature" in target:
        target_anchor = "most prominent density feature"
    elif "heterogeneity" in target:
        target_anchor = "heterogeneity"
    elif "turbulence" in target:
        target_anchor = "hotspot"
    else:
        target_anchor = target.removeprefix("the ")
    resolved_dimensions = set(family.principal_dimensions) - set(details["unresolved"])
    resolved: list[dict[str, Any]] = []
    for dimension, statement in details["dimension_clauses"].items():
        if dimension not in resolved_dimensions:
            continue
        normalized = normalize_case_text(str(statement))
        resolved.append(
            {
                "dimension_id": dimension.value,
                "canonical_id": f"{family.family_id}:{dimension.value}",
                "normalized_meaning": normalized,
                "question_visibility": "REQUIRED",
                "required_semantic_anchors": [
                    token for token in normalized.split() if len(token) >= 4
                ][:4],
            }
        )
    unresolved = [
        {
            "dimension_id": dimension.value,
            "normalized_meaning": f"model-selected {dimension.value.replace('_', ' ')}",
            "must_remain_open": True,
        }
        for dimension in details["unresolved"]
    ]
    if suffix == "o1_f2":
        finding_anchors = ["characterize", "findings"]
    else:
        requirement_text = " ".join(
            normalize_case_text(str(item.get("statement", "")))
            for item in details.get("requirements", ())
            if isinstance(item, Mapping)
        )
        finding_anchors = ["report"]
        finding_anchors.extend(
            token for token in ("coordinate", "identifier", "value", "quantity")
            if token in requirement_text
        )
        finding_anchors = finding_anchors[:2]
    return {
        "scientific_target": {
            "canonical_id": family.family_id,
            "normalized_meaning": target,
            "required_semantic_anchors": [target_anchor],
        },
        "resolved_operationalization_clauses": resolved,
        "unresolved_operationalization_dimensions": unresolved,
        "finding_responsibility": {
            "finding_mode": "F2" if suffix == "o1_f2" else "F1",
            "normalized_demand": normalize_case_text(details["finding_goal"]),
            "required_semantic_anchors": finding_anchors,
            "required_responsibility_mode": "SELECT_PRINCIPAL_FINDINGS" if suffix == "o1_f2" else None,
            "forbidden_responsibility_modes": [
                "EXHAUSTIVE_FIXED_FINDING_LIST" if suffix == "o1_f2" else "OPEN_FINDING_SELECTION"
            ],
        },
    }


def _build_representability_contract() -> dict[str, Any]:
    """Materialize the explicit production evaluation routes for each case."""

    return {
        "o_schema_available": True,
        "materialization": {
            "available": True,
            "handler_id": "deterministic_case_materializer_v1",
        },
        "finding_representation": {
            "available": True,
            "extractor_id": "atomic_finding_extractor_v1",
        },
        "deterministic_verification": {
            "required_claim_types": ["exact", "scalar_numeric", "spatial"],
            "registered_handlers": ["exact", "scalar_numeric", "spatial_tolerance"],
        },
        "semantic_adjudication": {
            "required": True,
            "handler_available": True,
            "handler_id": "scientific_adjudication_v1",
        },
        "valid_unenumerated_escalation": {
            "available": True,
            "route_id": "valid_unenumerated",
        },
        "uncertain_escalation": {
            "available": True,
            "route_id": "uncertain",
        },
    }


def _validate_model_visible_operationalization_coverage(
    family: FamilySpec,
    metadata: CaseConstructionMetadata,
    scientific_question: str,
    details: dict[str, Any],
) -> None:
    """Check authored O clauses without guessing from keywords or GT."""

    expected = set(metadata.principal_operationalization_dimensions) - set(
        metadata.unresolved_operationalization_dimensions
    )
    clauses = details["dimension_clauses"]
    if set(clauses) != expected:
        raise RuntimeError(
            f"{family.family_id}/{metadata.case_id}: model-visible O coverage mismatch "
            f"(expected={sorted(item.value for item in expected)!r}, "
            f"actual={sorted(item.value for item in clauses)!r})"
        )
    normalized_question = normalize_case_text(scientific_question)
    for dimension, clause in clauses.items():
        if normalize_case_text(clause) not in normalized_question:
            raise RuntimeError(
                f"{family.family_id}/{metadata.case_id}: authored clause for "
                f"{dimension.value} is not visible in the scientific question"
            )


def _decisions(family: FamilySpec, op_name: str) -> list[OperationalizationDecision]:
    if family.definition is not None:
        return [OperationalizationDecision(dimension=dimension, statement=statement)
                for dimension, statement in family.definition["operations"][op_name]["clauses"].items()]
    if family.family_id == "kitchen_turbulence_activity":
        criterion = {
            "maximum_turbulent_kinetic_energy": "Select the hotspot by maximizing turbulent kinetic energy.",
            "maximum_turbulent_dissipation": "Select the hotspot by maximizing turbulent dissipation rate.",
        }[op_name]
        return [
            OperationalizationDecision(dimension=OperationalizationDimension.FEATURE_DEFINITION, statement="Treat a turbulence-activity hotspot as a stored grid point."),
            OperationalizationDecision(dimension=OperationalizationDimension.CRITERION, statement=criterion),
            OperationalizationDecision(dimension=OperationalizationDimension.AGGREGATION_OR_REPRESENTATION, statement="Represent the selected hotspot by its exact stored grid-point coordinate."),
        ]
    if family.family_id == "kitchen_concentration_heterogeneity":
        measure = {
            "coefficient_of_variation": "Compare fields by the cell-volume-weighted coefficient of variation of their point-to-cell mean concentrations.",
            "relative_interdecile_spread": "Compare fields by the cell-volume-weighted relative interdecile concentration spread, normalized by the cell-volume-weighted mean.",
        }[op_name]
        return [
            OperationalizationDecision(dimension=OperationalizationDimension.FEATURE_DEFINITION, statement="Treat each stored c-prefixed scalar array as a gas-concentration field."),
            OperationalizationDecision(dimension=OperationalizationDimension.PROPERTY_MEASURE, statement=measure),
            OperationalizationDecision(dimension=OperationalizationDimension.AGGREGATION_OR_REPRESENTATION, statement="Represent the selected result by its stored field identifier after cell-volume-weighted aggregation."),
        ]
    criterion = {
        "maximum_density": "Select the density feature by maximizing stored density.",
        "maximum_density_gradient": "Select the density feature by maximizing density-gradient magnitude, computed by vtkGradientFilter on the stored Density point field.",
    }[op_name]
    return [
        OperationalizationDecision(dimension=OperationalizationDimension.FEATURE_DEFINITION, statement="Treat a density feature as a stored grid point."),
        OperationalizationDecision(dimension=OperationalizationDimension.CRITERION, statement=criterion),
        OperationalizationDecision(dimension=OperationalizationDimension.AGGREGATION_OR_REPRESENTATION, statement="Represent the selected density feature by its exact stored grid-point coordinate."),
    ]


def _scalar_verification(value: float) -> VerificationSpec:
    return VerificationSpec(
        absolute_tolerance=absolute_tolerance_for_significant_figures(value, significant_figures=5)
    )


def _findings(
    family: FamilySpec,
    metadata: CaseConstructionMetadata,
    op_name: str,
    result: dict[str, Any],
) -> tuple[list[ReferenceFinding], dict[str, str]]:
    if family.definition is not None:
        from flowintentbench.construction_recipes import computed_findings
        return computed_findings(family, metadata, op_name, result)
    prefix = f"{metadata.case_id}_{op_name}"
    property_mapping: dict[str, str] = {}
    required_properties = {
        item.property_id: item
        for item in _f1_finding_properties(
            family,
            o1_resolved=(metadata.operationalization_responsibility == OperationalizationResponsibility.USER_SPECIFIED),
        )
    } if metadata.finding_openness == FindingOpenness.BOUNDED else {}

    def authored_finding(
        *,
        finding_id: str,
        property_id: str | None,
        default_category: FindingRequirementCategory,
        statement: str,
        importance: FindingImportance,
        value: Any,
        verification: VerificationSpec | None,
    ) -> ReferenceFinding:
        property_spec = required_properties.get(property_id or "")
        if property_spec is not None:
            if property_id in property_mapping:
                raise RuntimeError(f"duplicate F1 property mapping: {property_id}")
            property_mapping[property_id] = finding_id
        return ReferenceFinding(
            finding_id=finding_id,
            category=property_spec.category if property_spec is not None else default_category,
            statement=statement,
            importance=FindingImportance.CORE if property_spec is not None else importance,
            value=value,
            verification=verification,
        )

    if family.family_id == "kitchen_concentration_heterogeneity":
        metric_name = "coefficient of variation" if op_name == "coefficient_of_variation" else "relative interdecile spread"
        metric_value = float(result[op_name])
        chosen = result["field_name"]
        findings = [
            authored_finding(
                finding_id=f"{prefix}_field",
                property_id="selected_concentration_field",
                default_category=FindingRequirementCategory.EXISTENCE_OR_IDENTITY,
                statement=f"The selected gas-concentration field is {chosen}.",
                importance=FindingImportance.CORE,
                value=chosen,
                verification=None,
            ),
            authored_finding(
                finding_id=f"{prefix}_heterogeneity",
                property_id="heterogeneity_value",
                default_category=FindingRequirementCategory.QUANTITY,
                statement=f"The selected field has cell-volume-weighted {metric_name} approximately {metric_value:.5g}.",
                importance=FindingImportance.CORE,
                value=metric_value,
                verification=_scalar_verification(metric_value),
            ),
        ]
        if metadata.finding_openness == FindingOpenness.OPEN:
            for quantile_name in ("q10", "q50", "q90"):
                quantile = float(result[quantile_name])
                findings.append(
                    ReferenceFinding(
                        finding_id=f"{prefix}_{quantile_name}",
                        category=FindingRequirementCategory.CHARACTERIZATION,
                        statement=f"The selected field has weighted concentration {quantile_name} approximately {quantile:.5g} in its stored concentration scale.",
                        importance=FindingImportance.SUPPORTING,
                        value=quantile,
                        verification=_scalar_verification(quantile),
                    )
                )
        return findings, property_mapping
    if family.family_id == "kitchen_turbulence_activity":
        feature_label = "turbulent kinetic energy" if op_name == "maximum_turbulent_kinetic_energy" else "turbulent dissipation rate"
        companion_label = "turbulent dissipation rate" if op_name == "maximum_turbulent_kinetic_energy" else "turbulent kinetic energy"
    else:
        feature_label = "density" if op_name == "maximum_density" else "density-gradient magnitude"
        companion_label = "density-gradient magnitude" if op_name == "maximum_density" else "density"
    feature_value = float(result["feature_value"])
    companion_value = float(result["companion_value"])
    findings = [
        authored_finding(
            finding_id=f"{prefix}_location",
            property_id=("selected_hotspot_location" if family.family_id == "kitchen_turbulence_activity" else "selected_density_feature_location"),
            default_category=FindingRequirementCategory.LOCATION,
            statement=f"The selected feature is at stored coordinate approximately {result['location']}.",
            importance=FindingImportance.CORE,
            value=result["location"],
            verification=VerificationSpec(spatial_tolerance=1e-4),
        ),
        authored_finding(
            finding_id=f"{prefix}_feature_value",
            property_id=(
                "turbulent_kinetic_energy"
                if family.family_id == "kitchen_turbulence_activity" and op_name == "maximum_turbulent_kinetic_energy"
                else "turbulent_dissipation_rate"
                if family.family_id == "kitchen_turbulence_activity"
                else "density_feature_value"
            ),
            default_category=FindingRequirementCategory.QUANTITY,
            statement=f"The selected feature has {feature_label} approximately {feature_value:.5g} in the stored scale.",
            importance=FindingImportance.CORE,
            value=feature_value,
            verification=_scalar_verification(feature_value),
        ),
    ]
    if metadata.finding_openness == FindingOpenness.OPEN:
        findings.append(authored_finding(finding_id=f"{prefix}_companion_value", property_id=None, default_category=FindingRequirementCategory.CHARACTERIZATION, statement=f"At that selected coordinate, {companion_label} is approximately {companion_value:.5g} in the stored scale.", importance=FindingImportance.SUPPORTING, value=companion_value, verification=_scalar_verification(companion_value)))
    elif family.family_id == "kitchen_turbulence_activity":
        companion_property_id = "turbulent_dissipation_rate" if op_name == "maximum_turbulent_kinetic_energy" else "turbulent_kinetic_energy"
        findings.append(authored_finding(finding_id=f"{prefix}_companion_value", property_id=companion_property_id, default_category=FindingRequirementCategory.QUANTITY, statement=f"At the selected coordinate, {companion_label} is approximately {companion_value:.5g} in the stored scale.", importance=FindingImportance.CORE, value=companion_value, verification=_scalar_verification(companion_value)))
    return findings, property_mapping


def _mapped_core_property_ids(
    branch: OperationalizationFindingBranch,
    property_mapping: dict[str, str],
) -> set[str]:
    """Resolve F1 properties solely through the authored construction map."""

    core_finding_ids = {
        finding.finding_id
        for finding in branch.findings
        if finding.importance == FindingImportance.CORE
    }
    return {
        property_id
        for property_id, finding_id in property_mapping.items()
        if finding_id in core_finding_ids
    }


def _validate_f1_gt_coverage(
    family: FamilySpec,
    metadata: CaseConstructionMetadata,
    details: dict[str, Any],
    ground_truth: GroundTruth,
    property_mapping: dict[str, dict[str, str]],
) -> None:
    """Ensure every bounded request exactly matches each branch's CORE GT."""

    if metadata.finding_openness != FindingOpenness.BOUNDED:
        return
    required = set(details["f1_properties"])
    if required != set(property_.property_id for property_ in _f1_finding_properties(
        family,
        o1_resolved=(metadata.operationalization_responsibility == OperationalizationResponsibility.USER_SPECIFIED),
    )):
        raise RuntimeError(f"F1 authored property specification drift for {metadata.case_id}")
    for branch in ground_truth.findings_by_operationalization:
        branch_mapping = property_mapping.get(branch.operationalization_id, {})
        actual = _mapped_core_property_ids(branch, branch_mapping)
        core_finding_ids = {
            finding.finding_id
            for finding in branch.findings
            if finding.importance == FindingImportance.CORE
        }
        mapped_finding_ids = set(branch_mapping.values())
        missing = required - actual
        unexpected = actual - required
        unexpected_unmapped = core_finding_ids - mapped_finding_ids
        if missing or unexpected or unexpected_unmapped:
            raise RuntimeError(
                f"F1 exact CORE GT mismatch for {metadata.case_id}/{branch.operationalization_id}: "
                f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}, "
                f"unexpected_core_findings={sorted(unexpected_unmapped)!r}"
            )
        if mapped_finding_ids != core_finding_ids:
            raise RuntimeError(
                f"F1 property mapping does not cover every CORE finding for "
                f"{metadata.case_id}/{branch.operationalization_id}"
            )
        if not core_finding_ids:
            raise RuntimeError(f"F1 branch has no core findings: {branch.operationalization_id}")


def _branch_results(family: FamilySpec, results: dict[str, dict[str, Any]], op_name: str) -> dict[str, Any]:
    if family.family_id == "kitchen_concentration_heterogeneity":
        item = dict(results[op_name])
        item[op_name] = float(item[op_name])
        return item
    return results[op_name]


def _accepted_operation_names(family: FamilySpec) -> tuple[str, str]:
    if family.definition is not None:
        return tuple(family.definition["operations"])
    if family.family_id == "kitchen_turbulence_activity":
        return ("maximum_turbulent_kinetic_energy", "maximum_turbulent_dissipation")
    if family.family_id == "kitchen_concentration_heterogeneity":
        return ("coefficient_of_variation", "relative_interdecile_spread")
    return ("maximum_density", "maximum_density_gradient")


def _ground_truth(
    family: FamilySpec,
    metadata: CaseConstructionMetadata,
    results: dict[str, dict[str, Any]],
) -> tuple[GroundTruth, dict[str, dict[str, str]]]:
    operation_names = _accepted_operation_names(family)
    if metadata.operationalization_responsibility == OperationalizationResponsibility.USER_SPECIFIED:
        operation_names = operation_names[:1]
    bundles: list[OperationalizationBundle] = []
    branches: list[OperationalizationFindingBranch] = []
    property_mapping: dict[str, dict[str, str]] = {}
    for op_name in operation_names:
        op_id = f"{metadata.case_id}_{op_name}"
        result = _branch_results(family, results, op_name)
        findings, branch_mapping = _findings(family, metadata, op_name, result)
        bundles.append(OperationalizationBundle(operationalization_id=op_id, decisions=_decisions(family, op_name), evidence_ids=list(family.evidence_ids)))
        branches.append(OperationalizationFindingBranch(operationalization_id=op_id, findings=findings))
        property_mapping[op_id] = branch_mapping
    return GroundTruth(dataset_id=metadata.dataset_id, case_id=metadata.case_id, case_family_id=metadata.case_family_id, acceptable_operationalizations=bundles, findings_by_operationalization=branches), property_mapping


def _materialize_context(family: FamilySpec, datasets_root: Path, case_id: str, question: str, output_dir: Path) -> ContextSelection:
    dataset_dir = datasets_root / family.dataset_id
    construction_dir = dataset_dir / "construction"
    sources = load_source_collection(construction_dir / "sources.json", dataset_id=family.dataset_id, require_formal_source=True)
    evidence = _records(construction_dir)
    built = DatasetContextBuilder().build(family.dataset_id, sources, evidence)
    selection = ContextSelection(case_id=case_id, dataset_id=family.dataset_id, include_fact_ids=list(family.context_fact_ids))
    selected = CaseContextSelector().select(built.context, built.provenance, selection, data_file_paths=[item.path for item in _flow_data(dataset_dir).data_files])
    _write_json(output_dir / "context_selection.json", selection.model_dump(mode="json"))
    _write_json(output_dir / "case_context.json", selected.context.model_dump(mode="json"))
    _write_json(output_dir / "case_context_provenance.json", selected.provenance.model_dump(mode="json"))
    case_input = BenchmarkCaseInput(scientific_question=question, flow_data=_flow_data(dataset_dir), case_context=selected.context)
    _write_json(output_dir / "case_input.json", case_input.model_dump(mode="json"))
    return selection


def _write_case(
    family: FamilySpec,
    datasets_root: Path,
    output_root: Path,
    suffix: str,
    results: dict[str, dict[str, Any]],
    *,
    screening_status: str,
) -> dict[str, Any]:
    case_id = f"{family.family_id}_{suffix}"
    case_dir = output_root / "candidate_cases" / family.dataset_id / case_id
    metadata, question, question_details = _metadata(family, case_id, suffix)
    selection = _materialize_context(family, datasets_root, case_id, question, case_dir)
    CaseConstructionMetadataValidator.validate(question, metadata, context_selection=selection)
    _validate_model_visible_operationalization_coverage(family, metadata, question, question_details)
    ground_truth, property_mapping = _ground_truth(family, metadata, results)
    evidence = _records(datasets_root / family.dataset_id / "construction")
    GroundTruthValidator.validate(ground_truth, metadata, evidence_records=evidence)
    # Persist the same canonical floating-point representation used by the
    # other construction artifacts.  Validation is repeated after the
    # round-trip so the written GT remains the object used by later checks.
    ground_truth = GroundTruth.model_validate(
        _canonicalize_json(ground_truth.model_dump(mode="json"))
    )
    GroundTruthValidator.validate(ground_truth, metadata, evidence_records=evidence)
    _validate_f1_gt_coverage(
        family,
        metadata,
        question_details,
        ground_truth,
        property_mapping,
    )
    _write_json(case_dir / "scientific_question.json", {
        "scientific_question": question,
        "f1_required_properties": question_details["f1_properties"],
        "model_visible_operationalization_clauses": {
            dimension.value: clause
            for dimension, clause in question_details["dimension_clauses"].items()
        },
    })
    _write_json(case_dir / "case_construction_metadata.json", metadata.model_dump(mode="json"))
    _write_json(case_dir / "f1_property_mapping.json", {
        "case_id": case_id,
        "mapping": property_mapping,
    })
    save_ground_truth(ground_truth, case_dir / "ground_truth.json", case_metadata=metadata, evidence_records=evidence)
    grounding = _grounding_labels(screening_status)
    _write_json(case_dir / "construction_pilot_status.json", {
        "status": PILOT_STATUS,
        "family_screening_status": screening_status,
        "scientific_grounding": grounding["construction_grounding"],
        "formal_scientific_review": grounding["formal_review"],
        "model_evaluation": "NOT RUN",
        "formal_release": "NOT ELIGIBLE FROM THIS ARTIFACT",
    })
    _write_markdown(case_dir / "reference_analysis.md", [
        f"# {case_id} Reference Analysis",
        "",
        f"Status: **{PILOT_STATUS}**.",
        "",
        "Every accepted complete operationalization was executed on the reviewed reader-visible dataset before its G(O) branch was written.",
        "",
        "## Accepted operationalizations",
        "",
        *[f"- `{bundle.operationalization_id}`: " + "; ".join(decision.statement for decision in bundle.decisions) for bundle in ground_truth.acceptable_operationalizations],
        "",
        "The artifact does not assert a formal release or a model-evaluation result.",
    ])
    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "metadata": metadata,
        "selection": selection,
        "ground_truth": ground_truth,
        "property_mapping": property_mapping,
        "question": question,
        "question_details": question_details,
    }


def _role_audit(family: FamilySpec) -> list[dict[str, str]]:
    if family.family_id == "kitchen_turbulence_activity":
        return [
            {"scientific_decision": "Analyze a local turbulence-activity hotspot represented by a stored grid point.", "dimension": "feature_definition", "reason": "This states what scientific object is being analyzed without choosing the ranking field."},
            {"scientific_decision": "Select the point by maximum turbulent kinetic energy or maximum turbulent dissipation rate.", "dimension": "criterion", "reason": "This is the rule that selects one point from the candidate object set."},
            {"scientific_decision": "Report the exact stored grid-point coordinate.", "dimension": "aggregation_or_representation", "reason": "This controls how the selected object is spatially represented."},
        ]
    if family.family_id == "kitchen_concentration_heterogeneity":
        return [
            {"scientific_decision": "Treat documented c-prefixed arrays as the candidate gas-concentration fields.", "dimension": "feature_definition", "reason": "This defines the scientific entities being compared."},
            {"scientific_decision": "Compare heterogeneity using weighted CV or mean-normalized weighted interdecile spread.", "dimension": "property_measure", "reason": "This is the object-level quantitative property used for comparison and ranking."},
            {"scientific_decision": "Use physical cell-volume weighting and report the stored field identifier.", "dimension": "aggregation_or_representation", "reason": "This fixes aggregation support and the representation of the selected entity."},
        ]
    return [
        {"scientific_decision": "Analyze a local density feature represented by a stored grid point.", "dimension": "feature_definition", "reason": "This states the candidate scientific object without selecting a prominence rule."},
        {"scientific_decision": "Select the point by maximum density or maximum density-gradient magnitude.", "dimension": "criterion", "reason": "This is the rule that chooses one feature point; its scientific grounding remains provisional."},
        {"scientific_decision": "Report the exact stored grid-point coordinate.", "dimension": "aggregation_or_representation", "reason": "This controls how the selected object is spatially represented."},
    ]


def _write_design_card(
    family: FamilySpec,
    results: dict[str, dict[str, Any]],
    output_root: Path,
    *,
    screening_status: str,
) -> dict[str, Any]:
    card_details = _family_card_details(family)
    grounding = _grounding_labels(screening_status)
    names = _accepted_operation_names(family)
    legal_masks = [
        [dimension.value for dimension in combination]
        for count in range(1, len(family.principal_dimensions))
        for combination in itertools.combinations(family.principal_dimensions, count)
    ]
    eligible_masks = [
        [dimension.value for dimension in mask]
        for mask in family.eligible_o2_masks
    ]
    legal_mask_keys = {tuple(mask) for mask in legal_masks}
    eligible_mask_keys = {tuple(mask) for mask in eligible_masks}
    if not eligible_mask_keys or not eligible_mask_keys <= legal_mask_keys:
        raise RuntimeError(f"invalid curated O2 masks for {family.family_id}")
    if tuple(item.value for item in family.o2_unresolved) not in eligible_mask_keys:
        raise RuntimeError(f"selected O2 mask is not eligible for {family.family_id}")
    summary: dict[str, Any] = {
        "concept_id": family.family_id,
        "family_id": family.family_id,
        "dataset_id": family.dataset_id,
        "title": family.title,
        "scientific_concept": family.concept,
        "concept": family.concept,
        "status": PILOT_STATUS,
        "scientific_grounding": grounding["construction_grounding"],
        "formal_scientific_review": grounding["formal_review"],
        "evidence_ids": list(family.evidence_ids),
        "principal_dimensions": [item.value for item in family.principal_dimensions],
        "o2_unresolved_dimensions": [item.value for item in family.o2_unresolved],
        "mathematically_legal_o2_masks": legal_masks,
        "eligible_o2_masks": eligible_masks,
        "o2_mask_rationale": card_details["o2_mask_rationale"],
        "accepted_complete_operationalizations": list(names),
        "candidate_accepted_o_branches": list(names),
        "execution_results": {name: _branch_results(family, results, name) for name in names},
        "scientific_coherence": family.scientific_coherence,
        "o_role_coherence": family.o_role_coherence,
        "ambiguity_quality": family.ambiguity_quality,
        "grounding_assessment": family.grounding_assessment,
        "screening_status": screening_status,
        "o_role_audit": _role_audit(family),
        **card_details,
    }
    if family.family_id == "kitchen_concentration_heterogeneity":
        divergent = results[names[0]]["field_name"] != results[names[1]]["field_name"]
    else:
        divergent = results[names[0]]["point_index"] != results[names[1]]["point_index"]
    summary["branch_divergence"] = _classify_branch_divergence(same_outcome=not divergent)
    _write_json(output_root / "concept_design_cards" / f"{family.family_id}.json", summary)
    _write_markdown(output_root / "concept_design_cards" / f"{family.family_id}.md", [
        f"# {family.title}",
        "",
        f"Status: **{PILOT_STATUS}**.",
        "",
        f"- Concept: {family.concept}",
        f"- Concept ID: `{summary['concept_id']}`",
        f"- Evidence IDs: {', '.join(f'`{item}`' for item in family.evidence_ids)}",
        f"- Principal O dimensions: {', '.join(f'`{item.value}`' for item in family.principal_dimensions)}",
        f"- O2 unresolved mask: {', '.join(f'`{item.value}`' for item in family.o2_unresolved)}",
        f"- O2 mask rationale: {summary['o2_mask_rationale']}",
        f"- Scientifically eligible O2 masks: {summary['eligible_o2_masks']}",
        f"- Executed branch divergence: **{summary['branch_divergence']}**",
        f"- Scientific coherence: {summary['scientific_coherence']}",
        f"- O-role coherence: {summary['o_role_coherence']}",
        f"- Ambiguity quality: {summary['ambiguity_quality']}",
        f"- Grounding assessment: {summary['grounding_assessment']}",
        f"- Screening status: **{summary['screening_status']}**",
        f"- Formal scientific review: **{summary['formal_scientific_review']}**",
        "",
        "## Operationalization role audit",
        "",
        "| Scientific decision | O dimension | Why this role is correct |",
        "|---|---|---|",
        *[
            f"| {item['scientific_decision']} | `{item['dimension']}` | {item['reason']} |"
            for item in summary["o_role_audit"]
        ],
        "",
        "## Scientific construction basis",
        "",
        f"{summary['why_scientifically_meaningful']}",
        "",
        f"- Scientific-question family: {summary['scientific_question_family']}",
        f"- Source/literature grounding: {summary['source_literature_grounding']}",
        f"- Observable data requirements: {summary['observable_data_requirements']}",
        f"- Available dataset support: {summary['available_dataset_support']}",
        f"- Defensible dimension choices: {summary['defensible_choices_per_dimension']}",
        f"- F1 finding goal: {summary['f1_finding_goal']}",
        f"- F2 finding goal: {summary['f2_finding_goal']}",
        f"- Expected core finding types: {', '.join(summary['expected_core_finding_types'])}",
        f"- Possible GT-outside / qualitative findings: {summary['possible_gt_outside_or_qualitative_findings']}",
        f"- Known construction risks: {summary['known_construction_risks']}",
        "",
        "This card is a provisional construction decision, not a frozen concept taxonomy or a formal release claim.",
    ])
    return summary


def _family_card_details(family: FamilySpec) -> dict[str, Any]:
    """Author the small, reviewed rationale set; no concept ontology is inferred."""

    if family.family_id == "kitchen_turbulence_activity":
        return {
            "scientific_question_family": "Which local turbulence-activity hotspot is most prominent in the documented Kitchen flow?",
            "why_scientifically_meaningful": "The reviewed dataset exposes two standard local turbulence quantities in a convection setting; choosing which one defines activity is a substantive scientific decision rather than a speed-threshold variant.",
            "source_literature_grounding": "Kitchen op_002 documents ke and ep for this dataset; Kitchen op_003 records the supporting k-epsilon literature.",
            "observable_data_requirements": "Finite ke and ep point arrays with stored Cartesian coordinates.",
            "available_dataset_support": "Both scalar arrays and a structured spatial grid are present in the reviewed Kitchen artifact.",
            "defensible_choices_per_dimension": "feature_definition: local turbulence-activity hotspot as a stored grid point; criterion: maximum turbulent kinetic energy or maximum turbulent dissipation rate; aggregation_or_representation: exact stored grid-point coordinate.",
            "expected_o_to_g_divergence": "The two extrema occur at different stored point indices and coordinates.",
            "f1_finding_goal": "Report the selected hotspot coordinate and local turbulence quantities.",
            "f2_finding_goal": "Characterize the selected O1 hotspot with its companion local turbulence quantity.",
            "expected_core_finding_types": ["location", "quantity"],
            "possible_gt_outside_or_qualitative_findings": "A model may report local directional or spatial-pattern descriptions, but they are not formal core findings without an authored verification contract.",
            "known_construction_risks": "Stored units and relation to a particular physical turbulence model require release review.",
            "required_capabilities": {"SPATIAL_COORDINATES": True, "SCALAR_FIELD_SEMANTICS:ke": True, "SCALAR_FIELD_SEMANTICS:ep": True, "OBSERVABLE_INTERPRETATION": True},
            "optional_capabilities": {},
            "irrelevant_capabilities": ["TIME_SEQUENCE", "WALL_GEOMETRY"],
            "scientific_issues": [{"issue_id": "kitchen_turbulence_activity_semantics", "issue_type": "OBSERVABLE_SEMANTICS_MISSING", "issue_resolution_class": "EVIDENCE_GAP", "related_claim_ids": ["kitchen_turbulence_activity_target"], "description": "Stored units and relation to a particular physical turbulence model require release review.", "severity": "CRITICAL", "blocking": True, "resolution_requirement": "Confirm the scientific interpretation of ke and ep for this dataset."}],
            "o2_mask_rationale": "Leave only the turbulence selection criterion unresolved; the hotspot object and coordinate representation stay fixed so O2 isolates the substantive field choice.",
            "scientific_support_claims": [
                {"claim_id": "kitchen_turbulence_activity_context", "claim_type": "DATASET_CONTEXT", "statement": "The reviewed Kitchen artifact contains ke and ep arrays on a structured grid.", "support_status": "SUPPORTED", "supporting_source_ids": ["src_003"], "supporting_evidence_record_ids": ["op_002"], "critical": True},
                {"claim_id": "kitchen_turbulence_activity_operationalization", "claim_type": "OPERATIONALIZATION_RATIONALE", "statement": "The k-epsilon literature supports interpreting ke and ep as turbulence-activity quantities.", "support_status": "SUPPORTED", "supporting_source_ids": ["src_004"], "supporting_evidence_record_ids": ["op_003"], "critical": True},
                {"claim_id": "kitchen_turbulence_activity_target", "claim_type": "SCIENTIFIC_TARGET_VALIDITY", "statement": "A local turbulence-activity hotspot is a scientifically valid target for this case.", "support_status": "UNKNOWN", "critical": True},
                {"claim_id": "kitchen_turbulence_activity_observable", "claim_type": "OBSERVABLE_SEMANTICS", "statement": "Stored ke and ep values have the required physical interpretation and units.", "support_status": "UNKNOWN", "critical": True},
                {"claim_id": "kitchen_turbulence_activity_same_target", "claim_type": "SAME_TARGET_RATIONALE", "statement": "The alternative criteria operationalize the same turbulence-activity target.", "support_status": "UNKNOWN", "critical": True},
                {"claim_id": "kitchen_turbulence_activity_findings", "claim_type": "FINDING_VERIFIABILITY", "statement": "The hotspot coordinate and companion quantities can be adjudicated from the data.", "support_status": "UNKNOWN", "critical": True},
            ],
            "claim_support_links": [
                {"claim_id": "kitchen_turbulence_activity_context", "evidence_record_id": "op_002", "source_id": "src_003", "support_relation": "DIRECT", "supported_statement": "The Kitchen artifact exposes ke and ep arrays."},
                {"claim_id": "kitchen_turbulence_activity_operationalization", "evidence_record_id": "op_003", "source_id": "src_004", "support_relation": "BACKGROUND", "supported_statement": "The reference supports the turbulence-activity operationalization."},
            ],
        }
    if family.family_id == "kitchen_concentration_heterogeneity":
        return {
            "scientific_question_family": "Which documented gas-concentration field is spatially most heterogeneous in the Kitchen flow?",
            "why_scientifically_meaningful": "Mixture heterogeneity is supported by documented gas-concentration observables; selecting a normalized spread statistic changes the scientifically selected field rather than merely formatting one answer.",
            "source_literature_grounding": "Kitchen op_002 documents c-prefixed gas-concentration fields; Kitchen op_004 records mixture-characteristics grounding for concentration spread.",
            "observable_data_requirements": "Finite c-prefixed concentration arrays and positive structured-cell volumes.",
            "available_dataset_support": "The Kitchen grid supplies c1 and c11-c17 plus physical cell volumes computed from the reviewed VTK grid.",
            "defensible_choices_per_dimension": "feature_definition: documented c-prefixed concentration arrays; property_measure: volume-weighted coefficient of variation or mean-normalized volume-weighted interdecile spread; aggregation_or_representation: stored field identifier.",
            "expected_o_to_g_divergence": "The two stable dispersion measures select c12 and c13, respectively.",
            "f1_finding_goal": "Report the selected concentration-field identifier and its heterogeneity value.",
            "f2_finding_goal": "Characterize the selected O1 concentration distribution using its weighted quantiles.",
            "expected_core_finding_types": ["existence_or_identity", "quantity"],
            "possible_gt_outside_or_qualitative_findings": "A model may describe a field as localized or broadly mixed, but those labels stay outside core GT without a separate explicit criterion.",
            "known_construction_risks": "Species labels, concentration units, and scientific preference among dispersion measures need release review; zero medians must not be used as a normalizer.",
            "required_capabilities": {"SPATIAL_COORDINATES": True, "SCALAR_FIELD_SEMANTICS": True, "FIELD_COMPARABILITY": True, "VOLUME_WEIGHTING_JUSTIFICATION": True, "OBSERVABLE_INTERPRETATION": True},
            "optional_capabilities": {},
            "irrelevant_capabilities": ["TIME_SEQUENCE", "WALL_GEOMETRY"],
            "scientific_issues": [{"issue_id": "kitchen_concentration_heterogeneity_comparability", "issue_type": "FIELD_COMPARABILITY_UNRESOLVED", "issue_resolution_class": "SCIENTIFIC_AMBIGUITY", "related_claim_ids": ["kitchen_concentration_heterogeneity_target"], "description": "Species labels, concentration units, and scientific preference among dispersion measures need release review.", "severity": "CRITICAL", "blocking": True, "resolution_requirement": "Confirm that the compared c-prefixed fields have a common scientific interpretation."}],
            "o2_mask_rationale": "Leave the heterogeneity measure unresolved while fixing the documented field set and volume-weighted representation; this yields distinct accepted G(O) branches.",
            "scientific_support_claims": [
                {"claim_id": "kitchen_concentration_heterogeneity_context", "claim_type": "DATASET_CONTEXT", "statement": "The Kitchen grid supplies comparable c-prefixed concentration arrays.", "support_status": "SUPPORTED", "supporting_source_ids": ["src_003"], "supporting_evidence_record_ids": ["op_002"], "critical": True},
                {"claim_id": "kitchen_concentration_heterogeneity_operationalization", "claim_type": "OPERATIONALIZATION_RATIONALE", "statement": "Mixture-characteristics literature supports normalized concentration spread measures.", "support_status": "SUPPORTED", "supporting_source_ids": ["src_005"], "supporting_evidence_record_ids": ["op_004"], "critical": True},
                {"claim_id": "kitchen_concentration_heterogeneity_target", "claim_type": "SCIENTIFIC_TARGET_VALIDITY", "statement": "Spatial concentration heterogeneity is a valid target for this case.", "support_status": "UNKNOWN", "critical": True},
                {"claim_id": "kitchen_concentration_heterogeneity_observable", "claim_type": "OBSERVABLE_SEMANTICS", "statement": "Species labels and units make the c-prefixed fields scientifically comparable.", "support_status": "UNKNOWN", "critical": True},
                {"claim_id": "kitchen_concentration_heterogeneity_same_target", "claim_type": "SAME_TARGET_RATIONALE", "statement": "The spread measures operationalize the same concentration-heterogeneity target.", "support_status": "UNKNOWN", "critical": True},
                {"claim_id": "kitchen_concentration_heterogeneity_findings", "claim_type": "FINDING_VERIFIABILITY", "statement": "The selected field and its spread value can be adjudicated from the data.", "support_status": "UNKNOWN", "critical": True},
            ],
            "claim_support_links": [
                {"claim_id": "kitchen_concentration_heterogeneity_context", "evidence_record_id": "op_002", "source_id": "src_003", "support_relation": "DIRECT", "supported_statement": "The Kitchen artifact exposes c-prefixed fields."},
                {"claim_id": "kitchen_concentration_heterogeneity_operationalization", "evidence_record_id": "op_004", "source_id": "src_005", "support_relation": "BACKGROUND", "supported_statement": "The reference supports concentration spread measures."},
            ],
        }
    return {
        "scientific_question_family": "Which stored-density feature is most prominent in the annular-combustor flow field?",
        "why_scientifically_meaningful": "The reviewed combustor material uses density for dataset-specific structure visualization. High density and strongest density change are distinct physical feature definitions with different selected points.",
        "source_literature_grounding": "Combustor op_002 and op_003 document density isosurface use for this exact dataset.",
        "observable_data_requirements": "Finite stored Density values, structured coordinates, and a reproducible point-gradient computation.",
        "available_dataset_support": "The PLOT3D reader exposes Density and coordinates; vtkGradientFilter provides a recorded deterministic gradient calculation.",
        "defensible_choices_per_dimension": "feature_definition: local density feature as a stored grid point; criterion: maximum density or maximum vtkGradientFilter density-gradient magnitude; aggregation_or_representation: exact stored grid-point coordinate.",
        "expected_o_to_g_divergence": "Maximum density and maximum density-gradient magnitude occur at different stored point indices and coordinates.",
        "f1_finding_goal": "Report the selected density-feature coordinate and defining density quantity.",
        "f2_finding_goal": "Characterize the selected O1 density feature with the companion local gradient quantity.",
        "expected_core_finding_types": ["location", "quantity"],
        "possible_gt_outside_or_qualitative_findings": "Interpretations such as flame structure, stratification, or flow direction are excluded because they are not supported by the reviewed context.",
        "known_construction_risks": "Density units, gradient sensitivity, and the scientific role of density features require formal domain review.",
        "required_capabilities": {"SPATIAL_COORDINATES": True, "SCALAR_FIELD_SEMANTICS:Density": True, "DERIVABLE_SPATIAL_GRADIENT": True, "OBSERVABLE_INTERPRETATION": True},
        "optional_capabilities": {},
        "irrelevant_capabilities": ["TIME_SEQUENCE", "WALL_GEOMETRY", "REFERENCE_DIRECTION"],
        "scientific_issues": [{"issue_id": "combustor_density_features_gradient", "issue_type": "OBSERVABLE_SEMANTICS_MISSING", "issue_resolution_class": "EVIDENCE_GAP", "related_claim_ids": ["combustor_density_features_target"], "description": "Density units, gradient sensitivity, and the scientific role of density features require formal domain review.", "severity": "CRITICAL", "blocking": True, "resolution_requirement": "Provide independent scientific support for the gradient-based branch."}],
        "o2_mask_rationale": "Leave only the density-feature selection criterion unresolved and fix the point object and representation, exposing the high-density versus strong-change distinction without threshold tuning.",
        "scientific_support_claims": [
            {"claim_id": "combustor_density_features_context", "claim_type": "DATASET_CONTEXT", "statement": "The reviewed Combustor artifact exposes stored Density values and coordinates.", "support_status": "SUPPORTED", "supporting_source_ids": ["src_002"], "supporting_evidence_record_ids": ["op_002"], "critical": True},
            {"claim_id": "combustor_density_features_target", "claim_type": "SCIENTIFIC_TARGET_VALIDITY", "statement": "A stored-density feature is a valid target for this case.", "support_status": "UNKNOWN", "critical": True},
            {"claim_id": "combustor_density_features_observable", "claim_type": "OBSERVABLE_SEMANTICS", "statement": "Density units and gradient sensitivity support the proposed feature interpretation.", "support_status": "UNKNOWN", "critical": True},
            {"claim_id": "combustor_density_features_operationalization", "claim_type": "OPERATIONALIZATION_RATIONALE", "statement": "The reviewed material supports density and density-gradient feature choices.", "support_status": "UNKNOWN", "critical": True},
            {"claim_id": "combustor_density_features_same_target", "claim_type": "SAME_TARGET_RATIONALE", "statement": "The density and gradient criteria operationalize the same target.", "support_status": "UNKNOWN", "critical": True},
            {"claim_id": "combustor_density_features_findings", "claim_type": "FINDING_VERIFIABILITY", "statement": "The selected point and quantity can be adjudicated from the data.", "support_status": "UNKNOWN", "critical": True},
        ],
        "claim_support_links": [
            {"claim_id": "combustor_density_features_context", "evidence_record_id": "op_002", "source_id": "src_002", "support_relation": "DIRECT", "supported_statement": "The Combustor artifact exposes the documented density field."},
        ],
    }


def _validate_family_cases(cases: list[dict[str, Any]], family_spec: FamilySpec | None = None) -> None:
    by_suffix = {item["case_id"].rsplit("_", 2)[-2] + "_" + item["case_id"].rsplit("_", 1)[-1]: item for item in cases}
    first = by_suffix["o1_f1"]
    if len(by_suffix) != len(cases):
        raise RuntimeError("duplicate condition within family")
    open_case = by_suffix.get("o1_f2")
    if open_case is None:
        open_case = first
    if "o1_f2" in by_suffix and first["question"] == open_case["question"]:
        raise RuntimeError("O1-F1 and O1-F2 must have different model-visible questions")
    if first["metadata"].principal_operationalization_dimensions != open_case["metadata"].principal_operationalization_dimensions:
        raise RuntimeError("O1-F1 and O1-F2 principal operationalization dimensions drifted")
    if first["metadata"].unresolved_operationalization_dimensions != open_case["metadata"].unresolved_operationalization_dimensions:
        raise RuntimeError("O1-F1 and O1-F2 unresolved operationalization dimensions drifted")
    if first["question_details"]["dimension_clauses"] != open_case["question_details"]["dimension_clauses"]:
        raise RuntimeError("O1-F1 and O1-F2 resolved operationalization clauses drifted")
    for suffix in ("o2_f1", "o3_f1"):
        if suffix not in by_suffix:
            continue
        other = by_suffix[suffix]
        ControlledCaseFamilyValidator.validate_pair(first["metadata"], other["metadata"], axis="operationalization", left_context_selection=first["selection"], right_context_selection=other["selection"])
    if "o1_f2" in by_suffix:
        ControlledCaseFamilyValidator.validate_pair(first["metadata"], open_case["metadata"], axis="finding", left_context_selection=first["selection"], right_context_selection=open_case["selection"])
    if "o2_f1" not in by_suffix:
        return
    o2 = by_suffix["o2_f1"]["ground_truth"]
    family = first["metadata"].case_family_id
    selected_mask = tuple(by_suffix["o2_f1"]["metadata"].unresolved_operationalization_dimensions)
    unresolved = set(selected_mask)
    family_spec = family_spec or next(item for item in FAMILIES if item.family_id == family)
    if selected_mask not in family_spec.eligible_o2_masks:
        raise RuntimeError(f"selected O2 mask is not curated for {family}: {selected_mask!r}")
    if not selected_mask or not set(selected_mask) < set(by_suffix["o2_f1"]["metadata"].principal_operationalization_dimensions):
        raise RuntimeError(f"selected O2 mask is not a non-empty proper subset for {family}")
    resolved = set(by_suffix["o2_f1"]["metadata"].principal_operationalization_dimensions) - unresolved
    for dimension in resolved:
        statements = {
            next(decision.statement for decision in bundle.decisions if decision.dimension == dimension)
            for bundle in o2.acceptable_operationalizations
        }
        if len(statements) != 1:
            raise RuntimeError(f"O2 resolved dimension drift: {dimension.value}")


def _validate_generated_screening_consistency(
    cards: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    authoritative_status: dict[str, str],
) -> None:
    """Fail closed when any generated family artifact contradicts screening."""

    for card in cards:
        family_id = card["family_id"]
        expected = authoritative_status[family_id]
        expected_grounding = _grounding_labels(expected)
        if card["screening_status"] != expected:
            raise RuntimeError(
                f"design-card screening mismatch for {family_id}: "
                f"expected={expected}, actual={card['screening_status']}"
            )
        if (
            card["scientific_grounding"] != expected_grounding["construction_grounding"]
            or card["formal_scientific_review"] != expected_grounding["formal_review"]
        ):
            raise RuntimeError(f"design-card grounding mismatch for {family_id}")
    for case in cases:
        family_id = case["metadata"].case_family_id
        expected = authoritative_status[family_id]
        expected_grounding = _grounding_labels(expected)
        status = _read_json(Path(case["case_dir"]) / "construction_pilot_status.json")
        if status["family_screening_status"] != expected:
            raise RuntimeError(
                f"case screening mismatch for {case['case_id']}: "
                f"expected={expected}, actual={status['family_screening_status']}"
            )
        if (
            status["scientific_grounding"] != expected_grounding["construction_grounding"]
            or status["formal_scientific_review"] != expected_grounding["formal_review"]
        ):
            raise RuntimeError(f"case grounding mismatch for {case['case_id']}")


def _write_o2_coverage_diagnostic(cards: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    """Record the small Phase 1 O2 design audit without creating a registry."""

    mask_size_counts: dict[str, int] = {}
    dimension_counts: dict[str, int] = {}
    families: list[dict[str, Any]] = []
    for card in cards:
        mask = list(card["o2_unresolved_dimensions"])
        mask_size_counts[str(len(mask))] = mask_size_counts.get(str(len(mask)), 0) + 1
        for dimension in mask:
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
        families.append(
            {
                "family_id": card["family_id"],
                "dataset_id": card["dataset_id"],
                "principal_dimensions": card["principal_dimensions"],
                "eligible_o2_masks": card["eligible_o2_masks"],
                "selected_o2_mask": mask,
                "rationale": card["o2_mask_rationale"],
            }
        )
    diagnostic = {
        "status": "PROVISIONAL CONSTRUCTION DIAGNOSTIC",
        "families": families,
        "selected_unresolved_dimension_counts": dimension_counts,
        "selected_mask_size_counts": mask_size_counts,
        "coverage_conclusion": (
            "Each selected O2 case leaves one nonempty proper subset unresolved. "
            "The pilot intentionally demonstrates one-dimension O2 masks only; it does not claim a balance algorithm."
        ),
    }
    _write_json(output_root / "o2_coverage_diagnostic.json", diagnostic)
    _write_markdown(output_root / "o2_coverage_diagnostic.md", [
        "# O2 Mask Design Diagnostic",
        "",
        "Status: **PROVISIONAL CONSTRUCTION DIAGNOSTIC**.",
        "",
        "| Family | Principal dimensions | Selected unresolved mask | Eligible nonempty proper masks | Rationale |",
        "|---|---|---|---|---|",
        *[
            "| `{family_id}` | {principal} | {selected} | {eligible} | {rationale} |".format(
                family_id=item["family_id"],
                principal=", ".join(f"`{value}`" for value in item["principal_dimensions"]),
                selected=", ".join(f"`{value}`" for value in item["selected_o2_mask"]),
                eligible="; ".join(", ".join(f"`{value}`" for value in mask) for mask in item["eligible_o2_masks"]),
                rationale=item["rationale"],
            )
            for item in families
        ],
        "",
        diagnostic["coverage_conclusion"],
    ])
    return diagnostic


def _finding_value_type(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return "list[numeric]"
        if all(isinstance(item, str) for item in value):
            return "list[string]"
        return "list[mixed]"
    return type(value).__name__


def _verification_mode(finding: ReferenceFinding) -> str:
    verification = finding.verification
    if verification is not None and verification.spatial_tolerance is not None:
        return "spatial_euclidean"
    if verification is not None and (
        verification.absolute_tolerance is not None
        or verification.relative_tolerance is not None
    ):
        return "scalar_tolerance"
    if isinstance(finding.value, int) and not isinstance(finding.value, bool):
        return "exact_discrete_numeric"
    return "semantic_only"


def _write_verification_audit(cases: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_numeric_rules: list[str] = []
    for case in cases:
        for branch in case["ground_truth"].findings_by_operationalization:
            for finding in branch.findings:
                value_type = _finding_value_type(finding.value)
                mode = _verification_mode(finding)
                rule = finding.verification.model_dump(mode="json") if finding.verification is not None else None
                numeric_value = value_type in {"int", "float", "list[numeric]"}
                if numeric_value and mode == "semantic_only":
                    missing_numeric_rules.append(finding.finding_id)
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "operationalization_id": branch.operationalization_id,
                        "finding_id": finding.finding_id,
                        "value_type": value_type,
                        "verification_mode": mode,
                        "verification": rule,
                        "evidence_ids": list(finding.evidence_ids),
                    }
                )
    if missing_numeric_rules:
        raise RuntimeError(
            "numeric construction findings lack deterministic verification: "
            + ", ".join(missing_numeric_rules)
        )
    audit = {
        "status": "CONSTRUCTION VERIFICATION AUDIT",
        "finding_count": len(rows),
        "rows": rows,
        "numeric_findings_without_deterministic_rule": missing_numeric_rules,
        "conclusion": "Every numeric finding in the materialized Phase 1 GT has an explicit supported deterministic rule.",
    }
    _write_json(output_root / "verification_audit.json", audit)
    _write_markdown(output_root / "verification_audit.md", [
        "# Numeric Finding Verification Audit",
        "",
        "Status: **CONSTRUCTION VERIFICATION AUDIT**.",
        "",
        "| Case | Finding | Value type | Verification mode | Tolerance / rule | Evidence |",
        "|---|---|---|---|---|---|",
        *[
            "| `{case_id}` | `{finding_id}` | `{value_type}` | `{verification_mode}` | `{verification}` | {evidence} |".format(
                case_id=row["case_id"],
                finding_id=row["finding_id"],
                value_type=row["value_type"],
                verification_mode=row["verification_mode"],
                verification=json.dumps(row["verification"], ensure_ascii=False) if row["verification"] is not None else "semantic-only (non-numeric)",
                evidence=", ".join(f"`{value}`" for value in row["evidence_ids"]) or "—",
            )
            for row in rows
        ],
        "",
        audit["conclusion"],
    ])
    return audit


def _write_f1_gt_coverage_audit(cases: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    """Persist the construction-only F1 request versus CORE-GT audit."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        metadata = case["metadata"]
        if metadata.finding_openness != FindingOpenness.BOUNDED:
            continue
        required = list(case["question_details"]["f1_properties"])
        for branch in case["ground_truth"].findings_by_operationalization:
            branch_mapping = case["property_mapping"].get(branch.operationalization_id, {})
            core = sorted(_mapped_core_property_ids(branch, branch_mapping))
            rows.append({
                "case_id": case["case_id"],
                "operationalization_id": branch.operationalization_id,
                "requested_f1_properties": required,
                "core_gt_properties": core,
                "exact_match": set(required) == set(core),
            })
    if not all(row["exact_match"] for row in rows):
        raise RuntimeError("F1 requested-property exact-match audit failed")
    audit = {
        "status": "CONSTRUCTION F1/CORE-GT COVERAGE AUDIT",
        "rows": rows,
        "conclusion": "Every bounded F1 request has exactly the same authored scientific-property set as CORE GT in every applicable G(O) branch.",
    }
    _write_json(output_root / "f1_gt_coverage_audit.json", audit)
    _write_markdown(output_root / "f1_gt_coverage_audit.md", [
        "# F1 Request to CORE Ground-Truth Coverage Audit",
        "",
        "Status: **CONSTRUCTION F1/CORE-GT COVERAGE AUDIT**.",
        "",
        "| Case | G(O) branch | Model-visible F1 properties | CORE GT properties | Match |",
        "|---|---|---|---|---|",
        *[
            "| `{case_id}` | `{operationalization_id}` | {requested} | {core} | **{coverage}** |".format(
                case_id=row["case_id"],
                operationalization_id=row["operationalization_id"],
                requested=", ".join(f"`{value}`" for value in row["requested_f1_properties"]),
                core=", ".join(f"`{value}`" for value in row["core_gt_properties"]),
                coverage="PASS" if row["exact_match"] else "FAIL",
            )
            for row in rows
        ],
        "",
        audit["conclusion"],
    ])
    return audit


def run_phase_1(datasets_root: Path, output_root: Path, *, dataset_ids: list[str] | None = None) -> dict[str, Any]:
    if dataset_ids is not None:
        return _run_authored_datasets(datasets_root, output_root, dataset_ids)
    profiles = _capability_profiles(datasets_root)
    screening = _screening_matrix()
    screening_status_by_family = _screening_status_by_family(screening)
    _write_json(output_root / "dataset_capability_profiles.json", {"status": PILOT_STATUS, "profiles": profiles})
    _write_markdown(output_root / "dataset_capability_profiles.md", [
        "# Dataset Capability Profiles",
        "",
        f"Status: **{PILOT_STATUS}**.",
        "",
        "This is an evidence audit of the seven reviewed datasets. It records available variables and approved context, rather than inferring missing geometric, boundary, or physical semantics.",
        "",
        "| Dataset | Variables | Temporal | Geometry / boundary support | Major limitation |",
        "|---|---|---|---|---|",
        *[
            "| `{dataset_id}` | {variables} | {temporal} | {geometry}; {boundaries} | {limitation} |".format(
                dataset_id=item["dataset_id"],
                variables=", ".join(f"`{value['name']}`" for value in item["available_variables"]),
                temporal=item["temporal_support"],
                geometry="geometry: " + (", ".join(item["geometry_information"]) if isinstance(item["geometry_information"], list) else item["geometry_information"]),
                boundaries="boundaries: " + ("approved" if isinstance(item["boundary_information"], list) else item["boundary_information"]),
                limitation=item["limitation_or_basis"],
            )
            for item in profiles
        ],
    ])
    _write_json(output_root / "concept_screening.json", {"status": "PROVISIONAL", "entries": screening})
    _write_markdown(output_root / "concept_screening.md", [
        "# Sparse Dataset x Concept Screening",
        "",
        "Status labels are provisional construction decisions, not a frozen taxonomy. A candidate is selected only when all four gates pass and its authored complete operationalizations can be executed.",
        "",
        "| Dataset | Candidate concept | Data feasibility | Context feasibility | Scientific grounding | O feasibility | Overall status |",
        "|---|---|---|---|---|---|",
        *[
            "| `{dataset_id}` | `{concept_id}` | {data} | {context} | {grounding} | {operationalization} | **{status}** |".format(
                dataset_id=item["dataset_id"],
                concept_id=item["concept_id"],
                data=item["data_feasibility"],
                context=item["context_feasibility"],
                grounding=item["scientific_grounding"],
                operationalization=item["operationalization_feasibility"],
                status=item["status"],
            )
            for item in screening
        ],
        "",
        "Rejected/unsupported entries retain their stated missing semantics in `concept_screening.json`; they are not silently converted into generic scalar tasks.",
    ])
    cards: list[dict[str, Any]] = []
    all_cases: list[dict[str, Any]] = []
    for family in FAMILIES:
        results = family.analyze(_load_dataset(datasets_root / family.dataset_id))
        screening_status = screening_status_by_family[family.family_id]
        cards.append(_write_design_card(family, results, output_root, screening_status=screening_status))
        family_cases = [
            _write_case(
                family,
                datasets_root,
                output_root,
                suffix,
                results,
                screening_status=screening_status,
            )
            for suffix in ("o1_f1", "o2_f1", "o3_f1", "o1_f2")
        ]
        _validate_family_cases(family_cases)
        all_cases.extend(family_cases)
    _validate_generated_screening_consistency(
        cards,
        all_cases,
        screening_status_by_family,
    )
    o2_diagnostic = _write_o2_coverage_diagnostic(cards, output_root)
    f1_gt_audit = _write_f1_gt_coverage_audit(all_cases, output_root)
    verification_audit = _write_verification_audit(all_cases, output_root)
    selected = [item for item in screening if item["status"] == "STRONG_CANDIDATE"]
    selected_ids = {item["concept_id"] for item in selected}
    selected_cards = [item for item in cards if item["family_id"] in selected_ids]
    recommendation = "READY FOR LIMITED CONCEPT-EXPANSION N=1" if len(selected_cards) >= 3 else "NOT READY FOR CONCEPT-EXPANSION N=1"
    summary = {
        "status": PILOT_STATUS,
        "scientific_grounding": "DERIVED PER FAMILY FROM SCREENING MATRIX",
        "selected_family_count": len(selected_cards),
        "selected_families": [item["family_id"] for item in selected_cards],
        "materialized_candidate_family_count": len(cards),
        "materialized_candidate_families": [item["family_id"] for item in cards],
        "strong_candidate_screen_count": len(selected),
        "candidate_case_count": len(all_cases),
        "readiness_case_count": len(selected_cards) * 4,
        "o2_selected_mask_size_counts": o2_diagnostic["selected_mask_size_counts"],
        "o2_selected_unresolved_dimension_counts": o2_diagnostic["selected_unresolved_dimension_counts"],
        "screening_status_source": "concept_screening.json entries (authored screening matrix)",
        "f1_exact_match_branch_count": len(f1_gt_audit["rows"]),
        "f1_exact_match_failures": sum(not row["exact_match"] for row in f1_gt_audit["rows"]),
        "verification_finding_count": verification_audit["finding_count"],
        "model_evaluation": "NOT RUN",
        "formal_release": "NOT ELIGIBLE FROM THIS ARTIFACT",
        "recommendation": recommendation,
    }
    _write_json(output_root / "concept_expansion_summary.json", summary)
    _write_markdown(output_root / "concept_expansion_summary.md", [
        "# Concept Expansion Construction Pilot - Phase 1",
        "",
        f"Status: **{PILOT_STATUS}**.",
        "",
        "## Result",
        "",
        f"- Strong Dataset x Concept candidates: `{len(selected)}`",
        f"- Families eligible for real-model readiness: `{len(selected_cards)}`",
        f"- Materialized candidate families for construction audit: `{len(cards)}`",
        f"- Candidate case artifacts with executed G(O): `{len(all_cases)}`",
        f"- Candidate cases belonging to readiness-eligible families: `{len(selected_cards) * 4}`",
        f"- Evaluated-model calls: **NOT RUN**",
        f"- Formal release: **NOT ELIGIBLE FROM THIS ARTIFACT**",
        f"- Decision: **{recommendation}**",
        "",
        "## Materialized Candidate Families and Executed O -> G(O)",
        "",
        "| Family | Dataset | Evidence | Accepted complete O | Observed G(O) difference |",
        "|---|---|---|---|---|",
        *[
            "| `{family_id}` | `{dataset_id}` | {evidence} | {operations} | **{difference}** |".format(
                family_id=item["family_id"],
                dataset_id=item["dataset_id"],
                evidence=", ".join(f"`{value}`" for value in item["evidence_ids"]),
                operations=", ".join(f"`{value}`" for value in item["accepted_complete_operationalizations"]),
                difference=item["branch_divergence"],
            )
            for item in cards
        ],
        "",
        "## O2 Mask Design",
        "",
        "| Family | Principal dimensions | Scientifically ambiguous dimensions | Eligible O2 masks | Selected mask | Why this is a controlled ambiguity |",
        "|---|---|---|---|---|---|",
        *[
            "| `{family_id}` | {principal} | {mask} | {eligible} | {selected} | {rationale} |".format(
                family_id=item["family_id"],
                principal=", ".join(f"`{value}`" for value in item["principal_dimensions"]),
                mask=", ".join(f"`{value}`" for value in item["o2_unresolved_dimensions"]),
                eligible="; ".join(", ".join(f"`{value}`" for value in mask) for mask in item["eligible_o2_masks"]),
                selected=", ".join(f"`{value}`" for value in item["o2_unresolved_dimensions"]),
                rationale=item["o2_mask_rationale"],
            )
            for item in cards
        ],
        "",
        f"Selected mask sizes: `{o2_diagnostic['selected_mask_size_counts']}`. No mask-balance algorithm is claimed.",
        "",
        "## Finding Diversity",
        "",
        "| Family | F1 target | F2 target | Core Finding types |",
        "|---|---|---|---|",
        *[
            "| `{family_id}` | {f1} | {f2} | {types} |".format(
                family_id=item["family_id"],
                f1=item["f1_finding_goal"],
                f2=item["f2_finding_goal"],
                types=", ".join(f"`{value}`" for value in item["expected_core_finding_types"]),
            )
            for item in cards
        ],
        "",
        "The candidate branches use location and quantity findings for the Kitchen turbulence and Combustor density families, and identity plus quantity findings for Kitchen concentration heterogeneity. Each O1-F2 artifact adds selective supporting characterization findings while preserving the O1 operationalization.",
        "",
        "## F1 Requested Properties versus CORE Ground Truth",
        "",
        "The following audit is construction-level: each bounded F1 request is generated from the same minimal property specification and must exactly match every applicable CORE G(O) branch; both missing and unexpected properties fail construction.",
        "",
        "| Case | G(O) branch | Requested F1 properties | CORE GT properties | Match |",
        "|---|---|---|---|---|",
        *[
            "| `{case_id}` | `{operationalization_id}` | {requested} | {core} | **{coverage}** |".format(
                case_id=row["case_id"],
                operationalization_id=row["operationalization_id"],
                requested=", ".join(f"`{value}`" for value in row["requested_f1_properties"]),
                core=", ".join(f"`{value}`" for value in row["core_gt_properties"]),
                coverage="PASS" if row["exact_match"] else "FAIL",
            )
            for row in f1_gt_audit["rows"]
        ],
        "",
        "See `f1_gt_coverage_audit.md` for the complete per-branch record.",
        "",
        "## O2 Resolved-Dimension Invariance and Eligibility",
        "",
        "Each selected mask is checked for non-empty proper-subset legality, membership in its family’s explicit eligible mask set, and invariant decision statements on every resolved dimension. Mathematical legality is not treated as a universal scientific eligibility rule; O2 mask policy remains provisional.",
        "",
        f"Selected mask sizes: `{o2_diagnostic['selected_mask_size_counts']}`; selected unresolved dimensions: `{o2_diagnostic['selected_unresolved_dimension_counts']}`.",
        "",
        "## Screening and Reproducibility",
        "",
        "Screening status is sourced only from the authored `concept_screening.json` matrix; `FamilySpec` does not independently control readiness. Construction floats are serialized canonically to 14 significant digits, while branch selection uses the unrounded computed values. The persisted Phase 1 tree is byte-compared with a fresh build by the construction-pilot regression test.",
        "",
        "| Family | Authoritative screening status | Generated grounding status | Formal-review/readiness status |",
        "|---|---|---|---|",
        *[
            "| `{family_id}` | **{screening}** | `{grounding}` | `{review}` |".format(
                family_id=item["family_id"],
                screening=item["screening_status"],
                grounding=item["scientific_grounding"],
                review=item["formal_scientific_review"],
            )
            for item in cards
        ],
        "",
        "## Construction Test Status",
        "",
        "Generation-time checks: **PASS** for F1/CORE-GT coverage, O2 mask legality and membership, resolved-dimension invariance, and numeric verification coverage. The persisted-vs-fresh byte-identity check is exercised by `test_phase_1_persisted_artifacts_match_fresh_build`; full repository test results are reported with the execution handoff.",
        "",
        "## Method-Generalization Finding",
        "",
        "The construction path generalizes across scalar extrema, a volume-weighted distributional comparison, and a VTK-derived spatial gradient. Scientific Operationalization plurality is required; consequential G(O) divergence is preferred as an informative construction diagnostic, not imposed as an absolute validity gate. It does not generalize by assuming unreviewed direction, boundary, surface, temporal, or variable semantics; unsupported Dataset x Concept pairs remain rejected in the screening matrix.",
        "",
        "## Frozen and Provisional Boundaries",
        "",
        "Frozen: the existing case schemas, O1/O2/O3 and F1/F2 contracts, source/evidence validation, O-to-G execution requirement, and no-model-call Phase 1 boundary.",
        "",
        "Provisional: the materialized candidate families, their evidence adequacy for formal release, screening labels, accepted-branch scientific adequacy, and any later N=1 model evaluation. This output validates construction feasibility only; it does not establish a formal benchmark release.",
        "",
        "## Scientific Family Review",
        "",
        "| Family | Scientific coherence | O-role coherence | O ambiguity quality | Grounding | Phase 1 status |",
        "|---|---|---|---|---|---|",
        *[
            "| `{family_id}` | {coherence} | {roles} | {ambiguity} | {grounding} | **{status}** |".format(
                family_id=item["family_id"],
                coherence=item["scientific_coherence"],
                roles=item["o_role_coherence"],
                ambiguity=item["ambiguity_quality"],
                grounding=item["grounding_assessment"],
                status=item["screening_status"],
            )
            for item in cards
        ],
        "",
        "## Corrective Pass Audit",
        "",
        "| Issue | Root cause | Fix | Regression protection |",
        "|---|---|---|---|",
        "| O1-F1 and O1-F2 were model-visible duplicates | Finding openness existed only in hidden metadata | O clauses are built independently from F1/F2 request clauses | Family validator compares questions and resolved O clause maps |",
        "| Kitchen concentration O1 omitted its measure | A principal dimension had no visible authored clause | O1 explicitly states cell-volume-weighted coefficient of variation | Model-visible dimension coverage check |",
        "| O2 eligibility was all legal subsets | Mathematical legality was mistaken for scientific eligibility | Curated per-family masks are explicit and selected masks are checked | Selected-mask membership/proper-subset assertions |",
        "| Numeric quantile list had no deterministic rule | Structured numeric value was treated as one semantic finding | q10, q50 and q90 are independent scalar findings with tolerances | Verification audit rejects numeric semantic-only findings |",
        "| Readiness used FamilySpec count | Candidate presence was confused with screening survival | Count only STRONG_CANDIDATE screening entries | Summary reports readiness-eligible families separately |",
        "",
        "## F1/F2 Model-Visible Audit",
        "",
        *sum(
            ([
                f"- `{item['family_id']}` O1-F1: {next(case['question'] for case in all_cases if case['case_id'] == item['family_id'] + '_o1_f1')}",
                f"- `{item['family_id']}` O1-F2: {next(case['question'] for case in all_cases if case['case_id'] == item['family_id'] + '_o1_f2')}",
                "",
            ] for item in cards),
            [],
        ),
        "## Verification Audit",
        "",
        f"Materialized findings audited: `{verification_audit['finding_count']}`. Numeric findings without an explicit deterministic rule: `{len(verification_audit['numeric_findings_without_deterministic_rule'])}`.",
        "See `verification_audit.md` for per-finding value types, verification modes, tolerances and evidence.",
        "",
        "## Readiness Decision",
        "",
        f"**{recommendation}**. Only `{len(selected_cards)}` families are STRONG_CANDIDATE under the explicit screening matrix; the Combustor family remains PROVISIONAL because the gradient branch lacks independent scientific grounding. The construction method generalizes to the materialized families without evaluator or metric changes, but the pragmatic Phase 1 heuristic of approximately three strong families is not met; this heuristic is not a frozen benchmark axiom.",
    ])
    return summary


def _run_authored_datasets(datasets_root: Path, output_root: Path, dataset_ids: list[str]) -> dict[str, Any]:
    """Use the same materializers/validators for dataset-local family designs."""
    import hashlib
    from flowintentbench.construction_recipes import execute_recipes

    if not dataset_ids or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("explicit dataset IDs must be nonempty and unique")
    all_cases, family_rows, errors = [], [], []
    seen_families = set()
    for dataset_id in dataset_ids:
        dataset_dir = datasets_root / dataset_id
        if dataset_dir.resolve().parent != datasets_root.resolve():
            raise ValueError("dataset ID must identify a direct child of datasets root")
        definitions = _read_json(dataset_dir / "construction/families.json")["families"]
        manifest = _read_json(dataset_dir / "dataset_manifest.json")
        inputs = []
        for record in manifest["files"]:
            path = datasets_root / record["path"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if record.get("checksum_algorithm") != "sha256" or digest != record.get("checksum"):
                raise ValueError(f"input checksum mismatch: {path}")
            inputs.append({"path": record["path"], "sha256": digest})
        dataset = _load_dataset(dataset_dir)
        for definition in definitions:
            family_id = definition["family_id"]
            if family_id in seen_families:
                raise ValueError(f"duplicate family ID: {family_id}")
            seen_families.add(family_id)
            dimensions = tuple(OperationalizationDimension(x) for x in definition["principal_dimensions"])
            unresolved = tuple(OperationalizationDimension(x) for x in definition["o2_unresolved_dimensions"])
            conditions = definition["eligible_conditions"]
            if not conditions or "o1_f1" not in conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= {"o1_f1", "o2_f1", "o3_f1", "o1_f2"}:
                raise ValueError(f"invalid authored condition membership: {family_id}")
            for operation in definition["operations"].values():
                if set(operation["clauses"]) != {d.value for d in dimensions}:
                    raise ValueError(f"incomplete operationalization: {family_id}")
            family = FamilySpec(
                family_id, dataset_id, definition["title"], definition["scientific_target"],
                tuple(definition["evidence_ids"]), dimensions, unresolved, (unresolved,),
                definition["scientific_coherence"], definition["o_role_coherence"],
                definition["ambiguity_quality"], "PENDING_SCIENTIFIC_REVIEW",
                tuple(definition["context_fact_ids"]),
                lambda data, spec=definition: execute_recipes(data, spec), definition,
            )
            try:
                results = family.analyze(dataset)
                # Numerical evidence is regenerated from the same checked bytes
                # as G(O); source documentation is never made to endorse it.
                evidence_path = dataset_dir / "construction/finding_evidence.json"
                evidence_rows = _read_json(evidence_path) if evidence_path.exists() else []
                own_ids = set(definition["finding_evidence_ids"])
                evidence_rows = [row for row in evidence_rows if row["evidence_id"] not in own_ids]
                for evidence_id in sorted(own_ids):
                    evidence_rows.append({"evidence_id": evidence_id, "dataset_id": dataset_id,
                        "evidence_type": "finding", "statement": "Executed numerical reference: " + json.dumps(results, sort_keys=True),
                        "source_id": "src_002", "locator": {"code_location": "flowintentbench/construction_recipes.py"},
                        "eligible_for_context": False, "context_facts": []})
                _write_json(evidence_path, evidence_rows)
                cases = [_write_case(family, datasets_root, output_root, suffix, results,
                                    screening_status="PROVISIONAL") for suffix in conditions]
                _validate_family_cases(cases, family)
                all_cases.extend(cases)
                family_rows.append({"family_id": family_id, "dataset_id": dataset_id,
                                    "condition_count": len(cases), "finding_archetype": definition["finding_archetype"],
                                    "status": "CONSTRUCTION_VALIDATED", "scientific_qualification": "NOT_ESTABLISHED"})
                _write_json(output_root / "execution_evidence" / f"{family_id}.json", {
                    "family_id": family_id, "inputs": inputs, "operations": definition["operations"],
                    "definition_sha256": hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
                    "executor_sha256": hashlib.sha256((ROOT / "flowintentbench/construction_recipes.py").read_bytes()).hexdigest(),
                    "results": results, "status": "EXECUTED", "scientific_qualification": "NOT_ESTABLISHED",
                })
            except (ValueError, RuntimeError, KeyError) as exc:
                errors.append({"family_id": family_id, "error": str(exc)})
    _write_f1_gt_coverage_audit(all_cases, output_root)
    _write_verification_audit(all_cases, output_root)
    rows = []
    for case in all_cases:
        directory = Path(case["case_dir"])
        row = {"case_id": case["case_id"], "family_id": case["metadata"].case_family_id,
               "dataset_id": case["metadata"].dataset_id,
               "condition": "-".join(case["case_id"].rsplit("_", 2)[-2:]).upper()}
        for filename in ("case_input", "case_construction_metadata", "case_context", "ground_truth"):
            path = directory / f"{filename}.json"
            row[f"{filename}_path"] = str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path.resolve())
            row[f"{filename}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    summary = {"status": PILOT_STATUS, "case_count": len(rows), "cases": rows,
               "family_count": len(family_rows), "families": family_rows, "errors": errors,
               "qualified_case_count": 0, "formal_release": "NOT ELIGIBLE FROM THIS ARTIFACT"}
    _write_json(output_root / "case_manifest.json", summary)
    return summary


def include_frozen_reference_cases(summary, output_root, reference_path):
    """Copy verified frozen inputs into the explicit expansion inventory."""
    import hashlib
    import shutil
    from flowintentbench.scientific_artifact_binding import artifact_sha256

    reference = _read_json(reference_path)
    rows = list(summary["cases"])
    identities = {row["case_id"] for row in rows}
    for frozen in reference["cases"]:
        case_id, dataset_id = frozen["case_id"], frozen["dataset_id"]
        if case_id in identities:
            raise ValueError(f"duplicate reference case identity: {case_id}")
        source = reference_path.parent / "cases" / dataset_id / case_id
        destination = output_root / "reference_cases" / dataset_id / case_id
        destination.mkdir(parents=True, exist_ok=True)
        metadata = _read_json(source / "case_construction_metadata.json")
        row = {
            "case_id": case_id,
            "dataset_id": dataset_id,
            "condition": frozen["condition"],
            "family_id": metadata["case_family_id"],
            "origin": "FROZEN_REFERENCE_COPY",
            "reference_manifest_sha256": hashlib.sha256(
                reference_path.read_bytes()
            ).hexdigest(),
        }
        check_names = {
            "case_input": "model_visible_case_sha256",
            "ground_truth": "ground_truth_sha256",
            "case_construction_metadata": "case_metadata_sha256",
            "scientific_semantic_projection": "semantic_contract_sha256",
        }
        for name in (
            "case_input",
            "case_construction_metadata",
            "case_context",
            "ground_truth",
            "scientific_semantic_projection",
        ):
            path = source / f"{name}.json"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            frozen_digest = (
                digest
                if name == "case_construction_metadata"
                else artifact_sha256(
                    GroundTruth.model_validate(_read_json(path)).model_dump(mode="json")
                )
                if name == "ground_truth"
                else artifact_sha256(_read_json(path))
            )
            if (
                name in check_names
                and frozen_digest != frozen["frozen_artifacts"][check_names[name]]
            ):
                raise ValueError(f"frozen artifact checksum mismatch: {path}")
            shutil.copy2(path, destination / path.name)
            row[f"{name}_path"] = str(
                (destination / path.name).resolve().relative_to(ROOT)
            )
            row[f"{name}_sha256"] = digest
        rows.append(row)
        identities.add(case_id)
    summary.update(
        cases=rows,
        case_count=len(rows),
        new_case_count=len(summary["cases"]),
        reference_case_count=len(reference["cases"]),
        family_count=len({row["family_id"] for row in rows}),
        dataset_count=len({row["dataset_id"] for row in rows}),
    )
    _write_json(output_root / "case_manifest.json", summary)
    return summary


def apply_question_revisions(summary, output_root, *, config_path=None):
    """Apply explicitly authored wording repairs after the existing fidelity audit."""
    import hashlib
    from flowintentbench.question_presentation import (
        audit_post_rewrite_semantic_fidelity,
        run_flow_expert_post_rewrite_semantic_fidelity,
        build_post_rewrite_semantic_fidelity_packet,
        post_rewrite_semantic_fidelity_instruction,
    )
    from flowintentbench.agent_profile import load_agent_profile
    from flowintentbench.review_lineage import live_review_call_matches

    profile_hash = load_agent_profile(
        "flow-scientific-reviewer-gpt-5.6-sol", repository_root=ROOT
    ).profile_sha256
    revisions = _read_json(output_root / "question_revisions.json")["revisions"]
    inventory = {row["case_id"]: row for row in summary["cases"]}
    for revision in revisions:
        row = inventory[revision["case_id"]]
        path = ROOT / row["case_input_path"]
        case_input = _read_json(path)
        projection_path = ROOT / row["scientific_semantic_projection_path"]
        if (
            hashlib.sha256(projection_path.read_bytes()).hexdigest()
            != revision["semantic_projection_sha256"]
        ):
            raise ValueError("wording revision semantic source changed")
        source_hash = hashlib.sha256(
            case_input["scientific_question"].encode()
        ).hexdigest()
        if (
            case_input["scientific_question"] != revision["scientific_question"]
            and source_hash != revision["source_question_sha256"]
        ):
            raise ValueError("wording revision source text changed")
        metadata = CaseConstructionMetadata.model_validate(
            _read_json(ROOT / row["case_construction_metadata_path"])
        )
        CaseConstructionMetadataValidator.validate(
            revision["scientific_question"], metadata
        )
        projection = _read_json(projection_path)
        review_context = {
            "case_context": _read_json(ROOT / row["case_context_path"]),
            "data_metadata": case_input["flow_data"]["data_metadata"],
        }
        packet = build_post_rewrite_semantic_fidelity_packet(
            projection,
            revision["scientific_question"],
            dataset_context=review_context,
            case_id=row["case_id"],
            condition=row["condition"],
        )
        review_path = path.parent / "presentation_revision_fidelity.json"
        result = _read_json(review_path) if review_path.is_file() else {}
        if not (
            result.get("status") == "PASS"
            and live_review_call_matches(
                result.get("review_call", {}),
                packet,
                post_rewrite_semantic_fidelity_instruction(packet),
                profile_hash,
            )
        ):
            result = audit_post_rewrite_semantic_fidelity(
                projection,
                revision["scientific_question"],
                dataset_context={
                    "case_context": _read_json(ROOT / row["case_context_path"]),
                    "data_metadata": case_input["flow_data"]["data_metadata"],
                },
                case_id=row["case_id"],
                condition=row["condition"],
                reviewer=lambda packet: run_flow_expert_post_rewrite_semantic_fidelity(
                    str(ROOT),
                    packet,
                    config_path=str(config_path) if config_path else None,
                    timeout=180,
                ),
            )
        _write_json(path.parent / "presentation_revision_fidelity.json", result)
        if result.get("status") != "PASS":
            raise ValueError(
                f"wording revision fidelity not established: {row['case_id']}"
            )
        case_input["scientific_question"] = revision["scientific_question"]
        _write_json(path, case_input)
        _write_json(path.parent / "scientific_question.json", revision)
        row.update(
            origin="FROZEN_REFERENCE_PRESENTATION_REVISION",
            case_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            presentation_revision_path=str(
                (path.parent / "presentation_revision_fidelity.json").relative_to(ROOT)
            ),
        )
        _write_json(output_root / "case_manifest.json", summary)
    return summary


def audit_constructed_portfolio(summary, output_root):
    """Use the existing diversity audit on the explicit unified case inventory."""
    import hashlib
    from collections import Counter
    from flowintentbench.diversity_audit import build_diversity_report
    from flowintentbench.dataset_capability import audit_dataset_capabilities
    from flowintentbench.case_repository import _family_index

    families = _family_index(ROOT)
    actual_families, cases, checks = {}, [], []
    for row in summary["cases"]:
        metadata = _read_json(ROOT / row["case_construction_metadata_path"])
        case_input = _read_json(ROOT / row["case_input_path"])
        family = dict(families.get(row["family_id"], {}))
        archetype = family.get("finding_archetype")
        if not archetype:
            # Legacy reference targets have reviewed explicit identities.
            target = metadata["scientific_target"].lower()
            archetype = (
                "surface_geometry"
                if "isosurface" in target
                else "spatial_distribution"
                if "heterogeneity" in target
                else "localized_region"
            )
        family.update(
            family_id=row["family_id"],
            dataset_id=row["dataset_id"],
            concept_id=family.get("concept_id", row["family_id"]),
            analysis_archetype=archetype,
            principal_dimensions=metadata["principal_operationalization_dimensions"],
            scientific_target=metadata["scientific_target"],
        )
        if row["family_id"] not in actual_families:
            actual_families[row["family_id"]] = {**family, "controlled_case_ids": []}
        actual_families[row["family_id"]]["controlled_case_ids"].append(row["case_id"])
        if row["condition"] == "O2-F1":
            actual_families[row["family_id"]]["o2_unresolved_dimensions"] = metadata[
                "unresolved_operationalization_dimensions"
            ]
        view = {
            **metadata,
            "analysis_archetype": archetype,
            "scientific_entity_type": archetype,
            "finding_requirement_contract": family.get(
                "finding_requirement_contract", {}
            ),
        }
        cases.append({**row, "metadata": view, "case_input": case_input})
        for name in (
            "case_input",
            "case_construction_metadata",
            "case_context",
            "ground_truth",
        ):
            path = ROOT / row[f"{name}_path"]
            checks.append(
                {
                    "case_id": row["case_id"],
                    "artifact": name,
                    "status": "PASS"
                    if hashlib.sha256(path.read_bytes()).hexdigest()
                    == row[f"{name}_sha256"]
                    else "FAIL",
                }
            )
    report = build_diversity_report(list(actual_families.values()), cases)
    report["condition_counts"] = dict(
        Counter(row["condition"] for row in summary["cases"])
    )
    parent_groups = {}
    for dataset_id in sorted({row["dataset_id"] for row in summary["cases"]}):
        manifest = _read_json(ROOT / "datasets" / dataset_id / "dataset_manifest.json")
        parent_groups[dataset_id] = manifest.get("provenance", {}).get(
            "parent_source_group", dataset_id
        )
    report["parent_source_groups"] = parent_groups
    report["parent_source_group_count"] = len(set(parent_groups.values()))
    report["independence_note"] = (
        "Conditions within a family and families from one source are correlated. Case count is not the number of independent source replications."
    )
    _write_json(output_root / "diversity_audit.json", report)
    _write_json(
        output_root / "artifact_integrity_audit.json",
        {
            "status": "PASS"
            if all(row["status"] == "PASS" for row in checks)
            else "FAIL",
            "checks": checks,
        },
    )
    _write_json(
        output_root / "dataset_capability_audit.json",
        audit_dataset_capabilities(ROOT, dataset_ids=list(parent_groups)),
    )
    _write_json(
        output_root / "dataset_capability_profiles.json",
        {"profiles": _capability_profiles(ROOT / "datasets", list(parent_groups))},
    )
    summary["families"] = [
        {
            "family_id": family["family_id"],
            "dataset_id": family["dataset_id"],
            "finding_archetype": family["analysis_archetype"],
            "controlled_case_ids": family["controlled_case_ids"],
            "condition_count": len(family["controlled_case_ids"]),
            "status": "CONSTRUCTION_VALIDATED",
            "scientific_qualification": "NOT_ESTABLISHED",
        }
        for family in actual_families.values()
    ]
    summary["family_count"] = len(actual_families)
    _write_json(output_root / "case_manifest.json", summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        help="construct dataset-local construction/families.json using existing materializers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "reference" / "concept_expansion_phase1",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly replace an existing reference tree during regeneration",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-portfolio", action="store_true")
    parser.add_argument("--apply-question-revisions", action="store_true")
    parser.add_argument(
        "--include-frozen-reference",
        action="store_true",
        help="include verified copies of the current frozen cases in the datasets inventory",
    )
    parser.add_argument("--review-science", action="store_true")
    parser.add_argument("--science-only", action="store_true")
    parser.add_argument(
        "--review-final",
        action="store_true",
        help="run the existing live final-wording review after construction",
    )
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="review the existing output manifest without rebuilding",
    )
    parser.add_argument("--review-workers", type=int, default=4)
    parser.add_argument("--server-config", type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if (
        output_root.exists()
        and output_root.is_relative_to(ROOT / "artifacts" / "reference")
        and not args.force
    ):
        raise SystemExit(
            f"refusing to overwrite immutable reference tree: {output_root}; use --force only for an intentional regeneration"
        )
    summary = (
        _read_json(output_root / "case_manifest.json")
        if args.review_only or args.science_only or args.audit_only
        else run_phase_1(
            args.datasets_root.resolve(), output_root, dataset_ids=args.dataset_ids
        )
    )
    if args.include_frozen_reference:
        summary = include_frozen_reference_cases(
            summary,
            output_root,
            ROOT
            / "artifacts/reference/portfolio_v7_gt_aligned_final/reference_science_baseline_manifest.json",
        )
    if args.apply_question_revisions:
        summary = apply_question_revisions(
            summary, output_root, config_path=args.server_config
        )
    if args.review_final or args.review_only:
        from flowintentbench.question_presentation import (
            review_constructed_case_manifest,
        )

        summary["wording_review"] = review_constructed_case_manifest(
            ROOT,
            output_root / "case_manifest.json",
            config_path=args.server_config,
            max_workers=args.review_workers,
        )
    if args.review_science or args.science_only:
        from flowintentbench.scientific_expert_review import review_constructed_families

        summary["scientific_review"] = review_constructed_families(
            ROOT,
            output_root / "case_manifest.json",
            config_path=args.server_config,
            max_workers=args.review_workers,
        )
    if args.audit_portfolio or args.audit_only:
        summary["portfolio_audit"] = audit_constructed_portfolio(summary, output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate and validate the seven dataset metadata bundles.

The generated ``data_metadata.json`` files contain only model-facing data
interpretation metadata. File roles, checksums, canonical reader settings and
exact PLOT3D mappings remain in ``dataset_manifest.json``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from vtkmodules.vtkIOParallel import vtkMultiBlockPLOT3DReader
from vtkmodules.vtkIOLegacy import vtkDataSetReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.manifest import (
    DatasetManifest,
    ManifestFile,
    ReaderConfiguration,
    VariableMapping,
    manifest_auxiliary_asset,
    manifest_file,
    sha256_file,
)
from flowintentbench.plot3d_reader import (
    PLOT3D_READER_NAME,
    apply_plot3d_reader_configuration,
    canonical_plot3d_reader_configuration,
)
from flowintentbench.schema import DataMetadata
from flowintentbench.vtk_reader import (
    apply_legacy_vtk_reader_configuration,
    canonical_vtk_reader_configuration,
)


DATASETS = ROOT / "datasets"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    output_dir: str
    files: tuple[tuple[str, str], ...]
    format: str
    source_urls: tuple[str, ...]
    auxiliary_assets: tuple[tuple[str, str], ...] = ()


SPECS = (
    DatasetSpec(
        "Blunt_Fin",
        "Blunt_Fin",
        (("Blunt_Fin/bluntfinxyz.bin", "grid"), ("Blunt_Fin/bluntfinq.bin", "solution")),
        "PLOT3D",
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/BluntStreamlines/",
            "https://vtk.org/doc/nightly/html/classvtkMultiBlockPLOT3DReader.html",
        ),
    ),
    DatasetSpec(
        "Combustor",
        "Combustor",
        (("Combustor/combxyz.bin", "grid"), ("Combustor/combq.bin", "solution")),
        "PLOT3D",
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/WarpCombustor/",
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/CombustorIsosurface/",
            "https://vtk.org/doc/nightly/html/classvtkMultiBlockPLOT3DReader.html",
        ),
    ),
    DatasetSpec(
        "FireFlow",
        "FireFlow",
        (("FireFlow/fire_ug.vtu", "flow_field"),),
        "VTU",
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/FireFlow/",
            "https://docs.enthought.com/mayavi/mayavi/example_fire.html",
        ),
        (("FireFlow/room_vis.wrl", "geometry"),),
    ),
    DatasetSpec(
        "NASA_LOx_Post",
        "NASA_LOx_Post",
        (("NASA_LOx_Post/postxyz.bin", "grid"), ("NASA_LOx_Post/postq.bin", "solution")),
        "PLOT3D",
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/LOx/",
            "https://book.vtk.org/en/latest/VTKBook/12Chapter12.html",
            "https://vtk.org/doc/nightly/html/classvtkMultiBlockPLOT3DReader.html",
        ),
    ),
    DatasetSpec(
        "Carotid",
        "Carotid",
        (("Carotid/carotid.vtk", "flow_field"),),
        "VTK",
        (
            "https://book.vtk.org/en/latest/VTKBook/06Chapter6.html",
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/CarotidFlow/",
        ),
    ),
    DatasetSpec(
        "Kitchen",
        "Kitchen",
        (("Kitchen/kitchen.vtk", "flow_field"),),
        "VTK",
        (
            "https://examples.vtk.org/site/Cxx/Visualization/Kitchen/",
            "https://www.csc.kth.se/utbildning/kth/kurser/DD2257/visual09/LAB1.pdf",
        ),
    ),
    DatasetSpec(
        "Office",
        "Office",
        (("Office/office.binary.vtk", "flow_field"),),
        "VTK",
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/Office/",
            "https://www.csc.kth.se/utbildning/kth/kurser/DD2257/visual09/LAB1.pdf",
        ),
    ),
)


def _dims(dataset: Any) -> list[int] | None:
    if not hasattr(dataset, "GetDimensions"):
        return None
    values = [0, 0, 0]
    dataset.GetDimensions(values)
    return [int(value) for value in values]


def _array_type(components: int) -> str:
    if components == 1:
        return "scalar"
    if components == 3:
        return "vector"
    return "tensor"


def _array_records(dataset: Any) -> list[dict[str, Any]]:
    variables: list[dict[str, Any]] = []
    for association, attributes in (("point", dataset.GetPointData()), ("cell", dataset.GetCellData())):
        for index in range(attributes.GetNumberOfArrays()):
            array = attributes.GetArray(index)
            name = attributes.GetArrayName(index)
            if array is None or not name:
                continue
            variables.append(
                {
                    "name": name,
                    "physical_quantity": None,
                    "type": _array_type(array.GetNumberOfComponents()),
                    "components": int(array.GetNumberOfComponents()),
                    "association": association,
                    "unit": None,
                    "source": "stored",
                }
            )
    if not variables:
        raise RuntimeError(f"reader returned no point/cell arrays for {dataset.GetClassName()}")
    return variables


def _base_metadata(
    format_type: str,
    grid_type: str,
    dimensions: list[int] | None,
    origin: list[float] | None,
    spacing: list[float] | None,
    variables: list[dict[str, Any]],
    format_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": {"type": format_type, "details": format_details or {}},
        "grid": {
            "type": grid_type,
            "dimensions": dimensions,
            "origin": origin,
            "spacing": spacing,
        },
        "variables": variables,
        "coordinate_system": {
            "type": "cartesian",
            "axis_meaning": {"x": None, "y": None, "z": None},
            "unit": None,
        },
        "temporal": {"type": "single_snapshot"},
    }


def _read_vtk(path: Path, format_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if format_type == "VTU":
        reader = vtkXMLUnstructuredGridReader()
    else:
        reader = vtkDataSetReader()
        apply_legacy_vtk_reader_configuration(
            reader, canonical_vtk_reader_configuration(format_type)
        )
    reader.SetFileName(str(path))
    reader.Update()
    dataset = reader.GetOutput()
    if dataset is None:
        raise RuntimeError(f"VTK reader returned no dataset: {path}")
    dimensions = _dims(dataset)
    origin = list(dataset.GetOrigin()) if hasattr(dataset, "GetOrigin") else None
    spacing = list(dataset.GetSpacing()) if hasattr(dataset, "GetSpacing") else None
    grid_type = "unstructured" if dataset.GetClassName() == "vtkUnstructuredGrid" else "structured"
    variables = _array_records(dataset)
    return (
        _base_metadata(format_type, grid_type, dimensions, origin, spacing, variables),
        {
            "reader": reader.__class__.__name__,
            "reader_configuration": canonical_vtk_reader_configuration(format_type),
        },
    )


def _plot3d_semantics(name: str) -> tuple[str | None, str]:
    stored = {
        "Density": "density",
        "Momentum": "momentum",
        "StagnationEnergy": "energy",
    }
    derived = {
        "Velocity": "velocity",
        "VelocityMagnitude": "velocity magnitude",
    }
    if name in stored:
        return stored[name], "stored"
    if name in derived:
        return derived[name], "reader_derived"
    return None, "reader_derived"


def _plot3d_mapping(name: str) -> VariableMapping:
    if name == "Density":
        return VariableMapping(
            output_name=name,
            source_file_role="solution",
            convention="PLOT3D Q stored density",
            raw_components={"density": "Q[0]"},
        )
    if name == "Momentum":
        return VariableMapping(
            output_name=name,
            source_file_role="solution",
            convention="PLOT3D Q stored momentum",
            raw_components={"x": "Q[1]", "y": "Q[2]", "z": "Q[3]"},
        )
    if name == "StagnationEnergy":
        return VariableMapping(
            output_name=name,
            source_file_role="solution",
            convention="PLOT3D Q stored energy",
            raw_components={"energy": "Q[4]"},
        )
    if name == "Velocity":
        return VariableMapping(
            output_name=name,
            source_file_role="solution",
            convention="vtkMultiBlockPLOT3DReader vector function 200",
            calculation="Canonical VTK PLOT3D reader derives velocity from stored Q variables.",
        )
    if name == "VelocityMagnitude":
        return VariableMapping(
            output_name=name,
            source_file_role="solution",
            convention="vtkMultiBlockPLOT3DReader scalar function 153",
            calculation="Canonical VTK PLOT3D reader derives velocity magnitude from stored Q variables.",
        )
    raise ValueError(f"unsupported PLOT3D variable mapping: {name}")


def _read_plot3d(xyz: Path, q: Path) -> tuple[dict[str, Any], dict[str, Any], list[VariableMapping]]:
    reader = vtkMultiBlockPLOT3DReader()
    apply_plot3d_reader_configuration(reader)
    reader.SetXYZFileName(str(xyz))
    reader.SetQFileName(str(q))
    reader.Update()
    output = reader.GetOutput()
    if output is None or output.GetNumberOfBlocks() != 1:
        count = 0 if output is None else output.GetNumberOfBlocks()
        raise RuntimeError(f"PLOT3D requires exactly one block; {xyz.parent.name} has {count}")
    dataset = output.GetBlock(0)
    dimensions = _dims(dataset)
    if dimensions is None:
        raise RuntimeError(f"PLOT3D reader did not return dimensions: {xyz}")
    # vtkMultiBlockPLOT3DReader may attach vtkGhostType as an adapter-level
    # cell array. It is not a stored PLOT3D quantity or a reader-recovered
    # physical variable, so keep it out of the model-facing metadata and
    # internal variable mappings.
    reader_variables = [
        variable for variable in _array_records(dataset)
        if variable["name"] != "vtkGhostType"
    ]
    grid_blanking = []
    variables = []
    for variable in reader_variables:
        if variable["name"] == "IBlank":
            grid_blanking.append(
                {
                    "name": variable["name"],
                    "type": variable["type"],
                    "components": variable["components"],
                    "association": variable["association"],
                    "source": "reader_derived",
                    "validity_rule": "valid when IBlank > 0; values <= 0 are blanked computational-domain points",
                }
            )
            continue
        physical_quantity, source = _plot3d_semantics(variable["name"])
        variable["physical_quantity"] = physical_quantity
        variable["source"] = source
        variables.append(variable)
    endianness = reader.GetByteOrderAsString().casefold()
    if endianness not in {"bigendian", "littleendian"}:
        raise RuntimeError(f"unrecognized PLOT3D byte order: {reader.GetByteOrderAsString()}")
    details = {
        "encoding": "binary" if reader.GetBinaryFile() else "ascii",
        "precision": "float64" if reader.GetDoublePrecision() else "float32",
        "endianness": "big" if endianness == "bigendian" else "little",
        "has_byte_count": bool(reader.GetHasByteCount()),
    }
    metadata = _base_metadata("PLOT3D", "structured", dimensions, None, None, variables, details)
    reader_config = {
        "reader": PLOT3D_READER_NAME,
        "reader_configuration": canonical_plot3d_reader_configuration(),
        "grid_blanking": grid_blanking,
    }
    return metadata, reader_config, [_plot3d_mapping(variable["name"]) for variable in variables]


SEMANTIC_SUPPORTS: dict[str, list[tuple[str, list[str], str]]] = {
    "Office": [
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/Office/",
            ["variables[name=vectors].physical_quantity", "variables[name=scalars].physical_quantity"],
            "The example identifies the vector field as flow velocity and the scalar field as pressure.",
        ),
        (
            "https://www.csc.kth.se/utbildning/kth/kurser/DD2257/visual09/LAB1.pdf",
            [],
            "The source list was retained for semantic cross-checking; no additional local array names were asserted.",
        ),
    ],
    "Kitchen": [
        (
            "https://examples.vtk.org/site/Cxx/Visualization/Kitchen/",
            ["variables[name=velocity].physical_quantity"],
            "The example uses the local velocity vector for streamline analysis in the kitchen flow field.",
        ),
        (
            "https://www.csc.kth.se/utbildning/kth/kurser/DD2257/visual09/LAB1.pdf",
            [
                "variables[name=p1].physical_quantity",
                "variables[name=u1].physical_quantity",
                "variables[name=v1].physical_quantity",
                "variables[name=w1].physical_quantity",
                "variables[name=ke].physical_quantity",
                "variables[name=ep].physical_quantity",
                "variables[name=h1].physical_quantity",
                "variables[name=c1].physical_quantity",
                "variables[name=c11].physical_quantity",
                "variables[name=c12].physical_quantity",
                "variables[name=c13].physical_quantity",
                "variables[name=c14].physical_quantity",
                "variables[name=c15].physical_quantity",
                "variables[name=c16].physical_quantity",
                "variables[name=c17].physical_quantity",
            ],
            "The source list maps p1, velocity components, turbulence quantities, enthalpy and concentration arrays; units are not asserted.",
        ),
    ],
    "Carotid": [
        (
            "https://book.vtk.org/en/latest/VTKBook/06Chapter6.html",
            ["variables[name=vectors].physical_quantity", "variables[name=scalars].physical_quantity"],
            "The vectors represent blood velocity; the scalar is proportional to velocity magnitude and is not labeled as exact speed.",
        ),
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/CarotidFlow/",
            ["variables[name=vectors].physical_quantity"],
            "The example uses the vector field to generate blood-flow streamtubes.",
        ),
    ],
    "Blunt_Fin": [
        (
            "https://vtk.org/doc/nightly/html/classvtkMultiBlockPLOT3DReader.html",
            [
                "format.details.encoding",
                "format.details.precision",
                "format.details.endianness",
                "format.details.has_byte_count",
                "variables[name=Density].physical_quantity",
                "variables[name=Momentum].physical_quantity",
                "variables[name=StagnationEnergy].physical_quantity",
                "variables[name=Velocity].physical_quantity",
                "variables[name=VelocityMagnitude].physical_quantity",
            ],
            "Canonical VTK PLOT3D documentation defines XYZ/Q roles, stored density/momentum/energy and reader functions 153/200.",
        ),
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/BluntStreamlines/",
            ["data_files[role=grid]", "data_files[role=solution]"],
            "The example confirms the local XYZ/Q pairing and VTK PLOT3D reader usage.",
        ),
    ],
    "Combustor": [
        (
            "https://vtk.org/doc/nightly/html/classvtkMultiBlockPLOT3DReader.html",
            [
                "format.details.encoding",
                "format.details.precision",
                "format.details.endianness",
                "format.details.has_byte_count",
                "variables[name=Density].physical_quantity",
                "variables[name=Momentum].physical_quantity",
                "variables[name=StagnationEnergy].physical_quantity",
                "variables[name=Velocity].physical_quantity",
                "variables[name=VelocityMagnitude].physical_quantity",
            ],
            "Canonical VTK PLOT3D documentation defines stored solution quantities and reader-derived velocity functions.",
        ),
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/WarpCombustor/",
            ["data_files[role=grid]", "data_files[role=solution]"],
            "The example confirms the local combustor XYZ/Q pairing and reader class.",
        ),
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/CombustorIsosurface/",
            [],
            "The example was used as an independent reader/data pairing cross-check; its threshold is not copied into metadata.",
        ),
    ],
    "NASA_LOx_Post": [
        (
            "https://vtk.org/doc/nightly/html/classvtkMultiBlockPLOT3DReader.html",
            [
                "format.details.encoding",
                "format.details.precision",
                "format.details.endianness",
                "format.details.has_byte_count",
                "variables[name=Density].physical_quantity",
                "variables[name=Momentum].physical_quantity",
                "variables[name=StagnationEnergy].physical_quantity",
                "variables[name=Velocity].physical_quantity",
                "variables[name=VelocityMagnitude].physical_quantity",
            ],
            "Canonical VTK PLOT3D documentation defines stored quantities and reader functions 153/200.",
        ),
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/LOx/",
            ["data_files[role=grid]", "data_files[role=solution]"],
            "The example confirms AutoDetectFormatOn and the local postxyz/postq pairing.",
        ),
        (
            "https://book.vtk.org/en/latest/VTKBook/12Chapter12.html",
            ["data_files[role=grid]", "data_files[role=solution]"],
            "The textbook confirms the postxyz/postq dataset usage.",
        ),
    ],
    "FireFlow": [
        (
            "https://examples.vtk.org/site/Cxx/VisualizationAlgorithms/FireFlow/",
            ["format.type", "grid.type"],
            "The example identifies fire_ug.vtu as the XML unstructured-grid solution read by VTK.",
        ),
        (
            "https://docs.enthought.com/mayavi/mayavi/example_fire.html",
            ["format.type", "grid.type"],
            "The source identifies fire_ug.vtu as an unstructured-grid VTK XML file; array physical semantics remain null when not explicit.",
        ),
    ],
}


def _set_semantics(dataset_id: str, metadata: dict[str, Any]) -> None:
    by_name = {item["name"]: item for item in metadata["variables"]}
    if dataset_id == "Office":
        by_name["vectors"]["physical_quantity"] = "velocity"
        by_name["scalars"]["physical_quantity"] = "pressure"
    elif dataset_id == "Kitchen":
        semantics = {
            "velocity": "velocity",
            "p1": "pressure",
            "u1": "velocity component",
            "v1": "velocity component",
            "w1": "velocity component",
            "ke": "turbulent kinetic energy",
            "ep": "turbulent dissipation rate",
            "h1": "enthalpy",
        }
        for name in ("c1", "c11", "c12", "c13", "c14", "c15", "c16", "c17"):
            semantics[name] = "gas concentration"
        for name, physical_quantity in semantics.items():
            by_name[name]["physical_quantity"] = physical_quantity
    elif dataset_id == "Carotid":
        by_name["vectors"]["physical_quantity"] = "velocity"
        by_name["scalars"]["physical_quantity"] = "speed-proportional scalar"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _manifest_for(
    spec: DatasetSpec,
    reader_info: dict[str, Any],
    mappings: list[VariableMapping],
) -> DatasetManifest:
    files = []
    for relative, role in spec.files:
        files.append(manifest_file(DATASETS / relative, relative, role))
    auxiliary_assets = []
    for relative, role in spec.auxiliary_assets:
        auxiliary_assets.append(
            manifest_auxiliary_asset(
                DATASETS / relative,
                relative,
                role,
                source_url=(spec.source_urls[0] if spec.source_urls else None),
            )
        )
    reader = ReaderConfiguration(
        format=spec.format,
        canonical_reader=reader_info["reader"],
        reader_configuration=reader_info["reader_configuration"],
        plot3d_convention=("VTK PLOT3D canonical Q/XYZ convention" if spec.format == "PLOT3D" else None),
        variable_mappings=mappings,
        grid_blanking=reader_info.get("grid_blanking", []),
    )
    return DatasetManifest(
        dataset_id=spec.dataset_id,
        file_root="datasets",
        files=files,
        auxiliary_assets=auxiliary_assets,
        reader=reader,
        provenance={"source_urls": list(spec.source_urls)},
    )


def _sources_for(spec: DatasetSpec) -> dict[str, Any]:
    sources = [
        {
            "url": "local://canonical-reader",
            "supports": [
                "format.type",
                "grid.type",
                "grid.dimensions",
                "grid.origin",
                "grid.spacing",
                "variables[].name",
                "variables[].type",
                "variables[].components",
                "variables[].association",
            ],
            "notes": "Extracted from the local file using the canonical VTK reader; null means not explicitly provided.",
        }
    ]
    for url, supports, notes in SEMANTIC_SUPPORTS[spec.dataset_id]:
        sources.append({"url": url, "supports": supports, "notes": notes})
    return {"dataset_id": spec.dataset_id, "sources": sources}


def generate(spec: DatasetSpec) -> tuple[Path, Path, Path]:
    if spec.format in {"VTK", "VTU"}:
        metadata, reader_info = _read_vtk(DATASETS / spec.files[0][0], spec.format)
        _set_semantics(spec.dataset_id, metadata)
        mappings: list[VariableMapping] = []
    else:
        metadata, reader_info, mappings = _read_plot3d(
            DATASETS / spec.files[0][0], DATASETS / spec.files[1][0]
        )
    DataMetadata.model_validate(metadata)
    manifest = _manifest_for(spec, reader_info, mappings)
    for item in [*manifest.files, *manifest.auxiliary_assets]:
        path = DATASETS / item.path
        if item.size_bytes != path.stat().st_size or item.checksum != sha256_file(path):
            raise RuntimeError(f"manifest checksum validation failed: {item.path}")
    output_dir = DATASETS / spec.output_dir
    metadata_path = output_dir / "data_metadata.json"
    manifest_path = output_dir / "dataset_manifest.json"
    sources_path = output_dir / "metadata_sources.json"
    _write_json(metadata_path, metadata)
    # ``validity_rule`` is intentionally excluded from the legacy in-memory
    # manifest projection, but it is part of the persisted reader contract so
    # regeneration cannot silently erase model-visible domain semantics.
    manifest_payload = manifest.model_dump(mode="json", exclude_none=False)
    for source_item, payload_item in zip(
        manifest.reader.grid_blanking,
        manifest_payload.get("reader", {}).get("grid_blanking", []),
    ):
        if source_item.validity_rule is not None:
            payload_item["validity_rule"] = source_item.validity_rule
    _write_json(manifest_path, manifest_payload)
    _write_json(sources_path, _sources_for(spec))
    # Validate exactly what was written, rather than only in-memory objects.
    DataMetadata.model_validate(json.loads(metadata_path.read_text(encoding="utf-8")))
    DatasetManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    source_payload = json.loads(sources_path.read_text(encoding="utf-8"))
    if not isinstance(source_payload.get("sources"), list):
        raise RuntimeError(f"metadata_sources.json has invalid sources: {sources_path}")
    return metadata_path, manifest_path, sources_path


def main() -> None:
    for spec in SPECS:
        paths = generate(spec)
        print(spec.dataset_id, "->", ", ".join(str(path.relative_to(ROOT)) for path in paths))


if __name__ == "__main__":
    main()

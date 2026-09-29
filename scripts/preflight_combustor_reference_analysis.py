"""Read-only Combustor reference-analysis preflight.

This tool exercises the repository's canonical PLOT3D reader and computes
data-mediated, reproducible geometry summaries.  It deliberately does not
write active cases, Ground Truth, evidence, or curator decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Keep direct ``python scripts/...`` invocation consistent with the repository
# scripts, while remaining a normal importable module under pytest.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.plot3d_reader import (
    PLOT3D_READER_NAME,
    apply_plot3d_reader_configuration,
    canonical_plot3d_reader_configuration,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _triangle_area_and_centroid(points: np.ndarray) -> tuple[float, np.ndarray]:
    p0, p1, p2 = points
    area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
    return area, (p0 + p1 + p2) / 3.0


def _contour_regions(grid: Any, level: float) -> list[dict[str, Any]]:
    """Extract connected contour components and area-weighted centroids."""
    try:
        from vtk.util.numpy_support import vtk_to_numpy
        from vtkmodules.vtkFiltersCore import (
            vtkContourFilter,
            vtkPolyDataConnectivityFilter,
            vtkTriangleFilter,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional VTK
        raise RuntimeError("VTK Python bindings are required for Combustor preflight") from exc

    contour = vtkContourFilter()
    contour.SetInputData(grid)
    contour.SetInputArrayToProcess(0, 0, 0, 0, "Density")
    contour.SetValue(0, float(level))
    contour.Update()

    connectivity = vtkPolyDataConnectivityFilter()
    connectivity.SetInputConnection(contour.GetOutputPort())
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()

    triangulate = vtkTriangleFilter()
    triangulate.SetInputConnection(connectivity.GetOutputPort())
    triangulate.Update()
    surface = triangulate.GetOutput()
    region_array = surface.GetPointData().GetArray("RegionId")
    if region_array is None:
        raise RuntimeError("VTK connectivity output did not contain RegionId")
    region_ids = vtk_to_numpy(region_array).astype(np.int64, copy=False)
    points = vtk_to_numpy(surface.GetPoints().GetData()).astype(float, copy=False)

    region_count = int(connectivity.GetNumberOfExtractedRegions())
    areas = np.zeros(region_count, dtype=float)
    moments = np.zeros((region_count, 3), dtype=float)
    triangle_counts = np.zeros(region_count, dtype=int)

    for cell_index in range(surface.GetNumberOfCells()):
        cell = surface.GetCell(cell_index)
        if cell.GetNumberOfPoints() != 3:
            raise RuntimeError("TriangleFilter produced a non-triangle cell")
        point_ids = [cell.GetPointId(index) for index in range(3)]
        rid_values = region_ids[point_ids]
        if not np.all(rid_values == rid_values[0]):
            raise RuntimeError("A contour triangle spans multiple connectivity regions")
        region_id = int(rid_values[0])
        if region_id < 0 or region_id >= region_count:
            raise RuntimeError(f"Unexpected VTK RegionId: {region_id}")
        area, centroid = _triangle_area_and_centroid(points[point_ids])
        areas[region_id] += area
        moments[region_id] += area * centroid
        triangle_counts[region_id] += 1

    regions: list[dict[str, Any]] = []
    for region_id in range(region_count):
        region_points = points[region_ids == region_id]
        area = float(areas[region_id])
        centroid = (moments[region_id] / area).tolist() if area > 0 else None
        bounds = None
        if region_points.size:
            bounds = {
                "x": [float(region_points[:, 0].min()), float(region_points[:, 0].max())],
                "y": [float(region_points[:, 1].min()), float(region_points[:, 1].max())],
                "z": [float(region_points[:, 2].min()), float(region_points[:, 2].max())],
            }
        regions.append(
            {
                "vtk_region_id": region_id,
                "point_count": int(region_points.shape[0]),
                "triangle_count": int(triangle_counts[region_id]),
                "surface_area": area,
                "area_weighted_centroid": centroid,
                "bounds": bounds,
            }
        )

    # Rank is useful for inspection; the VTK region id is retained only as an
    # implementation diagnostic and is not a scientific identity.
    regions.sort(key=lambda item: (-item["surface_area"], item["vtk_region_id"]))
    for rank, region in enumerate(regions, start=1):
        region["area_rank"] = rank
    return regions


def run_combustor_reference_preflight(
    repository_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the isolated Combustor analysis and optionally persist its report."""
    root = Path(repository_root).resolve()
    dataset_root = root / "datasets" / "Combustor"
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = _read_json(manifest_path)
    files = manifest.get("files") or []
    by_role = {str(item.get("role")): item for item in files}
    if set(by_role) != {"grid", "solution"}:
        raise ValueError("Combustor manifest must contain exactly grid and solution files")
    grid_path = root / "datasets" / str(by_role["grid"]["path"])
    solution_path = root / "datasets" / str(by_role["solution"]["path"])
    if not grid_path.is_file() or not solution_path.is_file():
        raise FileNotFoundError("Combustor PLOT3D input files are missing")

    try:
        from vtkmodules.vtkIOParallel import vtkMultiBlockPLOT3DReader
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:  # pragma: no cover - depends on optional VTK
        raise RuntimeError("VTK Python bindings are required for Combustor preflight") from exc

    reader_configuration = manifest.get("reader", {}).get("reader_configuration")
    expected_configuration = canonical_plot3d_reader_configuration()
    if reader_configuration != expected_configuration:
        raise ValueError("Combustor manifest reader configuration is not canonical")

    reader = vtkMultiBlockPLOT3DReader()
    require_single_block = apply_plot3d_reader_configuration(reader, reader_configuration)
    reader.SetXYZFileName(str(grid_path))
    reader.SetQFileName(str(solution_path))
    reader.Update()
    output = reader.GetOutput()
    if output is None or output.GetNumberOfBlocks() == 0:
        raise RuntimeError("vtkMultiBlockPLOT3DReader returned no Combustor block")
    if require_single_block and output.GetNumberOfBlocks() != 1:
        raise RuntimeError("canonical Combustor preflight requires one PLOT3D block")
    grid = output.GetBlock(0)
    if grid is None:
        raise RuntimeError("vtkMultiBlockPLOT3DReader returned an empty Combustor block")
    density_array = grid.GetPointData().GetArray("Density")
    if density_array is None:
        raise RuntimeError("canonical reader did not expose Density")
    density = vtk_to_numpy(density_array).astype(float, copy=False)
    finite = density[np.isfinite(density)]
    if finite.size == 0:
        raise RuntimeError("Combustor Density contains no finite values")
    q90 = float(np.quantile(finite, 0.90))

    dimensions = [0, 0, 0]
    grid.GetDimensions(dimensions)

    report: dict[str, Any] = {
        "schema_version": "combustor-reference-analysis-preflight-v1",
        "status": "PASS",
        "analysis_role": "DATA_MEDIATED_REPRODUCIBILITY_PREFLIGHT",
        "scientific_grounding_confirmation": False,
        "curator_confirmation": False,
        "active_case_mutation": False,
        "ground_truth_mutation": False,
        "dataset_id": "Combustor",
        "reader": {
            "canonical_reader": PLOT3D_READER_NAME,
            "reader_configuration": expected_configuration,
            "block_count": int(output.GetNumberOfBlocks()),
            "grid_class": grid.GetClassName(),
            "dimensions": [int(value) for value in dimensions],
            "point_count": int(grid.GetNumberOfPoints()),
        },
        "input_files": [
            {
                "path": str(by_role["grid"]["path"]),
                "role": "grid",
                "sha256": _sha256(grid_path),
                "size_bytes": grid_path.stat().st_size,
            },
            {
                "path": str(by_role["solution"]["path"]),
                "role": "solution",
                "sha256": _sha256(solution_path),
                "size_bytes": solution_path.stat().st_size,
            },
        ],
        "density": {
            "array_name": "Density",
            "finite_count": int(finite.size),
            "total_count": int(density.size),
            "minimum": float(finite.min()),
            "maximum": float(finite.max()),
            "q90": q90,
            "quantile_definition": "numpy.quantile over all finite stored Density point values",
        },
        "contour": {
            "scalar": "Density",
            "level": 0.38,
            "filter": "vtkContourFilter",
            "connectivity": "vtkPolyDataConnectivityFilter(all regions)",
            "triangulation": "vtkTriangleFilter",
            "region_count": 0,
            "regions": [],
        },
        "interpretation_boundary": {
            "is_data_mediated_reproducibility": True,
            "supports_dataset_specific_finding_generation": False,
            "does_not_confirm_target_grounding": True,
            "does_not_create_canonical_case_identity": True,
            "does_not_create_ground_truth": True,
            "requires_construction_and_curator_review": True,
        },
    }
    regions = _contour_regions(grid, 0.38)
    report["contour"]["region_count"] = len(regions)
    report["contour"]["regions"] = regions

    if output_path is not None:
        destination = Path(output_path)
    else:
        destination = root / "outputs/current/combustor_reference_analysis_preflight.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_combustor_reference_preflight(args.repository_root, args.output)
    destination = args.output or args.repository_root / "outputs/current/combustor_reference_analysis_preflight.json"
    print(json.dumps({
        "status": report["status"],
        "output": str(destination),
        "density_q90": report["density"]["q90"],
        "contour_level": report["contour"]["level"],
        "region_count": report["contour"]["region_count"],
        "largest_region_area": report["contour"]["regions"][0]["surface_area"] if report["contour"]["regions"] else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

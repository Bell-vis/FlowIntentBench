"""Face-connected cell regions with explicitly declared speed aggregation."""
from collections import defaultdict

import numpy as np


def face_connected_components(cell_faces):
    adjacency = {int(cid): set() for cid in cell_faces}
    faces = defaultdict(list)
    for cid, items in cell_faces.items():
        for face in items:
            faces[tuple(sorted(face))].append(int(cid))
    for owners in faces.values():
        if len(owners) > 2:
            raise ValueError("non-manifold face in cell region")
        if len(owners) == 2:
            a, b = owners
            adjacency[a].add(b)
            adjacency[b].add(a)
    seen, regions = set(), []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack, region = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            region.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        regions.append(np.array(sorted(region), dtype=int))
    return regions


def analyze_cell_speed_regions(root, dataset_id, effective, operation_id, plan):
    from flowintentbench.deterministic_materialization import _flow_arrays, _digest, _select_unique_max, VECTOR_NAMES
    from vtk.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    dataset, manifest, speed, valid, _ = _flow_arrays(root, dataset_id)
    if plan.aggregation_kind not in {"mean_of_point_speeds", "magnitude_of_mean_point_velocity"} or plan.measure_kind not in {"volume_weighted_mean", "volume_weighted_rms"}:
        raise ValueError("unsupported cell speed aggregation or measure")
    vectors = vtk_to_numpy(dataset.GetPointData().GetArray(VECTOR_NAMES[dataset_id])).astype(float)
    points = vtk_to_numpy(dataset.GetPoints().GetData()).astype(float)
    from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
    cells, centers, vertex_centers, cell_speeds, geometry_volumes, cell_indices = [], [], [], [], [], []
    for index in range(dataset.GetNumberOfCells()):
        cell = dataset.GetCell(index)
        ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        if len(ids) == 0 or not np.all(valid[ids]):
            continue
        if len(ids) != 8:
            raise ValueError("cell-speed executor requires eight-corner cells")
        coordinates = points[ids]
        tet_ids, tet_points = vtkIdList(), vtkPoints()
        tet_points.SetDataTypeToDouble()
        if not cell.Triangulate(0, tet_ids, tet_points):
            raise ValueError("hexahedron tetrahedralization failed")
        tetrahedra = vtk_to_numpy(tet_points.GetData()).astype(float).reshape(-1,4,3)
        tet_volumes = np.abs(np.linalg.det(tetrahedra[:,1:] - tetrahedra[:,:1])) / 6
        geometric_volume = float(tet_volumes.sum())
        if geometric_volume <= 0:
            continue
        cells.append(ids)
        cell_indices.append(index)
        centers.append(np.average(tetrahedra.mean(axis=1), axis=0, weights=tet_volumes))
        vertex_centers.append(coordinates.mean(axis=0))
        cell_speeds.append(float(np.linalg.norm(vectors[ids].mean(axis=0))) if plan.aggregation_kind == "magnitude_of_mean_point_velocity" else speed[ids].mean())
        geometry_volumes.append(geometric_volume)
    cells = np.array(cells, dtype=int)
    centers, cell_speeds = np.asarray(centers), np.asarray(cell_speeds)
    size = vtkCellSizeFilter()
    size.SetInputData(dataset)
    size.ComputeVolumeOn()
    size.Update()
    signed_volumes = vtk_to_numpy(size.GetOutput().GetCellData().GetArray("Volume")).astype(float)[cell_indices]
    volumes = np.abs(signed_volumes)
    if np.any(volumes <= 0) or not np.allclose(volumes, geometry_volumes, rtol=1e-6, atol=1e-12):
        raise ValueError("cell volumes disagree with the verified hexahedron geometry")
    population = cell_speeds[cell_speeds > 0] if plan.quantile_population == "positive" else cell_speeds
    if plan.quantile_population == "volume_weighted":
        order = np.argsort(cell_speeds, kind="stable")
        cumulative = np.cumsum(volumes[order])
        threshold = float(cell_speeds[order[np.searchsorted(cumulative, plan.quantile*cumulative[-1], side="left")]])
    else:
        threshold = float(np.quantile(population, plan.quantile,
                                     method="inverted_cdf" if plan.quantile_method == "nearest_rank" else "linear"))
    mask = cell_speeds > threshold if plan.comparator == "GT" else cell_speeds >= threshold
    selected_faces = {}
    for index in np.flatnonzero(mask):
        cell = dataset.GetCell(int(cell_indices[index]))
        selected_faces[int(index)] = [tuple(cell.GetFace(j).GetPointId(k) for k in range(cell.GetFace(j).GetNumberOfPoints()))
                                      for j in range(cell.GetNumberOfFaces())]
    regions = face_connected_components(selected_faces)
    strengths = np.array([np.average(cell_speeds[r], weights=volumes[r]) if plan.measure_kind == "volume_weighted_mean"
                          else np.sqrt(np.average(cell_speeds[r] ** 2, weights=volumes[r])) for r in regions])
    selected = regions[_select_unique_max(strengths, dimension="component_selection")]
    selected_points = np.unique(cells[selected])
    peak = int(selected_points[np.argmax(speed[selected_points])])
    volume = float(volumes[selected].sum())
    centroid = np.average(centers[selected], axis=0, weights=volumes[selected])
    bounds = np.stack([points[selected_points].min(axis=0), points[selected_points].max(axis=0)], axis=1)
    result = {"operation_id": operation_id, "criterion_kind": "quantile", "comparator": plan.comparator,
              "threshold": threshold, "measure_kind": plan.measure_kind, "representation": "volume_centroid",
              "reported_location": centroid.tolist(), "strength": float(strengths.max()),
              "maximum_cell_speed": float(cell_speeds[selected].max()), "region_volume": volume, "selected_region_size": int(len(selected)), "selected_point_count": int(len(selected_points)),
              "region_count": len(regions), "retained_cells": int(mask.sum()), "total_cell_count": len(cells),
              "total_volume": float(volumes.sum()), "excluded_cell_count": dataset.GetNumberOfCells()-len(cells),
              "vertex_center_weighted_location": np.average(np.asarray(vertex_centers)[selected],axis=0,weights=volumes[selected]).tolist(), "peak_speed": float(speed[peak]), "peak_point": points[peak].tolist(),
              "peak_velocity": vtk_to_numpy(dataset.GetPointData().GetArray(VECTOR_NAMES[dataset_id]))[peak].tolist(),
              "global_peak_speed": float(speed[valid].max()),
              "mean_speed": float(np.average(cell_speeds[selected], weights=volumes[selected])),
              "bounding_box": {axis: bounds[i].tolist() for i, axis in enumerate("xyz")},
              "cell_center_bounds": [centers[selected].min(axis=0).tolist(), centers[selected].max(axis=0).tolist()],
              "coordinate_span": (bounds[:, 1] - bounds[:, 0]).tolist(),
              "region_rankings": [{"rank": rank+1, "strength": float(strengths[i]), "cell_count": int(len(regions[i])),
                                   "volume": float(volumes[regions[i]].sum()),
                                   "peak_speed": float(speed[np.unique(cells[regions[i]])].max()),
                                   "mean_speed": float(np.average(cell_speeds[regions[i]], weights=volumes[regions[i]])),
                                   "reported_location": np.average(centers[regions[i]], axis=0, weights=volumes[regions[i]]).tolist()}
                                  for rank, i in enumerate(np.argsort(-strengths, kind="stable")[:32])]}
    provenance = {"dataset_id": dataset_id, "dataset_manifest_sha256": _digest(manifest),
                  "input_files": [item.get("path") for item in manifest.get("files", [])],
                  "reader_format": manifest.get("reader", {}).get("format"),
                  "reader_configuration": manifest.get("reader", {}).get("reader_configuration", {}),
                  "point_to_cell_rule": "Euclidean norm of mean corner velocity vectors" if plan.aggregation_kind == "magnitude_of_mean_point_velocity" else "mean of Euclidean point-speed magnitudes",
                  "geometry_check": "absolute signed VTK cell volume agrees with tetrahedralization; volume centroid from tetrahedral moments",
                  "negative_orientation_cell_count": int(np.count_nonzero(signed_volumes < 0)),
                  "materialization_plan": plan.to_dict(), "effective_o_sha256": _digest(effective),
                  "compiled_plan_sha256": _digest(plan.to_dict()), "executed_plan_sha256": _digest(plan.to_dict()),
                  "silent_substitutions": [], "unsupported_dimensions": []}
    return {"parameters": dict(effective), "result": result,
            "execution": {"status": "MATERIALIZED", "reproducible": True, "G_of_O": result, "data_provenance": provenance},
            "materialization_id": f"mat:{operation_id}"}

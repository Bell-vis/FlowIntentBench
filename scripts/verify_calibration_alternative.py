#!/usr/bin/env python3
"""Independent deterministic check of the pilot's open tetrahedral method.

Uses the frozen raw mesh. Does not modify GT or any solver answer/reviewer grade.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/model_metric_calibration/reports/alternative_method_verification.json"


def main():
    started = time.monotonic()
    manifest = json.loads((ROOT / "datasets/AIDEAS_Blow_Mold/dataset_manifest.json").read_text())
    source = ROOT / "datasets" / manifest["files"][0]["path"]
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == manifest["files"][0]["checksum"]
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(source))
    reader.Update()
    mesh = reader.GetOutput()
    triangulation = vtk.vtkDataSetTriangleFilter()
    triangulation.SetInputData(mesh)
    triangulation.Update()
    tetra = triangulation.GetOutput()
    assert set(vtk_to_numpy(tetra.GetCellTypes())) == {vtk.VTK_TETRA}
    connectivity = vtk_to_numpy(tetra.GetCells().GetConnectivityArray()).reshape(-1, 4)
    points = vtk_to_numpy(tetra.GetPoints().GetData()).astype(np.float64)[connectivity]
    field_name = next(tetra.GetPointData().GetArrayName(i) for i in range(tetra.GetPointData().GetNumberOfArrays())
                      if "NormShearRate" in tetra.GetPointData().GetArrayName(i))
    shear = vtk_to_numpy(tetra.GetPointData().GetArray(field_name)).astype(np.float64)[connectivity].mean(axis=1)
    vectors = points[:, 1:] - points[:, :1]
    volume = np.abs(np.einsum("ij,ij->i", np.cross(vectors[:, 0], vectors[:, 1]), vectors[:, 2])) / 6
    assert np.isfinite(shear).all() and np.isfinite(volume).all() and (volume >= 0).all()
    order = np.argsort(shear, kind="stable")
    cutoff = shear[order[np.searchsorted(np.cumsum(volume[order]), .9 * volume.sum(), side="left")]]
    selected = shear >= cutoff
    centroid = np.average(points.mean(axis=1)[selected], weights=volume[selected], axis=0)
    mean = np.average(shear[selected], weights=volume[selected])
    claimed = {"threshold": 312.9735, "centroid": [-0.5242, -0.9632, 17.1367], "mean_shear": 452.09}
    recomputed = {"threshold": float(cutoff), "centroid": centroid.tolist(), "mean_shear": float(mean),
        "total_volume": float(volume.sum()), "selected_volume": float(volume[selected].sum()),
        "tetrahedra": len(volume), "original_cells": mesh.GetNumberOfCells()}
    checks = {"threshold_rounds_to_claim": bool(abs(cutoff - claimed["threshold"]) <= 0.00005),
        "centroid_rounds_to_claim": bool(np.all(np.abs(centroid - claimed["centroid"]) <= 0.00005)),
        "mean_rounds_to_claim": bool(abs(mean - claimed["mean_shear"]) <= 0.005)}
    result = {"case_id": "aideas_high_shear_region_o3_f1", "model_id": "gpt-6-astra",
        "verification_role": "Independent calibration audit, separate from frozen benchmark scorer",
        "input_sha256": source_sha, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vtk_version": vtk.vtkVersion.GetVTKVersion(), "method": "VTK tetrahedral decomposition; tetra nodal mean; exact tetra volumes/centroids; volume-weighted left-inverse q90; retain ties",
        "claimed": claimed, "recomputed": recomputed, "checks": checks,
        "conclusion": "CLAIMED_ALTERNATIVE_REPRODUCED" if all(checks.values()) else "ALTERNATIVE_NOT_REPRODUCED",
        "seconds": time.monotonic() - started, "api_calls": 0,
        "interpretation": "O3 explicitly permits choosing a procedure. Differences from an original-cell reference alone do not refute a tetrahedral method. This verifies the reported primary values, not every additional scientific claim."}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

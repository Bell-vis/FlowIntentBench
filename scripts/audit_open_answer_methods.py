#!/usr/bin/env python3
"""Reproduce declared methods on frozen data, separate from benchmark scoring.

The answer annotations below were read from the saved answers. They are explicit
audit inputs, not a general answer extractor or new automatic acceptance rules.
No model code is executed; no GT, answer, or reviewer grade is modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.stats import rankdata
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.answer_evidence import value_is_bound

OUT = ROOT / "outputs/model_metric_calibration"
MODELS = ("gpt-6-astra", "gpt-5.6-luna", "gpt-5.6-terra", "claude-fable-5-1",
          "claude-sonnet-5", "deepseek-flash", "qwen3.8-max")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(dataset):
    manifest = ROOT / "datasets" / dataset / "dataset_manifest.json"
    spec = read(manifest)["files"][0]
    path = ROOT / "datasets" / spec["path"]
    assert sha(path) == spec["checksum"]
    reader = vtk.vtkXMLUnstructuredGridReader() if path.suffix == ".vtu" else vtk.vtkDataSetReader()
    reader.SetFileName(str(path))
    if path.suffix == ".vtk":
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
    reader.Update()
    return reader.GetOutput(), {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def array(data, name):
    return vtk_to_numpy(data.GetArray(name)).astype(np.float64)


def cell_view(mesh):
    size = vtk.vtkCellSizeFilter()
    size.SetInputData(mesh)
    size.Update()
    weights = np.abs(array(size.GetOutput().GetCellData(), "Volume"))
    centers = vtk.vtkCellCenters()
    centers.SetInputData(mesh)
    centers.Update()
    points = vtk_to_numpy(centers.GetOutput().GetPoints().GetData()).astype(float)
    transfer = vtk.vtkPointDataToCellData()
    transfer.SetInputData(mesh)
    transfer.Update()
    return weights, points, transfer.GetOutput().GetCellData()


def correlation(x, y, weights=None):
    weights = np.ones(len(x)) if weights is None else weights
    x, y = x - np.average(x, weights=weights), y - np.average(y, weights=weights)
    return float(np.dot(weights, x * y) / np.sqrt(np.dot(weights, x*x) * np.dot(weights, y*y)))


def cv(x, weights=None):
    mean = np.average(x, weights=weights)
    return float(np.sqrt(np.average((x - mean)**2, weights=weights)) / abs(mean))


def region(field, volumes, centers, q):
    order = np.argsort(field, kind="stable")
    threshold = field[order[np.searchsorted(np.cumsum(volumes[order]), q * volumes.sum(), side="left")]]
    mask = field >= threshold
    return {"threshold": float(threshold), "mean": float(np.average(field[mask], weights=volumes[mask])),
        "centroid": np.average(centers[mask], weights=volumes[mask], axis=0).tolist(),
        "selected_volume": float(volumes[mask].sum()), "total_volume": float(volumes.sum())}


def original_cell_tetra_integrals(mesh):
    # vtkCell.Triangulate(0) is the convention explicitly visible in trial 3.
    # Gather connectivity, then use an independent vectorized moment calculation.
    parents, indices = [], []
    ids, points = vtk.vtkIdList(), vtk.vtkPoints()
    for i in range(mesh.GetNumberOfCells()):
        assert mesh.GetCell(i).Triangulate(0, ids, points)
        n = ids.GetNumberOfIds()
        assert n % 4 == 0
        indices.extend(ids.GetId(j) for j in range(n))
        parents.extend([i] * (n // 4))
    conn = np.array(indices).reshape(-1, 4)
    parents = np.array(parents)
    xyz = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)[conn]
    vectors = xyz[:, 1:] - xyz[:, :1]
    volume = np.abs(np.einsum("ij,ij->i", np.cross(vectors[:, 0], vectors[:, 1]), vectors[:, 2])) / 6
    shear = array(mesh.GetPointData(), "NormShearRate [1/s]")[conn].mean(axis=1)
    weights = np.bincount(parents, volume)
    centers = np.column_stack([np.bincount(parents, volume * xyz.mean(axis=1)[:, k]) / weights for k in range(3)])
    mean_shear = np.bincount(parents, volume * shear) / weights
    assert (weights > 0).all()
    return mean_shear, weights, centers


def main():
    started = time.monotonic()
    from scripts.run_model_metric_calibration import inventory, load_development_case
    selected = ("aideas_pressure_heterogeneity_o3_f1", "tubes_pressure_speed_association_o3_f1", "mhd_density_magnetic_association_o3_f1")
    args = SimpleNamespace(manifest=ROOT / "experiments/expansion_v1_development/case_manifest.json",
        extra_collection=[], output=OUT / "open_answer_audit")
    args.case_rows = read(args.manifest)["cases"]
    cases = {r["case_id"]: load_development_case(ROOT, r) for r in args.case_rows if r["case_id"] in selected}
    selection, _ = inventory(args, cases, selected_case_ids=selected)
    assert not selection["missing_models"] and not selection["missing_selected_cells"]
    mesh, aideas_source = load("AIDEAS_Blow_Mold")
    weights, centers, cells = cell_view(mesh)
    pressure = array(mesh.GetCellData(), "Pressure [bar]")
    computations = {"pressure_weighted_cv": cv(pressure, weights), "pressure_equal_cell_cv": cv(pressure)}
    shear = array(cells, "NormShearRate [1/s]")
    shear_refs = {f"q{int(q*100)}": region(shear, weights, centers, q) for q in (.9, .95)}
    field, volumes, locations = original_cell_tetra_integrals(mesh)
    trial3 = region(field, volumes, locations, .95)
    tubes, tubes_source = load("OpenFOAM_Tubes")
    weights, _, _ = cell_view(tubes)
    p = array(tubes.GetCellData(), "p")
    speed = np.linalg.norm(array(tubes.GetCellData(), "U"), axis=1)
    computations.update(tubes_weighted_pearson=correlation(p, speed, weights),
        tubes_equal_cell_pearson=correlation(p, speed),
        tubes_equal_cell_spearman=correlation(rankdata(p), rankdata(speed)),
        tubes_weighted_rank_pearson=correlation(rankdata(p), rankdata(speed), weights))
    mhd, mhd_source = load("MHD_Turbulence")
    density = array(mhd.GetPointData(), "density")
    magnetic = np.linalg.norm(array(mhd.GetPointData(), "magnetic_field"), axis=1)
    weights, _, cells = cell_view(mhd)
    cell_density = array(cells, "density")
    cell_magnetic = np.linalg.norm(array(cells, "magnetic_field"), axis=1)
    computations.update(mhd_point_pearson=correlation(density, magnetic),
        mhd_cell_weighted_pearson=correlation(cell_density, cell_magnetic, weights),
        mhd_cell_weighted_rank_pearson=correlation(rankdata(cell_density), rankdata(cell_magnetic), weights))
    # Confirm VTK's point ordering from coordinates before naming any axis.
    assert np.allclose(mhd.GetPoint(1), [1/63, 0, 0])
    assert np.allclose(mhd.GetPoint(64*64), [0, 0, 1/63])
    rho_grid, b_grid = density.reshape(64, 64, 64), magnetic.reshape(64, 64, 64)
    plane_correlations = {}
    for axis, label in enumerate(("z", "y", "x")):
        coefficients = np.array([correlation(np.take(rho_grid, k, axis=axis).ravel(),
            np.take(b_grid, k, axis=axis).ravel()) for k in range(64)])
        plane_correlations[label] = {"min": float(coefficients.min()), "max": float(coefficients.max()),
            "mean": float(coefficients.mean()), "median": float(np.median(coefficients))}
    extra_answer = read(OUT / "open_answer_audit/answers/deepseek-flash__mhd_density_magnetic_association_o3_f1.json")
    extra_quote = "the 64 z-planes give a mean r of −0.725 (range −0.78 to −0.64)"
    assert extra_quote in extra_answer["answer"]

    # (primary method, printed coefficient, printed decimal places, primary method covered by GT).
    annotations = {
        "aideas_pressure_heterogeneity_o3_f1": [
            ("pressure_weighted_cv", .45477, 5, True), ("pressure_weighted_cv", .45477, 5, True),
            ("pressure_weighted_cv", .454767, 6, True), ("pressure_weighted_cv", .455, 3, True),
            ("pressure_weighted_cv", .455, 3, True), ("pressure_equal_cell_cv", .459, 3, False),
            ("pressure_equal_cell_cv", .4589, 4, False)],
        "tubes_pressure_speed_association_o3_f1": [
            ("tubes_weighted_pearson", -.9568, 4, True), ("tubes_equal_cell_pearson", -.9213336772, 10, False),
            ("tubes_weighted_pearson", -.9568, 4, True), ("tubes_equal_cell_pearson", -.921, 3, False),
            ("tubes_equal_cell_spearman", -.897, 3, False), ("tubes_equal_cell_pearson", -.921, 3, False),
            ("tubes_equal_cell_pearson", -.9213, 4, False)],
        "mhd_density_magnetic_association_o3_f1": [
            ("mhd_point_pearson", -.7202059474, 10, False), ("mhd_point_pearson", -.7202059474, 10, False),
            ("mhd_point_pearson", -.720205947393, 12, False), ("mhd_point_pearson", -.720, 3, False),
            ("mhd_point_pearson", -.720, 3, False), ("mhd_point_pearson", -.7202, 4, False),
            ("mhd_point_pearson", -.720, 3, False)]}
    rows = []
    for case_id, specs in annotations.items():
        for model, (method, value, decimals, covered) in zip(MODELS, specs):
            source = OUT / "open_answer_audit/answers" / f"{model}__{case_id}.json"
            answer = read(source)
            assert hashlib.sha256(answer["answer"].encode()).hexdigest() == answer["answer_sha256"]
            assert value_is_bound(value, answer["answer"]), (case_id, model, value)
            actual = computations[method]
            rows.append({"model_id": model, "case_id": case_id, "trial": 1, "source": str(source.relative_to(ROOT)),
                "answer_sha256": answer["answer_sha256"], "annotated_primary_method": method,
                "claimed_primary_value": value, "recomputed": actual, "rounding_radius": .5 * 10**-decimals,
                "reproduced_to_printed_precision": abs(value - actual) <= .5 * 10**-decimals + 1e-14,
                "primary_method_covered_by_frozen_gt": covered,
                "interpretation": "Numerical reproducibility only; scientific adequacy and extra claims require separate assessment."})

    root = ROOT / "outputs/expansion96_n3_gpt6_sol_windows_run"
    trials = []
    for slot in read(root / "collection_state.json")["slots"]:
        if slot["model_id"] != "gpt-6-astra" or slot["case_id"] != "aideas_high_shear_region_o3_f1":
            continue
        path = root / slot["run_record_path"].replace("\\", "/")
        assert sha(path) == slot["run_record_sha256"]
        run = read(path)
        trials.append({"trial": slot["trial_index"], "status": slot["status"], "run_record": str(path.relative_to(ROOT)),
            "run_sha256": sha(path), "answer_sha256": hashlib.sha256(run["final_response"].encode()).hexdigest(),
            "method": {1: "q90 over tetrahedra", 2: "No completed result; percentile intention only",
                       3: "q95 over original cells; tetra-integrated field and volume centroid"}[slot["trial_index"]],
            "primary_method_matches_gt": False if slot["status"] == "COMPLETED" else None})
    trial3_claimed = {"threshold": 402.8, "mean": 530.1, "centroid": [-1.305, -.919, 17.425]}
    trial3_checks = {"threshold": abs(trial3["threshold"] - 402.8) <= .05,
        "mean": abs(trial3["mean"] - 530.1) <= .05,
        "centroid": bool(np.all(np.abs(np.array(trial3["centroid"]) - trial3_claimed["centroid"]) <= .0005))}
    result = {"scope": "Post-pilot targeted audit: 21 additional O3 answers on 3 cases plus all 3 GPT-6 trials of high shear. No new evaluator scores.",
        "annotation_authority": "Explicit assistant audit annotations checked against original answer numbers; not human labels.",
        "data_sources": [aideas_source, tubes_source, mhd_source], "script_sha256": sha(__file__),
        "vtk_version": vtk.vtkVersion.GetVTKVersion(), "computations": computations,
        "aideas_original_cell_reference_recomputed": shear_refs,
        "mhd_plane_correlations_by_stored_axis": plane_correlations,
        "verified_extra_claim_error": {"model_id": "deepseek-flash", "case_id": "mhd_density_magnetic_association_o3_f1",
            "answer_sha256": extra_answer["answer_sha256"], "evidence_text": extra_quote,
            "error": "Axis mislabel: the quoted mean/range reproduce x-planes, not the declared z-planes.",
            "effect": "Global correlation remains correct; the secondary spatial diagnostic is wrong."},
        "gpt6_high_shear_trials": trials, "gpt6_trial3_recomputed": trial3,
        "gpt6_trial3_claimed": trial3_claimed, "gpt6_trial3_checks": trial3_checks,
        "answers": rows, "reproduced_primary_n": sum(r["reproduced_to_printed_precision"] for r in rows),
        "primary_method_outside_gt_n": sum(not r["primary_method_covered_by_frozen_gt"] for r in rows),
        "seconds": time.monotonic() - started, "api_calls": 0,
        "limits": ["Reproducing a method does not prove that its choice is scientifically adequate.",
                   "No primary method was admitted to GT or granted extra score for this audit.",
                   "Auxiliary scientific interpretations, bootstrap claims and all additional numbers are not exhaustively verified.",
                   "Existing saved trials may have different tool/runtime budgets; this is not a pure base-model ability comparison."]}
    dest = OUT / "reports/open_answer_audit.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("computations", "gpt6_trial3_recomputed", "gpt6_trial3_checks",
        "reproduced_primary_n", "primary_method_outside_gt_n", "seconds", "api_calls")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

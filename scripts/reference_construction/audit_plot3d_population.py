#!/usr/bin/env python3
"""Independent reference-construction audit of PLOT3D speed populations.

Never imported by production evaluation. Reads no model answers or judgments.
Compares canonical reader vectors with the stored momentum/density definition,
with and without IBlank; saves method identities and source checksums.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy import ndimage
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.deterministic_materialization import _reader_dataset


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(speed, mask, coordinates, dimensions):
    finite = mask & np.isfinite(speed)
    positive = finite & (speed > 0)
    threshold = float(np.quantile(speed[positive], .9, method="linear"))
    selected = finite & (speed >= threshold)
    labels, count = ndimage.label(selected.reshape(dimensions, order="F"),
                                 structure=ndimage.generate_binary_structure(3, 1))
    labels = labels.ravel(order="F")
    regions = []
    for index in range(1, count + 1):
        local = labels == index
        regions.append({"size": int(local.sum()), "peak_speed": float(speed[local].max()),
                        "mean_location": coordinates[local].mean(axis=0).tolist()})
    regions.sort(key=lambda row: row["peak_speed"], reverse=True)
    return {"finite_points": int(finite.sum()), "nonzero_points": int(positive.sum()),
            "threshold": threshold, "retained_points": int(selected.sum()),
            "region_count": count, "regions_by_peak": regions}


def main():
    dataset_id = "NASA_LOx_Post"
    dataset, manifest = _reader_dataset(ROOT, dataset_id)
    files = []
    for item in manifest["files"]:
        source = ROOT / "datasets" / item["path"]
        if item.get("role") not in {"grid", "solution"}:
            continue
        actual = checksum(source)
        if actual != item["checksum"]:
            raise ValueError("input differs from frozen dataset")
        files.append({"path": str(source.relative_to(ROOT)), "sha256": actual})
    pd = dataset.GetPointData()
    velocity = vtk_to_numpy(pd.GetArray("Velocity")).astype(np.float64)
    rho = vtk_to_numpy(pd.GetArray("Density")).astype(np.float64)
    momentum = vtk_to_numpy(pd.GetArray("Momentum")).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_speed = np.linalg.norm(momentum, axis=1) / rho
    canonical_speed = np.linalg.norm(velocity, axis=1)
    blank = pd.GetArray("IBlank")
    valid = vtk_to_numpy(blank) > 0 if blank is not None else np.ones(len(rho), dtype=bool)
    coordinates = vtk_to_numpy(dataset.GetPoints().GetData()).astype(np.float64)
    dimensions = [0, 0, 0]
    dataset.GetDimensions(dimensions)
    results = {}
    for method, speed in [("canonical_reader_velocity", canonical_speed),
                          ("stored_momentum_norm_over_density_float64", raw_speed)]:
        for population, mask in [("all_finite", np.ones(len(rho), dtype=bool)), ("iblank_valid", valid)]:
            results[method + ":" + population] = summarize(speed, mask, coordinates, dimensions)
    record = {"role": "INDEPENDENT_REFERENCE_CONSTRUCTION_AUDIT; NO_ANSWER_INPUT",
              "dataset_id": dataset_id, "input_files": files,
              "script_sha256": checksum(__file__), "dimensions": dimensions,
              "iblank_excluded": int((~valid).sum()), "density_zero": int((rho == 0).sum()),
              "density_negative": int((rho < 0).sum()), "results": results,
              "scientific_status": "COMPUTED_FOR_AUDIT; NOT_AUTOMATICALLY_ADDED_TO_SCORING"}
    output = ROOT / "outputs/model_answer_evaluation_v4/reference_construction/plot3d_population_audit.json"
    if output.exists():
        raise ValueError("preserve prior reference audit; inspect before replacing")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()

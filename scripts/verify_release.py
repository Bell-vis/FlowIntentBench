#!/usr/bin/env python3
"""Verify packaged cases and real numerical inputs without credentials or outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.expansion_evaluation import load_development_case


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(read_data=False):
    manifest = json.loads((ROOT / "experiments/expansion_v1_development/case_manifest.json").read_text())
    for row in manifest["cases"]:
        load_development_case(ROOT, row)
    userstudy = json.loads((ROOT / "experiments/userstudy/case_manifest.json").read_text())
    bindings = 0
    for row in userstudy["cases"]:
        for key, value in row.items():
            checksum = row.get(key.removesuffix("_path") + "_sha256") if key.endswith("_path") else None
            if value and checksum:
                path = ROOT / value
                if digest(path) != checksum:
                    raise ValueError(f"Case checksum mismatch: {value}")
                bindings += 1
    files = 0
    datasets = []
    for path in sorted((ROOT / "datasets").glob("*/dataset_manifest.json")):
        data = json.loads(path.read_text())
        for record in data["files"] + data.get("auxiliary_assets", []):
            target = ROOT / data.get("file_root", "datasets") / record["path"]
            if target.stat().st_size != record["size_bytes"] or digest(target) != record["checksum"]:
                raise ValueError(f"Dataset checksum/size mismatch: {record['path']}")
            files += 1
        summary = {"dataset_id": data["dataset_id"]}
        if read_data:
            from flowintentbench.deterministic_materialization import _reader_dataset
            mesh, _ = _reader_dataset(ROOT, data["dataset_id"])
            summary.update(points=mesh.GetNumberOfPoints(), cells=mesh.GetNumberOfCells())
            if summary["points"] <= 0 or summary["cells"] <= 0:
                raise ValueError(f"Empty dataset: {data['dataset_id']}")
        datasets.append(summary)
    return {"status": "PASS", "expansion_cases": len(manifest["cases"]),
            "userstudy_cases": len(userstudy["cases"]), "userstudy_bindings": bindings,
            "verified_data_files": files, "datasets": datasets}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-data", action="store_true", help="Also open all numerical data with VTK")
    args = parser.parse_args()
    print(json.dumps(verify(args.read_data), indent=2))

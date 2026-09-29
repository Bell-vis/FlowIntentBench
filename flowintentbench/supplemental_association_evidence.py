"""Independent association diagnostics; never judgments or reference findings."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .external_file_evaluator import digest


def method_sources(trajectory):
    """Expose successful answer-side method code, not stdout or model identity."""
    if trajectory is None:
        return None
    from .execution_evidence import successful_code_projection
    raw, events, source_sha = successful_code_projection(Path(trajectory))
    succeeded = {event.get("canonical_call_id") for event in events
                 if event.get("event") == "python_execution" and event.get("success") is True}
    sources = []
    for event in events:
        for call in event.get("tool_batch", {}).get("calls", []):
            if call.get("canonical_call_id") not in succeeded:
                continue
            code = call.get("arguments", {}).get("code", "")
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            names = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                     for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, (ast.Attribute, ast.Name))}
            if not names & {"corrcoef", "pearsonr", "spearmanr"}:
                continue
            sources.append({"code": code, "code_sha256": hashlib.sha256(code.encode()).hexdigest()})
    if not sources:
        return None
    return {"trajectory_sha256": hashlib.sha256(raw).hexdigest(), "source_journal_sha256": source_sha,
            "successful_method_blocks": sources,
            "scope": "Same answer's recorded method declarations only. Never execute this code or trust its printed/self-reported values. "
                     "Successful block completion does not prove a nested branch ran. Bind conventions from explicit code and source evidence, "
                     "not from agreement with the independent diagnostics. Unresolved method binding stays PENDING."}


def _pearson(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(pearsonr(x, y).statistic)


def association_diagnostics(dataset, recipe):
    """Fixed, claim-independent panel on complete rectilinear 3-D meshes.

    Alternative conventions are labeled, not chosen to match answer values.
    Reviewers must establish which, if any, the submitted method actually uses.
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy
    if not dataset.IsA("vtkRectilinearGrid"):
        raise ValueError("association diagnostics require a rectilinear grid")
    dimensions = dataset.GetDimensions()
    if min(dimensions) < 2:
        raise ValueError("association diagnostics require a complete 3-D mesh")
    coordinates = [vtk_to_numpy(getter()).astype(np.float64)
                   for getter in (dataset.GetXCoordinates, dataset.GetYCoordinates, dataset.GetZCoordinates)]
    if any(not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0) for axis in coordinates):
        raise ValueError("association diagnostics require finite increasing coordinates")
    dx, dy, dz = (np.diff(axis) for axis in coordinates)
    volumes = dz[:, None, None] * dy[None, :, None] * dx[None, None, :]

    def field(spec):
        if set(spec) - {"name", "association", "component"}:
            raise ValueError("unsupported supplemental field reduction")
        association = spec["association"]
        if association not in {"point", "cell"}:
            raise ValueError("unsupported supplemental field association")
        data = dataset.GetPointData() if association == "point" else dataset.GetCellData()
        if sum(data.GetArrayName(i) == spec["name"] for i in range(data.GetNumberOfArrays())) != 1:
            raise ValueError("missing or ambiguous supplemental field")
        values = vtk_to_numpy(data.GetArray(spec["name"])).astype(np.float64)
        if "component" in spec:
            component = spec["component"]
            if type(component) is not int or values.ndim != 2 or not 0 <= component < values.shape[1]:
                raise ValueError("invalid supplemental vector component")
            values = values[:, component]
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("supplemental fields must be finite scalars")
        if association == "cell":
            return values.reshape(volumes.shape), None
        points = values.reshape(tuple(reversed(dimensions)))
        cells = sum(points[k:k + volumes.shape[0], j:j + volumes.shape[1], i:i + volumes.shape[2]]
                    for k in (0, 1) for j in (0, 1) for i in (0, 1)) / 8.0
        return cells, values

    x, point_x = field(recipe["field"])
    y, point_y = field(recipe["other_field"])
    weights = volumes.ravel() / volumes.sum()
    flat_x, flat_y = x.ravel(), y.ravel()
    mx, my = np.dot(weights, flat_x), np.dot(weights, flat_y)
    covariance = np.dot(weights, (flat_x - mx) * (flat_y - my))
    result = {
        "domain": "All stored rectilinear cells; no masking or extrapolation; VTK x-fastest ordering",
        "point_to_cell": "Arithmetic mean of eight vertex scalar/component values in float64",
        "cell_count": int(volumes.size), "total_included_coordinate_volume": float(volumes.sum()),
        "volume_weighted_covariance": float(covariance),
        "covariance_definition": "sum(V*(X-mean_V(X))*(Y-mean_V(Y)))/sum(V)",
        "equal_weight_cell_pearson": _pearson(flat_x, flat_y),
        "equal_weight_cell_spearman": (float(spearmanr(flat_x, flat_y).statistic)
                                        if np.ptp(flat_x) and np.ptp(flat_y) else None),
        "spearman_definition": "Equal cell weights; ordinary average ranks for ties",
        "equal_weight_stored_point_pearson": (_pearson(point_x, point_y)
                                               if point_x is not None and point_y is not None else None),
        "auxiliary_conventions": [],
    }
    for precision in ("float64", "float32_roundtrip"):
        a, b = ((x, y) if precision == "float64" else
                (x.astype(np.float32).astype(np.float64), y.astype(np.float32).astype(np.float64)))
        ma, mb = a.mean(), b.mean()
        centered = (a - ma) * (b - mb)
        quadrant = ((a > ma) & (b < mb)) | ((a < ma) & (b > mb))
        panel = {"precision": precision,
                 "opposite_quadrants_fraction": float(quadrant.mean()),
                 "quadrant_definition": "Strict opposite signs relative to complete-mesh equal-cell means of X and Y; ties excluded",
                 "layer_regions": []}
        for physical_axis in range(3):
            storage_axis = 2 - physical_axis
            horizontal = tuple(axis for axis in range(3) if axis != storage_axis)
            profile = a.mean(axis=horizontal)
            if np.ptp(profile) == 0:
                continue
            scaled = (profile - profile.min()) / np.ptp(profile)
            selected = (scaled > 0.01) & (scaled < 0.99)
            shape = [1, 1, 1]
            shape[storage_axis] = len(selected)
            mask = np.broadcast_to(selected.reshape(shape), a.shape)
            total_covariance = centered.sum()
            centers = (coordinates[physical_axis][:-1] + coordinates[physical_axis][1:]) / 2
            panel["layer_regions"].append({
                "physical_axis": physical_axis, "thresholds": [0.01, 0.99],
                "definition": "Strict 1%-99% min-max normalized equal-cell layer mean of X; layer means use all transverse cells",
                "selected_layer_count": int(selected.sum()),
                "selected_layer_center_extent": ([float(centers[selected].min()), float(centers[selected].max())]
                                                   if selected.any() else None),
                "equal_cell_covariance_share": (float(centered[mask].sum() / total_covariance)
                                                 if total_covariance else None),
                "covariance_share_definition": "Selected contribution to complete-mesh equal-cell covariance, centered at global means",
                "within_region_equal_cell_pearson": _pearson(a[mask], b[mask]),
            })
        result["auxiliary_conventions"].append(panel)
    return result


def materialize_association_diagnostics(root, row, exchange, execution_provenance):
    """Cache independent raw-data calculations, not reviewer conclusions."""
    from .deterministic_materialization import _reader_dataset
    from .expansion_evaluation import file_sha256, read_json
    from .expansion_recipe_materializer import cached_recipe_result

    provenance = execution_provenance or {}
    recipe = provenance.get("recipe") or provenance.get("materialization_plan", {}).get("recipe", {})
    if recipe.get("kind") != "association" or recipe.get("measure") not in {"pearson", "spearman"}:
        return None
    root, exchange = Path(root).resolve(), Path(exchange).resolve()
    dataset_root = (root / "datasets" / row["dataset_id"]).resolve()
    if not dataset_root.is_relative_to(root / "datasets"):
        raise ValueError("supplemental dataset escapes repository")
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    expected = provenance.get("dataset_manifest_sha256")
    if expected != file_sha256(manifest_path):
        raise ValueError("supplemental dataset manifest binding mismatch")
    inputs = []
    for entry in manifest["files"]:
        path = (root / manifest.get("file_root", "datasets") / entry["path"]).resolve()
        if not path.is_relative_to(root) or entry.get("checksum_algorithm") != "sha256":
            raise ValueError("unsupported supplemental input authority")
        if file_sha256(path) != entry.get("checksum"):
            raise ValueError("supplemental raw input digest mismatch")
        inputs.append({"declared_path": entry["path"], "sha256": entry["checksum"]})
    identity = {"version": "independent-association-diagnostics-v1", "recipe": recipe,
                "dataset_manifest_sha256": expected, "inputs": inputs,
                "source_sha256": file_sha256(Path(__file__)),
                "reader_sha256": file_sha256(Path(__file__).with_name("deterministic_materialization.py")),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "vtk")}}
    def compute():
        dataset, _ = _reader_dataset(root, row["dataset_id"])
        try:
            values = association_diagnostics(dataset, recipe)
        except ValueError as exc:
            return {"status": "UNSUPPORTED", "reason": str(exc)}
        return {"status": "MATERIALIZED", "statistics": values}
    result = cached_recipe_result(exchange / "supplemental_association", identity, compute)
    return {"calculation_id": digest(identity), "provenance": identity, **result,
            "applicability": "Fixed claim-independent diagnostic panel, not scientific acceptance. Only compare a statistic after "
                             "establishing the same domain, fields, precision, weighting, region and thresholds from explicit source evidence. "
                             "Do not choose a convention based on numerical agreement or infer missing declarations from this panel. "
                             "Coordinate volume and covariance retain native stored scales; no undeclared SI units are assigned."}

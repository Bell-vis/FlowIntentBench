"""Offline reference-construction/audit utility; NEVER called by evaluation.

Independent, bounded calculations for explicitly bound auxiliary claims.

The reviewer selects a statistic and quotes its method; it never supplies the
expected result. These computations read checksum-bound data and execute only
host code. A successful calculation is conditional on the semantic binding,
which remains visible in the audit. No new reference/required role is created.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt

from .answer_evidence import bind_quote, bind_value, rounding_radius, value_is_bound, numeric_notation

VERSION = "auxiliary-numeric-verification-v1"


from .frozen_evidence import NumericCheck


def catalog(material):
    branches = {}
    for bid, entry in material.get("branch_execution_evidence", {}).items():
        recipe = entry.get("execution_provenance", {}).get("recipe", {})
        if recipe.get("kind") == "association" and recipe.get("measure") == "pearson":
            branches[bid] = {"dataset": entry["dataset_id"], "x": recipe["field"], "y": recipe["other_field"]}
    if not branches:
        return None
    return {"branches": branches, "population": "Complete rectilinear cells; arithmetic vertex means for point fields; volume weights, float64.",
        "statistics": "mean_x/y, std_x/y, covariance, pearson, r_squared, regression_slope/intercept; conditional_mean_y/conditional_max_abs_y with strict lower_x/upper_x; layer_pearson_min/max/median across ALL layers of explicit axis.",
        "binding": "Only bind an explicitly identical population, weighting, fields and method. Read preceding paragraphs/bullet headings for regional scope: whole-mesh O does not imply whole-mesh auxiliary findings. No subsamples, point statistics, inferred thresholds, masked layers or undeclared units. Quote method and population scope separately; scope_status must be EXPLICIT, otherwise leave unresolved. One scalar check per value component; leave unsupported claims without checks."}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def _arrays(path, file_sha, recipe_json):
    import numpy as np
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkDataSetReader
    if _sha(path) != file_sha:
        raise ValueError("auxiliary raw input checksum mismatch")
    reader = vtkDataSetReader()
    reader.SetFileName(path)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    mesh = reader.GetOutput()
    if not mesh.IsA("vtkRectilinearGrid") or min(mesh.GetDimensions()) < 2:
        raise ValueError("auxiliary calculations require a complete 3-D rectilinear grid")
    axes = [vtk_to_numpy(getter()).astype(float) for getter in
            (mesh.GetXCoordinates, mesh.GetYCoordinates, mesh.GetZCoordinates)]
    if any(not np.isfinite(a).all() or np.any(np.diff(a) <= 0) for a in axes):
        raise ValueError("invalid coordinates")
    dx, dy, dz = (np.diff(a) for a in axes)
    volume = dz[:, None, None] * dy[None, :, None] * dx[None, None, :]
    recipe = json.loads(recipe_json)
    def field(spec):
        if set(spec) - {"name", "association", "component"} or spec["association"] not in {"point", "cell"}:
            raise ValueError("unsupported field mapping")
        data = mesh.GetPointData() if spec["association"] == "point" else mesh.GetCellData()
        raw = data.GetArray(spec["name"])
        if raw is None:
            raise ValueError("missing field")
        a = vtk_to_numpy(raw).astype(float)
        if "component" in spec:
            c = spec["component"]
            if type(c) is not int or a.ndim != 2 or not 0 <= c < a.shape[1]:
                raise ValueError("invalid field component")
            a = a[:, c]
        if a.ndim != 1 or not np.isfinite(a).all():
            raise ValueError("nonfinite or nonscalar field")
        if spec["association"] == "cell":
            return a.reshape(volume.shape)
        a = a.reshape(tuple(reversed(mesh.GetDimensions())))
        nz, ny, nx = volume.shape
        return sum(a[k:k+nz, j:j+ny, i:i+nx] for k in (0, 1) for j in (0, 1) for i in (0, 1)) / 8
    return field(recipe["field"]), field(recipe["other_field"]), volume


def _correlation(x, y, w):
    import numpy as np
    xx, yy = x - np.average(x, weights=w), y - np.average(y, weights=w)
    den = np.sqrt(np.sum(w * xx * xx) * np.sum(w * yy * yy))
    if not den:
        raise ValueError("undefined correlation")
    return float(np.sum(w * xx * yy) / den)


@lru_cache(maxsize=256)
def _compute(path, file_sha, recipe_json, statistic, axis, lower, upper):
    import numpy as np
    x, y, w = _arrays(path, file_sha, recipe_json)
    if statistic.startswith("layer_"):
        storage_axis = {"x": 2, "y": 1, "z": 0}[axis]
        values = [_correlation(np.take(x, i, axis=storage_axis), np.take(y, i, axis=storage_axis),
                               np.take(w, i, axis=storage_axis)) for i in range(x.shape[storage_axis])]
        return float({"layer_pearson_min": np.min, "layer_pearson_max": np.max,
                      "layer_pearson_median": np.median}[statistic](values))
    if statistic.startswith("conditional_"):
        mask = np.ones(x.shape, dtype=bool)
        if lower is not None:
            mask &= x > lower
        if upper is not None:
            mask &= x < upper
        if not mask.any():
            raise ValueError("empty condition")
        return float(np.average(y[mask], weights=w[mask]) if statistic == "conditional_mean_y" else np.max(np.abs(y[mask])))
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    vx, vy = np.average((x-mx)**2, weights=w), np.average((y-my)**2, weights=w)
    cov = np.average((x-mx)*(y-my), weights=w)
    if statistic in {"pearson", "r_squared", "regression_slope", "regression_intercept"} and (not vx or not vy):
        raise ValueError("undefined normalized statistic")
    return float({"mean_x": mx, "mean_y": my, "std_x": np.sqrt(vx), "std_y": np.sqrt(vy),
                  "covariance": cov, "pearson": cov/np.sqrt(vx*vy) if vx*vy else 0,
                  "r_squared": cov*cov/(vx*vy) if vx*vy else 0,
                  "regression_slope": cov/vx if vx else 0, "regression_intercept": my-cov/vx*mx if vx else 0}[statistic])


def verify_checks(root, material, answer, claim, equivalent_branches, coordinate_axes=None):
    """Return local scientific truth plus complete provenance for a claim.

False is conclusive if one correctly bound scalar is contradicted. True needs
every extracted scalar component verified; partial success remains unknown.
    Auxiliary values have an explicit policy: reproduce the reported decimal
    precision (at least two significant digits), independent of frozen core GT.
"""
    queries = claim.numeric_checks
    if not queries:
        return None, []
    if len({q.branch_id for q in queries}) != 1:
        return None, [{"query": q.model_dump(mode="json"), "verdict": None,
                       "reason": "AUXILIARY_CHECK_UNRESOLVED: split different result groups before binding"} for q in queries]
    binding = bind_value(answer, claim.evidence_text, claim.value)
    values = claim.value if isinstance(claim.value, list) else [claim.value]
    rows, covered = [], set()
    for query in queries:
        row = {"query": query.model_dump(mode="json"), "verdict": None, "protocol": VERSION}
        try:
            if binding["status"] != "BOUND":
                raise ValueError(binding["reason"])
            if root is None or query.branch_id not in equivalent_branches:
                raise ValueError("method not independently bound to a supported branch")
            method = bind_quote(answer, query.method_evidence_text)
            if not method:
                raise ValueError("method quotation unbound")
            if query.scope_status != "EXPLICIT" or not bind_quote(answer, query.scope_evidence_text):
                raise ValueError("population scope unresolved or unbound")
            index = query.value_index if isinstance(claim.value, list) else 0
            axis = query.axis
            repairs = []
            # Recover omitted JSON selectors only from explicit source labels,
            # never from which independently computed number happens to match.
            if query.statistic.startswith("layer_"):
                source = numeric_notation(claim.evidence_text)
                if isinstance(claim.value, list) and index is None:
                    number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
                    limits = re.search(r"\brange\s+"+number+r"\s+to\s+"+number, source, re.I)
                    median = re.search(r"\bmedian\s+"+number, source, re.I)
                    value_from_label = None
                    if limits and query.statistic in {"layer_pearson_min", "layer_pearson_max"}:
                        value_from_label = (min if query.statistic.endswith("min") else max)(float(v) for v in limits.groups())
                    elif median and query.statistic == "layer_pearson_median":
                        value_from_label = float(median[1])
                    positions = [i for i, value in enumerate(values) if value == value_from_label]
                    if value_from_label is not None and len(positions) == 1:
                        index = positions[0]
                        repairs.append("VALUE_INDEX_FROM_EXPLICIT_RANGE_OR_MEDIAN_LABEL")
                axis_source = query.scope_evidence_text + '\n' + query.method_evidence_text
                axes = set(re.findall(r"\b([xyz])[- ](?:planes?|layers?|levels?)\b", axis_source, re.I))
                if re.search(r"\bvertical (?:planes?|layers?|levels?)\b", axis_source, re.I):
                    axes.update(a for a, meaning in (coordinate_axes or {}).items() if isinstance(meaning, str) and re.search(r"\bvertical\b", meaning, re.I))
                axes = {a.lower() for a in axes}
                if len(axes) == 1:
                    source_axis = next(iter(axes))
                    if axis is not None and axis != source_axis:
                        raise ValueError("axis contradicts quoted scope and public coordinate metadata")
                    if axis is None:
                        axis = source_axis
                        repairs.append("AXIS_FROM_QUOTED_SCOPE_AND_PUBLIC_COORDINATES")
            row.update(resolved_value_index=index, resolved_axis=axis, selector_repairs=repairs)
            if (isinstance(claim.value, list) and index is None
                    or not isinstance(claim.value, list) and query.value_index is not None
                    or type(index) is not int or not 0 <= index < len(values)):
                raise ValueError("invalid scalar component")
            value = values[index]
            if type(value) not in {int, float}:
                raise ValueError("nonnumeric extracted component")
            if claim.unit is not None:
                dimensionless = query.statistic in {"pearson", "r_squared"} or query.statistic.startswith("layer_")
                allowed = {"1", "dimensionless", "unitless"} if dimensionless else {
                    "stored units", "native units", "stored units as applicable", "stored velocity units"}
                if claim.unit not in allowed:
                    raise ValueError("supplemental unit conversion not established")
            conditional = query.statistic.startswith("conditional_")
            if query.statistic.startswith("layer_") != (axis is not None):
                raise ValueError("axis required only for layer statistics")
            if conditional != (query.lower_x is not None or query.upper_x is not None):
                raise ValueError("conditional statistics require explicit bounds only")
            if query.lower_x is not None and query.upper_x is not None and query.lower_x >= query.upper_x:
                raise ValueError("invalid condition bounds")
            for threshold in (query.lower_x, query.upper_x):
                if threshold is not None and not value_is_bound(threshold, method["text"] + '\n' + claim.evidence_text):
                    raise ValueError("condition threshold not present in source")
            entry = material["branch_execution_evidence"][query.branch_id]
            provenance = entry["execution_provenance"]
            recipe = provenance["recipe"]
            if recipe.get("kind") != "association" or recipe.get("measure") != "pearson":
                raise ValueError("unsupported branch recipe")
            dataset = Path(root) / "datasets" / entry["dataset_id"] / "dataset_manifest.json"
            if _sha(dataset) != provenance["dataset_manifest_sha256"]:
                raise ValueError("auxiliary manifest checksum mismatch")
            inputs = provenance["input_files"]
            if len(inputs) != 1:
                raise ValueError("unsupported multi-file dataset")
            path = (Path(root) / provenance.get("data_file_root", "datasets") / inputs[0]["path"]).resolve()
            if not path.is_relative_to(Path(root).resolve()) or path.suffix != ".vtk":
                raise ValueError("unsupported data path")
            # Revalidate input even when the numerical cache is warm.
            if _sha(path) != inputs[0]["sha256"]:
                raise ValueError("auxiliary raw input checksum mismatch")
            expected = _compute(str(path), inputs[0]["sha256"], json.dumps(recipe, sort_keys=True),
                                query.statistic, axis, query.lower_x, query.upper_x)
            if not math.isfinite(expected):
                raise ValueError("nonfinite independent statistic")
            radius = rounding_radius(value, binding["source"]["text"], allow_integer=True, min_digits=2)
            residual = abs(value-expected)
            numerical_epsilon = 1e-10 * max(abs(expected), 1e-8)
            verdict = residual <= numerical_epsilon + (radius or 0.0)
            row.update(verdict=verdict, expected=expected, reported=value, rounding_radius=radius,
                       reason="INDEPENDENT_REPORTED_PRECISION_MATCH" if verdict else "INDEPENDENT_NUMERIC_CONTRADICTION",
                       verification_policy="AUXILIARY_REPORTED_PRECISION_V1; two or more significant digits; no alteration of frozen core tolerance",
                       data_sha256=inputs[0]["sha256"], recipe=recipe, source_sha256=_sha(Path(__file__)),
                       semantic_binding="REVIEWER_QUOTED_METHOD; not proved by numerical agreement")
            if verdict is True:
                covered.add(index)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            row["reason"] = "AUXILIARY_CHECK_UNRESOLVED: " + str(exc)
        rows.append(row)
    if any(r["verdict"] is False for r in rows):
        return False, rows
    return (True if len(covered) == len(values) else None), rows

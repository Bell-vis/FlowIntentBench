"""Opt-in novel-O recipes; frozen construction recipes remain byte-identical.

Only documented runtime controls dispatch here. Data-derived answers are
computed from reader arrays, never reported Findings or calibration cutoffs.
"""
from __future__ import annotations

import math
import numpy as np

from .construction_recipes import _cell_view, execute_recipes as execute_frozen_recipes


def _validate_point_statistics_controls(recipe):
    """Validate the complete opt-in contract, including direct executor calls."""
    controls = {"point_domain", "ddof", "cv_denominator", "relative_denominator", "quantile_method"}
    if recipe.get("sampling", "cell_volume") != "point_equal":
        if controls & recipe.keys():
            raise ValueError("point statistics controls require point_equal sampling")
        return
    allowed = {"kind", "field", "measure", "sampling", "point_domain"}
    if recipe.get("point_domain", "all") not in {"all", "finite"}:
        raise ValueError("point domain must be all or finite; ROI is unsupported")
    if recipe["kind"] == "association":
        allowed.add("other_field")
        if recipe.get("measure") not in {"pearson", "spearman"}:
            raise ValueError("unsupported point association measure")
    elif recipe["kind"] == "dispersion":
        if recipe.get("measure") == "cv":
            allowed |= {"ddof", "cv_denominator"}
            if (type(recipe.get("ddof")) is not int or recipe["ddof"] not in {0, 1}
                    or recipe.get("cv_denominator") not in {"mean", "absolute_mean"}):
                raise ValueError("point CV requires explicit ddof (0 or 1) and cv_denominator")
        elif recipe.get("measure") == "relative_interdecile":
            allowed |= {"relative_denominator", "quantile_method"}
            if (recipe.get("relative_denominator") not in {"median", "absolute_median"}
                    or recipe.get("quantile_method") not in {"linear", "nearest_rank"}):
                raise ValueError("point relative dispersion requires explicit denominator and quantile method")
        else:
            raise ValueError("unsupported point dispersion measure")
    else:
        raise ValueError("unsupported point statistics kind")
    if set(recipe) - allowed:
        raise ValueError("unsupported or ignored point statistics controls")


def _point_statistics(dataset, definition, recipe):
    """Equal stored observations; reduce vectors before masking or statistics."""
    from vtkmodules.util.numpy_support import vtk_to_numpy

    _validate_point_statistics_controls(recipe)
    if definition.get("finite_cell_fields"):
        raise ValueError("point sampling cannot inherit a cell field domain")
    count = dataset.GetNumberOfPoints()
    if not count or dataset.GetPoints() is None:
        raise ValueError("point statistics require stored points")
    points = vtk_to_numpy(dataset.GetPoints().GetData())
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError("point statistics require complete finite coordinates")

    def field(spec):
        if spec.get("association") != "point":
            raise ValueError("point statistics require stored point fields; reconstruction is unsupported")
        if (set(spec) - {"name", "association", "component", "reduce"}
                or {"component", "reduce"} <= spec.keys()):
            raise ValueError("unsupported or ambiguous point field reduction")
        data = dataset.GetPointData()
        if sum(data.GetArrayName(i) == spec["name"] for i in range(data.GetNumberOfArrays())) != 1:
            raise ValueError("missing or ambiguous stored point field")
        array = data.GetArray(spec["name"])
        if array is None:
            raise ValueError("point field must be numeric")
        values = vtk_to_numpy(array).astype(float)
        if "component" in spec:
            component = spec["component"]
            if (values.ndim != 2 or type(component) is not int
                    or not 0 <= component < values.shape[1]):
                raise ValueError("unsupported point vector component")
            values = values[:, component]
        elif "reduce" in spec:
            if values.ndim != 2 or spec["reduce"] != "magnitude":
                raise ValueError("unsupported point vector reduction")
            values = np.linalg.norm(values, axis=1)
        if values.shape != (count,):
            raise ValueError("point field must be a colocated scalar or explicitly reduced vector")
        return values

    x = field(recipe["field"])
    y = field(recipe["other_field"]) if recipe["kind"] == "association" else None
    finite = np.isfinite(x)
    if y is not None:
        finite &= np.isfinite(y)
    if recipe.get("point_domain", "all") == "all" and not finite.all():
        raise ValueError("all-point statistics require complete finite fields")
    x = x[finite]
    if not len(x):
        raise ValueError("finite point domain is empty")
    if y is not None:
        y = y[finite]
        if recipe["measure"] == "spearman":
            from scipy.stats import rankdata

            x, y = rankdata(x, method="average"), rankdata(y, method="average")
        dx, dy = x - x.mean(), y - y.mean()
        denominator = np.sqrt(np.mean(dx * dx) * np.mean(dy * dy))
        if denominator == 0:
            raise ValueError("association is undefined for constant fields")
        value = np.mean(dx * dy) / denominator
    elif recipe["measure"] == "cv":
        if len(x) <= recipe["ddof"]:
            raise ValueError("point CV requires more samples than ddof")
        denominator = x.mean()
        if recipe["cv_denominator"] == "absolute_mean":
            denominator = abs(denominator)
        if denominator == 0:
            raise ValueError("CV is undefined for zero mean")
        value = np.std(x, ddof=recipe["ddof"]) / denominator
    else:
        method = "linear" if recipe["quantile_method"] == "linear" else "inverted_cdf"
        lower, median, upper = np.quantile(x, [.1, .5, .9], method=method)
        denominator = abs(median) if recipe["relative_denominator"] == "absolute_median" else median
        if denominator == 0:
            raise ValueError("relative interdecile spread requires nonzero median")
        value = (upper - lower) / denominator
    return {"value": float(value)}


def _normalize_point_mixture_range_recipe(recipe):
    """Resolve only a provably inactive omitted clipping control.

    For 0 < a < b < 1, a <= clip(c, 0, 1) <= b iff a <= c <= b.
    Stored-density and uniform weights on that selected set also do not use c.
    This equivalence does not supply any other missing scientific choice.
    """
    required = {"kind", "field", "weight_transform", "sampling", "axis", "measure",
                "endmembers", "mixture_mask", "center_weight", "center_domain"}
    allowed = required | {"clip_mixture", "tail_planes", "tail_bounds"}
    if set(recipe) - allowed:
        raise ValueError("unsupported or ignored point range controls")
    if not required <= recipe.keys():
        raise ValueError("point range requires explicit normalization, mask and center controls")
    if (recipe["kind"] != "weighted_width" or recipe["sampling"] != "point_equal"
            or recipe["measure"] != "range" or recipe["weight_transform"] != "mixture"):
        raise ValueError("point range requires point_equal mixture range controls")
    if (type(recipe["axis"]) is not int or recipe["axis"] not in {0, 1, 2}
            or recipe["center_weight"] not in {"density", "uniform", "mixture"}
            or recipe["center_domain"] not in {"all", "selected"}):
        raise ValueError("unsupported point range axis or center controls")

    def finite_pair(value):
        return (isinstance(value, (list, tuple)) and len(value) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v) for v in value)
                and value[0] < value[1])

    mask = recipe["mixture_mask"]
    if not finite_pair(mask) or not 0 <= mask[0] < mask[1] <= 1:
        raise ValueError("point range requires a finite increasing unit-interval mask")
    method = recipe["endmembers"]
    if isinstance(method, str):
        if method not in {"sample_extrema", "tail_plane_medians", "tail_coordinate_medians"}:
            raise ValueError("unsupported point endmember rule")
    elif not finite_pair(method):
        raise ValueError("mixture endmembers must be finite and increasing")
    if (method == "tail_plane_medians") != ("tail_planes" in recipe):
        raise ValueError("tail_planes must accompany only tail_plane_medians")
    if "tail_planes" in recipe and (type(recipe["tail_planes"]) is not int or recipe["tail_planes"] <= 0):
        raise ValueError("tail_planes must be a positive integer")
    if (method == "tail_coordinate_medians") != ("tail_bounds" in recipe):
        raise ValueError("tail_bounds must accompany only tail_coordinate_medians")
    if "tail_bounds" in recipe and not finite_pair(recipe["tail_bounds"]):
        raise ValueError("tail_bounds must be finite and increasing")
    if "clip_mixture" in recipe:
        if type(recipe["clip_mixture"]) is not bool:
            raise ValueError("explicit clip_mixture must be boolean")
        return recipe
    if not (0 < mask[0] < mask[1] < 1 and recipe["center_weight"] in {"density", "uniform"}
            and recipe["center_domain"] == "selected"):
        raise ValueError("point range requires explicit clipping unless it is provably inactive")
    return {**recipe, "clip_mixture": False}


def _point_mixture_range(dataset, recipe):
    """Explicit point-mask extent and independently specified first moment.

    Tail medians pool stored scalar samples, not horizontally averaged planes.
    No cell conversion, face extrapolation, or volume-dependent mask is used.
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy

    omitted_clipping = "clip_mixture" not in recipe
    recipe = _normalize_point_mixture_range_recipe(recipe)
    if recipe.get("weight_transform") != "mixture" or recipe["field"]["association"] != "point":
        raise ValueError("point mixture range requires a stored point mixture scalar")
    if set(recipe["field"]) != {"name", "association"}:
        raise ValueError("point mixture range requires a scalar field without reduction")
    array = dataset.GetPointData().GetArray(recipe["field"]["name"])
    if array is None:
        raise ValueError("missing point mixture scalar")
    x = vtk_to_numpy(array).astype(float)
    points = vtk_to_numpy(dataset.GetPoints().GetData()).astype(float)
    if x.ndim != 1 or points.shape != (len(x), 3) or not len(x) or not np.isfinite(x).all() or not np.isfinite(points).all():
        raise ValueError("point mixture range requires complete finite point data")
    coordinate = points[:, int(recipe["axis"])]
    method = recipe["endmembers"]
    if method == "tail_plane_medians":
        planes = np.unique(coordinate)
        count = recipe["tail_planes"]
        if count * 2 > len(planes):
            raise ValueError("endmember plane tails overlap")
        low = float(np.median(x[coordinate <= planes[count - 1]]))
        high = float(np.median(x[coordinate >= planes[-count]]))
    elif method == "tail_coordinate_medians":
        lower, upper = recipe["tail_bounds"]
        lower_values, upper_values = x[coordinate <= lower], x[coordinate >= upper]
        if not len(lower_values) or not len(upper_values):
            raise ValueError("endmember coordinate tail is empty")
        low, high = float(np.median(lower_values)), float(np.median(upper_values))
    elif method == "sample_extrema":
        low, high = float(x.min()), float(x.max())
    elif isinstance(method, (list, tuple)) and len(method) == 2:
        low, high = method
    else:
        raise ValueError("unsupported point endmember rule")
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("mixture endmembers must be finite and increasing")
    if omitted_clipping:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            span = high - low
            c = (x - low) / span
        if not np.isfinite(span) or not np.isfinite(c).all():
            raise ValueError("inactive clipping requires finite mixture normalization")
    else:
        c = (x - low) / (high - low)
    if recipe["clip_mixture"]:
        c = np.clip(c, 0, 1)
    lower, upper = recipe["mixture_mask"]
    selected = (c >= lower) & (c <= upper)
    if not selected.any():
        raise ValueError("intermediate-density mask is empty")
    center_domain = recipe["center_domain"]
    if center_domain not in {"all", "selected"}:
        raise ValueError("unknown center domain")
    included = selected if center_domain == "selected" else np.ones(len(x), dtype=bool)
    if recipe["center_weight"] == "mixture":
        mass = 4 * c * (1 - c)
    elif recipe["center_weight"] == "density":
        mass = x
    elif recipe["center_weight"] == "uniform":
        mass = np.ones(len(x))
    else:
        raise ValueError("unknown center weight")
    mass = mass[included]
    if not np.isfinite(mass).all() or (mass < 0).any() or mass.sum() <= 0:
        raise ValueError("center requires nonnegative weights with positive support")
    return {"width": float(np.ptp(coordinate[selected])),
            "center": float(np.average(coordinate[included], weights=mass))}


def _unweighted_upper_tail(dataset, definition, recipe):
    """Change percentile selection while retaining the source volume summaries."""
    method = recipe.get("quantile_method")
    if method not in {"linear", "nearest_rank"}:
        raise ValueError("unweighted percentile requires an explicit or evidenced estimator")
    weights, points, arrays = _cell_view(
        dataset, definition.get("cell_volume_convention", "positive_signed_volume"))
    valid = np.ones(len(weights), dtype=bool)
    for spec in definition.get("finite_cell_fields", []):
        values = arrays[(spec["association"], spec["name"])]
        finite = np.isfinite(values)
        valid &= finite.all(axis=1) if finite.ndim == 2 else finite
    spec = recipe["field"]
    x = arrays[(spec["association"], spec["name"])][valid]
    weights, points = weights[valid], points[valid]
    if "component" in spec:
        x = x[:, int(spec["component"])]
    elif x.ndim == 2:
        if spec.get("reduce") != "magnitude":
            raise ValueError("vector field requires explicit component or magnitude")
        x = np.linalg.norm(x, axis=1)
    if not len(x) or x.shape != weights.shape or not np.isfinite(x).all():
        raise ValueError("nonfinite or mismatched field in selected domain")
    cutoff = float(np.quantile(x, float(recipe["quantile"]),
                              method="inverted_cdf" if method == "nearest_rank" else "linear"))
    selected = x >= cutoff
    selected_weights = weights[selected]
    return {"fraction": float(selected_weights.sum() / weights.sum()),
            "centroid": np.average(points[selected], weights=selected_weights, axis=0).tolist(),
            "mean": float(np.average(x[selected], weights=selected_weights)), "cutoff": cutoff}


def execute_runtime_recipes(dataset, definition):
    results = {}
    for name, operation in definition["operations"].items():
        recipe = operation["recipe"]
        if recipe.get("sampling", "cell_volume") not in {"cell_volume", "point_equal"}:
            raise ValueError("unsupported sampling")
        if recipe["kind"] in {"association", "dispersion"}:
            _validate_point_statistics_controls(recipe)
        if recipe.get("sampling") == "point_equal":
            if recipe["kind"] in {"association", "dispersion"}:
                result = _point_statistics(dataset, definition, recipe)
            elif recipe["kind"] != "weighted_width" or definition.get("finite_cell_fields"):
                raise ValueError("point sampling cannot inherit a cell field domain or another recipe kind")
            else:
                result = _point_mixture_range(dataset, recipe)
        elif recipe["kind"] == "upper_tail" and recipe.get("quantile_weighting") == "unweighted":
            result = _unweighted_upper_tail(dataset, definition, recipe)
        else:
            if recipe["kind"] == "upper_tail" and (recipe.get("quantile_weighting", "cell_volume") != "cell_volume"
                    or recipe.get("quantile_method", "nearest_rank") != "nearest_rank"):
                raise ValueError("volume quantile requires the left cumulative inverse")
            result = execute_frozen_recipes(dataset, {**definition, "operations": {name: operation}})[name]
        if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in result.values()):
            raise ValueError("nonfinite computed result")
        results[name] = result
    return results

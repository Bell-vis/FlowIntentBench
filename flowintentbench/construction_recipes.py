"""Data-driven numerical recipes consumed by the existing construction pilot.

Recipes produce construction evidence, never scientific admission decisions.
All quantities are computed from reader-visible arrays; no stored answer is
accepted as a recipe result.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .case_design import (
    FindingOpenness,
    OperationalizationDimension,
    OperationalizationResponsibility,
)
from .construction_tolerances import absolute_tolerance_for_significant_figures
from .ground_truth import FindingImportance, ReferenceFinding, VerificationSpec


def weighted_quantile(values, weights, quantile):
    values, weights = np.asarray(values), np.asarray(weights)
    if (
        values.ndim != 1
        or values.shape != weights.shape
        or not len(values)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
        or not 0 <= quantile <= 1
    ):
        raise ValueError(
            "weighted quantile requires finite values, positive weights and a quantile in [0, 1]"
        )
    order = np.argsort(values, kind="stable")
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    index = np.searchsorted(np.cumsum(weights), quantile * weights.sum(), side="left")
    return float(values[min(int(index), len(values) - 1)])


def _cell_view(dataset, volume_convention="positive_signed_volume"):
    from vtkmodules.vtkFiltersCore import vtkCellCenters, vtkPointDataToCellData
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter
    from vtkmodules.util.numpy_support import vtk_to_numpy

    size = vtkCellSizeFilter()
    size.SetInputData(dataset)
    size.Update()
    weights = vtk_to_numpy(size.GetOutput().GetCellData().GetArray("Volume")).astype(
        float
    )
    if volume_convention == "absolute_signed_volume":
        weights = np.abs(weights)
    elif volume_convention != "positive_signed_volume":
        raise ValueError("unknown cell-volume convention")
    if not len(weights) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("recipe requires finite positive-volume 3D cells")
    centers = vtkCellCenters()
    centers.SetInputData(dataset)
    centers.Update()
    points = vtk_to_numpy(centers.GetOutput().GetPoints().GetData()).astype(float)
    converted = vtkPointDataToCellData()
    converted.SetInputData(dataset)
    converted.Update()
    arrays = {}
    for association, data in [
        ("cell", dataset.GetCellData()),
        ("point", converted.GetOutput().GetCellData()),
    ]:
        for index in range(data.GetNumberOfArrays()):
            array = data.GetArray(index)
            if array is not None:
                arrays[(association, data.GetArrayName(index))] = vtk_to_numpy(
                    array
                ).astype(float)
    return weights, points, arrays


def execute_recipes(dataset, definition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    weights, points, arrays = _cell_view(
        dataset, definition.get("cell_volume_convention", "positive_signed_volume")
    )

    valid_fields = definition.get("finite_cell_fields", [])
    if valid_fields:
        mask = np.ones(len(weights), dtype=bool)
        for spec in valid_fields:
            values = arrays[(spec["association"], spec["name"])]
            finite = np.isfinite(values)
            mask &= finite.all(axis=1) if finite.ndim == 2 else finite
        if not mask.any():
            raise ValueError("finite-domain selection is empty")
        weights, points = weights[mask], points[mask]
        arrays = {key: values[mask] for key, values in arrays.items()}

    def field(spec):
        array = arrays[(spec["association"], spec["name"])]
        if "component" in spec:
            array = array[:, int(spec["component"])]
        elif array.ndim == 2:
            if spec.get("reduce") != "magnitude":
                raise ValueError(
                    "vector field requires explicit component or magnitude"
                )
            array = np.linalg.norm(array, axis=1)
        if array.shape != weights.shape or not np.isfinite(array).all():
            raise ValueError("nonfinite or mismatched field in selected domain")
        return array

    def mean(array, w=weights):
        return float(np.dot(array, w) / w.sum())

    results = {}
    for name, operation in definition["operations"].items():
        recipe = operation["recipe"]
        kind = recipe["kind"]
        x = field(recipe["field"])
        if kind == "dispersion":
            if recipe["measure"] == "cv":
                average = mean(x)
                if average == 0:
                    raise ValueError("CV is undefined for zero mean")
                value = np.sqrt(mean((x - average) ** 2)) / abs(average)
            elif recipe["measure"] == "relative_interdecile":
                median = weighted_quantile(x, weights, 0.5)
                if median == 0:
                    raise ValueError(
                        "relative interdecile spread requires nonzero median"
                    )
                value = (
                    weighted_quantile(x, weights, 0.9)
                    - weighted_quantile(x, weights, 0.1)
                ) / abs(median)
            else:
                raise ValueError("unknown dispersion measure")
            result = {"value": float(value)}
        elif kind == "association":
            y = field(recipe["other_field"])
            if recipe["measure"] == "spearman":
                from scipy.stats import rankdata

                x, y = rankdata(x, method="average"), rankdata(y, method="average")
            elif recipe["measure"] != "pearson":
                raise ValueError("unknown association measure")
            dx, dy = x - mean(x), y - mean(y)
            denominator = np.sqrt(mean(dx * dx) * mean(dy * dy))
            if denominator == 0:
                raise ValueError("association is undefined for constant fields")
            result = {"value": float(mean(dx * dy) / denominator)}
        elif kind == "upper_tail":
            cutoff = weighted_quantile(x, weights, float(recipe["quantile"]))
            mask = x >= cutoff
            selected_weights = weights[mask]
            result = {
                "fraction": float(selected_weights.sum() / weights.sum()),
                "centroid": np.average(
                    points[mask], weights=selected_weights, axis=0
                ).tolist(),
                "mean": mean(x[mask], selected_weights),
                "cutoff": cutoff,
            }
        elif kind == "weighted_width":
            axis = int(recipe["axis"])
            if recipe.get("weight_transform") == "mixture":
                low, high = (
                    (float(x.min()), float(x.max()))
                    if recipe["endmembers"] == "sample_extrema"
                    else recipe["endmembers"]
                )
                if high <= low or x.min() < low - 1e-6 or x.max() > high + 1e-6:
                    raise ValueError("density is outside declared mixture endmembers")
                c = np.clip((x - low) / (high - low), 0, 1)
                mass = weights * 4 * c * (1 - c)
            else:
                if (x < 0).any():
                    raise ValueError("width requires a nonnegative weighting field")
                mass = weights * x
            if mass.sum() <= 0:
                raise ValueError("width has zero support")
            coordinate = points[:, axis]
            center = mean(coordinate, mass)
            if recipe["measure"] == "rms":
                width = np.sqrt(mean((coordinate - center) ** 2, mass))
            elif recipe["measure"] == "interdecile":
                keep = mass > 0
                width = weighted_quantile(
                    coordinate[keep], mass[keep], 0.9
                ) - weighted_quantile(coordinate[keep], mass[keep], 0.1)
            else:
                raise ValueError("unknown width measure")
            result = {"width": float(width), "center": center}
        elif kind == "advective_flux":
            y = field(recipe["other_field"])
            if recipe["product_rule"] == "cell_then_product":
                product = x * y
            elif recipe["product_rule"] == "point_then_cell":
                from vtkmodules.vtkFiltersCore import vtkPointDataToCellData
                from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

                first, second = recipe["field"], recipe["other_field"]
                if first["association"] != "point" or second["association"] != "point":
                    raise ValueError("pointwise flux rule requires two point fields")
                left = vtk_to_numpy(dataset.GetPointData().GetArray(first["name"]))
                right = vtk_to_numpy(dataset.GetPointData().GetArray(second["name"]))[
                    :, int(second["component"])
                ]
                copied = dataset.NewInstance()
                copied.ShallowCopy(dataset)
                temporary = numpy_to_vtk(left * right, deep=True)
                temporary.SetName("__reference_flux_product")
                copied.GetPointData().AddArray(temporary)
                conversion = vtkPointDataToCellData()
                conversion.SetInputData(copied)
                conversion.Update()
                product = vtk_to_numpy(
                    conversion.GetOutput().GetCellData().GetArray(temporary.GetName())
                )
            else:
                raise ValueError("unknown flux product rule")
            result = {"value": mean(product)}
        elif kind == "vector_alignment":
            first, second = recipe["field"], recipe["other_field"]
            left, right = (
                arrays[(first["association"], first["name"])],
                arrays[(second["association"], second["name"])],
            )
            products = np.sum(left * right, axis=1)
            if recipe["measure"] == "mean_cosine":
                norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
                if (norm <= 0).any():
                    raise ValueError("mean cosine is undefined at zero-vector cells")
                value = mean(products / norm)
            elif recipe["measure"] == "normalized_dot":
                norm = np.sqrt(
                    mean(np.sum(left * left, axis=1))
                    * mean(np.sum(right * right, axis=1))
                )
                if norm <= 0:
                    raise ValueError("normalized dot product has zero norm")
                value = mean(products) / norm
            else:
                raise ValueError("unknown vector alignment measure")
            result = {"value": float(value)}
        elif kind == "surface":
            from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkTriangleFilter
            from vtkmodules.util.numpy_support import vtk_to_numpy

            if recipe["field"]["association"] != "point":
                raise ValueError("surface recipe requires a stored point scalar")
            contour = vtkContourFilter()
            contour.SetInputData(dataset)
            contour.SetInputArrayToProcess(0, 0, 0, 0, recipe["field"]["name"])
            contour.SetValue(0, float(recipe["level"]))
            triangle = vtkTriangleFilter()
            triangle.SetInputConnection(contour.GetOutputPort())
            triangle.Update()
            output = triangle.GetOutput()
            if output.GetNumberOfPolys() == 0:
                raise ValueError("empty isosurface")
            vertices = vtk_to_numpy(output.GetPoints().GetData())
            connectivity = vtk_to_numpy(
                output.GetPolys().GetConnectivityArray()
            ).reshape(-1, 3)
            tris = vertices[connectivity]
            area = (
                np.linalg.norm(
                    np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
                )
                / 2
            )
            if area.sum() <= 0:
                raise ValueError("degenerate surface")
            result = {
                "area": float(area.sum()),
                "centroid": np.average(
                    tris.mean(axis=1), weights=area, axis=0
                ).tolist(),
            }
        else:
            raise ValueError(f"unregistered construction recipe {kind!r}")
        if not all(
            np.isfinite(np.asarray(value, dtype=float)).all()
            for value in result.values()
        ):
            raise ValueError("nonfinite computed Finding")
        results[name] = result
    return results


def authored_question(family, suffix):
    definition = family.definition
    authored = definition["questions"][suffix]
    unresolved = (
        ()
        if suffix.startswith("o1")
        else family.o2_unresolved
        if suffix == "o2_f1"
        else family.principal_dimensions
    )
    responsibility = {
        "o1": OperationalizationResponsibility.USER_SPECIFIED,
        "o2": OperationalizationResponsibility.PARTIALLY_SPECIFIED,
        "o3": OperationalizationResponsibility.MODEL_SELECTED,
    }[suffix[:2]]
    baseline = next(iter(definition["operations"].values()))["clauses"]
    clauses = {
        OperationalizationDimension(key): value
        for key, value in baseline.items()
        if OperationalizationDimension(key) not in unresolved
    }
    properties = definition["finding_properties"] if suffix != "o1_f2" else []
    methods = [
        {
            "category": "analysis_procedure"
            if key.value
            in {"aggregation_or_representation", "comparison_or_normalization"}
            else key.value,
            "statement": value,
            "question_fragment": value,
        }
        for key, value in clauses.items()
    ]
    return authored, {
        "scientific_target": definition["scientific_target"],
        "finding_goal": definition["finding_goal"],
        "responsibility": responsibility,
        "unresolved": unresolved,
        "openness": FindingOpenness.OPEN
        if suffix == "o1_f2"
        else FindingOpenness.BOUNDED,
        "methods": methods,
        "dimension_clauses": clauses,
        "requirements": [
            {
                "category": p["category"],
                "statement": p["requirement_statement"],
                "question_fragment": p["request_phrase"],
            }
            for p in properties
        ],
        "f1_properties": [p["property_id"] for p in properties],
        "scope": definition["scope"],
        "selection": definition["selection"],
        "selection_fragment": definition["selection"],
    }


def computed_findings(family, metadata, operation, result):
    findings, mapping = [], {}
    for property_spec in family.definition["finding_properties"]:
        property_id = property_spec["property_id"]
        value = result[property_spec["result_key"]]
        fid = f"{metadata.case_id}_{operation}_{property_id}"
        if isinstance(value, list):
            verification = VerificationSpec(
                spatial_tolerance=float(property_spec["spatial_tolerance"])
            )
        else:
            verification = VerificationSpec(
                absolute_tolerance=absolute_tolerance_for_significant_figures(
                    float(value), significant_figures=5
                )
            )
        findings.append(
            ReferenceFinding(
                finding_id=fid,
                category=property_spec["category"],
                statement=property_spec["requirement_statement"],
                importance=(
                    FindingImportance.SUPPORTING
                    if metadata.finding_openness == FindingOpenness.OPEN
                    and property_id
                    not in family.definition.get(
                        "f2_core_properties",
                        [
                            p["property_id"]
                            for p in family.definition["finding_properties"]
                        ],
                    )
                    else FindingImportance.CORE
                ),
                value=value,
                unit=property_spec.get("unit"),
                verification=verification,
                evidence_ids=list(family.definition.get("finding_evidence_ids", [])),
            )
        )
        mapping[property_id] = fid
    return findings, mapping

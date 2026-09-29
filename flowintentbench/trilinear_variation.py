"""Volume-normalized total variation of trilinear scalar fields."""
from itertools import product

import numpy as np


def normalized_trilinear_variation(axes, values, *, relative_tolerance=1e-4):
    """Integrate on an axis-aligned grid; values has shape (nz, ny, nx, fields)."""
    x, y, z = [np.asarray(a, dtype=float) for a in axes]
    values = np.asarray(values, dtype=float)
    if values.shape[:3] != (len(z), len(y), len(x)) or values.ndim != 4:
        raise ValueError("point fields and structured grid dimensions differ")
    if not np.isfinite(values).all():
        raise ValueError("trilinear fields must have finite vertex values")
    if any(len(a) < 2 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0) for a in (x, y, z)):
        raise ValueError("trilinear integral requires positive finite cell widths")
    if np.any(np.ptp(values.reshape(-1, values.shape[-1]), axis=0) == 0):
        raise ValueError("normalized total variation is undefined for a constant field")
    nz, ny, nx = values.shape[:3]
    corners = list(product((0, 1), repeat=3))  # z, y, x
    nodes = np.stack([values[k:nz-1+k, j:ny-1+j, i:nx-1+i].reshape(-1, values.shape[-1])
                      for k, j, i in corners], axis=1)
    dz, dy, dx = np.meshgrid(np.diff(z), np.diff(y), np.diff(x), indexing="ij")
    widths = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
    volumes = widths.prod(axis=1)
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("cell volumes must be positive and finite")
    volume = float(volumes.sum())
    length = volume ** (1 / 3)
    signs = np.array([[2*i-1, 2*j-1, 2*k-1] for k, j, i in corners])

    def quadrature(order):
        points, weights = np.polynomial.legendre.leggauss(order)
        for ix, iy, iz in product(range(order), repeat=3):
            q = points[[ix, iy, iz]]
            factors = (1 + signs * q) / 2
            shape = factors.prod(axis=1)
            derivative = np.column_stack([
                signs[:, axis] / 2 * factors[:, [a for a in range(3) if a != axis]].prod(axis=1)
                for axis in range(3)
            ])
            yield shape, derivative, weights[ix] * weights[iy] * weights[iz] / 8

    # Two-point Gauss integration is exact for both c and c^2 on each
    # axis-aligned trilinear cell. Use centered moments to avoid cancellation.
    mean = np.zeros(values.shape[-1])
    for shape, _, weight in quadrature(2):
        interpolated = np.einsum("caf,a->cf", nodes, shape)
        mean += np.einsum("c,cf->f", volumes * weight, interpolated) / volume
    variance = np.zeros_like(mean)
    for shape, _, weight in quadrature(2):
        centered = np.einsum("caf,a->cf", nodes, shape) - mean
        variance += np.einsum("c,cf->f", volumes * weight, centered ** 2) / volume
    standard_deviation = np.sqrt(variance)
    if np.any(standard_deviation <= 0):
        raise ValueError("normalized total variation is undefined for a zero-variance field")
    history, previous = [], None
    for order in (3, 5, 7, 11, 15):
        gradient_mean = np.zeros_like(mean)
        for _, derivative, weight in quadrature(order):
            # Reference coordinates span [-1,1]: inverse Jacobian is 2/dx,
            # not 1/dx. This factor is independently tested on affine fields.
            gradient = np.einsum("caf,ad->cdf", nodes, derivative) * 2 / widths[:, :, None]
            magnitude = np.linalg.norm(gradient, axis=1)
            gradient_mean += np.einsum("c,cf->f", volumes * weight, magnitude) / volume
        metric = length * gradient_mean / standard_deviation
        history.append({"order": order, "metric_values": metric.tolist()})
        if previous is not None and np.allclose(metric, previous, rtol=relative_tolerance, atol=1e-8):
            return {"mean": mean, "standard_deviation": standard_deviation,
                    "mean_gradient_magnitude": gradient_mean, "metric_value": metric,
                    "volume": volume, "characteristic_length": length,
                    "cell_count": len(volumes), "quadrature_history": history,
                    "convergence_relative_tolerance": relative_tolerance}
        previous = metric
    raise ValueError("trilinear gradient quadrature did not meet the declared convergence tolerance")


def materialize_trilinear_variation(dataset, manifest, fields, effective, plan, operation_id):
    from .deterministic_materialization import _point_scalar, _digest, _select_unique_max
    from vtk.util.numpy_support import vtk_to_numpy

    dimensions = [0, 0, 0]
    dataset.GetDimensions(dimensions)
    nx, ny, nz = dimensions
    coordinates = vtk_to_numpy(dataset.GetPoints().GetData()).reshape(nz, ny, nx, 3)
    x, y, z = coordinates[0, 0, :, 0], coordinates[0, :, 0, 1], coordinates[:, 0, 0, 2]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    if not np.allclose(coordinates, np.stack([xx, yy, zz], axis=-1), rtol=1e-10, atol=1e-10):
        raise ValueError("trilinear gradient executor requires axis-aligned cell coordinates")
    scalars = np.stack([_point_scalar(dataset, name).reshape(nz, ny, nx) for name in fields], axis=-1)
    computed = normalized_trilinear_variation((x, y, z), scalars)
    metrics = [{"field_name": name,
                **{key: float(computed[key][i]) for key in
                   ("mean", "standard_deviation", "mean_gradient_magnitude", "metric_value")}}
               for i, name in enumerate(fields)]
    selected = metrics[_select_unique_max(np.asarray(computed["metric_value"]) * (-1 if plan.component_selection_kind == "minimum_metric" else 1), dimension="property_measure")]
    result = {"operation_id": operation_id, "metric_name": "normalized_total_variation",
              **selected, "all_field_metrics": metrics, "total_volume": computed["volume"],
              "characteristic_length": computed["characteristic_length"],
              "cell_count": computed["cell_count"], "normalization": "L * volume_mean(abs(gradient(c))) / volume_std(c)",
              "numerical_integration": {"method": "tensor Gauss-Legendre on trilinear cells",
                                        "gradient_transform": "reference-coordinate derivative times 2/cell_width",
                                        "convergence_test": "successive quadrature order agreement",
                                        "relative_tolerance": computed["convergence_relative_tolerance"],
                                        "history": computed["quadrature_history"]}}
    provenance = {"dataset_id": "Kitchen", "dataset_manifest_sha256": _digest(manifest),
                  "reader_format": manifest.get("reader", {}).get("format"),
                  "reader_configuration": manifest.get("reader", {}).get("reader_configuration", {}),
                  "input_files": [item.get("path") for item in manifest.get("files", [])],
                  "scalar_fields": list(fields), "aggregation": plan.aggregation_kind,
                  "materialization_plan": plan.to_dict(), "effective_o_sha256": _digest(effective),
                  "compiled_plan_sha256": _digest(plan.to_dict()), "executed_plan_sha256": _digest(plan.to_dict()),
                  "silent_substitutions": [], "unsupported_dimensions": []}
    return {"parameters": dict(effective), "result": result,
            "execution": {"status": "MATERIALIZED", "reproducible": True,
                          "G_of_O": result, "data_provenance": provenance},
            "materialization_id": f"mat:{operation_id}"}

"""Independent descriptive evidence; never changes a selected scientific method."""
import math
import re

import numpy as np


def requested_numeric_thresholds(answer):
    """Extract finite literals as queries, not as trusted results or instructions."""
    values = set()
    # A grouped literal such as 10,000 is a possible threshold too. Retain
    # ordinary comma-separated literals below: this is an evidence query
    # inventory, never a decision about what the author meant or a result.
    for match in re.finditer(r"(?<![\w.])[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])", answer):
        value = float(match.group().replace(",", ""))
        if math.isfinite(value):
            values.add(value)
    for match in re.finditer(r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])", answer):
        value = float(match.group())
        if math.isfinite(value):
            values.add(value)
    return tuple(sorted(values))


def region_statistics(speed, valid, selected, retained, *, vectors=None, thresholds=()):
    speed = np.asarray(speed, dtype=float)
    selected = np.asarray(selected, dtype=int)
    values = speed[selected]
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("regional statistics require nonempty finite values")
    result = {
        "association": "retained grid points; equal point weights",
        "total_point_count": int(len(speed)), "valid_point_count": int(np.count_nonzero(valid)),
        "nonzero_valid_point_count": int(np.count_nonzero(np.asarray(valid) & (speed != 0))),
        "retained_point_count": int(np.count_nonzero(retained)), "selected_point_count": int(len(values)),
        "minimum_speed": float(values.min()), "mean_speed": float(values.mean()),
        "median_speed": float(np.median(values)), "maximum_speed": float(values.max()),
        "population_standard_deviation": float(values.std(ddof=0)),
        "quantile_method": "linear", "point_speed_quantiles": {
            str(q): float(np.quantile(values, q/100, method="linear")) for q in (10,25,50,75,90,95,99)
        },
        "speed_exceedance_queries": [
            {"threshold": float(t), "ge_count": int(np.count_nonzero(values >= t)),
             "gt_count": int(np.count_nonzero(values > t))}
            for t in sorted(set(thresholds)) if math.isfinite(t) and values.min() <= t <= values.max()
        ],
    }
    if vectors is not None:
        vectors = np.asarray(vectors, dtype=float)
        if vectors.shape != (len(speed), 3) or not np.all(np.isfinite(vectors[selected])):
            raise ValueError("regional vectors must be finite three-component values")
        peak = vectors[int(selected[np.argmax(values)])]
        norm = float(np.linalg.norm(peak))
        result.update(mean_velocity=vectors[selected].mean(axis=0).tolist(), peak_velocity=peak.tolist(),
                      peak_velocity_unit_direction=(peak/norm).tolist() if norm > 0 else None)
    return result


def cell_distribution(values, weights, *, thresholds=()):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if values.shape != weights.shape or not values.size or not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("cell distributions require finite values and positive weights")
    total = float(weights.sum())
    integral = float(np.dot(weights, values))
    if not math.isfinite(total) or not math.isfinite(integral):
        raise ValueError("cell distributions require finite volume and scalar integrals")
    fraction_defined = integral > 0 and bool(np.all(values >= 0))
    queries = []
    for threshold in sorted(set(thresholds)):
        if not math.isfinite(threshold) or not values.min() <= threshold <= values.max():
            continue
        ge, gt = values >= threshold, values > threshold
        ge_integral = float(np.dot(weights[ge], values[ge]))
        gt_integral = float(np.dot(weights[gt], values[gt]))
        query = {"threshold": float(threshold),
                 "ge_volume_fraction": float(weights[ge].sum()/total),
                 "gt_volume_fraction": float(weights[gt].sum()/total),
                 "ge_volume_integral": ge_integral, "gt_volume_integral": gt_integral}
        if fraction_defined:
            query.update(ge_integral_fraction=ge_integral/integral, gt_integral_fraction=gt_integral/integral)
        queries.append(query)
    order = np.argsort(values)
    sorted_values, cumulative = values[order], np.cumsum(weights[order]) / total
    mean = float(np.average(values, weights=weights))
    probabilities = (0.001,0.005,0.01,0.02,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.98,0.99,0.995,0.999)
    return {"cell_count": int(values.size), "total_volume": total,
            "volume_integrated_scalar": integral,
            "integral_fraction_status": "DEFINED_NONNEGATIVE_SIGNAL" if fraction_defined else "UNDEFINED_NONPOSITIVE_OR_SIGNED_SIGNAL",
            "minimum_cell_mean": float(values.min()), "maximum_cell_mean": float(values.max()),
            "volume_weighted_mean": mean,
            "weighted_quantiles": {
                "inverse_empirical_cdf_left": {str(q):float(sorted_values[min(int(np.searchsorted(cumulative,q,side='left')),len(values)-1)]) for q in probabilities},
                "linear_interpolation_cumulative_weights": {str(q):float(np.interp(q,cumulative,sorted_values)) for q in probabilities},
            },
            "quantile_convention_note": "Both conventions are explicitly labeled; they are distinct quantities, not interchangeable reference values.",
            "mean_relative_volume_fractions": [
                {"mean_multiplier": factor,"threshold": factor*mean,
                 "lt":float(weights[values < factor*mean].sum()/total),
                 "le":float(weights[values <= factor*mean].sum()/total),
                 "gt":float(weights[values > factor*mean].sum()/total),
                 "ge":float(weights[values >= factor*mean].sum()/total)}
                for factor in (0.5,1.0,2.0,10.0)
            ],
            "volume_fraction_equal_to_one": float(weights[values == 1].sum()/total),
            "cell_mean_exceedance_queries": queries}



def supplemental_kitchen_evidence(root, record, *, requested_thresholds=()):
    from .deterministic_materialization import (_reader_dataset, _point_scalar, _kitchen_cell_values,
        _digest, analyze_kitchen_concentration, MaterializationPlan)
    execution = record.get("execution", {})
    provenance = execution.get("data_provenance", {})
    old = execution.get("G_of_O", {})
    plan = provenance.get("materialization_plan", {})
    if plan.get("domain") != "kitchen_concentration" or not old:
        return None
    dataset, manifest = _reader_dataset(root, "Kitchen")
    if provenance.get("dataset_manifest_sha256") != _digest(manifest):
        raise RuntimeError("supplemental Kitchen dataset differs from frozen evidence")
    computed = analyze_kitchen_concentration(root, record.get("parameters", {}), old["operation_id"], _compiled_plan=MaterializationPlan(**plan))["result"]
    if computed["field_name"] != old["field_name"]:
        raise RuntimeError("supplemental Kitchen field differs from frozen evidence")
    for key in ("metric_value", "mean", "standard_deviation", "q10", "q50", "q90", "cell_count"):
        if key not in old or not np.allclose(old[key], computed[key], rtol=1e-10, atol=1e-10):
            raise RuntimeError("supplemental Kitchen disagrees with frozen " + key)
    details = {}
    for name in ("c1", "c11", "c12", "c13", "c14", "c15", "c16", "c17"):
        weights, values = _kitchen_cell_values(dataset, _point_scalar(dataset, name))
        details[name] = cell_distribution(values, weights, thresholds=requested_thresholds)
    return {**old, "field_distribution_details": details, "all_field_metrics": computed["all_field_metrics"],
            "supplemental_execution_provenance": {"source_execution_sha256": _digest(execution),
                "dataset_manifest_sha256": _digest(manifest), "verified_frozen_results": True}}

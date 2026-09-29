"""Population binding and actual execution using only synthetic point speeds."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import flowintentbench.deterministic_materialization as materializer


def operation(population: str) -> dict[str, str]:
    return {
        "criterion": f"Retain speed at or above P75 of {population} using linear interpolation.",
        "feature_definition": "Face-connected grid points form regions.",
        "property_measure": "Rank by peak speed.",
        "aggregation_or_representation": "Report mean spatial location.",
    }


@pytest.fixture
def synthetic_flow(monkeypatch):
    # Invalid finite and nonfinite samples must never enter either population.
    speed = np.array([0., 0., 0., 0., 1., 2., 3., 4., 1000., np.nan])
    valid = np.array([True] * 8 + [False, False])

    class Grid:
        def GetPoint(self, index):
            return (float(index), 0., 0.)

    monkeypatch.setattr(materializer, "_flow_arrays", lambda *_: (
        Grid(), {"files": [], "reader": {}}, speed, valid, (10, 1, 1)))
    masks = []
    original_regions = materializer._regions

    def capture(mask, *args):
        masks.append(mask.copy())
        return original_regions(mask, *args)

    monkeypatch.setattr(materializer, "_regions", capture)
    return speed, valid, masks


@pytest.mark.parametrize("population", [
    "all finite speeds", "finite speed", "all point speeds", "all grid-point velocity values",
    "valid locations", "valid grid locations", "valid grid points", "valid samples",
    "all finite samples", "every grid point", "domain-wide speeds",
    "all valid grid locations", "finite velocity magnitudes", "all finite speeds; zero speeds are included",
])
def test_explicit_finite_aliases_include_zeros(population, synthetic_flow):
    _, _, masks = synthetic_flow
    result = materializer.analyze_high_speed(Path("/unused"), "Synthetic", operation(population), "test")
    assert result["result"]["threshold"] == 2.25
    assert result["result"]["retained_points"] == 2
    assert result["result"]["reported_location"] == [6.5, 0., 0.]
    np.testing.assert_array_equal(masks[0], [False] * 6 + [True, True, False, False])
    assert result["execution"]["data_provenance"]["materialization_plan"]["quantile_population"] == "finite"


@pytest.mark.parametrize("population", [
    "nonzero speeds", "non-zero speeds", "non zero speeds", "positive speeds",
    "all finite nonzero speeds", "nonzero finite speeds", "strictly positive point speeds",
    "all finite speeds excluding zero", "finite speeds; exclude zero values",
    "positive grid locations", "nonzero grid points", "all finite speeds; zero speeds are excluded",
    "all finite speeds; zero-speed points are excluded", "all finite speeds; zeros are omitted",
    "all finite speeds; without zero values",
])
def test_explicit_nonzero_population_keeps_existing_behavior(population, synthetic_flow):
    _, _, masks = synthetic_flow
    result = materializer.analyze_high_speed(Path("/unused"), "Synthetic", operation(population), "test")
    assert result["result"]["threshold"] == 3.25
    assert result["result"]["retained_points"] == 1
    assert result["result"]["reported_location"] == [7., 0., 0.]
    np.testing.assert_array_equal(masks[0], [False] * 7 + [True, False, False])
    assert result["execution"]["data_provenance"]["materialization_plan"]["quantile_population"] == "positive"


@pytest.mark.parametrize("population", [
    "speeds", "the global speed distribution", "fluid-only samples", "a custom support",
    "all finite speeds and nonzero speeds", "nonzero speeds including zero values",
    "valid locations; the population consists of positive speeds",
    "all finite speeds within a subdomain", "all finite speeds using a custom support",
    "masked valid locations", "finite speeds from the ROI",
    "all finite speeds in the ROI", "valid grid locations in a selected region",
    "all finite speeds restricted to a subset", "volume-weighted samples of all finite speeds",
    "nonzero speeds; zero speeds are included", "positive speeds; zero values are not excluded",
    "all finite speeds; do not exclude zero speeds",
])
def test_missing_unknown_or_conflicting_population_fails_closed(population):
    with pytest.raises(materializer.MaterializationUnsupported, match="population"):
        materializer.compile_effective_operationalization(operation(population))


def test_population_in_feature_definition_and_unrelated_positive_density():
    effective = operation("valid grid locations")
    effective["feature_definition"] += " " + effective.pop("criterion")
    effective["criterion"] = "Select the strongest region; stored density is positive."
    plan = materializer.compile_effective_operationalization(effective)
    assert plan.quantile_population == "finite"


def test_interpolation_evidence_does_not_supply_missing_population():
    effective = operation("speeds")
    effective["criterion"] = "Retain speed at or above P75(s)."
    with pytest.raises(materializer.MaterializationUnsupported, match="population"):
        materializer.compile_effective_operationalization(effective, observed_quantile_method="linear")


def test_explicit_population_does_not_supply_missing_interpolation():
    effective = operation("valid grid locations")
    effective["criterion"] = effective["criterion"].replace(" using linear interpolation", "")
    with pytest.raises(materializer.MaterializationUnsupported, match="interpolation"):
        materializer.compile_effective_operationalization(effective)


@pytest.mark.parametrize("population, expected", [("valid grid locations", 2.), ("nonzero speeds", 3.)])
def test_nearest_rank_uses_the_declared_population(population, expected, synthetic_flow):
    effective = operation(population)
    effective["criterion"] = effective["criterion"].replace("linear interpolation", "nearest-rank")
    result = materializer.analyze_high_speed(Path("/unused"), "Synthetic", effective, "test")
    assert result["result"]["threshold"] == expected


def test_legacy_nonzero_percentile_word_order(synthetic_flow):
    effective = operation("speeds")
    effective["criterion"] = "Use the non-zero 75th percentile of speed with linear interpolation."
    result = materializer.analyze_high_speed(Path("/unused"), "Synthetic", effective, "test")
    assert result["result"]["threshold"] == 3.25


def test_valid_locations_with_descriptive_cutoff_still_recomputes(synthetic_flow):
    effective = operation("valid grid locations")
    effective["criterion"] = (
        "Compute speed as s = sqrt(mx² + my² + mz²) / rho. "
        "Retain s >= P75(s), the fastest 25% of valid grid locations; the stated cutoff is 999."
    )
    result = materializer.analyze_high_speed(Path("/unused"), "Synthetic", effective, "test",
                                            observed_quantile_method="linear")
    assert result["result"]["criterion_kind"] == "quantile"
    assert result["result"]["threshold"] == 2.25
    assert result["parameters"] == effective


@pytest.mark.parametrize("population", [None, "unknown", "volume_weighted"])
def test_executor_rejects_unknown_compiled_population(population, synthetic_flow):
    effective = operation("all finite speeds")
    plan = replace(materializer.compile_effective_operationalization(effective), quantile_population=population)
    with pytest.raises(materializer.MaterializationUnsupported, match="unknown speed population"):
        materializer.analyze_high_speed(Path("/unused"), "Synthetic", effective, "test", _compiled_plan=plan)


def test_mean_std_keeps_evidenced_full_population():
    effective = operation("all finite speeds")
    effective["criterion"] = "Retain speed at or above mean + 2 standard deviations at every point."
    proof = {"multiplier": 2., "ddof": 0, "population": "all_speed_points"}
    plan = materializer.compile_effective_operationalization(effective, observed_threshold_statistics=proof)
    assert plan.quantile_population == "finite"
    effective["criterion"] = "Retain speed at or above mean + 2 standard deviations of nonzero speeds."
    with pytest.raises(materializer.MaterializationUnsupported, match="population"):
        materializer.compile_effective_operationalization(effective, observed_threshold_statistics=proof)


def test_mean_std_executes_full_population_with_zeros(synthetic_flow):
    speed, valid, _ = synthetic_flow
    speed[-2:] = 0.
    valid[:] = True
    effective = operation("speeds")
    effective["criterion"] = "Retain speed at or above mean + 2 standard deviations at every point."
    result = materializer.analyze_high_speed(
        Path("/unused"), "Synthetic", effective, "test",
        observed_threshold_statistics={"multiplier": 2., "ddof": 0, "population": "all_speed_points"},
    )["result"]
    assert result["threshold"] == pytest.approx(1. + 2. * np.sqrt(2.))
    assert result["threshold_statistics"]["population_count"] == 10
    assert result["retained_points"] == 1


def test_all_zero_finite_population_executes_and_positive_population_is_empty(synthetic_flow):
    speed, _, _ = synthetic_flow
    speed[:8] = 0.
    result = materializer.analyze_high_speed(Path("/unused"), "Synthetic", operation("valid points"), "test")
    assert result["result"]["threshold"] == 0.
    assert result["result"]["retained_points"] == 8
    with pytest.raises(RuntimeError, match="no positive speed values"):
        materializer.analyze_high_speed(Path("/unused"), "Synthetic", operation("nonzero speeds"), "test")


def test_fixed_threshold_needs_no_quantile_population(synthetic_flow):
    effective = operation("speeds")
    effective["criterion"] = "Retain speed at least 0."
    result = materializer.analyze_high_speed(Path("/unused"), "Synthetic", effective, "test")
    assert result["result"]["retained_points"] == 8

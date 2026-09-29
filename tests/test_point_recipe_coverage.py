"""Point statistics coverage using synthetic data, independent of answers/GT."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from flowintentbench.construction_recipes import execute_recipes as frozen_execute
from flowintentbench.expansion_recipe_materializer import (
    OVERRIDES, _valid_override, _validate_recipe_controls)
from flowintentbench.runtime_construction_recipes import execute_runtime_recipes


def mesh_with_fields(x, y=None):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import numpy_to_vtk

    x = np.asarray(x, dtype=float)
    mesh = vtk.vtkRectilinearGrid()
    mesh.SetDimensions(2, 2, len(x))
    # Deliberately unequal cell volumes; each plane has four observations.
    for setter, coordinates in [(mesh.SetXCoordinates, [0., 1.]),
                                (mesh.SetYCoordinates, [0., 1.]),
                                (mesh.SetZCoordinates, np.arange(len(x), dtype=float) ** 2)]:
        setter(numpy_to_vtk(np.asarray(coordinates), deep=True))
    for name, values in [("x", x), ("y", y)]:
        if values is not None:
            array = numpy_to_vtk(np.repeat(np.asarray(values, dtype=float), 4, axis=0), deep=True)
            array.SetName(name)
            mesh.GetPointData().AddArray(array)
    return mesh


def recipe(kind="association", **overrides):
    result = {"kind": kind, "field": {"name": "x", "association": "point"},
              "sampling": "point_equal"}
    if kind == "association":
        result.update(measure="pearson", other_field={"name": "y", "association": "point"})
    else:
        result.update(measure="cv", ddof=0, cv_denominator="mean")
    return {**result, **overrides}


def definition(operation, **extra):
    return {"operations": {"runtime": {"recipe": operation}}, **extra}


def value(mesh, operation, **extra):
    return execute_runtime_recipes(mesh, definition(operation, **extra))["runtime"]["value"]


@pytest.mark.parametrize("measure", ["pearson", "spearman"])
def test_point_association_diverges_from_nonuniform_volume_cell_statistics(measure):
    from scipy.stats import rankdata

    x, y = np.array([1., 2., 2., 8., 12.]), np.array([5., 1., 4., 4., 20.])
    mesh = mesh_with_fields(x, y)
    operation = recipe(measure=measure)
    expected_x, expected_y = np.repeat(x, 4), np.repeat(y, 4)
    if measure == "spearman":
        expected_x, expected_y = rankdata(expected_x), rankdata(expected_y)
    expected = np.corrcoef(expected_x, expected_y)[0, 1]
    actual = value(mesh, operation)
    assert actual == pytest.approx(expected)
    del operation["sampling"]
    old = frozen_execute(mesh, definition(operation))
    assert execute_runtime_recipes(mesh, definition(operation)) == old
    assert actual != pytest.approx(old["runtime"]["value"])


def test_magnitude_is_reduced_per_point_before_statistics():
    x = [1., 2., 4., 9.]
    vector = [[1., 0., 0.], [-3., 0., 0.], [0., 4., 0.], [0., -10., 0.]]
    mesh = mesh_with_fields(x, vector)
    operation = recipe(other_field={"name": "y", "association": "point", "reduce": "magnitude"})
    assert value(mesh, operation) == pytest.approx(np.corrcoef(x, [1., 3., 4., 10.])[0, 1])
    old = {**operation, "sampling": "cell_volume"}
    assert value(mesh, operation) != pytest.approx(value(mesh, old))
    dispersion = recipe("dispersion", field=operation["other_field"])
    assert value(mesh, dispersion) == pytest.approx(np.std([1., 3., 4., 10.]) / 4.5)


@pytest.mark.parametrize("measure", ["pearson", "spearman"])
def test_finite_pairs_share_one_mask_before_ranking(measure):
    from scipy.stats import rankdata

    x, y = [1., np.nan, 2., 2., 8., 9.], [5., 99., np.inf, 1., 1., 4.]
    mesh = mesh_with_fields(x, y)
    operation = recipe(measure=measure)
    with pytest.raises(ValueError, match="complete finite"):
        value(mesh, operation)
    a, b = np.repeat([1., 2., 8., 9.], 4), np.repeat([5., 1., 1., 4.], 4)
    if measure == "spearman":
        a, b = rankdata(a), rankdata(b)
    assert value(mesh, {**operation, "point_domain": "finite"}) == pytest.approx(np.corrcoef(a, b)[0, 1])


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("denominator", ["mean", "absolute_mean"])
def test_cv_explicit_normalization_and_sign(ddof, denominator):
    x = np.array([-1., -2., -5., -11.])
    mesh = mesh_with_fields(x)
    divisor = abs(x.mean()) if denominator == "absolute_mean" else x.mean()
    actual = value(mesh, recipe("dispersion", ddof=ddof, cv_denominator=denominator))
    assert actual == pytest.approx(np.std(np.repeat(x, 4), ddof=ddof) / divisor)


def test_dispersion_cell_defaults_preserved_and_point_population_differs():
    mesh = mesh_with_fields([1., 2., 5., 11.])
    for measure in ["cv", "relative_interdecile"]:
        source = {"kind": "dispersion", "field": {"name": "x", "association": "point"}, "measure": measure}
        assert execute_runtime_recipes(mesh, definition(source)) == frozen_execute(mesh, definition(source))
        assert value(mesh, {**source, "sampling": "cell_volume"}) == frozen_execute(mesh, definition(source))["runtime"]["value"]
    assert value(mesh, recipe("dispersion")) != pytest.approx(value(mesh, {**source, "measure": "cv"}))


@pytest.mark.parametrize("method", ["linear", "nearest_rank"])
@pytest.mark.parametrize("denominator", ["median", "absolute_median"])
def test_relative_interdecile_explicit_estimator_and_denominator(method, denominator):
    x = np.array([-11., -5., -2., -1.])
    mesh = mesh_with_fields(x)
    operation = {"kind": "dispersion", "field": {"name": "x", "association": "point"},
                 "sampling": "point_equal", "measure": "relative_interdecile",
                 "quantile_method": method, "relative_denominator": denominator}
    low, median, high = np.quantile(np.repeat(x, 4), [.1, .5, .9],
                                    method="linear" if method == "linear" else "inverted_cdf")
    expected = (high-low) / (abs(median) if denominator == "absolute_median" else median)
    assert value(mesh, operation) == pytest.approx(expected)


def test_finite_scalar_selection_and_empty_domain():
    mesh = mesh_with_fields([1., np.nan, 3., np.inf])
    assert value(mesh, recipe("dispersion", point_domain="finite")) == pytest.approx(.5)
    with pytest.raises(ValueError, match="complete finite"):
        value(mesh, recipe("dispersion"))
    with pytest.raises(ValueError, match="empty"):
        value(mesh_with_fields([np.nan, np.inf]), recipe("dispersion", point_domain="finite"))


@pytest.mark.parametrize("measure", ["pearson", "spearman"])
def test_constant_association_is_undefined(measure):
    with pytest.raises(ValueError, match="constant"):
        value(mesh_with_fields([2., 2.], [1., 3.]), recipe(measure=measure))


def test_zero_and_constant_dispersion_denominators():
    assert value(mesh_with_fields([2., 2.]), recipe("dispersion")) == 0
    with pytest.raises(ValueError, match="zero mean"):
        value(mesh_with_fields([-1., 1.]), recipe("dispersion"))
    relative = {"kind": "dispersion", "field": {"name": "x", "association": "point"},
                "sampling": "point_equal", "measure": "relative_interdecile",
                "relative_denominator": "median", "quantile_method": "linear"}
    with pytest.raises(ValueError, match="nonzero median"):
        value(mesh_with_fields([-1., 1.]), relative)


@pytest.mark.parametrize("changes", [
    {"ddof": None}, {"ddof": 2}, {"ddof": True}, {"ddof": 0.0},
    {"cv_denominator": None}, {"cv_denominator": "rms"},
    {"quantile_method": "linear"}, {"sampling": "cell_volume"},
    {"point_domain": "roi"}, {"roi": [0, 1]},
])
def test_cv_missing_ambiguous_or_ignored_controls_fail_closed(changes):
    operation = recipe("dispersion", **changes)
    for validate in [_validate_recipe_controls,
                     lambda r: value(mesh_with_fields([1., 2.]), r)]:
        with pytest.raises(ValueError):
            validate(operation)


@pytest.mark.parametrize("changes", [
    {}, {"quantile_method": "linear"}, {"relative_denominator": "median"},
    {"quantile_method": "from_execution", "relative_denominator": "median"},
    {"quantile_method": "linear", "relative_denominator": "mean"},
    {"quantile_method": "linear", "relative_denominator": "median", "ddof": 0},
])
def test_relative_controls_are_required_and_measure_specific(changes):
    operation = {"kind": "dispersion", "field": {"name": "x", "association": "point"},
                 "measure": "relative_interdecile", "sampling": "point_equal", **changes}
    with pytest.raises(ValueError):
        _validate_recipe_controls(operation)


@pytest.mark.parametrize("spec", [
    {"name": "x", "association": "cell"},
    {"name": "absent", "association": "point"},
    {"name": "x", "association": "point", "reduce": "mean"},
    {"name": "x", "association": "point", "component": 0, "reduce": "magnitude"},
    {"name": "x", "association": "point", "component": -1},
])
def test_unsupported_field_access_fails_closed(spec):
    with pytest.raises(ValueError):
        value(mesh_with_fields([1., 2.], [2., 4.]), recipe(field=spec))


def test_inherited_cell_domain_and_unknown_sampling_are_rejected():
    mesh = mesh_with_fields([1., 2.], [2., 4.])
    with pytest.raises(ValueError, match="cell field domain"):
        value(mesh, recipe(), finite_cell_fields=[{"name": "x", "association": "point"}])
    with pytest.raises(ValueError, match="sampling"):
        value(mesh, recipe(sampling="point_volume"))
    with pytest.raises(ValueError):
        value(mesh, recipe(rank_method="dense", measure="spearman"))
    with pytest.raises(ValueError):
        value(mesh, recipe(sampling="cell_volume", point_domain="finite"))


def test_override_registry_advertises_only_supported_explicit_choices():
    for kind in ["association", "dispersion"]:
        assert _valid_override("point_equal", OVERRIDES[kind]["sampling"])
        assert _valid_override("finite", OVERRIDES[kind]["point_domain"])
        assert not _valid_override("roi", OVERRIDES[kind]["point_domain"])
    assert not _valid_override(True, OVERRIDES["dispersion"]["ddof"])
    assert not _valid_override("from_execution", OVERRIDES["dispersion"]["quantile_method"])


@pytest.mark.parametrize("kind", ["association", "dispersion"])
def test_synthetic_binding_callback_executes_points_and_caches_the_explicit_plan(tmp_path, monkeypatch, kind):
    from flowintentbench import expansion_recipe_materializer as materializer
    from flowintentbench.evaluator import EvaluationPendingAdjudication
    from flowintentbench.ground_truth import OperationalizationBundle

    x, y = [1., 2., 5., 11.], [4., 1., 8., 9.]
    mesh = mesh_with_fields(x, y)
    source = recipe(kind)
    for key in ["sampling", "ddof", "cv_denominator"]:
        source.pop(key, None)
    family = {"operations": {"source": {"recipe": source, "clauses": {}}},
              "finding_properties": [{"property_id": "statistic", "result_key": "value",
                                       "category": "quantity", "requirement_statement": "Report the statistic."}]}
    dataset_dir = tmp_path / "datasets" / "synthetic"
    dataset_dir.mkdir(parents=True)
    data = json.dumps({"x": x, "y": y}).encode()
    (dataset_dir / "samples.json").write_bytes(data)
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps({"files": [{
        "path": "synthetic/samples.json", "checksum_algorithm": "sha256",
        "checksum": hashlib.sha256(data).hexdigest()}]}))
    monkeypatch.setattr(materializer, "_reader_dataset", lambda *_: (mesh, None))
    calls = []
    overrides = {"sampling": "point_equal"}
    if kind == "dispersion":
        overrides.update(ddof=0, cv_denominator="mean")

    def request(operation, payload, **kwargs):
        assert operation == "recipe_binding"
        assert "findings" not in payload and "G_of_O" not in payload
        assert "point_equal" in payload["supported_overrides"][kind]["sampling"]
        calls.append(payload)
        return {"source_operation": "source", "parameter_overrides": dict(overrides),
                "semantic_binding_rationale": "Controlled synthetic binding fixture, no scientific judgment."}

    metadata = SimpleNamespace(dataset_id="synthetic", case_id="synthetic_o3_f1",
                               scientific_target="synthetic statistic", finding_openness="F1")
    transport = SimpleNamespace(directory=tmp_path / "fixture_cache", request=request)
    bundle = OperationalizationBundle.model_validate({"operationalization_id": "synthetic:points",
        "decisions": [{"dimension": "property_measure", "statement":
            "Use ordinary point Pearson correlation." if kind == "association" else
            "Use population point standard deviation divided by the mean, with equal point weights."}]})
    callback = materializer.build_expansion_recipe_materializer(
        tmp_path, metadata, {"family_definition": family}, transport)
    artifact = callback(bundle)
    expected = np.corrcoef(x, y)[0, 1] if kind == "association" else np.std(x) / np.mean(x)
    assert artifact["execution"]["G_of_O"]["value"] == pytest.approx(expected)
    plan = artifact["execution"]["data_provenance"]["materialization_plan"]
    assert all(plan["recipe"][key] == val for key, val in overrides.items())
    assert artifact["findings"][0]["value"] == pytest.approx(expected)
    assert callback(bundle) == artifact
    assert len(calls) == 2
    if kind == "dispersion":
        # A cached complete plan must not rescue an incomplete new binding.
        del overrides["ddof"]
        with pytest.raises(EvaluationPendingAdjudication, match="explicit ddof"):
            callback(bundle)

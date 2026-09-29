"""Novel-O numerical coverage checks independent of GT and reported answers."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flowintentbench.construction_recipes import execute_recipes as frozen_execute
from flowintentbench.runtime_construction_recipes import execute_runtime_recipes
from flowintentbench.expansion_recipe_materializer import (
    _valid_override, _validate_recipe_controls, build_expansion_recipe_materializer)
from flowintentbench.evaluator import EvaluationPendingAdjudication
from flowintentbench.ground_truth import OperationalizationBundle

ROOT = Path(__file__).resolve().parents[1]


def point_grid(values, z=None):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import numpy_to_vtk
    values = np.asarray(values, dtype=float)
    mesh = vtk.vtkRectilinearGrid()
    mesh.SetDimensions(2, 2, len(values))
    for setter, coordinates in [(mesh.SetXCoordinates, [0., 1.]),
                                (mesh.SetYCoordinates, [0., 1.]),
                                (mesh.SetZCoordinates, np.arange(len(values), dtype=float) if z is None else z)]:
        setter(numpy_to_vtk(np.asarray(coordinates, dtype=float), deep=True))
    a = numpy_to_vtk(np.repeat(values, 4), deep=True)
    a.SetName("density")
    mesh.GetPointData().AddArray(a)
    return mesh


def width_recipe(**overrides):
    return {"kind": "weighted_width", "field": {"name": "density", "association": "point"},
            "weight_transform": "mixture", "sampling": "point_equal", "axis": 2,
            "measure": "range", "endmembers": [0., 10.], "clip_mixture": True,
            "mixture_mask": [.1, .9], "center_weight": "density", "center_domain": "selected", **overrides}


def definition(recipe):
    return {"operations": {"runtime": {"recipe": recipe}}}


def test_point_range_uses_stored_coordinates_inclusive_mask_and_independent_center():
    mesh = point_grid([-1, 0, 1, 4, 9, 10, 11], [0, 1, 2, 4, 7, 9, 12])
    result = execute_runtime_recipes(mesh, definition(width_recipe()))["runtime"]
    assert result["width"] == 5  # selected stored coordinates are 2, 4, 7
    assert result["center"] == pytest.approx((1*2 + 4*4 + 9*7) / 14)
    weighted = width_recipe(center_weight="mixture", center_domain="all")
    alternate = execute_runtime_recipes(mesh, definition(weighted))["runtime"]
    assert alternate["center"] == pytest.approx((.36*2 + .96*4 + .36*7) / 1.68)
    assert alternate["center"] != result["center"]
    assert execute_runtime_recipes(mesh, definition(width_recipe(mixture_mask=[.4, .9])))["runtime"]["width"] == 3


def test_point_center_domain_is_not_silently_masked():
    mesh = point_grid([0, .5, 1, 4, 9, 9.8, 10])
    all_points = execute_runtime_recipes(mesh, definition(width_recipe(center_weight="mixture", center_domain="all")))["runtime"]
    masked = execute_runtime_recipes(mesh, definition(width_recipe(center_weight="mixture")))["runtime"]
    assert all_points["width"] == masked["width"]
    assert all_points["center"] != masked["center"]


@pytest.mark.parametrize("extra", [
    {"endmembers": "tail_plane_medians", "tail_planes": 2},
    {"endmembers": "tail_coordinate_medians", "tail_bounds": [1., 5.]},
])
def test_tail_medians_are_computed_from_point_values(extra):
    mesh = point_grid([-2, 2, 2, 4, 8, 8, 12])
    # Pooled low/high medians are 0 and 10; neither sample extrema nor means
    # over the entire grid determine the mask.
    actual = execute_runtime_recipes(mesh, definition(width_recipe(**extra)))["runtime"]
    expected = execute_runtime_recipes(mesh, definition(width_recipe()))["runtime"]
    assert actual == expected
    changed = point_grid([-2, 6, 2, 4, 8, 8, 12])
    assert execute_runtime_recipes(changed, definition(width_recipe(**extra)))["runtime"] != actual


def test_incomplete_or_ignored_point_controls_fail_closed():
    for key in ["mixture_mask", "center_weight", "center_domain"]:
        recipe = width_recipe()
        del recipe[key]
        with pytest.raises(ValueError, match="explicit"):
            _validate_recipe_controls(recipe)
    with pytest.raises(ValueError, match="point_equal"):
        _validate_recipe_controls(width_recipe(sampling="cell_volume"))
    with pytest.raises(ValueError, match="tail_planes"):
        _validate_recipe_controls(width_recipe(tail_planes=2))
    with pytest.raises(ValueError, match="overlap"):
        execute_runtime_recipes(point_grid([0, 2, 8, 10]), definition(width_recipe(endmembers="tail_plane_medians", tail_planes=3)))
    with pytest.raises(ValueError, match="nonnegative"):
        execute_runtime_recipes(point_grid([-2, 2, 8, 12]), definition(width_recipe(clip_mixture=False, center_weight="mixture", center_domain="all")))
    assert not _valid_override(True, "positive_integer")
    assert not _valid_override([.9, .1], "unit_interval_increasing_pair")
    assert not _valid_override([0, float("nan")], "finite_increasing_pair")


def unequal_cell_grid():
    mesh = point_grid([0, 1, 2, 3], [0, 1, 3, 10])
    from vtk.util.numpy_support import numpy_to_vtk
    a = numpy_to_vtk(np.array([1., 4., 10.]), deep=True)
    a.SetName("shear")
    mesh.GetCellData().AddArray(a)
    return mesh


def test_unweighted_percentile_changes_selection_only_and_preserves_volume_summaries():
    mesh = unequal_cell_grid()
    recipe = {"kind": "upper_tail", "field": {"association": "cell", "name": "shear"}, "quantile": .5}
    old = frozen_execute(mesh, definition(recipe))["runtime"]
    assert execute_runtime_recipes(mesh, definition(recipe))["runtime"] == old
    assert old["cutoff"] == 10
    novel = {**recipe, "quantile_weighting": "unweighted", "quantile_method": "linear"}
    result = execute_runtime_recipes(mesh, definition(novel))["runtime"]
    assert result["cutoff"] == 4
    assert result["fraction"] == pytest.approx(.9)
    assert result["mean"] == pytest.approx((4*2 + 10*7)/9)
    assert result["centroid"] == pytest.approx([.5, .5, (2*2 + 6.5*7)/9])
    for method, expected in [("linear", 7), ("nearest_rank", 10)]:
        assert execute_runtime_recipes(mesh, definition({**novel, "quantile": .75, "quantile_method": method}))["runtime"]["cutoff"] == expected
    with pytest.raises(ValueError, match="estimator"):
        execute_runtime_recipes(mesh, definition({**recipe, "quantile_weighting": "unweighted"}))


class BindingFixture:
    """A controlled transport, not a production semantic-binding decision."""
    def __init__(self, directory, overrides):
        self.directory, self.overrides, self.inputs = directory, overrides, []

    def request(self, operation, payload, **kwargs):
        assert operation == "recipe_binding"
        assert "G_of_O" not in payload and "findings" not in payload
        self.inputs.append(payload)
        return {"source_operation": next(iter(payload["operations"])),
                "parameter_overrides": self.overrides,
                "semantic_binding_rationale": "Controlled numerical callback fixture; no scientific acceptance."}


def family_metadata(dataset_id, family_id):
    families = json.loads((ROOT / "datasets" / dataset_id / "construction/families.json").read_text())["families"]
    family = next(f for f in families if f["family_id"] == family_id)
    return family, SimpleNamespace(dataset_id=dataset_id, case_id=family_id + "_o3_f1",
                                   scientific_target=family["scientific_target"], finding_openness="F1")


@pytest.mark.parametrize("mode", ["tail_planes", "tail_coordinates"])
def test_real_rt_callback_matches_independent_point_calculation(tmp_path, mode):
    from vtk.util.numpy_support import vtk_to_numpy
    from flowintentbench.deterministic_materialization import _reader_dataset
    family, metadata = family_metadata("Rayleigh_Taylor", "rt_mixing_width")
    overrides = {"sampling": "point_equal", "measure": "range", "clip_mixture": mode == "tail_planes",
                 "mixture_mask": [.05, .95] if mode == "tail_planes" else [.1, .9],
                 "center_weight": "mixture" if mode == "tail_planes" else "density",
                 "center_domain": "all" if mode == "tail_planes" else "selected"}
    overrides.update({"endmembers": "tail_plane_medians", "tail_planes": 20} if mode == "tail_planes"
                     else {"endmembers": "tail_coordinate_medians", "tail_bounds": [.15, .85]})
    transport = BindingFixture(tmp_path, overrides)
    bundle = OperationalizationBundle.model_validate({"operationalization_id": "controlled:" + mode,
            "decisions": [{"dimension": "feature_definition", "statement": "Controlled point range numerical fixture: " + mode}]})
    callback = build_expansion_recipe_materializer(ROOT, metadata, {"family_definition": family}, transport)
    artifact = callback(bundle)
    mesh, _ = _reader_dataset(ROOT, metadata.dataset_id)
    rho = vtk_to_numpy(mesh.GetPointData().GetArray("density")).astype(float)
    z = vtk_to_numpy(mesh.GetPoints().GetData())[:, 2].astype(float)
    planes = np.unique(z)
    low_z, high_z = (planes[19], planes[-20]) if mode == "tail_planes" else (.15, .85)
    low, high = np.median(rho[z <= low_z]), np.median(rho[z >= high_z])
    c = (rho-low)/(high-low)
    if mode == "tail_planes":
        c = np.clip(c, 0, 1)
    lower, upper = overrides["mixture_mask"]
    selected = (c >= lower) & (c <= upper)
    mass = 4*c*(1-c) if mode == "tail_planes" else rho[selected]
    center_z = z if mode == "tail_planes" else z[selected]
    actual = artifact["execution"]["G_of_O"]
    assert actual == pytest.approx({"width": float(z[selected].max()-z[selected].min()),
                                   "center": float(np.dot(mass, center_z)/mass.sum())})
    assert artifact["parameters"] == {"feature_definition": bundle.decisions[0].statement}
    assert callback(bundle) == artifact


def test_real_aideas_callback_recomputes_cell_percentile_and_preserves_absolute_volume(tmp_path):
    from vtk.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkPointDataToCellData, vtkCellCenters
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter
    from flowintentbench.deterministic_materialization import _reader_dataset
    family, metadata = family_metadata("AIDEAS_Blow_Mold", "aideas_high_shear_region")
    overrides = {"quantile": .9, "quantile_weighting": "unweighted", "quantile_method": "from_execution"}
    bundle = OperationalizationBundle.model_validate({"operationalization_id": "controlled:unweighted",
        "decisions": [{"dimension": "criterion", "statement": "Use the unweighted 90th percentile of cell shear values."}]})
    transport = BindingFixture(tmp_path, overrides)
    callback = build_expansion_recipe_materializer(ROOT, metadata, {"family_definition": family}, transport)
    with pytest.raises(EvaluationPendingAdjudication, match="execution evidence"):
        callback(bundle)
    # This controlled successful journal fixture supplies only NumPy's method,
    # never a data result or threshold. Production uses a run-bound trajectory.
    code = "import numpy as np\ncutoff=np.quantile(cell_shear,.9)"
    trajectory = tmp_path / "projection.json"
    trajectory.write_text(json.dumps({"source_journal_sha256": "0"*64, "code_only_execution_projection": [
        {"execution_index": 1, "code": code, "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
         "returncode": 0, "finished_epoch": 1}]}))
    callback = build_expansion_recipe_materializer(ROOT, metadata, {"family_definition": family}, transport,
                                                   execution_trajectory=trajectory)
    artifact = callback(bundle)
    mesh, _ = _reader_dataset(ROOT, metadata.dataset_id)
    converted = vtkPointDataToCellData(); converted.SetInputData(mesh); converted.Update()
    x = vtk_to_numpy(converted.GetOutput().GetCellData().GetArray("NormShearRate [1/s]")).astype(float)
    volumes = vtkCellSizeFilter(); volumes.SetInputData(mesh); volumes.Update()
    w = np.abs(vtk_to_numpy(volumes.GetOutput().GetCellData().GetArray("Volume")))
    centers = vtkCellCenters(); centers.SetInputData(mesh); centers.Update()
    p = vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
    cutoff = float(np.percentile(x, 90, method="linear")); selected = x >= cutoff
    result = artifact["execution"]["G_of_O"]
    assert result["cutoff"] == cutoff
    assert result["mean"] == pytest.approx(np.dot(x[selected], w[selected])/w[selected].sum())
    assert result["fraction"] == pytest.approx(w[selected].sum()/w.sum())
    assert result["centroid"] == pytest.approx(np.average(p[selected], weights=w[selected], axis=0))
    plan = artifact["execution"]["data_provenance"]["materialization_plan"]
    assert plan["recipe"]["quantile_method"] == "linear"
    assert plan["numerical_convention_evidence"]["method"] == "linear"
    assert transport.inputs[-1]["observed_quantile_method"]["method"] == "linear"

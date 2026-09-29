import hashlib
import json

import numpy as np
import pytest
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid

from flowintentbench.supplemental_association_evidence import (
    association_diagnostics, materialize_association_diagnostics, method_sources)


def grid():
    data = vtkRectilinearGrid()
    axes = [np.array([0., 1., 3.]), np.array([0., 1., 2., 4.]), np.array([0., 1., 3., 4., 6.])]
    data.SetDimensions(*(len(axis) for axis in axes))
    for setter, values in zip((data.SetXCoordinates, data.SetYCoordinates, data.SetZCoordinates), axes):
        setter(numpy_to_vtk(values, deep=True))
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    for name, values in (("X", x + 2*y + 3*z), ("Y", -2*(x + 2*y + 3*z))):
        array = numpy_to_vtk(values.ravel(), deep=True)
        array.SetName(name)
        data.GetPointData().AddArray(array)
    recipe = {"kind": "association", "measure": "pearson",
              "field": {"name": "X", "association": "point"},
              "other_field": {"name": "Y", "association": "point"}}
    return data, recipe, axes


def test_independent_panel_matches_analytic_linear_field():
    data, recipe, axes = grid()
    result = association_diagnostics(data, recipe)
    assert result["cell_count"] == 24
    assert result["total_included_coordinate_volume"] == 72
    for field in ("equal_weight_cell_pearson", "equal_weight_stored_point_pearson", "equal_weight_cell_spearman"):
        assert result[field] == pytest.approx(-1)
    values, weights = [], []
    for k in range(4):
        for j in range(3):
            for i in range(2):
                centers = [(axis[n]+axis[n+1])/2 for axis, n in zip(axes, (i, j, k))]
                values.append(centers[0] + 2*centers[1] + 3*centers[2])
                weights.append(np.prod([axis[n+1]-axis[n] for axis, n in zip(axes, (i,j,k))]))
    mean = np.average(values, weights=weights)
    covariance = -2*np.average((np.array(values)-mean)**2, weights=weights)
    assert result["volume_weighted_covariance"] == pytest.approx(covariance)
    for panel in result["auxiliary_conventions"]:
        assert panel["opposite_quadrants_fraction"] <= 1
        for region in panel["layer_regions"]:
            if region["within_region_equal_cell_pearson"] is not None:
                assert region["within_region_equal_cell_pearson"] == pytest.approx(-1)
            assert region["thresholds"] == [.01, .99]
    json.dumps(result, allow_nan=False)


def test_constant_fields_are_explicitly_undefined_not_nan():
    data, recipe, _ = grid()
    data.GetPointData().GetArray("Y").FillComponent(0, 1)
    result = association_diagnostics(data, recipe)
    assert result["equal_weight_cell_pearson"] is None
    assert result["equal_weight_cell_spearman"] is None
    assert result["volume_weighted_covariance"] == pytest.approx(0)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["coordinates", "nonfinite", "missing", "reduction"])
def test_invalid_inputs_do_not_generate_supported_evidence(fault):
    data, recipe, _ = grid()
    if fault == "coordinates":
        data.GetXCoordinates().SetValue(1, 0)
    elif fault == "nonfinite":
        data.GetPointData().GetArray("X").SetValue(0, float("nan"))
    elif fault == "missing":
        recipe["field"]["name"] = "absent"
    else:
        recipe["field"]["reduce"] = "invented"
    with pytest.raises(ValueError):
        association_diagnostics(data, recipe)


def test_method_projection_never_executes_code_or_copies_stdout(tmp_path):
    code = "import numpy as np\nraise RuntimeError('DO_NOT_EXECUTE')\nr = np.corrcoef(x, y)"
    row = dict(code=code, code_sha256=hashlib.sha256(code.encode()).hexdigest(), returncode=0,
               execution_index=1, finished_epoch=1, stdout="SELF_REPORTED_VALUE_SECRET")
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(dict(source_journal_sha256="a"*64, code_only_execution_projection=[row])))
    result = method_sources(path)
    assert result["successful_method_blocks"][0]["code"] == code
    assert "SELF_REPORTED_VALUE_SECRET" not in str(result)
    row["returncode"] = 1
    path.write_text(json.dumps(dict(source_journal_sha256="a"*64, code_only_execution_projection=[row])))
    assert method_sources(path) is None


def test_materialization_checks_inputs_even_when_cached(tmp_path, monkeypatch):
    from flowintentbench.expansion_evaluation import file_sha256
    from flowintentbench import deterministic_materialization as reader
    dataset, recipe, _ = grid()
    root = tmp_path / "repo"
    folder = root / "datasets" / "Synthetic"
    folder.mkdir(parents=True)
    raw = folder / "field.vtk"
    raw.write_bytes(b"raw fixture")
    manifest = folder / "dataset_manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "Synthetic/field.vtk", "checksum_algorithm": "sha256",
                                              "checksum": file_sha256(raw)}]}))
    provenance = {"recipe": recipe, "dataset_manifest_sha256": file_sha256(manifest)}
    calls = []
    def load(*args):
        calls.append(1)
        return dataset, {}
    monkeypatch.setattr(reader, "_reader_dataset", load)
    args = (root, {"dataset_id": "Synthetic"}, tmp_path / "exchange", provenance)
    first = materialize_association_diagnostics(*args)
    assert first["status"] == "MATERIALIZED"
    assert materialize_association_diagnostics(*args) == first
    assert len(calls) == 1
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="raw input digest"):
        materialize_association_diagnostics(*args)

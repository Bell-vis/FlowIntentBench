import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
from vtkmodules.vtkIOLegacy import vtkDataSetWriter

from flowintentbench.numeric_diagnostics import NumericCheck, verify_checks
from flowintentbench.trusted_scoring import Finding, score
from test_trusted_scoring import fixture


def numeric_fixture(tmp_path):
    grid = vtkRectilinearGrid()
    axes = [np.array([0., 1., 2.]), np.array([0., 1., 3.]), np.array([0., 1., 2.])]
    grid.SetDimensions(3, 3, 3)
    for setter, axis in zip((grid.SetXCoordinates, grid.SetYCoordinates, grid.SetZCoordinates), axes):
        setter(numpy_to_vtk(axis, deep=True))
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    for name, values in (("X", x+2*y+3*z), ("Y", -2*(x+2*y+3*z))):
        array = numpy_to_vtk(values.ravel(), deep=True)
        array.SetName(name)
        grid.GetPointData().AddArray(array)
    folder = tmp_path / "datasets/Synthetic"
    folder.mkdir(parents=True)
    path = folder / "field.vtk"
    writer = vtkDataSetWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.Write()
    manifest = folder / "dataset_manifest.json"
    manifest.write_text('{}')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    material = {"branch_execution_evidence": {"o": {"dataset_id": "Synthetic", "execution_provenance": {
        "dataset_manifest_sha256": sha(manifest), "input_files": [{"path": "Synthetic/field.vtk", "sha256": sha(path)}],
        "recipe": {"kind": "association", "measure": "pearson", "field": {"name": "X", "association": "point"},
                   "other_field": {"name": "Y", "association": "point"}}}}}}
    # Supplied construction evidence. Production score must only compare it.
    material["branch_execution_evidence"]["o"]["evidence_binding_sha256"] = "synthetic-context"
    material["frozen_auxiliary_evidence"] = [{"query": {"branch_id": "o", "statistic": "mean_x",
        "axis": None, "lower_x": None, "upper_x": None}, "expected": 7.0,
        "branch_evidence_sha256": "synthetic-context", "source": {"authority": "analytical synthetic fixture"}}]
    return material, path


def claim(value=7.0):
    quote = f"The volume weighted mean X is {value}."
    return Finding(finding_id="extra", statement="mean X", evidence_text=quote, value=value, unit=None,
        eligible=True, matches=[], numeric_checks=[NumericCheck(branch_id="o", statistic="mean_x", method_evidence_text=quote,
            scope_status="EXPLICIT", scope_evidence_text=quote)])


def test_independent_error_reaches_precision_and_c_but_not_required_recall(tmp_path):
    numeric_material, _ = numeric_fixture(tmp_path)
    answer, meta, gt, material, review = fixture()
    material.update(numeric_material)
    extra = claim(9.0)
    answer += " " + extra.evidence_text
    review["findings"].append(extra.model_dump())
    result = score(answer, meta, gt, material, review, condition="O2-F1", root=tmp_path)
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    assert result["metrics"]["o_score"]["value"] == 1
    assert result["metrics"]["finding_precision"]["value"] == .5
    assert result["metrics"]["c_score"]["value"] == .5
    assert result["error_diagnostics"]["independent_contradicted_finding_ids"] == ["extra"]


def test_correct_extra_without_gt_gains_precision_but_no_new_required_role(tmp_path):
    numeric_material, _ = numeric_fixture(tmp_path)
    answer, meta, gt, material, review = fixture()
    material.update(numeric_material)
    extra = claim()
    review["findings"].append(extra.model_dump())
    result = score(answer+" "+extra.evidence_text, meta, gt, material, review, condition="O2-F1", root=tmp_path)
    assert result["metrics"]["finding_precision"]["value"] == 1
    assert result["metrics"]["c_score"]["value"] == 1
    assert result["branch_metrics"]["o"]["matched_reference_ids"] == ["cv"]


def test_cached_calculation_still_rejects_changed_raw_data(tmp_path):
    material, path = numeric_fixture(tmp_path)
    c = claim()
    assert verify_checks(tmp_path, material, c.evidence_text, c, {"o"})[0] is True
    path.write_bytes(path.read_bytes()+b' changed')
    verdict, rows = verify_checks(tmp_path, material, c.evidence_text, c, {"o"})
    assert verdict is None
    assert "checksum mismatch" in rows[0]["reason"]


def test_unbound_method_or_threshold_never_becomes_a_numeric_verdict(tmp_path):
    material, _ = numeric_fixture(tmp_path)
    c = claim()
    assert verify_checks(tmp_path, material, c.evidence_text, c, set())[0] is None
    c.numeric_checks[0].statistic = "conditional_mean_y"
    c.numeric_checks[0].lower_x = 4.2
    verdict, rows = verify_checks(tmp_path, material, c.evidence_text, c, {"o"})
    assert verdict is None
    assert "threshold not present" in rows[0]["reason"]
    c.numeric_checks[0].scope_status = "UNRESOLVED"
    verdict, rows = verify_checks(tmp_path, material, c.evidence_text, c, {"o"})
    assert verdict is None
    assert "population scope" in rows[0]["reason"]


def test_explicit_axis_controls_layer_calculation(tmp_path):
    material, _ = numeric_fixture(tmp_path)
    c = claim(-1.0)
    c.numeric_checks[0].statistic = "layer_pearson_min"
    c.numeric_checks[0].axis = "z"
    assert verify_checks(tmp_path, material, c.evidence_text, c, {"o"})[0] is True
    c.numeric_checks[0].axis = None
    assert verify_checks(tmp_path, material, c.evidence_text, c, {"o"})[0] is None


def test_missing_layer_selectors_use_source_labels_and_public_axes_only(tmp_path):
    material, _ = numeric_fixture(tmp_path)
    text = "All vertical levels have range -0.63 to -0.44, median -0.48."
    c = claim()
    c.value, c.evidence_text = [-.63, -.44], text
    c.numeric_checks = [NumericCheck(branch_id="o", statistic=s, method_evidence_text=text,
        scope_status="EXPLICIT", scope_evidence_text=text) for s in ("layer_pearson_min", "layer_pearson_max", "layer_pearson_median")]
    verdict, rows = verify_checks(tmp_path, material, text, c, {"o"}, {"z": "vertical coordinate"})
    assert verdict is False
    assert [r["resolved_value_index"] for r in rows] == [0, 1, None]
    assert rows[0]["resolved_axis"] == "z"
    assert rows[2]["verdict"] is None  # Do not invent an unextracted median component.
    c.numeric_checks[0].axis = "x"
    assert "axis contradicts" in verify_checks(tmp_path, material, text, c, {"o"}, {"z": "vertical coordinate"})[1][0]["reason"]

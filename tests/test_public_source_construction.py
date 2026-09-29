import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flowintentbench.construction_recipes import execute_recipes, weighted_quantile
from flowintentbench.source_acquisition import HTTPRangeFile, acquire_source_assets


def grid():
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import numpy_to_vtk

    mesh = vtk.vtkImageData()
    mesh.SetDimensions(4, 2, 2)
    for name, values in [("a", [1.0, 2.0, 3.0]), ("b", [3.0, 2.0, 1.0])]:
        array = numpy_to_vtk(np.array(values), deep=True)
        array.SetName(name)
        mesh.GetCellData().AddArray(array)
    return mesh


def recipe(kind, **kw):
    return {
        "operations": {
            "test": {
                "recipe": {
                    "kind": kind,
                    "field": {"association": "cell", "name": "a"},
                    **kw,
                }
            }
        }
    }


def test_numerical_reference_matches_analytic_cells():
    mesh = grid()
    assert execute_recipes(mesh, recipe("dispersion", measure="cv"))["test"][
        "value"
    ] == pytest.approx(np.sqrt(2 / 3) / 2)
    correlation = recipe(
        "association",
        measure="pearson",
        other_field={"association": "cell", "name": "b"},
    )
    assert execute_recipes(mesh, correlation)["test"]["value"] == pytest.approx(-1)
    result = execute_recipes(mesh, recipe("upper_tail", quantile=0.5))["test"]
    assert result["centroid"] == pytest.approx([2.0, 0.5, 0.5])
    assert result["mean"] == pytest.approx(2.5)
    assert result["fraction"] == pytest.approx(2 / 3)
    assert weighted_quantile([3, 1, 2], [1, 5, 1], 0.5) == 1


def test_nan_domain_requires_explicit_selection():
    mesh = grid()
    mesh.GetCellData().GetArray("a").SetValue(0, float("nan"))
    definition = recipe("upper_tail", quantile=0.5)
    with pytest.raises(ValueError, match="nonfinite"):
        execute_recipes(mesh, definition)
    definition["finite_cell_fields"] = [{"association": "cell", "name": "a"}]
    assert execute_recipes(mesh, definition)["test"]["mean"] == 2.5


def test_cell_conversion_precedes_nonlinear_measure():
    pytest.importorskip("vtk")
    from vtk.util.numpy_support import numpy_to_vtk

    mesh = grid()
    x = np.array([mesh.GetPoint(i)[0] for i in range(mesh.GetNumberOfPoints())])
    array = numpy_to_vtk(x, deep=True)
    array.SetName("point_x")
    mesh.GetPointData().AddArray(array)
    definition = recipe("dispersion", measure="cv")
    definition["operations"]["test"]["recipe"]["field"] = {
        "association": "point",
        "name": "point_x",
    }
    assert execute_recipes(mesh, definition)["test"]["value"] == pytest.approx(
        np.std([0.5, 1.5, 2.5]) / 1.5
    )


def test_verified_cache_and_corruption_fail_closed(tmp_path):
    (tmp_path / "construction").mkdir()
    (tmp_path / "raw").mkdir()
    value = b"published data"
    target = tmp_path / "raw/data.bin"
    target.write_bytes(value)
    asset = {
        "path": "raw/data.bin",
        "url": "https://example.org/data",
        "size_bytes": len(value),
        "checksum": "sha256:" + hashlib.sha256(value).hexdigest(),
    }
    (tmp_path / "construction/source_assets.json").write_text(
        json.dumps({"assets": [asset]})
    )
    assert acquire_source_assets(tmp_path)["status"] == "VERIFIED"
    target.write_bytes(b"x" * len(value))
    with pytest.raises(ValueError, match="checksum"):
        acquire_source_assets(tmp_path)
    assert target.read_bytes() == b"x" * len(value)
    asset["path"] = "../escape.bin"
    (tmp_path / "construction/source_assets.json").write_text(
        json.dumps({"assets": [asset]})
    )
    with pytest.raises(ValueError, match="escaping"):
        acquire_source_assets(tmp_path)


def test_range_cached_seek_and_corruption(tmp_path):
    stream = HTTPRangeFile("https://example.org/pinned", 8, tmp_path, block_size=4)
    for start, content in [(0, b"abcd"), (4, b"efgh")]:
        p = stream.cache_dir / f"{start}-{start + 3}.bin"
        p.write_bytes(content)
        p.with_suffix(".sha256").write_text(hashlib.sha256(content).hexdigest())
    stream.seek(2)
    assert stream.read(4) == b"cdef"
    stream.seek(-2, 2)
    assert stream.read() == b"gh"
    (stream.cache_dir / "0-3.bin").write_bytes(b"xxxx")
    stream.seek(0)
    with pytest.raises(ValueError, match="corrupt"):
        stream.read(1)


def test_range_rejects_server_ignoring_range(tmp_path, monkeypatch):
    import subprocess

    def fake_run(command, **kwargs):
        headers = Path(command[command.index("--dump-header") + 1])
        headers.write_text("HTTP/1.1 200 OK\n")
        return subprocess.CompletedProcess(command, 0, stdout=b"abcd")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="requested byte range"):
        HTTPRangeFile("https://example.org/pinned", 8, tmp_path, block_size=4).read(1)


def test_open_finding_core_does_not_inherit_every_bounded_request():
    from types import SimpleNamespace
    from flowintentbench.case_design import FindingOpenness
    from flowintentbench.construction_recipes import computed_findings
    from flowintentbench.ground_truth import FindingImportance

    family = SimpleNamespace(
        definition={
            "finding_properties": [
                {
                    "property_id": "width",
                    "result_key": "width",
                    "category": "property",
                    "requirement_statement": "width",
                },
                {
                    "property_id": "center",
                    "result_key": "center",
                    "category": "location",
                    "requirement_statement": "center",
                },
            ],
            "f2_core_properties": ["width"],
        }
    )
    for openness, expected in [
        (FindingOpenness.BOUNDED, [FindingImportance.CORE, FindingImportance.CORE]),
        (FindingOpenness.OPEN, [FindingImportance.CORE, FindingImportance.SUPPORTING]),
    ]:
        findings, _ = computed_findings(
            family,
            SimpleNamespace(case_id="test", finding_openness=openness),
            "rms",
            {"width": 2.0, "center": 1.0},
        )
        assert [finding.importance for finding in findings] == expected

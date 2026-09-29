import numpy as np
import pytest
import vtk

from scripts.reference_construction.audit_width_geometry import vertex_average, geometric_weights


def grid(points, cells):
    mesh = vtk.vtkUnstructuredGrid(); coordinates = vtk.vtkPoints()
    coordinates.SetDataTypeToDouble()
    for point in points:
        coordinates.InsertNextPoint(*point)
    mesh.SetPoints(coordinates)
    for kind, ids in cells:
        indexes = vtk.vtkIdList()
        for index in ids:
            indexes.InsertNextId(index)
        mesh.InsertNextCell(kind, indexes)
    return mesh


def test_tetra_volume_and_vertex_average():
    mesh = grid([(0,0,0),(2,0,0),(0,3,0),(0,0,4)], [(10,[0,1,2,3])])
    for volume in geometric_weights(mesh).values():
        assert volume == pytest.approx([4])
    assert vertex_average(mesh, np.array([1.,2.,3.,10.]), np.float64) == pytest.approx([4])


def test_hex_decompositions_preserve_cube_volume():
    mesh = grid([(0,0,0),(1,0,0),(1,1,0),(0,1,0),
                 (0,0,1),(1,0,1),(1,1,1),(0,1,1)], [(12,list(range(8)))])
    for volume in geometric_weights(mesh).values():
        assert volume == pytest.approx([1])


def test_mixed_cell_average_and_precision():
    mesh = grid([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,1)],
                [(10,[0,1,2,3]),(14,[0,1,4,2,3])])
    values = np.arange(15, dtype=np.float64).reshape(5,3)
    expected = np.array([values[[0,1,2,3]].mean(0), values[[0,1,4,2,3]].mean(0)])
    assert vertex_average(mesh, values, np.float32) == pytest.approx(expected)

#!/usr/bin/env python3
"""Independently audit unstructured width references; no answer inputs.

Measure arithmetic sensitivity separately from geometric discretization.
This writes construction evidence only, never changes a scoring tolerance.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.construction_recipes import _cell_view
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from scripts.reference_construction.audit_field_sampling import average_point_values, weighted_width
from scripts.reference_construction.build_high_shear_references import tetra_geometry, aggregate_geometry
from scripts.reference_construction.build_pressure_support_references import sha


def vertex_average(mesh, values, dtype):
    values = np.asarray(values).astype(np.result_type(values.dtype, dtype))
    offsets = vtk_to_numpy(mesh.GetCells().GetOffsetsArray())
    connectivity = vtk_to_numpy(mesh.GetCells().GetConnectivityArray())
    sizes = np.diff(offsets)
    result = np.empty((len(sizes), *values.shape[1:]), dtype=float)
    for size in np.unique(sizes):
        ids = np.flatnonzero(sizes == size)
        result[ids] = values[connectivity[offsets[ids, None] + np.arange(size)]].mean(axis=1)
    return result


def geometric_weights(mesh):
    """Two explicit decompositions; neither is assumed equivalent to VTK size."""
    n = mesh.GetNumberOfCells()
    copied = vtk.vtkUnstructuredGrid(); copied.ShallowCopy(mesh)
    ids = numpy_to_vtk(np.arange(n, dtype=np.int64), deep=True); ids.SetName('__parent')
    copied.GetCellData().AddArray(ids)
    split = vtk.vtkDataSetTriangleFilter(); split.SetInputData(copied); split.Update()
    tet = split.GetOutput()
    types = vtk_to_numpy(tet.GetCellTypesArray())
    if not np.all(types == vtk.VTK_TETRA):
        raise ValueError('only volume cells supported')
    conn = vtk_to_numpy(tet.GetCells().GetConnectivityArray()).reshape(-1, 4)
    volumes, centers = tetra_geometry(vtk_to_numpy(tet.GetPoints().GetData())[conn])
    weights, _ = aggregate_geometry(volumes, centers,
        vtk_to_numpy(tet.GetCellData().GetArray('__parent')).astype(int), n)
    patterns = {10: [[0, 1, 2, 3]], 14: [[0, 1, 2, 4], [0, 2, 3, 4]],
        13: [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
        12: [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
             [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]}
    types = vtk_to_numpy(mesh.GetCellTypesArray())
    if set(types) - set(patterns):
        raise ValueError('unsupported cell geometry')
    offsets = vtk_to_numpy(mesh.GetCells().GetOffsetsArray())
    connectivity = vtk_to_numpy(mesh.GetCells().GetConnectivityArray())
    xyz = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    fixed = np.empty(n)
    for kind in np.unique(types):
        indexes = np.flatnonzero(types == kind)
        size = int(offsets[indexes[0]+1] - offsets[indexes[0]])
        conn = connectivity[offsets[indexes, None] + np.arange(size)]
        vertices = xyz[conn[:, np.array(patterns[kind])]]
        volume, _ = tetra_geometry(vertices.reshape(-1, 4, 3))
        fixed[indexes] = volume.reshape(len(indexes), -1).sum(axis=1)
    if not np.isfinite(fixed).all() or np.any(fixed <= 0):
        raise ValueError('positive cell volume required')
    return {'vtk_tetrahedra': weights, 'fixed_diagonal': fixed}


def build(output):
    output = output.resolve(); snapshot = output.with_name(output.stem + '_builder.py')
    if output.exists() or snapshot.exists():
        raise ValueError('preserve previous audits')
    manifest = json.loads((ROOT/'experiments/expansion_v1_development/case_manifest.json').read_text())
    cases = [x for x in manifest['cases'] if x['case_id'].startswith('tubes_axial_activity_width_')]
    cache = {}; records = []
    for item in cases:
        ci, _, gt, mat = load_development_case(ROOT, item)
        dataset = mat['source_case']['dataset_id']
        if dataset not in cache:
            mesh, dm = _reader_dataset(ROOT, dataset)
            files = []
            for spec in dm['files']:
                path = ROOT/dm.get('file_root', 'datasets')/spec['path']
                if sha(path) != spec['checksum']:
                    raise ValueError('source identity mismatch')
                files.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
            w, centers, arrays = _cell_view(mesh, mat['family_definition'].get('cell_volume_convention', 'positive_signed_volume'))
            xyz = vtk_to_numpy(mesh.GetPoints().GetData())
            locations = {'vtk_parametric': centers,
                'vertex_float32': vertex_average(mesh, xyz, np.float32),
                'vertex_float64': vertex_average(mesh, xyz, np.float64)}
            volumes = {'vtk_size': w, **geometric_weights(mesh)}
            cache[dataset] = mesh, arrays, locations, volumes, files
        mesh, arrays, locations, volumes, files = cache[dataset]
        for branch in gt.findings_by_operationalization:
            recipe = mat['branch_execution_evidence'][branch.operationalization_id]['execution_provenance']['recipe']
            spec = recipe['field']; name = spec['name']
            raw = vtk_to_numpy(mesh.GetPointData().GetArray(name))
            vectors = {'point_vtk': arrays[('point', name)],
                'point_float32': vertex_average(mesh, raw, np.float32),
                'point_float64': vertex_average(mesh, raw, np.float64),
                'stored_cell': arrays[('cell', name)]}
            values = {}
            for vname, vector in vectors.items():
                speed = np.linalg.norm(vector, axis=1)
                for wname, weights in volumes.items():
                    for cname, coordinate in locations.items():
                        values['/'.join((vname, wname, cname))] = {
                            **weighted_width(speed, weights, coordinate, recipe['axis'], recipe['measure']),
                            'total_volume': float(weights.sum())}
            baseline = values[('point_vtk' if spec['association']=='point' else 'stored_cell')+'/vtk_size/vtk_parametric']
            for finding in branch.findings:
                key = 'center' if 'center' in finding.finding_id else 'width'
                if not np.isclose(finding.value, baseline[key], rtol=1e-8, atol=1e-10):
                    raise ValueError('original GT not reproduced')
            records.append({'case_id': item['case_id'], 'branch_id': branch.operationalization_id,
                'recipe': recipe, 'question': ci.scientific_question, 'input_files': files,
                'variants': values})
    output.parent.mkdir(parents=True, exist_ok=True); snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(output, {'role': 'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT',
        'source_snapshot': str(snapshot.relative_to(ROOT)), 'script_sha256': sha(snapshot),
        'vtk_version': vtk.vtkVersion.GetVTKVersion(), 'records': records,
        'original_branches_reproduced': len(records),
        'interpretation': 'Geometry changes are distinct methods, not floating point tolerance. Bind any future branch by method evidence, never by closeness to a claimed value.'})
    return {'branches_reproduced': len(records), 'output': str(output), 'api_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output)))

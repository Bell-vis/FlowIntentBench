#!/usr/bin/env python3
"""Independently reproduce rectilinear advective-flux references with NumPy.

No answer, claimed value or model identity is an input. No VTK cell averaging or
production numerical recipe is called; VTK is used only to read stored arrays.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from scripts.reference_construction.build_pressure_support_references import sha


def rectilinear_flux(density, velocity, xyz):
    x, y, z = [np.asarray(a, dtype=float) for a in xyz]
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    if not all(np.isfinite(a).all() and len(a) and (a > 0).all() for a in (dx, dy, dz)):
        raise ValueError('finite strictly increasing rectilinear axes required')
    shape = (len(z), len(y), len(x))
    rho, vz = [np.asarray(a, dtype=float).reshape(shape) for a in (density, velocity)]
    if not np.isfinite(rho).all() or not np.isfinite(vz).all():
        raise ValueError('complete finite point fields required')
    def average(a):
        result = np.zeros((len(z)-1, len(y)-1, len(x)-1))
        for kz in (0, 1):
            for jy in (0, 1):
                for ix in (0, 1):
                    result += a[kz:len(z)-1+kz, jy:len(y)-1+jy, ix:len(x)-1+ix] / 8
        return result
    weights = dz[:, None, None] * dy[None, :, None] * dx[None, None, :]
    total = weights.sum()
    return {'cell_then_product': float(np.sum(weights * average(rho) * average(vz)) / total),
            'point_then_cell': float(np.sum(weights * average(rho * vz)) / total),
            'included_volume': float(total), 'point_count': rho.size, 'cell_count': weights.size}


def audit(output):
    snapshot = output.with_name(output.stem + '_builder.py')
    if output.exists() or snapshot.exists():
        raise ValueError('preserve frozen audit')
    manifest = json.loads((ROOT / 'experiments/expansion_v1_development/case_manifest.json').read_text())
    records, cache = [], {}
    for item in manifest['cases']:
        _, _, gt, material = load_development_case(ROOT, item)
        for b in gt.findings_by_operationalization:
            recipe = material.get('branch_execution_evidence', {}).get(b.operationalization_id, {}).get('execution_provenance', {}).get('recipe', {})
            if recipe.get('kind') != 'advective_flux':
                continue
            dataset = material['source_case']['dataset_id']
            key = (dataset, json.dumps({k: recipe[k] for k in ('field', 'other_field')}, sort_keys=True))
            if key not in cache:
                mesh, dm = _reader_dataset(ROOT, dataset)
                if not mesh.IsA('vtkRectilinearGrid'):
                    raise ValueError('this independent audit requires a rectilinear grid')
                for field in ('field', 'other_field'):
                    if recipe[field]['association'] != 'point':
                        raise ValueError('point fields required')
                files = []
                for f in dm['files']:
                    path = ROOT / dm.get('file_root', 'datasets') / f['path']
                    if sha(path) != f['checksum']:
                        raise ValueError('dataset checksum mismatch')
                    files.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
                rho = vtk_to_numpy(mesh.GetPointData().GetArray(recipe['field']['name']))
                v = vtk_to_numpy(mesh.GetPointData().GetArray(recipe['other_field']['name']))[:, recipe['other_field']['component']]
                xyz = [vtk_to_numpy(a) for a in (mesh.GetXCoordinates(), mesh.GetYCoordinates(), mesh.GetZCoordinates())]
                cache[key] = (rectilinear_flux(rho, v, xyz), files)
            values, files = cache[key]
            if len(b.findings) != 1:
                raise ValueError('one core flux expected')
            f = b.findings[0]
            computed = values[recipe['product_rule']]
            tolerance = f.verification.absolute_tolerance
            if tolerance is None or abs(computed - f.value) > tolerance:
                raise ValueError('independent flux does not reproduce frozen GT: ' + b.operationalization_id)
            records.append({'case_id': item['case_id'], 'branch_id': b.operationalization_id, 'recipe': recipe,
                'original_value': f.value, 'independent_value': computed, 'tolerance': tolerance,
                'values': values, 'input_files': files})
    snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(output, {'role': 'INDEPENDENT_REFERENCE_AUDIT_NO_ANSWER_INPUT', 'source_snapshot': str(snapshot.relative_to(ROOT)),
                       'source_sha256': sha(snapshot), 'records': records})
    return {'reproduced_branches': len(records), 'api_calls': 0, 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output.resolve())))

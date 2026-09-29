#!/usr/bin/env python3
"""Audit numeric precision of released rectilinear upper-tail references.

Independent construction only: no answer files or claimed results as inputs.
Compare VTK averaging to standard NumPy float32/float64 averaging on the same
finite-vertex cells. Quantile selection can amplify sub-ULP field differences.
This script reports sensitivity; it does not change scoring policy.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.construction_recipes import _cell_view
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from scripts.reference_construction.build_high_shear_references import upper_tail
from scripts.reference_construction.build_pressure_support_references import sha


def vertex_means(values, dimensions, dtype):
    # VTK flattened point order has x varying fastest.
    x, y, z = dimensions
    a = np.asarray(values, dtype=dtype).reshape(z, y, x)
    cells = np.stack([a[dz:z-1+dz, dy:y-1+dy, dx:x-1+dx]
                      for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)])
    finite = np.isfinite(cells).all(axis=0).ravel()
    return cells.mean(axis=0, dtype=dtype).ravel(), finite


def audit(output):
    if output.exists():
        raise ValueError('preserve precision audit')
    manifest = json.loads((ROOT / 'experiments/expansion_v1_development/case_manifest.json').read_text())
    cache, records = {}, []
    for item in manifest['cases']:
        ci, _, gt, material = load_development_case(ROOT, item)
        for branch in gt.findings_by_operationalization:
            recipe = material.get('branch_execution_evidence', {}).get(branch.operationalization_id, {}).get('execution_provenance', {}).get('recipe', {})
            spec = recipe.get('field', {})
            if recipe.get('kind') != 'upper_tail' or spec.get('association') != 'point':
                continue
            dataset = material['source_case']['dataset_id']
            if dataset not in cache:
                mesh, dm = _reader_dataset(ROOT, dataset)
                if not mesh.IsA('vtkRectilinearGrid'):
                    cache[dataset] = None
                    continue
                files = []
                for f in dm['files']:
                    path = ROOT / dm.get('file_root', 'datasets') / f['path']
                    if sha(path) != f['checksum']:
                        raise ValueError('source checksum mismatch')
                    files.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
                cache[dataset] = (mesh, files)
            if cache[dataset] is None:
                continue
            mesh, files = cache[dataset]
            if set(spec) != {'name', 'association'}:
                raise ValueError('scalar field required')
            w, centers, arrays = _cell_view(mesh, material['family_definition'].get('cell_volume_convention', 'positive_signed_volume'))
            stored = vtk_to_numpy(mesh.GetPointData().GetArray(spec['name']))
            variants, masks = {}, []
            for dtype in (np.float32, np.float64):
                x, mask = vertex_means(stored, mesh.GetDimensions(), dtype)
                masks.append(mask)
                variants[np.dtype(dtype).name] = upper_tail(x[mask].astype(float), w[mask], centers[mask], recipe['quantile'])
            if not np.array_equal(*masks):
                raise ValueError('precision changed the finite-cell population')
            mask = masks[0]
            variants['vtk'] = upper_tail(arrays[('point', spec['name'])][mask], w[mask], centers[mask], recipe['quantile'])
            for f in branch.findings:
                key = 'centroid' if f.category.value == 'location' else 'mean'
                if not np.allclose(f.value, variants['vtk'][key], rtol=1e-8, atol=1e-10):
                    raise ValueError('original core reference not reproduced')
            records.append({'case_id': item['case_id'], 'branch_id': branch.operationalization_id,
                'recipe': recipe, 'input_files': files, 'stored_dtype': str(stored.dtype), 'variants': variants})
    snapshot = output.with_name(output.stem + '_builder.py')
    if snapshot.exists():
        raise ValueError('preserve source snapshot')
    snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(output, {'role': 'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT',
        'source_snapshot': str(snapshot.relative_to(ROOT)), 'script_sha256': sha(snapshot), 'records': records})
    return {'branches': len(records), 'output': str(output), 'api_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output.resolve())))

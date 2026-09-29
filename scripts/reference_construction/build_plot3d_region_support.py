#!/usr/bin/env python3
"""Independently freeze common PLOT3D region statistics, without answer inputs.

Construction only: reproduce every released branch before adding supporting
quantities. No changes to core obligations, scientific tolerances, or methods.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.reference_packages import load_supplements, reporting_precision_rules
from scripts.reference_construction.build_plot3d_population_references import checksum, field_for_reference


def region_statistics(speed, xyz, shape, validity, quantile, measure, representation):
    """Structured face adjacency, positive-speed linear quantile, inclusive cut."""
    valid = np.asarray(validity, dtype=bool) & np.isfinite(speed)
    population = valid & (speed > 0)
    if not population.any():
        raise ValueError('empty positive-speed population')
    threshold = float(np.quantile(speed[population], quantile, method='linear'))
    selected = valid & (speed >= threshold)
    labels, count = ndimage.label(selected.reshape(shape, order='F'),
                                 structure=ndimage.generate_binary_structure(3, 1))
    labels = labels.ravel(order='F')
    regions = [np.flatnonzero(labels == k) for k in range(1, count + 1)]
    if measure not in {'peak', 'mean'} or representation not in {'peak', 'mean'}:
        raise ValueError('unsupported branch operation')
    scores = np.asarray([speed[r].max() if measure == 'peak' else speed[r].mean() for r in regions])
    winner = int(np.argmax(scores))
    if np.count_nonzero(scores == scores[winner]) != 1:
        raise ValueError('ambiguous winning region')
    points = regions[winner]
    coords = xyz[points]
    peaks = [float(speed[r].max()) for j, r in enumerate(regions) if j != winner]
    indices = np.array(np.unravel_index(points, shape, order='F')).T
    peak = xyz[points[np.argmax(speed[points])]].tolist()
    result = {'threshold': threshold, 'retained_points': int(selected.sum()), 'region_count': int(count),
        'selected_region_size': len(points), 'strength': float(scores[winner]),
        'location': peak if representation == 'peak' else coords.mean(axis=0).tolist(),
        'extent': np.ptp(coords, axis=0).tolist(), 'mean_speed': float(speed[points].mean()),
        'peak_point': peak, 'peak_speed': float(speed[points].max()),
        'nonzero_points': int(population.sum()), 'zero_speed_points': int((valid & (speed == 0)).sum()),
        'other_region_count': len(peaks), 'mean_location': coords.mean(axis=0).tolist(),
        'index_min': indices.min(axis=0).tolist(), 'index_max': indices.max(axis=0).tolist(),
        'coordinate_min': coords.min(axis=0).tolist(), 'coordinate_max': coords.max(axis=0).tolist()}
    if peaks:
        result.update(other_peak_max=max(peaks), other_peak_min=min(peaks))
    return result


def build(base, output, dataset_id='Blunt_Fin'):
    output = output.resolve()
    records_path = output.with_name(output.stem + '_records.json')
    audit_path = output.with_name(output.stem + '_construction.json')
    snapshot = output.with_name(output.stem + '_builder.py')
    if any(p.exists() for p in (output, records_path, audit_path, snapshot)):
        raise ValueError('preserve frozen versions')
    records = {cid: deepcopy(s['record']) for cid, s in load_supplements(base, ROOT).items()}
    manifest = json.loads((ROOT / 'experiments/expansion_v1_development/case_manifest.json').read_text())
    dataset, dm = _reader_dataset(ROOT, dataset_id)
    inputs = []
    for spec in dm['files']:
        if spec.get('role') not in {'grid', 'solution'}:
            continue
        path = ROOT / dm.get('file_root', 'datasets') / spec['path']
        if checksum(path) != spec['checksum']:
            raise ValueError('dataset checksum mismatch')
        inputs.append({'path': str(path.relative_to(ROOT)), 'sha256': checksum(path)})
    pd = dataset.GetPointData()
    speed = np.linalg.norm(vtk_to_numpy(pd.GetArray('Velocity')).astype(float), axis=1)
    xyz = vtk_to_numpy(dataset.GetPoints().GetData()).astype(float)
    blank = pd.GetArray('IBlank')
    valid = vtk_to_numpy(blank) > 0 if blank is not None else np.ones(len(speed), dtype=bool)
    shape = [0, 0, 0]
    dataset.GetDimensions(shape)
    specs = {
        'nonzero_points': ('Number of positive finite speed grid points used for the percentile', 'count'),
        'zero_speed_points': ('Number of finite zero-speed grid points excluded from the percentile', 'count'),
        'other_region_count': ('Number of retained connected regions other than the selected strongest region', 'count'),
        'other_peak_max': ('Maximum of the peak speeds of all other retained connected regions, excluding the selected region', None),
        'other_peak_min': ('Minimum of the peak speeds of all other retained connected regions, excluding the selected region', None),
        'extent': ('Physical coordinate spans (max minus min) along x, y, z of the selected region', None),
        'coordinate_min': ('Minimum x, y, z coordinates (bounding-box lower corner) of the selected region', None),
        'coordinate_max': ('Maximum x, y, z coordinates (bounding-box upper corner) of the selected region', None),
        'index_min': ('Minimum zero-based structured indices i, j, k of the selected region', '1'),
        'index_max': ('Maximum zero-based structured indices i, j, k of the selected region', '1'),
        'mean_location': ('Arithmetic mean x, y, z coordinates of the selected region', None),
        'peak_speed': ('Maximum speed in the selected region', None),
    }
    audits = []
    for item in manifest['cases']:
        ci, meta, gt, material = load_development_case(ROOT, item)
        if material['source_case']['dataset_id'] != dataset_id:
            continue
        record = records[item['case_id']]
        additions = []
        for branch in gt.findings_by_operationalization:
            bid = branch.operationalization_id
            entry = material['branch_execution_evidence'][bid]
            plan = entry['execution_provenance']['original_execution_provenance']['materialization_plan']
            if (plan['domain'] != 'high_speed_region' or plan['criterion_kind'] != 'quantile'
                    or plan['connectivity_kind'] != 'structured_face_6' or plan['quantile_method'] != 'linear'
                    or plan['comparator'] != 'GE'):
                raise ValueError('unsupported branch: ' + bid)
            result = region_statistics(speed, xyz, shape, valid, plan['quantile'],
                                       plan['measure_kind'], plan['representation_kind'])
            for ref in branch.findings:
                if not np.allclose(result[field_for_reference(bid, ref.finding_id)], ref.value, rtol=1e-10, atol=1e-10):
                    raise ValueError('independent reproduction failed: ' + ref.finding_id)
            audits.append({'case_id': item['case_id'], 'branch_id': bid, 'plan': plan, 'result': result})
            existing_fields={field_for_reference(bid,ref.finding_id) for ref in branch.findings}
            for key, (label, unit) in specs.items():
                if key not in result:
                    continue
                # Do not redefine a core quantity as supporting evidence with
                # a different tolerance. Its existing reference is authoritative.
                if (key in existing_fields or key=='mean_location' and plan['representation_kind']=='mean'
                        or key=='peak_speed' and plan['measure_kind']=='peak'):
                    continue
                value = result[key]
                vector = isinstance(value, list)
                exact = unit == 'count' or key.startswith('index_')
                tol = 0 if exact else 1e-7
                verification = {'absolute_tolerance': None if vector else tol, 'relative_tolerance': None,
                                'spatial_tolerance': tol if vector else None}
                fid = bid + '_region_support_' + key
                additions.append({'branch_id': bid, 'finding': {'finding_id': fid, 'category': 'location' if vector else 'quantity',
                    'importance': 'supporting', 'statement': label + '; uses this branch\'s point population, threshold, connectivity and region selection',
                    'value': value, 'unit': unit, 'verification': verification, 'evidence_ids': []},
                    'policy': {'operationalization_id': bid, 'finding_id': fid,
                        'verification_mode': 'componentwise_vector' if vector else 'scalar_tolerance',
                        'verification_parameters': verification}})
        record.setdefault('supporting_findings', []).extend(additions)
        record.setdefault('reporting_policies', []).extend(reporting_precision_rules(
            SimpleNamespace(findings_by_operationalization=[]),
            [a for a in additions if not a['finding']['finding_id'].endswith(('index_min', 'index_max'))]))
        record['authority'] += '; INDEPENDENT_PLOT3D_REGION_SUPPORT'
        record['reviewed_by'] += '; scipy-face-components-reference-reproduction'
        record['rationale'] += ' Reproduced original branch findings before freezing common supporting counts, peaks and bounds. No answer/model inputs; no new core obligations. Canonical reader sampling shared with construction; not independent human calibration.'
    if not audits:
        raise ValueError('no supported cases')
    write_json(audit_path, {'role': 'INDEPENDENT_REFERENCE_CONSTRUCTION; NO_ANSWER_INPUT',
        'input_files': inputs, 'script_sha256': checksum(__file__), 'dataset_id': dataset_id, 'branches': audits})
    for cid in {a['case_id'] for a in audits}:
        records[cid]['rationale'] += f' Audit {audit_path.relative_to(ROOT)} SHA256 {checksum(audit_path)}.'
    write_json(records_path, records)
    write_json(output, {'protocol': 'evaluation-reference-package-v1', 'entries': [
        {'case_id': cid, 'task_sha256': r['task_sha256'], 'source': {'path': str(records_path.relative_to(ROOT)),
         'sha256': checksum(records_path), 'pointer': '/' + cid}} for cid, r in records.items()]})
    snapshot.write_bytes(Path(__file__).read_bytes())
    return {'verified_branches': len(audits), 'supplemented_cases': len({a['case_id'] for a in audits}), 'output': str(output)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', default='Blunt_Fin')
    args = p.parse_args()
    print(json.dumps(build(args.base, args.output, args.dataset)))

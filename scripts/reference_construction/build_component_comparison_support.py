#!/usr/bin/env python3
"""Independent RMS and component-comparison references, with no answer inputs.

Only existing finite-quantile branches are supplemented. The original frozen
construction must reproduce; branch methods, core values and tolerances remain
unchanged. Comparisons labelled 'the other region' exist only for two regions.
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
from scripts.reference_construction.build_plot3d_population_references import materialize, checksum
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.reference_packages import load_supplements, reporting_precision_rules
from flowintentbench.external_file_evaluator import write_json


def component_statistics(speed, xyz, shape, validity, quantile, measure, representation):
    base = materialize(speed, xyz, shape, validity, quantile, measure, representation,
                       quantile_population='finite')
    selected = validity & np.isfinite(speed) & (speed >= base['threshold'])
    labels, count = ndimage.label(selected.reshape(shape, order='F'),
                                 structure=ndimage.generate_binary_structure(3, 1))
    labels = labels.ravel(order='F')
    regions = [np.flatnonzero(labels == k) for k in range(1, count + 1)]
    scores = [float(speed[r].max() if measure == 'peak' else speed[r].mean()) for r in regions]
    winner = int(np.argmax(scores))
    points = regions[winner]
    extra = {'rms_speed': float(np.sqrt(np.mean(np.square(speed[points]))))}
    if count == 2:
        other = regions[1 - winner]
        extra.update(other_region_size=len(other), other_mean_location=xyz[other].mean(axis=0).tolist(),
                     other_peak_speed=float(speed[other].max()), other_mean_speed=float(speed[other].mean()),
                     other_rms_speed=float(np.sqrt(np.mean(np.square(speed[other])))),
                     exceeds_other_peak_and_mean=bool(speed[points].max() > speed[other].max()
                         and speed[points].mean() > speed[other].mean()))
    return base, extra


def build(base, construction, output):
    output = Path(output).resolve()
    rp, ap = output.with_name(output.stem + '_records.json'), output.with_name(output.stem + '_construction.json')
    if any(p.exists() for p in (output, rp, ap)):
        raise ValueError('preserve frozen versions')
    records = {cid: deepcopy(s['record']) for cid, s in load_supplements(base, ROOT).items()}
    original_audit = json.loads(Path(construction).read_text())
    construction_sha = checksum(construction)
    if original_audit['role'] != 'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT':
        raise ValueError('unsupported construction evidence')
    for spec in original_audit['input_files']:
        if checksum(ROOT / spec['path']) != spec['sha256']:
            raise ValueError('dataset checksum mismatch')
    dataset, _ = _reader_dataset(ROOT, 'NASA_LOx_Post')
    speed = np.linalg.norm(vtk_to_numpy(dataset.GetPointData().GetArray('Velocity')).astype(float), axis=1)
    xyz = vtk_to_numpy(dataset.GetPoints().GetData()).astype(float)
    masked = vtk_to_numpy(dataset.GetPointData().GetArray('IBlank')) > 0
    shape = [0, 0, 0]; dataset.GetDimensions(shape)
    manifest = json.loads((ROOT / 'experiments/expansion_v1_development/case_manifest.json').read_text())
    specs = {'rms_speed': ('Root-mean-square speed in the selected strongest region', None),
        'other_region_size': ('Number of retained grid points in the only other retained connected region', 'count'),
        'other_mean_location': ('Mean x, y, z location of the only other retained connected region', None),
        'other_peak_speed': ('Peak speed of the only other retained connected region', None),
        'other_mean_speed': ('Mean speed of the only other retained connected region', None),
        'other_rms_speed': ('Root-mean-square speed of the only other retained connected region', None)}
    audit = {'role': 'INDEPENDENT_COMPONENT_COMPARISON_CONSTRUCTION_NO_ANSWER_INPUT',
        'script_sha256': checksum(__file__), 'base_sha256': checksum(base),
        'source_construction_sha256': construction_sha, 'input_files': original_audit['input_files'], 'branches': []}
    for item in manifest['cases']:
        cid = item['case_id']
        record = records.get(cid, {})
        branches = [b for b in record.get('new_branches', [])
                    if b['operationalization']['operationalization_id'] in original_audit['variants']]
        if not branches:
            continue
        if construction_sha not in record['rationale']:
            raise ValueError('reference does not bind original construction')
        _, _, _, mat = load_development_case(ROOT, item)
        entries = []
        for branch in branches:
            bid = branch['operationalization']['operationalization_id']
            parent = branch['requirement_branch_id']
            plan = mat['branch_execution_evidence'][parent]['execution_provenance']['original_execution_provenance']['materialization_plan']
            if bid.endswith('_finite_quantile_all_stored'):
                valid = np.ones(len(speed), dtype=bool)
            elif bid.endswith('_finite_quantile_iblank_valid'):
                valid = masked
            else:
                raise ValueError('unrecognized population')
            reproduced, extra = component_statistics(speed, xyz, shape, valid, plan['quantile'],
                                                     plan['measure_kind'], plan['representation_kind'])
            for key, value in original_audit['variants'][bid].items():
                if value is None:
                    if reproduced[key] is not None: raise ValueError('construction changed')
                elif not np.allclose(reproduced[key], value, rtol=1e-10, atol=1e-10):
                    raise ValueError('construction failed reproduction: ' + bid + ':' + key)
            audit['branches'].append({'case_id': cid, 'branch_id': bid, 'statistics': extra})
            for key, (label, unit) in specs.items():
                if key not in extra: continue
                value = extra[key]; vector = isinstance(value, list)
                tol = 0 if unit == 'count' else 1e-7
                verification = {'absolute_tolerance': None if vector else tol,
                                'relative_tolerance': None, 'spatial_tolerance': tol if vector else None}
                fid = bid + '_comparison_support_' + key
                entries.append({'branch_id': bid, 'finding': {'finding_id': fid,
                    'category': 'location' if vector else 'quantity', 'importance': 'supporting',
                    'statement': label + '; uses this branch\'s population, quantile, connectivity and selection rule.',
                    'value': value, 'unit': unit, 'verification': verification},
                    'policy': {'operationalization_id': bid, 'finding_id': fid,
                        'verification_mode': 'componentwise_vector' if vector else 'scalar_tolerance',
                        'verification_parameters': verification}})
            if 'exceeds_other_peak_and_mean' in extra:
                fid = bid + '_comparison_support_joint_dominance'
                text = ('The selected strongest region has both a strictly larger peak speed and a strictly larger mean speed than the only other retained connected region.'
                    if extra['exceeds_other_peak_and_mean'] else
                    'The selected strongest region does not have both a strictly larger peak speed and a strictly larger mean speed than the only other retained connected region.')
                entries.append({'branch_id': bid, 'finding': {'finding_id': fid,
                    'category': 'quantity', 'importance': 'supporting', 'statement': text,
                    'value': None, 'unit': None, 'verification': {}},
                    'policy': {'operationalization_id': bid, 'finding_id': fid,
                        'verification_mode': 'semantic_only', 'verification_parameters': {
                            'absolute_tolerance': None, 'relative_tolerance': None, 'spatial_tolerance': None}}})
        record['supporting_findings'].extend(entries)
        record['reporting_policies'] += reporting_precision_rules(SimpleNamespace(findings_by_operationalization=[]), entries)
        record['authority'] += '; INDEPENDENT_COMPONENT_COMPARISON_CONSTRUCTION'
        record['rationale'] += (' Added RMS and comparisons with the sole other component only when exactly two components exist. '
            'All original frozen variant statistics reproduced; original methods, core requirements and tolerances preserved. '
            'New numeric supporting tolerance is 1e-7, counts exact; independent construction takes no answers or model identity.')
    write_json(ap, audit)
    for cid in {x['case_id'] for x in audit['branches']}:
        records[cid]['rationale'] += f' Audit {ap.relative_to(ROOT)} SHA256 {checksum(ap)}.'
    write_json(rp, records)
    write_json(output, {'protocol': 'evaluation-reference-package-v1', 'entries': [
        {'case_id': cid, 'task_sha256': r['task_sha256'], 'source': {'path': str(rp.relative_to(ROOT)),
         'sha256': checksum(rp), 'pointer': '/' + cid}} for cid, r in records.items()]})
    return {'reproduced_variants': len(audit['branches']), 'output': str(output), 'api_calls': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--construction', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.base, args.construction, args.output)))

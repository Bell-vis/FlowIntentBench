#!/usr/bin/env python3
"""Freeze independently measured floating-point sensitivity for support fractions.

Only audited rectilinear upper-tail volume fractions are handled. Original core
references and policies are unchanged. No model outputs are construction inputs.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.reference_packages import load_supplements
from scripts.reference_construction.build_pressure_support_references import sha


def fraction_tolerance(reference, baseline, variants):
    if (not 0 <= reference <= 1 or not math.isfinite(baseline) or baseline < 0 or not variants
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in variants)):
        raise ValueError('finite volume fractions and nonnegative baseline required')
    # Retain the old numerical budget in addition to the observed implementation
    # spread. This is a construction convention, not statistical confidence.
    return baseline + max(abs(v - reference) for v in variants)


def build(base, audit, output):
    output = output.resolve()
    records_path = output.with_name(output.stem + '_records.json')
    changes_path = output.with_name(output.stem + '_precision_audit.json')
    snapshot = output.with_name(output.stem + '_builder.py')
    if any(p.exists() for p in (output, records_path, changes_path, snapshot)):
        raise ValueError('preserve frozen versions')
    evidence = json.loads(audit.read_text())
    if evidence['role'] != 'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT':
        raise ValueError('independent precision audit required')
    if sha(ROOT / evidence['source_snapshot']) != evidence['script_sha256']:
        raise ValueError('precision source changed')
    records = {cid: deepcopy(s['record']) for cid, s in load_supplements(base, ROOT).items()}
    changes = []
    for e in evidence['records']:
        if e['stored_dtype'] != 'float32' or set(e['variants']) != {'vtk', 'float32', 'float64'}:
            raise ValueError('unsupported precision comparison')
        for source in e['input_files']:
            if sha(ROOT / source['path']) != source['sha256']:
                raise ValueError('audited data changed')
        candidates = [f for f in records[e['case_id']].get('supporting_findings', [])
                      if f['branch_id'] == e['branch_id'] and f['finding']['finding_id'] == e['branch_id'] + '_support_fraction']
        if len(candidates) != 1:
            raise ValueError('exact original support fraction required')
        addition = candidates[0]
        ref = addition['finding']
        if (ref['importance'] != 'supporting' or ref['unit'] != '1'
                or ref['statement'] != 'Fraction of total mesh volume occupied by the selected region'
                or not math.isclose(ref['value'], e['variants']['vtk']['selected_fraction'], rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError('support fraction does not reproduce audited quantity')
        spec = ref['verification']
        if spec.get('relative_tolerance') is not None or spec.get('spatial_tolerance') is not None:
            raise ValueError('expected scalar absolute policy')
        old = spec['absolute_tolerance']
        values = [v['selected_fraction'] for v in e['variants'].values()]
        new = fraction_tolerance(ref['value'], old, values)
        if new <= old * (1 + 1e-4):
            continue
        ref['verification']['absolute_tolerance'] = new
        addition['policy']['verification_parameters']['absolute_tolerance'] = new
        changes.append({'case_id': e['case_id'], 'branch_id': e['branch_id'], 'finding_id': ref['finding_id'],
                        'old_tolerance': old, 'new_tolerance': new, 'independent_fractions': values})
    for cid in {c['case_id'] for c in changes}:
        record = records[cid]
        record['authority'] += '; NUMERICAL_PRECISION_SENSITIVITY'
        record['reviewed_by'] += '; independent-rectilinear-precision-audit'
        record['rationale'] += (f' Supporting selected-volume fractions include the measured difference between VTK, '
            f'NumPy float32 and float64 vertex averaging on identical cells, plus original numerical tolerance. '
            f'The released float32 task does not prescribe arithmetic precision. Core policies unchanged. '
            f'Evidence {audit.relative_to(ROOT)} SHA256 {sha(audit)}. No answer inputs or score-based tolerance selection.')
    snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(changes_path, {'construction_source': str(snapshot.relative_to(ROOT)),
        'construction_sha256': sha(snapshot), 'source_audit': str(audit.relative_to(ROOT)), 'changes': changes})
    write_json(records_path, records)
    write_json(output, {'protocol': 'evaluation-reference-package-v1', 'entries': [
        {'case_id': cid, 'task_sha256': r['task_sha256'], 'source': {
            'path': str(records_path.relative_to(ROOT)), 'sha256': sha(records_path), 'pointer': '/' + cid}}
        for cid, r in records.items()]})
    return {'output': str(output), 'changed_supporting_policies': len(changes), 'core_policies_changed': 0, 'api_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base.resolve(), args.audit.resolve(), args.output.resolve())))

#!/usr/bin/env python3
"""Expose audited volume conventions in frozen width reference descriptions.

No answer input; no changed scientific values, obligations or tolerances.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.reference_packages import load_supplements
from scripts.reference_construction.build_pressure_support_references import sha


def build(base, audit, output):
    output = output.resolve(); records_path = output.with_name(output.stem + '_records.json')
    snapshot = output.with_name(output.stem + '_builder.py')
    if any(p.exists() for p in (output, records_path, snapshot)):
        raise ValueError('preserve frozen versions')
    evidence = json.loads(audit.read_text())
    if (evidence['role'] != 'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT'
            or sha(ROOT/evidence['source_snapshot']) != evidence['script_sha256']):
        raise ValueError('independent construction evidence required')
    for item in evidence['records']:
        for source in item['input_files']:
            if sha(ROOT/source['path']) != source['sha256']:
                raise ValueError('audited source changed')
    records = {cid: deepcopy(s['record']) for cid, s in load_supplements(base, ROOT).items()}
    count = 0
    for cid in sorted({r['case_id'] for r in evidence['records']}):
        record = records[cid]
        for branch, interpretation in record['branch_interpretations'].items():
            if not branch.startswith('tubes_axial_activity_width_'):
                raise ValueError('unexpected interpretation')
            interpretation['description'] += (' Cell volumes are the absolute signed volumes returned by '
                'VTK vtkCellSizeFilter on the original mesh cells. This volume implementation is part of '
                'the frozen numerical reference and is not prescribed by the public question.')
            count += 1
        record['authority'] += '; INDEPENDENT_VOLUME_CONVENTION_AUDIT'
        record['reviewed_by'] += '; width-geometry-scope-audit'
        record['rationale'] += (f' Original width references reproduced; separate volume algorithms give '
            f'different numerical results. Make the reference implementation explicit without changing '
            f'core values, tolerances, or public obligations. Evidence {audit.relative_to(ROOT)} SHA256 {sha(audit)}.')
    snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(records_path, records)
    write_json(output, {'protocol': 'evaluation-reference-package-v1', 'entries': [
        {'case_id': cid, 'task_sha256': r['task_sha256'], 'source': {
            'path': str(records_path.relative_to(ROOT)), 'sha256': sha(records_path), 'pointer': '/'+cid}}
        for cid, r in records.items()]})
    return {'interpretations_clarified': count, 'output': str(output), 'api_calls': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.base.resolve(), a.audit.resolve(), a.output.resolve())))

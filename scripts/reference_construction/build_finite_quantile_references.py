#!/usr/bin/env python3
"""Freeze zero-inclusive quantile alternatives for publicly open criteria.

Inputs are the benchmark, frozen references, and dataset files. No answer,
model identity, claimed value, or score is accepted. Every original positive-
population numeric reference must reproduce before adding finite alternatives.
Production evaluation neither imports nor invokes this constructor.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reference_construction.build_plot3d_population_references import (
    materialize, checksum, field_for_reference, variant_statement)
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.reference_packages import load_supplements, apply_supplement, reporting_precision_rules
from flowintentbench.external_file_evaluator import write_json


def finite_criterion(statement):
    """Change only the declared percentile population, not its cutoff/rule."""
    updated, count = re.subn(r'non[- ]zero recorded speed values',
                            'all finite recorded speed values, including zeros', statement)
    if count != 1:
        raise ValueError('unrecognized positive-speed criterion')
    return updated


def build(base, output):
    output = Path(output).resolve()
    record_path = output.with_name(output.stem + '_records.json')
    audit_path = output.with_name(output.stem + '_construction.json')
    if any(p.exists() for p in (output, record_path, audit_path)):
        raise ValueError('preserve existing reference versions')
    supplements = load_supplements(base, ROOT)
    records = {cid: deepcopy(s['record']) for cid, s in supplements.items()}
    manifest = json.loads((ROOT / 'experiments/expansion_v1_development/case_manifest.json').read_text())
    dataset, dm = _reader_dataset(ROOT, 'NASA_LOx_Post')
    inputs = []
    for f in dm['files']:
        if f.get('role') not in {'grid', 'solution'}:
            continue
        p = ROOT / 'datasets' / f['path']
        if checksum(p) != f['checksum']:
            raise ValueError('dataset checksum mismatch')
        inputs.append({'path': str(p.relative_to(ROOT)), 'sha256': checksum(p)})
    pd = dataset.GetPointData()
    speed = np.linalg.norm(vtk_to_numpy(pd.GetArray('Velocity')).astype(float), axis=1)
    xyz = vtk_to_numpy(dataset.GetPoints().GetData()).astype(float)
    masked = vtk_to_numpy(pd.GetArray('IBlank')) > 0
    shape = [0, 0, 0]
    dataset.GetDimensions(shape)
    audit = {'role': 'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT',
             'script_sha256': checksum(__file__),
             'materializer_sha256': checksum(ROOT / 'scripts/reference_construction/build_plot3d_population_references.py'),
             'base_sha256': checksum(base), 'input_files': inputs,
             'verified_original_branches': [], 'variants': {}}
    affected = []
    for item in manifest['cases']:
        cid = item['case_id']
        if not cid.startswith('nasa_lox_post_'):
            continue
        ci, meta, original, material = load_development_case(ROOT, item)
        if 'criterion' not in {d.value for d in meta.unresolved_operationalization_dimensions}:
            continue
        public = json.dumps(ci.model_dump(mode='json')).lower()
        if any(term in public for term in ('iblank', 'blanked', 'validity mask')):
            raise ValueError('publicly specified mask needs separate construction policy')
        _, mat = apply_supplement(ci, meta, original, material, supplements[cid])
        record = records[cid]
        bundles = {b.operationalization_id: b for b in original.acceptable_operationalizations}
        for branch in original.findings_by_operationalization:
            bid = branch.operationalization_id
            plan = material['branch_execution_evidence'][bid]['execution_provenance']['original_execution_provenance']['materialization_plan']
            if (plan['criterion_kind'] != 'quantile' or plan['connectivity_kind'] != 'structured_face_6'
                    or plan['comparator'] != 'GE' or plan['measure_kind'] not in {'peak', 'mean'}
                    or plan['representation_kind'] not in {'peak', 'mean'}):
                raise ValueError('unsupported materialization plan')
            args = (plan['quantile'], plan['measure_kind'], plan['representation_kind'])
            expected = materialize(speed, xyz, shape, masked, *args)
            for ref in branch.findings:
                if not np.allclose(expected[field_for_reference(bid, ref.finding_id)], ref.value,
                                   rtol=1e-10, atol=1e-10):
                    raise ValueError('original numeric reference failed reproduction: ' + ref.finding_id)
            audit['verified_original_branches'].append(bid)
            for scope, validity in [('iblank_valid', masked), ('all_stored', np.ones(len(speed), dtype=bool))]:
                computed = materialize(speed, xyz, shape, validity, *args, quantile_population='finite')
                new_bid = bid + '_finite_quantile_' + scope
                audit['variants'][new_bid] = computed
                decisions = bundles[bid].model_dump(mode='json')['decisions']
                for d in decisions:
                    if d['dimension'] == 'criterion':
                        d['statement'] = finite_criterion(d['statement'])
                findings, policies, mapping = [], [], {}
                prefix = ('On finite IBlank-valid grid points' if scope == 'iblank_valid'
                          else 'On all finite stored grid points before IBlank filtering')
                for ref in branch.findings:
                    fid = new_bid + ref.finding_id[len(bid):]
                    key = field_for_reference(bid, ref.finding_id)
                    statement = variant_statement(key, *args[1:]).split(': ', 1)[1]
                    statement = statement.replace('positive-speed quantile', 'finite-speed quantile including zeros')
                    f = ref.model_copy(update={'finding_id': fid, 'value': computed[key],
                        'statement': prefix + ': ' + statement, 'evidence_ids': []}).model_dump(mode='json')
                    findings.append(f)
                    old = next(p for p in mat['finding_verification_policy']['policies']
                        if p['operationalization_id'] == bid and p['finding_id'] == ref.finding_id)
                    policies.append({k: v for k, v in {**old, 'operationalization_id': new_bid,
                        'finding_id': fid}.items() if k in {'operationalization_id', 'finding_id',
                        'verification_mode', 'verification_parameters'}})
                    if ref.importance.value == 'core':
                        mapping[fid] = ref.finding_id
                record.setdefault('new_branches', []).append({
                    'operationalization': {'operationalization_id': new_bid, 'decisions': decisions},
                    'findings': {'operationalization_id': new_bid, 'findings': findings},
                    'requirement_branch_id': bid, 'requirement_map': mapping, 'policies': policies})
                record.setdefault('branch_interpretations', {})[new_bid] = {
                    'dimension': 'feature_definition', 'description': prefix + '; zeros are included in percentile estimation.',
                    'publicly_specified': False}
                record['reporting_policies'] += reporting_precision_rules(
                    SimpleNamespace(findings_by_operationalization=[]),
                    [{'branch_id': new_bid, 'finding': f} for f in findings])
        record['authority'] += '; INDEPENDENT_FINITE_QUANTILE_CONSTRUCTION'
        record['reviewed_by'] += '; open-criterion-population-audit'
        record['rationale'] += (' Finite-value quantiles including zeros are added only when criterion is OPEN. '
            'The existing quantiles, selection comparator, connectivity, measures and representations are retained. '
            'Both public mask interpretations are included uniformly across models. Original references reproduce first. '
            'No answer or model identity enters construction; numeric reproduction is not human calibration.')
        affected.append(cid)
    write_json(audit_path, audit)
    for cid in affected:
        records[cid]['rationale'] += f' Evidence: {audit_path.relative_to(ROOT)} SHA256 {checksum(audit_path)}.'
    write_json(record_path, records)
    write_json(output, {'protocol': 'evaluation-reference-package-v1', 'entries': [
        {'case_id': cid, 'task_sha256': r['task_sha256'], 'source': {
            'path': str(record_path.relative_to(ROOT)), 'sha256': checksum(record_path), 'pointer': '/' + cid}}
        for cid, r in records.items()]})
    return {'verified_original_branches': len(audit['verified_original_branches']),
            'new_branches': len(audit['variants']), 'affected_cases': affected,
            'output': str(output), 'api_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output)))

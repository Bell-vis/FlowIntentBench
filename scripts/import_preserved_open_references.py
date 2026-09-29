#!/usr/bin/env python3
"""Freeze previously computed independent audit values as open-method references.

This imports a fixed, inspected evidence artifact. It never executes its audit
script, reads flow arrays, or obtains truth from a model's claimed values.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.reference_packages import VERSION, load_supplements


def read(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(base_manifest, audit_path, output):
    audit = read(audit_path)
    if audit['script_sha256'] != checksum(ROOT / 'scripts/audit_open_answer_methods.py'):
        raise ValueError('preserved audit implementation is not the inspected version')
    records = {cid: deepcopy(value['record']) for cid, value in load_supplements(base_manifest, ROOT).items()}
    cases = {r['case_id']: r for r in read(ROOT / 'experiments/expansion_v1_development/case_manifest.json')['cases']}
    specs = [
        ('aideas_pressure_heterogeneity_o3_f1', 'equal_cell_cv', 'pressure_equal_cell_cv', {
            'property_measure': 'Use the population standard deviation of stored cell pressure divided by its absolute arithmetic mean.',
            'aggregation_or_representation': 'Give every original cell equal weight and normalize by cell count.'}),
        ('tubes_pressure_speed_association_o3_f1', 'equal_cell_pearson', 'tubes_equal_cell_pearson', {
            'property_measure': 'Measure association by ordinary Pearson correlation between stored cell pressure and cell velocity magnitude.',
            'aggregation_or_representation': 'Give every original cell equal weight and normalize by cell count.'}),
        ('tubes_pressure_speed_association_o3_f1', 'equal_cell_spearman', 'tubes_equal_cell_spearman', {
            'property_measure': 'Assign ordinary midranks to cell pressure and cell velocity magnitude, then compute ordinary Pearson correlation of those ranks.',
            'aggregation_or_representation': 'Give every original cell equal weight and normalize by cell count.'}),
        ('mhd_density_magnetic_association_o3_f1', 'point_pearson', 'mhd_point_pearson', {
            'feature_definition': 'Use all stored mesh points, their stored density, and the Euclidean magnitude of each stored magnetic_field vector; do not average onto cells.',
            'property_measure': 'Compute ordinary Pearson correlation between point density and point magnetic-field magnitude.',
            'aggregation_or_representation': 'Give every stored point equal weight and normalize by point count.'}),
    ]
    additions = []
    for cid, suffix, key, overrides in specs:
        ci, meta, gt, material = load_development_case(ROOT, cases[cid])
        if {d.value for d in meta.unresolved_operationalization_dimensions} != {d.value for d in meta.principal_operationalization_dimensions}:
            raise ValueError('this importer only adds fully open O3 method branches')
        execution = next(iter(material['branch_execution_evidence'].values()))['execution_provenance']
        preserved = {r['sha256'] for r in audit['data_sources']}
        if not all(f['sha256'] in preserved for f in execution['input_files']):
            raise ValueError('preserved audit belongs to different scientific data')
        original = gt.acceptable_operationalizations[0]
        original_ref = gt.findings_by_operationalization[0].findings[0]
        bid, fid = cid + '_' + suffix, cid + '_' + suffix + '_coefficient'
        decisions = original.model_dump(mode='json')['decisions']
        for d in decisions:
            d['statement'] = overrides.get(d['dimension'], d['statement'])
        ref = original_ref.model_copy(update={'finding_id': fid, 'value': audit['computations'][key], 'evidence_ids': []})
        addition = {'operationalization': {'operationalization_id': bid, 'decisions': decisions},
            'findings': {'operationalization_id': bid, 'findings': [ref.model_dump(mode='json')]},
            'requirement_branch_id': original.operationalization_id, 'requirement_map': {fid: original_ref.finding_id},
            'policies': [{'operationalization_id': bid, 'finding_id': fid, 'verification_mode': 'scalar_tolerance',
                          'verification_parameters': ref.verification.model_dump(mode='json')}]}
        record = records[cid]
        if any(b['operationalization']['operationalization_id'] == bid for b in record.get('new_branches', [])):
            raise ValueError('do not append the same preserved method twice')
        record.setdefault('new_branches', []).append(addition)
        record['reporting_policies'].append({'branch_id': bid, 'finding_id': fid, 'min_significant_digits': 3,
                                             'max_rounding_radius': max(abs(ref.value)*.005, 1e-12)})
        record['authority'] += '; PRESERVED_INDEPENDENT_METHOD_AUDIT'
        record['rationale'] += f' Import {key} only from independently recomputed /computations/{key} in {audit_path}, SHA256 {checksum(audit_path)}. No model identity or claimed value is used for scoring.'
        additions.append({'case_id': cid, 'branch_id': bid, 'source_pointer': '/computations/' + key, 'expected': ref.value})
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('reference artifacts must be in repository')
    record_path = output.with_name(output.stem + '_records.json')
    if output.exists() or record_path.exists():
        raise ValueError('preserve reference versions: select a new output path')
    write_json(record_path, records)
    write_json(output, {'protocol': VERSION, 'entries': [{'case_id': cid, 'task_sha256': record['task_sha256'],
        'source': {'path': str(record_path.relative_to(ROOT)), 'sha256': checksum(record_path), 'pointer': '/' + cid}}
        for cid, record in records.items()]})
    receipt = {'source_audit': str(audit_path), 'source_audit_sha256': checksum(audit_path), 'base_manifest_sha256': checksum(base_manifest),
        'role': 'FROZEN_REFERENCE_IMPORT_ONLY; NO_NEW_SCIENTIFIC_COMPUTATION', 'added_branches': additions,
        'calibration': 'Inspected code and prior independent recomputation; not independent human labels.',
        'not_applied_to_running_evaluations': True}
    write_json(output.with_name(output.stem + '_import_receipt.json'), receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--audit', type=Path, default=ROOT / 'outputs/model_metric_calibration/reports/open_answer_audit.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(build(args.base, args.audit, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()

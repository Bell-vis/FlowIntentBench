#!/usr/bin/env python3
"""Freeze public scalar point/cell alternatives outside evaluation.

Inputs are the released task, original GT and hash-checked dataset only. Both
associations must be public and the task must leave their selection open.
Original core values are independently reproduced before any addition is saved.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.construction_recipes import _cell_view
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.reference_packages import load_supplements, reporting_precision_rules
from scripts.reference_construction.build_common_field_references import coefficient, moments
from scripts.reference_construction.build_pressure_support_references import sha


def scalar_variants(arrays, weights, name, measure, original_association, original_value):
    """Require both finite scalar populations and reproduce the original oracle."""
    values = {}
    for association in ('cell', 'point'):
        x = np.asarray(arrays[(association, name)], dtype=float)
        stats = moments(x, weights)  # Reject nonfinite values; never silently filter.
        values[association] = {'coefficient': coefficient(x, weights, measure), 'moments': stats}
    if not np.isclose(values[original_association]['coefficient'], original_value, rtol=1e-8, atol=1e-10):
        raise ValueError('original scalar coefficient was not reproduced')
    return values


def build(base, output):
    output = output.resolve()
    records_path = output.with_name(output.stem + '_records.json')
    audit_path = output.with_name(output.stem + '_construction.json')
    source_path = output.with_name(output.stem + '_builder.py')
    if any(p.exists() for p in (output, records_path, audit_path, source_path)):
        raise ValueError('preserve frozen reference versions')
    records = {cid: deepcopy(s['record']) for cid, s in load_supplements(base, ROOT).items()}
    manifest = json.loads((ROOT / 'experiments/expansion_v1_development/case_manifest.json').read_text())
    cache, audits = {}, []
    for item in manifest['cases']:
        ci, _, gt, material = load_development_case(ROOT, item)
        definition = material.get('family_definition', {})
        for branch in gt.findings_by_operationalization:
            bid = branch.operationalization_id
            recipe = material.get('branch_execution_evidence', {}).get(bid, {}).get('execution_provenance', {}).get('recipe', {})
            if recipe.get('kind') != 'dispersion':
                continue
            spec = recipe['field']
            public = [v for v in ci.flow_data.data_metadata.variables if v.name == spec['name']]
            if {v.association for v in public} != {'cell', 'point'}:
                continue
            if any(v.components != 1 for v in public) or set(spec) != {'name', 'association'}:
                raise ValueError('scalar association construction does not cover vector reductions')
            # Restrict automatic authoring to the released generic clause, or O3.
            # Explicit association restrictions require separate authoring review.
            if ('Use all cells, with stored cell values directly and arithmetic means of vertex values for point fields.'
                    not in ci.scientific_question and not item['case_id'].endswith('_o3_f1')):
                raise ValueError('public scalar association wording requires review')
            if len({v.unit for v in public}) != 1 or definition.get('finite_cell_fields'):
                raise ValueError('association-specific units or filtering require separate review')
            dataset = material['source_case']['dataset_id']
            convention = definition.get('cell_volume_convention', 'positive_signed_volume')
            key = (dataset, convention)
            if key not in cache:
                data, dm = _reader_dataset(ROOT, dataset)
                files = []
                for source in dm['files']:
                    path = ROOT / dm.get('file_root', 'datasets') / source['path']
                    if sha(path) != source['checksum']:
                        raise ValueError('source checksum mismatch')
                    files.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
                w, _, arrays = _cell_view(data, convention)
                cache[key] = (w, arrays, files)
            w, arrays, files = cache[key]
            core = [f for f in branch.findings if f.importance.value == 'core']
            if len(core) != 1 or len(branch.findings) != 1:
                raise ValueError('scalar dispersion must have one original core coefficient')
            ref = core[0]
            variants = scalar_variants(arrays, w, spec['name'], recipe['measure'], spec['association'], ref.value)
            record = records[item['case_id']]
            if bid in record.get('branch_interpretations', {}):
                raise ValueError('do not overwrite an existing population interpretation')
            alternate = 'point' if spec['association'] == 'cell' else 'cell'
            new_bid = bid + '_' + alternate + '_scalar'
            if any(b['operationalization']['operationalization_id'] == new_bid for b in record.get('new_branches', [])):
                raise ValueError('duplicate scalar branch')
            bundle = next(b for b in gt.acceptable_operationalizations if b.operationalization_id == bid).model_dump(mode='json')
            bundle['operationalization_id'] = new_bid
            fid = new_bid + '_coefficient'
            finding = ref.model_copy(update={'finding_id': fid, 'value': variants[alternate]['coefficient']}).model_dump(mode='json')
            old_policy = next(p for p in material['finding_verification_policy']['policies'] if p['finding_id'] == ref.finding_id)
            policy = {k: v for k, v in {**old_policy, 'operationalization_id': new_bid, 'finding_id': fid}.items()
                      if k in {'operationalization_id', 'finding_id', 'verification_mode', 'verification_parameters'}}
            record.setdefault('new_branches', []).append({'operationalization': bundle,
                'findings': {'operationalization_id': new_bid, 'findings': [finding]},
                'requirement_branch_id': bid, 'requirement_map': {fid: ref.finding_id}, 'policies': [policy]})
            record.setdefault('reporting_policies', []).extend(reporting_precision_rules(
                SimpleNamespace(findings_by_operationalization=[]), [{'branch_id': new_bid, 'finding': finding}]))
            for association, branch_id in ((spec['association'], bid), (alternate, new_bid)):
                description = (f'Use stored cell-associated {spec["name"]} scalar values directly.' if association == 'cell'
                               else f'Use stored point-associated {spec["name"]}; form each cell scalar as the arithmetic mean of its vertex values.')
                record.setdefault('branch_interpretations', {})[branch_id] = {
                    'dimension': 'feature_definition', 'description': description, 'publicly_specified': False}
            audits.append({'case_id': item['case_id'], 'branch_id': bid, 'new_branch_id': new_bid,
                'question': ci.scientific_question, 'recipe': recipe, 'input_files': files,
                'cell_volume_convention': convention, 'cell_count': len(w), 'variants': variants})
    source_path.write_bytes(Path(__file__).read_bytes())
    write_json(audit_path, {'role': 'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT',
        'source_snapshot': str(source_path.relative_to(ROOT)), 'script_sha256': sha(source_path), 'records': audits})
    for cid in {a['case_id'] for a in audits}:
        records[cid]['authority'] += '; PUBLIC_SCALAR_ASSOCIATION_INTERPRETATION'
        records[cid]['reviewed_by'] += '; independent-scalar-association-audit'
        records[cid]['rationale'] += (f' Both stored associations are publicly supplied without an association selection. '
            f'Original coefficients independently reproduced; original core obligations and tolerance retained. '
            f'Bind association from source method evidence, never numerical proximity. Evidence {audit_path.relative_to(ROOT)} SHA256 {sha(audit_path)}.')
    write_json(records_path, records)
    write_json(output, {'protocol': 'evaluation-reference-package-v1', 'entries': [
        {'case_id': cid, 'task_sha256': r['task_sha256'], 'source': {
            'path': str(records_path.relative_to(ROOT)), 'sha256': sha(records_path), 'pointer': '/' + cid}}
        for cid, r in records.items()]})
    return {'output': str(output), 'new_branches': len(audits), 'api_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base.resolve(), args.output.resolve())))

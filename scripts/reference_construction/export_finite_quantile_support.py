#!/usr/bin/env python3
"""Export already frozen component statistics; no datasets or answers are read.

New finite-quantile branches inherit exactly the supporting quantities and
tolerances already authored for their original branch. Only values come from
the separately frozen, hash-bound construction artifact.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.reference_packages import load_supplements, reporting_precision_rules
from flowintentbench.external_file_evaluator import write_json
from scripts.reference_construction.build_plot3d_population_references import checksum, field_for_reference


def support_entries(record, variants):
    result = []
    for branch in record.get('new_branches', []):
        bid = branch['operationalization']['operationalization_id']
        if bid not in variants:
            continue
        original = branch['requirement_branch_id']
        prefix = 'On ' + record['branch_interpretations'][bid]['description'].split(';', 1)[0].removeprefix('On ') + ': '
        for entry in record.get('supporting_findings', []):
            if entry['branch_id'] != original:
                continue
            old_id = entry['finding']['finding_id']
            key = field_for_reference(original, old_id)
            if key not in variants[bid]:
                raise ValueError('frozen construction lacks supporting field: ' + key)
            peer = deepcopy(entry)
            fid = bid + old_id[len(original):]
            peer['branch_id'] = bid
            peer['finding'].update(finding_id=fid, value=variants[bid][key],
                                   statement=prefix + entry['finding']['statement'])
            peer['policy'].update(operationalization_id=bid, finding_id=fid)
            result.append(peer)
    return result


def export(base, construction, output):
    output = Path(output).resolve()
    rp = output.with_name(output.stem + '_records.json')
    ap = output.with_name(output.stem + '_export.json')
    if any(p.exists() for p in (output, rp, ap)):
        raise ValueError('preserve existing reference versions')
    records = {cid: deepcopy(s['record']) for cid, s in load_supplements(base, ROOT).items()}
    frozen = json.loads(Path(construction).read_text())
    if frozen['role'] != 'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT':
        raise ValueError('unsupported construction evidence')
    evidence_sha = checksum(construction)
    changes = []
    for cid, record in records.items():
        branches = {b['operationalization']['operationalization_id'] for b in record.get('new_branches', [])}
        selected = branches & frozen['variants'].keys()
        if not selected:
            continue
        if evidence_sha not in record['rationale']:
            raise ValueError('base reference does not bind construction evidence')
        entries = support_entries(record, frozen['variants'])
        existing = {e['finding']['finding_id'] for e in record.get('supporting_findings', [])}
        if any(e['finding']['finding_id'] in existing for e in entries):
            raise ValueError('support reference already exists')
        record['supporting_findings'].extend(entries)
        record['reporting_policies'] += reporting_precision_rules(
            SimpleNamespace(findings_by_operationalization=[]), entries)
        record['authority'] += '; FROZEN_QUANTILE_SUPPORT_EXPORT'
        record['rationale'] += (' Supporting quantities and tolerances inherit the original branch; '
            'values are exported from its independently frozen finite-population variant. '
            'No answer, dataset read, new scientific calculation or core obligation change occurs in export.')
        changes.append({'case_id': cid, 'new_supporting_references': len(entries)})
    write_json(ap, {'role': 'FROZEN_REFERENCE_EXPORT_NO_SCIENTIFIC_RECOMPUTATION',
        'base_sha256': checksum(base), 'construction': str(Path(construction).resolve()),
        'construction_sha256': evidence_sha, 'script_sha256': checksum(__file__), 'changes': changes})
    write_json(rp, records)
    write_json(output, {'protocol': 'evaluation-reference-package-v1', 'entries': [
        {'case_id': cid, 'task_sha256': r['task_sha256'], 'source': {'path': str(rp.relative_to(ROOT)),
            'sha256': checksum(rp), 'pointer': '/' + cid}} for cid, r in records.items()]})
    return {'output': str(output), 'new_supporting_references': sum(c['new_supporting_references'] for c in changes),
            'changed_cases': len(changes), 'api_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--construction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.base, args.construction, args.output)))

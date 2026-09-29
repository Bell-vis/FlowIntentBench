#!/usr/bin/env python3
"""Build frozen population variants for underspecified PLOT3D validity masks.

Independent construction only: no model answers, judgments, or model IDs enter
this program. Reproduce masked GT first, then materialize the same public O on
all finite stored points. The scorer only imports the resulting frozen JSON.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.reference_packages import load_supplements, apply_supplement, reporting_precision_rules
from flowintentbench.external_file_evaluator import write_json


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize(speed, coordinates, shape, validity, quantile, measure, representation,
                *, quantile_population='positive'):
    if quantile_population not in {'positive', 'finite'}:
        raise ValueError('unsupported quantile population')
    valid = validity & np.isfinite(speed)
    positive = valid & (speed > 0)
    population = positive if quantile_population == 'positive' else valid
    threshold = float(np.quantile(speed[population], quantile, method="linear"))
    selected = valid & (speed >= threshold)
    labels, count = ndimage.label(selected.reshape(shape, order="F"),
                                 structure=ndimage.generate_binary_structure(3, 1))
    labels = labels.ravel(order="F")
    regions = [np.flatnonzero(labels == k) for k in range(1, count + 1)]
    scores = [float(speed[r].max() if measure == 'peak' else speed[r].mean()) for r in regions]
    winner = int(np.argmax(scores))
    if scores.count(scores[winner]) != 1:
        raise ValueError('tied component maximum requires a separately authored tie rule')
    points = regions[winner]
    xyz = coordinates[points]
    peak_point = coordinates[points[np.argmax(speed[points])]].tolist()
    peaks = sorted([float(speed[r].max()) for r in regions], reverse=True)
    return {'threshold': threshold, 'retained_points': int(selected.sum()), 'region_count': count,
            'selected_region_size': len(points), 'strength': scores[winner],
            'location': peak_point if representation == 'peak' else xyz.mean(axis=0).tolist(),
            'extent': np.ptp(xyz, axis=0).tolist(), 'mean_speed': float(speed[points].mean()),
            'peak_point': peak_point, 'nonzero_points': int(positive.sum()),
            'runner_up_peak': peaks[1] if len(peaks) > 1 else None}


def field_for_reference(bid, fid):
    suffix = fid.removeprefix(bid + '_')
    if suffix.startswith('support_'):
        return suffix.removeprefix('support_')
    return {'retained_points': 'selected_region_size'}.get(suffix, suffix)


def variant_statement(key, measure, representation):
    """Describe the new quantity without copying the old population's number."""
    labels = {
        'location': ('Peak-speed point location' if representation == 'peak' else 'Mean spatial location')
                    + ' of the selected strongest connected region',
        'strength': ('Peak' if measure == 'peak' else 'Mean') + ' speed of the selected strongest connected region',
        'threshold': 'High-speed threshold from the declared positive-speed quantile',
        'retained_points': 'Number of all grid points retained by the high-speed threshold',
        'region_count': 'Number of face-connected high-speed regions',
        'selected_region_size': 'Number of grid points in the selected strongest connected region',
        'extent': 'Coordinate extent of the selected strongest connected region',
        'mean_speed': 'Mean speed of the selected strongest connected region',
        'peak_point': 'Peak-speed point location in the selected strongest connected region',
        'nonzero_points': 'Number of nonzero-speed points used for percentile estimation',
        'runner_up_peak': 'Second-highest connected-component peak speed',
    }
    if key not in labels or measure not in {'peak', 'mean'} or representation not in {'peak', 'mean'}:
        raise ValueError('unsupported reference quantity/representation')
    return 'On all finite stored grid points before IBlank filtering: ' + labels[key] + '.'


def deduplicate_variant_references(record):
    """Remove semantically identical supporting aliases; keep core policies."""
    removed=[]
    for branch in record.get('new_branches',[]):
        if not branch['operationalization']['operationalization_id'].endswith('_all_finite_stored'):
            continue
        findings=branch['findings']['findings'];canonical={};drop=set()
        for f in sorted(findings,key=lambda f:(f['importance']!='core',f.get('unit')!='count')):
            prior=canonical.get(f['statement'])
            if prior is None:
                canonical[f['statement']]=f
                continue
            count_alias=(set((f.get('unit'),prior.get('unit')))=={None,'count'}
                and 'Number of ' in f['statement'] and type(f['value']) is int)
            if (f['importance']=='core' or f['value']!=prior['value']
                    or f.get('unit')!=prior.get('unit') and not count_alias):
                raise ValueError('duplicate description has conflicting truth or core obligations')
            if f['finding_id'] in branch['requirement_map']:
                raise ValueError('supporting alias unexpectedly owns a required result')
            drop.add(f['finding_id'])
            removed.append({'removed_id':f['finding_id'],'retained_id':prior['finding_id'],
                            'value':f['value'],'reason':'IDENTICAL_QUANTITY_SUPPORTING_ALIAS'})
        branch['findings']['findings']=[f for f in findings if f['finding_id'] not in drop]
        branch['policies']=[p for p in branch['policies'] if p['finding_id'] not in drop]
    dropped={r['removed_id'] for r in removed}
    record['reporting_policies']=[p for p in record.get('reporting_policies',[]) if p['finding_id'] not in dropped]
    return removed


def build(base, output):
    supplements = load_supplements(base, ROOT)
    records = {cid: deepcopy(s['record']) for cid, s in supplements.items()}
    manifest = json.loads((ROOT/'experiments/expansion_v1_development/case_manifest.json').read_text())
    dataset, dm = _reader_dataset(ROOT, 'NASA_LOx_Post')
    inputs = []
    for f in dm['files']:
        if f.get('role') not in {'grid', 'solution'}:
            continue
        p = ROOT/'datasets'/f['path']
        if checksum(p) != f['checksum']:
            raise ValueError('dataset checksum mismatch')
        inputs.append({'path': str(p.relative_to(ROOT)), 'sha256': checksum(p)})
    pd = dataset.GetPointData()
    speed = np.linalg.norm(vtk_to_numpy(pd.GetArray('Velocity')).astype(float), axis=1)
    xyz = vtk_to_numpy(dataset.GetPoints().GetData()).astype(float)
    valid = vtk_to_numpy(pd.GetArray('IBlank')) > 0
    shape = [0, 0, 0]
    dataset.GetDimensions(shape)
    evidence = {'role': 'INDEPENDENT_REFERENCE_CONSTRUCTION; NO_MODEL_ANSWER_INPUT', 'input_files': inputs,
                'script_sha256': checksum(__file__), 'verified_masked_branches': [], 'variants': {}}
    for item in manifest['cases']:
        if not item['case_id'].startswith('nasa_lox_post_'):
            continue
        ci, meta, original, material = load_development_case(ROOT, item)
        # This rule addresses an omission in the released input, not permission
        # to ignore an explicitly supplied validity mask in another question.
        public = json.dumps(ci.model_dump(mode='json')).lower()
        if any(term in public for term in ('iblank', 'blanked', 'validity mask')):
            raise ValueError('public task specifies a validity population; do not add an alternative')
        gt, mat = apply_supplement(ci, meta, original, material, supplements[item['case_id']])
        record = records[item['case_id']]
        bundles = {b.operationalization_id: b for b in gt.acceptable_operationalizations}
        for branch in gt.findings_by_operationalization:
            bid = branch.operationalization_id
            entry = material['branch_execution_evidence'][bid]
            plan = entry['execution_provenance']['original_execution_provenance']['materialization_plan']
            if (plan['criterion_kind'] != 'quantile' or plan['connectivity_kind'] != 'structured_face_6'
                    or plan['comparator'] != 'GE' or plan['measure_kind'] not in {'peak', 'mean'}
                    or plan['representation_kind'] not in {'peak', 'mean'}):
                raise ValueError('unsupported population-variant operation')
            args = (plan['quantile'], plan['measure_kind'], plan['representation_kind'])
            masked = materialize(speed, xyz, shape, valid, *args)
            all_points = materialize(speed, xyz, shape, np.ones(len(speed), dtype=bool), *args)
            for ref in branch.findings:
                key = field_for_reference(bid, ref.finding_id)
                if not np.allclose(masked[key], ref.value, rtol=1e-10, atol=1e-10):
                    raise ValueError('independent construction does not reproduce masked reference: '+ref.finding_id)
            evidence['verified_masked_branches'].append(bid)
            new_bid = bid + '_all_finite_stored'
            evidence['variants'][new_bid] = all_points
            findings, policies, mapping = [], [], {}
            for ref in branch.findings:
                fid = new_bid + ref.finding_id[len(bid):]
                value = all_points[field_for_reference(bid, ref.finding_id)]
                f = ref.model_copy(update={'finding_id': fid, 'value': value, 'evidence_ids': []}).model_dump(mode='json')
                f['statement'] = variant_statement(field_for_reference(bid, ref.finding_id), *args[1:])
                findings.append(f)
                old_policy = next(p for p in mat['finding_verification_policy']['policies']
                                  if p['operationalization_id'] == bid and p['finding_id'] == ref.finding_id)
                policies.append({k:v for k,v in {**old_policy, 'operationalization_id':new_bid, 'finding_id':fid}.items()
                                 if k in {'operationalization_id','finding_id','verification_mode','verification_parameters'}})
                if ref.importance.value == 'core':
                    mapping[fid] = ref.finding_id
            # Independently computed common supporting facts, including those
            # needed to distinguish a domain choice from an arithmetic error.
            for key, text, unit in [('nonzero_points', 'Number of non-zero-speed points used for percentile estimation', 'count'),
                                    ('runner_up_peak', 'Second-highest connected-component peak speed', None)]:
                value = all_points[key]
                if value is None:
                    continue
                fid = new_bid + '_support_' + key
                spec = {'absolute_tolerance': 0 if unit == 'count' else 1e-7, 'relative_tolerance': None, 'spatial_tolerance': None}
                findings.append({'finding_id':fid,'category':'quantity','importance':'supporting','statement':text,
                                 'value':value,'unit':unit,'verification':spec,'evidence_ids':[]})
                policies.append({'operationalization_id':new_bid,'finding_id':fid,'verification_mode':'scalar_tolerance',
                                 'verification_parameters':spec})
            record.setdefault('new_branches', []).append({'operationalization':{
                'operationalization_id':new_bid,'decisions':bundles[bid].model_dump(mode='json')['decisions']},
                'findings':{'operationalization_id':new_bid,'findings':findings},
                'requirement_branch_id':bid,'requirement_map':mapping,'policies':policies})
            for branch_id, desc in [(bid, 'Finite points with IBlank > 0; the original reference excludes blanked grid points.'),
                                    (new_bid, 'All finite stored grid points, with no IBlank filtering; zero speeds excluded only for percentile estimation.')]:
                record.setdefault('branch_interpretations', {})[branch_id] = {
                    'dimension':'feature_definition','description':desc,'publicly_specified':False}
            from types import SimpleNamespace
            record['reporting_policies'] += reporting_precision_rules(SimpleNamespace(findings_by_operationalization=[]),
                [{'branch_id':new_bid,'finding':f} for f in findings])
        record['authority'] += '; INDEPENDENT_POPULATION_VARIANT_CONSTRUCTION'
        record['reviewed_by'] += '; source-question-and-validity-mask-audit'
        record['rationale'] += (' The released question and public metadata omit IBlank. Both finite stored-point and '
            'IBlank-valid materializations preserve all stated O constraints. Added populations apply uniformly to every model. '
            'Independent ndimage component construction reproduces every original masked numeric reference before adding the variant. '
            'No model answers or model IDs enter construction. This is a reference audit, not independent human judge calibration.')
    output = Path(output).resolve()
    record_path = output.with_name(output.stem+'_records.json')
    audit_path = output.with_name(output.stem+'_construction.json')
    if any(p.exists() for p in (output,record_path,audit_path)):
        raise ValueError('preserve existing reference versions')
    evidence['deduplicated_supporting_references']=[r for cid,record in records.items()
        if cid.startswith('nasa_lox_post_') for r in deduplicate_variant_references(record)]
    write_json(audit_path, evidence)
    for cid, record in records.items():
        if cid.startswith('nasa_lox_post_'):
            record['rationale'] += f' Frozen independent evidence: {audit_path.relative_to(ROOT)} SHA256 {checksum(audit_path)}.'
    write_json(record_path, records)
    write_json(output, {'protocol':'evaluation-reference-package-v1','entries':[
        {'case_id':cid,'task_sha256':r['task_sha256'],'source':{'path':str(record_path.relative_to(ROOT)),
         'sha256':checksum(record_path),'pointer':'/'+cid}} for cid,r in records.items()]})
    return {'checked_masked_branches':len(evidence['verified_masked_branches']), 'new_population_variants':len(evidence['variants']),
            'output':str(output),'api_calls':0}


def refresh_descriptions(base, output):
    """Correct copied prose in an existing frozen package; no numeric work."""
    output=Path(output).resolve()
    rp=output.with_name(output.stem+'_records.json')
    ap=output.with_name(output.stem+'_description_audit.json')
    if any(p.exists() for p in (output,rp,ap)):
        raise ValueError('preserve existing reference versions')
    records={cid:deepcopy(s['record']) for cid,s in load_supplements(base,ROOT).items()}
    manifest=json.loads((ROOT/'experiments/expansion_v1_development/case_manifest.json').read_text())
    changes=[]
    for item in manifest['cases']:
        cid=item['case_id']
        if not cid.startswith('nasa_lox_post_'):continue
        _,_,_,material=load_development_case(ROOT,item)
        for branch in records[cid].get('new_branches',[]):
            bid=branch['operationalization']['operationalization_id']
            suffix='_all_finite_stored'
            if not bid.endswith(suffix):continue
            original=bid[:-len(suffix)]
            plan=material['branch_execution_evidence'][original]['execution_provenance']['original_execution_provenance']['materialization_plan']
            for finding in branch['findings']['findings']:
                text=variant_statement(field_for_reference(bid,finding['finding_id']),plan['measure_kind'],plan['representation_kind'])
                if finding['statement']!=text:
                    changes.append({'case_id':cid,'finding_id':finding['finding_id'],
                        'before':finding['statement'],'after':text,'unchanged_value':finding['value']})
                    finding['statement']=text
    duplicates=[r for cid,record in records.items() if cid.startswith('nasa_lox_post_')
                for r in deduplicate_variant_references(record)]
    write_json(ap,{'role':'FROZEN_DESCRIPTION_CORRECTION_NO_NUMERIC_RECOMPUTATION',
        'base':str(Path(base).resolve()),'base_sha256':checksum(base),'script_sha256':checksum(__file__),
        'changes':changes,'deduplicated_supporting_references':duplicates})
    for cid in {c['case_id'] for c in changes}:
        records[cid]['rationale']+=f' Copied old-population numeric prose replaced by quantity labels; identical supporting aliases removed, retaining core values and core policies. Audit {ap.relative_to(ROOT)} SHA256 {checksum(ap)}.'
    write_json(rp,records)
    write_json(output,{'protocol':'evaluation-reference-package-v1','entries':[
        {'case_id':cid,'task_sha256':r['task_sha256'],'source':{'path':str(rp.relative_to(ROOT)),
          'sha256':checksum(rp),'pointer':'/'+cid}} for cid,r in records.items()]})
    return {'description_changes':len(changes),'deduplicated_supporting_references':len(duplicates),
            'numeric_changes':0,'api_calls':0,'output':str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refresh-descriptions',action='store_true',help='Correct copied labels in a frozen package without reading scientific arrays')
    args = parser.parse_args()
    print(json.dumps((refresh_descriptions if args.refresh_descriptions else build)(args.base,args.output)))

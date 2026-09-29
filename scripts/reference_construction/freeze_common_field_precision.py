#!/usr/bin/env python3
"""Freeze audited arithmetic sensitivity for supporting common-field moments.

Original values, core obligations and methods remain fixed. Only independently
reproduced common supporting statistics receive a measured numerical budget.
No answers, model identities or scores are inputs.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.reference_packages import load_supplements
from scripts.reference_construction.build_pressure_support_references import sha


def precision_tolerance(reference,baseline,variants):
    if not variants or baseline<0 or not all(math.isfinite(v) for v in [reference,baseline,*variants]):
        raise ValueError('finite independently measured values required')
    return baseline+max(abs(v-reference) for v in variants)


def build(base,audit,output):
    records_path=output.with_name(output.stem+'_records.json')
    changes_path=output.with_name(output.stem+'_precision_audit.json')
    snapshot=output.with_name(output.stem+'_builder.py')
    if any(p.exists() for p in [output,records_path,changes_path,snapshot]):raise ValueError('preserve frozen versions')
    evidence=json.loads(audit.read_text())
    if evidence['role']!='INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT':raise ValueError('independent precision audit required')
    for path,checksum in {evidence['source_snapshot']:evidence['script_sha256'],
            evidence['source_construction']:evidence['source_construction_sha256'],**evidence['dependencies']}.items():
        if sha(ROOT/path)!=checksum:raise ValueError('audit dependency changed')
    records={cid:deepcopy(s['record']) for cid,s in load_supplements(base,ROOT).items()};changes=[]
    for e in evidence['records']:
        if set(e['variants'])!={'vtk','float32','float64'}:raise ValueError('unsupported precision audit')
        for f in e['input_files']:
            if sha(ROOT/f['path'])!=f['sha256']:raise ValueError('audited data changed')
        for name,reference in e['variants']['vtk'].items():
            fid=e['branch_id']+'_common_'+name
            matched=[x for x in records[e['case_id']].get('supporting_findings',[])
                     if x['branch_id']==e['branch_id'] and x['finding']['finding_id']==fid]
            if len(matched)!=1:raise ValueError('exact common supporting reference required')
            entry=matched[0];f=entry['finding'];spec=f['verification']
            if f['importance']!='supporting' or not math.isclose(f['value'],reference,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('reference not reproduced')
            if spec.get('relative_tolerance') is not None or spec.get('spatial_tolerance') is not None:
                raise ValueError('absolute numerical budget required')
            old=spec['absolute_tolerance'];values=[v[name] for v in e['variants'].values()]
            new=precision_tolerance(f['value'],old,values)
            if new<=old*(1+1e-4):continue
            f['verification']['absolute_tolerance']=new
            entry['policy']['verification_parameters']['absolute_tolerance']=new
            changes.append({'case_id':e['case_id'],'branch_id':e['branch_id'],'finding_id':fid,
                'old_tolerance':old,'new_tolerance':new,'variants':{k:v[name] for k,v in e['variants'].items()}})
    for cid in {x['case_id'] for x in changes}:
        records[cid]['authority']+='; INDEPENDENT_COMMON_FIELD_PRECISION_AUDIT'
        records[cid]['rationale']+=(f' Common supporting moments allow independently measured VTK/NumPy float32/float64 '
            f'vertex-averaging sensitivity plus the original tolerance, on identical populations. '
            f'Evidence {audit.relative_to(ROOT)} SHA256 {sha(audit)}. No answer inputs. Core policies unchanged.')
    output.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(changes_path,{'source_audit':str(audit.relative_to(ROOT)),'source_audit_sha256':sha(audit),
        'script_snapshot':str(snapshot.relative_to(ROOT)),'script_sha256':sha(snapshot),
        'core_policies_changed':0,'changes':changes})
    write_json(records_path,records)
    write_json(output,{'protocol':'evaluation-reference-package-v1','entries':[
        {'case_id':cid,'task_sha256':r['task_sha256'],'source':{'path':str(records_path.relative_to(ROOT)),
         'sha256':sha(records_path),'pointer':'/'+cid}} for cid,r in records.items()]})
    return {'supporting_policies_changed':len(changes),'core_policies_changed':0,'api_calls':0,'output':str(output)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(build(a.base.resolve(),a.audit.resolve(),a.output.resolve())))

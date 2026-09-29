#!/usr/bin/env python3
"""Freeze common cell-field statistics outside the production scoring process.

Uses the released VTK sampling/volume conventions, but independently computes
weighted moments and reproduces each original coefficient before publishing.
No submitted answer, model identity, or desired score is an input.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.construction_recipes import _cell_view
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.reference_packages import load_supplements, reporting_precision_rules
from flowintentbench.external_file_evaluator import write_json
from scripts.reference_construction.build_pressure_support_references import sha


def moments(x,w):
    x,w=np.asarray(x,dtype=float),np.asarray(w,dtype=float)
    if x.ndim!=1 or x.shape!=w.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all() or (w<=0).any():
        raise ValueError('explicit finite positive-weight scalar population required')
    mean=float(np.average(x,weights=w))
    variance=float(np.average((x-mean)**2,weights=w))
    order=np.argsort(x,kind='stable');cumulative=np.cumsum(w[order])
    result={'mean':mean,'variance':variance,'std':float(np.sqrt(variance)),'minimum':float(x.min()),'maximum':float(x.max())}
    result.update({name:float(x[order[np.searchsorted(cumulative,q*w.sum(),side='left')]])
                   for q,name in [(.1,'q10'),(.5,'median'),(.9,'q90')]})
    return result


def coefficient(x,w,measure,y=None):
    if y is None:
        m=moments(x,w)
        if measure=='cv':return m['std']/abs(m['mean'])
        if measure=='relative_interdecile':return (m['q90']-m['q10'])/abs(m['median'])
    else:
        if measure=='spearman':x,y=rankdata(x,method='average'),rankdata(y,method='average')
        elif measure!='pearson':raise ValueError('unsupported coefficient')
        dx=x-np.average(x,weights=w);dy=y-np.average(y,weights=w)
        return float(np.dot(w,dx*dy)/np.sqrt(np.dot(w,dx*dx)*np.dot(w,dy*dy)))
    raise ValueError('unsupported coefficient')


def association_moments(x,y,w):
    """Raw-field joint moments, independent of a primary rank coefficient."""
    mx,my=moments(x,w),moments(y,w)
    covariance=float(np.average((x-mx['mean'])*(y-my['mean']),weights=w))
    if min(mx['variance'],my['variance'])<=0:raise ValueError('joint reference requires nonconstant fields')
    slope=covariance/mx['variance']
    return {'covariance':covariance,'regression_slope':slope,
            'regression_intercept':my['mean']-slope*mx['mean'],
            'pearson_squared':covariance**2/(mx['variance']*my['variance'])}


def build(base,output):
    output=output.resolve();rp=output.with_name(output.stem+'_records.json');ap=output.with_name(output.stem+'_construction.json')
    if any(p.exists() for p in [output,rp,ap]):raise ValueError('preserve frozen reference versions')
    records={cid:deepcopy(s['record']) for cid,s in load_supplements(base,ROOT).items()}
    manifest=json.loads((ROOT/'experiments/expansion_v1_development/case_manifest.json').read_text())
    cache={};audits=[]
    for item in manifest['cases']:
        if item['case_id'].startswith('aideas_pressure_heterogeneity_'):continue
        ci,meta,gt,material=load_development_case(ROOT,item)
        definition=material.get('family_definition',{})
        operations=definition.get('operations',{})
        if not operations or any(o.get('recipe',{}).get('kind') not in {'dispersion','association'} for o in operations.values()):continue
        dataset_id=material['source_case']['dataset_id']
        convention=definition.get('cell_volume_convention','positive_signed_volume')
        key=(dataset_id,convention)
        if key not in cache:
            data,dm=_reader_dataset(ROOT,dataset_id)
            sources=[]
            for spec in dm['files']:
                path=ROOT/dm.get('file_root','datasets')/spec['path']
                if sha(path)!=spec['checksum']:raise ValueError('input dataset hash mismatch')
                sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
            weights,_,arrays=_cell_view(data,convention)
            cache[key]=(weights,arrays,sources)
        w,arrays,sources=cache[key]
        mask=np.ones(len(w),dtype=bool)
        for spec in definition.get('finite_cell_fields',[]):
            a=arrays[(spec['association'],spec['name'])]
            mask &= np.isfinite(a).all(axis=1) if a.ndim==2 else np.isfinite(a)
        w=w[mask]
        def field(spec):
            a=arrays[(spec['association'],spec['name'])][mask]
            if 'component' in spec:a=a[:,spec['component']]
            elif a.ndim==2:
                if spec.get('reduce')!='magnitude':raise ValueError('vector needs an explicit reduction')
                a=np.linalg.norm(a,axis=1)
            return a
        record=records[item['case_id']];additions=[]
        for branch in gt.findings_by_operationalization:
            bid=branch.operationalization_id
            entry=material['branch_execution_evidence'][bid]
            recipe=entry['execution_provenance']['recipe']
            x=field(recipe['field']);y=field(recipe['other_field']) if recipe['kind']=='association' else None
            expected=coefficient(x,w,recipe['measure'],y)
            core=[f for f in branch.findings if f.importance.value=='core']
            if len(core)!=1 or not np.isclose(core[0].value,expected,rtol=1e-8,atol=1e-10):
                raise ValueError('independent coefficient mismatch: '+bid)
            evidence={'case_id':item['case_id'],'branch_id':bid,'input_files':sources,'volume_convention':convention,
                'finite_cell_fields':definition.get('finite_cell_fields',[]),'reproduced_coefficient':expected,'fields':{}}
            for slot,spec in [('x',recipe['field']),*([('y',recipe['other_field'])] if y is not None else [])]:
                values=moments(field(spec),w);evidence['fields'][slot]={'spec':spec,'moments':values}
                v=next(v for v in ci.flow_data.data_metadata.variables if v.name==spec['name'] and v.association==spec['association'])
                unit=v.unit
                label=spec['name']+(f' component {spec["component"]} (zero-based)' if 'component' in spec else
                    ' magnitude after cell averaging' if spec.get('reduce')=='magnitude' and spec['association']=='point' else
                    ' magnitude' if spec.get('reduce')=='magnitude' else '')
                conversion=('stored cell scalar values' if spec['association']=='cell' else 'arithmetic mean of vertex values at each cell before vector reduction')
                for name,value in values.items():
                    fid=bid+'_common_'+slot+'_'+name
                    label_stat={'mean':'mean','variance':'population variance','std':'population standard deviation','minimum':'minimum','maximum':'maximum',
                                'q10':'10th percentile','median':'median','q90':'90th percentile'}[name]
                    statement=f'{label_stat} of {label}; {conversion}; same complete cell population as this branch; {convention} weights'
                    if name in {'q10','median','q90'}:statement+='; inverse cumulative weight without interpolation'
                    spec_tol={'absolute_tolerance':max(abs(value)*1e-9,1e-10),'relative_tolerance':None,'spatial_tolerance':None}
                    additions.append({'branch_id':bid,'method_dimensions':['feature_definition','aggregation_or_representation'],
                        'finding':{'finding_id':fid,'category':'quantity','importance':'supporting','statement':statement,
                            'value':value,'unit':f'({unit})^2' if name=='variance' and unit else unit,'verification':spec_tol,'evidence_ids':[]},
                        'policy':{'operationalization_id':bid,'finding_id':fid,'verification_mode':'scalar_tolerance','verification_parameters':spec_tol}})
            joint = association_moments(x,y,w) if y is not None else {}
            evidence['joint_moments']=joint
            scope='same finite complete cell population and field construction as this branch; '+convention+' weights'
            measures={'included_volume':(float(w.sum()),'Sum of included cell volume; '+scope,None),
                      'included_cell_count':(len(w),'Number of included cells; '+scope,'count')}
            if joint:
                xspec,yspec=recipe['field'],recipe['other_field']
                def label(spec):
                    return spec['name']+(f' component {spec["component"]} (zero-based)' if 'component' in spec else
                        ' magnitude after cell averaging' if spec.get('reduce')=='magnitude' else '')
                ux=next(v.unit for v in ci.flow_data.data_metadata.variables if v.name==xspec['name'] and v.association==xspec['association'])
                uy=next(v.unit for v in ci.flow_data.data_metadata.variables if v.name==yspec['name'] and v.association==yspec['association'])
                labels={'covariance':('Population covariance of '+label(xspec)+' and '+label(yspec),f'({ux})*({uy})' if ux and uy else None),
                    'regression_slope':('Least-squares slope of '+label(yspec)+' regressed on '+label(xspec),f'({uy})/({ux})' if ux and uy else None),
                    'regression_intercept':('Least-squares intercept of '+label(yspec)+' regressed on '+label(xspec),uy),
                    'pearson_squared':('Squared raw-field Pearson correlation of '+label(xspec)+' and '+label(yspec)+' (not squared rank correlation)','1')}
                measures.update({name:(value,labels[name][0]+'; '+scope,labels[name][1]) for name,value in joint.items()})
            for name,(value,statement,unit) in measures.items():
                fid=bid+'_common_'+name
                spec_tol={'absolute_tolerance':0 if unit=='count' else max(abs(value)*1e-9,1e-10),
                          'relative_tolerance':None,'spatial_tolerance':None}
                additions.append({'branch_id':bid,'method_dimensions':['feature_definition','aggregation_or_representation'],
                    'finding':{'finding_id':fid,'category':'quantity','importance':'supporting','statement':statement,
                        'value':value,'unit':unit,'verification':spec_tol,'evidence_ids':[]},
                    'policy':{'operationalization_id':bid,'finding_id':fid,'verification_mode':'scalar_tolerance','verification_parameters':spec_tol}})
            audits.append(evidence)
        record.setdefault('supporting_findings',[]).extend(additions)
        record.setdefault('reporting_policies',[]).extend(reporting_precision_rules(SimpleNamespace(findings_by_operationalization=[]),additions))
        record['authority']+='; COMMON_FIELD_MOMENT_REFERENCE_CONSTRUCTION'
        record['reviewed_by']+='; declared-sampling-independent-moment-audit'
        record['rationale']+=' Added common marginal statistics, using released VTK sampling with independently computed weighted moments; original coefficients reproduced. No answer/model inputs. Core obligations unchanged. Common statistics depend on field/population and weighting, not choice of primary association/dispersion coefficient. Shared sampling is not independent human calibration.'
    write_json(ap,{'role':'INDEPENDENT_REFERENCE_CONSTRUCTION; NO_ANSWER_INPUT','script_sha256':sha(__file__),'branches':audits})
    for cid in {a['case_id'] for a in audits}:records[cid]['rationale']+=f' Evidence {ap.relative_to(ROOT)} SHA256 {sha(ap)}.'
    write_json(rp,records)
    write_json(output,{'protocol':'evaluation-reference-package-v1','entries':[
        {'case_id':cid,'task_sha256':r['task_sha256'],'source':{'path':str(rp.relative_to(ROOT)),'sha256':sha(rp),'pointer':'/'+cid}}
        for cid,r in records.items()]})
    return {'verified_branches':len(audits),'supplemented_cases':len({a['case_id'] for a in audits}),'output':str(output)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.base,args.output)))

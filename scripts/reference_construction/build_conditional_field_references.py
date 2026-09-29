#!/usr/bin/env python3
"""Construct fixed conditional-statistic references without model-answer inputs.

Released rectilinear association tasks only. Equal-width strata and weighted
quartiles are defined before computation, including whole-cell versus fractional
volume boundary conventions. VTK/float32/float64 sensitivity is measured before
freezing supporting tolerances. The evaluator never calls this program.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flowintentbench.construction_recipes import _cell_view
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.reference_packages import load_supplements,reporting_precision_rules
from scripts.reference_construction.audit_common_field_precision import cell_field
from scripts.reference_construction.build_common_field_references import coefficient
from scripts.reference_construction.build_pressure_support_references import sha


def conditional_statistics(x,y,w,scheme,count):
    x,y,w=(np.asarray(a,dtype=float) for a in (x,y,w))
    if (x.ndim!=1 or x.shape!=y.shape or x.shape!=w.shape or not len(x)
        or not all(np.isfinite(a).all() for a in (x,y,w)) or (w<=0).any()
        or type(count)!=int or count<2 or x.min()==x.max()):
        raise ValueError('finite nonconstant scalar population and positive weights required')
    total=w.sum();means=[];fractions=[];lower=[];upper=[]
    if scheme=='equal_volume':
        # Equal x ties receive identical fractional membership, independent of
        # their arbitrary mesh order. Boundary splitting is a separate recipe.
        order=np.argsort(x,kind='stable');sx,sy,sw=x[order],y[order],w[order]
        starts=np.r_[0,np.flatnonzero(sx[1:]!=sx[:-1])+1]
        mass=np.add.reduceat(sw,starts);ymass=np.add.reduceat(sw*sy,starts)
        right=np.cumsum(mass);left=right-mass
        for i in range(count):
            overlap=np.maximum(0,np.minimum(right,(i+1)*total/count)-np.maximum(left,i*total/count))
            included=overlap.sum()
            means.append(float(np.dot(overlap,ymass/mass)/included));fractions.append(float(included/total))
            active=overlap>0
            lower.append(float(sx[starts][active].min()));upper.append(float(sx[starts][active].max()))
    else:
        if scheme=='equal_width':edges=np.linspace(x.min(),x.max(),count+1)
        elif scheme=='weighted_quantile':
            order=np.argsort(x,kind='stable');cumulative=np.cumsum(w[order])
            inner=x[order[np.searchsorted(cumulative,np.arange(1,count)*total/count,side='left')]]
            edges=np.r_[x.min(),inner,x.max()]
            if len(np.unique(edges))!=len(edges):raise ValueError('tied quantile edges need explicit fractional allocation')
        else:raise ValueError('unsupported partition')
        groups=np.minimum(np.searchsorted(edges,x,side='right')-1,count-1)
        for i in range(count):
            mask=groups==i
            if not mask.any():raise ValueError('empty bin')
            mass=w[mask].sum();means.append(float(np.average(y[mask],weights=w[mask])))
            fractions.append(float(mass/total))
            lower.append(float(edges[i]));upper.append(float(edges[i+1]))
    if not np.isclose(sum(fractions),1,atol=1e-12):raise ValueError('partition lost volume')
    if not np.isclose(np.dot(means,fractions),np.average(y,weights=w),atol=1e-12,rtol=1e-10):
        raise ValueError('partition lost weighted first moment')
    return {'y_mean':means,'volume_fraction':fractions,'x_lower':lower,'x_upper':upper}


def build(base,output):
    rp=output.with_name(output.stem+'_records.json');ap=output.with_name(output.stem+'_construction.json')
    snapshot=output.with_name(output.stem+'_builder.py')
    if any(p.exists() for p in [output,rp,ap,snapshot]):raise ValueError('preserve frozen versions')
    records={cid:deepcopy(s['record']) for cid,s in load_supplements(base,ROOT).items()}
    manifest=json.loads((ROOT/'experiments/expansion_v1_development/case_manifest.json').read_text())
    cache={};audit=[];skipped=[];added=0
    for item in manifest['cases']:
        ci,_,gt,material=load_development_case(ROOT,item)
        definition=material.get('family_definition',{});ops=definition.get('operations',{})
        if not ops or any(v.get('recipe',{}).get('kind')!='association' for v in ops.values()):continue
        dataset=material['source_case']['dataset_id'];convention=definition.get('cell_volume_convention','positive_signed_volume')
        key=(dataset,convention)
        if key not in cache:
            mesh,dm=_reader_dataset(ROOT,dataset)
            if not mesh.IsA('vtkRectilinearGrid'):cache[key]=None;continue
            sources=[]
            for f in dm['files']:
                path=ROOT/dm.get('file_root','datasets')/f['path']
                if sha(path)!=f['checksum']:raise ValueError('source changed')
                sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
            w,_,arrays=_cell_view(mesh,convention);cache[key]=(mesh,w,arrays,sources,{})
        if cache[key] is None:continue
        mesh,w,arrays,sources,computed=cache[key]
        additions=[]
        for branch in gt.findings_by_operationalization:
            bid=branch.operationalization_id;recipe=material['branch_execution_evidence'][bid]['execution_provenance']['recipe']
            xs,ys=recipe['field'],recipe['other_field']
            field_key=json.dumps([xs,ys,definition.get('finite_cell_fields',[])],sort_keys=True)
            if field_key not in computed:
                mask=np.ones(len(w),dtype=bool)
                for spec in definition.get('finite_cell_fields',[]):
                    a=arrays[(spec['association'],spec['name'])]
                    mask &= np.isfinite(a).all(axis=1) if a.ndim==2 else np.isfinite(a)
                def native(spec):
                    a=arrays[(spec['association'],spec['name'])][mask]
                    if 'component' in spec:return a[:,spec['component']]
                    if a.ndim==2:
                        if spec.get('reduce')!='magnitude':raise ValueError('explicit vector reduction required')
                        return np.linalg.norm(a,axis=1)
                    return a
                fields={'vtk':(native(xs),native(ys))}
                for name,dtype in [('float32',np.float32),('float64',np.float64)]:
                    fields[name]=(cell_field(mesh,xs,dtype)[mask],cell_field(mesh,ys,dtype)[mask])
                results={};failures=[]
                for scheme,count in [('weighted_quantile',4),('equal_volume',4),('equal_width',5),('equal_width',10)]:
                    try:results[(scheme,count)]={mode:conditional_statistics(x,y,w[mask],scheme,count) for mode,(x,y) in fields.items()}
                    except ValueError as exc:failures.append({'scheme':scheme,'count':count,'reason':str(exc)})
                computed[field_key]=(fields['vtk'],w[mask],results,failures)
            (x,y),weights,results,failures=computed[field_key]
            core=[f for f in branch.findings if f.importance.value=='core']
            observed=coefficient(x,weights,recipe['measure'],y)
            if len(core)!=1 or not np.isclose(core[0].value,observed,rtol=1e-8,atol=1e-10):raise ValueError('core coefficient mismatch')
            def label(spec):
                return spec['name']+(f' component {spec["component"]} (zero-based)' if 'component' in spec else
                    ' magnitude after cell averaging' if spec.get('reduce')=='magnitude' else '')
            units={slot:next(v.unit for v in ci.flow_data.data_metadata.variables if v.name==spec['name'] and v.association==spec['association'])
                   for slot,spec in [('x',xs),('y',ys)]}
            for (scheme,count),variants in results.items():
                description=({'equal_width':f'{count} equal-width {label(xs)} bins spanning its complete cell minimum to maximum; left closed/right open except last endpoint',
                    'weighted_quantile':f'{count} volume-weighted quantile {label(xs)} bins; inverse cumulative-weight boundaries; whole cells, left closed/right open except last endpoint',
                    'equal_volume':f'{count} equal-volume groups sorted by {label(xs)}; fractional boundary-cell allocation, tied {label(xs)} values split proportionally'}[scheme])
                for stat,vector in variants['vtk'].items():
                    for index in [None,'intermediate',*range(count)]:
                        def select(vector):return vector if index is None else vector[1:-1] if index=='intermediate' else vector[index]
                        value=select(vector)
                        values=[select(v[stat]) for v in variants.values()]
                        spread=max(float(np.max(np.abs(np.asarray(v)-value))) for v in values)
                        tolerance=max(float(np.max(np.abs(value)))*1e-9,1e-10)+spread
                        fid=f'{bid}_conditional_{scheme}_{count}_{stat}_'+('all' if index is None else 'intermediate' if index=='intermediate' else str(index+1))
                        boundary=(f'of {label(xs)} actually allocated to the group' if scheme=='equal_volume' else f'boundary of the {label(xs)} partition interval')
                        statement={'y_mean':f'Volume-weighted mean of {label(ys)}','volume_fraction':'Fraction of total included volume',
                                   'x_lower':'Minimum '+boundary,'x_upper':'Maximum '+boundary}[stat]
                        statement+=('; all bins in increasing order' if index is None else '; intermediate bins in increasing order, excluding lowest and highest' if index=='intermediate'
                                    else f'; bin {index+1} of {count} in increasing order')
                        statement+='; '+description+f'; complete finite cell population and field construction of this branch; {convention} weights'
                        is_vector=isinstance(value,list)
                        spec={'absolute_tolerance':tolerance if not is_vector else None,
                              'relative_tolerance':None,'spatial_tolerance':tolerance if is_vector else None}
                        additions.append({'branch_id':bid,'method_dimensions':['feature_definition','aggregation_or_representation'],
                            'finding':{'finding_id':fid,'category':'quantity','importance':'supporting','statement':statement,'value':value,
                                'unit':units['y'] if stat=='y_mean' else '1' if stat=='volume_fraction' else units['x'],'verification':spec,'evidence_ids':[]},
                            'policy':{'operationalization_id':bid,'finding_id':fid,
                                'verification_mode':'componentwise_vector' if is_vector else 'scalar_tolerance','verification_parameters':spec}})
                audit.append({'case_id':item['case_id'],'branch_id':bid,'input_files':sources,'scheme':scheme,'count':count,
                    'fields':[xs,ys],'variants':variants,'reproduced_coefficient':observed})
            skipped.extend({'case_id':item['case_id'],'branch_id':bid,**f} for f in failures)
        if additions:
            r=records[item['case_id']];r.setdefault('supporting_findings',[]).extend(additions)
            r.setdefault('reporting_policies',[]).extend(reporting_precision_rules(SimpleNamespace(findings_by_operationalization=[]),additions))
            r['authority']+='; INDEPENDENT_CONDITIONAL_FIELD_STATISTICS'
            r['rationale']+=' Fixed conditional bins, separately declared boundary conventions; VTK/float32/float64 arithmetic spread frozen. No answer inputs; core obligations unchanged.'
            added+=len(additions)
    output.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(ap,{'role':'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT','source_snapshot':str(snapshot.relative_to(ROOT)),
        'script_sha256':sha(snapshot),'dependencies':{str(Path(m.__file__).relative_to(ROOT)):sha(m.__file__) for m in
            [sys.modules['scripts.reference_construction.audit_common_field_precision'],sys.modules['scripts.reference_construction.build_common_field_references']]},
        'records':audit,'skipped':skipped})
    for cid in {a['case_id'] for a in audit}:records[cid]['rationale']+=f' Evidence {ap.relative_to(ROOT)} SHA256 {sha(ap)}.'
    write_json(rp,records)
    write_json(output,{'protocol':'evaluation-reference-package-v1','entries':[{'case_id':cid,'task_sha256':r['task_sha256'],
        'source':{'path':str(rp.relative_to(ROOT)),'sha256':sha(rp),'pointer':'/'+cid}} for cid,r in records.items()]})
    return {'added_supporting':added,'audited_partitions':len(audit),'skipped':len(skipped),'api_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.base.resolve(),a.output.resolve())))

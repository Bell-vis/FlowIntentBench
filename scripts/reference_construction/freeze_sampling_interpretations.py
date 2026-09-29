#!/usr/bin/env python3
"""Freeze audited point-vector reduction alternatives when public order is unspecified.

Consumes independent construction evidence only. It never reads model answers.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flowintentbench.reference_packages import load_supplements,reporting_precision_rules
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from scripts.reference_construction.build_pressure_support_references import sha


def add_width_interpretations(e,ci,gt,mat,record):
    """Same named stored point/cell vectors; preserve averaging-before-norm."""
    recipe=e['recipe'];spec=recipe['field'];name=spec['name']
    associations={v.association for v in ci.flow_data.data_metadata.variables if v.name==name}
    if associations!={'point','cell'} or spec.get('reduce')!='magnitude':
        raise ValueError('both public vector associations required')
    if 'arithmetic means of vertex values for point fields' not in ci.scientific_question and not e['case_id'].endswith('_o3_f1'):
        raise ValueError('public field-association review required')
    old=e['branch_id'];oldbranch=next(b for b in gt.findings_by_operationalization if b.operationalization_id==old)
    bundle=next(b for b in gt.acceptable_operationalizations if b.operationalization_id==old).model_dump(mode='json')
    choices=['stored_cell_vector_magnitude','stored_cell_vector_magnitude_vertex_coordinate_means',
             'average_vector_then_magnitude','average_vector_then_magnitude_vertex_coordinate_means']
    new=[]
    for choice in choices:
        bid=old if choice==e['original_convention'] else old+'_'+choice
        data=e['variants'][choice]
        description=(f'Use stored cell-associated {name} vectors directly before taking magnitude. ' if choice.startswith('stored_cell')
                     else f'Use stored point-associated {name} vectors; average vertex vector components at each cell, then take magnitude. ')
        description+=('Represent each cell coordinate by the arithmetic mean of its vertex coordinates.' if choice.endswith('_vertex_coordinate_means')
                      else 'Represent each cell coordinate by its VTK parametric center.')
        record.setdefault('branch_interpretations',{})[bid]={'dimension':'feature_definition','description':description,'publicly_specified':False}
        if bid!=old:
            findings=[];policies=[];mapping={}
            for ref in oldbranch.findings:
                key='center' if 'center' in ref.finding_id else 'width';fid=bid+'_'+key
                findings.append(ref.model_copy(update={'finding_id':fid,'value':data[key]}).model_dump(mode='json'))
                if ref.importance.value=='core':mapping[fid]=ref.finding_id
                policy=next(x for x in mat['finding_verification_policy']['policies'] if x['finding_id']==ref.finding_id)
                policies.append({k:v for k,v in {**policy,'operationalization_id':bid,'finding_id':fid}.items()
                                 if k in {'operationalization_id','finding_id','verification_mode','verification_parameters'}})
            newbundle=deepcopy(bundle);newbundle['operationalization_id']=bid
            record.setdefault('new_branches',[]).append({'operationalization':newbundle,'findings':{'operationalization_id':bid,'findings':findings},
                'requirement_branch_id':old,'requirement_map':mapping,'policies':policies})
            record.setdefault('reporting_policies',[]).extend(reporting_precision_rules(SimpleNamespace(findings_by_operationalization=[]),
                [{'branch_id':bid,'finding':f} for f in findings]))
            new.append(bid)
        fid=bid+'_support_total_weight';value=data['total_weight']
        tol={'absolute_tolerance':max(abs(value)*1e-9,1e-10),'relative_tolerance':None,'spatial_tolerance':None}
        f={'finding_id':fid,'category':'quantity','importance':'supporting','statement':'Sum of velocity-magnitude times cell volume over the complete cell population for this field association',
           'value':value,'unit':None,'verification':tol,'evidence_ids':[]}
        addition={'branch_id':bid,'method_dimensions':['feature_definition','aggregation_or_representation'],'finding':f,
                  'policy':{'operationalization_id':bid,'finding_id':fid,'verification_mode':'scalar_tolerance','verification_parameters':tol}}
        record.setdefault('supporting_findings',[]).append(addition)
        record.setdefault('reporting_policies',[]).extend(reporting_precision_rules(SimpleNamespace(findings_by_operationalization=[]),[addition]))
    return new


def build(base,audit,output):
    output=output.resolve();rp=output.with_name(output.stem+'_records.json');sp=output.with_name(output.stem+'_builder.py')
    if any(p.exists() for p in (output,rp,sp)):raise ValueError('preserve frozen reference versions')
    evidence=json.loads(audit.read_text())
    if evidence['role']!='INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT':raise ValueError('independent construction required')
    if sha(ROOT/evidence['source_snapshot'])!=evidence['script_sha256']:raise ValueError('construction source snapshot mismatch')
    records={cid:deepcopy(s['record']) for cid,s in load_supplements(base,ROOT).items()}
    manifest=json.loads((ROOT/'experiments/expansion_v1_development/case_manifest.json').read_text())
    cases={i['case_id']:i for i in manifest['cases']};new=[]
    for e in evidence['records']:
        recipe=e['recipe'];spec=recipe.get('other_field',{})
        if recipe['kind']!='weighted_width' and (recipe['kind']!='association' or spec.get('association')!='point' or spec.get('reduce')!='magnitude'):continue
        for src in e['input_files']:
            if sha(ROOT/src['path'])!=src['sha256']:raise ValueError('audited data changed')
        ci,meta,gt,mat=load_development_case(ROOT,cases[e['case_id']])
        if ci.scientific_question!=e['question']:raise ValueError('public question changed')
        if recipe['kind']=='weighted_width':
            record=records[e['case_id']]
            new.extend(add_width_interpretations(e,ci,gt,mat,record))
            record['authority']+='; PUBLIC_FIELD_ASSOCIATION_AND_COORDINATE_INTERPRETATION'
            record['reviewed_by']+='; independent-width-sampling-audit'
            record['rationale']+=f' Both point and cell vector fields are publicly supplied with the same name; task wording does not select association or a parametric-versus-vertex-mean cell coordinate. Freeze these interpretations with original obligations and tolerances. Evidence {audit.relative_to(ROOT)} SHA256 {sha(audit)}. Bind methods from answer text; no numerical-proximity inference.'
            continue
        # This authoring rule is limited to the released wording: the named
        # point-derived magnitude is requested, without an explicit ordering.
        if ('arithmetic means of vertex values for point fields' not in ci.scientific_question
                and not e['case_id'].endswith('_o3_f1')):
            raise ValueError('sampling interpretation needs public-wording review')
        record=records[e['case_id']];old=e['branch_id'];bid=old+'_point_magnitude_then_average'
        oldbranch=next(b for b in gt.findings_by_operationalization if b.operationalization_id==old)
        oldbundle=next(b for b in gt.acceptable_operationalizations if b.operationalization_id==old)
        values=e['variants']['magnitude_then_average'];findings=[];policies=[];mapping={}
        for ref in oldbranch.findings:
            fid=bid+'_coefficient'
            f=ref.model_copy(update={'finding_id':fid,'value':values['coefficient'],
                'statement':'Numerical association coefficient with pointwise vector magnitude computed before vertex-to-cell averaging'}).model_dump(mode='json')
            findings.append(f)
            if ref.importance.value=='core':mapping[fid]=ref.finding_id
            p=next(x for x in mat['finding_verification_policy']['policies'] if x['finding_id']==ref.finding_id)
            policies.append({k:v for k,v in {**p,'operationalization_id':bid,'finding_id':fid}.items()
                             if k in {'operationalization_id','finding_id','verification_mode','verification_parameters'}})
        bundle=oldbundle.model_dump(mode='json');bundle['operationalization_id']=bid
        record.setdefault('new_branches',[]).append({'operationalization':bundle,'findings':{'operationalization_id':bid,'findings':findings},
            'requirement_branch_id':old,'requirement_map':mapping,'policies':policies})
        record.setdefault('reporting_policies',[]).extend(reporting_precision_rules(SimpleNamespace(findings_by_operationalization=[]),
            [{'branch_id':bid,'finding':f} for f in findings]))
        interpretations=record.setdefault('branch_interpretations',{})
        for key,description in ((old,'Average each stored point-vector component over cell vertices, then compute the magnitude of the cell vector.'),
                                (bid,'Compute vector magnitude at each stored point first, then average those scalar magnitudes over cell vertices.')):
            interpretations[key]={'dimension':'feature_definition','description':description,
                'publicly_specified':False}
        additions=[]
        for group,prefix in (('scalar_moments','density'),('vector_magnitude_moments','pointwise magnetic magnitude averaged per cell')):
            for stat,value in values[group].items():
                additions.append((group+'_'+stat, prefix+' '+stat,value,None))
            additions.append((group+'_variance',prefix+' population variance',values[group]['std']**2,None))
        additions.extend([('raw_covariance','Volume-weighted raw covariance of density and pointwise magnetic magnitude averaged per cell',values['raw_covariance'],None),
                          ('included_volume','Total included cell volume',values['included_volume'],None),
                          ('cell_count','Included cell count',values['cell_count'],'count')])
        support=[]
        for key,statement,value,unit in additions:
            fid=bid+'_support_'+key
            tol={'absolute_tolerance':0 if unit=='count' else max(abs(value)*1e-9,1e-10),'relative_tolerance':None,'spatial_tolerance':None}
            f={'finding_id':fid,'category':'quantity','importance':'supporting','statement':statement+'; complete cell population, cell-volume weights',
               'value':value,'unit':unit,'verification':tol,'evidence_ids':[]}
            support.append({'branch_id':bid,'method_dimensions':['feature_definition','aggregation_or_representation'],'finding':f,
                'policy':{'operationalization_id':bid,'finding_id':fid,'verification_mode':'scalar_tolerance','verification_parameters':tol}})
        record.setdefault('supporting_findings',[]).extend(support)
        record['reporting_policies'].extend(reporting_precision_rules(SimpleNamespace(findings_by_operationalization=[]),support))
        record['authority']+='; PUBLIC_NONLINEAR_SAMPLING_INTERPRETATION'
        record['reviewed_by']+='; independent-vector-reduction-audit'
        record['rationale']+=f' Public wording requests a derived point-field magnitude and vertex means without specifying reduction order. Both explicit orders are represented and must bind to answer method evidence, never numerical proximity. Original fixed decision text, obligations and tolerances retained. Audit {audit.relative_to(ROOT)} SHA256 {sha(audit)}.'
        new.append(bid)
    sp.write_bytes(Path(__file__).read_bytes())
    write_json(rp,records)
    write_json(output,{'protocol':'evaluation-reference-package-v1','entries':[{'case_id':cid,'task_sha256':r['task_sha256'],
        'source':{'path':str(rp.relative_to(ROOT)),'sha256':sha(rp),'pointer':'/'+cid}} for cid,r in records.items()]})
    return {'new_branches':new,'output':str(output),'api_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.base.resolve(),a.audit.resolve(),a.output)))

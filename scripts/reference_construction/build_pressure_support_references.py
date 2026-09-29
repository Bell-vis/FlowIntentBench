#!/usr/bin/env python3
"""Independently freeze common pressure statistics; never invoked by scoring.

Inputs are the released dataset and reference package, never submitted answers.
The two original heterogeneity coefficients must be reproduced before adding
supporting results. Core obligations and original tolerances are unchanged.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.reference_packages import load_supplements, reporting_precision_rules


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(pressure, signed_volumes):
    p, signed = np.asarray(pressure, dtype=float), np.asarray(signed_volumes, dtype=float)
    if p.ndim != 1 or p.shape != signed.shape or not len(p):
        raise ValueError('expected matching nonempty scalar cell arrays')
    if not np.isfinite(p).all() or not np.isfinite(signed).all():
        raise ValueError('nonfinite cells require an explicitly authored population policy')
    w = np.abs(signed)
    if not w.sum() > 0:
        raise ValueError('zero included volume')
    mean = float(np.average(p, weights=w))
    variance = float(np.average((p - mean)**2, weights=w))
    order = np.argsort(p, kind='stable')
    cumulative = np.cumsum(w[order])
    quantiles = [float(p[order[np.searchsorted(cumulative, q*w.sum(), side='left')]])
                 for q in (.1, .5, .9)]
    if mean == 0 or quantiles[1] == 0 or p.mean() == 0:
        raise ValueError('undefined relative dispersion')
    return {'weighted_mean': mean, 'weighted_variance': variance,
        'weighted_std': float(np.sqrt(variance)), 'minimum': float(p.min()), 'maximum': float(p.max()),
        'weighted_q10': quantiles[0], 'weighted_median': quantiles[1], 'weighted_q90': quantiles[2],
        'absolute_volume': float(w.sum()), 'negative_volume_cells': int((signed < 0).sum()),
        'cell_count': len(p), 'arithmetic_mean': float(p.mean()), 'unweighted_std': float(p.std()),
        'unweighted_cv': float(p.std()/abs(p.mean())),
        'weighted_mad_ratio': float(np.average(np.abs(p-mean), weights=w)/abs(mean)),
        'cv': float(np.sqrt(variance)/abs(mean)),
        'relative_interdecile': float((quantiles[2]-quantiles[0])/abs(quantiles[1]))}


def spatial_statistics(p, signed, centers, bounds, cell_types):
    """Fixed full-domain queries; scopes are chosen without answer values."""
    w = np.abs(signed); total = w.sum()
    mean = np.average(p, weights=w); delta = p-mean
    variance = np.average(delta**2, weights=w)
    out, labels = {}, {}
    def put(key, value, statement, unit='1'):
        out[key] = float(value); labels[key] = (statement,unit)
    put('signed_volume', signed.sum(), 'Sum of signed VTK cell volumes over all cells, stored coordinate-volume units', None)
    put('effective_n', total**2/np.dot(w,w), 'Effective sample size (sum absolute volumes)^2 / sum squared absolute volumes', '1')
    put('reliability_cv', np.sqrt(variance/(1-np.dot(w,w)/total**2))/abs(mean),
        'Reliability-weight corrected pressure CV; variance divided by 1-sum normalized squared volume weights')
    for sign, mask in [('positive',signed>0),('negative',signed<0)]:
        m=np.average(p[mask],weights=w[mask]);v=np.average((p[mask]-m)**2,weights=w[mask])
        put(sign+'_volume_subset_cv',np.sqrt(v)/abs(m),f'Absolute-volume-weighted pressure CV restricted to cells with {sign} signed VTK volume')
    for axis, name in enumerate('xyz'):
        x = centers[:,axis]; dx=x-np.average(x,weights=w)
        covariance=np.average(dx*delta,weights=w); var_x=np.average(dx**2,weights=w)
        put('correlation_'+name,covariance/np.sqrt(var_x*variance),
            f'Absolute-volume-weighted Pearson correlation of cell pressure and VTK cell parametric-center {name} coordinate')
        if name=='z':
            slope=covariance/var_x;intercept=mean-slope*np.average(x,weights=w)
            residual=np.average((p-(intercept+slope*x))**2,weights=w)
            put('z_fit_slope',slope,'Full-domain absolute-volume-weighted linear fit p = intercept + slope*z; VTK cell parametric centers',None)
            put('z_fit_intercept',intercept,'Pressure intercept of full-domain absolute-volume-weighted linear fit against VTK cell parametric-center z','bar')
            put('z_fit_r_squared',1-residual/variance,'Fraction of volume-weighted pressure variance explained by full-domain linear fit against cell-center z')
            put('z_fit_residual_std',np.sqrt(residual),'Volume-weighted pressure residual standard deviation after full-domain linear fit against cell-center z','bar')
    for count in (15,30):
        edges=np.linspace(bounds[4],bounds[5],count+1)
        groups=np.clip(np.searchsorted(edges,centers[:,2],side='right')-1,0,count-1)
        if (centers[:,2]<edges[0]).any() or (centers[:,2]>edges[-1]).any():
            raise ValueError('cell centers outside authored slab domain')
        masses=[];means=[];variances=[]
        for k in range(count):
            mask=groups==k; mass=w[mask].sum()
            if mass<=0:raise ValueError('empty spatial slab needs a separate convention')
            m=np.average(p[mask],weights=w[mask]);v=np.average((p[mask]-m)**2,weights=w[mask])
            masses.append(mass);means.append(m);variances.append(v)
        between=np.average((np.array(means)-mean)**2,weights=masses)
        within=np.average(variances,weights=masses)
        if not np.isclose(between+within,variance,rtol=1e-10):raise ValueError('total variance identity failed')
        scope=f'{count} equal-width z slabs over [{edges[0]:g},{edges[-1]:g}], VTK parametric-center assignment, left closed/right open except final endpoint, absolute-volume weights'
        for key,value,description,unit in [
            ('between_variance',between,'Between-slab pressure variance','bar^2'),
            ('within_variance',within,'Within-slab pressure variance','bar^2'),
            ('between_fraction',between/variance,'Between-slab fraction of total pressure variance','1'),
            ('within_fraction',within/variance,'Within-slab fraction of total pressure variance','1'),
            ('first_mean',means[0],'First slab weighted mean pressure','bar'),
            ('last_mean',means[-1],'Last slab weighted mean pressure','bar'),
            ('std_min',np.sqrt(min(variances)),'Minimum of within-slab population pressure standard deviations','bar'),
            ('std_max',np.sqrt(max(variances)),'Maximum of within-slab population pressure standard deviations','bar')]:
            put(f'slabs{count}_{key}',value,description+'; '+scope,unit)
    for code,name in [(10,'tetrahedral'),(12,'hexahedral'),(13,'wedge'),(14,'pyramidal')]:
        mask=cell_types==code
        put(name+'_volume_fraction',w[mask].sum()/total,f'Fraction of total absolute cell volume carried by {name} cells')
        put(name+'_variance_fraction',np.dot(w[mask],delta[mask]**2)/(total*variance),
            f'Fraction of global-mean-centered volume-weighted pressure variance contributed by {name} cells')
    return out, labels


SUPPORT = {
    'weighted_mean': ('Mean stored cell pressure, weighted by absolute signed VTK cell volume', 'bar'),
    'weighted_std': ('Population standard deviation of stored cell pressure, weighted by absolute signed VTK cell volume', 'bar'),
    'weighted_variance': ('Population variance of stored cell pressure, weighted by absolute signed VTK cell volume', 'bar^2'),
    'minimum': ('Minimum stored pressure over all cells', 'bar'),
    'maximum': ('Maximum stored pressure over all cells', 'bar'),
    'weighted_q10': ('Volume-weighted pressure 10th percentile, inverse cumulative weight, no interpolation', 'bar'),
    'weighted_median': ('Volume-weighted pressure median, inverse cumulative weight, no interpolation', 'bar'),
    'weighted_q90': ('Volume-weighted pressure 90th percentile, inverse cumulative weight, no interpolation', 'bar'),
    'absolute_volume': ('Sum of absolute signed VTK volumes over all cells, in stored coordinate-volume units', None),
    'negative_volume_cells': ('Number of all mesh cells with strictly negative signed VTK cell volume', 'count'),
    'cell_count': ('Total number of mesh cells', 'count'),
    'arithmetic_mean': ('Unweighted arithmetic mean stored cell pressure; equal weight per cell', 'bar'),
    'unweighted_std': ('Unweighted population standard deviation of stored cell pressure; equal weight per cell', 'bar'),
    'unweighted_cv': ('Unweighted population pressure standard deviation divided by absolute arithmetic mean; equal weight per cell', '1'),
    'weighted_mad_ratio': ('Volume-weighted mean absolute deviation of pressure about weighted mean, divided by absolute weighted mean', '1'),
}


def build(base, output):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    output = output.resolve()
    record_path = output.with_name(output.stem+'_records.json')
    audit_path = output.with_name(output.stem+'_construction.json')
    if any(p.exists() for p in (output, record_path, audit_path)):
        raise ValueError('preserve existing reference versions')
    dataset_manifest = ROOT/'datasets/AIDEAS_Blow_Mold/dataset_manifest.json'
    spec = json.loads(dataset_manifest.read_text())['files'][0]
    path = ROOT/'datasets'/spec['path']
    if sha(path) != spec['checksum']:
        raise ValueError('dataset identity mismatch')
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    size = vtk.vtkCellSizeFilter()
    size.SetInputData(mesh)
    size.Update()
    p = vtk_to_numpy(mesh.GetCellData().GetArray('Pressure [bar]')).astype(float)
    signed = vtk_to_numpy(size.GetOutput().GetCellData().GetArray('Volume')).astype(float)
    values = statistics(p, signed)
    centers_filter=vtk.vtkCellCenters();centers_filter.SetInputData(mesh);centers_filter.Update()
    centers=vtk_to_numpy(centers_filter.GetOutput().GetPoints().GetData()).astype(float)
    extras, extra_labels=spatial_statistics(p,signed,centers,mesh.GetBounds(),vtk_to_numpy(mesh.GetCellTypesArray()))
    values.update(extras)
    support={**SUPPORT,**extra_labels}
    # Cross-check weighted moments through an independent raw second moment.
    raw_variance = float(np.dot(np.abs(signed), p*p)/np.abs(signed).sum()-values['weighted_mean']**2)
    if not np.isclose(raw_variance, values['weighted_variance'], rtol=1e-10):
        raise ValueError('independent variance formulas disagree')
    records = {cid: deepcopy(s['record']) for cid,s in load_supplements(base, ROOT).items()}
    manifest = json.loads((ROOT/'experiments/expansion_v1_development/case_manifest.json').read_text())
    checked = []
    for item in manifest['cases']:
        if not item['case_id'].startswith('aideas_pressure_heterogeneity_'):
            continue
        ci, meta, gt, material = load_development_case(ROOT, item)
        record = records[item['case_id']]
        additions = []
        bundles={b.operationalization_id:b for b in gt.acceptable_operationalizations}
        target_branches=list(gt.findings_by_operationalization)
        if 'property_measure' in {d.value for d in meta.unresolved_operationalization_dimensions}:
            original=next(b for b in target_branches if b.operationalization_id.endswith('_cv'))
            old_bid=original.operationalization_id;bid=old_bid+'_signed_mean'
            bundle=bundles[old_bid].model_dump(mode='json');bundle['operationalization_id']=bid
            for decision in bundle['decisions']:
                if decision['dimension']=='property_measure':
                    decision['statement']='Use the volume-weighted population standard deviation divided by the signed volume-weighted mean (no absolute value in the denominator).'
            core=next(f for f in original.findings if f.importance.value=='core')
            new=core.model_copy(update={'finding_id':bid+'_coefficient','value':values['weighted_std']/values['weighted_mean'],
                'statement':'Volume-weighted pressure population standard deviation divided by signed volume-weighted mean'})
            policy=next(x for x in material['finding_verification_policy']['policies'] if x['finding_id']==core.finding_id)
            policy={k:v for k,v in policy.items() if k in {'verification_mode','verification_parameters'}}
            policy.update(operationalization_id=bid,finding_id=new.finding_id)
            record.setdefault('new_branches',[]).append({'operationalization':bundle,
                'findings':{'operationalization_id':bid,'findings':[new.model_dump(mode='json')]},
                'requirement_branch_id':old_bid,'requirement_map':{new.finding_id:core.finding_id},'policies':[policy]})
            record.setdefault('reporting_policies',[]).extend(reporting_precision_rules(
                SimpleNamespace(findings_by_operationalization=[]),[{'branch_id':bid,'finding':new.model_dump(mode='json')}]))
            target_branches.append(SimpleNamespace(operationalization_id=bid,findings=[]))
        for branch in target_branches:
            bid = branch.operationalization_id
            recipe = material['branch_execution_evidence'].get(bid,{}).get('execution_provenance',{}).get('recipe',{'measure':'cv'})
            if recipe['measure'] not in ('cv', 'relative_interdecile'):
                raise ValueError('unsupported original method')
            core = [f for f in branch.findings if f.importance.value == 'core']
            if not bid.endswith('_signed_mean') and (len(core) != 1 or not np.isclose(core[0].value, values[recipe['measure']], rtol=1e-9, atol=1e-10)):
                raise ValueError('independent construction does not reproduce original reference: '+bid)
            if not bid.endswith('_signed_mean'):checked.append(bid)
            for key, (description, unit) in support.items():
                fid = bid+'_support_'+key
                value = values[key]
                tolerance = 0 if unit == 'count' else max(abs(value)*1e-9, 1e-10)
                spec = {'absolute_tolerance':tolerance, 'relative_tolerance':None, 'spatial_tolerance':None}
                # All entries use the same cell population. Unweighted
                # statistics/counts explicitly define their own weights;
                # weighted moments also depend on the declared volume rule.
                dependencies = ['feature_definition']
                if key not in {'minimum','maximum','cell_count','arithmetic_mean','unweighted_std','unweighted_cv'}:
                    dependencies.append('aggregation_or_representation')
                additions.append({'branch_id':bid, 'method_dimensions':dependencies, 'finding':{
                    'finding_id':fid, 'category':'quantity', 'importance':'supporting', 'statement':description,
                    'value':value, 'unit':unit, 'verification':spec, 'evidence_ids':[]},
                    'policy':{'operationalization_id':bid, 'finding_id':fid,
                              'verification_mode':'scalar_tolerance', 'verification_parameters':spec}})
        record.setdefault('supporting_findings', []).extend(additions)
        record.setdefault('reporting_policies', []).extend(reporting_precision_rules(
            SimpleNamespace(findings_by_operationalization=[]), additions))
        record['authority'] += '; INDEPENDENT_PRESSURE_SUPPORT_CONSTRUCTION'
        record['reviewed_by'] += '; source-bound-cell-pressure-statistics-audit'
        record['rationale'] += (' Common scalar pressure statistics were independently constructed without answer inputs; '
            'both original coefficients reproduced. Weights are absolute signed VTK cell volumes; unweighted statistics '
            'are explicitly labeled and cannot substitute for weighted quantities. Core obligations unchanged. '
            'Numerical tolerance covers reconstruction roundoff, with the existing uniform reporting policy. '
            'This is not independent human calibration.')
    write_json(audit_path, {'role':'INDEPENDENT_REFERENCE_CONSTRUCTION; NO_MODEL_ANSWER_INPUT',
        'input':{'path':str(path.relative_to(ROOT)), 'sha256':sha(path)}, 'script_sha256':sha(__file__),
        'verified_original_branches':checked, 'statistics':values, 'raw_second_moment_variance':raw_variance})
    for cid, record in records.items():
        if cid.startswith('aideas_pressure_heterogeneity_'):
            record['rationale'] += f' Evidence {audit_path.relative_to(ROOT)} SHA256 {sha(audit_path)}.'
    write_json(record_path, records)
    write_json(output, {'protocol':'evaluation-reference-package-v1', 'entries':[
        {'case_id':cid, 'task_sha256':r['task_sha256'], 'source':{'path':str(record_path.relative_to(ROOT)),
         'sha256':sha(record_path), 'pointer':'/'+cid}} for cid,r in records.items()]})
    return {'output':str(output), 'original_branches_reproduced':len(checked),
            'support_quantities_per_branch':len(support), 'api_calls':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output)))

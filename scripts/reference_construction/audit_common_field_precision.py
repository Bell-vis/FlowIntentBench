#!/usr/bin/env python3
"""Independent precision audit of frozen common field statistics.

No model answers or claimed values are inputs. Compare VTK point-to-cell
sampling with NumPy float32 and float64 arithmetic on the same rectilinear cells.
This only records sensitivity; it does not modify evaluation policies.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flowintentbench.construction_recipes import _cell_view
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from scripts.reference_construction.audit_rectilinear_precision import vertex_means
from scripts.reference_construction.build_common_field_references import moments,association_moments
from scripts.reference_construction.build_pressure_support_references import sha


def cell_field(mesh,spec,dtype):
    if spec['association']=='cell':
        a=vtk_to_numpy(mesh.GetCellData().GetArray(spec['name'])).astype(float)
    else:
        stored=vtk_to_numpy(mesh.GetPointData().GetArray(spec['name']))
        dtype=np.result_type(stored.dtype,dtype)  # Never downcast a double source.
        if stored.ndim==1:
            a=vertex_means(stored,mesh.GetDimensions(),dtype)[0].astype(float)
        else:
            a=np.column_stack([vertex_means(stored[:,i],mesh.GetDimensions(),dtype)[0]
                               for i in range(stored.shape[1])]).astype(float)
    if 'component' in spec:return a[:,spec['component']]
    if a.ndim==2:
        if spec.get('reduce')!='magnitude':raise ValueError('explicit vector reduction required')
        return np.linalg.norm(a,axis=1)
    return a


def audit(construction,output):
    snapshot=output.with_name(output.stem+'_builder.py')
    if output.exists() or snapshot.exists():raise ValueError('preserve frozen audit versions')
    source=json.loads(construction.read_text())
    if source['role']!='INDEPENDENT_REFERENCE_CONSTRUCTION; NO_ANSWER_INPUT':
        raise ValueError('independent common-field construction required')
    manifest=json.loads((ROOT/'experiments/expansion_v1_development/case_manifest.json').read_text())
    cases={x['case_id']:x for x in manifest['cases']};cache={};results=[]
    for record in source['branches']:
        _,_,_,material=load_development_case(ROOT,cases[record['case_id']])
        dataset=material['source_case']['dataset_id'];key=(dataset,record['volume_convention'])
        if key not in cache:
            mesh,_=_reader_dataset(ROOT,dataset)
            if not mesh.IsA('vtkRectilinearGrid'):
                cache[key]=None
            else:
                for f in record['input_files']:
                    if sha(ROOT/f['path'])!=f['sha256']:raise ValueError('dataset changed')
                w,_,arrays=_cell_view(mesh,record['volume_convention'])
                cache[key]=(mesh,w,arrays,{})
        if cache[key] is None:continue
        mesh,w,arrays,computed=cache[key]
        specs={k:v['spec'] for k,v in record['fields'].items()}
        spec_key=json.dumps(specs,sort_keys=True)+json.dumps(record['finite_cell_fields'],sort_keys=True)
        if spec_key not in computed:
            variants={}
            original_mask=None
            for mode,dtype in [('vtk',None),('float32',np.float32),('float64',np.float64)]:
                mask=np.ones(len(w),dtype=bool)
                for spec in record['finite_cell_fields']:
                    a=arrays[(spec['association'],spec['name'])]
                    mask &= np.isfinite(a).all(axis=1) if a.ndim==2 else np.isfinite(a)
                fields={}
                for slot,spec in specs.items():
                    if mode=='vtk':
                        a=arrays[(spec['association'],spec['name'])]
                        if 'component' in spec:a=a[:,spec['component']]
                        elif a.ndim==2:a=np.linalg.norm(a,axis=1)
                    else:a=cell_field(mesh,spec,dtype)
                    if not np.isfinite(a[mask]).all():raise ValueError('precision changed finite population')
                    fields[slot]=a[mask]
                if original_mask is not None and not np.array_equal(mask,original_mask):
                    raise ValueError('population mismatch')
                original_mask=mask
                values={slot+'_'+name:v for slot,a in fields.items() for name,v in moments(a,w[mask]).items()}
                if 'y' in fields:values.update(association_moments(fields['x'],fields['y'],w[mask]))
                variants[mode]=values
            computed[spec_key]=variants
        variants=computed[spec_key]
        expected={slot+'_'+name:v for slot,d in record['fields'].items() for name,v in d['moments'].items()}
        expected.update(record['joint_moments'])
        if any(not np.isclose(variants['vtk'][k],v,rtol=1e-12,atol=1e-12) for k,v in expected.items()):
            raise ValueError('frozen common moments not reproduced')
        results.append({'case_id':record['case_id'],'branch_id':record['branch_id'],
            'input_files':record['input_files'],'fields':specs,'variants':variants})
    output.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(output,{'role':'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT',
        'source_snapshot':str(snapshot.relative_to(ROOT)),'script_sha256':sha(snapshot),
        'source_construction':str(construction.relative_to(ROOT)),'source_construction_sha256':sha(construction),
        'dependencies':{str(Path(m.__file__).relative_to(ROOT)):sha(m.__file__) for m in
            [sys.modules['scripts.reference_construction.audit_rectilinear_precision'],
             sys.modules['scripts.reference_construction.build_common_field_references']]},
        'records':results})
    return {'branches':len(results),'api_calls':0,'output':str(output)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--construction',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.construction.resolve(),a.output.resolve())))

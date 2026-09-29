#!/usr/bin/env python3
"""Audit vector-reduction and duplicate-field sampling conventions without answers.

Produces candidate construction evidence, not automatic GT acceptance policies.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flowintentbench.deterministic_materialization import _reader_dataset
from flowintentbench.construction_recipes import _cell_view
from flowintentbench.expansion_evaluation import load_development_case
from scripts.reference_construction.build_common_field_references import coefficient,moments
from scripts.reference_construction.build_pressure_support_references import sha
from flowintentbench.external_file_evaluator import write_json


def weighted_width(x,w,centers,axis,measure):
    mass=x*w
    if (mass<0).any() or mass.sum()<=0:raise ValueError('positive width support required')
    coordinate=centers[:,axis];center=float(np.average(coordinate,weights=mass))
    if measure=='rms':width=float(np.sqrt(np.average((coordinate-center)**2,weights=mass)))
    elif measure=='interdecile':
        v=moments(coordinate[mass>0],mass[mass>0]);width=v['q90']-v['q10']
    else:raise ValueError('unsupported width')
    return {'width':width,'center':center,'total_weight':float(mass.sum())}


def average_point_values(dataset,values):
    temporary=numpy_to_vtk(values,deep=True);temporary.SetName('__independent_point_average')
    copied=dataset.NewInstance();copied.ShallowCopy(dataset);copied.GetPointData().AddArray(temporary)
    convert=vtk.vtkPointDataToCellData();convert.SetInputData(copied);convert.Update()
    return vtk_to_numpy(convert.GetOutput().GetCellData().GetArray(temporary.GetName())).astype(float)


def point_magnitude_then_average(dataset,name):
    raw=vtk_to_numpy(dataset.GetPointData().GetArray(name)).astype(float)
    if raw.ndim!=2 or raw.shape[1]!=3:raise ValueError('point vector required')
    return average_point_values(dataset,np.linalg.norm(raw,axis=1))


def build(output):
    output=output.resolve();snapshot=output.with_name(output.stem+'_builder.py')
    if output.exists() or snapshot.exists():raise ValueError('preserve audit versions')
    manifest=json.loads((ROOT/'experiments/expansion_v1_development/case_manifest.json').read_text())
    cache={};records=[]
    families={'mhd_density_magnetic_association','tubes_axial_activity_width'}
    for item in manifest['cases']:
        if not any(item['case_id'].startswith(f+'_') for f in families):continue
        ci,meta,gt,mat=load_development_case(ROOT,item)
        dataset_id=mat['source_case']['dataset_id'];definition=mat['family_definition']
        if dataset_id not in cache:
            data,dm=_reader_dataset(ROOT,dataset_id)
            files=[]
            for spec in dm['files']:
                p=ROOT/dm.get('file_root','datasets')/spec['path']
                if sha(p)!=spec['checksum']:raise ValueError('source identity mismatch')
                files.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
            w,centers,arrays=_cell_view(data,definition.get('cell_volume_convention','positive_signed_volume'))
            cache[dataset_id]=(data,w,centers,arrays,files,{})
        data,w,centers,arrays,files,derived=cache[dataset_id]
        for branch in gt.findings_by_operationalization:
            recipe=mat['branch_execution_evidence'][branch.operationalization_id]['execution_provenance']['recipe']
            vector=recipe.get('other_field',recipe['field']);name=vector['name']
            if vector.get('reduce')!='magnitude':raise ValueError('expected vector magnitude experiment')
            if name not in derived:
                derived[name]={'average_vector_then_magnitude':np.linalg.norm(arrays[('point',name)],axis=1),
                               'magnitude_then_average':point_magnitude_then_average(data,name)}
                if ('cell',name) in arrays:derived[name]['stored_cell_vector_magnitude']=np.linalg.norm(arrays[('cell',name)],axis=1)
            variants={}
            for convention,x in derived[name].items():
                if recipe['kind']=='association':
                    scalar=arrays[(recipe['field']['association'],recipe['field']['name'])]
                    result={'coefficient':coefficient(scalar,w,recipe['measure'],x),'vector_magnitude_moments':moments(x,w),
                        'scalar_moments':moments(scalar,w),'raw_covariance':float(np.average(
                            (scalar-np.average(scalar,weights=w))*(x-np.average(x,weights=w)),weights=w)),
                        'cell_count':len(w),'included_volume':float(w.sum())}
                elif recipe['kind']=='weighted_width':result=weighted_width(x,w,centers,recipe['axis'],recipe['measure'])
                else:raise ValueError('unsupported recipe')
                variants[convention]=result
            if recipe['kind']=='weighted_width':
                if '__vertex_centers' not in derived:
                    derived['__vertex_centers']=average_point_values(data,vtk_to_numpy(data.GetPoints().GetData()).astype(float))
                for convention,x in derived[name].items():
                    variants[convention+'_vertex_coordinate_means']=weighted_width(x,w,derived['__vertex_centers'],recipe['axis'],recipe['measure'])
            original=('average_vector_then_magnitude' if vector['association']=='point' else 'stored_cell_vector_magnitude')
            for f in branch.findings:
                key='coefficient' if recipe['kind']=='association' else 'center' if 'center' in f.finding_id else 'width'
                if not np.isclose(f.value,variants[original][key],rtol=1e-8,atol=1e-10):raise ValueError('original GT not reproduced')
            records.append({'case_id':item['case_id'],'branch_id':branch.operationalization_id,'question':ci.scientific_question,
                'recipe':recipe,'input_files':files,'original_convention':original,'variants':variants})
    snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(output,{'role':'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWER_INPUT','status':'CANDIDATE_SAMPLING_AUDIT_NOT_GT',
        'script_sha256':sha(__file__),'source_snapshot':str(snapshot.relative_to(ROOT)),'vtk_version':vtk.vtkVersion.GetVTKVersion(),
        'original_branches_reproduced':len(records),'records':records,
        'interpretation':'Alternative values are not automatically accepted by the public task. Resolve whether sampling order/association is specified before authoring reference branches. Never infer an answer method from numerical proximity.'})
    return {'original_branches_reproduced':len(records),'output':str(output),'api_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.output)))

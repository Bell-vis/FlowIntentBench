#!/usr/bin/env python3
"""Construct high-shear references with explicit discretization, outside scoring.

No answers, model names or claimed values are inputs. Reproduce original GT
before freezing geometric-cell alternatives for conditions with open methods.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.reference_packages import load_supplements, reporting_precision_rules
from flowintentbench.external_file_evaluator import write_json
from scripts.reference_construction.build_pressure_support_references import sha


def tetra_geometry(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (4, 3) or not np.isfinite(points).all():
        raise ValueError('finite tetrahedral vertices required')
    v = points[:, 1:] - points[:, :1]
    volume = np.abs(np.einsum('ij,ij->i', np.cross(v[:, 0], v[:, 1]), v[:, 2])) / 6
    return volume, points.mean(axis=1)


def aggregate_geometry(volume, centers, parents, n):
    weights = np.bincount(parents, weights=volume, minlength=n)
    if len(weights) != n or (weights <= 0).any():
        raise ValueError('every original cell needs positive geometric volume')
    center = np.column_stack([np.bincount(parents, weights=volume*centers[:, j], minlength=n)/weights
                              for j in range(3)])
    return weights, center


def upper_tail(field, weights, centers, quantile):
    field, weights, centers = map(np.asarray, (field, weights, centers))
    if (field.ndim != 1 or field.shape != weights.shape or centers.shape != (len(field), 3)
            or not all(np.isfinite(x).all() for x in (field, weights, centers))
            or (weights <= 0).any() or not 0 < quantile < 1):
        raise ValueError('finite positive-weight population and quantile required')
    order = np.argsort(field, kind='stable')
    threshold = field[order[np.searchsorted(np.cumsum(weights[order]), quantile*weights.sum(), side='left')]]
    selected = field >= threshold
    return {'centroid':np.average(centers[selected], weights=weights[selected], axis=0).tolist(),
            'mean':float(np.average(field[selected], weights=weights[selected])), 'threshold':float(threshold),
            'selected_volume':float(weights[selected].sum()), 'total_volume':float(weights.sum()),
            'selected_fraction':float(weights[selected].sum()/weights.sum()), 'selected_count':int(selected.sum()),
            'domain_mean':float(np.average(field,weights=weights)), 'maximum':float(field.max())}


def populations(source):
    reader=vtk.vtkXMLUnstructuredGridReader();reader.SetFileName(str(source));reader.Update()
    mesh=reader.GetOutput();n=mesh.GetNumberOfCells()
    field='NormShearRate [1/s]'
    pc=vtk.vtkPointDataToCellData();pc.SetInputData(mesh);pc.Update()
    cell_field=vtk_to_numpy(pc.GetOutput().GetCellData().GetArray(field)).astype(float)
    size=vtk.vtkCellSizeFilter();size.SetInputData(mesh);size.Update()
    weights=np.abs(vtk_to_numpy(size.GetOutput().GetCellData().GetArray('Volume')).astype(float))
    cc=vtk.vtkCellCenters();cc.SetInputData(mesh);cc.Update()
    centers=vtk_to_numpy(cc.GetOutput().GetPoints().GetData()).astype(float)
    ids=numpy_to_vtk(np.arange(n,dtype=np.int64),deep=True);ids.SetName('reference_construction_parent_cell')
    copied=vtk.vtkUnstructuredGrid();copied.ShallowCopy(mesh);copied.GetCellData().AddArray(ids)
    triangles=vtk.vtkDataSetTriangleFilter();triangles.SetInputData(copied);triangles.Update();tet=triangles.GetOutput()
    if any(tet.GetCellType(i)!=vtk.VTK_TETRA for i in range(tet.GetNumberOfCells())):
        raise ValueError('decomposition produced non-tetrahedral cells')
    conn=vtk_to_numpy(tet.GetCells().GetConnectivityArray()).reshape(-1,4)
    points=vtk_to_numpy(tet.GetPoints().GetData()).astype(float)[conn]
    volume, centroids=tetra_geometry(points)
    parents=vtk_to_numpy(tet.GetCellData().GetArray('reference_construction_parent_cell')).astype(int)
    geom_weights, geom_centers=aggregate_geometry(volume,centroids,parents,n)
    tetra_field=vtk_to_numpy(tet.GetPointData().GetArray(field)).astype(float)[conn].mean(axis=1)
    positive=volume>0
    # A second, fully specified standard decomposition is necessary because
    # nonplanar quadrilateral faces make tetrahedralization non-unique.
    offsets=vtk_to_numpy(mesh.GetCells().GetOffsetsArray())
    cells=vtk_to_numpy(mesh.GetCells().GetConnectivityArray())
    xyz=vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    nodal=vtk_to_numpy(mesh.GetPointData().GetArray(field)).astype(float)
    types=np.array([mesh.GetCellType(i) for i in range(n)])
    patterns={vtk.VTK_TETRA:[[0,1,2,3]],vtk.VTK_PYRAMID:[[0,1,2,4],[0,2,3,4]],
        vtk.VTK_WEDGE:[[0,1,2,3],[1,2,3,4],[2,3,4,5]],
        vtk.VTK_HEXAHEDRON:[[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6],[0,5,1,6]]}
    if set(types)-set(patterns):raise ValueError('unsupported mixed-cell geometry')
    explicit_volume=np.zeros(n);explicit_center=np.zeros((n,3));precise_field=np.zeros(n)
    for kind,pattern in patterns.items():
        indexes=np.flatnonzero(types==kind)
        width=int(offsets[indexes[0]+1]-offsets[indexes[0]])
        conn=cells[offsets[indexes,None]+np.arange(width)]
        vertices=xyz[conn[:,np.array(pattern)]]
        vols,ct=tetra_geometry(vertices.reshape(-1,4,3))
        vols=vols.reshape(len(indexes),-1);ct=ct.reshape(len(indexes),-1,3)
        explicit_volume[indexes]=vols.sum(axis=1)
        explicit_center[indexes]=np.sum(vols[:,:,None]*ct,axis=1)/vols.sum(axis=1)[:,None]
        precise_field[indexes]=nodal[conn].mean(axis=1)
    return {'vtk_cells':(cell_field,weights,centers), 'geometric_cells':(cell_field,geom_weights,geom_centers),
            'fixed_diagonal_cells':(precise_field,explicit_volume,explicit_center),
            'linear_tetrahedra':(tetra_field[positive],volume[positive],centroids[positive])}, {
            'original_cells':n,'original_points':mesh.GetNumberOfPoints(),'tetrahedra':int(positive.sum()),
            'vtk_version':vtk.vtkVersion.GetVTKVersion(),'input_sha256':sha(source)}


def build(base,output):
    output=output.resolve();rp=output.with_name(output.stem+'_records.json');ap=output.with_name(output.stem+'_construction.json')
    if any(p.exists() for p in (output,rp,ap)):raise ValueError('preserve frozen reference versions')
    dm=json.loads((ROOT/'datasets/AIDEAS_Blow_Mold/dataset_manifest.json').read_text())
    source=ROOT/'datasets'/dm['files'][0]['path']
    if sha(source)!=dm['files'][0]['checksum']:raise ValueError('dataset hash mismatch')
    pops, provenance=populations(source)
    values={name:{str(q):upper_tail(*args,q) for q in (.9,.95)} for name,args in pops.items()}
    records={cid:deepcopy(s['record']) for cid,s in load_supplements(base,ROOT).items()}
    manifest=json.loads((ROOT/'experiments/expansion_v1_development/case_manifest.json').read_text())
    reproduced=[];new=[]
    for item in manifest['cases']:
        if not item['case_id'].startswith('aideas_high_shear_region_'):continue
        ci,meta,gt,material=load_development_case(ROOT,item)
        record=records[item['case_id']]
        bundles={b.operationalization_id:b for b in gt.acceptable_operationalizations}
        for b in gt.findings_by_operationalization:
            q=.95 if b.operationalization_id.endswith('_q95') else .9
            for f in b.findings:
                key='centroid' if f.category.value=='location' else 'mean'
                if not np.allclose(f.value,values['vtk_cells'][str(q)][key],rtol=1e-8,atol=1e-8):
                    raise ValueError('original reference not reproduced: '+f.finding_id)
            reproduced.append(b.operationalization_id)
            # Only fully open O3 permits replacing field sampling and weights.
            if {d.value for d in meta.unresolved_operationalization_dimensions}!={d.value for d in meta.principal_operationalization_dimensions}:continue
            for method in ('geometric_cells','fixed_diagonal_cells','linear_tetrahedra'):
                # An existing independently frozen tetrahedral q90 branch is
                # retained; this construction audits rather than duplicates it.
                if method=='linear_tetrahedra' and q==.9:continue
                bid=b.operationalization_id+'_'+method
                bundle=deepcopy(bundles[b.operationalization_id].model_dump(mode='json'));bundle['operationalization_id']=bid
                for d in bundle['decisions']:
                    if d['dimension']=='feature_definition':
                        d['statement']=('Use the original mesh cells with arithmetic means of all original vertices for the point field.'
                            if method in {'geometric_cells','fixed_diagonal_cells'} else 'Decompose the supplied mesh into VTK tetrahedra; use the arithmetic mean of each tetrahedron vertex field as its scalar value.')
                    elif d['dimension']=='aggregation_or_representation':
                        d['statement']=('For every original cell, sum absolute VTK-decomposed tetrahedral volumes and their first moments to obtain its geometric volume and centroid; volume-weight original cells.'
                            if method=='geometric_cells' else 'Use exact absolute geometric tetrahedral volumes and arithmetic tetrahedral vertex centroids; volume-weight the tetrahedra.')
                        if method=='fixed_diagonal_cells':
                            d['statement']='Compute each original cell geometric volume and centroid using absolute tetrahedral moments: hex 6 tetrahedra around diagonal (0,6), pyramid base diagonal (0,2), wedge tetrahedra (0,1,2,3),(1,2,3,4),(2,3,4,5), using VTK local vertex numbering; volume-weight original-cell values, averaging vertex scalars in float64.'
                fs=[];policies=[];mapping={}
                for f in b.findings:
                    key='centroid' if f.category.value=='location' else 'mean';rid=bid+'_'+key
                    ref=f.model_copy(update={'finding_id':rid,'value':values[method][str(q)][key]}).model_dump(mode='json')
                    fs.append(ref);mapping[rid]=f.finding_id
                    policy=next(x for x in material['finding_verification_policy']['policies'] if x['finding_id']==f.finding_id)
                    policies.append({**{k:policy[k] for k in ('verification_mode','verification_parameters')},'operationalization_id':bid,'finding_id':rid})
                record.setdefault('new_branches',[]).append({'operationalization':bundle,'findings':{'operationalization_id':bid,'findings':fs},
                    'requirement_branch_id':b.operationalization_id,'requirement_map':mapping,'policies':policies})
                record.setdefault('reporting_policies',[]).extend(reporting_precision_rules(SimpleNamespace(findings_by_operationalization=[]),
                    [{'branch_id':bid,'finding':f} for f in fs]))
                new.append(bid)
        record['authority']+='; GEOMETRIC_CELL_REFERENCE_CONSTRUCTION'
        record['reviewed_by']+='; tetrahedral-first-moment-audit'
        record['rationale']+=' Original high-shear references reproduced before freezing geometric-cell alternatives for fully open O3 only. No model answers or claimed values are construction inputs; original core obligations and tolerances retained. VTK decomposition conventions explicit; this is not human calibration.'
    # Validate the previously frozen q90 tetrahedral reference too.
    candidate=next(r for cid,r in records.items() if cid=='aideas_high_shear_region_o3_f1')
    old_tetra=[b for b in candidate.get('new_branches',[]) if 'tetra' in b['operationalization']['operationalization_id'] and b['operationalization']['operationalization_id'] not in new]
    for branch in old_tetra:
        for f in branch['findings']['findings']:
            if f['importance']=='core':
                key='centroid' if f['category']=='location' else 'mean'
                if not np.allclose(f['value'],values['linear_tetrahedra']['0.9'][key],rtol=1e-8,atol=1e-8):raise ValueError('prior tetrahedral reference mismatch')
    write_json(ap,{'role':'INDEPENDENT_REFERENCE_CONSTRUCTION_NO_ANSWERS','script_sha256':sha(__file__),
        'input':provenance,'statistics':values,'original_branches_reproduced':reproduced,'new_branches':new})
    for cid,r in records.items():
        if cid.startswith('aideas_high_shear_region_'):r['rationale']+=f' Evidence {ap.relative_to(ROOT)} SHA256 {sha(ap)}.'
    write_json(rp,records)
    write_json(output,{'protocol':'evaluation-reference-package-v1','entries':[{'case_id':cid,'task_sha256':r['task_sha256'],
        'source':{'path':str(rp.relative_to(ROOT)),'sha256':sha(rp),'pointer':'/'+cid}} for cid,r in records.items()]})
    return {'original_branches_reproduced':len(reproduced),'new_branches':new,'output':str(output),'api_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.base,a.output)))

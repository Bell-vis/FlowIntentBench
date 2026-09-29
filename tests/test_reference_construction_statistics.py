import numpy as np
import pytest

from scripts.reference_construction.build_pressure_support_references import statistics
from scripts.reference_construction.build_pressure_support_references import spatial_statistics
from scripts.reference_construction.build_common_field_references import moments, coefficient


def test_pressure_population_weights_and_dispersion_are_distinct():
    r = statistics([1, 3, 5], [-1, 2, 1])
    assert r['weighted_mean'] == 3
    assert r['weighted_variance'] == 2
    assert r['weighted_std'] == pytest.approx(np.sqrt(2))
    assert r['cv'] == pytest.approx(np.sqrt(2)/3)
    assert r['unweighted_std'] == pytest.approx(np.sqrt(8/3))
    assert r['weighted_mad_ratio'] == pytest.approx(1/3)
    assert r['absolute_volume'] == 4
    assert r['negative_volume_cells'] == 1
    assert [r['weighted_q10'], r['weighted_median'], r['weighted_q90']] == [1, 3, 5]


@pytest.mark.parametrize('pressure,volumes', [([1], [0]), ([np.nan], [1]), ([1,2], [1]), ([[1]], [[1]])])
def test_invalid_populations_are_not_silently_filtered(pressure, volumes):
    with pytest.raises(ValueError):
        statistics(pressure, volumes)


def test_spatial_reference_fit_and_total_variance_have_independent_known_solution():
    z=(np.arange(120)+.5)/4
    xyz=np.column_stack([np.sin(z),np.cos(z),z])
    signed=np.where(np.arange(120)%2,1.,-1.)
    values,_=spatial_statistics(10+2*z,signed,xyz,(-1,1,-1,1,0,30),np.tile([10,12,13,14],30))
    assert values['z_fit_slope']==pytest.approx(2)
    assert values['z_fit_intercept']==pytest.approx(10)
    assert values['z_fit_r_squared']==pytest.approx(1)
    assert values['correlation_z']==pytest.approx(1)
    assert values['slabs15_first_mean']==pytest.approx(12)
    assert values['slabs15_last_mean']==pytest.approx(68)
    for n in (15,30):
        assert values[f'slabs{n}_between_fraction']+values[f'slabs{n}_within_fraction']==pytest.approx(1)
    assert sum(values[n+'_variance_fraction'] for n in ['tetrahedral','hexahedral','wedge','pyramidal'])==pytest.approx(1)


def test_common_field_moments_and_association_are_weighted_and_rank_aware():
    x=np.array([1.,2.,4.]);w=np.array([1.,2.,1.])
    m=moments(x,w)
    assert m['mean']==2.25
    assert m['std']==pytest.approx(np.sqrt(1.1875))
    assert m['variance']==pytest.approx(1.1875)
    assert m['median']==2
    assert coefficient(x,w,'cv')==pytest.approx(np.sqrt(1.1875)/2.25)
    assert coefficient(x,w,'relative_interdecile')==1.5
    assert coefficient(x,w,'spearman',np.array([9.,3.,1.]))==pytest.approx(-1)


def test_joint_raw_field_moments_have_known_unequal_weight_solution():
    from scripts.reference_construction.build_common_field_references import association_moments
    x=np.array([1.,2.,4.]);w=np.array([1.,2.,1.])
    values=association_moments(x,3*x-5,w)
    assert values==pytest.approx({'covariance':3*1.1875,'regression_slope':3.,
        'regression_intercept':-5.,'pearson_squared':1.})
    with pytest.raises(ValueError,match='nonconstant'):
        association_moments(x,np.ones(3),w)
    assert coefficient(x,w,'pearson',np.array([9.,3.,1.]))!=-1


def test_scalar_association_reference_preserves_distinct_populations_and_reproduces_gt():
    from scripts.reference_construction.build_scalar_association_references import scalar_variants
    arrays = {('cell', 'k'): np.array([1., 3.]), ('point', 'k'): np.array([2., 2.])}
    variants = scalar_variants(arrays, np.ones(2), 'k', 'cv', 'cell', .5)
    assert variants['cell']['coefficient'] == .5
    assert variants['point']['coefficient'] == 0
    with pytest.raises(ValueError, match='not reproduced'):
        scalar_variants(arrays, np.ones(2), 'k', 'cv', 'cell', .4)
    arrays[('point', 'k')][0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        scalar_variants(arrays, np.ones(2), 'k', 'cv', 'cell', .5)


def test_rectilinear_vertex_means_preserve_vtk_order_and_finite_vertex_selection():
    from scripts.reference_construction.audit_rectilinear_precision import vertex_means
    # Two cells along x; field x + 10*y + 100*z, x-fastest storage.
    x = np.array([i + 10*j + 100*k for k in range(2) for j in range(2) for i in range(3)], dtype=np.float32)
    values, finite = vertex_means(x, (3, 2, 2), np.float64)
    assert values.tolist() == [55.5, 56.5]
    assert finite.tolist() == [True, True]
    x[0] = np.nan
    _, finite = vertex_means(x, (3, 2, 2), np.float64)
    assert finite.tolist() == [False, True]


def test_precision_tolerance_only_covers_independently_measured_numerical_spread():
    from scripts.reference_construction.freeze_precision_tolerances import fraction_tolerance
    tolerance = fraction_tolerance(.1, 1e-9, [.1, .1000006])
    assert tolerance == pytest.approx(6.01e-7)
    assert abs(.1000006 - .1) <= tolerance
    assert abs(.101 - .1) > tolerance
    for values in ([float('nan')], [1.1], []):
        with pytest.raises(ValueError):
            fraction_tolerance(.1, 1e-9, values)


def test_common_field_precision_preserves_vector_order_and_double_source():
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    from scripts.reference_construction.audit_common_field_precision import cell_field
    mesh=vtk.vtkRectilinearGrid();mesh.SetDimensions(2,2,2)
    for setter in (mesh.SetXCoordinates,mesh.SetYCoordinates,mesh.SetZCoordinates):
        setter(numpy_to_vtk(np.array([0.,1.]),deep=True))
    vectors=np.array([[1.,0.,0.],[-1.,0.,0.]]*4,dtype=np.float32)
    field=numpy_to_vtk(vectors,deep=True);field.SetName('v');mesh.GetPointData().AddArray(field)
    assert cell_field(mesh,{'association':'point','name':'v','reduce':'magnitude'},np.float64).tolist()==[0.]
    # Casting this double source to single would lose the increment entirely.
    doubles=np.full(8,1.+2**-40)
    field=numpy_to_vtk(doubles,deep=True);field.SetName('d');mesh.GetPointData().AddArray(field)
    assert cell_field(mesh,{'association':'point','name':'d'},np.float32).tolist()==[1.+2**-40]


def test_common_moment_precision_accepts_measured_spread_but_rejects_large_error():
    from scripts.reference_construction.freeze_common_field_precision import precision_tolerance
    reference=-.01394329050162978
    tol=precision_tolerance(reference,1e-10,[reference,-.013943291306634303,-.013943289581296089])
    assert abs(-.01394328958-reference)<=tol
    assert abs(-.014-reference)>tol
    for baseline,values in [(-1,[0]),(0,[]),(0,[np.nan])]:
        with pytest.raises(ValueError):precision_tolerance(0,baseline,values)


def test_conditional_bins_conserve_weight_and_include_the_last_endpoint():
    from scripts.reference_construction.build_conditional_field_references import conditional_statistics
    r=conditional_statistics([0,1,2,3],[1,3,5,7],[1,3,2,2],'equal_width',2)
    assert r['y_mean']==pytest.approx([2.5,6])
    assert r['volume_fraction']==pytest.approx([.5,.5])
    assert r['x_lower']==[0,1.5] and r['x_upper']==[1.5,3]
    # The boundary at x=1.5 belongs to the second equal-width bin.
    r=conditional_statistics([0,1.5,3],[1,9,3],[1,1,1],'equal_width',2)
    assert r['y_mean']==pytest.approx([1,6])


def test_equal_volume_groups_split_ties_without_mesh_order_bias():
    from scripts.reference_construction.build_conditional_field_references import conditional_statistics
    x=np.array([0,0,1,2]);y=np.array([2,6,10,20]);w=np.array([1,3,2,2])
    r=conditional_statistics(x,y,w,'equal_volume',4)
    assert r['y_mean']==pytest.approx([5,5,10,20])
    assert r['volume_fraction']==pytest.approx([.25]*4)
    assert r['x_lower']==[0,0,1,2] and r['x_upper']==[0,0,1,2]
    p=[1,0,3,2]
    assert conditional_statistics(x[p],y[p],w[p],'equal_volume',4)==r
    with pytest.raises(ValueError,match='tied quantile'):
        conditional_statistics(x,y,w,'weighted_quantile',4)


@pytest.mark.parametrize('x,y,w,scheme,count',[
    ([0,1],[1,2],[0,1],'equal_width',2),([0,1],[1,float('nan')],[1,1],'equal_width',2),
    ([0,0],[1,2],[1,1],'equal_width',2),([0,1],[1],[1,1],'equal_width',2),
    ([0,1],[1,2],[1,1],'equal_width',3),([0,1],[1,2],[1,1],'invented',2)])
def test_conditional_reference_rejects_undefined_partitions(x,y,w,scheme,count):
    from scripts.reference_construction.build_conditional_field_references import conditional_statistics
    with pytest.raises(ValueError):conditional_statistics(x,y,w,scheme,count)


def test_independent_flux_distinguishes_product_orders_and_uses_volume_weights():
    from scripts.reference_construction.audit_advective_flux import rectilinear_flux
    # Two cells with widths 1 and 2, rho=x and vz=x at their vertices.
    x = np.array([0., 1., 3.])
    values = np.tile(x, 4)
    result = rectilinear_flux(values, values, (x, [0., 1.], [0., 1.]))
    assert result['included_volume'] == 3
    assert result['cell_then_product'] == pytest.approx((.25 + 2*4)/3)
    assert result['point_then_cell'] == pytest.approx((.5 + 2*5)/3)
    assert result['point_count'] == 12 and result['cell_count'] == 2
    with pytest.raises(ValueError, match='strictly increasing'):
        rectilinear_flux(values, values, ([0., 1., 1.], [0., 1.], [0., 1.]))


def test_tetrahedral_geometry_aggregates_first_moments_and_retains_quantile_ties():
    from scripts.reference_construction.build_high_shear_references import tetra_geometry, aggregate_geometry, upper_tail
    tet=np.array([[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
                  [[2,0,0],[4,0,0],[2,2,0],[2,0,2]]],dtype=float)
    v,c=tetra_geometry(tet)
    assert v==pytest.approx([1/6,8/6])
    w,g=aggregate_geometry(v,c,np.array([0,0]),1)
    assert w==pytest.approx([1.5])
    assert g[0]==pytest.approx(np.average(c,weights=[1,8],axis=0))
    # Vertex orientation must not change the physical moment.
    flipped=tet.copy();flipped[:,[1,2]]=flipped[:,[2,1]]
    vf,cf=tetra_geometry(flipped)
    assert vf==pytest.approx(v);assert cf==pytest.approx(c)
    r=upper_tail(np.array([1.,2.,2.]),np.array([1.,1.,2.]),np.zeros((3,3)),.5)
    assert r['threshold']==2 and r['selected_count']==2 and r['selected_fraction']==.75


def test_sampling_audit_separates_nonlinear_reduction_from_averaging():
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    from scripts.reference_construction.audit_field_sampling import point_magnitude_then_average, weighted_width
    points=vtk.vtkPoints()
    for xyz in ((0,0,0),(1,0,0),(0,1,0),(0,0,1)):points.InsertNextPoint(*xyz)
    mesh=vtk.vtkUnstructuredGrid();mesh.SetPoints(points)
    cell=vtk.vtkTetra()
    for i in range(4):cell.GetPointIds().SetId(i,i)
    mesh.InsertNextCell(cell.GetCellType(),cell.GetPointIds())
    vectors=np.array([[1.,0,0],[0,1.,0],[-1.,0,0],[0,-1.,0]])
    field=numpy_to_vtk(vectors,deep=True);field.SetName('v');mesh.GetPointData().AddArray(field)
    assert point_magnitude_then_average(mesh,'v')==pytest.approx([1])
    assert np.linalg.norm(vectors.mean(axis=0))==0
    r=weighted_width(np.ones(2),np.array([1.,3.]),np.array([[0,0,0],[2,0,0]]),0,'rms')
    assert r['center']==1.5 and r['width']==pytest.approx(np.sqrt(.75))


def test_region_reference_uses_face_adjacency_positive_population_and_physical_coordinates():
    from scripts.reference_construction.build_plot3d_region_support import region_statistics
    # Three disconnected peaks; index-fastest ordering and nonlinear physical
    # coordinates distinguish grid indices from the reported physical centroid.
    speed=np.array([9.,8.,0.,1.,7.,0.,2.,6.])
    xyz=np.array([[i*i,2*i,0.] for i in range(8)])
    r=region_statistics(speed,xyz,(8,1,1),np.ones(8,dtype=bool),.5,'peak','mean')
    assert r['threshold']==6.5
    assert r['nonzero_points']==6 and r['zero_speed_points']==2
    assert r['retained_points']==3 and r['region_count']==2
    assert r['selected_region_size']==2 and r['location']==[.5,1.,0.]
    assert r['other_peak_max']==r['other_peak_min']==7
    assert r['coordinate_max']==[1,2,0] and r['index_max']==[1,0,0]
    speed[4]=9
    with pytest.raises(ValueError,match='ambiguous winning region'):
        region_statistics(speed,xyz,(8,1,1),np.ones(8,dtype=bool),.5,'peak','mean')


def test_population_variant_labels_preserve_statistic_and_do_not_copy_old_numbers():
    from scripts.reference_construction.build_plot3d_population_references import variant_statement
    assert 'Mean spatial location' in variant_statement('location','peak','mean')
    assert 'Peak-speed point location' in variant_statement('location','peak','peak')
    assert 'Mean speed' in variant_statement('strength','mean','mean')
    assert 'Peak speed' in variant_statement('strength','peak','mean')
    assert not any(c.isdigit() for c in variant_statement('location','peak','mean'))
    with pytest.raises(ValueError):variant_statement('unsupported','peak','mean')


def test_population_alias_dedup_preserves_required_result_and_core_tolerance():
    from scripts.reference_construction.build_plot3d_population_references import deduplicate_variant_references
    core={'finding_id':'core','statement':'Mean speed','value':2.0,'unit':None,'importance':'core'}
    support={**core,'finding_id':'support','importance':'supporting'}
    record={'new_branches':[{'operationalization':{'operationalization_id':'o_all_finite_stored'},
        'findings':{'findings':[support,core]},'requirement_map':{'core':'old_core'},
        'policies':[{'finding_id':'support','tol':1e-9},{'finding_id':'core','tol':.01}]}],
        'reporting_policies':[{'finding_id':'support'},{'finding_id':'core'}]}
    removed=deduplicate_variant_references(record)
    b=record['new_branches'][0]
    assert removed[0]['retained_id']=='core'
    assert b['findings']['findings']==[core]
    assert b['policies']==[{'finding_id':'core','tol':.01}]
    assert b['requirement_map']=={'core':'old_core'}
    assert record['reporting_policies']==[{'finding_id':'core'}]
    b['findings']['findings'].append({**support,'value':3.0})
    with pytest.raises(ValueError,match='conflicting truth'):deduplicate_variant_references(record)

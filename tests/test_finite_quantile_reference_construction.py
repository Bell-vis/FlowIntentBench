import numpy as np
import pytest
from copy import deepcopy

from scripts.reference_construction.build_plot3d_population_references import materialize
from scripts.reference_construction.build_finite_quantile_references import finite_criterion
from scripts.reference_construction.export_finite_quantile_support import support_entries
from scripts.reference_construction.build_component_comparison_support import component_statistics


def test_zero_inclusive_quantile_changes_selection_without_changing_peak_rule():
    speed = np.array([0., 0., 1., 2., 3., 4., 5., 6.])
    xyz = np.column_stack((np.arange(8), np.zeros(8), np.zeros(8)))
    args = (speed, xyz, (8, 1, 1), np.ones(8, dtype=bool), .5, 'peak', 'mean')
    positive = materialize(*args)
    finite = materialize(*args, quantile_population='finite')
    assert positive['threshold'] == 3.5
    assert finite['threshold'] == 2.5
    assert positive['selected_region_size'] == 3
    assert finite['selected_region_size'] == 4
    assert positive['location'] == [6., 0., 0.]
    assert finite['location'] == [5.5, 0., 0.]
    assert positive['strength'] == finite['strength'] == 6
    assert positive['nonzero_points'] == finite['nonzero_points'] == 6
    with pytest.raises(ValueError, match='unsupported quantile population'):
        materialize(*args, quantile_population='invented')


def test_finite_population_still_excludes_nan_and_respects_validity_mask():
    speed = np.array([0., 1., 2., 3., float('nan'), 999.])
    xyz = np.column_stack((np.arange(6), np.zeros(6), np.zeros(6)))
    validity = np.array([True, True, True, True, True, False])
    result = materialize(speed, xyz, (6, 1, 1), validity, .5, 'peak', 'mean',
                         quantile_population='finite')
    assert result['threshold'] == 1.5
    assert result['selected_region_size'] == 2
    assert result['strength'] == 3


def test_criterion_change_preserves_cutoff_and_comparator():
    old = 'Retain speed >= the 90th percentile of non-zero recorded speed values.'
    assert finite_criterion(old) == (
        'Retain speed >= the 90th percentile of all finite recorded speed values, including zeros.')
    with pytest.raises(ValueError, match='unrecognized'):
        finite_criterion('Retain speed >= 1.')


def test_frozen_support_export_preserves_tolerance_and_does_not_mutate_core():
    record = {'new_branches': [{'operationalization': {'operationalization_id': 'finite'},
        'requirement_branch_id': 'positive', 'findings': {'findings': [{'value': 5}]}}],
        'branch_interpretations': {'finite': {'description': 'On all finite stored points; zeros included'}},
        'supporting_findings': [{'branch_id': 'positive', 'finding': {
            'finding_id': 'positive_support_threshold', 'value': 3.5, 'importance': 'supporting',
            'statement': 'Selection threshold', 'verification': {'absolute_tolerance': .001}},
            'policy': {'operationalization_id': 'positive', 'finding_id': 'positive_support_threshold',
                'verification_mode': 'scalar_tolerance', 'verification_parameters': {'absolute_tolerance': .001}}}]}
    original = deepcopy(record)
    entries = support_entries(record, {'finite': {'threshold': 2.5}})
    assert record == original
    assert entries[0]['finding']['value'] == 2.5
    assert entries[0]['finding']['verification'] == {'absolute_tolerance': .001}
    assert entries[0]['policy']['verification_parameters'] == {'absolute_tolerance': .001}
    assert entries[0]['branch_id'] == 'finite'
    assert entries[0]['finding']['importance'] == 'supporting'
    assert entries[0]['finding']['statement'] == 'On all finite stored points: Selection threshold'
    with pytest.raises(ValueError, match='lacks supporting field'):
        support_entries(record, {'finite': {}})


def test_component_comparison_respects_separation_and_rms_definition():
    speed = np.array([5., 4., 0., 3., 2.])
    xyz = np.column_stack((np.arange(5), np.zeros(5), np.zeros(5)))
    base, stats = component_statistics(speed, xyz, (5, 1, 1), np.ones(5, dtype=bool), .4, 'peak', 'mean')
    assert base['region_count'] == 2
    assert stats['rms_speed'] == pytest.approx(np.sqrt(20.5))
    assert stats['other_region_size'] == 1
    assert stats['other_mean_location'] == [3., 0., 0.]
    assert stats['exceeds_other_peak_and_mean'] is True
    _, stats = component_statistics(np.array([5., 0., 4., 0., 3.]), xyz, (5, 1, 1),
        np.ones(5, dtype=bool), .3, 'peak', 'mean')
    assert set(stats) == {'rms_speed'}  # Three regions: 'the other' is ambiguous.


def test_peak_winner_is_not_automatically_mean_winner():
    speed = np.array([10., 1., 1., 0., 5., 5.])
    xyz = np.column_stack((np.arange(6), np.zeros(6), np.zeros(6)))
    _, stats = component_statistics(speed, xyz, (6, 1, 1), np.ones(6, dtype=bool), .1, 'peak', 'mean')
    assert stats['exceeds_other_peak_and_mean'] is False
    assert stats['other_mean_speed'] == 5

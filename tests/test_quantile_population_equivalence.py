"""A finite-population quantile must not inherit nonzero-only numeric truth."""
from types import SimpleNamespace

import pytest

from flowintentbench.answer_evidence import operationalization_match_uncertainty
from test_rubric_evaluation import example, evaluate


REFERENCE = 'Use the 90th percentile of non-zero recorded speed values.'


@pytest.mark.parametrize('quote,guarded', [
    ('Use the upper empirical decile of finite speed values.', True),
    ('Use the 90th percentile of all finite values.', True),
    ('Use the 90th percentile of finite samples, excluding zeros.', False),
    ('Use the 90th percentile of positive speed values.', False),
    ('Use the 90th percentile of finite values with speed > 0.', False),
    ('Use the mean of all finite values.', False),
])
def test_population_guard_requires_explicit_quantile_and_population(quote, guarded):
    assert bool(operationalization_match_uncertainty(quote, REFERENCE)) is guarded
    assert operationalization_match_uncertainty(quote, 'Use all finite speed values.') is None


@pytest.mark.parametrize('claimed', [.5, .9])
def test_open_finite_quantile_cannot_inherit_consistency_or_error_from_nonzero_gt(claimed):
    answer, meta, gt, material, review = example()
    quote = 'Use the upper empirical decile of finite speed values.'
    answer = answer.replace('0.5', str(claimed)) + ' ' + quote
    gt.acceptable_operationalizations[0].decisions = [SimpleNamespace(
        dimension=SimpleNamespace(value='property_measure'), statement=REFERENCE)]
    review['result_groups'][0]['dimensions'][1]['evidence_text'] = quote
    review['findings'][0].update(value=claimed, evidence_text=f'CV is {claimed}.')
    result = evaluate((answer, meta, gt, material, review))
    assert result['metrics']['o_score']['value'] == 1
    # F can still identify a match against the independent reference catalog;
    # it does not establish that this was the answer's actual method (C).
    assert result['metrics']['finding_precision']['value'] == (1 if claimed == .5 else None)
    assert result['metrics']['c_score']['value'] is None
    assert result['error_diagnostics']['independent_contradicted_finding_ids'] == []
    assert 'FINITE_QUANTILE_POPULATION_NOT_ESTABLISHED_AS_NONZERO' in str(result)


def test_publicly_fixed_nonzero_population_is_unresolved_not_credited_by_open_adequacy():
    answer, meta, gt, material, review = example()
    meta.unresolved_operationalization_dimensions = []
    quote = 'Use the 90th percentile of all finite values.'
    gt.acceptable_operationalizations[0].decisions = [SimpleNamespace(
        dimension=SimpleNamespace(value='property_measure'), statement=REFERENCE)]
    review['result_groups'][0]['dimensions'][1]['evidence_text'] = quote
    result = evaluate((answer + ' ' + quote, meta, gt, material, review), condition='O1-F1')
    assert result['metrics']['o_score']['value'] is None

"""Do not assume an order of vector reduction absent from reference wording."""
from types import SimpleNamespace

import pytest

from flowintentbench.answer_evidence import operationalization_match_uncertainty
from test_rubric_evaluation import example, evaluate


REFERENCE = 'Use all cells, with stored cell values directly and arithmetic means of vertex values for point fields.'


@pytest.mark.parametrize('quote,guarded', [
    ('For each cell, used the arithmetic mean of its eight vertex magnitudes.', True),
    ('Use the average of the point magnitudes.', True),
    ('Use the mean of the vertex vectors, then their magnitude.', False),
    ('Use point magnitudes without averaging.', False),
    ('Use the mean of the vertex values.', False),
])
def test_reduction_order_requires_an_explicit_source_declaration(quote, guarded):
    assert bool(operationalization_match_uncertainty(quote, REFERENCE)) is guarded
    assert operationalization_match_uncertainty(quote, 'Use arithmetic means of vertex magnitudes.') is None


@pytest.mark.parametrize('claimed', [.5, .9])
@pytest.mark.parametrize('fixed', [False, True])
def test_ambiguous_reference_reduction_does_not_establish_consistency_or_error(claimed, fixed):
    answer, meta, gt, material, review = example()
    if fixed:
        meta.unresolved_operationalization_dimensions = []
    quote = 'Use the arithmetic mean of its eight vertex magnitudes.'
    answer = answer.replace('0.5', str(claimed)) + '\n' + quote
    dimension = review['result_groups'][0]['dimensions'][0]
    dimension['evidence_text'] = quote
    gt.acceptable_operationalizations[0].decisions = [SimpleNamespace(
        dimension=SimpleNamespace(value=dimension['dimension']), statement=REFERENCE)]
    review['findings'][0].update(value=claimed, evidence_text=f'CV is {claimed}.')
    result = evaluate((answer, meta, gt, material, review))
    assert result['metrics']['c_score']['value'] is None
    if claimed == .9:
        assert result['metrics']['finding_precision']['value'] is None
    assert result['error_diagnostics']['independent_contradicted_finding_ids'] == []
    assert 'VECTOR_REDUCTION_ORDER_UNRESOLVED' in str(result)

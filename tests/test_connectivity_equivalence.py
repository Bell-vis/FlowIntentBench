"""Different graph definitions must not inherit numerical reference truth."""
import pytest
from flowintentbench.answer_evidence import operationalization_match_conflict

REFERENCE = 'Treat retained grid points as one six-neighbor connected speed region.'


@pytest.mark.parametrize('quote', [
    'I used ±1-index z neighbors while holding x and y fixed.',
    'I used y neighbors while holding x and z fixed.',
    'I evaluated immediate-neighbor connectivity separately along each single grid axis, with no diagonal or cross-axis connections.',
])
def test_explicit_one_dimensional_runs_are_not_six_neighbor_components(quote):
    assert operationalization_match_conflict(quote, REFERENCE) == 'SINGLE_AXIS_RUNS_VS_SIX_NEIGHBOR_COMPONENTS'
    assert operationalization_match_conflict(quote, 'Use separate single-axis runs.') is None


@pytest.mark.parametrize('quote', [
    'Each pair of neighboring points differs by one index along exactly one axis.',
    'Use face-sharing immediate neighbors: the six ±i, ±j, ±k neighbors.',
    'Use neighbors without holding x and y fixed.',
    'I compared one-axis runs, then used full six-neighbor connected components.',
    'For comparison I used z neighbors while holding x and y fixed; the primary result uses six-neighbor components.',
])
def test_full_connectivity_and_nonbinding_comparisons_are_not_rejected(quote):
    assert operationalization_match_conflict(quote, REFERENCE) is None

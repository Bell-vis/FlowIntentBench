from copy import deepcopy

import pytest

from flowintentbench import recipe_binding_validation as materializer


def packet():
    return dict(candidate_operationalization={'decisions': ['Use cell-volume Pearson']},
        observed_quantile_method=None,
        operations={'pearson': {'recipe': {'kind': 'association', 'measure': 'pearson',
            'field': {'name': 'density', 'association': 'point'},
            'other_field': {'name': 'velocity', 'association': 'point', 'component': 2}}}})


def binding(**overrides):
    return dict(source_operation='pearson', parameter_overrides=overrides,
                semantic_binding_rationale='The candidate explicitly specifies cell-volume Pearson.')


class Transport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def request(self, operation, payload, **kwargs):
        assert operation == 'recipe_binding'
        self.calls.append(deepcopy(payload))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_invalid_controls_are_returned_to_agent_once_without_host_changes():
    original = packet()
    snapshot = deepcopy(original)
    invalid = binding(sampling='cell_volume', point_domain='all')
    corrected = binding(sampling='cell_volume')
    tr = Transport([invalid, corrected])
    recipe, evidence = materializer.request_supported_recipe_binding(tr, original, {})
    assert original == snapshot
    assert recipe['sampling'] == 'cell_volume' and 'point_domain' not in recipe
    assert evidence is None and len(tr.calls) == 2
    feedback = tr.calls[1]['binding_validation_feedback']
    assert feedback['previous_binding'] == invalid
    assert 'point_equal' in feedback['error']
    assert tr.calls[1]['candidate_operationalization'] == original['candidate_operationalization']


def test_repeated_invalid_binding_is_not_retried_or_silently_normalized():
    bad = binding(sampling='cell_volume', point_domain='finite')
    tr = Transport([bad, bad])
    with pytest.raises(ValueError, match='point_equal'):
        materializer.request_supported_recipe_binding(tr, packet(), {})
    assert len(tr.calls) == 2


def test_valid_binding_does_not_add_a_model_call():
    tr = Transport([binding(sampling='cell_volume')])
    recipe, _ = materializer.request_supported_recipe_binding(tr, packet(), {})
    assert recipe['sampling'] == 'cell_volume' and len(tr.calls) == 1


def test_missing_evidence_verdict_is_never_retried():
    from flowintentbench.evaluator import EvaluationPendingAdjudication
    tr = Transport([EvaluationPendingAdjudication('Missing declaration', pending_type='external_recipe_binding')])
    with pytest.raises(EvaluationPendingAdjudication):
        materializer.request_supported_recipe_binding(tr, packet(), {})
    assert len(tr.calls) == 1


def test_unknown_algorithm_control_remains_coverage_gap_without_new_review():
    tr = Transport([binding(time_integral=True)])
    with pytest.raises(ValueError, match='unsupported recipe parameters'):
        materializer.request_supported_recipe_binding(tr, packet(), {})
    assert len(tr.calls) == 1

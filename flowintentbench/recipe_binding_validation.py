"""Bounded agent correction of non-executable recipe bindings, never science."""
from copy import deepcopy
import math


class IncompatibleRecipeControls(ValueError):
    """Known controls were combined in a way the executor cannot accept."""


def recipe_from_binding(operations, binding, numerical_evidence=None):
    from .expansion_recipe_materializer import OVERRIDES, _valid_override, _validate_recipe_controls

    name = binding.get('source_operation')
    if name not in operations:
        raise ValueError('Unknown construction source recipe')
    if not isinstance(binding.get('semantic_binding_rationale'), str) or not binding['semantic_binding_rationale'].strip():
        raise ValueError('Independent recipe binding has no semantic rationale')
    recipe = deepcopy(operations[name]['recipe'])
    allowed = OVERRIDES.get(recipe['kind'], {})
    overrides = binding.get('parameter_overrides', {})
    if not isinstance(overrides, dict) or set(overrides) - set(allowed):
        raise ValueError('Candidate requires unsupported recipe parameters')
    for key, value in overrides.items():
        if not _valid_override(value, allowed[key]):
            raise ValueError(f'Unsupported recipe parameter: {key}')
        recipe[key] = value
    convention = None
    if recipe.get('quantile_method') == 'from_execution':
        if not numerical_evidence or recipe.get('quantile_weighting') != 'unweighted':
            raise ValueError('Unweighted percentile estimator lacks authenticated execution evidence')
        if any(not math.isclose(call['quantile'], recipe['quantile'], abs_tol=1e-12)
               for call in numerical_evidence['calls']):
            raise ValueError('Execution convention percentile does not match the bound recipe')
        recipe['quantile_method'] = numerical_evidence['method']
        convention = numerical_evidence
    try:
        _validate_recipe_controls(recipe)
    except ValueError as exc:
        raise IncompatibleRecipeControls(str(exc)) from exc
    return recipe, convention


def request_supported_recipe_binding(transport, payload, schema):
    binding = transport.request('recipe_binding', payload, output_schema=schema)
    try:
        return recipe_from_binding(payload['operations'], binding, payload.get('observed_quantile_method'))
    except IncompatibleRecipeControls as exc:
        correction = deepcopy(payload)
        correction['binding_validation_feedback'] = dict(
            previous_binding=deepcopy(binding), error=str(exc), max_corrections=1,
            instruction='The host rejected this binding before numerical execution. '
                        'Correct only the incompatible recipe controls using the unchanged candidate '
                        'and documented supported semantics. Do not invent missing scientific choices, '
                        'change the candidate, or use reported results. Return PENDING when no exact '
                        'supported binding can be established. No score or numerical result is provided.')
    corrected = transport.request('recipe_binding', correction, output_schema=schema)
    return recipe_from_binding(payload['operations'], corrected, payload.get('observed_quantile_method'))

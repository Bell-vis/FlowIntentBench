from copy import deepcopy
import json

import pytest

from flowintentbench.reference_extension_review import request, apply, additions, semantic_reference_texts
from flowintentbench.rubric_scoring import Finding, Match, _verify_match
from flowintentbench.ground_truth import ReferenceFinding
from test_reference_extension_review import fixture


def semantic_example():
    answer, judgment, old, new = fixture()
    answer = 'The selected region wins by both peak and mean speed.'
    judgment['findings'][0].update(value=None, statement='Joint peak and mean dominance', evidence_text=answer)
    judgment['result_groups'][0]['evidence_text'] = answer
    ref = {'finding_id': 'dominance', 'importance': 'supporting', 'category': 'quantity',
        'statement': 'The selected region has both a larger peak speed and a larger mean speed than the other region.',
        'value': None, 'unit': None, 'host_policy': {'verification_mode': 'semantic_only'}}
    new['branches'][0]['findings'].append(ref)
    return answer, judgment, old, new, ref


def test_semantic_mode_has_separate_truth_evidence_and_keeps_numeric_targets_hidden():
    answer, j, old, new, ref = semantic_example()
    default = request(answer, j, old, new, {'f'})
    assert default == request(answer, j, old, new, {'f'}, semantic_support=False)
    prompt, schema, aliases, binding = request(answer, j, old, new, {'f'}, semantic_support=True)
    assert '987654.321' not in prompt
    assert binding['semantic_support'] is True
    packet = json.loads(prompt.rsplit('\n\n', 1)[1])
    text = [r for r in packet['new_references'] if r['verification_kind'] == 'SEMANTIC']
    assert text[0]['reference_text'].strip() == ref['statement']
    assert 'semantic_assessments' in schema['properties']['mappings']['items']['required']


@pytest.mark.parametrize('verdict, expected', [('SUPPORTED', True), ('REFUTED', False), ('UNVERIFIABLE', None)])
def test_only_source_bound_semantic_verdict_reaches_scorer(verdict, expected):
    answer, j, old, new, ref = semantic_example()
    _, _, aliases, _ = request(answer, j, old, new, {'f'}, semantic_support=True)
    texts = semantic_reference_texts(additions(old, new), aliases)
    alias = next(iter(texts))
    payload = {'mappings': [{'finding_id': 'f', 'status': 'MATCHED', 'reference_ids': [alias],
        'source_start_line': 1, 'source_end_line': 1, 'semantic_assessments': [
            {'reference_id': alias, 'verdict': verdict, 'reference_evidence_text': ref['statement']}]}]}
    result, audit = apply(answer, j, aliases, payload, {'f'}, semantic_texts=texts)
    finding = Finding.model_validate(result['findings'][0])
    match = Match.model_validate(result['findings'][0]['matches'][0])
    reference = ReferenceFinding.model_validate({k: v for k, v in ref.items() if k != 'host_policy'})
    assert _verify_match(answer, finding, match, reference, ref['host_policy'], None, {})[0] is expected
    payload['mappings'][0]['semantic_assessments'][0]['reference_evidence_text'] = 'Invented scientific proof.'
    result, audit = apply(answer, j, aliases, payload, {'f'}, semantic_texts=texts)
    assert result['findings'][0]['matches'][0]['semantic_verdict'] == 'UNVERIFIABLE'
    assert j['findings'][0]['matches'] == []


def test_numeric_truth_and_missing_or_duplicate_semantic_assessments_are_rejected():
    answer, j, old, new, ref = semantic_example()
    _, _, aliases, _ = request(answer, j, old, new, {'f'}, semantic_support=True)
    texts = semantic_reference_texts(additions(old, new), aliases)
    numeric = next(a for a in aliases if a not in texts)
    semantic = next(iter(texts))
    row = {'finding_id': 'f', 'status': 'MATCHED', 'reference_ids': [semantic],
        'source_start_line': 1, 'source_end_line': 1, 'semantic_assessments': []}
    with pytest.raises(ValueError, match='exactly'):
        apply(answer, j, aliases, {'mappings': [row]}, {'f'}, semantic_texts=texts)

    assessment = {'reference_id': semantic, 'verdict': 'SUPPORTED', 'reference_evidence_text': ref['statement']}
    row['semantic_assessments'] = [assessment, deepcopy(assessment)]
    with pytest.raises(ValueError, match='exactly'):
        apply(answer, j, aliases, {'mappings': [row]}, {'f'}, semantic_texts=texts)
    row.update(reference_ids=[numeric], semantic_assessments=[{**assessment, 'reference_id': numeric}])
    with pytest.raises(ValueError, match='exactly'):
        apply(answer, j, aliases, {'mappings': [row]}, {'f'}, semantic_texts=texts)


def test_bad_semantic_quote_does_not_block_another_numeric_mapping():
    answer, j, old, new, ref = semantic_example()
    answer += '\nMean density is 2.0.'
    j['findings'].append({'finding_id': 'numeric', 'group_id': 'g', 'statement': 'mean density',
        'value': 2., 'unit': None, 'evidence_text': 'Mean density is 2.0.', 'eligible': True,
        'matches': [], 'mapping_complete': True})
    _, _, aliases, _ = request(answer, j, old, new, {'f', 'numeric'}, semantic_support=True)
    texts = semantic_reference_texts(additions(old, new), aliases)
    semantic = next(iter(texts)); numeric = next(a for a in aliases if a not in texts)
    payload = {'mappings': [
        {'finding_id': 'f', 'status': 'MATCHED', 'reference_ids': [semantic],
         'source_start_line': 1, 'source_end_line': 1, 'semantic_assessments': [
            {'reference_id': semantic, 'verdict': 'SUPPORTED', 'reference_evidence_text': 'Invented proof.'}]},
        {'finding_id': 'numeric', 'status': 'MATCHED', 'reference_ids': [numeric],
         'source_start_line': 2, 'source_end_line': 2, 'semantic_assessments': []}]}
    result, audit = apply(answer, j, aliases, payload, {'f', 'numeric'}, semantic_texts=texts)
    assert result['findings'][0]['matches'][0]['semantic_verdict'] == 'UNVERIFIABLE'
    assert result['findings'][1]['matches'] == [{'branch_id': 'b', 'finding_id': 'mean'}]
    assert any(a['reason'] == 'SEMANTIC_REFERENCE_QUOTE_UNBOUND' for a in audit)

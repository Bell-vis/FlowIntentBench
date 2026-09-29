import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from flowintentbench.external_file_evaluator import FileJudgmentTransport
from flowintentbench.evaluator import EvaluationPendingAdjudication
from scripts.run_codex_file_judgments import run_judgments
from scripts.batched_file_judgments import packet_for
from scripts.grading_policy import batch_key


def requests(tmp_path, count=4, *, answer='answer', operation='semantic_match'):
    ex = tmp_path / 'exchange'
    tr = FileJudgmentTransport(ex, context=dict(case_id='case', response_sha256=answer,
        evaluation_manifest_sha256='frozen', model_id='HIDDEN_MODEL'))
    for index in range(count):
        if operation == 'semantic_match':
            payload = dict(purpose='finding', scientific_question='Shared question', case_context={},
                finding_goal='Goal', dimension=None, finding_category='scalar',
                predicted_statement=f'Claim {index}', reference_statement=f'Reference {index}',
                predicted_value=None, predicted_unit=None, reference_value=None, reference_unit=None,
                verification_mode='semantic_only')
        else:
            payload = dict(scientific_question='Shared question', case_context={}, finding_goal='Goal',
                finding=dict(prediction_id=f'F{index}', statement=f'Claim {index}', value=None, unit=None))
        with pytest.raises(EvaluationPendingAdjudication):
            tr(operation, payload)
    qs = [json.loads((ex / 'requests' / (rid + '.json')).read_text()) for rid in tr.pending_request_ids]
    inv = tmp_path / 'inventory.json'
    inv.write_text(json.dumps({'by_operation': {operation: [{'request_id': r['request_id']} for r in qs]}}))
    return ex, inv, qs


def resolved(value=None):
    return dict(status='RESOLVED', response=value or {'result': 'MATCH'}, reason='Supported by supplied text', evidence_files=[])


def receipt(value, kwargs):
    return dict(completed=True, returncode=0, timed_out=False, thread_id='test',
                transport='codex_exec_chatgpt_login', final_text=json.dumps(value),
                events_path=str(kwargs['output_dir'] / 'events.jsonl'),
                usage={'input_tokens': 100, 'output_tokens': 20})


def test_four_atomic_judgments_use_one_call_preserve_original_contract(tmp_path):
    ex, inv, qs = requests(tmp_path)
    calls = []
    def run(**kw):
        calls.append(kw)
        return receipt({key: resolved() for key in kw['output_schema']['properties']}, kw)
    result = run_judgments(ex, inv, tmp_path / 'work', 'judge', lean=True, runner=run)
    assert len(calls) == 1 and result['resolved'] == 4 and result['dispatches'] == 1
    assert calls[0]['reasoning_effort'] == 'medium'
    assert 'HIDDEN_MODEL' not in calls[0]['prompt']
    assert calls[0]['prompt'].count('Shared question') == 1
    for request in qs:
        env = json.loads((ex / 'responses' / (request['request_id'] + '.json')).read_text())
        assert env['response'] == {'result': 'MATCH'}
        assert env['provenance']['review_policy']['effective_effort'] == 'medium'
        assert Path(env['provenance']['audit_path']).exists()
    assert sum(len(r['transports']) for r in result['results']) == 1
    usage = json.loads((tmp_path / 'judgment_usage.json').read_text())
    assert usage['physical_receipts'] == 1 and usage['input_tokens'] == 100


def test_batch_reconstructs_exact_input_and_cannot_cross_answers_or_roles(tmp_path):
    _, _, qs = requests(tmp_path)
    packet = packet_for(qs)
    for i, request in enumerate(qs):
        assert {**packet['shared_input'], **packet['items'][f'q{i}']} == request['input']
    other = json.loads(json.dumps(qs[1]))
    for mutate in [lambda q: q['context'].update(response_sha256='different'),
                   lambda q: q['input'].update(purpose='finding_deduplication'),
                   lambda q: q['context'].update(evaluation_manifest_sha256='other')]:
        changed = json.loads(json.dumps(other)); mutate(changed)
        with pytest.raises(ValueError, match='scope'):
            packet_for([qs[0], changed])
    other['context'] = {}
    assert batch_key(other) is None


@pytest.mark.parametrize('bad', ['missing', 'invalid', 'evidence'])
def test_bad_batch_child_does_not_discard_valid_siblings(tmp_path, bad):
    ex, inv, qs = requests(tmp_path)
    def run(**kw):
        value = {key: resolved() for key in kw['output_schema']['properties']}
        bad_key = 'q1' if 'q1' in value else next(iter(value))
        if bad == 'missing': del value[bad_key]
        if bad == 'invalid': value[bad_key]['response'] = {'result': 'WRONG_ENUM'}
        if bad == 'evidence': value[bad_key]['evidence_files'] = ['../secret']
        return receipt(value, kw)
    r = run_judgments(ex, inv, tmp_path / 'work', 'judge', lean=True, runner=run,
                      continue_on_format_error=True)
    assert r['resolved'] == 3 and r['errors'] == 1 and r['fatal_errors'] == 0
    assert len(list((ex / 'responses').glob('*.json'))) == 3


def test_pending_child_remains_pending_and_existing_responses_not_repeated(tmp_path):
    ex, inv, _ = requests(tmp_path)
    calls = []
    def run(**kw):
        calls.append(kw)
        value = {key: resolved() for key in kw['output_schema']['properties']}
        value['q2'] = dict(status='PENDING', response=None, reason='Missing evidence', evidence_files=[])
        return receipt(value, kw)
    r = run_judgments(ex, inv, tmp_path / 'work', 'judge', lean=True, runner=run)
    assert r['resolved'] == 3 and r['pending'] == 1
    r = run_judgments(ex, inv, tmp_path / 'work', 'judge', lean=True, runner=run)
    assert len(calls) == 1 and r['skipped_existing'] == 4


def test_solver_effort_default_unchanged_and_text_effort_is_explicit():
    from scripts.codex_console_transport import command
    assert 'model_reasoning_effort="xhigh"' in command('solver', '.')
    assert 'model_reasoning_effort="high"' in command('judge', '.', reasoning_effort='high')


def test_eligibility_keeps_original_prompt_with_explicit_medium_effort(tmp_path):
    ex, inv, qs = requests(tmp_path, count=2, operation='eligibility')
    assert all(batch_key(q) is None for q in qs)
    calls = []
    value = dict(scientifically_interpretable=True, relevant_to_finding_goal=True,
                 in_principle_verifiable=True)
    def run(**kw):
        calls.append(kw)
        return receipt(resolved(value), kw)
    result = run_judgments(ex, inv, tmp_path / 'work', 'judge', lean=True, runner=run)
    assert result['resolved'] == 2 and result['dispatches'] == 2
    assert all(kw['reasoning_effort'] == 'medium' for kw in calls)
    assert all('Verification runtime' in kw['prompt'] for kw in calls)
    assert all('This is a text-only review' not in kw['prompt'] for kw in calls)


def test_dependency_refresh_happens_while_other_call_is_running(tmp_path):
    ex, inv, qs = requests(tmp_path, 2)
    # Different answer contexts force separate independent calls.
    _, _, other = requests(tmp_path, 1, answer='other')
    inv.write_text(json.dumps({'by_operation': {'semantic_match':
        [{'request_id': q['request_id']} for q in [qs[0], other[0]]]}}))
    barrier, refreshed = threading.Barrier(2), threading.Event()
    # Assign which call waits under a lock rather than rely on task completion order.
    gate = threading.Lock(); seen = []
    def controlled(**kw):
        with gate:
            first = not seen; seen.append(True)
        barrier.wait(timeout=5)
        if not first: assert refreshed.wait(5)
        return receipt(resolved(), kw)
    r = run_judgments(ex, inv, tmp_path / 'work', 'judge', lean=True, workers=2,
        runner=controlled, on_resolved=lambda: refreshed.set())
    assert r['resolved'] == 2 and refreshed.is_set()


def test_numeric_cache_reuses_facts_and_binds_parameters_and_version(tmp_path):
    from flowintentbench.expansion_recipe_materializer import cached_recipe_result
    calls = []
    def compute(): calls.append(1); return {'value': 2.5}
    identity = {'data_hash': 'abc', 'weights': 'volume', 'version': 1}
    assert cached_recipe_result(tmp_path, identity, compute) == {'value': 2.5}
    assert cached_recipe_result(tmp_path, identity, compute) == {'value': 2.5}
    assert len(calls) == 1
    cached_recipe_result(tmp_path, dict(identity, weights='equal'), compute)
    assert len(calls) == 2
    path = next(tmp_path.glob('*.json')); value = json.loads(path.read_text()); value['result']['value'] = 99
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='digest'):
        cached_recipe_result(tmp_path, value['identity'], compute)


def test_numeric_cache_concurrent_compute_once(tmp_path):
    from flowintentbench.expansion_recipe_materializer import cached_recipe_result
    calls = []
    def compute(): calls.append(1); return {'value': 3}
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: cached_recipe_result(tmp_path, {'version': 1}, compute), range(2)))
    assert len(calls) == 1 and results == [{'value': 3}, {'value': 3}]


def test_batch_repairs_only_invalid_child_once(tmp_path):
    ex, inv, qs = requests(tmp_path)
    calls = []
    def run(**kw):
        calls.append(kw)
        values = {key: resolved() for key in kw['output_schema']['properties']}
        if len(calls) == 1:
            values['q1']['response'] = {'result': 'WRONG_ENUM'}
        return receipt(values, kw)
    r = run_judgments(ex, inv, tmp_path/'work', 'judge', lean=True, runner=run)
    assert r['resolved'] == 4 and r['errors'] == 0
    assert [len(c['output_schema']['properties']) for c in calls] == [4, 1]
    assert calls[1]['timeout_seconds'] < calls[0]['timeout_seconds']
    usage = json.loads((tmp_path/'judgment_usage.json').read_text())
    assert usage['physical_receipts'] == 2 and usage['input_tokens'] == 200


def test_slow_replay_does_not_block_ready_model_dispatch(tmp_path):
    ex, inv, first = requests(tmp_path, 1, answer='first')
    _, _, second = requests(tmp_path, 1, answer='second')
    inv.write_text(json.dumps({'by_operation': {'semantic_match':
        [{'request_id': q['request_id']} for q in first + second]}}))
    second_started = threading.Event()
    count = []
    def run(**kw):
        count.append(1)
        if len(count) == 2: second_started.set()
        return receipt(resolved(), kw)
    def refresh():
        assert second_started.wait(3), 'Replay blocked the next ready model task'
    r = run_judgments(ex, inv, tmp_path/'work', 'judge', lean=True, workers=1,
                      runner=run, on_resolved=refresh)
    assert r['resolved'] == 2

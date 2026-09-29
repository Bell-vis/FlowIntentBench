"""Offline end-to-end dependency and fairness tests; no inference calls."""
from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from flowintentbench.external_file_evaluator import FileJudgmentTransport, adjudication_schema, digest
from flowintentbench.evaluator import EvaluationAdjudications, EvaluationPendingAdjudication, continuation_request_id
from scripts.replay_judgment_completion import ReplayJudgmentCompletion, literal_candidate_response
from scripts.benchmark_hybrid import HybridController
from scripts.run_claude_benchmark import ClaudeAdmission, ClaudeController
from scripts import collect_subagent_runs as collector
from tests.test_file_judgment_carry_forward import request, resolve


def context(version):
    return dict(case_id='case', response_sha256='same-answer',
                evaluation_material_sha256='same-material', evaluation_manifest_sha256=version*64)


def test_reuse_closes_multiple_dependency_stages_in_one_traversal(tmp_path):
    for value in ['speed', 'region', 'finding', 'unit']:
        rid = request(tmp_path, 'a', value=value)
        resolve(tmp_path, rid)
    completion = ReplayJudgmentCompletion(tmp_path)
    transport = FileJudgmentTransport(tmp_path, context=context('b'))
    transport.response_reuser = completion
    for value in ['speed', 'region', 'finding', 'unit']:
        assert transport.request('semantic_match', {'statement':value}, output_schema={'type':'object'}) == {'result':'MATCH'}
    assert completion.stats['reused'] == 4
    assert completion.stats['indexed_responses'] == 4  # One index, not one scan per stage.
    assert transport.pending_request_ids == [] and len(transport.used_response_ids) == 4


def test_changed_evidence_and_conflicting_verdicts_are_not_reused(tmp_path):
    resolve(tmp_path, request(tmp_path, 'a'), 'MATCH')
    resolve(tmp_path, request(tmp_path, 'b'), 'NO_MATCH')
    tr = FileJudgmentTransport(tmp_path, context=context('c'))
    tr.response_reuser = ReplayJudgmentCompletion(tmp_path)
    for payload in [{'statement':'speed'}, {'statement':'changed evidence'}]:
        with pytest.raises(EvaluationPendingAdjudication):
            tr.request('semantic_match',payload,output_schema={'type':'object'})
    assert tr.response_reuser.stats['conflicts'] == 1
    assert tr.response_reuser.stats['reused'] == 0


def test_invocation_index_avoids_rescan_and_revalidates_source_bytes(tmp_path, monkeypatch):
    rid = request(tmp_path, 'a')
    resolve(tmp_path, rid)
    completion = ReplayJudgmentCompletion(tmp_path)
    completion.save_index(tmp_path/'index.json')
    restored = ReplayJudgmentCompletion(tmp_path)
    restored.load_index(tmp_path/'index.json')
    monkeypatch.setattr(restored, 'build_index', lambda: pytest.fail('Historical tree scanned twice'))
    tr = FileJudgmentTransport(tmp_path, context=context('b'))
    tr.response_reuser = restored
    assert tr.request('semantic_match', {'statement':'speed'},output_schema={'type':'object'}) == {'result':'MATCH'}
    path = tmp_path/'requests'/(rid+'.json')
    corrupted = json.loads(path.read_text());corrupted['input']['statement'] = 'changed'
    path.write_text(json.dumps(corrupted))
    tr.context = context('c')
    with pytest.raises(ValueError):
        tr.request('semantic_match', {'statement':'speed'},output_schema={'type':'object'})


def test_parallel_history_index_matches_serial_and_reports_all_files(tmp_path):
    for version in ('a', 'b', 'c'):
        for value in ('speed', 'region', 'unit'):
            resolve(tmp_path, request(tmp_path, version, value=value))
    serial = ReplayJudgmentCompletion(tmp_path)
    serial.build_index(workers=1)
    parallel = ReplayJudgmentCompletion(tmp_path)
    progress = []
    parallel.build_index(workers=4, progress=lambda done, total: progress.append((done, total)))
    assert parallel.index == serial.index
    assert parallel.stats['indexed_responses'] == 9
    assert progress[0] == (0, 9) and progress[-1] == (9, 9)


def test_parallel_history_index_fails_closed_on_corrupt_provenance(tmp_path):
    rid = request(tmp_path, 'a')
    resolve(tmp_path, rid)
    path = tmp_path / 'responses' / (rid + '.json')
    value = json.loads(path.read_text())
    value['request_id'] = 'incorrect'
    path.write_text(json.dumps(value))
    completion = ReplayJudgmentCompletion(tmp_path)
    with pytest.raises(ValueError, match='identity mismatch'):
        completion.build_index(workers=4)
    assert completion.index is None


def candidate_request(tmp_path):
    supplied = {'decisions':[dict(dimension='feature_definition', statement='Use connected points.'),
        dict(dimension='criterion', statement='Keep speed > 1.') ]}
    payload = dict(pending_type='novel_operationalization_candidate',
        continuation_context=dict(continuation_request=supplied,
            continuation_request_id=continuation_request_id('operationalization_completion', supplied)),
        accumulated_adjudications=EvaluationAdjudications().to_dict(), instruction='Complete the candidate only.')
    tr = FileJudgmentTransport(tmp_path, context=context('a'))
    with pytest.raises(EvaluationPendingAdjudication):
        tr.request('adjudication', payload, output_schema=adjudication_schema())
    return json.loads((tmp_path/'requests'/(tr.pending_request_ids[0]+'.json')).read_text())


def test_literal_completion_preserves_all_choices_and_cannot_accept_science(tmp_path):
    req = candidate_request(tmp_path)
    before = deepcopy(req)
    envelope = literal_candidate_response(req)
    assert req == before
    o = envelope['response']['novel_operationalization']
    assert o['status'] == 'UNRESOLVED'
    assert o['operationalization']['decisions'] == req['input']['continuation_context']['continuation_request']['decisions']
    assert envelope['provenance']['api_calls'] == 0
    assert not envelope['provenance']['scientific_judgment']
    assert envelope['response']['novel_findings'] == []
    scientific = deepcopy(req)
    scientific['input']['pending_type'] = 'novel_operationalization_scientific_adjudication'
    assert literal_candidate_response(scientific) is None


def test_literal_completion_is_consumed_by_original_typed_transport(tmp_path):
    req = candidate_request(tmp_path)
    tr = FileJudgmentTransport(tmp_path, context=req['context'])
    tr.response_reuser = ReplayJudgmentCompletion(tmp_path)
    reply = tr.request('adjudication', req['input'], output_schema=req['output_schema'])
    assert reply['novel_operationalization']['status'] == 'UNRESOLVED'
    assert tr.response_reuser.stats['candidates_transcribed'] == 1
    assert tr.pending_request_ids == []


def test_literal_completion_refuses_missing_choice_or_wrong_identity(tmp_path):
    req = candidate_request(tmp_path)
    req['input']['continuation_context']['continuation_request_id'] = 'wrong'
    with pytest.raises(ValueError, match='identity mismatch'):
        literal_candidate_response(req)
    supplied = req['input']['continuation_context']['continuation_request']
    supplied['decisions'][0]['statement'] = ''
    req['input']['continuation_context']['continuation_request_id'] = continuation_request_id('operationalization_completion', supplied)
    assert literal_candidate_response(req) is None


def test_recovered_solver_gets_lane_before_second_task_of_other_model(tmp_path):
    h = HybridController.__new__(HybridController)
    h.model_cursor = 0
    luna, terra = collector.MODELS
    h.governor = SimpleNamespace(available=lambda _: True)
    state = dict(slots=[dict(slot_id='a',model_id=luna,status='PENDING'),
                        dict(slot_id='b',model_id=terra,status='PENDING')])
    occupied = Counter({('api',luna):1})
    assert h.next_api_slot(state,set(),occupied,Counter(),2)[0] == terra
    h.governor.available = lambda model: model == luna
    assert h.next_api_slot(state,set(),occupied,Counter(),2)[0] == luna  # Borrow idle lane.


def test_claude_supplier_outage_does_not_gate_gpt_grading(tmp_path):
    h = ClaudeController.__new__(ClaudeController)
    h.output = tmp_path
    h.cli_disabled = True
    h.governor = ClaudeAdmission(tmp_path/'claude_health.json')
    h.governor.failed('claude-fable-5-1',dict(error='claude_http_503',provider_unavailable=True))
    assert not h.governor.available('claude-sonnet-5')
    h.judge_governor = SimpleNamespace(check_health=lambda: True, available=lambda _: True)
    calls = []
    h.agent = lambda **kw: calls.append(kw) or dict(completed=True)
    assert h.judge_runner(model='gpt-6-astra', reasoning_effort='high')['completed']
    assert len(calls) == 1


def test_gpt_grading_outage_does_not_gate_claude_solver(tmp_path):
    h = ClaudeController.__new__(ClaudeController)
    h.governor = ClaudeAdmission(tmp_path/'claude_health.json')
    h.judge_governor = SimpleNamespace(check_health=lambda: False)
    assert h.refresh_health()  # Collector checks its own transport, not GPT health.
    assert h.governor.available('claude-sonnet-5')


def test_raw_hash_cache_reuses_unchanged_bytes_but_detects_mutation(tmp_path, monkeypatch):
    from scripts import console_evaluation_cache as cache
    ex = tmp_path/'exchange';data = ex/'review_data'/'raw.bin'
    data.parent.mkdir(parents=True);data.write_bytes(b'original')
    calls = []
    real_hash = cache.hashlib.sha256
    def counted(*a, **k):
        calls.append(1)
        return real_hash(*a, **k)
    monkeypatch.setattr(cache.hashlib, 'sha256', counted)
    first = cache.replay_guard(tmp_path, ex)
    assert cache.replay_guard(tmp_path, ex) == first and len(calls) == 1
    data.write_bytes(b'modified')
    assert cache.replay_guard(tmp_path, ex) != first and len(calls) == 2

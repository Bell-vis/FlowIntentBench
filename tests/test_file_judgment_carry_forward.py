import json

import pytest

from flowintentbench.external_file_evaluator import FileJudgmentTransport, write_json
from flowintentbench.evaluator import EvaluationPendingAdjudication
from scripts.carry_forward_file_judgments import carry_forward


def request(exchange, version, *, value="speed", role="semantic_match", case="case"):
    transport = FileJudgmentTransport(exchange, context={
        "case_id": case, "response_sha256": "same-answer",
        "evaluation_material_sha256": "same-material", "evaluation_manifest_sha256": version * 64})
    with pytest.raises(EvaluationPendingAdjudication) as exc:
        transport.request(role, {"statement": value}, output_schema={"type": "object"})
    return exc.value.continuation_context["external_request_id"]


def resolve(exchange, request_id, result="MATCH"):
    write_json(exchange / "responses" / (request_id + ".json"), {
        "request_id": request_id, "reviewer_id": "/root/independent_reviewer",
        "status": "RESOLVED", "response": {"result": result}})


def inventory(exchange, ids):
    path = exchange / "inventory.json"
    write_json(path, {"by_operation": {"semantic_match": [{"request_id": x} for x in ids]}})
    return path


def test_identical_task_reuse_retains_reviewer_and_audit_without_overwrite(tmp_path):
    old = request(tmp_path, "a")
    resolve(tmp_path, old)
    old_bytes = (tmp_path / "responses" / (old + ".json")).read_bytes()
    new = request(tmp_path, "b")
    inv = inventory(tmp_path, [new])
    assert carry_forward(tmp_path, inv)["results"][0]["status"] == "IDENTICAL_TASK_REUSABLE"
    assert not (tmp_path / "responses" / (new + ".json")).exists()
    result = carry_forward(tmp_path, inv, apply=True)
    assert result["new_scientific_judgments"] == 0
    answer = json.loads((tmp_path / "responses" / (new + ".json")).read_text())
    assert answer["reviewer_id"] == "/root/independent_reviewer"
    assert answer["carry_forward"]["source_request_id"] == old
    assert (tmp_path / "carry_forward" / (new + ".json")).exists()
    assert (tmp_path / "responses" / (old + ".json")).read_bytes() == old_bytes
    assert carry_forward(tmp_path, inv, apply=True)["results"][0]["status"] == "EXISTING_RESPONSE_PRESERVED"


def test_changed_evidence_case_or_role_requires_new_review(tmp_path):
    old = request(tmp_path, "a")
    resolve(tmp_path, old)
    changed = [request(tmp_path, "b", value="pressure"),
               request(tmp_path, "b", case="other-case"),
               request(tmp_path, "b", role="eligibility")]
    results = carry_forward(tmp_path, inventory(tmp_path, changed), apply=True)["results"]
    assert all(r["status"] == "REQUIRES_REVIEW" for r in results)


def test_conflicting_reviews_are_not_selected_for_favorable_outcome(tmp_path):
    first = request(tmp_path, "a")
    resolve(tmp_path, first, "MATCH")
    second = request(tmp_path, "b")
    resolve(tmp_path, second, "NO_MATCH")
    new = request(tmp_path, "c")
    result = carry_forward(tmp_path, inventory(tmp_path, [new]), apply=True)
    assert result["results"][0]["status"] == "CONFLICT_REQUIRES_REVIEW"
    assert not (tmp_path / "responses" / (new + ".json")).exists()


def test_modified_request_without_rebinding_is_rejected(tmp_path):
    old = request(tmp_path, "a")
    resolve(tmp_path, old)
    path = tmp_path / "requests" / (old + ".json")
    payload = json.loads(path.read_text())
    payload["input"]["statement"] = "altered evidence"
    write_json(path, payload)
    with pytest.raises(ValueError, match="visible-input hash mismatch"):
        carry_forward(tmp_path, inventory(tmp_path, [old]), apply=True)


@pytest.mark.parametrize('mode', ['carry_forward', 'replay'])
def test_pending_verdict_is_reused_without_resampling(tmp_path, mode):
    from scripts.replay_judgment_completion import ReplayJudgmentCompletion
    old = request(tmp_path, 'a')
    source = {'request_id': old, 'reviewer_id': 'third-party-api:gpt-6-astra:test',
              'status': 'PENDING', 'response': None, 'reason': 'Missing unit declaration'}
    write_json(tmp_path / 'responses' / (old + '.json'), source)
    new = request(tmp_path, 'b')
    if mode == 'carry_forward':
        carry_forward(tmp_path, inventory(tmp_path, [new]), apply=True)
    else:
        completion = ReplayJudgmentCompletion(tmp_path)
        completion(json.loads((tmp_path / 'requests' / (new + '.json')).read_text()))
    result = json.loads((tmp_path / 'responses' / (new + '.json')).read_text())
    assert result['status'] == 'PENDING' and result['response'] is None
    assert result['reason'] == source['reason']
    assert result['carry_forward']['source_request_id'] == old
    changed = request(tmp_path, 'c', value='new source unit evidence')
    if mode == 'carry_forward':
        carry_forward(tmp_path, inventory(tmp_path, [changed]), apply=True)
    else:
        completion(json.loads((tmp_path / 'requests' / (changed + '.json')).read_text()))
    assert not (tmp_path / 'responses' / (changed + '.json')).exists()


@pytest.mark.parametrize('mode', ['carry_forward', 'replay'])
def test_resolved_and_pending_conflict_is_not_silently_selected(tmp_path, mode):
    from scripts.replay_judgment_completion import ReplayJudgmentCompletion
    old = request(tmp_path, 'a')
    resolve(tmp_path, old)
    held = request(tmp_path, 'b')
    write_json(tmp_path / 'responses' / (held + '.json'), {
        'request_id': held, 'reviewer_id': 'test-reviewer', 'status': 'PENDING',
        'response': None, 'reason': 'Missing evidence'})
    new = request(tmp_path, 'c')
    if mode == 'carry_forward':
        carry_forward(tmp_path, inventory(tmp_path, [new]), apply=True)
    else:
        ReplayJudgmentCompletion(tmp_path)(
            json.loads((tmp_path / 'requests' / (new + '.json')).read_text()))
    assert not (tmp_path / 'responses' / (new + '.json')).exists()

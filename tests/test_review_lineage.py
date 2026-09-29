import json

import pytest

from flowintentbench.review_lineage import audit_question_review_lineage
from flowintentbench.scientific_run_snapshots import canonical_json_sha256


def _setup(tmp_path, *, current="Where is the region?", reviewed="Where is the region?", live=True):
    (tmp_path / "reviews").mkdir()
    (tmp_path / "case.json").write_text(json.dumps({"scientific_question": current}))
    (tmp_path / "manifest.json").write_text(json.dumps({"cases": [{"case_id": "a", "condition": "O1-F1", "case_input_path": "case.json"}]}))
    digest = canonical_json_sha256({"model_visible_text": reviewed})
    record = {
        "reviewed_model_visible_text": reviewed,
        "reviewed_model_visible_text_sha256": digest,
        "invocation_status": "SUCCESS",
        "final_flow_expert_review_status": "PASS",
        "provenance": {
            "review_type": "FINAL_FLOW_EXPERT_WORDING_REVIEW",
            "execution_mode": "LIVE_MODEL_CALL" if live else "INJECTED_REVIEWER",
            "live_model_calls": live,
            "invocation_status": "SUCCESS",
            "reviewed_model_visible_text_sha256": digest,
        },
    }
    (tmp_path / "reviews/live.json").write_text(json.dumps({"nested": [record]}))
    return record


@pytest.mark.parametrize("current,live,count", [
    ("Where is the region?", True, 1),
    ("Where is the strongest region?", True, 0),
    ("Where is the region?", False, 0),
])
def test_current_text_requires_matching_live_review(tmp_path, current, live, count):
    _setup(tmp_path, current=current, live=live)
    result = audit_question_review_lineage(tmp_path, tmp_path / "manifest.json", [tmp_path / "reviews"])
    assert result["current_text_with_live_final_review_count"] == count
    assert result["user_study_readiness"] == "NOT_ESTABLISHED_BY_THIS_AUDIT"


def test_shallow_pass_and_corrupt_text_hash_cannot_establish_review(tmp_path):
    record = _setup(tmp_path)
    record["reviewed_model_visible_text"] = "A different text with a stale hash"
    (tmp_path / "reviews/live.json").write_text(json.dumps([record, {"review_status": "PASS"}]))
    result = audit_question_review_lineage(tmp_path, tmp_path / "manifest.json", [tmp_path / "reviews"])
    assert result["current_text_with_live_final_review_count"] == 0


def test_unreadable_review_is_reported_as_incomplete_search(tmp_path):
    _setup(tmp_path)
    (tmp_path / "reviews/broken.json").write_text("{")
    result = audit_question_review_lineage(tmp_path, tmp_path / "manifest.json", [tmp_path / "reviews"])
    assert result["search_complete"] is False
    assert result["read_errors"][0]["error"] == "JSONDecodeError"


def test_live_review_cache_binds_packet_instruction_and_profile():
    import hashlib
    from flowintentbench.live_agents import _digest, build_live_role_system_instruction
    from flowintentbench.proxy_expert import build_firewalled_payload
    from flowintentbench.review_lineage import live_review_call_matches
    packet = {'model_visible_text': 'Describe the supplied velocity field.'}
    call = {'invocation_status': 'SUCCESS', 'execution_mode': 'LIVE_MODEL_CALL', 'live_model_calls': True,
            'agent_profile_sha256': 'profile',
            'visible_input_sha256': _digest(build_firewalled_payload('scientific_reviewer', packet)),
            'system_instruction_sha256': hashlib.sha256(build_live_role_system_instruction('scientific_reviewer', 'review policy').encode()).hexdigest()}
    assert live_review_call_matches(call, packet, 'review policy', 'profile')
    assert not live_review_call_matches(call, packet, 'changed policy', 'profile')
    assert not live_review_call_matches(call, {'model_visible_text': 'Changed'}, 'review policy', 'profile')
    assert not live_review_call_matches(call, packet, 'review policy', 'changed profile')
    assert not live_review_call_matches({**call, 'invocation_status': 'PROVIDER_ERROR'}, packet, 'review policy', 'profile')

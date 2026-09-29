from argparse import Namespace
import json

from flowintentbench.external_file_evaluator import write_json
from flowintentbench.trusted_scoring import empty_metrics
from scripts import run_model_evaluation_sequence as sequence


def row(model, source="TRUSTED_HTTP_REVIEW"):
    return {"model_id": model, "protocol": sequence.VERSION, "case_id": "c", "trial": 1, "collection_status": "COMPLETED",
            "status": "SCORED", "provenance": {"source": source}, "metrics": empty_metrics("O1-F1", "unverified")}


def test_historical_fallback_does_not_end_model_stage_and_bounds_do_not_become_points(tmp_path):
    write_json(tmp_path / "cases/a.json", row("a", "CACHED_OUTCOME_EVIDENCE"))
    assert not sequence.snapshot(tmp_path, "a")["review_completion"]
    write_json(tmp_path / "cases/a.json", row("a"))
    report = sequence.snapshot(tmp_path, "a")
    assert report["review_completion"]
    assert not report["scientific_identification"]
    assert report["metric_status_counts"]["INTERVAL"] > 0


def test_receipts_count_cost_after_interrupted_batch_and_distinguish_live_attempt(tmp_path):
    write_json(tmp_path / "api/one/receipt.json", {"completed": True, "finished_epoch": 2,
        "usage": {"input_tokens": 100, "output_tokens": 20}})
    write_json(tmp_path / "api/two/receipt.json", {"completed": False, "usage": {}})
    value = sequence.snapshot(tmp_path, "a")
    assert value["new_api_attempts"] == 2
    assert value["finished_api_attempts"] == 1
    assert value["new_total_tokens"] == 120
    assert value["api_failures"] == 0
    assert value["usage_missing_finished_attempts"] == 0
    write_json(tmp_path / "api/three/receipt.json", {"completed": False, "finished_epoch": 5,
        "error": "TimeoutError: Shared API admission deadline exceeded before sending", "usage": {}})
    value = sequence.snapshot(tmp_path, "a")
    assert value["local_admission_failures"] == 1
    assert value["api_failures"] == value["usage_missing_finished_attempts"] == 0


def test_sequence_finishes_current_model_before_advancing_and_stops_on_provider_error(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    write_json(plan, {"models": [{"model_id": "luna", "collection": "source"},
                                  {"model_id": "terra", "collection": "source"},
                                  {"model_id": "next", "collection": "source"}]})
    args = Namespace(plan=plan, output=tmp_path / "out")
    calls = []
    def child(args, item, state):
        model = item["model_id"]
        calls.append(model)
        write_json(args.output / model / "cases/a.json", row(model,
            "CACHED_OUTCOME_EVIDENCE" if len(calls) == 1 or model == "terra" else "TRUSTED_HTTP_REVIEW"))
        write_json(args.output / model / "reports/experiment_report.json",
                   {"evaluation_cost": {"provider_circuit_open": model == "terra"}})
        return 2
    monkeypatch.setattr(sequence, "run_child", child)
    assert sequence.run(args) == 2
    assert calls == ["luna", "luna", "terra"]
    state = json.loads((args.output / "sequence_state.json").read_text())
    assert state["status"] == "NEEDS_INTERVENTION"
    assert state["models"]["luna"]["review_completion"]
    assert "next" not in state["models"]


def test_host_repair_replay_is_explicit_per_model(tmp_path):
    args = Namespace(output=tmp_path/'out', auth_path=tmp_path/'auth.json', api_config=tmp_path/'api.toml',
        batch_calls=12, batch_seconds=600, timeout=150, workers=4, shared_concurrency=4,
        reuse_trusted_from=[], evidence_bank=None, reference_package=None, replay_root=tmp_path/'previous')
    item = {'model_id': 'luna', 'collection': str(tmp_path/'collection')}
    assert '--replay-from' not in sequence.command(args, item)
    (args.replay_root/'luna').mkdir(parents=True)
    cmd = sequence.command(args, item)
    assert cmd[cmd.index('--replay-from')+1] == str((args.replay_root/'luna').resolve())
    args.review_repairs_root = tmp_path/'repairs'
    assert '--review-repairs' not in sequence.command(args, item)
    (args.review_repairs_root/'luna').mkdir(parents=True)
    cmd = sequence.command(args, item)
    assert cmd[cmd.index('--review-repairs')+1] == str((args.review_repairs_root/'luna').resolve())
    args.structured_output=True
    assert '--structured-output' in sequence.command(args,item)

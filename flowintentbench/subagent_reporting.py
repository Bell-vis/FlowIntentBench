"""Read-only reports over the complete frozen collaboration slot ledger.

The ledger owns requested observations. File judgments supply scientific scores;
this module only validates identity, replays stored calculations and aggregates.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .evaluation_metrics import BenchmarkMetricValues, SCIENTIFIC_METRIC_NAMES, benchmark_summary
from .evaluator import CaseEvaluationRecord, TrialEvaluation, aggregate_trials_for_case
from .quality_efficiency import EFFICIENCY_FIELDS, QUALITY_POLICY_SHA256, quality_efficiency_summary
from scripts.path_migration import resolve_repository_path

CONDITIONS = {"O1-F1", "O2-F1", "O3-F1", "O1-F2"}


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _resolve(path, root):
    return resolve_repository_path(
        path,
        repository_root=Path(__file__).resolve().parents[1],
        relative_root=root,
    )


def _ledger(collection_root):
    state = _read(collection_root / "collection_state.json")
    config = state["configuration"]
    # The collector freezes the default JSON separators; file-evaluator
    # response and run identities use compact separators in _digest instead.
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if state.get("configuration_sha256") is not None and state["configuration_sha256"] != config_hash:
        raise ValueError("collection configuration hash mismatch")
    if config.get("quality_policy_sha256", QUALITY_POLICY_SHA256) != QUALITY_POLICY_SHA256:
        raise ValueError("frozen collection quality policy differs from reporting policy")
    manifest_path = _resolve(config["manifest_path"], collection_root)
    if config.get("manifest_sha256") != _sha(manifest_path):
        raise ValueError("collection manifest hash mismatch")
    manifest = _read(manifest_path)
    cases = {case["case_id"]: case for case in manifest["cases"]}
    models = config["models"]
    slots = state["slots"]
    if (len(cases) != 96 or len(manifest["cases"]) != 96 or len(models) != 2
            or len(set(models)) != 2 or config.get("repetitions") != 3 or len(slots) != 576):
        raise ValueError("report requires exactly 96 cases × 2 models × 3 trials = 576 slots")
    if set(config["cases"]) != set(cases):
        raise ValueError("ledger and manifest case inventories differ")
    expected = {(model, case, trial) for model in models for case in cases for trial in (1, 2, 3)}
    actual = [(slot["model_id"], slot["case_id"], slot["trial_index"]) for slot in slots]
    if (any(type(s["trial_index"]) is not int for s in slots) or set(actual) != expected
            or len(set(actual)) != 576 or len({s["slot_id"] for s in slots}) != 576):
        raise ValueError("duplicate, missing or foreign requested slot")
    if any(case["condition"] not in CONDITIONS for case in cases.values()):
        raise ValueError("unknown scientific condition")
    return state, cases


def current_slot_run_paths(collection_root):
    """Select only current ledger-bound attempts, retaining legacy roots."""
    root = Path(collection_root).resolve()
    if not (root / "collection_state.json").exists():
        return sorted(root.rglob("run_record.json"))
    state, _ = _ledger(root)
    paths = []
    for slot in state["slots"]:
        if slot.get("run_record_path"):
            path = _resolve(slot["run_record_path"], root)
            if _sha(path) != slot.get("run_record_sha256"):
                raise ValueError("current attempt run record hash mismatch")
            paths.append(path)
    return paths


def _infrastructure_history(state, root):
    records = state.get("infra_history", [])
    for history in records:
        content = {key: value for key, value in history.items() if key != "history_sha256"}
        expected = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if history.get("history_sha256") != expected:
            raise ValueError("infrastructure history hash mismatch")
        path = root / "infra_history" / f"{history['failed_slot']['slot_id']}.json"
        if _read(path) != history:
            raise ValueError("infrastructure history audit copy differs from ledger")
    return records


def build_requested_quality_rows(collection_root, evaluation_root):
    """Return exactly 576 rows, preserving unstarted and unevaluated slots."""
    collection_root, evaluation_root = Path(collection_root).resolve(), Path(evaluation_root).resolve()
    state, cases = _ledger(collection_root)
    history = _infrastructure_history(state, collection_root)
    historical_runs = {item["run_id"] for item in history if item.get("run_id")}
    historical_artifacts = {
        _digest({"run_id": item["run_id"], "source_sha256": item["failed_slot"]["run_record_sha256"]}): item["run_id"]
        for item in history if item.get("run_id")
    }
    imported, expected_artifacts = {}, {}
    for slot in state["slots"]:
        if not slot.get("run_record_path"):
            continue
        path = _resolve(slot["run_record_path"], collection_root)
        source_hash = _sha(path)
        if source_hash != slot.get("run_record_sha256"):
            raise ValueError("imported run record hash mismatch")
        run = _read(path)
        for key, expected in (("case_id", slot["case_id"]), ("model_id", slot["model_id"]),
                              ("trial_index", slot["trial_index"]), ("experiment_id", state["experiment_id"]),
                              ("run_status", slot["status"])):
            if run.get(key) != expected:
                raise ValueError(f"run/slot {key} mismatch")
        if run["run_id"] in imported:
            raise ValueError("duplicate imported run identity")
        imported[run["run_id"]] = (slot, run)
        expected_artifacts[_digest({"run_id": run["run_id"], "source_sha256": source_hash})] = run["run_id"]
    evaluations = {}
    index_path = evaluation_root / "evaluation_index.json"
    index = _read(index_path) if index_path.exists() else {"results": []}
    for entry in index["results"]:
        path = _resolve(entry["evaluation_path"], evaluation_root)
        payload = _read(path)
        run_id = payload.get("run_id") or expected_artifacts.get(path.parent.name) or historical_artifacts.get(path.parent.name)
        if run_id in historical_runs:
            continue
        if run_id not in imported:
            raise ValueError("evaluation is not bound to an imported ledger run")
        if run_id in evaluations:
            raise ValueError("duplicate evaluation for a requested run")
        slot, run = imported[run_id]
        if payload.get("case_id") != slot["case_id"] or entry.get("case_id") != slot["case_id"]:
            raise ValueError("evaluation/slot case mismatch")
        if payload.get("trial_index", slot["trial_index"]) != slot["trial_index"]:
            raise ValueError("evaluation/slot trial mismatch")
        if payload.get("response_sha256") is not None and payload["response_sha256"] != _digest(run.get("final_response")):
            raise ValueError("evaluation response hash mismatch")
        if entry.get("status") != payload.get("status"):
            raise ValueError("evaluation index status mismatch")
        evaluations[run_id] = payload
    by_slot = {slot["slot_id"]: run for slot, run in imported.values()}
    rows = []
    for slot in state["slots"]:
        condition = cases[slot["case_id"]]["condition"]
        awaiting_agent = (slot["status"] == "RUNNING" and "agent_id" in slot
                          and slot["agent_id"] is None)
        row = {"model": slot["model_id"], "case_id": slot["case_id"], "trial": slot["trial_index"],
               "experiment_id": state["experiment_id"], "configuration_sha256": state.get("configuration_sha256"),
               "slot_id": slot["slot_id"], "condition": condition, "collection_status": slot["status"],
               "status": "NOT_STARTED" if slot["status"] == "PENDING" or awaiting_agent else slot["status"],
               "dispatch_awaiting_agent": awaiting_agent,
               "_infra_history": history,
               "_scientific_terminal": False, **{name: None for name in SCIENTIFIC_METRIC_NAMES},
               **{name: None for name in EFFICIENCY_FIELDS}}
        run = by_slot.get(slot["slot_id"])
        if run is not None:
            row.update({name: run.get(name) for name in EFFICIENCY_FIELDS})
            # Platform usage is never inferred from text, helper counts or time.
            if row["input_tokens"] is not None and row["output_tokens"] is not None:
                row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
            else:
                row["total_tokens"] = None
            row["run_id"] = run["run_id"]
            row["status"] = run["run_status"] if run["run_status"] != "COMPLETED" else "PENDING"
            payload = evaluations.get(run["run_id"])
            if payload is not None:
                row["evaluation_status"] = payload["status"]
                row["pending_type"] = payload.get("pending_type")
                if payload["status"] == "SCORED":
                    record = CaseEvaluationRecord.from_dict(payload["evaluation_record"])
                    from .metric_diagnostics import record_diagnostics
                    row["metric_diagnostics"] = record_diagnostics(payload["evaluation_record"])
                    result = record.replay()
                    if (record.run_id != run["run_id"] or record.trial_index != slot["trial_index"]
                            or result.case_id != slot["case_id"] or result.condition != condition
                            or result.run_status.value != run["run_status"]
                            or record.experiment_id != state["experiment_id"]):
                        raise ValueError("scored evaluation identity mismatch")
                    if payload.get("evaluation_manifest_digest") != record.evaluation_manifest_digest:
                        raise ValueError("scored evaluation manifest digest mismatch")
                    if run.get("target_fingerprint") is not None and record.target_fingerprint != run["target_fingerprint"]:
                        raise ValueError("scored evaluation target mismatch")
                    row["evaluation_manifest_digest"] = record.evaluation_manifest_digest
                    if result.metrics is not None and result.eligible_for_scientific_aggregation:
                        metrics = result.metrics.to_dict()
                        for block in ("scientific_operationalization", "scientific_findings", "o_f_consistency"):
                            row.update(metrics[block])
                        consistency = result.metrics.o_f_consistency
                        terminal = (consistency.c_score is not None or consistency.operationalization_determinate is False
                                    or consistency.unavailable_reason == "NO_APPLICABLE_FINDINGS")
                        row["_scientific_terminal"] = terminal
                        row["status"] = result.run_status.value if terminal else "PENDING"
                        row["_result"] = result
        elif slot["status"] not in {"PENDING", "RUNNING", "FINISHED"}:
            raise ValueError("terminal collection slot has no imported run record")
        if row["status"] == "FINISHED":
            row["status"] = "PENDING"
        rows.append(row)
    return rows


def rows_for_evaluator_snapshot(rows, snapshot):
    """Keep every requested slot; withhold other versions only in derived views.

    None selects no scientific version. Original trial rows, infrastructure
    outcomes and raw timing remain intact. No stored evaluation is rewritten.
    """
    selected = []
    for row in rows:
        other = (row.get('evaluation_manifest_digest') is not None
                 and row['evaluation_manifest_digest'] != snapshot)
        if other and row['status'] in {'COMPLETED', 'MODEL_NONCOMPLETION', 'PENDING'}:
            row = dict(row, status='PENDING', _scientific_terminal=False,
                       evaluation_status='WITHHELD_OTHER_EVALUATOR_SNAPSHOT',
                       metric_diagnostics=None,
                       **{name: None for name in SCIENTIFIC_METRIC_NAMES})
            row.pop('_result', None)
        selected.append(row)
    return selected


def _summary_views(rows):
    from .metric_diagnostics import metric_validity_summary
    quality = quality_efficiency_summary(rows)
    for model, groups in quality['by_model'].items():
        for group, selected in [(groups['overall'], [r for r in rows if r['model'] == model]),
                                *((block, [r for r in rows if r['model'] == model and r['condition'] == condition])
                                  for condition, block in groups['by_condition'].items())]:
            group['requested_slots_scientifically_complete'] = all(r['_scientific_terminal'] for r in selected)
            if not group['requested_slots_scientifically_complete']:
                group['quality_pass_rate'] = None
    return dict(scientific_by_model={model: _scientific_summary([r for r in rows if r['model'] == model])
                                   for model in sorted({r['model'] for r in rows})},
                quality_efficiency=quality, metric_validity_diagnostics=metric_validity_summary(rows))


def _scientific_summary(rows):
    complete_cases = []
    for case_id in sorted({row["case_id"] for row in rows}):
        trials = [row for row in rows if row["case_id"] == case_id]
        if all(row["_scientific_terminal"] for row in trials):
            complete_cases.append(aggregate_trials_for_case(tuple(
                TrialEvaluation(row["trial"], row["_result"]) for row in trials
            )).metrics)

    def collect(selected_rows, selected_cases):
        def scores(name):
            return tuple(getattr(case, name) for case in selected_cases if getattr(case, name) is not None)
        requested = {row["case_id"]: row["condition"] for row in selected_rows}
        counts = {name: len(requested) for name in SCIENTIFIC_METRIC_NAMES}
        counts["urs"] = sum(c.startswith(("O2", "O3")) for c in requested.values())
        counts["resolved_o_compliance"] = sum(not c.startswith("O3") for c in requested.values())
        counts["core_finding_recall"] = sum(c.endswith("F1") for c in requested.values())
        counts["adequate_core_complete"] = sum(c.endswith("F2") for c in requested.values())
        for name in ("c_score", "branch_alignment"):
            counts[name] -= sum(c.metric_trial_denominators[name]["applicable_trial_n"] == 0 for c in selected_cases)
        return BenchmarkMetricValues(
            o_scores=scores("o_score"), urs_scores=scores("urs"),
            resolved_o_compliance_scores=scores("resolved_o_compliance"),
            finding_precision_scores=scores("finding_precision"),
            core_finding_recall_scores=scores("core_finding_recall"),
            finding_requirement_recall_scores=scores("finding_requirement_recall"),
            adequate_core_complete_scores=scores("adequate_core_complete_rate"),
            c_scores=scores("c_score"), branch_alignment_scores=scores("branch_alignment"),
            metric_applicable_counts=counts,
        )
    by_condition = {condition: [row for row in rows if row["condition"] == condition]
                    for condition in sorted({row["condition"] for row in rows})}
    result = benchmark_summary(collect(rows, complete_cases), by_condition={
        condition: collect(group, [c for c in complete_cases if c.condition == condition])
        for condition, group in by_condition.items()
    })
    for block, group in [(result["overall"], rows), *((result["by_condition"][key], group) for key, group in by_condition.items())]:
        block["requested_trial_count"] = len(group)
        block["scientifically_terminal_trial_count"] = sum(r["_scientific_terminal"] for r in group)
        block["requested_case_count"] = len({r["case_id"] for r in group})
        block["finalized_n3_case_count"] = sum(c.case_id in {r["case_id"] for r in group} for c in complete_cases)
        block["mean_scope"] = "Finalized N3 cases only; incomplete requested cases remain in metric denominators"
    result["case_aggregates"] = [case.to_dict() for case in complete_cases]
    return result


def report_subagent_experiment(collection_root, evaluation_root):
    rows = build_requested_quality_rows(collection_root, evaluation_root)
    complete = all(row["_scientific_terminal"] for row in rows)
    snapshots = sorted({r['evaluation_manifest_digest'] for r in rows if r.get('evaluation_manifest_digest')})
    scientific_snapshots = sorted({r['evaluation_manifest_digest'] for r in rows if r['_scientific_terminal']})
    mixed_science = len(scientific_snapshots) > 1
    # Infrastructure-only records from a newer evaluator do not invalidate
    # existing scientific results. Multiple scientific versions are reported
    # separately, never silently averaged or made into a mixed-version N=3.
    summary_rows = rows_for_evaluator_snapshot(rows, None) if mixed_science else rows
    # Use the same ledger snapshot that supplied the current slots, so a
    # concurrent recovery cannot count an attempt as both current and retired.
    history = rows[0]["_infra_history"]
    history_rows = [dict(model=h["logical_trial"][0], case_id=h["logical_trial"][1],
                         trial=h["logical_trial"][2], condition=next(r["condition"] for r in rows if r["case_id"] == h["logical_trial"][1]),
                         status="INFRASTRUCTURE_INVALID", **h["efficiency"]) for h in history]
    history_report = quality_efficiency_summary(history_rows)
    return {
        "artifact_type": "SubagentExperimentReport", "formal_release": False,
        "experiment_id": rows[0]["experiment_id"], "configuration_sha256": rows[0]["configuration_sha256"],
        "status": "COMPLETE" if complete else "INCOMPLETE", "requested_slot_count": 576,
        "scientifically_terminal_slot_count": sum(row["_scientific_terminal"] for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "trial_rows": [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows],
        **_summary_views(summary_rows),
        "mixed_evaluator_snapshots": len(snapshots) > 1,
        "mixed_scientific_evaluator_snapshots": mixed_science,
        "scientific_aggregation_policy": "Separate evaluator snapshots; all requested slots retained in each denominator; no mixed-version N3 or quality-qualified pooling",
        "evaluation_snapshots": {snapshot: dict(
            scored_record_count=sum(r.get('evaluation_manifest_digest') == snapshot for r in rows),
            scientifically_terminal_slot_count=sum(r.get('evaluation_manifest_digest') == snapshot and r['_scientific_terminal'] for r in rows),
            status_counts=dict(Counter(r['status'] for r in rows if r.get('evaluation_manifest_digest') == snapshot)))
            for snapshot in snapshots},
        "by_evaluator_snapshot": {snapshot: _summary_views(rows_for_evaluator_snapshot(rows, snapshot))
                                  for snapshot in scientific_snapshots} if mixed_science else {},
        "infra_history": {"attempt_count": len(history), "attempts": history,
                          "by_model": {model: block["overall"]["infrastructure_efficiency"]
                                       for model, block in history_report["by_model"].items()}},
        "api_calls": 0,
    }

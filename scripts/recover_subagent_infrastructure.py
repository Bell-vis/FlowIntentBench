#!/usr/bin/env python3
"""Create a bounded replacement attempt for a documented infrastructure failure."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import collect_subagent_runs as collector
from flowintentbench.subagent_reporting import _ledger
from flowintentbench.quality_efficiency import EFFICIENCY_FIELDS

POLICY_VERSION = "collaboration-infrastructure-recovery-v1"
MAX_RECOVERIES_PER_LOGICAL_TRIAL = 3

AUTO_RECOVERY_POLICY = {
    "version": "imported-infrastructure-requeue-v1",
    "eligible_status": "INFRASTRUCTURE_INVALID",
    "max_recoveries_per_logical_trial": MAX_RECOVERIES_PER_LOGICAL_TRIAL,
    "original_attempts": "Preserve all artifacts and efficiency in infra_history; new opaque slot for the same logical trial",
    "unknown_http_outcome": "Never resend the ambiguous HTTP request; only a terminal imported infrastructure attempt may receive an audited replacement",
    "excluded": "Completed/model-noncompletion observations, unfinished or quarantined claims, unverified artifacts",
    "extensions": "Only explicit per-trial audited authorizations; restart never resets the limit",
}


def recovery_limit(output, logical):
    path = Path(output) / 'infrastructure_recovery_authorizations.json'
    if not path.exists():
        return MAX_RECOVERIES_PER_LOGICAL_TRIAL
    document = collector.read(path)
    body = {k: v for k, v in document.items() if k != 'sha256'}
    if document.get('sha256') != collector.digest(body):
        raise ValueError('Recovery authorization digest mismatch')
    for row in document['trials']:
        if tuple(row['logical_trial']) == tuple(logical):
            if collector.file_hash(Path(output) / row['run_record_path']) != row['run_record_sha256']:
                raise ValueError('Authorized infrastructure evidence changed')
            limit = row['max_recoveries']
            if not isinstance(limit, int) or not 3 <= limit <= 6 or not document.get('reason'):
                raise ValueError('Invalid bounded recovery authorization')
            return limit
    return MAX_RECOVERIES_PER_LOGICAL_TRIAL


def authorize_exhausted_recovery(output, *, reason):
    """Offline maintenance only; authorize three more attempts, never dispatch."""
    if not reason.strip():
        raise ValueError('An explicit amendment reason is required')
    output = Path(output)
    with collector.locked(output):
        state, _ = _ledger(output)
        if any(s['status'] in {'RUNNING', 'FINISHED'} for s in state['slots']):
            raise ValueError('Drain active trials before authorizing replacements')
        path = output / 'infrastructure_recovery_authorizations.json'
        if path.exists():
            return collector.read(path)  # Idempotent; cannot grant again on restart.
        rows = []
        for slot in state['slots']:
            logical = (slot['model_id'], slot['case_id'], slot['trial_index'])
            prior = sum(tuple(h['logical_trial']) == logical for h in state.get('infra_history', []))
            if slot['status'] != 'INFRASTRUCTURE_INVALID' or prior < 3:
                continue
            record = output / slot['run_record_path']
            if collector.file_hash(record) != slot['run_record_sha256']:
                raise ValueError('Infrastructure record hash mismatch')
            claim = output / 'task_claims/solver' / (slot['slot_id'] + '.json')
            if claim.exists() and collector.read(claim).get('status') != 'FINISHED':
                raise ValueError('Unfinished claim requires review')
            rows.append(dict(logical_trial=list(logical), failed_slot_id=slot['slot_id'],
                run_record_path=slot['run_record_path'], run_record_sha256=slot['run_record_sha256'],
                max_recoveries=min(6, prior + 3)))
        document = dict(version='bounded-recovery-amendment-v1', reason=reason, epoch=time.time(), trials=rows)
        document['sha256'] = collector.digest(document)
        collector.write(path, document)
        return document


def queue_imported_infrastructure(output, *, slot_ids=None):
    """Caller holds the console publication barrier and exclusive runner lock.

    Admission/cooldown stays in the collector. This only creates audited
    PENDING replacements; it cannot invoke a provider or extend a trial budget.
    """
    output = Path(output)
    state = collector.read(output / "collection_state.json")
    selected = None if slot_ids is None else set(slot_ids)
    candidates = [s for s in state["slots"] if s["status"] == "INFRASTRUCTURE_INVALID"
                  and (selected is None or s["slot_id"] in selected)]
    rows = []
    for slot in candidates:
        row = dict(slot_id=slot["slot_id"], model_id=slot["model_id"],
                   case_id=slot["case_id"], trial_index=slot["trial_index"])
        logical = (slot["model_id"], slot["case_id"], slot["trial_index"])
        prior = sum(tuple(h["logical_trial"]) == logical for h in state.get("infra_history", []))
        limit = recovery_limit(output, logical)
        if prior >= limit:
            row.update(status="LIMIT_REACHED", reason=f"Audited replacement limit reached: {limit}")
        else:
            claim_path = output / "task_claims" / "solver" / (slot["slot_id"] + ".json")
            try:
                if claim_path.exists() and collector.read(claim_path).get("status") != "FINISHED":
                    raise ValueError("Unfinished or quarantined solver claim requires receipt review")
                result = recover_infrastructure_attempt(output, slot["slot_id"])
                row.update(status="QUEUED", **result)
            except (ValueError, OSError) as exc:
                # A bad artifact must not be sampled again, nor block the other
                # model's verified tasks. Keep the original record and reason.
                row.update(status="BLOCKED_REQUIRES_REVIEW", reason=str(exc))
        rows.append(row)
    # Merge by failed slot so repeated resume/empty scans do not hide blockers.
    path = output / "infrastructure_recovery_queue.json"
    previous = collector.read(path).get("results", []) if path.exists() else []
    merged = {r["slot_id"]: r for r in previous}
    merged.update({r["slot_id"]: r for r in rows})
    current = collector.read(output / "collection_state.json")
    invalid_ids = {s["slot_id"] for s in current["slots"] if s["status"] == "INFRASTRUCTURE_INVALID"}
    summary = dict(updated_epoch=time.time(), policy=AUTO_RECOVERY_POLICY,
                   results=list(merged.values()), last_scan_results=rows,
                   current_blocked_slot_ids=sorted(invalid_ids & {
                       sid for sid, r in merged.items() if r["status"] != "QUEUED"}))
    collector.write(path, summary)
    return summary


def _spawn_evidence(output, slot):
    path = output / "orchestrator_events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    matches = [event for event in events
               if event.get("event") == "SPAWN_INFRASTRUCTURE_FAILURE"
               and event.get("slot_id") == slot["slot_id"]
               and event.get("model_id") == slot["model_id"]
               and event.get("agent_id") is None]
    if not matches:
        raise ValueError("undispatched attempt requires SPAWN_INFRASTRUCTURE_FAILURE evidence")
    if any(event.get("slot_id") == slot["slot_id"] and event.get("agent_id") is not None for event in events):
        raise ValueError("event history records an attached agent; cannot recover as undispatched")
    return {"path": str(path.relative_to(output)), "sha256": collector.file_hash(path), "events": matches}


def recover_infrastructure_attempt(output, slot_id):
    """Replace one current opaque attempt; never resample a model observation."""
    with collector.locked(output) as output:
        state, _ = _ledger(output)
        # Reissuing the same request cannot create another replacement.
        previous = next((h for h in state.get("infra_history", []) if h["failed_slot"]["slot_id"] == slot_id), None)
        if previous is not None:
            return {"already_recovered": True, "replacement_slot_id": previous["replacement_slot_id"]}
        slot = collector.get_slot(state, slot_id)
        collector.verify(state, slot)
        history_path = output / "infra_history" / f"{slot_id}.json"
        if history_path.exists():
            # Finish an interrupted ledger switch using its already staged
            # replacement, without minting another attempt or overwriting it.
            saved = collector.read(history_path)
            content = {key: value for key, value in saved.items() if key != "history_sha256"}
            if saved.get("history_sha256") != collector.digest(content) or saved["failed_slot"] != slot:
                raise ValueError("interrupted recovery audit does not match current attempt")
            replacement = saved["replacement_slot"]
            collector.verify(state, replacement)
            state.setdefault("infra_history", []).append(saved)
            state["slots"][state["slots"].index(slot)] = replacement
            collector.save_state(output, state)
            _ledger(output)
            return {"already_recovered": False, "resumed_recovery": True,
                    "replacement_slot_id": replacement["slot_id"]}
        config = state["configuration"]
        if (config["timeout_seconds"] != collector.SOLVER_TIMEOUT_SECONDS
                or config["max_python_executions"] != collector.SOLVER_MAX_PYTHON_EXECUTIONS):
            raise ValueError(
                "recovery requires the configured solver budget "
                f"{collector.SOLVER_TIMEOUT_SECONDS}-second/"
                f"{collector.SOLVER_MAX_PYTHON_EXECUTIONS}-execution budget")
        logical = (slot["model_id"], slot["case_id"], slot["trial_index"])
        prior = [h for h in state.get("infra_history", [])
                 if tuple(h["logical_trial"]) == logical]
        limit = recovery_limit(output, logical)
        if len(prior) >= limit:
            raise ValueError("bounded infrastructure recovery limit reached")
        public = Path(state["public_root"])
        old_work = public / "work" / slot_id
        evidence, run = None, None
        if slot["status"] == "INFRASTRUCTURE_INVALID" and slot.get("run_record_path"):
            path = output / slot["run_record_path"]
            if collector.file_hash(path) != slot.get("run_record_sha256"):
                raise ValueError("imported infrastructure run record hash mismatch")
            run = collector.read(path)
            for key, value in (("run_status", "INFRASTRUCTURE_INVALID"), ("model_id", slot["model_id"]),
                               ("case_id", slot["case_id"]), ("trial_index", slot["trial_index"]),
                               ("experiment_id", state["experiment_id"])):
                if run.get(key) != value:
                    raise ValueError(f"imported infrastructure run {key} mismatch")
            reason = "IMPORTED_INFRASTRUCTURE_INVALID"
            events_path = output / "orchestrator_events.jsonl"
            if events_path.exists():
                events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
                evidence = {"path": str(events_path.relative_to(output)), "sha256": collector.file_hash(events_path),
                            "events": [event for event in events if event.get("slot_id") == slot_id]}
        elif (slot["status"] in {"PENDING", "RUNNING"} and "agent_id" in slot
              and slot["agent_id"] is None and not slot.get("run_record_path")):
            evidence = _spawn_evidence(output, slot)
            reason = "UNDISPATCHED_SPAWN_INFRASTRUCTURE_FAILURE"
        else:
            raise ValueError("only imported infrastructure failures or evidenced undispatched attempts may recover")
        with (old_work / ".execution.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if reason == "UNDISPATCHED_SPAWN_INFRASTRUCTURE_FAILURE":
                if any((directory / name).exists() or (directory / name).is_symlink()
                       for directory in (old_work, output / "collected" / slot_id)
                       for name in ("answer.md", "execution_journal.jsonl")):
                    raise ValueError("undispatched attempt contains answer or execution journal")
            # Create new paths exclusively; old work, bundles, prompt, runs,
            # snapshots and the original event log are never overwritten.
            new_id = uuid4().hex
            replacement = {key: deepcopy(slot[key]) for key in (
                "model_id", "case_id", "dataset_id", "trial_index", "ordering_seed",
                "case_execution_position", "raw_files")}
            replacement.update(slot_id=new_id, status="PENDING", agent_id=None, run_record_path=None,
                               recovery_attempt_index=len(prior) + 1, previous_attempt_slot_id=slot_id)
            bundle, work = public / "bundles" / new_id, public / "work" / new_id
            bundle.mkdir()
            work.mkdir()
            shutil.copyfile(public / "bundles" / slot_id / "case_input.json", bundle / "case_input.json")
            (bundle / "case_input.json").chmod(0o444)
            shutil.copyfile(collector.HELPER, work / "run_python.py")
            prompt = collector.prompt_for(replacement, public)
            replacement["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            prompt_path = output / "prompts" / f"{new_id}.txt"
            with prompt_path.open("x") as stream:
                stream.write(prompt)
            collector.verify(state, replacement)
            history = {"policy_version": POLICY_VERSION, "max_recoveries_per_logical_trial": limit,
                       "logical_trial": list(logical), "failed_slot": deepcopy(slot),
                       "replacement_slot": deepcopy(replacement),
                       "replacement_slot_id": new_id, "retired_epoch": time.time(), "reason": reason,
                       "orchestrator_event_evidence": evidence, "run_id": None if run is None else run["run_id"],
                       "efficiency": {name: None if run is None else run.get(name) for name in EFFICIENCY_FIELDS}}
            history["reservation_elapsed_seconds"] = (
                max(0., history["retired_epoch"] - slot["started_epoch"])
                if slot.get("started_epoch") is not None else None
            )
            history["history_sha256"] = collector.digest(history)
            # Independent append-only audit copy precedes the atomic ledger
            # switch. An interrupted write leaves only unreferenced new paths.
            history_path.parent.mkdir(exist_ok=True)
            with history_path.open("x") as stream:
                json.dump(history, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            state.setdefault("infra_history", []).append(history)
            state["slots"][state["slots"].index(slot)] = replacement
            collector.save_state(output, state)
            _ledger(output)  # assert all 576 logical trials remain present
            return {"already_recovered": False, "replacement_slot_id": new_id,
                    "logical_trial": list(logical), "recovery_attempt_index": len(prior) + 1,
                    "configuration_sha256": state["configuration_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--slot-id", required=True)
    args = parser.parse_args()
    print(json.dumps(recover_infrastructure_attempt(args.collection_root, args.slot_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

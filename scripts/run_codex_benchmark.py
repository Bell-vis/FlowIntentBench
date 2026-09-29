#!/usr/bin/env python3
"""Console entrypoint for the existing 96-case collector and scientific evaluator."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from collections import Counter
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading

try:
    from scripts.portable_fcntl import fcntl, process_exists
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from portable_fcntl import fcntl, process_exists

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_subagent_runs as collector
from scripts.codex_console_transport import run_codex, parse_events
from scripts.report_subagent_experiment import render_markdown
from flowintentbench.model_runner import RunRecord
from flowintentbench.subagent_reporting import report_subagent_experiment, current_slot_run_paths

DEFAULT_OUTPUT = ROOT / "outputs/expansion96_n3_subagents"
CLI_PROTOCOL = "CODEX_CLI_STAGED_INSTRUCTIONS_V1"
PUBLICATION_LOCK = threading.RLock()
REPLAY_CACHE = None
REPLAY_REUSE_INDEXES = {}
HYBRID = None


def publication_transaction(function):
    @wraps(function)
    def locked(*args, **kwargs):
        with PUBLICATION_LOCK:
            return function(*args, **kwargs)
    return locked


def emit(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def replay_source_fingerprint():
    """Fingerprint code that can change a local evaluation result.

    The fingerprint is deliberately independent of answer data and judge
    responses. Those are validated per result before a cached replay is used.
    """
    files = []
    for root_name in ("flowintentbench", "scripts"):
        root = ROOT / root_name
        files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    files.append(ROOT / "datasets/expansion_v1/case_manifest.json")
    entries = [(str(path.relative_to(ROOT)), collector.file_hash(path))
               for path in sorted(set(files)) if path.is_file()]
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


@contextmanager
def runner_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".console_runner.lock"
    lock = lock_path.open("a")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # A killed sandbox can leave the advisory lock inode held while
            # the recorded runner PID is already gone.  Reusing that inode
            # would make every resumable evaluation fail before doing any
            # work.  Rotate it only after proving the recorded owner is dead;
            # a live owner still gets the original mutual-exclusion error.
            marker_path = output / "console_process.json"
            stale = False
            try:
                marker = json.loads(marker_path.read_text())
                pid = int(marker.get("pid", 0))
                if not process_exists(pid):
                    raise ProcessLookupError(pid)
            except (FileNotFoundError, json.JSONDecodeError, ValueError, ProcessLookupError, PermissionError):
                stale = True
            if stale:
                lock.close()
                stale_path = output / ".console_runner.lock.stale"
                try:
                    lock_path.replace(stale_path)
                except FileNotFoundError:
                    pass
                lock = lock_path.open("a")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    raise RuntimeError("Stale console runner lock could not be recovered") from None
            else:
                lock.close()
                raise RuntimeError("Another console runner owns this experiment") from None
        marker = dict(pid=os.getpid(), started_epoch=time.time(), status="RUNNING")
        collector.write(output / "console_process.json", marker)
        try:
            yield
        except Exception as exc:
            marker.update(exit_kind='ERROR', error_type=type(exc).__name__, error_message=str(exc))
            collector.write(output / 'console_logs' / f'runner_failure_{time.time_ns()}.json',
                            dict(marker, failed_epoch=time.time()))
            raise
        finally:
            collector.write(output / "console_process.json", dict(marker, status="STOPPED", stopped_epoch=time.time()))
    finally:
        lock.close()


def require_chatgpt_login():
    result = subprocess.run(["codex", "login", "status"], capture_output=True, text=True, timeout=15)
    if result.returncode or "Logged in using ChatGPT" not in result.stdout + result.stderr:
        raise RuntimeError("This runner requires `codex login` with ChatGPT; API-key login is not used")


def prepare(output):
    sidecar = output / "console_protocol.json"
    ledger = output / "collection_state.json"
    state = collector.prepare(manifest_path=ROOT / "datasets/expansion_v1/case_manifest.json",
                              datasets_root=ROOT / "datasets", output=output,
                              seed=20260914, reasoning_effort="xhigh")
    protocol = dict(protocol=CLI_PROTOCOL, transport="codex_exec_chatgpt_login",
                    configuration_sha256=state["configuration_sha256"],
                    cli_version=(collector.read(sidecar).get('cli_version') if sidecar.exists()
                        and HYBRID is not None and HYBRID.cli_disabled and not HYBRID.cli_solvers
                        else subprocess.check_output(["codex", "--version"], text=True).strip()),
                    runner_sha256=collector.file_hash(Path(__file__)),
                    transport_sha256=collector.file_hash(ROOT / "scripts/codex_console_transport.py"),
                    scheduling_source_hashes={name: collector.file_hash(ROOT / 'scripts' / name) for name in
                        ('console_evaluation_cache.py', 'carry_forward_file_judgments.py', 'run_codex_file_judgments.py',
                         'grading_policy.py', 'batched_file_judgments.py', 'lean_judgment_scheduler.py')},
                    models=list(collector.MODELS), repetitions=3,
                    timeout_seconds=collector.SOLVER_TIMEOUT_SECONDS,
                    max_python_executions=collector.SOLVER_MAX_PYTHON_EXECUTIONS, project_api_calls=0,
                    network_isolation=False, direct_api_fallback=False,
                    target_identity_policy="Preserve frozen model target identifiers; record actual launch transport in runtime provenance",
                    amendment="Native collaboration to fresh codex exec continuation; mixed launch environments must be disclosed")
    if (output / 'hybrid_protocol.json').exists():
        protocol.update(transport='mixed_cli_and_third_party_api',
                        direct_api_fallback='explicit_quota_rejection_only',
                        project_api_calls='recorded_per_http_attempt')
        protocol['scheduling_source_hashes'].update({name: collector.file_hash(ROOT / 'scripts' / name)
            for name in ('benchmark_hybrid.py', 'third_party_benchmark_transport.py', 'shared_api_admission.py')})
    if sidecar.exists():
        previous = collector.read(sidecar)
        if any(previous.get(key) != value for key, value in protocol.items()):
            raise ValueError("CLI protocol/code/version changed: do not mix execution conditions")
    else:
        protocol["adopted_epoch"] = time.time()
        protocol["existing_ledger_sha256"] = collector.file_hash(ledger)
        protocol["existing_run_hashes"] = {s["slot_id"]: s["run_record_sha256"] for s in state["slots"] if s.get("run_record_sha256")}
        collector.write(sidecar, protocol)
    return state


@publication_transaction
def import_cli(output, slot_id, receipt):
    """Use frozen validation/target identities and disclose the real launch transport.

    Only newly finished CLI slots enter here. Existing native records are never
    rewritten. The protocol amendment preserves the model target, not a claim of
    identical tool environments. Runtime provenance and report disclose the split.
    """
    slot = collector.import_slot(output, slot_id)
    with collector.locked(output):
        state = collector.read(output / "collection_state.json")
        slot = collector.get_slot(state, slot_id)
        run_path = output / slot["run_record_path"]
        old = collector.read(run_path)
        if old.get("protocol_version") == CLI_PROTOCOL:
            return slot
        provenance = dict(old["runtime_environment_fingerprint"])
        provenance.update(transport="codex_exec_chatgpt_login", codex_thread_id=receipt.get("thread_id"),
                          cli_protocol_sha256=collector.file_hash(output / "console_protocol.json"),
                          cli_receipt_sha256=collector.file_hash(output / "cli_runs" / slot_id / "receipt.json"),
                          events_sha256=collector.file_hash(receipt["events_path"]),
                          tool_call_count_scope="recorded Python helper executions; CLI events stored separately")
        provenance.pop("collaboration_agent_id", None)
        schedule_path = output / 'cli_runs' / slot_id / 'schedule.json'
        if schedule_path.exists():
            provenance['console_schedule'] = collector.read(schedule_path)
            provenance['console_schedule_sha256'] = collector.file_hash(schedule_path)
        record = RunRecord.from_dict(old)
        record.runtime_environment_fingerprint = provenance
        record.trajectory = json.loads((run_path.parent / "trajectory.json").read_text())
        # Keep the frozen journal envelope expected by execution_evidence.
        # Actual launch transport remains explicit in the provenance payload.
        record.trajectory[0] = {"event": "collaboration_provenance", **provenance}
        record.protocol_version = CLI_PROTOCOL
        record.runtime_profile_id = "codex-cli-staged-python-720s-48-v2"
        usage = receipt.get("usage") or {}
        if receipt.get("completed"):
            record.input_tokens = usage.get("input_tokens")
            record.output_tokens = usage.get("output_tokens")
        record.write_trajectory(run_path.parent / "trajectory.json")
        record.write_json(run_path)
        slot["run_record_sha256"] = collector.file_hash(run_path)
        slot["console_transport"] = "codex_exec_chatgpt_login"
        collector.save_state(output, state)
        return slot


def collect_one(output, slot_id):
    spec = collector.start(output, slot_id)
    schedule_path = output / 'console_schedule.json'
    if schedule_path.exists():
        collector.write(output / 'cli_runs' / slot_id / 'schedule.json', collector.read(schedule_path))
    state = collector.read(output / "collection_state.json")
    work = Path(state["public_root"]) / "work" / slot_id
    receipt = run_codex(spec["message"], spec["model"], work, output / "cli_runs" / slot_id,
                        timeout_seconds=max(0.01, spec["deadline_epoch"] - time.time()))
    identity = "codex-cli:" + (receipt.get("thread_id") or "failed-launch:" + slot_id)
    if receipt["timed_out"] and receipt.get("error"):
        outcome, reason = "INFRASTRUCTURE_INVALID", receipt["error"]
    elif receipt["timed_out"]:
        outcome, reason = "MODEL_NONCOMPLETION", "900-second CLI execution budget exceeded"
    elif not receipt["completed"]:
        outcome, reason = "INFRASTRUCTURE_INVALID", receipt.get("error") or "CLI did not emit successful turn completion"
    else:
        outcome, reason = "COMPLETED", None
    collector.finish(output, slot_id, outcome=outcome, reason=reason, agent_id=identity)
    slot = import_cli(output, slot_id, receipt)
    emit(event="collected", model=slot["model_id"], case=slot["case_id"], trial=slot["trial_index"],
         status=slot["status"], slot_id=slot_id)
    return slot


def collect(output, max_runs=576, *, stop_event=None, on_ready=None, on_commit=None):
    if HYBRID is not None:
        return HYBRID.collect(output, max_runs, stop_event=stop_event, on_ready=on_ready, on_commit=on_commit)
    state = prepare(output)
    state = finalize_receipts(output, state)
    if any(x["status"] in {"RUNNING", "FINISHED"} for x in state["slots"]):
        unfinished = [(x['slot_id'], x['status']) for x in state['slots'] if x['status'] in {'RUNNING', 'FINISHED'}]
        raise RuntimeError(f"Unfinished attempt found: {unfinished}. No complete recoverable event log/receipt; inspect cli_runs/<slot>/ and execution journal. No trial was rerun.")
    stop = False
    stop_event = stop_event if stop_event is not None else threading.Event()
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        stop_event.set()
        emit(event="stop_requested", message="Finishing current trials; no new trials will start")
    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    launched = 0
    try:
        if on_ready is not None:
            on_ready()
        with ThreadPoolExecutor(max_workers=2) as pool:
            while not stop and not stop_event.is_set() and not (output / 'STOP_REQUESTED').exists() and launched < max_runs:
                state = collector.read(output / "collection_state.json")
                selected = []
                for model in collector.MODELS:
                    row = next((s for s in state["slots"] if s["model_id"] == model and s["status"] == "PENDING"), None)
                    if row and launched + len(selected) < max_runs:
                        selected.append(row)
                if not selected:
                    break
                futures = [pool.submit(collect_one, output, row["slot_id"]) for row in selected]
                launched += len(futures)
                for future in as_completed(futures):
                    try:
                        row = future.result()
                    except BaseException:
                        # Signal the grader before executor shutdown waits for
                        # the other solver to finish its current trial.
                        stop_event.set()
                        raise
                    if on_commit is not None:
                        on_commit()
                    if row["status"] == "INFRASTRUCTURE_INVALID":
                        stop = True
                        stop_event.set()
                        emit(event="infrastructure_stop", slot_id=row["slot_id"], reason=row.get("failure_reason"))
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return launched


def finalize_receipts(output, state):
    """Finish durable CLI receipts after a controller restart, never rerun a trial."""
    for slot in state["slots"]:
        path = output / "cli_runs" / slot["slot_id"] / "receipt.json"
        if not path.exists():
            if slot["status"] not in {"RUNNING", "FINISHED"}:
                continue
            if not recover_event_receipt(output, state, slot):
                continue
        receipt = collector.read(path)
        if slot.get("console_transport"):
            continue
        if slot["status"] == "RUNNING":
            active = active_work_processes(Path(state['public_root']) / 'work' / slot['slot_id'])
            if active:
                raise RuntimeError(f"Previous CLI attempt {slot['slot_id']} is still running (PIDs {active})")
            if receipt.get("completed") and time.time() <= slot["started_epoch"] + 900:
                outcome, reason = "COMPLETED", None
            else:
                outcome = "INFRASTRUCTURE_INVALID"
                reason = "Controller interrupted before freezing the CLI result; no backdated completion"
            collector.finish(output, slot["slot_id"], outcome=outcome, reason=reason,
                             agent_id="codex-cli:" + (receipt.get("thread_id") or "failed-launch:" + slot["slot_id"]))
        if slot["status"] in {"RUNNING", "FINISHED"} or slot.get("run_record_path"):
            import_cli(output, slot["slot_id"], receipt)
    return collector.read(output / "collection_state.json")


def active_work_processes(work):
    """Inspect only process arguments; never take over a still-running analysis."""
    needle = str(Path(work).resolve()).encode()
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            raise RuntimeError('Cannot verify whether the previous CLI attempt is still running') from None
        if any(needle in arg for arg in args):
            found.append(int(entry.name))
    return found


def recover_event_receipt(output, state, slot):
    """Recover a missing receipt from a complete CLI event log, never rerun it.

    An observed completion before the original deadline is sufficient without
    inventing an earlier finish time. Late observation remains infrastructure
    uncertainty. The unknown process exit code stays null.
    """
    audit = output / 'cli_runs' / slot['slot_id']
    events = audit / 'events.jsonl'
    prompt = audit / 'prompt.txt'
    if not events.is_file() or not prompt.is_file():
        return False
    work = Path(state['public_root']) / 'work' / slot['slot_id']
    active = active_work_processes(work)
    if active:
        raise RuntimeError(f"Previous CLI attempt {slot['slot_id']} is still running (PIDs {active}); do not launch a duplicate")
    if collector.file_hash(prompt) != slot['prompt_sha256']:
        raise ValueError('Cannot recover CLI receipt: prompt binding mismatch')
    result = parse_events(events)
    if not result['completed'] or result['error'] or not result['thread_id']:
        return False
    observed = time.time()
    result.update(returncode=None, timed_out=False, started_epoch=slot['started_epoch'],
                  finished_epoch=observed, model=slot['model_id'], events_path=str(events.resolve()),
                  transport='codex_exec_chatgpt_login', structured_output_enforced=False,
                  recovery={'kind': 'completed-event-log-without-receipt',
                            'observed_epoch': observed, 'events_sha256': collector.file_hash(events),
                            'recovery_source_sha256': collector.file_hash(Path(__file__)),
                            'exit_code_unavailable': True, 'completion_time_not_backdated': True})
    collector.write(audit / 'receipt.json', result)
    emit(event='recovered_receipt', slot_id=slot['slot_id'], completion_observed_within_budget=observed <= slot['started_epoch'] + 900)
    return True


def replay_and_report(output, *, emit_report=True):
    replay_started = time.monotonic()
    from scripts.preparation_progress import PreparationProgress
    preparation = PreparationProgress(output)
    preparation.update('VALIDATING_RUN_RECORDS')
    # Import validates then annotates CLI metadata. Never sample the intermediate
    # record/hash; once published, run records are immutable during evaluation.
    with PUBLICATION_LOCK:
        paths = current_slot_run_paths(output)
    if not paths:
        return None
    exchange = exchange_for(output)
    dirty, reusable = paths, {}
    if REPLAY_CACHE is not None:
        from scripts.console_evaluation_cache import replay_guard
        preparation.update('CHECKING_INPUTS')
        dirty, reusable = REPLAY_CACHE.select(paths, exchange, replay_guard(ROOT, exchange))
    from scripts.console_evaluation_cache import preserved_scored_results
    preserved = preserved_scored_results(output, paths)
    reusable.update(preserved)
    dirty = [path for path in dirty if str(path) not in preserved]
    cmd = [sys.executable, '-E', '-s', '-X', 'utf8', '-u',
           str(ROOT / "scripts/replay_existing_with_progress.py"), "replay",
           "--exchange", str(exchange_for(output)),
           "--output", str(output / "evaluation"),
           "--workers", "4"]
    replay_marker = output / "evaluation/replay_fingerprint.json"
    current_replay_fingerprint = replay_source_fingerprint()
    previous_replay_fingerprint = None
    if replay_marker.exists():
        try:
            previous_replay_fingerprint = collector.read(replay_marker).get("fingerprint")
        except (OSError, ValueError, AttributeError):
            previous_replay_fingerprint = None
    # The marker is published before replay so an interrupted subprocess can
    # resume its already-written result files on the next invocation.
    collector.write(replay_marker, {
        "fingerprint": current_replay_fingerprint,
        "updated_epoch": time.time(),
        "reuse_contract": "answer-hash-and-active-judge-state-v1",
    })
    if previous_replay_fingerprint == current_replay_fingerprint:
        cmd.append("--reuse-existing")
    for path in dirty:
        cmd.extend(['--run-record', str(path)])
    logs = output / "console_logs"
    logs.mkdir(exist_ok=True)
    returncode = 0
    local_completion = {}
    if dirty:
        key = str(exchange.resolve())
        if key not in REPLAY_REUSE_INDEXES:
            from scripts.replay_judgment_completion import ReplayJudgmentCompletion
            # The completion index is content-addressed and immutable for an
            # exchange. Reuse the newest published index across evaluator
            # restarts; rebuilding tens of thousands of historical attempts is
            # local preparation overhead and never changes a judgment.
            existing_indexes = sorted(
                logs.glob('historical_reuse_index_*.json'),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if existing_indexes:
                reuse_index = existing_indexes[0]
            else:
                reuse_index = logs / f'historical_reuse_index_{time.time_ns()}.json'
                preparation.update('INDEXING_HISTORY')
                ReplayJudgmentCompletion(exchange).save_index(
                    reuse_index, progress=lambda completed, total: preparation.update(
                        'INDEXING_HISTORY', completed=completed, total=total))
            REPLAY_REUSE_INDEXES[key] = reuse_index
        cmd.extend(['--reuse-index', str(REPLAY_REUSE_INDEXES[key])])
        preparation.update('REPLAYING_ANSWERS', total=len(dirty))
        # Host replay includes verified data staging and deterministic
        # materialization. Keep a generous bound so a slow local disk cannot
        # prevent the reviewer from ever seeing the completed inventory.
        replay_timeout = max(900, 30 * len(dirty))
        with (logs / "replay.log").open("a") as log:
            try:
                # Allow per-record local I/O while retaining a finite cap.
                result = subprocess.run(cmd, stdout=log, stderr=log, cwd=ROOT,
                                        env={**os.environ, 'PYTHONHASHSEED': '0'},
                                        timeout=replay_timeout)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                log.write(f'\nREPLAY_TIMEOUT seconds={replay_timeout}\n')
                log.flush()
                returncode = 124
        if returncode:
            preparation.update('REPLAY_FAILED', status='ERROR')
            raise RuntimeError('Evaluator replay failed; see console_logs/replay.log. '
                               'No stale index was merged and no reviewer API was started.')
        index_path = output / 'evaluation/evaluation_index.json'
        if index_path.exists():
            local_completion = collector.read(index_path).get('local_dependency_completion', {})
    if REPLAY_CACHE is not None or preserved:
        preparation.update('REFRESHING_DEPENDENCIES')
        from scripts.console_evaluation_cache import restore_inventory
        updated = collector.read(output / 'evaluation/evaluation_index.json')['results'] if dirty else []
        if len(updated) != len(dirty):
            raise RuntimeError('Incremental replay result count mismatch')
        by_path = {**reusable, **dict(zip(map(str, dirty), updated))}
        merged = [by_path[str(path)] for path in paths]
        restore_inventory(output / 'evaluation', merged, exchange)
        if REPLAY_CACHE is not None:
            REPLAY_CACHE.remember(paths, merged, exchange,
                progress=lambda completed, total: preparation.update(
                    'REFRESHING_DEPENDENCIES', completed=completed, total=total))
        emit(event='incremental_replay', evaluated=len(dirty), reused=len(reusable), total=len(paths))
    collector.write(output / 'replay_progress.json', dict(updated_epoch=time.time(),
        elapsed_seconds=time.monotonic()-replay_started, evaluated=len(dirty), reused=len(reusable),
        local_dependency_completion=local_completion, api_calls=0))
    preparation.update('WRITING_REPORT')
    report = write_report(output) if emit_report else None
    if returncode:
        preparation.update('REPLAY_FAILED', status='ERROR')
        raise RuntimeError("Evaluator error: see console_logs/replay.log and evaluation/evaluation_index.json")
    preparation.update('READY_FOR_REVIEW', status='COMPLETE')
    return report


@publication_transaction
def write_report(output):
    state = collector.read(output / 'collection_state.json')
    report = report_subagent_experiment(output, output / "evaluation")
    transports = Counter()
    transport_rows = []
    for slot in state["slots"]:
        if not slot.get("run_record_path"):
            continue
        run = collector.read(output / slot["run_record_path"])
        transport = run.get("runtime_environment_fingerprint", {}).get("transport", "unknown")
        transports[transport] += 1
        transport_rows.append(dict(slot_id=slot["slot_id"], model=slot["model_id"], trial=slot["trial_index"],
                                   case_id=slot["case_id"], launch_transport=transport,
                                   schedule=run.get('runtime_environment_fingerprint', {}).get('console_schedule'),
                                   wall_clock_time=run.get("wall_clock_time"),
                                   python_execution_count=run.get("python_execution_count")))
    report["launch_transport_counts"] = dict(transports)
    report["launch_transport_rows"] = transport_rows
    report["mixed_launch_environments"] = len(transports) > 1
    if (output / 'hybrid_protocol.json').exists():
        from scripts.benchmark_hybrid import api_accounting, annotate_transport_efficiency
        report['api_transport_usage'] = api_accounting(output)
        report['api_calls'] = report['api_transport_usage']['http_attempts']
        report['hybrid_protocol'] = collector.read(output / 'hybrid_protocol.json')
        report['transport_comparability'] = 'CLI and third-party agent harnesses differ; retain transport strata and do not pool wall-time efficiency as identical environments.'
        annotate_transport_efficiency(report, state, output)
    schedule_path = output / 'console_schedule.json'
    report['console_schedule'] = collector.read(schedule_path) if schedule_path.exists() else None
    from scripts.grading_policy import POLICY, POLICY_SHA256
    report['reviewer_execution_policy'] = dict(POLICY, sha256=POLICY_SHA256,
        scope='New reviewer dispatches; historical responses retain original provenance',
        solver_efficiency='Reviewer usage is separate from model-solving efficiency')
    invocation_path = output / 'evaluation_invocation.json'
    budget_path = output / 'review_budget.json'
    if invocation_path.exists():
        report['evaluation_invocation'] = collector.read(invocation_path)
    if budget_path.exists():
        report['review_budget'] = collector.read(budget_path)
    directory = output / "reports"
    directory.mkdir(exist_ok=True)
    collector.write(directory / "experiment_report.json", report)
    rows = report["trial_rows"]
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row if not key.startswith("_")))
        with (directory / "trial_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    note = "\n实际执行入口：" + json.dumps(dict(transports), ensure_ascii=False) + "。\n"
    if len(transports) > 1:
        note += "本实验续跑更换过启动入口；模型目标标识沿用冻结协议，实际运行环境按试次保留。合并科学指标属于混合运行环境结果，效率比较应结合 JSON 中的入口明细。\n"
    if report['console_schedule'] and report['console_schedule'].get('mode') == 'parallel':
        note += '当前调度允许回答计算与评分重叠；壁钟耗时可能受本机资源竞争影响，不能当作纯模型推理耗时。\n'
    note = "\nObserved launch transports: " + json.dumps(dict(transports), ensure_ascii=False) + ".\n"
    if len(transports) > 1:
        note += (
            "The resumed experiment used multiple launch transports. Scientific metrics "
            "retain frozen model identities; efficiency comparisons must use the transport "
            "detail in JSON.\n"
        )
    if report['console_schedule'] and report['console_schedule'].get('mode') == 'parallel':
        note += (
            "Answer computation and evaluation may overlap; wall time can include local "
            "resource contention and is not pure model inference time.\n"
        )
    (directory / "experiment_report.md").write_text(render_markdown(report) + note)
    invocation = collector.read(output / 'evaluation_invocation.json') \
        if (output / 'evaluation_invocation.json').exists() else {}
    emit(event="report", status=report["status"],
         scientifically_scored=report["scientifically_terminal_slot_count"],
         historical_scientifically_scored_at_start=invocation.get('historical_scientifically_scored_at_start'),
         newly_scored_slots=invocation.get('newly_scored_slots'),
         requested=report["requested_slot_count"], path=str(directory / "experiment_report.md"))
    return report


def evaluate(output, reviewer_model, max_rounds=32000, judgment_limit=32, *,
             stop_event=None, collection_done=None, changed=None, judgment_workers=1,
             max_api_calls=None, max_wall_seconds=None, max_no_score_seconds=None):
    from scripts.run_codex_file_judgments import run_judgments
    from scripts.console_evaluation_cache import ordered_requests
    stop_event = stop_event if stop_event is not None else threading.Event()
    progress_guard = None
    if max_no_score_seconds is not None:
        from scripts.score_progress_guard import ScoreProgressGuard
        progress_guard = ScoreProgressGuard(output, max_no_score_seconds)
    # The reviewer wall budget starts immediately before provider dispatch,
    # below.  Initial local replay can take tens of seconds for an existing
    # exchange and must not consume the API reviewer window.
    budget_configured = False

    def ensure_review_budget():
        nonlocal budget_configured
        if not budget_configured and HYBRID is not None and hasattr(HYBRID, 'configure_review_budget'):
            HYBRID.configure_review_budget(max_api_calls=max_api_calls,
                                           max_wall_seconds=max_wall_seconds,
                                           stop_event=stop_event)
            budget_configured = True
    last_full_report = [time.monotonic()]
    initial_scored = [None]

    def track_invocation(report):
        if report is None:
            return None
        current = int(report.get('scientifically_terminal_slot_count', 0) or 0)
        if initial_scored[0] is None:
            initial_scored[0] = current
        collector.write(output / 'evaluation_invocation.json', {
            'updated_epoch': time.time(),
            'historical_scientifically_scored_at_start': initial_scored[0],
            'scientifically_scored_now': current,
            'newly_scored_slots': max(0, current - initial_scored[0]),
            'status': report.get('status'),
            'review_budget': collector.read(output / 'review_budget.json')
                if (output / 'review_budget.json').exists() else None,
        })
        return report

    def finish(report):
        return track_invocation(report)
    def refresh_dependencies():
        from scripts.grading_policy import POLICY
        report_due = time.monotonic()-last_full_report[0] >= POLICY['report_min_interval_seconds']
        replay_and_report(output, emit_report=report_due)
        if report_due:
            last_full_report[0] = time.monotonic()
        # Replay consumes identical historical dependencies as it encounters
        # them, including later stages of this same answer, in one traversal.
        return ordered_requests(output)
    blocked = {}
    collector.write(output / 'judgment_errors.json', dict(
        updated_epoch=time.time(), scope='current_invocation', errors=[],
        policy='Format failures deferred after one correction; no scientific response fabricated'))
    for round_number in range(1, max_rounds + 1):
        if (stop_event is not None and stop_event.is_set()) or (output / 'STOP_REQUESTED').exists():
            return finish(write_report(output))
        if changed is not None:
            changed.clear()
        if HYBRID is not None and HYBRID.cli_disabled and not HYBRID.api_judges:
            emit(event='scoring_paused_cli_quota', message='API solvers continue; pending judgments retained for CLI quota recovery')
            if wait_for_collection(collection_done, changed, stop_event):
                continue
            return finish(replay_and_report(output))
        report = replay_and_report(output)
        track_invocation(report)
        if report is not None and report["status"] == "COMPLETE":
            return finish(report)
        if report is None:
            if wait_for_collection(collection_done, changed, stop_event):
                continue
            return finish(report)
        kwargs = dict(stop_event=stop_event) if stop_event is not None else {}
        ensure_review_budget()
        kwargs.update(workers=judgment_workers, request_order=ordered_requests(output),
                      excluded_ids=blocked, continue_on_format_error=True,
                      lean=True, on_resolved=refresh_dependencies, continuous=True,
                      progress_guard=progress_guard)
        if HYBRID is not None:
            kwargs.update(runner=HYBRID.judge_runner, hybrid=HYBRID,
                          workers=min(8, judgment_workers + HYBRID.api_judges))
        summary = run_judgments(exchange=exchange_for(output), inventory=output / "evaluation/pending_requests.json",
                                work_root=output / "judgment_work", reviewer_model=reviewer_model,
                                limit=judgment_limit, timeout_seconds=900, **kwargs)
        from scripts.grading_policy import POLICY
        displayed = summary
        if POLICY.get('compact_round_logs'):
            displayed = {k: v for k, v in summary.items() if k != 'results'}
            displayed['audit_directory'] = str(output / 'judgment_work')
        emit(event="judgment_round", round=round_number, result=displayed)
        if summary.get("errors"):
            # A safe admission retry may have succeeded later in this window.
            latest = {item['request_id']: item for item in summary.get('results', [])}
            errors = [item for item in latest.values() if item['status'] == 'ERROR']
            blocked.update({item['request_id']: {key: item.get(key) for key in
                ('request_id', 'failure_kind', 'reason', 'attempt_path')} for item in errors
                if item.get('failure_kind') != 'api_transport'})
            collector.write(output / 'judgment_errors.json', dict(
                updated_epoch=time.time(), scope='current_invocation', errors=list(blocked.values()),
                policy='Format failures deferred after one correction; no scientific response fabricated'))
            if summary.get('fatal_errors', summary['errors']):
                replay_and_report(output)
                details = '; '.join(f"{item['request_id']}: {item.get('reason')} ({item.get('attempt_path')})" for item in errors)
                raise RuntimeError('Judgment transport/schema failure: ' + (details or 'inspect judgment_work before retrying'))
            budget_errors = [item for item in errors
                             if (item.get('transport') or {}).get('error') == 'review_api_budget_exhausted']
            if budget_errors:
                emit(event='review_budget_exhausted', request_ids=[item['request_id'] for item in budget_errors],
                     message='Hard reviewer budget reached; no further provider calls will be dispatched')
            else:
                emit(event='judgment_format_deferred', request_ids=[item['request_id'] for item in errors],
                     message='Other requests continue; failed requests remain unscored and are not retried again in this invocation')
        # Adapter reports newly resolved judgments; PENDING cannot mean success.
        if not summary.get("attempted", 0):
            if wait_for_collection(collection_done, changed, stop_event):
                continue
            return finish(replay_and_report(output))
    return finish(replay_and_report(output))


def wait_for_collection(done, changed, stop):
    """Wait without replaying unchanged data; final commits also trigger replay."""
    if done is None or changed is None:
        return False
    while stop is None or not stop.is_set():
        if changed.is_set():
            return True
        if done.is_set():
            return False
        changed.wait(timeout=1)
    return False


def all_parallel(output, max_runs, reviewer_model, max_rounds, judgment_limit, *, judgment_workers=1):
    stop, changed, done = threading.Event(), threading.Event(), threading.Event()
    worker = None
    def grade():
        try:
            kwargs = {'judgment_workers': judgment_workers} if judgment_workers != 1 else {}
            return evaluate(output, reviewer_model, max_rounds, judgment_limit,
                            stop_event=stop, collection_done=done, changed=changed, **kwargs)
        except BaseException:
            stop.set()
            raise
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='scientific-grader') as pool:
        def start_grader():
            nonlocal worker
            emit(event='parallel_started', solver_concurrency=(2 if HYBRID is None or HYBRID.cli_solvers else 0) + (HYBRID.api_solvers if HYBRID else 0),
                 grader_concurrency=judgment_workers + (HYBRID.api_judges if HYBRID else 0),
                 message='Existing collected answers are graded while new answers are collected')
            worker = pool.submit(grade)
        try:
            collect(output, max_runs, stop_event=stop, on_ready=start_grader, on_commit=changed.set)
        except BaseException:
            stop.set()
            raise
        finally:
            done.set()
            changed.set()
        try:
            report = worker.result() if worker is not None else None
        except BaseException:
            stop.set()
            raise
    # Refresh only local aggregation after the final collector import; no new
    # judging calls on a graceful stop or service failure.
    return write_report(output) if report is not None else report


def exchange_for(output):
    checkpoint = output / "orchestration_checkpoint.json"
    if checkpoint.exists():
        configured = collector.read(checkpoint).get("judgment_exchange")
        if configured:
            path = Path(configured)
            if not (path / "requests").is_dir():
                raise ValueError("Existing judgment exchange is missing; restore it before continuing")
            return path
    return output / "exchange"


def main():
    global REPLAY_CACHE, HYBRID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "api-preflight", "prepare", "recover", "collect", "evaluate", "all", "status", "report"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-runs", type=int, default=576)
    parser.add_argument("--reviewer-model", default="gpt-6-astra")
    parser.add_argument("--max-rounds", type=int, default=32000)
    parser.add_argument("--judgment-limit", type=int, default=32)
    parser.add_argument('--judgment-workers', type=int, default=2,
                        help='Independent grading CLI sessions (0..8); 0 disables CLI grading')
    parser.add_argument('--full-replay', action='store_true', help='Disable process-local incremental replay')
    parser.add_argument('--third-party-api', action='store_true', help='Use registered third-party API alongside CLI')
    parser.add_argument('--api-config', type=Path, default=ROOT / 'config/yiapi.toml')
    parser.add_argument('--api-solver-workers', type=int, choices=range(0, 9), default=1)
    parser.add_argument('--api-solvers-per-model', type=int, choices=range(1, 5), default=1,
                        help='Bounded independent solver trials per model; total workers and HTTP limit still apply')
    parser.add_argument('--api-judge-workers', type=int, choices=range(0, 9), default=1)
    parser.add_argument('--api-shared-concurrency', type=int, choices=range(0, 9), default=0,
                        help='Cross-process generation request cap for the same endpoint/account; 0 disables shared admission')
    parser.add_argument('--solver-transport', choices=('mixed', 'api'), default='mixed',
                        help='api reserves ChatGPT CLI exclusively for scoring')
    parser.add_argument('--api-http-concurrency', type=int, choices=range(1, 9), default=1,
                        help='Endpoint-wide in-flight HTTP limit; shared pacing/cooldown retained')
    parser.add_argument('--api-recovery-first-model', action='append', choices=collector.MODELS, default=[])
    parser.add_argument('--api-recovery-only-model', action='append', choices=collector.MODELS, default=[])
    parser.add_argument('--auto-recover-infrastructure', action='store_true',
                        help='Queue bounded audited replacements for imported infrastructure failures; never retry model noncompletion')
    parser.add_argument('--retry-cli-after-quota', action='store_true',
                        help='Explicitly re-enable CLI admission after its previously detected quota exhaustion')
    parser.add_argument('--schedule', choices=('parallel', 'sequential'), default='parallel',
                        help='all: bounded 2 solver + 1 grader concurrency, or sequential phases')
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.max_runs < 1 or args.max_rounds < 1 or args.judgment_limit < 1:
        parser.error("Limits must be positive")
    if not 0 <= args.judgment_workers <= 8:
        parser.error('--judgment-workers must be between 0 and 8')
    if args.command in {'all', 'evaluate'} and not 1 <= args.judgment_workers + (args.api_judge_workers if args.third_party_api else 0) <= 8:
        parser.error('Total enabled grading workers must be between 1 and 8')
    if args.solver_transport == 'api' and (not args.third_party_api or not args.api_solver_workers):
        parser.error('--solver-transport api requires --third-party-api and API solver workers')
    if args.auto_recover_infrastructure and not args.third_party_api:
        parser.error('--auto-recover-infrastructure requires --third-party-api')
    if args.command == "status":
        state = collector.read(output / "collection_state.json")
        report_path = output / "reports/experiment_report.json"
        report = collector.read(report_path) if report_path.exists() else {}
        schedule = output / 'console_schedule.json'
        def status_file(name):
            path = output / name
            value = collector.read(path) if path.exists() else None
            if name == 'judgment_usage.json' and value is not None:
                value.pop('seen_transports', None)
                value.pop('by_answer', None)  # Detailed rows remain in the durable ledger.
            return value
        emit(collection=state["status_counts"], total=state["total_slots"],
             scientifically_scored=report.get("scientifically_terminal_slot_count"),
             fully_scored_completed_answers=sum(r.get('status') == 'COMPLETED' for r in report.get('trial_rows', [])),
             report_status=report.get("status"),
             schedule=collector.read(schedule) if schedule.exists() else None,
             process=status_file('console_process.json'),
             judgment_batch=status_file('judgment_progress.json'),
             judgment_usage=status_file('judgment_usage.json'),
             infrastructure_recovery=status_file('infrastructure_recovery_queue.json'),
             judgment_errors=status_file('judgment_errors.json'),
             api_health=status_file('api_health.json'),
             cli_health=status_file('cli_health.json'),
             api_quarantine=status_file('api_judgment_quarantine.json'),
             reports=str(output / "reports/experiment_report.md"))
        return 0
    if args.command == "doctor":
        require_chatgpt_login()
        stamp = str(time.time_ns())
        for model in dict.fromkeys((*collector.MODELS, args.reviewer_model)):
            directory = output / "console_preflight" / stamp / model
            result = run_codex("Do not use tools. Reply with exactly READY.", model,
                               directory / "work", directory / "audit", timeout_seconds=60)
            emit(model=model, completed=result["completed"], response=result["final_text"])
            if not result["completed"] or result["final_text"].strip() != "READY":
                return 2
        return 0
    with runner_lock(output):
        HYBRID = None
        if args.retry_cli_after_quota:
            collector.write(output / 'cli_health.json', dict(quota_exhausted=False,
                updated_epoch=time.time(), reason='Explicit console request to retry CLI after quota reset'))
        if args.third_party_api or args.command == 'api-preflight':
            from scripts.benchmark_hybrid import HybridController, verify_hybrid_protocol
            verify_hybrid_protocol(output, args.api_config)
            HYBRID = HybridController(sys.modules[__name__], output, args.api_config,
                cli_judges=args.judgment_workers, api_solvers=args.api_solver_workers,
                api_judges=args.api_judge_workers, cli_solvers=args.solver_transport != 'api',
                api_http_concurrency=args.api_http_concurrency,
                recovery_first_models=args.api_recovery_first_model,
                recovery_only_models=args.api_recovery_only_model,
                auto_recover_infrastructure=args.auto_recover_infrastructure,
                api_solvers_per_model=args.api_solvers_per_model,
                api_shared_concurrency=args.api_shared_concurrency)
            if args.command in {'all', 'collect', 'evaluate', 'api-preflight'}:
                ready = HYBRID.refresh_health(startup=True)
                emit(event='api_preflight', ready=ready, network=HYBRID.governor.network,
                     health=collector.read(output / 'api_health.json'),
                     message='API health verified before model trial budget starts' if ready else
                             'API admission unavailable; CLI work can continue')
                if args.command == 'api-preflight':
                    return 0 if ready else 2
        REPLAY_CACHE = None
        if args.command in {'all', 'evaluate'} and not args.full_replay:
            from scripts.console_evaluation_cache import IncrementalReplay
            REPLAY_CACHE = IncrementalReplay()
        # Explicit user invocation resumes after the prior graceful stop.
        if args.command in {'all', 'collect', 'evaluate'}:
            (output / 'STOP_REQUESTED').unlink(missing_ok=True)
        if args.command in {"collect", "evaluate", "all"}:
            if HYBRID is None or not HYBRID.cli_disabled:
                require_chatgpt_login()
        if args.command == "prepare":
            state = prepare(output)
            emit(prepared=state["total_slots"], output=str(output))
        elif args.command == "recover":
            state = finalize_receipts(output, prepare(output))
            emit(event='recovery_status', collection=state['status_counts'])
            return int(any(s['status'] in {'RUNNING', 'FINISHED'} for s in state['slots']))
        elif args.command in {"collect", "all"}:
            schedule = dict(
                schedule_id=str(time.time_ns()),
                runner_sha256=collector.file_hash(Path(__file__)),
                mode=args.schedule if args.command == 'all' else 'collection_only',
                solver_concurrency=2, grader_concurrency=args.judgment_workers if args.command == 'all' else 0,
                judgment_order='unblocked_later_stage_fewer_dependencies_then_oldest', incremental_replay=not args.full_replay,
                replay_pythonhashseed=0,
                judgment_batch=args.judgment_limit, started_epoch=time.time(),
                efficiency_note='Overlapping local grading can affect solver wall-clock time')
            from scripts.grading_policy import POLICY, POLICY_SHA256
            schedule['reviewer_execution_policy'] = dict(POLICY, sha256=POLICY_SHA256)
            if HYBRID is not None:
                schedule.update(solver_concurrency=(2 if HYBRID.cli_solvers else 0)+HYBRID.api_solvers,
                    grader_concurrency=args.judgment_workers+HYBRID.api_judges if args.command=='all' else 0,
                    api_solver_workers=HYBRID.api_solvers, api_judge_workers=HYBRID.api_judges,
                    api_http_concurrency=args.api_http_concurrency, api_config_sha256=collector.file_hash(args.api_config),
                    api_shared_concurrency=args.api_shared_concurrency,
                    codex_cli_judge_workers=args.judgment_workers,
                    grading_dispatch='rolling atomic window; no round-drain barrier',
                    cli_solver_workers=2 if HYBRID.cli_solvers else 0,
                    recovery_first_models=args.api_recovery_first_model,
                    recovery_only_models=args.api_recovery_only_model,
                    auto_recover_infrastructure=args.auto_recover_infrastructure,
                    api_solvers_per_model=args.api_solvers_per_model,
                    reviewer_model=args.reviewer_model,
                    transport_assignment='Distinct pending slots; recorded per-model/total concurrency limits; stable ordinary/recovery phases')
            collector.write(output / 'console_schedule_history' / (schedule['schedule_id'] + '.json'), schedule)
            collector.write(output / 'console_schedule.json', schedule)
            if args.command == 'all' and args.schedule == 'parallel':
                report = all_parallel(output, args.max_runs, args.reviewer_model, args.max_rounds, args.judgment_limit,
                                      judgment_workers=args.judgment_workers)
                return 0 if report and report['status'] == 'COMPLETE' else 2
            stop = threading.Event()
            collect(output, args.max_runs, stop_event=stop)
            if stop.is_set():
                return 2
            if args.command == "all":
                report = evaluate(output, args.reviewer_model, args.max_rounds, args.judgment_limit,
                                  judgment_workers=args.judgment_workers)
                return 0 if report and report["status"] == "COMPLETE" else 2
        elif args.command == "evaluate":
            prepare(output)
            report = evaluate(output, args.reviewer_model, args.max_rounds, args.judgment_limit,
                              judgment_workers=args.judgment_workers)
            return 0 if report and report["status"] == "COMPLETE" else 2
        elif args.command == "report":
            replay_and_report(output)
    return 0


if __name__ == "__main__":
    def _terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _terminate)
    raise SystemExit(main())

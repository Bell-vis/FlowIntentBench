#!/usr/bin/env python3
"""Evaluate collected Claude answers with bounded, locally closed metrics.

This script never launches a solver and never creates a new answer.  It only
consumes the existing evaluation exchange under --output and fills unresolved
review requests.  It is safe to stop and rerun: completed requests are reused
by the exchange/replay cache and the published CSV is kept in place.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_runtime import require_benchmark_runtime  # noqa: E402

if __name__ == "__main__":
    require_benchmark_runtime()

from scripts import console_evaluation_cache as cec  # noqa: E402
from scripts import grading_policy  # noqa: E402
from scripts import run_claude_benchmark as claude  # noqa: E402
from scripts import run_codex_benchmark as console  # noqa: E402
from scripts.console_evaluation_cache import IncrementalReplay  # noqa: E402
from flowintentbench.external_file_evaluator import digest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=("trusted", "legacy"), default="trusted",
                   help="trusted uses one bounded review per answer; legacy resumes the original xhigh audit")
    p.add_argument("--trusted-output", type=Path)
    p.add_argument("--donor", type=Path, default=ROOT / "outputs/claude_outcomes_gpt6_complete_177")
    p.add_argument("--effort", choices=("low", "medium", "high"), default="medium")
    p.add_argument("--max-output-tokens", type=int, default=6000)
    p.add_argument("--api-config", type=Path, default=ROOT / "config/yiapi_direct.toml")
    p.add_argument("--auth-path", type=Path, default=ROOT / "auth_yapi.json")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/claude_resume_eval",
                   help="existing Claude output directory")
    p.add_argument("--reviewer-model", default="gpt-6-astra", choices=("gpt-6-astra",),
                   help="fixed GPT-6 evaluator identity")
    p.add_argument("--max-rounds", type=int, default=2000)
    p.add_argument("--judgment-limit", type=int, default=32,
                   help="maximum in-flight atomic judgments (not API calls)")
    p.add_argument("--resume", action="store_true", help="clear a prior stop request after acquiring the run lock")
    p.add_argument("--prepare-only", action="store_true",
                   help="validate and replay existing answers locally, then exit without API calls")
    p.add_argument("--api-judge-workers", type=int, default=2)
    p.add_argument("--api-shared-concurrency", type=int, default=2)
    p.add_argument("--max-api-calls", type=int, default=8,
                   help="hard cap on provider reviewer calls for this invocation")
    p.add_argument("--max-wall-seconds", type=float, default=300,
                   help="hard wall-clock cap for reviewer dispatch in this invocation")
    p.add_argument("--max-no-score-seconds", type=float, default=1800,
                   help="stop dispatch after this many seconds without a newly complete answered evaluation")
    return p.parse_args()


def configure_xhigh():
    claude.configure_process()
    # Consumers import POLICY by reference. Update that object in place so
    # actual requests, batch provenance and progress report the same policy.
    grading_policy.POLICY.update(
        version="claude-xhigh-structured-review-v17",
        text_effort="xhigh", eligibility_effort="xhigh",
        extraction_effort="xhigh", scientific_effort="xhigh",
        # Large 32-item JSON responses were truncated or malformed by the
        # gateway, turning one transport failure into dozens of blocked child
        # requests.  Eight items keeps each response below the provider's
        # practical output limit while retaining useful batching. Unit/frame
        # adjudications use the same per-item identity checks and can share a
        # transport packet when they belong to one authenticated answer.
        structured_only=True, batch_size=8,
        batch_roles=["semantic_match", "eligibility", "recipe_binding", "unit_relationship"],
        batch_answer_scope=True, legacy_prompt_roles=[],
        scientific_batch_size=4, compact_round_logs=True, batch_format_repairs=2,
        pending_rule=(
            "PENDING_ONLY_FOR_MISSING_DECISIVE_EVIDENCE; resolve explicit support or "
            "contradiction from supplied answer-bound evidence; answer omissions are terminal negatives; "
            "never invent execution or source facts"
        ),
    )
    grading_policy.POLICY_SHA256 = digest(grading_policy.POLICY)
    from scripts import batched_file_judgments, lean_judgment_scheduler
    for module in (batched_file_judgments, lean_judgment_scheduler):
        module.POLICY = grading_policy.POLICY
        module.POLICY_SHA256 = grading_policy.POLICY_SHA256


def run(args) -> int:
    output = args.output.resolve()
    required = [output / "collection_state.json", output / "evaluation/evaluation_index.json",
                output / "exchange/requests"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("已有答案评估目录不完整，缺少: " + ", ".join(missing))

    # The policy is part of every evaluation manifest.  Keep the reviewer at
    # the requested GPT-6 xhigh setting and make streaming explicit so the
    # gateway does not hit its non-streaming idle timeout.
    configure_xhigh()
    # Publish Claude-specific solver efficiency. This excludes reviewer and
    # host execution accounting from the primary efficiency measurements.
    console.write_report = console.publication_transaction(claude.write_report)
    if getattr(args, 'prepare_only', False):
        console.REPLAY_CACHE = IncrementalReplay()
        report = console.replay_and_report(output)
        print(json.dumps(dict(event='local_preparation_complete', api_calls=0,
                              answered_evaluation=report.get('answered_evaluation'))), flush=True)
        return 0

    # Preserve the normal dependency ordering, then append any newly-created
    # active request IDs.  This is essential for resumability after a prior
    # interrupted run, and does not submit already-resolved requests again.
    original_order = cec.ordered_requests

    def ordered_existing(output_dir: Path):
        ids = original_order(output_dir) or []
        inventory = output_dir / "evaluation/pending_requests.json"
        if inventory.exists():
            doc = json.loads(inventory.read_text())
            active = [item["request_id"]
                      for rows in doc.get("by_operation", {}).values()
                      for item in rows]
            ids.extend(rid for rid in active if rid not in ids)
        ids = list(dict.fromkeys(ids))
        # Prefer requests that can close a trial immediately: exactly one
        # missing response and no held PENDING dependency. This is scheduling
        # only; the evaluator's scoring and dependency rules remain unchanged.
        try:
            index = json.loads((output_dir / "evaluation/evaluation_index.json").read_text())
            response_dir = output_dir / "exchange/responses"
            priority = {}
            for row in index.get("results", []):
                active_ids = row.get("active_request_ids") or []
                missing, held = [], 0
                for rid in active_ids:
                    try:
                        status = json.loads((response_dir / (rid + ".json")).read_text()).get("status")
                    except (FileNotFoundError, json.JSONDecodeError, OSError):
                        status = None
                    if status == "PENDING":
                        held += 1
                    elif status != "RESOLVED":
                        missing.append(rid)
                if held == 0 and missing:
                    rank = (0 if len(missing) == 1 else 1, len(missing))
                    for rid in missing:
                        priority[rid] = min(priority.get(rid, (2, 999)), rank)
            ranked = sorted(enumerate(ids), key=lambda item: (priority.get(item[1], (2, 999)), item[0]))
            return [rid for _, rid in ranked]
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
            return ids

    cec.ordered_requests = ordered_existing
    console.HYBRID = claude.ClaudeController(
        output, ROOT / "settings.json", None, ROOT / "config/yiapi.toml", 0,
        api_judge_workers=args.api_judge_workers,
        api_shared_concurrency=args.api_shared_concurrency,
    )
    # The evaluation entrypoint never grants the reviewer code execution.
    # Numerical materialization and score computation belong to the host.
    console.HYBRID.agent.structured_only = True
    console.REPLAY_CACHE = IncrementalReplay()

    # Probe the reviewer endpoint before the expensive local replay.  The
    # probe is a read-only /models request; when a restricted namespace cannot
    # reach the host-local proxy, exit immediately instead of spending a
    # minute hashing 178 trials and then discovering the same infrastructure
    # failure.  No evaluation request or score is created here.
    # Do not include the governor's pacing window in this startup probe:
    # check_health() may succeed while available() is briefly false because
    # the previous probe just reserved the minimum inter-request interval.
    # Pacing is handled by normal admission once replay begins.
    governor = console.HYBRID.judge_governor
    if not governor.check_health(startup=True):
        health = json.loads((output / 'api_health.json').read_text()) \
            if (output / 'api_health.json').exists() else {}
        reason = health.get('failure_reason') or 'api_preflight_unavailable'
        print(json.dumps({'status': 'STOPPED', 'reason': reason,
                          'message': 'reviewer preflight failed; no replay or API judgment was started'},
                         ensure_ascii=False), flush=True)
        return 3

    print(json.dumps({
        "output": str(output), "solver": "disabled",
        "reviewer_model": args.reviewer_model, "reasoning_effort": "xhigh",
        "api_judge_workers": args.api_judge_workers,
        "api_shared_concurrency": args.api_shared_concurrency,
        "resume": True,
        "reviewer_tools": [], "scoring": "deterministic_host_code",
        "batch_size": grading_policy.POLICY['batch_size'],
        "max_api_calls": args.max_api_calls,
        "max_wall_seconds": args.max_wall_seconds,
    }, ensure_ascii=False), flush=True)
    report = console.evaluate(
        output, args.reviewer_model, max_rounds=args.max_rounds,
        judgment_limit=args.judgment_limit, judgment_workers=0,
        max_api_calls=args.max_api_calls, max_wall_seconds=args.max_wall_seconds,
        max_no_score_seconds=args.max_no_score_seconds,
    )
    invocation = json.loads((output / 'evaluation_invocation.json').read_text()) \
        if (output / 'evaluation_invocation.json').exists() else {}
    budget = json.loads((output / 'review_budget.json').read_text()) \
        if (output / 'review_budget.json').exists() else {}
    # Always mirror the terminal/active budget snapshot into the invocation
    # record.  The evaluator may exhaust its bounded budget and return
    # normally; leaving the older ACTIVE snapshot makes monitoring report a
    # phantom running budget.  This is bookkeeping only and does not affect
    # scheduling, judgments, or score computation.
    if budget:
        invocation['review_budget'] = budget
        from flowintentbench.external_file_evaluator import write_json
        write_json(output / 'evaluation_invocation.json', invocation)
    # A cooperative stop may arrive while the scheduler is draining in-flight
    # work.  Persist the terminal budget state so monitoring cannot mistake a
    # stopped invocation for an active one.  This is runtime bookkeeping only;
    # it does not alter any judgment or score.
    if (output / 'STOP_REQUESTED').exists():
        budget.update(status='STOPPED', reason='STOP_REQUESTED', updated_epoch=time.time())
        from flowintentbench.external_file_evaluator import write_json
        write_json(output / 'review_budget.json', budget)
        # Keep the invocation snapshot consistent with the terminal budget
        # state; monitoring reads both files.
        invocation['review_budget'] = budget
        write_json(output / 'evaluation_invocation.json', invocation)
    print(json.dumps({
        "status": report.get("status") if report else None,
        "answered_evaluation_status": report.get("answered_evaluation", {}).get("status") if report else None,
        "answered_quality_complete": report.get("answered_evaluation", {}).get("quality_complete_count") if report else None,
        "answered_slot_count": report.get("answered_evaluation", {}).get("answered_slot_count") if report else None,
        "scientifically_scored": report.get("scientifically_terminal_slot_count") if report else None,
        "historical_scientifically_scored_at_start": invocation.get('historical_scientifically_scored_at_start'),
        "newly_scored_slots": invocation.get('newly_scored_slots', 0),
        "review_api_calls": budget.get('api_calls', 0),
        "review_budget_status": budget.get('status'),
        "review_budget_reason": budget.get('reason'),
        "requested": report.get("requested_slot_count") if report else None,
        "report": str(output / "reports/experiment_report.md"),
    }, ensure_ascii=False), flush=True)
    # INCOMPLETE is not successful completion of the requested evaluations.
    if (output / 'STOP_REQUESTED').exists():
        print(json.dumps(dict(event='evaluation_exit', reason='STOP_REQUESTED')), flush=True)
        return 130
    answered_complete = (
        report is not None
        and report.get('answered_evaluation', {}).get('status') == 'COMPLETE'
    )
    print(json.dumps(dict(
        event='evaluation_exit',
        reason='ANSWERED_EVALUATION_COMPLETE' if answered_complete else
        'INCOMPLETE: answered trials still have unresolved quality dependencies',
    )), flush=True)
    return 0 if answered_complete else 2


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.terminal.write(value)

    def flush(self):
        self.log.flush()
        self.terminal.flush()


def main() -> int:
    require_benchmark_runtime()
    args = parse_args()
    if getattr(args, "engine", "legacy") == "trusted":
        from scripts.evaluate_model_answers import parser as trusted_parser, run as trusted_run
        from scripts.portable_fcntl import fcntl
        trusted_args = trusted_parser().parse_args([])
        trusted_args.collection = args.output
        trusted_args.output = args.trusted_output or args.output / "rubric_metrics_v4"
        trusted_args.donor = args.donor
        trusted_args.offline = args.prepare_only
        trusted_args.max_api_calls = args.max_api_calls
        trusted_args.max_wall_seconds = args.max_wall_seconds
        trusted_args.timeout = min(150, args.max_wall_seconds)
        trusted_args.workers = args.api_judge_workers
        trusted_args.reviewer_model = args.reviewer_model
        trusted_args.effort = args.effort
        trusted_args.max_output_tokens = args.max_output_tokens
        trusted_args.api_config = args.api_config
        trusted_args.auth_path = args.auth_path
        trusted_args.shared_concurrency = args.api_shared_concurrency
        trusted_args.output.mkdir(parents=True, exist_ok=True)
        with (trusted_args.output / ".trusted_evaluation.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = trusted_run(trusted_args)
        return 0 if report["status"] in {"COMPLETE", "COMPLETE_WITH_BOUNDS"} else 2
    logs = args.output.resolve() / 'console_logs'
    logs.mkdir(parents=True, exist_ok=True)
    prefix = logs / f'xhigh_run_{time.time_ns()}'
    state = dict(pid=os.getpid(), started_epoch=time.time(), status='RUNNING',
                 log_path=str(prefix.with_suffix('.log')))
    def save():
        from flowintentbench.external_file_evaluator import write_json
        write_json(prefix.with_suffix('.json'), state)
    save()
    with prefix.with_suffix('.log').open('a', buffering=1) as log, \
            redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            with console.runner_lock(args.output.resolve()):
                stop = args.output.resolve() / 'STOP_REQUESTED'
                if stop.exists():
                    if not getattr(args, 'resume', False):
                        raise SystemExit('存在停止标记；续评请添加 --resume。')
                    stop.unlink()
                code = run(args)
            state.update(status='EXITED', exit_code=code)
            return code
        except BaseException as exc:
            # Do not leave an apparently active budget after a local replay or
            # initialization failure.  This is bookkeeping only; no judgment
            # is synthesized and no score is changed.
            budget_path = args.output.resolve() / 'review_budget.json'
            if budget_path.exists():
                try:
                    from flowintentbench.external_file_evaluator import write_json
                    budget = json.loads(budget_path.read_text())
                    if budget.get('status') == 'ACTIVE':
                        budget.update(status='ERROR', reason=type(exc).__name__ + ': ' + str(exc)[:300],
                                      updated_epoch=time.time())
                        write_json(budget_path, budget)
                        invocation_path = args.output.resolve() / 'evaluation_invocation.json'
                        if invocation_path.exists():
                            invocation = json.loads(invocation_path.read_text())
                            invocation['review_budget'] = budget
                            write_json(invocation_path, invocation)
                except Exception:
                    pass
            state.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'ERROR',
                         error_type=type(exc).__name__, error_message=str(exc))
            traceback.print_exc()
            raise
        finally:
            state['finished_epoch'] = time.time()
            save()


if __name__ == "__main__":
    raise SystemExit(main())

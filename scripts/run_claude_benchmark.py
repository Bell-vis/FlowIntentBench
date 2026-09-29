#!/usr/bin/env python3
"""Independent Claude experiment composed from the existing benchmark workflow.

Process-local configuration only: never amend the running Luna/Terra protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_subagent_runs as collector
from scripts import run_codex_benchmark as console
from scripts.benchmark_hybrid import HybridController, api_accounting
from scripts.third_party_benchmark_transport import APIGovernor, ThirdPartyAgent, api_visible_prompt
from scripts.claude_code_transport import MODELS, resolve_cli, configuration, stage_scope, run_claude
from scripts.console_evaluation_cache import IncrementalReplay
from flowintentbench.model_runner import RunRecord
from flowintentbench.runtime_config import resolve_provider_configuration

DEFAULT_OUTPUT = ROOT / 'outputs/expansion96_n3_claude'
ORIGINAL_PROMPT = collector.prompt_for
ORIGINAL_REPORT = console.write_report


def claude_prompt(slot, public):
    prompt = api_visible_prompt(ORIGINAL_PROMPT(slot, public))
    start = prompt.index('Write your final answer to ')
    return prompt[:start] + '''Use read_case to read the assigned case_input.json. Use the python tool for
all computations. It can read only this case's declared files and installed
libraries, and can write only the scratch directory (its working directory).
Use relative paths for intermediate outputs. No network or other host files
are accessible from analysis Python. Do not invoke run_python.py yourself.
Submit your final answer through submit_answer with exactly two nonempty sections:
## Operationalization
Your concrete interpretation and computational method.
## Finding
Your scientific answer with quantitative evidence where possible.
The host saves answer.md and complete execution evidence. Do not write that
file using Python. Your final chat message should only confirm submission.
'''


def configure_process():
    collector.MODELS = MODELS
    collector.HELPER = ROOT / 'scripts/claude_blind_python.py'
    collector.prompt_for = claude_prompt
    from scripts import grading_policy
    # Text extraction and eligibility are contract checks; reserve high effort
    # for scientific adjudication where numerical evidence can be required.
    grading_policy.POLICY = dict(grading_policy.POLICY, version='claude-api-mixed-review-v8',
        text_effort='medium', eligibility_effort='medium', extraction_effort='medium', scientific_effort='high')
    grading_policy.POLICY_SHA256 = grading_policy.digest(grading_policy.POLICY)
    # This module keeps imported policy references. Set them explicitly for
    # embedders/tests too, without touching the shared policy file on disk.
    from scripts import batched_file_judgments, lean_judgment_scheduler
    for module in (batched_file_judgments, lean_judgment_scheduler):
        module.POLICY = grading_policy.POLICY
        module.POLICY_SHA256 = grading_policy.POLICY_SHA256


def protocol(settings, cli, api_config):
    from scripts.grading_policy import POLICY, POLICY_SHA256
    from scripts.third_party_benchmark_transport import network_configuration
    base, _, _ = configuration(settings)
    sources = [*sorted((ROOT / 'flowintentbench').glob('*.py')),
        *(ROOT / 'scripts' / name for name in (
            'run_claude_benchmark.py', 'run_claude_benchmark.sh', 'claude_code_transport.py',
            'claude_case_tools.py', 'claude_blind_python.py', 'collect_subagent_runs.py',
            'run_real_model_pilot.py', 'run_codex_benchmark.py', 'benchmark_hybrid.py',
            'third_party_benchmark_transport.py', 'run_codex_file_judgments.py',
            'run_expansion_file_evaluation.py', 'replay_judgment_completion.py', 'api_judgment_recovery.py', 'console_evaluation_cache.py', 'grading_policy.py',
            'lean_judgment_scheduler.py', 'batched_file_judgments.py',
            'carry_forward_file_judgments.py', 'recover_subagent_infrastructure.py', 'shared_api_admission.py'))]
    return dict(version='claude-code-scoped-mcp-v2', models=list(MODELS), solver_effort='max',
        reviewer_model='gpt-6-astra', reviewer_effort='mixed_medium_high', codex_cli_judges=0,
        repetitions=3, timeout_seconds=collector.SOLVER_TIMEOUT_SECONDS,
        max_python_executions=collector.SOLVER_MAX_PYTHON_EXECUTIONS,
        claude_endpoint=base, cli_path=str(cli), cli_version=subprocess.check_output([str(cli), '--version'], text=True).strip(),
        cli_sha256=collector.file_hash(cli.resolve()),
        api_runtime=resolve_provider_configuration(config_path=api_config).without_secrets(),
        network=network_configuration(), reviewer_policy=dict(POLICY, sha256=POLICY_SHA256),
        sources={str(p.relative_to(ROOT)): collector.file_hash(p) for p in sources},
        access_policy='Three explicit MCP tools; no builtins/skills/agents/ambient settings; '
                      'analysis Landlock allowlist, scratch-only writes and seccomp network/process restrictions; no namespaces',
        efficiency_policy='Raw Claude CLI time and cache-inclusive token usage; opaque transport overhead unknown; '
                          'no pure inference timing or transport-adjusted primary efficiency claim',
        retry_policy='No blind CLI conversation restart. Audited imported infrastructure replacements only, maximum three. '
                     'Unfinished claims without durable receipts require review.')


def prepare(output, *, settings, cli, api_config):
    output = Path(output).resolve()
    if output == console.DEFAULT_OUTPUT.resolve():
        raise ValueError('Claude must use its own output directory')
    state_path = output / 'collection_state.json'
    if state_path.exists() and tuple(collector.read(state_path)['configuration']['models']) != MODELS:
        raise ValueError('Refusing to reuse another model experiment')
    expected = protocol(settings, cli, api_config)
    pp = output / 'claude_protocol.json'
    if pp.exists() and collector.read(pp) != expected:
        raise ValueError('Claude protocol/code/config changed; do not mix versions into this experiment')
    state = collector.prepare(manifest_path=ROOT / 'datasets/expansion_v1/case_manifest.json',
        datasets_root=ROOT / 'datasets', output=output, seed=20260914, reasoning_effort='max')
    if not pp.exists():
        collector.write(pp, expected)
    # The stricter boundary supplements the existing collector envelope.
    # No secret, GT or reviewer file is copied to the public root.
    return state


def amend_protocol(output, *, settings, cli, api_config, reason):
    """Explicit stopped-run source amendment; retain every original observation."""
    if not reason or not reason.strip():
        raise ValueError('Source amendment requires a reason')
    output = Path(output)
    state = collector.read(output / 'collection_state.json')
    if any(s['status'] in {'RUNNING', 'FINISHED'} for s in state['slots']):
        raise ValueError('Drain active trials before amending Claude protocol')
    collector.verify(state)
    for slot in state['slots']:
        if slot.get('run_record_sha256') and collector.file_hash(output / slot['run_record_path']) != slot['run_record_sha256']:
            raise ValueError('Existing Claude run hash mismatch')
    path = output / 'claude_protocol.json'
    old, new = collector.read(path), protocol(settings, cli, api_config)
    # No silent provider, model, effort, budget, blinding or policy change.
    if {k:v for k,v in old.items() if k != 'sources'} != {k:v for k,v in new.items() if k != 'sources'}:
        raise ValueError('Only a source amendment is supported; experiment settings differ')
    digest = collector.file_hash(path)
    archive = output / 'claude_protocol_versions' / (digest + '.json')
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
    collector.write(output / 'preflight' / f'claude_amendment_{time.time_ns()}.json', dict(
        reason=reason, epoch=time.time(), previous_sha256=digest, new_protocol=new,
        preserved_runs=sum(bool(s.get('run_record_sha256')) for s in state['slots'])))
    collector.write(path, new)
    return new


class ClaudeAdmission:
    """Solver admission independent of the GPT grading endpoint's health."""
    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        state = collector.read(path) if path.exists() else {}
        self.disabled = False
        self.disabled_models = set(state.get('disabled_models', []))
        self.until = state.get('cooldown_until', {})
        self.provider_until = state.get('provider_cooldown_until', 0)
        self.provider_failures = state.get('provider_failures', 0)
        self.probe_inflight = False
    def available(self, model):
        return (model not in self.disabled_models and not self.probe_inflight
                and time.time() >= max(self.until.get(model, 0), self.provider_until))
    def reserve(self, model):
        with self.lock:
            if not self.available(model):
                return False
            # After an outage admit one real trial, not both models at once.
            if self.provider_failures:
                self.probe_inflight = True
            return True
    def persist(self, **extra):
        collector.write(self.path, dict(disabled_models=sorted(self.disabled_models), cooldown_until=self.until,
            provider_cooldown_until=self.provider_until, provider_failures=self.provider_failures,
            probe_inflight=self.probe_inflight, updated_epoch=time.time(), **extra))
    def succeeded(self, model):
        with self.lock:
            self.probe_inflight = False
            # An older concurrent success cannot cancel a newer failure's cooldown.
            if time.time() >= self.provider_until:
                self.provider_failures = 0
            self.persist(last_success={'model': model, 'epoch': time.time()})
    def failed(self, model, receipt):
        with self.lock:
            self.probe_inflight = False
            self.until[model] = time.time() + 120
            if receipt.get('provider_unavailable'):
                self.provider_failures += 1
                delay = min(900, 120 * 2 ** min(self.provider_failures - 1, 3))
                self.provider_until = max(self.provider_until, time.time() + delay)
            if receipt.get('error') in {'returned_model_mismatch_or_missing', 'unexpected_or_missing_tools', 'claude_cli_launch_failure'}:
                self.disabled_models.add(model)
            if receipt.get('http_status') in {401, 403}:
                self.disabled = True
                self.disabled_models.update(MODELS)
            self.persist(last_failure={'model': model, 'reason': receipt.get('error'), 'epoch': time.time()})


class ClaudeController(HybridController):
    def __init__(self, output, settings, cli, api_config, solver_workers, *, api_judge_workers=2,
                 api_shared_concurrency=4):
        self.console, self.output, self.settings, self.cli = console, output, settings, cli
        self.governor = ClaudeAdmission(output / 'claude_health.json')
        # A provider failure gets one bounded retry inside a logical review;
        # repeated recovery belongs to a later explicit invocation with a new
        # budget, otherwise one bad gateway can multiply token and wall costs.
        self.judge_governor = APIGovernor(output / 'api_health.json', attempts=2,
                                          max_concurrency=api_judge_workers)
        self.agent = ThirdPartyAgent(api_config, self.judge_governor)
        from scripts.shared_api_admission import SharedAPIAdmission
        self.judge_governor.shared_gate = SharedAPIAdmission(self.agent.runtime.base_url, self.agent.key,
                                                           limit=api_shared_concurrency)
        self.judge_governor.configure_healthcheck(self.agent.runtime.base_url, self.agent.key)
        self.api_solvers, self.api_solvers_per_model, self.api_judges = solver_workers, 1, api_judge_workers
        self.cli_solvers, self.cli_disabled = False, True
        self.auto_recover_infrastructure = True
        self.recovery_first_models, self.recovery_only_models = set(), set()
        self.model_cursor = 0
        self.judgment_failures = {}
        self.error_lock = threading.Lock()
        self._review_budget_lock = threading.Lock()
        self._review_budget = None
        quarantine = output / 'api_judgment_quarantine.json'
        if quarantine.exists():
            self.judgment_failures.update(collector.read(quarantine).get('requests', {}))

    def configure_review_budget(self, *, max_api_calls=None, max_wall_seconds=None,
                                stop_event=None):
        """Install a hard per-invocation budget for reviewer API calls.

        ``judgment_limit`` is an in-flight window and can refill forever in
        continuous mode.  This separate budget caps the actual provider calls
        and elapsed time so a dependency graph can never consume an unbounded
        number of retries in one terminal invocation.
        """
        if max_api_calls is not None and max_api_calls < 1:
            raise ValueError('max_api_calls must be positive')
        if max_wall_seconds is not None and max_wall_seconds <= 0:
            raise ValueError('max_wall_seconds must be positive')
        self._review_budget = {
            'max_api_calls': max_api_calls,
            'max_wall_seconds': max_wall_seconds,
            'api_calls': 0,
            'started_monotonic': time.monotonic(),
            'started_epoch': time.time(),
            'stop_event': stop_event,
            'status': 'ACTIVE',
        }
        collector.write(self.output / 'review_budget.json', {
            'status': 'ACTIVE', 'max_api_calls': max_api_calls,
            'max_wall_seconds': max_wall_seconds, 'api_calls': 0,
            'started_epoch': self._review_budget['started_epoch'],
        })

    def _reserve_review_call(self):
        budget = getattr(self, '_review_budget', None)
        if budget is None:
            return True
        with self._review_budget_lock:
            elapsed = time.monotonic() - budget['started_monotonic']
            limit_hit = (budget['max_api_calls'] is not None
                         and budget['api_calls'] >= budget['max_api_calls'])
            wall_hit = (budget['max_wall_seconds'] is not None
                        and elapsed >= budget['max_wall_seconds'])
            stop = budget.get('stop_event')
            if stop is not None and stop.is_set():
                return False
            if limit_hit or wall_hit:
                budget['status'] = 'EXHAUSTED'
                reason = 'max_api_calls' if limit_hit else 'max_wall_seconds'
                budget['reason'] = reason
                if stop is not None:
                    stop.set()
                collector.write(self.output / 'review_budget.json', {
                    'status': 'EXHAUSTED', 'reason': reason,
                    'max_api_calls': budget['max_api_calls'],
                    'max_wall_seconds': budget['max_wall_seconds'],
                    'api_calls': budget['api_calls'],
                    'elapsed_seconds': elapsed,
                    'finished_epoch': time.time(),
                })
                return False
            budget['api_calls'] += 1
            collector.write(self.output / 'review_budget.json', {
                'status': 'ACTIVE', 'max_api_calls': budget['max_api_calls'],
                'max_wall_seconds': budget['max_wall_seconds'],
                'api_calls': budget['api_calls'],
                'elapsed_seconds': elapsed,
                'updated_epoch': time.time(),
            })
            return True

    def review_budget_exhausted(self):
        """Return true when the scheduler must stop even without a new call."""
        budget = self._review_budget
        if budget is None:
            return False
        with self._review_budget_lock:
            if budget.get('status') == 'EXHAUSTED':
                return True
            limit = budget.get('max_wall_seconds')
            if limit is None or time.monotonic() - budget['started_monotonic'] < limit:
                return False
            budget['status'] = 'EXHAUSTED'
            budget['reason'] = 'max_wall_seconds'
            stop = budget.get('stop_event')
            if stop is not None:
                stop.set()
            collector.write(self.output / 'review_budget.json', {
                'status': 'EXHAUSTED', 'reason': 'max_wall_seconds',
                'max_api_calls': budget['max_api_calls'],
                'max_wall_seconds': limit, 'api_calls': budget['api_calls'],
                'elapsed_seconds': time.monotonic() - budget['started_monotonic'],
                'finished_epoch': time.time(),
            })
            return True

    def refresh_health(self, *, startup=False):
        # Scoring performs its own admission check; its outage must not block
        # submission of independent Claude cases in the collector's main loop.
        return True

    def judge_runner(self, **kwargs):
        if kwargs['model'] != 'gpt-6-astra' or kwargs.get('reasoning_effort') not in {'medium', 'high', 'xhigh'}:
            raise ValueError('Claude experiment requires GPT-6 Astra API with medium/high/xhigh reviewer effort')
        started = time.monotonic()
        budget = kwargs.get('timeout_seconds', 900)
        # A dead local proxy must not consume the full per-judgment timeout.
        # The caller's bounded invocation can retry later after infrastructure
        # recovery; waiting here only burns wall time without making a call.
        preflight_wait = min(budget, 20.0)
        while not self.reviewer_ready(kwargs['model']):
            if self.review_budget_exhausted():
                return dict(completed=False, returncode=None, timed_out=False,
                            transport='hybrid_scheduler', error='review_api_budget_exhausted',
                            ambiguous=False, api_http_attempts=0)
            if (self.reviewer_disabled(kwargs['model']) or time.monotonic() - started >= preflight_wait
                    or (self.output / 'STOP_REQUESTED').exists()):
                return dict(completed=False, returncode=None, timed_out=False, transport='hybrid_scheduler',
                            error='gpt_api_preflight_unavailable', ambiguous=False, api_http_attempts=0)
            time.sleep(min(1, max(0, budget - (time.monotonic() - started))))
        if not self._reserve_review_call():
            return dict(completed=False, returncode=None, timed_out=False,
                        transport='hybrid_scheduler', error='review_api_budget_exhausted',
                        ambiguous=False, api_http_attempts=0)
        remaining = max(.001, budget - (time.monotonic() - started))
        review_budget = getattr(self, '_review_budget', None)
        if review_budget is not None and review_budget.get('max_wall_seconds') is not None:
            remaining = min(remaining, max(.001, review_budget['max_wall_seconds'] -
                                           (time.monotonic() - review_budget['started_monotonic'])))
        return self.agent(**dict(kwargs, timeout_seconds=remaining))

    @staticmethod
    def _execution_failure_reason(work):
        """Return a reason when Claude's staged Python never executed successfully.

        A Claude CLI session can terminate cleanly even when every scoped Python
        call failed before running user code (for example when the host lacks
        Landlock ABI 3). Such a receipt must not enter scientific scoring.
        """
        journal = work / 'execution_journal.jsonl'
        if not journal.exists():
            return None
        rows = []
        for line in journal.read_text(errors='replace').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('event') == 'python_execution':
                rows.append(row)
        if not rows or any(row.get('returncode') == 0 for row in rows):
            return None
        errors = '\n'.join(str(row.get('stderr', '')) for row in rows)
        if 'Landlock ABI >= 3 required' in errors:
            return 'all staged Python executions failed: host Landlock ABI >=3 is unavailable'
        return 'all staged Python executions failed before producing valid evidence'

    def collect_one_api(self, slot_id):
        state = collector.read(self.output / 'collection_state.json')
        slot = collector.get_slot(state, slot_id)
        public = Path(state['public_root'])
        work = public / 'work' / slot_id
        stage_scope(work, public / 'bundles' / slot_id / 'case_input.json',
                    [public / 'raw' / name for name in slot['raw_files']])
        spec = collector.start(self.output, slot_id, agent_id='claude-code:' + slot_id)
        audit = self.output / 'claude_runs' / slot_id
        receipt = run_claude(spec['message'], spec['model'], work, audit, settings=self.settings,
            cli=self.cli, timeout_seconds=max(.01, spec['deadline_epoch'] - time.time()))
        execution_failure = self._execution_failure_reason(work)
        if receipt['completed'] and not execution_failure:
            self.governor.succeeded(spec['model'])
        outcome = 'COMPLETED' if receipt['completed'] and not execution_failure else 'INFRASTRUCTURE_INVALID'
        with console.PUBLICATION_LOCK:
            collector.finish(self.output, slot_id, outcome=outcome, reason=receipt.get('error'),
                             agent_id='claude-code:' + slot_id)
            row = self.import_claude(slot_id, receipt)
        if execution_failure:
            receipt['error'] = execution_failure
        if outcome == 'INFRASTRUCTURE_INVALID':
            self.governor.failed(spec['model'], receipt)
        return row

    def import_claude(self, slot_id, receipt):
        collector.import_slot(self.output, slot_id)
        with collector.locked(self.output):
            state = collector.read(self.output / 'collection_state.json')
            slot = collector.get_slot(state, slot_id)
            path = self.output / slot['run_record_path']
            record = RunRecord.from_dict(collector.read(path))
            if record.protocol_version == 'CLAUDE_CODE_SCOPED_MCP_V1':
                return slot
            provenance = dict(record.runtime_environment_fingerprint)
            provenance.pop('collaboration_agent_id', None)
            provenance.update(transport='third_party_claude_code', api_equivalent=False,
                requested_model=slot['model_id'], observed_models=receipt['observed_models'],
                reasoning_effort='max', cli_version=receipt['cli_version'],
                endpoint=configuration(self.settings)[0], session_id=receipt['thread_id'],
                tool_environment='Claude Code bare with three case MCP tools and confined Python',
                blinding=dict(fresh_context=True, staged_raw_copies=True, instruction_scoped_access=True,
                    kernel_filesystem_isolation=True, kernel_network_isolation=False,
                    python_network_syscalls_denied=True, isolation_scope='analysis child; no namespaces'),
                claude_protocol_sha256=collector.file_hash(self.output / 'claude_protocol.json'),
                receipt_sha256=collector.file_hash(self.output / 'claude_runs' / slot_id / 'receipt.json'),
                transport_timing=receipt['transport_timing'], provider_usage=receipt.get('provider_usage'),
                console_schedule=collector.read(self.output / 'console_schedule.json'),
                telemetry_unavailable=['pure_model_inference_seconds', 'hidden_provider_waits', 'actual_third_party_cost'],
                tool_call_count_scope='Journaled Python calls; raw Claude tool messages retained separately')
            record.runtime_environment_fingerprint = provenance
            record.trajectory = collector.read(path.parent / 'trajectory.json')
            record.trajectory[0] = dict(event='collaboration_provenance', **provenance)
            record.protocol_version = 'CLAUDE_CODE_SCOPED_MCP_V1'
            record.runtime_profile_id = 'claude-code-scoped-python-720s-48-v2'
            record.input_tokens = receipt['usage'].get('input_tokens')
            record.output_tokens = receipt['usage'].get('output_tokens')
            record.model_turn_count = receipt.get('model_turn_count')
            record.write_trajectory(path.parent / 'trajectory.json')
            record.write_json(path)
            slot.update(run_record_sha256=collector.file_hash(path), console_transport='third_party_claude_code')
            collector.save_state(self.output, state)
        console.emit(event='collected', slot_id=slot_id, model=slot['model_id'], status=slot['status'])
        return slot

    def recover(self):
        state = collector.read(self.output / 'collection_state.json')
        public = Path(state['public_root'])
        for slot in state['slots']:
            if not str(slot.get('agent_id', '')).startswith('claude-code:'):
                continue
            if slot.get('run_record_path') and collector.read(self.output / slot['run_record_path']).get('protocol_version') == 'CLAUDE_CODE_SCOPED_MCP_V1':
                continue
            path = self.output / 'claude_runs' / slot['slot_id'] / 'receipt.json'
            if not path.exists():
                raise RuntimeError('Unfinished Claude attempt without receipt; inspect claude_runs/' + slot['slot_id'])
            receipt = collector.read(path)
            if receipt['model'] != slot['model_id'] or receipt['started_epoch'] < slot['started_epoch']:
                raise ValueError('Claude receipt does not match trial')
            if slot['status'] == 'RUNNING':
                valid = receipt['completed'] and time.time() <= slot['started_epoch'] + collector.SOLVER_TIMEOUT_SECONDS
                execution_failure = self._execution_failure_reason(public / 'work' / slot['slot_id'])
                valid = valid and not execution_failure
                collector.finish(self.output, slot['slot_id'], outcome='COMPLETED' if valid else 'INFRASTRUCTURE_INVALID',
                    reason=None if valid else (execution_failure or 'Controller recovery after observation interruption'), agent_id=slot['agent_id'])
            self.import_claude(slot['slot_id'], receipt)


def write_report(output):
    report = ORIGINAL_REPORT(output)
    state = collector.read(output / 'collection_state.json')
    from flowintentbench.claude_efficiency import (
        build_claude_solver_efficiency, write_solver_efficiency_csv)
    efficiency, efficiency_rows = build_claude_solver_efficiency(report, state, output)
    report['solver_efficiency'] = efficiency
    report['answered_evaluation'] = efficiency['answered_evaluation']
    write_solver_efficiency_csv(output / 'reports/solver_efficiency.csv', efficiency_rows)
    report['claude_protocol'] = collector.read(output / 'claude_protocol.json')
    report['api_transport_usage'] = api_accounting(output)
    report['efficiency_interpretation'] = {
        'primary': 'solver_efficiency; successful target Claude sessions without observed faults',
        'solver_time': 'Claude Code duration_api_ms; excludes local Python/tool and evaluator time',
        'input_tokens': 'Includes uncached, cache-read and cache-creation tokens once; components remain in each receipt',
        'unobservable_limit': 'Upstream queueing and hidden provider-internal activity cannot be separated',
        'cross_transport_pooling': False,
        'reviewer_usage': 'GPT-6 reviewer usage is separate and never enters solver_efficiency'}
    collector.write(output / 'reports/experiment_report.json', report)
    with (output / 'reports/experiment_report.md').open('a') as stream:
        stream.write('\nClaude 时间为原始端到端耗时；CLI 内部重试及上游等待不可分离。质量达标效率仅作该环境下的描述性结果，不能当作纯推理效率。\n')
    from scripts.report_subagent_experiment import render_markdown
    note = (
        "\nPrimary solver efficiency is in `solver_efficiency` and "
        "`solver_efficiency.csv`. It includes only successful target Claude sessions "
        "without observed faults. GPT-6 review, host scoring, and local code/tool time "
        "are excluded. Provider-internal queueing remains unobservable.\n"
    )
    (output / 'reports/experiment_report.md').write_text(render_markdown(report) + note)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='all', choices=('all', 'prepare', 'status', 'evaluate'))
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--settings', type=Path, default=ROOT / 'settings.json')
    parser.add_argument('--api-config', type=Path, default=ROOT / 'config/yiapi.toml')
    parser.add_argument('--claude-bin')
    parser.add_argument('--solver-workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--api-judge-workers', type=int, choices=range(1, 9), default=2)
    parser.add_argument('--api-shared-concurrency', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--max-runs', type=int, default=2304, help='Per invocation cap including bounded infrastructure replacements')
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    if args.command == 'status':
        if not (output / 'collection_state.json').exists():
            console.emit(status='NOT_PREPARED', output=str(output))
            return 0
        state = collector.read(output / 'collection_state.json')
        report = output / 'reports/experiment_report.json'
        console.emit(collection=state['status_counts'], by_model={m:dict(Counter(s['status'] for s in state['slots'] if s['model_id']==m)) for m in MODELS},
                     scientifically_terminal=collector.read(report).get('scientifically_terminal_slot_count') if report.exists() else 0,
                     process=collector.read(output / 'console_process.json') if (output / 'console_process.json').exists() else None)
        return 0
    if args.max_runs < 1:
        parser.error('--max-runs must be positive')
    # Reject the other experiment before even writing a console-process marker.
    if output == console.DEFAULT_OUTPUT.resolve():
        parser.error('Claude must use its own output directory')
    existing = output / 'collection_state.json'
    if existing.exists() and tuple(collector.read(existing)['configuration']['models']) != MODELS:
        parser.error('Refusing to reuse another model experiment')
    configure_process()
    cli = resolve_cli(args.claude_bin)
    def configured_prepare(root):
        return prepare(root, settings=args.settings.resolve(), cli=cli, api_config=args.api_config.resolve())
    console.prepare = configured_prepare
    console.finalize_receipts = lambda root, state: state  # ClaudeController owns Claude receipts
    console.write_report = console.publication_transaction(write_report)
    with console.runner_lock(output):
        state = configured_prepare(output)
        if args.command == 'prepare':
            console.emit(prepared=state['total_slots'], models=MODELS, output=str(output))
            return 0
        console.HYBRID = ClaudeController(output, args.settings.resolve(), cli, args.api_config.resolve(), args.solver_workers,
            api_judge_workers=args.api_judge_workers, api_shared_concurrency=args.api_shared_concurrency)
        console.REPLAY_CACHE = IncrementalReplay()
        (output / 'STOP_REQUESTED').unlink(missing_ok=True)
        schedule = dict(mode='parallel', solver_concurrency=args.solver_workers, grader_concurrency=args.api_judge_workers,
            cli_solver_workers=args.solver_workers, api_judge_workers=args.api_judge_workers, codex_cli_judge_workers=0,
            api_http_concurrency=args.api_judge_workers, api_shared_concurrency=args.api_shared_concurrency,
            grading_dispatch='rolling atomic window; no round-drain barrier', solver_effort='max', reviewer_effort='mixed_medium_high',
            models=list(MODELS), reviewer_model='gpt-6-astra', started_epoch=time.time(),
            shared_gpt_endpoint_limit='Cross-process generation admission and 429 cooldown; Claude Code internal requests excluded')
        collector.write(output / 'console_schedule.json', schedule)
        collector.write(output / 'console_schedule_history' / f'{time.time_ns()}.json', schedule)
        if args.command == 'all':
            report = console.all_parallel(output, args.max_runs, 'gpt-6-astra', 32000, 32, judgment_workers=0)
        else:
            report = console.evaluate(output, 'gpt-6-astra', 32000, 32, judgment_workers=0)
        return 0 if report and report['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())

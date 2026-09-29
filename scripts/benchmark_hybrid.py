"""Mixed transport scheduling inside the existing console workflow."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import json
import os
from pathlib import Path
import signal
import threading
import time

try:
    from scripts.portable_fcntl import fcntl
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from portable_fcntl import fcntl

from scripts import collect_subagent_runs as collector
from scripts.third_party_benchmark_transport import APIGovernor, ThirdPartyAgent, network_configuration
from flowintentbench.model_runner import RunRecord


def retryable_judgment_error(row):
    """Retry known admission/rejection failures, never unknown generations."""
    receipt = row.get('transport') or {}
    if row.get('failure_kind') != 'api_transport' or receipt.get('ambiguous', True):
        return False
    error = str(receipt.get('error') or '')
    return any(kind in error for kind in (
        'preflight_required', 'preflight_unavailable', 'admission_unavailable',
        'admission_budget_exhausted', 'rate_limit_retries_exhausted',
        'rate_limit_wait_exceeds_budget', 'circuit_wait_exceeds_budget',
        'pacing_wait_exceeds_budget', 'request_lane_budget_exhausted'))


def cli_quota_exhausted(receipt, audit_dir):
    """Match explicit provider errors, never a model's answer or generic timeout."""
    if receipt.get('completed'):
        return False
    error = str(receipt.get('error') or '').lower()
    stderr = Path(audit_dir) / 'stderr.log'
    if stderr.exists():
        error += '\n' + stderr.read_text(errors='replace').lower()
    return any(marker in error for marker in (
        'usage_limit_reached', 'usage limit', 'insufficient_quota', 'quota_exceeded',
        'exceeded your current quota', 'weekly limit reached', 'weekly usage limit'))


def cli_rejected_before_generation(receipt, audit_dir, workdir):
    """Only explicit rejection with no generated events/artifacts may change channel."""
    if not cli_quota_exhausted(receipt, audit_dir) or receipt.get('final_text') or receipt.get('usage'):
        return False
    work = Path(workdir)
    if any((work / name).exists() for name in ('answer.md', 'execution_journal.jsonl')):
        return False
    events = Path(audit_dir) / 'events.jsonl'
    if not events.exists():
        return False
    try:
        rows = [json.loads(line) for line in events.read_text().split('\n') if line.strip()]
        return bool(rows) and all(row.get('type') in {'thread.started', 'turn.started', 'turn.failed', 'error'} for row in rows)
    except (ValueError, AttributeError):
        return False


def api_trial_outcome(receipt, *, wall_budget_exceeded=False):
    timing=receipt.get('transport_timing') or {}
    affected=timing.get('transport_affected',False)
    budget_error=wall_budget_exceeded or receipt.get('error')=='execution_budget_exhausted'
    if budget_error and (affected or timing.get('known_infrastructure_overhead_seconds',0)>1):
        return 'INFRASTRUCTURE_INVALID', 'Measured API/client overhead affected trial deadline; not a model capability failure'
    if receipt['completed'] and not wall_budget_exceeded:
        return 'COMPLETED', None
    if budget_error or receipt.get('error')=='output_truncated':
        return ('INFRASTRUCTURE_INVALID' if affected else 'MODEL_NONCOMPLETION'), receipt.get('error') or f'{collector.SOLVER_TIMEOUT_SECONDS}-second deadline exceeded'
    return 'INFRASTRUCTURE_INVALID', receipt.get('error')


@contextmanager
def task_claim(output, kind, identity):
    """OS lock prevents concurrent sampling; journal prevents silent crash replay."""
    directory = Path(output) / 'task_claims' / kind
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (identity + '.json')
    with (directory / (identity + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        if path.exists() and collector.read(path)['status'] in {'STARTED', 'QUARANTINED'}:
            yield False
            return
        receipt = dict(pid=os.getpid(), identity=identity, started_epoch=time.time(), status='STARTED')
        collector.write(path, receipt)
        try:
            yield True
        except BaseException:
            receipt.update(status='QUARANTINED', finished_epoch=time.time())
            collector.write(path, receipt)
            raise
        else:
            receipt.update(status='FINISHED', finished_epoch=time.time())
            collector.write(path, receipt)


class HybridController:
    def __init__(self, console, output, config_path, *, cli_judges=2, api_solvers=1, api_judges=1,
                 cli_solvers=True, api_http_concurrency=1, recovery_first_models=(),
                 recovery_only_models=(), auto_recover_infrastructure=False, api_solvers_per_model=1,
                 api_shared_concurrency=0):
        self.console, self.output = console, Path(output)
        self.governor = APIGovernor(self.output / 'api_health.json', max_concurrency=api_http_concurrency)
        self.agent = ThirdPartyAgent(config_path, self.governor)
        if api_shared_concurrency:
            from scripts.shared_api_admission import SharedAPIAdmission
            self.governor.shared_gate = SharedAPIAdmission(self.agent.runtime.base_url, self.agent.key,
                                                         limit=api_shared_concurrency)
        self.governor.configure_healthcheck(self.agent.runtime.base_url, self.agent.key)
        self.api_solvers, self.api_judges = api_solvers, api_judges
        self.api_solvers_per_model = api_solvers_per_model
        self.cli_solvers = cli_solvers
        self.recovery_first_models = set(recovery_first_models)
        self.recovery_only_models = set(recovery_only_models)
        self.auto_recover_infrastructure = auto_recover_infrastructure
        self.cli_judge_gate = threading.BoundedSemaphore(cli_judges)
        self.api_judge_gate = threading.BoundedSemaphore(max(1, api_judges))
        self.model_cursor = 0
        self.judgment_failures = {}
        self.error_lock = threading.Lock()
        self.cli_health_lock = threading.Lock()
        health = self.output / 'cli_health.json'
        self.cli_disabled = cli_judges == 0 or (bool(collector.read(health).get('quota_exhausted')) if health.exists() else False)
        quarantine = self.output / 'api_judgment_quarantine.json'
        if quarantine.exists():
            self.judgment_failures.update(collector.read(quarantine).get('requests', {}))

    def excluded_judgments(self):
        with self.error_lock:
            # Cooldowns and limits persist across restarts. This releases only
            # finished failed reviews with no final verdict, not HTTP retries.
            now = time.time()
            if now >= getattr(self, '_next_judgment_recovery_scan', 0):
                self._next_judgment_recovery_scan = now + 5
                self.judgment_recovery_waiting = False
                from scripts.api_judgment_recovery import recover_unreturned_review
                from scripts.run_codex_benchmark import exchange_for
                released = []
                inventory = self.output/'evaluation/pending_requests.json'
                active = ({r['request_id'] for group in collector.read(inventory)['by_operation'].values()
                           for r in group} if inventory.exists() else set())
                for rid, failure in list(self.judgment_failures.items()):
                    if rid not in active:
                        continue
                    recovery = recover_unreturned_review(self.output, exchange_for(self.output), rid, failure, now=now)
                    if recovery and recovery['status'] == 'WAITING_FOR_COOLDOWN':
                        self.judgment_recovery_waiting = True
                    elif recovery:
                        del self.judgment_failures[rid]
                        released.append(rid)
                if released:
                    self._recovered_judgments = getattr(self, '_recovered_judgments', set()) | set(released)
                    collector.write(self.output/'api_judgment_quarantine.json',
                                    dict(requests=self.judgment_failures, updated_epoch=now))
                    self.console.emit(event='review_replacements_authorized', request_ids=released,
                                      policy='Terminal unreturned review only; original artifacts retained')
            unresolved = dict(self.judgment_failures)
        for path in (self.output / 'task_claims/judgment').glob('*.json'):
            if collector.read(path)['status'] in {'STARTED', 'QUARANTINED'}:
                unresolved[path.stem] = {'reason': 'unfinished claim; inspect receipt before recovering'}
        return unresolved

    def take_recovered_judgments(self):
        """Consume explicit audited releases, never ordinary task-lock release."""
        with self.error_lock:
            released = getattr(self, '_recovered_judgments', set())
            self._recovered_judgments = set()
            return released

    def reviewer_ready(self, model):
        """Check admission before reserving a scientific request or its budget."""
        if not self.cli_disabled:
            return True
        governor = getattr(self, 'judge_governor', self.governor)
        return bool(governor.check_health() and governor.available(model))

    def reviewer_disabled(self, model):
        governor = getattr(self, 'judge_governor', self.governor)
        return self.cli_disabled and (governor.disabled or model in governor.disabled_models)

    def judge_runner(self, **kwargs):
        # API admission happens before starting its own wall-time budget. A
        # waiting CLI lane is not another model call and cannot duplicate a task.
        model = kwargs['model']
        if self.api_judges:
            self.refresh_health()
        if self.cli_disabled:
            return self.api_judge_after_quota(**kwargs)
        if self.api_judges and self.governor.available(model) and self.api_judge_gate.acquire(False):
            try:
                return self.agent(**kwargs)
            finally:
                self.api_judge_gate.release()
        queued=time.monotonic()
        budget=kwargs.get('timeout_seconds',900)
        if not self.cli_judge_gate.acquire(timeout=budget):
            return dict(completed=False,returncode=None,timed_out=False,
                        transport='hybrid_scheduler',error='reviewer_admission_budget_exhausted',ambiguous=False)
        try:
            remaining=budget-(time.monotonic()-queued)
            if remaining<=0:
                return dict(completed=False,returncode=None,timed_out=False,
                            transport='hybrid_scheduler',error='reviewer_admission_budget_exhausted',ambiguous=False)
            if self.cli_disabled:
                return self.api_judge_after_quota(**dict(kwargs,timeout_seconds=remaining))
            receipt = self.console.run_codex(**dict(kwargs,timeout_seconds=remaining))
            if not cli_quota_exhausted(receipt, kwargs['output_dir']):
                if not receipt.get('completed'):
                    # Preserve the CLI evidence while isolating scoring outages
                    # from authorized API solver work.
                    return dict(receipt, transport='hybrid_scheduler',
                                original_transport='codex_cli', ambiguous=True)
                return receipt
            self.note_cli_quota(kwargs['output_dir'])
            remaining = budget - (time.monotonic() - queued)
            if cli_rejected_before_generation(receipt, kwargs['output_dir'], kwargs['workdir']) and remaining > 0:
                result = self.api_judge_after_quota(**dict(kwargs, timeout_seconds=remaining,
                    output_dir=Path(kwargs['output_dir']) / 'quota_fallback'))
                result['prior_cli_receipt'] = receipt
                return result
            return dict(receipt, transport='hybrid_scheduler', ambiguous=True,
                        error='cli_quota_after_possible_generation; preserve attempt for review')
        finally:
            self.cli_judge_gate.release()

    def note_cli_quota(self, audit_dir):
        with self.cli_health_lock:
            self.cli_disabled = True
            collector.write(self.output / 'cli_health.json', dict(quota_exhausted=True,
                updated_epoch=time.time(), evidence_path=str(audit_dir),
                policy='CLI admission disabled across restart; API workers continue; explicit retry flag re-enables CLI'))

    def api_judge_after_quota(self, **kwargs):
        budget = kwargs.get('timeout_seconds', 900)
        started = time.monotonic()
        if not self.api_judges or not self.api_judge_gate.acquire(timeout=budget):
            return dict(completed=False, returncode=None, timed_out=False, transport='hybrid_scheduler',
                        error='api_reviewer_admission_unavailable', ambiguous=False)
        try:
            while not (self.refresh_health() and self.governor.available(kwargs['model'])):
                if (self.governor.disabled or kwargs['model'] in self.governor.disabled_models
                        or time.monotonic() - started >= budget
                        or (self.output / 'STOP_REQUESTED').exists()):
                    return dict(completed=False, returncode=None, timed_out=False, transport='hybrid_scheduler',
                                error='api_reviewer_preflight_unavailable', ambiguous=False)
                time.sleep(min(1, max(0, budget - (time.monotonic() - started))))
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                return dict(completed=False, returncode=None, timed_out=False, transport='hybrid_scheduler',
                            error='reviewer_admission_budget_exhausted', ambiguous=False)
            return self.agent(**dict(kwargs, timeout_seconds=remaining))
        finally:
            self.api_judge_gate.release()

    def refresh_health(self, *, startup=False):
        return self.governor.check_health(startup=startup)

    def note_judgment_errors(self, results):
        with self.error_lock:
            self._note_judgment_errors(results)

    def _note_judgment_errors(self, results):
        for result in results:
            if result.get('failure_kind') != 'api_transport':
                continue
            transport = result.get('transport', {})
            if transport.get('ambiguous'):
                self.judgment_failures[result['request_id']] = {
                    'reason': result.get('reason'), 'attempt_path': result.get('attempt_path'),
                    'status': 'UNKNOWN_PROVIDER_OUTCOME', 'epoch': time.time()}
        collector.write(self.output / 'api_judgment_quarantine.json',
                        dict(requests=self.judgment_failures, updated_epoch=time.time()))

    def collect_one_api(self, slot_id):
        console, output = self.console, self.output
        spec = collector.start(output, slot_id, agent_id='third-party:' + slot_id)
        state = collector.read(output / 'collection_state.json')
        work = Path(state['public_root']) / 'work' / slot_id
        audit = output / 'api_runs' / slot_id
        if (output / 'console_schedule.json').exists():
            collector.write(audit / 'schedule.json', collector.read(output / 'console_schedule.json'))
        receipt = self.agent(spec['message'], spec['model'], work, audit,
                             timeout_seconds=max(.001, spec['deadline_epoch']-time.time()))
        outcome, reason = api_trial_outcome(receipt,wall_budget_exceeded=time.time()>spec['deadline_epoch'])
        # Publish finish/import/provenance under the same barrier used by replay.
        with console.PUBLICATION_LOCK:
            collector.finish(output, slot_id, outcome=outcome, reason=reason,
                             agent_id='third-party:' + slot_id)
            return self.import_api(slot_id, receipt)

    def import_api(self, slot_id, receipt):
        output = self.output
        collector.import_slot(output, slot_id)
        with collector.locked(output):
            state = collector.read(output / 'collection_state.json')
            slot = collector.get_slot(state, slot_id)
            path = output / slot['run_record_path']
            record = RunRecord.from_dict(collector.read(path))
            if record.protocol_version == 'THIRD_PARTY_STAGED_V1':
                return slot
            provenance = dict(record.runtime_environment_fingerprint)
            provenance.update(transport=self.agent.transport_name, api_equivalent=False,
                provider=self.agent.runtime.provider, endpoint=self.agent.runtime.base_url,
                requested_model=slot['model_id'], conversation_id=receipt['thread_id'],
                api_receipt_sha256=collector.file_hash(output / 'api_runs' / slot_id / 'receipt.json'),
                hybrid_protocol_sha256=collector.file_hash(output / 'hybrid_protocol.json'),
                api_http_attempts=receipt.get('api_http_attempts', 0),
                effective_prompt_sha256=receipt.get('effective_prompt_sha256'),
                transport_instruction_sha256=receipt.get('transport_instruction_sha256'),
                transport_timing=receipt.get('transport_timing'),
                api_network=receipt.get('network'),
                console_schedule=collector.read(output / 'console_schedule.json'),
                tool_call_count_scope='Journaled API Python tool executions; HTTP attempts reported separately',
                tool_environment='Project Chat Completions adapter plus staged Python helper; distinct from Codex CLI')
            provenance.pop('collaboration_agent_id', None)
            record.runtime_environment_fingerprint = provenance
            record.trajectory = collector.read(path.parent / 'trajectory.json')
            record.trajectory[0] = dict(event='collaboration_provenance', **provenance)
            record.protocol_version = 'THIRD_PARTY_STAGED_V1'
            record.runtime_profile_id = 'third-party-staged-python-720s-48-v2'
            record.model_turn_count = receipt.get('model_turn_count')
            record.input_tokens = receipt['usage'].get('input_tokens')
            record.output_tokens = receipt['usage'].get('output_tokens')
            record.write_trajectory(path.parent / 'trajectory.json')
            record.write_json(path)
            slot.update(run_record_sha256=collector.file_hash(path), console_transport=self.agent.transport_name)
            collector.save_state(output, state)
        self.console.emit(event='collected', model=slot['model_id'], case=slot['case_id'],
                          trial=slot['trial_index'], status=slot['status'], transport=self.agent.transport_name)
        return slot

    def recover(self):
        """Only import a durable receipt. Never repeat an uncertain API request."""
        output = self.output
        state = collector.read(output / 'collection_state.json')
        for slot in state['slots']:
            if not str(slot.get('agent_id', '')).startswith('third-party:'):
                continue
            # Terminal slots have already been imported (including audited
            # infrastructure-invalid attempts); never demand a receipt or
            # resample them during a later restart.
            if slot.get('status') not in {'RUNNING', 'FINISHED'}:
                continue
            # Newly queued recovery slots are intentionally PENDING and have
            # neither a receipt nor a run record yet. They must be dispatched
            # by the collector; recovery only imports durable attempts.
            if not slot.get('run_record_path') and slot.get('status') not in {'RUNNING', 'FINISHED'}:
                continue
            path = output / 'api_runs' / slot['slot_id'] / 'receipt.json'
            run_record_path = slot.get('run_record_path')
            runpath = output / run_record_path if run_record_path else None
            if slot.get('run_record_sha256') and runpath and collector.read(runpath).get('protocol_version') == 'THIRD_PARTY_STAGED_V1':
                continue
            if not path.exists():
                raise RuntimeError('Unfinished API trial: inspect api_runs/' + slot['slot_id'] + '; do not resample')
            receipt = collector.read(path)
            if receipt.get('model') != slot['model_id'] or receipt.get('started_epoch',0) < slot['started_epoch']:
                raise RuntimeError('API receipt identity mismatch; inspect ' + str(path))
            if slot['status'] == 'RUNNING':
                # Finish time is observed now, never backdated after a crash.
                recoverable=receipt['completed'] and time.time() <= slot['started_epoch']+collector.SOLVER_TIMEOUT_SECONDS
                collector.finish(output, slot['slot_id'], outcome='COMPLETED' if recoverable else 'INFRASTRUCTURE_INVALID',
                                 reason=receipt.get('error') or (None if recoverable else 'Controller recovery after observation deadline'), agent_id=slot['agent_id'])
            self.import_api(slot['slot_id'], receipt)

    def queue_infrastructure(self, slot_ids=None):
        if not getattr(self, 'auto_recover_infrastructure', False):
            return None
        from scripts.recover_subagent_infrastructure import queue_imported_infrastructure
        with self.console.PUBLICATION_LOCK:
            result = queue_imported_infrastructure(self.output, slot_ids=slot_ids)
        self.console.emit(event='infrastructure_recovery_queue',
                          results=result['last_scan_results'],
                          current_blocked_slot_ids=result['current_blocked_slot_ids'])
        return result

    def collect(self, output, maximum, *, stop_event=None, on_ready=None, on_commit=None):
        c = self.console
        c.prepare(output)
        self.recover()
        state = c.finalize_receipts(output, collector.read(output / 'collection_state.json'))
        if any(s['status'] in {'RUNNING', 'FINISHED'} for s in state['slots']):
            raise RuntimeError('Unfinished trial; inspect its durable receipt before resuming')
        stop = stop_event if stop_event is not None else threading.Event()
        handlers = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        launched, active = 0, {}
        def stopping():
            return stop.is_set() or (output / 'STOP_REQUESTED').exists()
        def execute(slot_id, backend):
            with task_claim(output, 'solver', slot_id) as claimed:
                if not claimed:
                    raise RuntimeError('Unfinished or concurrent solver claim: ' + slot_id)
                if backend == 'api':
                    return self.collect_one_api(slot_id)
                row = c.collect_one(output, slot_id)
                audit = output / 'cli_runs' / slot_id
                receipt = collector.read(audit / 'receipt.json') if (audit / 'receipt.json').exists() else {}
                if cli_quota_exhausted(receipt, audit):
                    self.note_cli_quota(audit)
                    state = collector.read(output / 'collection_state.json')
                    work = Path(state['public_root']) / 'work' / slot_id
                    if row['status'] == 'INFRASTRUCTURE_INVALID' and cli_rejected_before_generation(receipt, audit, work):
                        from scripts.recover_subagent_infrastructure import recover_infrastructure_attempt
                        with c.PUBLICATION_LOCK:
                            with (output / 'orchestrator_events.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(event='CLI_QUOTA_REJECTED_BEFORE_GENERATION',
                                    slot_id=slot_id, model_id=row['model_id'], epoch=time.time(),
                                    evidence_path=str(audit), reason='Explicit quota rejection; no generated events or execution artifacts')) + '\n')
                            recover_infrastructure_attempt(output, slot_id)
                    row = dict(row, cli_quota_exhausted=True)
                return row
        try:
            if not stopping() and maximum > 0:
                self.queue_infrastructure()
            if on_ready:
                on_ready()
            with ThreadPoolExecutor(max_workers=2+self.api_solvers) as pool:
                while True:
                    if not stopping() and launched < maximum:
                        self.refresh_health()
                        state = collector.read(output / 'collection_state.json')
                        reserved = {entry[0] for entry in active.values()}
                        occupied = Counter((entry[1], entry[2]) for entry in active.values())
                        lanes = ([] if self.cli_disabled or not getattr(self, 'cli_solvers', True)
                                 else [('cli', model, None) for model in collector.MODELS])
                        api_active = sum(entry[1] == 'api' for entry in active.values())
                        planned = Counter()
                        available = self.api_solvers - api_active
                        cap = getattr(self, 'api_solvers_per_model', 1)
                        for _ in range(max(0, available)):
                            chosen = self.next_api_slot(state, reserved, occupied, planned, cap)
                            if chosen is None:
                                break
                            model, slot = chosen
                            reserved.add(slot['slot_id'])
                            lanes.append(('api', model, slot))
                            planned[model] += 1
                        for backend, model, selected in lanes:
                            if launched >= maximum or (backend == 'cli' and occupied[backend, model]):
                                continue
                            slot = selected if selected is not None else self.next_pending(state, model, reserved)
                            if slot is None:
                                continue
                            reserve = getattr(self.governor, 'reserve', None)
                            if backend == 'api' and reserve is not None and not reserve(model):
                                continue
                            reserved.add(slot['slot_id'])
                            active[pool.submit(execute, slot['slot_id'], backend)] = (slot['slot_id'], backend, model)
                            launched += 1
                            if backend == 'api':
                                self.model_cursor = (collector.MODELS.index(model) + 1) % len(collector.MODELS)
                    if not active:
                        # A shared API cooldown is temporary, not end-of-work.
                        state = collector.read(output / 'collection_state.json')
                        if not stopping() and launched < maximum and self.api_solvers and not self.governor.disabled and any(
                                model not in self.governor.disabled_models and self.next_pending(state, model, set())
                                for model in collector.MODELS):
                            stop.wait(1)
                            continue
                        break
                    done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        slot_id, backend, _ = active.pop(future)
                        row = future.result()
                        # The worker has finalized its claim and imported the
                        # terminal receipt before a replacement can be queued.
                        if row['status'] == 'INFRASTRUCTURE_INVALID' and not stopping():
                            self.queue_infrastructure([slot_id])
                        if on_commit:
                            on_commit()
                        if row['status']=='INFRASTRUCTURE_INVALID' and backend=='cli' and not row.get('cli_quota_exhausted'):
                            stop.set()
                        # Failed API observations remain terminal and auditable;
                        # unrelated CLI tasks continue instead of global abort.
        except BaseException:
            stop.set()
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        return launched

    def next_api_slot(self, state, reserved, occupied, planned, cap):
        # Give each ready model one lane before lending surplus lanes. A
        # recovering Terra must not sit behind a second long-running Luna.
        ordered = sorted(enumerate(collector.MODELS), key=lambda item: (
            occupied['api', item[1]] + planned[item[1]],
            (item[0] - self.model_cursor) % len(collector.MODELS)))
        for _, model in ordered:
            if occupied['api', model] + planned[model] >= cap or not self.governor.available(model):
                continue
            slot = self.next_pending(state, model, reserved)
            if slot is not None:
                return model, slot
        return None

    def next_pending(self, state, model, reserved):
        pending = [s for s in state['slots'] if s['model_id'] == model
                   and s['status'] == 'PENDING' and s['slot_id'] not in reserved]
        if model in getattr(self, 'recovery_only_models', ()):
            pending = [s for s in pending if s.get('recovery_attempt_index', 0)]
        recovery_first = model in getattr(self, 'recovery_first_models', ())
        # Stable ordering within each phase; Terra defaults to ordinary work
        # first, while the requested Luna lane prioritizes audited replacements.
        pending.sort(key=lambda s: bool(s.get('recovery_attempt_index', 0)) != recovery_first)
        return pending[0] if pending else None


def api_accounting(output):
    """Count actual HTTP receipts, including failures, separately from CLI usage."""
    output = Path(output)
    paths = list((output / 'api_runs').glob('*/http/*/http_receipt.json'))
    paths += list((output / 'judgment_work').glob('*/*/*/**/http/*/http_receipt.json'))
    paths += list((output / 'judgment_batches').glob('*/transport/**/http/*/http_receipt.json'))
    diagnostic_paths = list((output / 'diagnostics').glob('*/judgment_work/*/*/*/**/http/*/http_receipt.json'))
    diagnostic_paths += list((output / 'diagnostics').glob('*/judgment_batches/*/transport/**/http/*/http_receipt.json'))
    paths = sorted(set(paths + diagnostic_paths))
    groups = Counter()
    for path in paths:
        row = collector.read(path)
        groups[(row['model'], row['status'])] += 1
    return dict(http_attempts=len(paths), diagnostic_reviewer_http_attempts=len(set(diagnostic_paths)),
                by_model_status=[dict(model=m,status=s,count=n) for (m,s),n in sorted(groups.items())],
                scope='Benchmark solver and reviewer HTTP attempts, including targeted reviewer diagnostics; preflight excluded; requests with unknown outcomes included')


def verify_hybrid_protocol(output, config_path):
    from flowintentbench.runtime_config import resolve_provider_configuration
    path = Path(output) / 'hybrid_protocol.json'
    if not path.exists():
        raise RuntimeError('Register the reviewed hybrid protocol before enabling third-party API')
    saved = collector.read(path)
    runtime = resolve_provider_configuration(config_path=Path(config_path))
    if saved.get('api_network') != network_configuration():
        raise RuntimeError('API network configuration changed; register the reviewed protocol amendment')
    if runtime.without_secrets() != saved['api_runtime']:
        raise RuntimeError('Third-party endpoint/configuration changed; protocol amendment required')
    for name, expected in saved['sources'].items():
        if collector.file_hash(Path(__file__).parent / name) != expected:
            raise RuntimeError('Hybrid implementation changed; protocol amendment required: ' + name)


def annotate_transport_efficiency(report, state, output):
    """Preserve raw measures, exclude known service failures from primary timing.

    This is a new reporting view, not a rewrite of scientific results or the
    frozen quality threshold. Never assert that provider queue time is known.
    """
    from flowintentbench.quality_efficiency import quality_efficiency_summary
    output=Path(output)
    by_id={s['slot_id']:s for s in state['slots']}
    groups={}
    for row in report['trial_rows']:
        slot=by_id[row['slot_id']]
        if not slot.get('run_record_path'):
            continue
        run=collector.read(output/slot['run_record_path'])
        provenance=run.get('runtime_environment_fingerprint',{})
        transport=provenance.get('transport','unknown')
        timing=provenance.get('transport_timing') or {}
        adjusted_wall = None
        if timing.get('known_infrastructure_overhead_seconds') is not None and row.get('wall_clock_time') is not None:
            adjusted_wall=max(0.0,row['wall_clock_time']-timing['known_infrastructure_overhead_seconds'])
        row.update(launch_transport=transport,
                   transport_affected=timing.get('transport_affected'),
                   raw_wall_clock_time=row.get('wall_clock_time'),
                   known_infrastructure_overhead_seconds=timing.get('known_infrastructure_overhead_seconds'),
                   observed_time_excluding_known_overhead=adjusted_wall,
                   pure_model_inference_seconds=None)
        api=transport=='third_party_chat_completions'
        bucket=groups.setdefault(transport,{'all':[],'primary':[],'affected':[]})
        summary_row = row
        if report.get('mixed_scientific_evaluator_snapshots'):
            from flowintentbench.subagent_reporting import rows_for_evaluator_snapshot
            summary_row = rows_for_evaluator_snapshot([row], None)[0]
        bucket['all'].append(dict(summary_row))
        if api and (not timing or timing.get('transport_affected') or row['status']=='INFRASTRUCTURE_INVALID'):
            bucket['affected'].append(dict(summary_row))
        else:
            adjusted=dict(summary_row)
            if api:
                adjusted['wall_clock_time']=adjusted_wall
            bucket['primary'].append(adjusted)
    report['transport_efficiency']={name:dict(
        observed_count=len(g['all']), affected_count=len(g['affected']),
        raw=quality_efficiency_summary(g['all']),
        primary=quality_efficiency_summary(g['primary']) if g['primary'] else None,
        affected=quality_efficiency_summary(g['affected']) if g['affected'] else None,
        primary_time_definition='Observed time minus measured client waits and failed HTTP time' if name=='third_party_chat_completions' else 'Original CLI/collaboration wall-clock time; hidden service time not measurable'
        ) for name,g in groups.items()}
    report['primary_efficiency_policy']={
        'quality_gate':'Unchanged frozen quality policy; failed/low-quality answers never count as quality-qualified efficient successes',
        'service_failures':'Retain scientific results if completed; exclude API transport-affected observations from primary efficiency and report them separately',
        'missing_service_timing':'Unknown, never assumed zero',
        'cross_transport_pooling':False,
        'cross_evaluator_snapshot_quality_pooling':False,
        'legacy_quality_efficiency':'Raw end-to-end audit view; not primary transport-comparable efficiency',
        'pure_model_inference_time':'Unavailable; hidden upstream queueing cannot be subtracted'}

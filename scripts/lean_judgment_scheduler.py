"""Bounded reviewer scheduling; resolve dependencies after each completed call."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
import json
import threading
import time

from scripts.grading_policy import POLICY, POLICY_SHA256, batch_key, scientific_batch
from scripts.batched_file_judgments import judge_batch, packet_for
from flowintentbench.external_file_evaluator import write_json


def run_lean_judgments(exchange, inventory, work_root, reviewer_model, limit, timeout_seconds,
                       runner, stop_event, workers, request_order, excluded_ids,
                       continue_on_format_error, hybrid, on_resolved=None, *, continuous=False,
                       progress_guard=None):
    from scripts.run_codex_file_judgments import judge_request, visible_task_signature
    stop_event = stop_event if stop_event is not None else threading.Event()
    seen, excluded = set(), set(excluded_ids)
    held_by_transport = set()
    results, deferred, skipped_errors = [], set(), set()
    active_ids, skipped = set(), set()
    ranks = {rid: i for i, rid in enumerate(request_order or [])}
    calls = 0
    retry_at, retry_counts = {}, Counter()
    scheduling_state = 'READY'
    answer_affinity = {}
    answer_blockers = set()

    def answer_scope(request):
        context = request.get('context', {})
        key = tuple(context.get(k) for k in ('case_id', 'response_sha256', 'evaluation_manifest_sha256'))
        return key if all(key) else None

    def record_usage(rows):
        path = work_root.parent / 'judgment_usage.json'
        ledger = json.loads(path.read_text()) if path.exists() else dict(
            scope='Reviewer transports since lean policy activation; excludes solver usage and historical calls',
            seen_transports=[], physical_receipts=0, usage_missing=0,
            input_tokens=0, cached_input_tokens=0, output_tokens=0,
            worker_seconds=0.0, atomic_results=0)
        known = set(ledger['seen_transports'])
        # Breakdowns start with v5; do not pretend historical totals have the
        # same role mixture or use per-call time as complete-answer latency.
        ledger.setdefault('breakdown_since_epoch', time.time())
        def add_bucket(bucket, receipt, usage):
            bucket['calls'] = bucket.get('calls', 0) + 1
            for field in ('input_tokens', 'cached_input_tokens', 'output_tokens'):
                value = usage.get(field)
                if isinstance(value, int) and value >= 0:
                    bucket[field] = bucket.get(field, 0) + value
            if receipt.get('started_epoch') is not None and receipt.get('finished_epoch') is not None:
                start, end = receipt['started_epoch'], receipt['finished_epoch']
                bucket['worker_seconds'] = bucket.get('worker_seconds', 0) + max(0, end-start)
                bucket['first_started_epoch'] = min(bucket.get('first_started_epoch', start), start)
                bucket['last_finished_epoch'] = max(bucket.get('last_finished_epoch', end), end)
        for row in rows:
            ledger['atomic_results'] += int(row['status'] in {'RESOLVED', 'PENDING'})
            for transport in row.get('transports', []):
                receipts = [transport]
                if transport.get('prior_cli_receipt'):
                    receipts.append(transport['prior_cli_receipt'])
                for receipt in receipts:
                    key = receipt.get('events_path')
                    if not key or key in known:
                        continue
                    known.add(key)
                    ledger['physical_receipts'] += 1
                    usage = receipt.get('usage') or receipt.get('observed_usage') or {}
                    request_path = exchange / 'requests' / (row['request_id'] + '.json')
                    request = json.loads(request_path.read_text())
                    role = request['operation'] + (':' + request['input']['pending_type']
                        if request['input'].get('pending_type') else '')
                    add_bucket(ledger.setdefault('by_operation', {}).setdefault(role, {}), receipt, usage)
                    channel = receipt.get('transport', 'unknown')
                    add_bucket(ledger.setdefault('by_transport', {}).setdefault(channel, {}), receipt, usage)
                    context = request.get('context', {})
                    if context.get('response_sha256') and context.get('case_id'):
                        identity = context['case_id'] + ':' + context['response_sha256']
                        bucket = ledger.setdefault('by_answer', {}).setdefault(identity, {
                            'case_id': context['case_id'], 'response_sha256': context['response_sha256']})
                        add_bucket(bucket, receipt, usage)
                    ledger['usage_missing'] += int(not usage)
                    for field in ('input_tokens', 'cached_input_tokens', 'output_tokens'):
                        value = usage.get(field)
                        if isinstance(value, int) and value >= 0:
                            ledger[field] += value
                    if receipt.get('started_epoch') is not None and receipt.get('finished_epoch') is not None:
                        ledger['worker_seconds'] += max(0, receipt['finished_epoch'] - receipt['started_epoch'])
        ledger.update(seen_transports=sorted(known), updated_epoch=time.time(),
                      review_policy_sha256=POLICY_SHA256,
                      counting_note='Cached inputs are part of input tokens, not additional tokens; worker-time is not elapsed parallel time')
        write_json(path, ledger)

    def stopped():
        return stop_event.is_set() or (work_root.parent / 'STOP_REQUESTED').exists()

    def candidates():
        nonlocal held_by_transport
        for rid, deadline in list(retry_at.items()):
            if time.time() >= deadline:
                seen.discard(rid)
                del retry_at[rid]
        active = json.loads(inventory.read_text())
        ids = {r['request_id'] for rows in active['by_operation'].values() for r in rows}
        active_ids.clear()
        active_ids.update(ids)
        if hybrid is not None:
            currently_held = set(hybrid.excluded_judgments())
            take_recovered = getattr(hybrid, 'take_recovered_judgments', lambda: ())
            for rid in take_recovered():
                seen.discard(rid)  # Explicit audited replacement, not a silent HTTP resend.
                skipped_errors.discard(rid)
            held_by_transport = currently_held
        ready = []
        answer_blockers.clear()
        for rid in sorted(ids, key=lambda k: (ranks.get(k, len(ranks)), k)):
            response_path = exchange / 'responses' / (rid + '.json')
            blocked = rid in excluded or rid in held_by_transport
            if response_path.exists():
                blocked |= json.loads(response_path.read_text()).get('status') == 'PENDING'
            if blocked:
                request = json.loads((exchange/'requests'/(rid+'.json')).read_text())
                answer_blockers.add(answer_scope(request))
            if rid in seen:
                continue
            if rid in excluded or rid in held_by_transport:
                skipped_errors.add(rid)
                continue
            if len(rid) != 64 or any(c not in '0123456789abcdef' for c in rid):
                raise ValueError('Invalid request identity')
            if (exchange / 'responses' / (rid + '.json')).exists():
                skipped.add(rid)
                continue
            request = json.loads((exchange / 'requests' / (rid + '.json')).read_text())
            if request['request_id'] != rid:
                raise ValueError('Request identity differs from inventory')
            visible_task_signature(request)
            if (request['operation'] == 'adjudication' and request['input'].get('pending_type')
                    == 'novel_operationalization_materialization'):
                deferred.add(rid)
                continue
            ready.append(request)
        return ready

    def pick():
        # Continuous mode bounds in-flight atomic requests, not lifetime
        # dispatches. A completed worker can refill before slow siblings finish.
        remaining = limit - (sum(pending.values()) if continuous else len(seen))
        if remaining <= 0:
            return []
        ready = candidates()
        if not ready:
            return []
        # Continue an opened answer across recursive dependency stages before
        # spreading work over new answers. Blocked answers have no ready task
        # and cannot hold a worker. No scores enter this scheduling preference.
        ready.sort(key=lambda r: (answer_scope(r) in answer_blockers,
                                 answer_affinity.get(answer_scope(r), float('inf'))))
        group = ready[:1]
        scope = answer_scope(group[0])
        if scope is not None:
            answer_affinity.setdefault(scope, len(answer_affinity))
        key = batch_key(group[0])
        if key:
            batch_limit = POLICY['scientific_batch_size'] if scientific_batch(group[0]) else POLICY['batch_size']
            for request in ready[1:]:
                if len(group) >= min(remaining, batch_limit):
                    break
                if batch_key(request) == key:
                    proposed = group + [request]
                    if len(json.dumps(packet_for(proposed), ensure_ascii=False).encode()) <= POLICY['batch_max_prompt_bytes']:
                        group = proposed
        seen.update(r['request_id'] for r in group)
        return group

    def execute(group):
        with ExitStack() as stack:
            claimed, ignored = [], []
            for request in group:
                rid = request['request_id']
                if hybrid is not None:
                    from scripts.benchmark_hybrid import task_claim
                    if not stack.enter_context(task_claim(work_root.parent, 'judgment', rid)):
                        ignored.append(dict(request_id=rid, status='SKIPPED'))
                        continue
                if (exchange / 'responses' / (rid + '.json')).exists():
                    ignored.append(dict(request_id=rid, status='SKIPPED'))
                else:
                    claimed.append(request)
            if len(claimed) > 1:
                completed = judge_batch(claimed, exchange, work_root, reviewer_model, timeout_seconds, runner)
            elif claimed:
                completed = [judge_request(claimed[0], exchange, work_root, reviewer_model,
                                           timeout_seconds, runner, lean=True)]
            else:
                completed = []
            if hybrid is not None:
                hybrid.note_judgment_errors(completed)
            return completed + ignored

    def fatal(row):
        return row['status'] == 'ERROR' and not (
            (continue_on_format_error and row.get('failure_kind') == 'format') or
            (hybrid is not None and row.get('failure_kind') == 'api_transport'))

    def progress(running):
        counts = Counter(r['status'] for r in results)
        write_json(work_root.parent / 'judgment_progress.json', dict(
            updated_epoch=time.time(), active=len(active_ids), attempted=len(results),
            resolved=counts['RESOLVED'], pending=counts['PENDING'], errors=counts['ERROR'],
            skipped_existing=len(skipped), workers=workers, running=running,
            skipped_error_requests=sorted(skipped_errors), deferred_execution_requests=sorted(deferred & active_ids),
            scope='rolling_invocation' if continuous else 'current_batch',
            atomic_window_limit=limit, review_policy=dict(POLICY, sha256=POLICY_SHA256),
            dispatches=calls, atomic_requests_dispatched=len(seen),
            scheduling_state=scheduling_state, retry_waiting=len(retry_at),
            dependency_refresh='coalesced background replay; model dispatch remains nonblocking'))
        latest = {row['request_id']: row for row in results}
        write_json(work_root.parent / 'judgment_errors.json', dict(updated_epoch=time.time(),
            scope='rolling_invocation', errors=[{k: row.get(k) for k in
                ('request_id', 'failure_kind', 'reason', 'attempt_path')}
                for row in latest.values() if row['status'] == 'ERROR'],
            retry_after_epoch=dict(retry_at), scheduling_state=scheduling_state))

    def refresh_job():
        started = time.monotonic()
        return on_resolved(), time.monotonic()-started

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='lean-review') as pool, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix='review-replay') as replay_pool:
        pending = {}
        refresh_future = None
        revision, refreshed_revision = 0, 0
        refresh_revision = 0
        last_refresh = float('-inf')
        try:
            while True:
                if progress_guard is not None and progress_guard():
                    scheduling_state = 'NO_SCORE_PROGRESS'
                    stop_event.set()
                if (hybrid is not None and hasattr(hybrid, 'review_budget_exhausted')
                        and hybrid.review_budget_exhausted()):
                    scheduling_state = 'BUDGET_EXHAUSTED'
                    stop_event.set()
                if refresh_future is not None and refresh_future.done():
                    order, elapsed = refresh_future.result()
                    refresh_future = None
                    refreshed_revision = refresh_revision
                    last_refresh = time.monotonic()
                    path = work_root.parent / 'judgment_usage.json'
                    ledger = json.loads(path.read_text())
                    ledger['dependency_refresh_seconds'] = ledger.get('dependency_refresh_seconds', 0) + elapsed
                    ledger['dependency_refresh_count'] = ledger.get('dependency_refresh_count', 0) + 1
                    write_json(path, ledger)
                    if order is not None:
                        ranks = {rid: i for i, rid in enumerate(order)}
                # Keep model workers occupied while the single replay worker
                # updates the atomic inventory. Claims and response IDs prevent
                # duplicate judging even when the inventory is briefly stale.
                while len(pending) < workers and not stopped():
                    if (hybrid is not None and hasattr(hybrid, 'review_budget_exhausted')
                            and hybrid.review_budget_exhausted()):
                        scheduling_state = 'BUDGET_EXHAUSTED'
                        stop_event.set()
                        break
                    # Health recovery cannot depend on new solver commits.
                    if hybrid is not None and hasattr(hybrid, 'reviewer_ready'):
                        if not hybrid.reviewer_ready(reviewer_model):
                            scheduling_state = 'WAITING_FOR_API'
                            break
                    scheduling_state = 'READY'
                    group = pick()
                    if not group:
                        break
                    pending[pool.submit(execute, group)] = len(group)
                    calls += 1
                needs_refresh = on_resolved is not None and revision > refreshed_revision and not stopped()
                if (needs_refresh and refresh_future is None and
                        (not pending or time.monotonic()-last_refresh >= POLICY['dependency_refresh_min_interval_seconds'])):
                    refresh_revision = revision
                    refresh_future = replay_pool.submit(refresh_job)
                progress(len(pending))
                if not pending and refresh_future is None and not needs_refresh:
                    ready = candidates() if not stopped() else []
                    disabled = (hybrid is not None and hasattr(hybrid, 'reviewer_disabled')
                                and hybrid.reviewer_disabled(reviewer_model))
                    waiting_recovery = hybrid is not None and getattr(hybrid, 'judgment_recovery_waiting', False)
                    if not stopped() and not disabled and (retry_at or waiting_recovery or (ready and scheduling_state == 'WAITING_FOR_API')):
                        stop_event.wait(1)
                        continue
                    break
                waiting = set(pending) | ({refresh_future} if refresh_future is not None else set())
                if not waiting:
                    stop_event.wait(.25)
                    continue
                finished, _ = wait(waiting, timeout=.5, return_when=FIRST_COMPLETED)
                for future in finished & pending.keys():
                    pending.pop(future)
                    rows = future.result()
                    results.extend(rows)
                    from scripts.benchmark_hybrid import retryable_judgment_error
                    for row in rows:
                        rid = row['request_id']
                        if row['status'] == 'ERROR' and retryable_judgment_error(row) and retry_counts[rid] < 3:
                            retry_counts[rid] += 1
                            retry_at[rid] = time.time() + 120 * retry_counts[rid]
                    record_usage(rows)
                    revision += int(any(r['status'] == 'RESOLVED' for r in rows))
                    # Per-request output made a large run spend substantial
                    # time formatting and writing thousands of lines.  The
                    # compact policy already emits a durable round summary;
                    # retain verbose rows only for legacy debugging runs.
                    if not POLICY.get('compact_round_logs'):
                        for row in rows:
                            print(json.dumps({k: row[k] for k in ('request_id', 'status')}), flush=True)
                    if any(fatal(r) for r in rows):
                        stop_event.set()
        except BaseException:
            stop_event.set()
            raise

    progress(0)
    counts = Counter(r['status'] for r in results)
    return dict(active=len(active_ids), attempted=len(results), skipped_existing=len(skipped),
        resolved=counts['RESOLVED'], pending=counts['PENDING'], errors=counts['ERROR'],
        fatal_errors=sum(fatal(r) for r in results), results=results,
        made_progress=counts['RESOLVED'] > 0, workers=workers, dispatches=calls,
        skipped_error_requests=sorted(skipped_errors), deferred_execution_requests=sorted(deferred & active_ids),
        review_policy=dict(POLICY, sha256=POLICY_SHA256),
        direct_project_api_calls=sum(t.get('api_http_attempts', 0)
            for r in results for t in r.get('transports', [])))

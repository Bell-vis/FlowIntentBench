"""Process-local replay optimization; no scientific decisions or durable cache trust."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

try:
    from scripts.path_migration import resolve_repository_path
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from path_migration import resolve_repository_path


ROOT = Path(__file__).resolve().parents[1]

_STAGED_HASHES = {}
_ORDER_DATA = {}


def read(path):
    return json.loads(Path(path).read_text())


def local_path(path, relative_root=None):
    return resolve_repository_path(
        path, repository_root=ROOT, relative_root=relative_root
    )


def preserved_scored_results(output, paths):
    """Explicit stopped-run snapshot; never replace an already published score."""
    from flowintentbench.external_file_evaluator import digest
    registry = Path(output) / 'preserved_scored_evaluations.json'
    if not registry.exists():
        return {}
    document = read(registry)
    if document.get('sha256') != digest({k:v for k,v in document.items() if k != 'sha256'}):
        raise ValueError('Preserved score registry digest mismatch')
    active = {str(Path(path).resolve()): str(path) for path in paths}
    saved = {}
    for entry in document['entries']:
        run_path = local_path(entry['run_path'])
        run_key = str(run_path.resolve())
        if run_key not in active:
            continue
        for path, expected in ((run_path, entry['run_sha256']),
                               (local_path(entry['result']['evaluation_path']), entry['evaluation_sha256'])):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('Preserved answer/score hash mismatch: ' + str(path))
        if entry['result']['status'] != 'SCORED':
            raise ValueError('Only complete evaluations can be preserved')
        result = dict(entry['result'])
        result['evaluation_path'] = str(local_path(result['evaluation_path']))
        saved[active[run_key]] = result
    return saved


def file_stamp(path):
    try:
        stat = Path(path).stat()
    except FileNotFoundError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def tree_stamp(directory):
    return [(str(p), file_stamp(p)) for p in sorted(Path(directory).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts]


def bound_paths(value, _seen=None):
    if _seen is None:
        _seen = set()
    if isinstance(value, dict):
        for item in value.values():
            yield from bound_paths(item, _seen)
    elif isinstance(value, list):
        for item in value:
            yield from bound_paths(item, _seen)
    elif isinstance(value, str) and len(value) < 4096 and '\n' not in value:
        if value in _seen:
            return
        _seen.add(value)
        try:
            path = local_path(value)
            if path.is_file():
                yield str(path)
        except OSError:
            pass


def replay_guard(root, exchange):
    # Metadata invalidation includes ctime/inode, not just mtime. A new process
    # always performs a full replay; no disk cache can authorize skipping it.
    # Reuse hashes only within this process and while inode/size/mtime/ctime
    # remain unchanged. Staging avoids redundant chmod on read-only files.
    staged = []
    for path in sorted((exchange / 'review_data').rglob('*')):
        if path.is_file():
            stamp = file_stamp(path)
            cached = _STAGED_HASHES.get(str(path.resolve()))
            if cached is None or cached[0] != stamp:
                checksum = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        checksum.update(block)
                if file_stamp(path) != stamp:
                    raise ValueError('Staged data changed during hashing')
                cached = stamp, checksum.hexdigest()
                _STAGED_HASHES[str(path.resolve())] = cached
            staged.append((str(path), cached[1]))
    return [tree_stamp(root / name) for name in
            ('flowintentbench', 'scripts', 'datasets', 'experiments/expansion_v1_development')] + [
                staged]


class IncrementalReplay:
    def __init__(self):
        self.guard = None
        self.entries = {}
        self.bound_cache = {}

    def _bound_files(self, path):
        stamp = file_stamp(path)
        cached = self.bound_cache.get(str(path))
        if cached is None or cached[0] != stamp:
            cached = (stamp, read(path))
            self.bound_cache[str(path)] = cached
        # Recheck path existence: previously missing evidence may appear without
        # changing the immutable record that references it.
        return set(bound_paths(cached[1]))

    def _stamp(self, path, result, exchange):
        evaluation = local_path(result['evaluation_path'])
        payload = read(evaluation)
        bound = set(self._bound_files(path))
        trajectory = path.parent / 'trajectory.json'
        if trajectory.exists():
            bound.update(self._bound_files(trajectory))
        ids = set(payload.get('used_external_response_ids', ())) | set(payload.get('active_request_ids', ()))
        return (tree_stamp(path.parent), [(p, file_stamp(p)) for p in sorted(bound)],
                hashlib.sha256(evaluation.read_bytes()).hexdigest(),
                [(rid, file_stamp(exchange / 'requests' / (rid + '.json')),
                  file_stamp(exchange / 'responses' / (rid + '.json'))) for rid in sorted(ids)])

    def select(self, paths, exchange, guard):
        if self.guard != guard:
            # A newly staged file cannot change an already validated file or
            # an existing content-bound request. Keep prior entries on pure
            # additions; edits/removals and source/material changes invalidate.
            additions_only = False
            if (isinstance(self.guard, list) and isinstance(guard, list)
                    and len(self.guard) == len(guard) == 5 and self.guard[:-1] == guard[:-1]):
                previous, current = dict(self.guard[-1]), dict(guard[-1])
                additions_only = all(current.get(path) == checksum for path, checksum in previous.items())
            if not additions_only:
                self.entries.clear()
            self.guard = guard
        def check(path):
            entry = self.entries.get(str(path))
            try:
                unchanged = entry is not None and entry[1] == self._stamp(path, entry[0], exchange)
            except (OSError, ValueError, KeyError):
                unchanged = False
            return path, entry[0] if unchanged else None
        dirty, reusable = [], {}
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix='dependency-check') as pool:
            for path, result in pool.map(check, paths):
                if result is not None:
                    reusable[str(path)] = result
                else:
                    dirty.append(path)
        return dirty, reusable

    def remember(self, paths, results, exchange, *, progress=None):
        pairs = list(zip(paths, results, strict=True))
        def record(pair):
            path, result = pair
            if result['status'] != 'EVALUATOR_ERROR':
                return str(path), (result, self._stamp(path, result, exchange))
            return None
        entries = {}
        reported = time.monotonic()
        if progress:
            progress(0, len(pairs))
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix='dependency-cache') as pool:
            for count, entry in enumerate(pool.map(record, pairs), 1):
                if entry is not None:
                    entries[entry[0]] = entry[1]
                now = time.monotonic()
                if progress and (now - reported >= 1 or count == len(pairs)):
                    progress(count, len(pairs))
                    reported = now
        self.entries = entries


def restore_inventory(output, results, exchange):
    from flowintentbench.external_file_evaluator import write_json
    write_json(output / 'evaluation_index.json', dict(
        evaluation_mode='DEVELOPMENT_EVALUATION', formal_release=False, api_calls=0,
        submission_count=len(results), results=results))
    ids = sorted({rid for row in results for rid in row.get('active_request_ids', ())})
    operations = {}
    for rid in ids:
        path = exchange / 'requests' / (rid + '.json')
        request = read(path)
        operations.setdefault(request['operation'], []).append(dict(
            request_id=rid, request_path=str(path), pending_type=request['input'].get('pending_type')))
    write_json(output / 'pending_requests.json', dict(active_request_count=len(ids), by_operation=operations))


def ordered_requests(output):
    """Advance unblocked, later-stage answers; never inspect score values."""
    state_path = output / 'collection_state.json'
    index_path = output / 'evaluation/evaluation_index.json'
    if not state_path.exists() or not index_path.exists():
        return None
    age = {row['slot_id']: (row.get('finished_epoch') or row.get('started_epoch') or 0, row['slot_id'])
           for row in read(state_path)['slots']}
    from scripts.run_codex_benchmark import exchange_for
    exchange = exchange_for(Path(output))
    quarantine_path = Path(output)/'api_judgment_quarantine.json'
    quarantined = set(read(quarantine_path).get('requests', {})) if quarantine_path.exists() else set()
    stages = {'external_adjudication': 0, 'external_recipe_binding': 1,
              'external_semantic_match': 2, 'external_eligibility': 3, 'external_extraction': 4}
    ranked = []
    for result in read(index_path)['results']:
        if not result.get('active_request_ids'):
            continue
        path = str(local_path(result['evaluation_path']))
        stamp = file_stamp(path)
        cached = _ORDER_DATA.get(path)
        if cached is None or cached[0] != stamp:
            payload = read(path)
            run_id = payload.get('run_id') or payload.get('pending_record', {}).get('run_id')
            run_id = run_id or payload.get('evaluation_record', {}).get('run_id')
            cached = (stamp, run_id, len(payload.get('used_external_response_ids', [])))
            _ORDER_DATA[path] = cached
        run_id = result.get('run_id') or cached[1]
        active = result['active_request_ids']
        blocked = bool(set(active) & quarantined)
        unanswered = 0
        for rid in active:
            response = exchange/'responses'/f'{rid}.json'
            if response.exists():
                blocked |= read(response).get('status') == 'PENDING'
            else:
                unanswered += 1
        rank = (blocked, stages.get(result.get('pending_type'), 5),
                unanswered, -cached[2], age.get(run_id, (float('inf'), run_id or '')))
        for rid in result.get('active_request_ids', ()):
            ranked.append((rank, rid))
    return list(dict.fromkeys(rid for _, rid in sorted(ranked)))

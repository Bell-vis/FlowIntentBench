"""Cross-process HTTP admission for experiments sharing an endpoint/account.

OS locks release on process death; only owner PIDs and timing are persisted.
No request content or credentials are written to shared state.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

try:
    from scripts.portable_fcntl import fcntl
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from portable_fcntl import fcntl


class SharedAPIAdmission:
    def __init__(self, base_url, key, *, limit=4, root=None, min_interval=3., allow_limit_increase=False):
        if not 1 <= limit <= 8:
            raise ValueError('Shared HTTP concurrency must be between 1 and 8')
        identity = hashlib.sha256((base_url.rstrip('/') + '\0' + key).encode()).hexdigest()
        root = Path(root) if root else Path(__file__).resolve().parents[1] / 'outputs/.shared_api'
        self.directory = root / identity
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit, self.min_interval = limit, min_interval
        with self.state() as state:
            can_increase = (allow_limit_increase and state and state['limit'] < limit
                            and state['min_interval'] == min_interval)
            if state and (state['limit'] != limit or state['min_interval'] != min_interval) and not can_increase:
                raise ValueError('Shared endpoint limits differ across experiments; use the same limits')
            state.update(limit=limit, min_interval=min_interval)

    @contextmanager
    def state(self):
        with (self.directory / 'state.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.directory / 'state.json'
            value = json.loads(path.read_text()) if path.exists() else {}
            before = json.dumps(value)
            yield value
            encoded = json.dumps(value)
            if encoded != before or not path.exists():
                temp = path.with_suffix('.tmp')
                temp.write_text(encoded)
                temp.replace(path)

    def cooldown(self, seconds):
        with self.state() as state:
            state['cooldown_until'] = max(state.get('cooldown_until', 0), time.time() + seconds)

    def ready(self):
        """Read shared cooldown before a trial's own execution clock starts."""
        with self.state() as state:
            return time.time() >= max(state.get('cooldown_until', 0), state.get('next_request_epoch', 0))

    @contextmanager
    def admit(self, deadline):
        started, owned = time.monotonic(), None
        try:
            while owned is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Shared API admission deadline exceeded before sending')
                with self.state() as state:
                    now = time.time()
                    ready = now >= max(state.get('cooldown_until', 0), state.get('next_request_epoch', 0))
                    if ready:
                        for index in range(self.limit):
                            lock = (self.directory / f'lane_{index}.lock').open('a+')
                            try:
                                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                lock.close()
                                continue
                            owned = lock
                            lock.seek(0)
                            lock.truncate()
                            lock.write(json.dumps({'pid': os.getpid(), 'started_epoch': now}))
                            lock.flush()
                            state['next_request_epoch'] = now + self.min_interval
                            break
                if owned is None:
                    time.sleep(min(.2, max(0, deadline-time.monotonic())))
            yield time.monotonic() - started
        finally:
            if owned is not None:
                owned.close()

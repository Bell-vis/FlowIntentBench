"""Progress for local preparation, kept separate from provider and solver usage."""

import os
from pathlib import Path
import time
import sys

from flowintentbench.external_file_evaluator import write_json


class PreparationProgress:
    def __init__(self, output):
        self.path = Path(output) / 'preparation_progress.json'
        self.started = time.monotonic()
        self.phase = None
        self.phase_started = self.started

    def update(self, phase, *, status='RUNNING', completed=None, total=None):
        now = time.monotonic()
        if phase != self.phase:
            self.phase = phase
            self.phase_started = now
        elapsed = now - self.phase_started
        eta = None
        if completed and total is not None:
            eta = max(0, total - completed) * elapsed / completed
        payload = dict(
            pid=os.getpid(), updated_epoch=time.time(), phase=phase, status=status,
            elapsed_seconds=now - self.started, phase_elapsed_seconds=elapsed,
            completed=completed, total=total, phase_eta_seconds=eta,
            scope='Local preparation only; excluded from solver efficiency and API usage',
        )
        try:
            write_json(self.path, payload)
        except OSError as exc:
            print(f'Preparation snapshot unavailable: {type(exc).__name__}', file=sys.stderr, flush=True)

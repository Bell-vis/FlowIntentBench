"""Stop reviewer dispatch when complete answered-case scores stop progressing."""
import json
import math
from pathlib import Path
import time

from flowintentbench.external_file_evaluator import write_json


class ScoreProgressGuard:
    def __init__(self, output, timeout_seconds, *, clock=time.monotonic):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('No-score timeout must be finite and positive')
        self.output = Path(output)
        self.timeout = timeout_seconds
        self.clock = clock
        self.last_progress = None
        self.next_check = float('-inf')
        self.high_water = 0
        self.stopped = False

    def __call__(self):
        now = self.clock()
        if self.stopped or now < self.next_check:
            return self.stopped
        self.next_check = now + min(60, self.timeout)
        report = json.loads((self.output / 'reports/experiment_report.json').read_text(encoding='utf-8'))
        answered = report.get('answered_evaluation', {})
        complete = int(answered.get('quality_complete_count', 0))
        if self.last_progress is None or complete > self.high_water:
            self.last_progress = now
            self.high_water = max(self.high_water, complete)
        all_complete = bool(answered.get('answered_slot_count')) and complete == answered['answered_slot_count']
        self.stopped = not all_complete and now - self.last_progress >= self.timeout
        event = dict(status='STOP_REQUESTED' if self.stopped else 'ACTIVE',
                     reason='NO_COMPLETE_ANSWER_SCORE_PROGRESS' if self.stopped else None,
                     quality_complete_count=complete, high_water_count=self.high_water,
                     seconds_without_score_progress=now-self.last_progress,
                     timeout_seconds=self.timeout, updated_epoch=time.time(),
                     scope='Starts at reviewer dispatch; excludes initial preparation; never a solver metric')
        write_json(self.output / 'score_progress_guard.json', event)
        if self.stopped:
            try:
                with (self.output / 'STOP_REQUESTED').open('x', encoding='utf-8') as stream:
                    stream.write('NO_COMPLETE_ANSWER_SCORE_PROGRESS\n')
            except FileExistsError:
                pass
        return self.stopped

"""Observe local replay without changing the fingerprinted scoring workflow."""

import argparse
import threading
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from flowintentbench.external_file_evaluator import write_json
    from scripts import run_expansion_file_evaluation as replay

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run-record', type=Path, action='append', default=[])
    parser.add_argument('--exchange', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument(
        '--reuse-existing', action='store_true',
        help='Reuse validated evaluation files from an interrupted replay.',
    )
    args, _ = parser.parse_known_args()
    total = len(dict.fromkeys(args.run_record))
    completed = 0
    started = time.monotonic()
    original = replay.evaluate_submission
    progress_lock = threading.Lock()

    def reusable_payload(destination, response, record):
        """Reuse only results whose answer and active judge state are unchanged."""
        if not args.reuse_existing or not destination.exists():
            return None
        try:
            payload = replay.read_json(destination)
        except (OSError, ValueError):
            return None
        if payload.get('status') not in {'PENDING', 'SCORED'}:
            return None
        if payload.get('response_sha256') != replay.digest(response):
            return None
        if record is not None and payload.get('run_id') != record.run_id:
            return None
        # A newly resolved response is a state transition. Re-evaluate that
        # case; unanswered/PENDING requests remain safely reusable.
        for request_id in payload.get('active_request_ids', ()):
            response_path = args.exchange / 'responses' / (request_id + '.json')
            if not response_path.exists():
                continue
            try:
                reply = replay.read_json(response_path)
            except (OSError, ValueError):
                return None
            if reply.get('status') in {'RESOLVED', 'ERROR'}:
                return None
        return payload

    def publish():
        elapsed = time.monotonic() - started
        try:
            write_json(args.output / 'replay_case_progress.json', dict(
                pid=os.getpid(), parent_pid=os.getppid(), updated_epoch=time.time(),
                completed=completed, total=total, elapsed_seconds=elapsed,
                phase_eta_seconds=(total - completed) * elapsed / completed
                    if completed else None))
        except OSError as exc:
            print(f'Progress snapshot unavailable: {type(exc).__name__}', file=sys.stderr, flush=True)

    def observed(*args, **kwargs):
        nonlocal completed
        try:
            # evaluate_submission's public positional contract places the
            # immutable answer and destination at positions 3 and 6.
            if len(args) >= 7:
                cached = reusable_payload(Path(args[6]), args[3], args[4])
                if cached is not None:
                    return cached
            return original(*args, **kwargs)
        finally:
            with progress_lock:
                completed += 1
                publish()

    publish()
    replay.evaluate_submission = observed
    original_argv = sys.argv
    try:
        # The inner evaluator has its own parser and must not see this
        # wrapper-only control flag.
        if args.reuse_existing:
            sys.argv = [value for value in sys.argv if value != '--reuse-existing']
        return replay.main()
    finally:
        sys.argv = original_argv
        replay.evaluate_submission = original


if __name__ == '__main__':
    raise SystemExit(main())

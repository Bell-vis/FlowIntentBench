#!/usr/bin/env python3
"""Collect independent rounds: Luna 2 -> retry Terra -> Luna 3 if still limited.

Only raw responses are collected here. run_userstudy.py must subsequently
score and audit every requested observation before declaring the study complete.
"""
from __future__ import annotations
import argparse
import fcntl
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.repeated_experiment import read_json, file_hash, run_collection_fallback
from flowintentbench.experiment_scope import case_inventory
from flowintentbench.model_runner import EvaluationTarget
from flowintentbench.provider_retry import failure_category
from scripts import run_full_dataset_n1_experiment as canonical
from scripts.run_full_dataset_n1_pilot import _validate_observation


def inspect_collection(collection, manifest, model):
    state_path = collection / 'collection_state.json'
    if not state_path.is_file():
        return {'status': 'INCOMPLETE', 'complete_case_count': 0, 'record_hashes': []}
    state = read_json(state_path)
    if state['model_id'] != 'gpt-5.6-' + model:
        raise RuntimeError('collection model mismatch')
    target = EvaluationTarget(state['provider'], state['model_id'], model_configuration=state['model_configuration'])
    expected = case_inventory(manifest)
    seen, hashes, failures = set(), [], []
    for observation in state['observations']:
        case_id = observation['case_id']
        if case_id in seen or case_id not in expected:
            raise RuntimeError('duplicate or unknown collection case')
        seen.add(case_id)
        record = _validate_observation(observation, output_root=collection, expected_case=expected[case_id], target=target, target_fingerprint=state['target_fingerprint'])
        if record.run_status.value == 'COMPLETED':
            folder = (collection / observation['run_record_path']).parent
            if not (folder / 'final_answer.md').read_text().strip() or not (folder / 'trajectory.json').is_file():
                raise RuntimeError('completed response artifacts missing')
            if record.network_isolation_active is not False or record.runtime_profile_id != 'flow-python-host-network-v1':
                raise RuntimeError('collection host-network condition mismatch')
            hashes.append(observation['run_record_sha256'])
        elif record.failure_reason:
            failures.append(record.failure_reason)
    blocks = sorted(collection.rglob('incomplete_external_block.json'), key=lambda p: p.stat().st_mtime)
    if blocks:
        failures.append(read_json(blocks[-1]).get('failure_message', ''))
    return {'status': 'COLLECTION_COMPLETE' if len(hashes) == len(expected) and seen == set(expected) else 'INCOMPLETE',
            'complete_case_count': len(hashes), 'record_hashes': hashes,
            'failure_category': failure_category(failures[-1]) if failures else None,
            'output_root': str(collection)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'experiments/userstudy/case_manifest.json')
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    root, manifest_path = args.output_root.resolve(), args.manifest.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read_json(manifest_path)
    suite_path = root / 'suite_preflight.json'
    canonical._assert_preflight_binding(read_json(suite_path), manifest_file=manifest_path, repository_root=ROOT)
    if 'not sandbox' not in read_json(suite_path)['frozen_tests']['command']:
        raise RuntimeError('host-network preflight required')
    initial_results = []
    for model in ('luna', 'terra'):
        for number in range(1, args.rounds + 1):
            existing = inspect_collection(root / f'round-{number:02d}' / model / 'initial/collection', manifest, model)
            if existing['status'] != 'COLLECTION_COMPLETE':
                break
            initial_results.append({**existing, 'round': number, 'model': model})
    def execute(number, model):
        folder = root / f'round-{number:02d}' / model
        collection = folder / 'initial/collection'
        previous = inspect_collection(collection, manifest, model)
        if previous['status'] == 'COLLECTION_COMPLETE':
            return previous
        # Preserve immutable collection identity and failed-attempt history.
        # Only this SAME model/round can seed its next attempt.
        source = None
        if collection.exists():
            source = folder / 'collection_history' / f'collection-{time.time_ns()}'
            source.parent.mkdir(parents=True, exist_ok=True)
            collection.rename(source)
        config = read_json(ROOT / f'config/evaluator_for_{model}.json')
        command = canonical._collection_command(manifest_path=manifest_path, collection_root=collection,
            agent=f'gpt-5.6-{model}-xhigh-chat-host-network', config=config, resume=source is not None,
            resume_from=source, preflight_report=suite_path, skip_preflight_tests=True)
        # Allow the alternate model to run after bounded retries, instead of
        # spending the same exhausted-provider attempts on all remaining cases.
        command.remove('--continue-on-infrastructure-exhaustion')
        launch = folder / f'collection_launch-{time.time_ns()}'
        canonical._atomic_json(launch.with_suffix('.json'), {'command': command, 'round': number,
            'model': model, 'same_round_resume_from': str(source) if source else None,
            'case_manifest_sha256': file_hash(manifest_path)})
        code = canonical._run_command(command, log_path=launch.with_suffix('.log'))
        result = inspect_collection(collection, manifest, model)
        result['script_exit_code'] = code
        return result
    status = run_collection_fallback(rounds=args.rounds, execute=execute, initial_results=initial_results, expected_case_count=len(case_inventory(manifest)),
        persist=lambda value: canonical._atomic_json(root / 'collection_schedule_status.json', value))
    print(status['status'], status['completed_collection_rounds'], flush=True)
    return 0 if status['status'] == 'COLLECTION_COMPLETE' else 2

if __name__ == '__main__':
    raise SystemExit(main())

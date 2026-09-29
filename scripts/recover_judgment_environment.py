"""Archive a proven dependency-failure PENDING, without deciding its science."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_codex_file_judgments import dependency_failure, sha256, visible_task_signature, _exclusive_json


def recover(exchange, request_id, *, apply=False):
    import re
    if not re.fullmatch('[0-9a-f]{64}', request_id):
        raise ValueError('invalid request id')
    exchange = Path(exchange)
    request_path = exchange / 'requests' / (request_id + '.json')
    request = json.loads(request_path.read_text())
    visible_task_signature(request)
    response_path = exchange / 'responses' / (request_id + '.json')
    response_hash = sha256(response_path)
    response = json.loads(response_path.read_text())
    if response.get('status') != 'PENDING' or response.get('request_id') != request_id:
        raise ValueError('Only a proven infrastructure PENDING can be recovered')
    audit_path = Path(response['provenance']['audit_path'])
    audit = json.loads(audit_path.read_text())
    if audit.get('request_id') != request_id or audit.get('status') != 'PENDING':
        raise ValueError('Missing matching original reviewer audit')
    evidence = response['provenance']['evidence_files']
    if evidence != audit.get('evidence_files'):
        raise ValueError('Evidence must match original audit')
    for item in evidence:
        path = Path(item['path'])
        if (path.is_symlink() or not path.resolve().is_relative_to(audit_path.parent.resolve())
                or sha256(path) != item['sha256']):
            raise ValueError('Original dependency evidence changed')
    failure = dependency_failure(evidence)
    if not failure or failure['module'].split('.')[0] not in {'vtkmodules', 'vtk', 'numpy', 'scipy'}:
        raise ValueError('No explicit recoverable scientific-library dependency failure')
    probe = subprocess.run([sys.executable, '-I', '-c',
        'import importlib,sys; importlib.import_module(sys.argv[1]); print(sys.executable)', failure['module']],
        capture_output=True, text=True, timeout=60)
    if probe.returncode:
        raise ValueError('Dependency is still unavailable in the authorized runtime: ' + probe.stderr)
    directory = exchange / 'infrastructure_recovery' / request_id / response_hash
    event = dict(request_id=request_id, original_response_sha256=response_hash,
        original_request_sha256=sha256(request_path), original_audit_path=str(audit_path),
        failure=failure, verified_python=sys.executable, runtime_probe=probe.stdout.strip(),
        status='READY_TO_REVIEW_AGAIN', scientific_judgment=None, applied=apply, epoch=time.time(),
        archive=str(directory / 'original_response.json'))
    if apply:
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / 'original_response.json'
        if not archive.exists():
            os.link(response_path, archive)
        if sha256(archive) != response_hash or sha256(response_path) != response_hash:
            raise ValueError('Response changed during recovery')
        _exclusive_json(directory / ('recovery_' + str(time.time_ns()) + '.json'), event)
        # Preserve the exact old envelope in the archive. The request becomes
        # unanswered again; only a fresh independent reviewer can resolve it.
        response_path.unlink()
    return event


if __name__ == '__main__':
    from scripts.run_codex_benchmark import DEFAULT_OUTPUT, exchange_for, runner_lock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    with runner_lock(args.output_root):
        print(json.dumps(recover(exchange_for(args.output_root), args.request_id, apply=args.apply), ensure_ascii=False))

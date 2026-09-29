#!/usr/bin/env python3
"""Freeze, run, return and locally rescore distributed saved-answer evaluations."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ('deepseek-flash', 'qwen3.8-max')
MANIFEST = 'experiments/expansion_v1_development/case_manifest.json'
BANK = 'outputs/model_answer_evaluation/frozen_auxiliary_references.json'
REFERENCE = 'outputs/model_answer_evaluation_v4/reference_construction/width_scope_v1.json'
COLLECTION = 'outputs/expansion96_n1_deepseek_qwen_host_network'
SCHEMA = 'distributed-answer-evaluation-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def key(row):
    return row['model_id'], row['case_id'], row.get('trial', row.get('trial_index'))


def safe(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative or not p.parts:
        raise ValueError(f'unsafe relative path: {relative}')
    result = (Path(root) / relative).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError('path escapes root')
    return result


def inventory(root, paths):
    return [{'path': str(p.relative_to(root)), 'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(paths)]


def check_files(root, files):
    seen = set()
    for f in files:
        if f['path'] in seen:
            raise ValueError('duplicate file in manifest')
        seen.add(f['path'])
        p = safe(root, f['path'])
        if not p.is_file() or p.stat().st_size != f['bytes'] or sha(p) != f['sha256']:
            raise ValueError(f"file/hash mismatch: {f['path']}")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def jsonl(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values), encoding='utf-8')


def archive(root, destination, paths):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError(f'archive already exists: {destination}')
    with tarfile.open(destination, 'w:gz') as tf:
        for p in sorted(paths):
            tf.add(p, arcname=str(p.relative_to(root)), recursive=False)
    destination.with_name(destination.name + '.sha256').write_text(f'{sha(destination)}  {destination.name}\n')


def freeze_dependencies():
    from packaging.requirements import Requirement
    # flowintentbench.__init__ imports construction modules eagerly, even when
    # the evaluator only consumes frozen JSON and never opens a mesh.
    todo = ['numpy', 'pydantic', 'PyYAML', 'packaging', 'jsonschema', 'vtk', 'scipy', 'matplotlib', 'h5py']
    found = {}
    while todo:
        name = todo.pop()
        d = importlib.metadata.distribution(name)
        name = d.metadata['Name']
        if name in found:
            continue
        found[name] = d.version
        for raw in d.requires or []:
            req = Requirement(raw)
            if req.marker is None or req.marker.evaluate({'extra': ''}):
                todo.append(req.name)
    return found


def build(args):
    root = ROOT
    refs = read(args.reference_manifest)
    listed = Path(args.reference_files).read_text(encoding='utf-8-sig').splitlines()
    if len(listed) != len(set(listed)) or set(listed) != {f['path'] for f in refs['files']}:
        raise ValueError('reference file list and hash manifest differ')
    check_files(root, refs['files'])
    source = root / COLLECTION
    state = read(source / 'collection_state.json')
    tasks = read(root / MANIFEST)
    expected = {(m, c['case_id'], 1) for m in MODELS for c in tasks['cases']}
    actual = [key(s) for s in state['slots']]
    if len(expected) != 192 or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('collection is not the complete 192-slot N=1 ledger')
    if state['configuration']['manifest_sha256'] not in {sha(root / MANIFEST), tasks.get('source_manifest_sha256')}:
        raise ValueError('collection task manifest differs from frozen evaluation tasks')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    def copy(relative):
        src, dst = safe(root, relative), safe(output, relative)
        if not src.is_file():
            raise ValueError(f'missing source: {src}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for f in refs['files']:
        copy(f['path'])
    # Snapshot the working tree, including uncommitted code and lazy imports.
    for directory in ('flowintentbench', 'scripts'):
        for p in (root / directory).rglob('*.py'):
            if '__pycache__' not in p.parts:
                copy(str(p.relative_to(root)))
    for p in ('pyproject.toml', 'runtime-requirements.txt', 'config/yiapi_direct.toml', 'config/path_aliases.json'):
        if (root / p).is_file():
            copy(p)
    for p in (root / 'runtime_profiles').glob('*.yaml'):
        copy(str(p.relative_to(root)))
    for model in ('deepseek-v4.1-flash-max-host-network', 'qwen3.8-max-xhigh-host-network'):
        copy(f'agents/{model}/agent.yaml')
    exported = []
    for slot in sorted(state['slots'], key=key):
        if slot['status'] != 'COMPLETED':
            raise ValueError('incomplete collection slot')
        p = safe(source, slot['run_record_path'])
        record = read(p)
        if sha(p) != slot['run_record_sha256'] or key(record) != key(slot) or record['run_status'] != 'COMPLETED':
            raise ValueError('original run record mismatch')
        answer_path = p.with_name('final_answer.md')
        answer = answer_path.read_text(encoding='utf-8')
        answer_hash = hashlib.sha256(answer.encode()).hexdigest()
        if not answer.strip() or sha(answer_path) != record['final_answer_sha256'] or answer_hash != record['final_answer_sha256']:
            raise ValueError('original answer mismatch')
        if record.get('final_response') != answer or not isinstance(read(p.with_name('trajectory.json')), list):
            raise ValueError('run answer/trajectory mismatch')
        exported.append({'model_id': slot['model_id'], 'case_id': slot['case_id'], 'trial': 1,
                         'collection_status': slot['status'], 'answer': answer, 'answer_sha256': answer_hash,
                         'run_id': record['run_id'], 'dataset_id': slot['dataset_id'],
                         'source_run_record': str(p.relative_to(root)), 'source_run_sha256': sha(p),
                         'attempt_count': len(slot.get('attempt_history', [])) + 1})
    # Original final and prior attempts are retained for solver provenance/costs.
    for p in (source / 'runs').rglob('*'):
        if p.is_file() and p.name in {'run_record.json', 'final_answer.md', 'trajectory.json'}:
            copy(str(p.relative_to(root)))
    for name in ('collection_state.json', 'collection_config.json', 'completion_summary.json',
                 'deepseek-flash_answers.jsonl', 'qwen3.8-max_answers.jsonl'):
        copy(f'{COLLECTION}/{name}')
    jsonl(output / 'handoff/answers/all.jsonl', exported)
    assignments = []
    for model in MODELS:
        selected = [r for r in exported if r['model_id'] == model]
        for index in range(args.shards_per_model):
            part = selected[index::args.shards_per_model]
            if not part:
                raise ValueError('empty shard')
            sid = f'{model}-{index + 1:02d}'
            path = f'handoff/answers/{sid}.jsonl'
            jsonl(output / path, part)
            assignments.append({'id': sid, 'answers': path, 'slots': [list(key(r)) for r in part],
                                'answer_count': len(part), 'answers_sha256': sha(output / path)})
    write(output / 'handoff/assignments.json', {'schema': SCHEMA, 'shards': assignments, 'total_slots': len(exported)})
    write(output / 'handoff/reference_manifest.json', refs)
    (output / 'handoff/reference_files.txt').write_text('\n'.join(listed) + '\n')
    deps = freeze_dependencies()
    (output / 'handoff/requirements.lock.txt').write_text(''.join(f'{n}=={v}\n' for n, v in sorted(deps.items())))
    policy = {'schema': SCHEMA, 'model': 'gpt-6-astra', 'effort': 'medium', 'max_output_tokens': 6000,
              'max_prompt_chars': 200000, 'structured_output': False, 'workers': 2, 'shared_concurrency': 2,
              'timeout': 420, 'max_wall_seconds': 14400, 'max_api_calls_per_invocation': 96,
              'tools': [], 'boundary': 'ANSWER_AND_FROZEN_EVIDENCE_ONLY'}
    write(output / 'handoff/judge.json', policy)
    write(output / 'handoff/auth.example.json', {'OPENAI_API_KEY': 'REPLACE_ON_REMOTE_SERVER'})
    for name in ('README.md', 'AGENTS.md', 'setup.sh'):
        shutil.copyfile(root / 'config/distributed_evaluation' / name, output / name)
        copy('config/distributed_evaluation/' + name)
    write(output / 'handoff/build_environment.json', {'python': platform.python_version(), 'platform': platform.platform(),
        'packages': deps, 'created_utc': datetime.now(timezone.utc).isoformat(), 'source_root': str(root),
        'reference_input_sha256': sha(args.reference_manifest), 'reference_list_input_sha256': sha(args.reference_files)})
    files = inventory(output, [p for p in output.rglob('*') if p.is_file()])
    bundle = {'schema': SCHEMA, 'files': files, 'slot_count': len(exported), 'shards': len(assignments)}
    write(output / 'handoff/bundle_manifest.json', bundle)
    archive(output, output.with_name(output.name + '.tar.gz'), [p for p in output.rglob('*') if p.is_file()])
    print(json.dumps({'bundle': str(output), 'archive': str(output) + '.tar.gz', 'answers': len(exported),
                      'reference_files': len(refs['files']), 'shards': len(assignments)}, ensure_ascii=False))


def verify(root=ROOT, *, environment=True):
    doc = read(root / 'handoff/bundle_manifest.json')
    if doc['schema'] != SCHEMA:
        raise ValueError('unsupported bundle')
    check_files(root, doc['files'])
    if environment:
        if sys.version_info[:2] != (3, 12):
            raise ValueError('Python 3.12 is required')
        for name, version in read(root / 'handoff/build_environment.json')['packages'].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f'dependency version mismatch: {name}=={version}')
    return doc


def assignment(sid):
    return next(s for s in read(ROOT / 'handoff/assignments.json')['shards'] if s['id'] == sid)


def evaluator_args(answer_files, output, *, offline=True, auth=None, calls=None):
    from scripts import evaluate_model_answers as ev
    p = read(ROOT / 'handoff/judge.json')
    argv = ['--manifest', str(ROOT / MANIFEST), '--evidence-bank', str(ROOT / BANK),
            '--reference-package', str(ROOT / REFERENCE), '--output', str(output),
            '--reviewer-model', p['model'], '--effort', p['effort'], '--max-output-tokens', str(p['max_output_tokens']),
            '--max-prompt-chars', str(p['max_prompt_chars']), '--workers', str(p['workers']),
            '--shared-concurrency', str(p['shared_concurrency']), '--timeout', str(p['timeout']),
            '--max-wall-seconds', str(p['max_wall_seconds']), '--max-api-calls',
            str(0 if offline else (calls if calls is not None else p['max_api_calls_per_invocation'])),
            '--api-config', str(ROOT / 'config/yiapi_direct.toml')]
    for path in answer_files:
        argv += ['--answers', str(path)]
    if offline:
        argv += ['--offline']
    if auth:
        argv += ['--auth-path', str(Path(auth).resolve())]
    if p['structured_output']:
        argv += ['--structured-output']
    return ev.parser().parse_args(argv)


def run_evaluator(args):
    from scripts import evaluate_model_answers as ev
    from scripts.portable_fcntl import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.trusted_evaluation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return ev.run(args)


def run_shard(args):
    verify()
    s = assignment(args.shard)
    output = ROOT / 'outputs/distributed_eval' / s['id']
    if not args.offline and not args.auth:
        raise ValueError('--auth is required for online evaluation; configure it on this server')
    call = evaluator_args([ROOT / s['answers']], output, offline=args.offline, auth=args.auth, calls=args.max_api_calls)
    report = run_evaluator(call)
    write(output / 'execution.json', {'schema': SCHEMA, 'shard': s['id'],
          'bundle_manifest_sha256': sha(ROOT / 'handoff/bundle_manifest.json'),
          'project_root': str(ROOT), 'output_root': str(output), 'command': sys.argv,
          'python': platform.python_version(), 'packages': read(ROOT / 'handoff/build_environment.json')['packages'],
          'report_status': report['status'], 'updated_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'shard': s['id'], 'status': report['status'], 'counts': report['status_counts']}))
    if args.offline:
        return 0 if set(report['status_counts']) <= {'NOT_REVIEWED', 'SCORED'} else 2
    return 0 if report['status'] in {'COMPLETE', 'COMPLETE_WITH_BOUNDS'} else 2


def pack(args):
    verify()
    s = assignment(args.shard)
    output = ROOT / 'outputs/distributed_eval' / s['id']
    from scripts.portable_fcntl import fcntl
    with (output / '.trusted_evaluation.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = read(output / 'execution.json')
        if execution['bundle_manifest_sha256'] != sha(ROOT / 'handoff/bundle_manifest.json'):
            raise ValueError('execution belongs to another bundle')
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            shutil.copytree(output, stage / 'evaluation', ignore=shutil.ignore_patterns('.trusted_evaluation.lock'))
            shutil.copyfile(ROOT / s['answers'], stage / 'answers.jsonl')
            write(stage / 'return_manifest.json', {'schema': SCHEMA, 'shard': s,
                'bundle_manifest_sha256': sha(ROOT / 'handoff/bundle_manifest.json'),
                'original_output_root': str(output), 'files': inventory(stage, [p for p in stage.rglob('*') if p.is_file()])})
            archive(stage, args.archive, [p for p in stage.rglob('*') if p.is_file()])
    print(json.dumps({'archive': str(args.archive.resolve()), 'sha256': sha(args.archive)}))


def extract(archive_path, destination):
    with tarfile.open(archive_path, 'r:gz') as tf:
        seen = set()
        for member in tf.getmembers():
            safe(destination, member.name)
            if not member.isfile() or member.name in seen:
                raise ValueError('return archive contains link, directory, special file or duplicate')
            seen.add(member.name)
        tf.extractall(destination, filter='data')
    doc = read(destination / 'return_manifest.json')
    check_files(destination, doc['files'])
    expected = {f['path'] for f in doc['files']} | {'return_manifest.json'}
    if expected != {str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}:
        raise ValueError('return has unlisted files')
    return doc


def merge(args):
    verify()
    from scripts import evaluate_model_answers as ev
    output = args.output.resolve()
    if output.exists():
        raise ValueError('merge output must be new; preserve previous imports/reports')
    output.mkdir(parents=True)
    expected = {s['id']: s for s in read(ROOT / 'handoff/assignments.json')['shards']}
    received, mappings, replay_dirs, remote_rows, remote_contracts = {}, {}, [], {}, []
    hashes = {f['path']: f['sha256'] for f in read(ROOT / 'handoff/bundle_manifest.json')['files']}
    reviewer = read(ROOT / 'handoff/judge.json')
    for index, path in enumerate(args.archives):
        dest = output / 'received' / str(index)
        doc = extract(path, dest)
        sid = doc['shard']['id']
        if sid in received:
            raise ValueError(f'duplicate shard: {sid}; do not choose a favorable review')
        if doc['bundle_manifest_sha256'] != sha(ROOT / 'handoff/bundle_manifest.json') or doc['shard'] != expected.get(sid):
            raise ValueError('foreign bundle/shard')
        s = expected[sid]
        if sha(dest / 'answers.jsonl') != s['answers_sha256']:
            raise ValueError('returned answers changed')
        directory = dest / 'evaluation'
        contract = read(directory / 'evaluation_contract.json')
        wanted = {k: reviewer[k] for k in ('model', 'effort', 'max_output_tokens')}
        if reviewer['structured_output']:
            wanted['structured_output'] = True
        if contract['protocol'] != ev.VERSION or contract['reviewer'] != wanted:
            raise ValueError('reviewer/protocol mismatch')
        source = contract['source_identity']
        for field, rel in [('manifest_sha256', MANIFEST), ('evidence_bank_sha256', BANK), ('reference_package_sha256', REFERENCE)]:
            if source[field] != hashes[rel]:
                raise ValueError(f'frozen source mismatch: {field}')
        if any(hashes.get(p) != h for p, h in source['scoring_dependencies_sha256'].items()):
            raise ValueError('scoring code differs')
        if len(source['inputs']) != 1 or source['inputs'][0]['sha256'] != s['answers_sha256']:
            raise ValueError('evaluation input export mismatch')
        execution = read(directory / 'execution.json')
        if (execution['bundle_manifest_sha256'] != doc['bundle_manifest_sha256']
                or execution['shard'] != sid or execution['output_root'] != doc['original_output_root']):
            raise ValueError('execution metadata does not bind this return')
        if execution['packages'] != read(ROOT / 'handoff/build_environment.json')['packages'] or execution['python'].split('.')[:2] != ['3', '12']:
            raise ValueError('remote environment mismatch')
        rr = [read(p) for p in (directory / 'cases').glob('*.json')]
        if len(rr) != len(s['slots']) or {key(r) for r in rr} != {tuple(k) for k in s['slots']}:
            raise ValueError('duplicate/missing/foreign returned slots')
        answer_map = {key(r): r for r in rows(dest / 'answers.jsonl')}
        for row in rr:
            if key(row) in remote_rows or row.get('answer_sha256') != answer_map[key(row)]['answer_sha256']:
                raise ValueError('duplicate slot or answer identity mismatch')
            remote_rows[key(row)] = row
        if doc['original_output_root'] in mappings:
            raise ValueError('ambiguous remote path mapping')
        mappings[doc['original_output_root']] = directory
        received[sid] = {'archive_sha256': sha(path), 'manifest': doc}
        replay_dirs.append(directory)
        remote_contracts.append(contract)
    if not args.allow_incomplete and set(received) != set(expected):
        raise ValueError('missing shards')
    old_resolve = ev.resolve
    def mapped(value, relative=None):
        raw = str(value)
        for prefix, target in mappings.items():
            if raw.startswith(prefix.rstrip('/') + '/'):
                return safe(target, raw[len(prefix.rstrip('/')) + 1:])
        # Imported review receipts must live inside a returned evidence tree.
        p = Path(raw).resolve()
        if any(p.is_relative_to(d.resolve()) for d in replay_dirs):
            return p
        raise ValueError(f'unbound external receipt path: {raw}')
    ev.resolve = mapped
    try:
        # Check every saved review's receipt and payload before replay. Metrics
        # still come exclusively from a fresh local evaluation below.
        for directory in replay_dirs:
            for path in (directory / 'judgments').glob('*.json'):
                stored = read(path)
                request = read(directory / 'requests' / path.name)
                identity = {'prompt': request['prompt'], 'schema': request['schema'], 'model': reviewer['model'],
                            'effort': reviewer['effort'], 'max_output_tokens': reviewer['max_output_tokens'], 'protocol': ev.VERSION}
                if reviewer['structured_output']:
                    identity['structured_output'] = True
                if path.stem != ev.digest(identity):
                    raise ValueError('review request identity mismatch')
                ev.validate_review_record(stored, request, path.stem, request['prompt'], request['schema'],
                    reviewer['model'], effort=reviewer['effort'], max_output_tokens=reviewer['max_output_tokens'],
                    structured_output=reviewer['structured_output'])
                receipt_path = mapped(stored['receipt'])
                if not receipt_path.is_relative_to(directory):
                    raise ValueError('review is bound to another shard receipt')
                receipt, payload = read(receipt_path), read(receipt_path.with_name('request.json'))
                response = read(receipt_path.with_name('response.json'))
                if not receipt_path.with_name('response.sse').is_file():
                    raise ValueError('missing original API stream')
                instructions = ('Return only JSON matching the supplied response schema. No tools. Use the required line citations; do not add commentary.'
                    if reviewer['structured_output'] else 'Return only JSON matching the supplied schema. No tools.\n' + json.dumps(request['schema'], ensure_ascii=False))
                final = '\n'.join(part['text'] for item in response.get('output', []) if item.get('type') == 'message'
                    for part in item.get('content', []) if part.get('type') == 'output_text' and isinstance(part.get('text'), str))
                if (response.get('status') != 'completed' or response.get('model') != reviewer['model']
                        or response.get('id') != receipt.get('response_id') or final != receipt['final_text']
                        or (response.get('usage') or {}) != (receipt.get('usage') or {})
                        or payload.get('model') != reviewer['model']
                        or payload.get('instructions') != instructions
                        or payload.get('reasoning', {}).get('effort') != reviewer['effort']):
                    raise ValueError('receipt differs from recorded API response/payload')
        call = evaluator_args([ROOT / 'handoff/answers/all.jsonl'], output / 'evaluation')
        call.replay_from = replay_dirs[0] if replay_dirs else None
        call.additional_replay_from = replay_dirs[1:]
        report = run_evaluator(call)
    finally:
        ev.resolve = old_resolve
    discrepancies = []
    for contract in remote_contracts:
        if contract['source_identity']['scoring_dependencies_sha256'] != report['source_identity']['scoring_dependencies_sha256']:
            raise ValueError('incomplete or different scoring dependency set')
    for row in report['rows']:
        old = remote_rows.get(key(row))
        if old and (row['metrics'] != old['metrics'] or row['status'] != old['status']):
            discrepancies.append(list(key(row)))
    # Account for each actual attempt once; cost_history contains cumulative snapshots.
    attempts, response_ids = [], {}
    for directory in replay_dirs:
        if any(not p.with_name('receipt.json').is_file() for p in (directory / 'api').glob('*/request.json')):
            raise ValueError('API attempt missing receipt/cost evidence')
        for p in sorted((directory / 'api').glob('*/receipt.json')):
            r = read(p)
            rid = r.get('response_id')
            if rid and rid in response_ids:
                if sha(p) != response_ids[rid]:
                    raise ValueError('conflicting duplicate API response ID')
                continue
            if rid:
                response_ids[rid] = sha(p)
            usage = r.get('usage') or {}
            known = all(type(usage.get(k)) is int for k in ('input_tokens', 'output_tokens'))
            attempts.append({'receipt': str(p), 'sha256': sha(p), 'response_id': rid,
                             'completed': bool(r.get('completed')), 'usage_known': known,
                             'input_tokens': usage.get('input_tokens'), 'output_tokens': usage.get('output_tokens'),
                             'seconds': r.get('finished_epoch', 0) - r.get('started_epoch', 0) if r.get('finished_epoch') and r.get('started_epoch') else None})
    costs = {'api_calls': len(attempts), 'input_tokens': sum(a['input_tokens'] or 0 for a in attempts),
             'output_tokens': sum(a['output_tokens'] or 0 for a in attempts),
             'usage_missing_calls': sum(not a['usage_known'] for a in attempts),
             'token_totals_complete': all(a['usage_known'] for a in attempts), 'attempts': attempts}
    write(output / 'evaluation/reports/distributed_api_cost.json', costs)
    audit = {'schema': SCHEMA, 'bundle_manifest_sha256': sha(ROOT / 'handoff/bundle_manifest.json'),
             'received_shards': sorted(received), 'missing_shards': sorted(set(expected) - set(received)),
             'requested_slots': report['requested_slots'], 'recomputed_status': report['status'],
             'metric_or_status_conflicts': discrepancies, 'path_mapping': {k: str(v) for k, v in mappings.items()},
             'accepted': not discrepancies and report['status'] in {'COMPLETE', 'COMPLETE_WITH_BOUNDS'} and set(received) == set(expected),
             'cost_report': 'evaluation/reports/distributed_api_cost.json'}
    audit['archives'] = {sid: data['archive_sha256'] for sid, data in received.items()}
    write(output / 'merge_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit['accepted'] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--reference-files', type=Path, required=True)
    b.add_argument('--reference-manifest', type=Path, required=True)
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--shards-per-model', type=int, choices=range(1, 97), default=1)
    sub.add_parser('verify')
    r = sub.add_parser('run')
    r.add_argument('--shard', required=True)
    r.add_argument('--offline', action='store_true')
    r.add_argument('--auth', type=Path)
    r.add_argument('--max-api-calls', type=int)
    r = sub.add_parser('pack-return')
    r.add_argument('--shard', required=True)
    r.add_argument('--archive', type=Path, required=True)
    r = sub.add_parser('merge')
    r.add_argument('--archives', type=Path, nargs='+', required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--allow-incomplete', action='store_true', help='diagnostic import; never marks incomplete data accepted')
    args = p.parse_args()
    os.chdir(ROOT)
    if args.command == 'build':
        build(args)
    elif args.command == 'verify':
        doc = verify()
        print(json.dumps({'status': 'PASS', 'files': len(doc['files']), 'slots': doc['slot_count']}))
    elif args.command == 'run':
        return run_shard(args)
    elif args.command == 'pack-return':
        pack(args)
    else:
        return merge(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

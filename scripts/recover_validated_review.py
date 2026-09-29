"""Recover an unchanged, provenance-bound reply after a validator compatibility fix."""
from pathlib import Path
import json

from flowintentbench.external_file_evaluator import digest
from scripts.collect_subagent_runs import file_hash
from scripts.run_codex_file_judgments import (visible_task_signature, validate_response,
                                            validate_schema, _output_schema, _exclusive_json, _verify_raw_data)


def recover(exchange, audit_path, *, apply=False):
    exchange, audit_path = Path(exchange).resolve(), Path(audit_path).resolve()
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    rid = audit['request_id']
    if len(rid) != 64 or any(c not in '0123456789abcdef' for c in rid):
        raise ValueError('Invalid request identity')
    request_path = exchange / 'requests' / (rid + '.json')
    request = json.loads(request_path.read_text(encoding='utf-8'))
    signature = visible_task_signature(request)
    _verify_raw_data(request, exchange)
    if audit.get('status') != 'ERROR' or audit.get('visible_task_signature') != signature:
        raise ValueError('Only a matching failed audit can be recovered')
    if audit.get('reviewer_model') != 'gpt-6-astra' or audit.get('review_policy', {}).get('effective_effort') != 'xhigh':
        raise ValueError('Reviewer must be GPT-6 Astra xhigh')
    if (exchange / 'responses' / (rid + '.json')).exists():
        raise ValueError('An existing verdict cannot be replaced')
    candidates = []
    for receipt in audit.get('transports', []):
        if (not receipt.get('completed') or receipt.get('returncode') != 0
                or receipt.get('timed_out') or receipt.get('error') or not receipt.get('thread_id')):
            continue
        envelope = json.loads(receipt['final_text'])
        validate_schema(envelope, _output_schema(request))
        if envelope.get('status') != 'RESOLVED' or envelope.get('evidence_files') != []:
            raise ValueError('Recovery requires a resolved structured-only verdict')
        validate_response(request, envelope['response'])
        candidates.append((envelope, receipt))
    if not candidates or len({digest(item[0]) for item in candidates}) != 1:
        raise ValueError('No single unchanged validated response')
    envelope, receipt = candidates[0]
    receipt_path = Path(receipt['events_path']).with_name('receipt.json').resolve()
    if not receipt_path.is_relative_to(audit_path.parent):
        raise ValueError('Receipt must belong to this audit attempt')
    saved_receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if saved_receipt != receipt:
        raise ValueError('Transport receipt differs from original audit')
    if receipt.get('reasoning_effort') != 'xhigh' or receipt.get('review_execution') != 'structured_only':
        raise ValueError('Original receipt must confirm structured-only xhigh review')
    recovery = dict(policy='unchanged-reply-after-validator-fix-v1', api_calls=0,
        source_audit_path=str(audit_path), source_audit_sha256=file_hash(audit_path),
        source_receipt_path=str(receipt_path), source_receipt_sha256=file_hash(receipt_path),
        source_request_sha256=file_hash(request_path), original_final_text_sha256=digest(receipt['final_text']),
        validator_sha256=file_hash(Path(__file__).with_name('run_expansion_file_evaluation.py')),
        recovery_source_sha256=file_hash(Path(__file__)), visible_task_signature=signature)
    result = dict(request_id=rid, reviewer_id=f"third-party-api:gpt-6-astra:{receipt['thread_id']}",
        status='RESOLVED', response=envelope['response'], reason=envelope['reason'],
        provenance=dict(audit_path=str(audit_path), evidence_files=[],
                        visible_task_signature=signature, review_policy=audit['review_policy'],
                        validator_recovery=recovery))
    if apply:
        _exclusive_json(exchange / 'validator_recovery' / (rid + '.json'), recovery)
        _exclusive_json(exchange / 'responses' / (rid + '.json'), result)
    return dict(request_id=rid, status='RECOVERED' if apply else 'VALIDATED', api_calls=0,
                source_audit_sha256=recovery['source_audit_sha256'])

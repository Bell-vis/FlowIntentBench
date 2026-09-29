"""Bounded new reviewer attempts after a terminal, unreturned API generation.

Never retry an ambiguous HTTP request in place. Never replace a published
scientific answer or sample alternatives to a PENDING/low score.
"""
from pathlib import Path
import time

from scripts.collect_subagent_runs import read, write, file_hash
from scripts.carry_forward_file_judgments import visible_task_signature

POLICY = dict(version='terminal-unreturned-review-replacement-v1', max_replacements=2,
              cooldown_seconds=120, selection='First valid published verdict; never resample science',
              cost='All original and replacement attempts retained in reviewer usage')
# A streamed HTTP 200 can still end before the first complete model turn.  The
# adapter records this as ModelAdapterError with ``ambiguous=true``; it is safe
# to use the same bounded, audited replacement path as other unreturned
# provider outcomes because no scientific verdict was produced.
ERRORS = {'response_outcome_unknown', 'http_500', 'http_502', 'http_503', 'http_504', 'ModelAdapterError', 'invalid_stream_response'}


def recover_unreturned_review(output, exchange, request_id, failure, *, now=None):
    """Return an audited release only for a finished claim with no final output."""
    output, exchange = Path(output).resolve(), Path(exchange).resolve()
    now = time.time() if now is None else now
    if len(request_id) != 64 or any(c not in '0123456789abcdef' for c in request_id):
        return None
    if (exchange/'responses'/f'{request_id}.json').exists():
        return None
    attempt = Path(failure.get('attempt_path', '')).resolve()
    if not attempt.is_relative_to(output/'judgment_work'):
        return None
    claim_path = output/'task_claims/judgment'/f'{request_id}.json'
    audit_path = attempt/'audit.json'
    request_path = exchange/'requests'/f'{request_id}.json'
    if not all(p.is_file() for p in (claim_path, audit_path, request_path)):
        return None
    claim, audit, request = read(claim_path), read(audit_path), read(request_path)
    if (claim.get('status') != 'FINISHED' or claim.get('identity') != request_id
            or audit.get('request_id') != request_id or audit.get('status') != 'ERROR'
            or audit.get('failure_kind') != 'api_transport'):
        return None
    receipt = audit.get('transport') or {}
    if (receipt.get('completed') or receipt.get('final_text') or not receipt.get('ambiguous')
            or receipt.get('error') not in ERRORS or not receipt.get('finished_epoch')):
        return None
    if claim.get('finished_epoch', 0) < receipt['finished_epoch']:
        return None
    signature = visible_task_signature(request)
    directory = output/'judgment_recovery'/signature
    audit_hash = file_hash(audit_path)
    event_path = directory/f'{audit_hash}.json'
    if event_path.exists():
        event = read(event_path)
        return event if event.get('source_audit_sha256') == audit_hash else None
    previous = list(directory.glob('*.json'))
    if len(previous) >= POLICY['max_replacements']:
        return None
    ordinal = len(previous) + 1
    retry_epoch = receipt['finished_epoch'] + POLICY['cooldown_seconds'] * ordinal
    if now < retry_epoch:
        return dict(status='WAITING_FOR_COOLDOWN', retry_epoch=retry_epoch)
    event = dict(policy=POLICY, request_id=request_id, visible_task_signature=signature,
                 replacement_index=ordinal, epoch=now, source_audit_path=str(audit_path),
                 source_audit_sha256=audit_hash, source_claim_sha256=file_hash(claim_path),
                 source_request_sha256=file_hash(request_path), original_failure=failure,
                 status='REPLACEMENT_AUTHORIZED', scientific_response=None)
    write(event_path, event)
    return event

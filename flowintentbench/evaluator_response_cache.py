"""Persist only validated evaluator responses within one observation.

A transport/schema failure never becomes a cached judgment. Exact input and
contract hashes prevent reuse after model, prompt, reference or protocol edits.
"""
from __future__ import annotations
import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


class ValidatedResponseCache:
    def __init__(self, path: Path, namespace: str):
        self.path, self.namespace = Path(path), namespace

    def resolve(self, operation, payload, parser, compute, repair_audit):
        request = canonical(payload)
        key = digest(canonical([self.namespace, operation, request]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=1800)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS responses (cache_key TEXT PRIMARY KEY, namespace TEXT NOT NULL, operation TEXT NOT NULL, request_json TEXT NOT NULL, response_json TEXT NOT NULL, response_sha256 TEXT NOT NULL, repair_audit_json TEXT NOT NULL, created_at REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0)')
            # Serializes duplicate requests for this observation. Different
            # cases/rounds have separate databases and remain independent.
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT namespace, operation, request_json, response_json, response_sha256, repair_audit_json FROM responses WHERE cache_key=?', (key,)).fetchone()
            if row is not None:
                namespace, cached_operation, cached_request, response, checksum, repairs = row
                if (namespace != self.namespace or cached_operation != operation
                        or cached_request != request or digest(response) != checksum):
                    raise ValueError('validated evaluator cache identity/checksum mismatch')
                result = parser(json.loads(response))
                repair_audit.extend(json.loads(repairs))
                db.execute('UPDATE responses SET hits=hits+1 WHERE cache_key=?', (key,))
                return result
            audit_start = len(repair_audit)
            def validate_and_store(value):
                result = parser(value)  # Never store unvalidated provider output.
                response = canonical(value)
                db.execute('INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)',
                           (key, self.namespace, operation, request, response, digest(response),
                            canonical(repair_audit[audit_start:]), time.time()))
                return result
            return compute(validate_and_store)

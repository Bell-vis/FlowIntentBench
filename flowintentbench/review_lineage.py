"""Read-only, necessary-condition audit of final wording review lineage.

A matching text is not a study-readiness verdict: semantic/context packet
binding, scientific qualification and participant validation remain separate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .experiment_scope import case_inventory
from .scientific_run_snapshots import canonical_json_sha256


def _objects(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def audit_question_review_lineage(
    repository_root: Path, manifest_path: Path, review_roots: list[Path]
) -> dict[str, Any]:
    """Find exact text matches to successful live final reviews, without mutation."""
    root = repository_root.resolve()
    manifest_path = (root / manifest_path).resolve()
    inventory = case_inventory(json.loads(manifest_path.read_text(encoding="utf-8")))
    paths: set[Path] = set()
    errors: list[dict[str, str]] = []
    for directory in review_roots:
        directory = (root / directory).resolve()
        if not directory.is_dir():
            errors.append({"path": str(directory), "error": "MISSING_REVIEW_DIRECTORY"})
        else:
            paths.update(directory.rglob("*.json"))
    reviews: dict[str, set[str]] = {}
    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append({"path": str(path), "error": type(exc).__name__})
            continue
        for record in _objects(payload):
            provenance = record.get("provenance")
            if not isinstance(provenance, dict):
                continue
            # A provenance fragment, injected fixture, failed call or shallow
            # PASS must never stand in for the complete live review envelope.
            if not (
                provenance.get("review_type") == "FINAL_FLOW_EXPERT_WORDING_REVIEW"
                and provenance.get("execution_mode") == "LIVE_MODEL_CALL"
                and provenance.get("live_model_calls") is True
                and record.get("invocation_status") == "SUCCESS"
                and provenance.get("invocation_status") == "SUCCESS"
                and record.get("final_flow_expert_review_status") == "PASS"
            ):
                continue
            text = record.get("reviewed_model_visible_text")
            digest = record.get("reviewed_model_visible_text_sha256")
            if not isinstance(text, str) or not text.strip():
                continue
            if digest != canonical_json_sha256({"model_visible_text": text}):
                continue
            if provenance.get("reviewed_model_visible_text_sha256") != digest:
                continue
            reviews.setdefault(digest, set()).add(str(path))
    cases = []
    for case_id, row in inventory.items():
        case_path = row.get("case_input_path") or row.get("case_input")
        if not isinstance(case_path, str) or not case_path:
            raise ValueError(f"{case_id}: missing case_input path")
        payload = json.loads((root / case_path).read_text(encoding="utf-8"))
        question = payload.get("scientific_question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{case_id}: missing scientific_question")
        digest = canonical_json_sha256({"model_visible_text": question})
        matching = sorted(reviews.get(digest, set()))
        cases.append({
            "case_id": case_id,
            "question_sha256": digest,
            "word_count": len(question.split()),
            "live_final_review_text_status": "MATCH_FOUND" if matching else "NOT_ESTABLISHED",
            "matching_review_files": matching,
        })
    return {
        "artifact_type": "READ_ONLY_FINAL_WORDING_LINEAGE_AUDIT",
        "manifest_path": str(manifest_path),
        "searched_roots": [str((root / path).resolve()) for path in review_roots],
        "search_complete": not errors,
        "read_errors": errors,
        "case_count": len(cases),
        "unique_live_reviewed_texts": len(reviews),
        "current_text_with_live_final_review_count": sum(bool(c["matching_review_files"]) for c in cases),
        "semantic_and_context_binding_status": "NOT_ASSESSED",
        "user_study_readiness": "NOT_ESTABLISHED_BY_THIS_AUDIT",
        "cases": cases,
    }


def live_review_call_matches(call: dict, packet: dict, instruction: str, profile_sha256: str) -> bool:
    """Reuse only an identical successful live invocation, including its policy."""
    import hashlib
    from .live_agents import _digest, build_live_role_system_instruction
    from .proxy_expert import build_firewalled_payload
    return bool(
        call.get('invocation_status') == 'SUCCESS'
        and call.get('execution_mode') == 'LIVE_MODEL_CALL'
        and call.get('live_model_calls') is True
        and call.get('agent_profile_sha256') == profile_sha256
        and call.get('visible_input_sha256') == _digest(build_firewalled_payload('scientific_reviewer', packet))
        and call.get('system_instruction_sha256') == hashlib.sha256(build_live_role_system_instruction('scientific_reviewer', instruction).encode()).hexdigest()
    )

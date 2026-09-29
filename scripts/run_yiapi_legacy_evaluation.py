#!/usr/bin/env python3
"""Evaluate the existing Fable/Sonnet answers through YiAPI GPT-6.

This script never runs a model-under-test. It only reviews completed
``RunRecord`` files, writes one durable result per answer, and can be resumed.
YiAPI JSON mode is used by default because its Responses endpoint does not
accept the full OpenAI strict JSON-schema subset used by the local scorer.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_yiapi_rebuild_pipeline import (  # noqa: E402
    HISTORICAL_ROOTS,
    REVIEWER_MODEL,
    _health,
    _read_auth,
    _score_collection,
    _state,
    _write_runtime_files,
)
from flowintentbench.expansion_evaluation import read_json  # noqa: E402


def _require_benchmark() -> None:
    from scripts.benchmark_runtime import require_benchmark_runtime
    require_benchmark_runtime()

def _archive_previous(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    archive = output_root.parent / "legacy_evaluation_archive" / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(output_root), str(archive))
    return archive


def main(argv: list[str] | None = None) -> int:
    _require_benchmark()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    parser.add_argument("--auth", type=Path, default=ROOT / "auth_yapi.json")
    parser.add_argument("--source-root", type=Path, default=HISTORICAL_ROOTS["fable_sonnet"])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", action="append", choices=("claude-fable-5-1", "claude-sonnet-5"))
    parser.add_argument("--case-id", action="append", help="evaluate only these case IDs; repeatable")
    parser.add_argument("--limit", type=int, help="evaluate at most this many cases per selected model")
    parser.add_argument("--reset", action="store_true", help="archive old YiAPI results before starting")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_root = (args.output_root or root / "legacy_evaluation").resolve()
    if args.reset:
        archived = _archive_previous(output_root)
        if archived:
            print(json.dumps({"archived_previous_results": str(archived)}, ensure_ascii=False))

    # JSON mode is intentional and is also recorded in the run identity.
    import os
    os.environ.setdefault("YIAPI_REVIEW_FORMAT", "json_object")
    key = _read_auth(args.auth.resolve())
    runtime = _write_runtime_files(root)
    health = _health(root, key, required=(REVIEWER_MODEL,))
    if health.get("status") != "HEALTHY":
        raise SystemExit(f"YiAPI preflight failed; inspect {root / 'health/api_health.json'}")
    rows = {row["case_id"]: row for row in read_json(args.manifest.resolve())["cases"]}
    selected = set(args.model or ("claude-fable-5-1", "claude-sonnet-5"))
    summaries = {}
    for model in ("claude-fable-5-1", "claude-sonnet-5"):
        if model not in selected:
            continue
        output = output_root / model
        case_ids = set(args.case_id) if args.case_id else None
        if args.limit is not None and case_ids is None:
            # Keep selection deterministic so a smoke test is reproducible.
            from flowintentbench.model_runner import RunRecord
            candidates = []
            for path in args.source_root.resolve().rglob("run_record.json"):
                try:
                    record = RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    continue
                if record.model_id == model and record.final_response and record.case_id in rows:
                    candidates.append(record.case_id)
            case_ids = set(sorted(set(candidates))[:args.limit])
        summaries[model] = _score_collection(
            args.source_root.resolve(), output, rows, key, workers=args.workers, model=model, case_ids=case_ids
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps({
            "schema_version": "yiapi-legacy-evaluation-v1",
            "reviewer_model": REVIEWER_MODEL,
            "reviewer_effort": "xhigh",
            "review_format": os.environ["YIAPI_REVIEW_FORMAT"],
            "models": summaries,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _state(root, status="LEGACY_EVALUATION_COMPLETE", legacy_evaluation=str(output_root), reviewer_model=REVIEWER_MODEL, review_format=os.environ["YIAPI_REVIEW_FORMAT"])
    print(json.dumps({"status": "OK", "output": str(output_root), "models": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

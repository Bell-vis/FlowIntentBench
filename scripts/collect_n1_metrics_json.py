#!/usr/bin/env python3
"""Package the already-rendered N=1 evaluation artifacts into one JSON file.

This command is a read-only reporting layer.  It never invokes an evaluator,
recomputes a scientific metric, or turns a pending/null value into zero.  The
scientific and efficiency values are copied from ``render_evaluation_report``
CSV artifacts and the existing evaluator ``summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc


_INTEGER_COLUMNS = {
    "trial",
    "eligible_case_count",
    "case_count",
    "applicable_n",
    "finalized_n",
    "null_n",
    "input_tokens",
    "output_tokens",
    "model_turn_count",
    "tool_call_count",
    "python_execution_count",
    "total_tokens",
}
_FLOAT_COLUMNS = {
    "o_score",
    "urs",
    "resolved_o_compliance",
    "finding_precision",
    "core_finding_recall",
    "finding_requirement_recall",
    "c_score",
    "branch_alignment",
    "branch_alignment_rate",
    "adequate_core_complete_rate",
    "wall_clock_time",
    "python_execution_total_time",
    "provider_reported_cost",
    "mean",
    "median",
}


def _typed_cell(column: str, value: str) -> Any:
    """Preserve CSV nulls as JSON null and restore numeric JSON types."""

    if value == "":
        return None
    if column in _INTEGER_COLUMNS:
        return int(value)
    if column in _FLOAT_COLUMNS:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite metric value in {column}")
        return number
    if column == "value":
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError("metric matrix value must be numeric or empty") from exc
        if not math.isfinite(number):
            raise ValueError("metric matrix value must be finite")
        return number
    if column == "adequate_core_complete":
        lowered = value.casefold()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [
                {key: _typed_cell(key, value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    except OSError as exc:
        raise ValueError(f"cannot read rendered CSV artifact: {path}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_snapshot(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return the persisted evaluator snapshot identity as report metadata.

    This is intentionally descriptive: it does not decide whether two
    snapshots are scientifically comparable and it never rewrites a record.
    A summary containing more than one manifest variant is reported as
    heterogeneous rather than silently collapsed into one identity.
    """

    variants = summary.get("evaluation_manifest_variants")
    variant_count = len(variants) if isinstance(variants, Mapping) else 1
    digest = summary.get("evaluation_manifest_digest")
    if not digest and isinstance(summary.get("evaluation_manifest"), Mapping):
        manifest = summary["evaluation_manifest"]
        digest = manifest.get("manifest_digest")
    return {
        "status": "PASS" if variant_count <= 1 else "HETEROGENEOUS",
        "evaluation_manifest_digest": digest,
        "variant_count": variant_count,
        "scientific_comparability": variant_count <= 1,
    }


def collect_n1_metrics_json(
    *,
    evaluation_root: str | Path,
    summary_path: str | Path,
    report_dir: str | Path,
    output_path: str | Path,
    model: str,
    collection_state_path: str | Path | None = None,
    case_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Copy existing evaluator/report artifacts into one inspectable JSON."""

    evaluation = Path(evaluation_root).resolve()
    summary_file = Path(summary_path).resolve()
    report = Path(report_dir).resolve()
    output = Path(output_path).resolve()
    summary = _read_json(summary_file)
    if not isinstance(summary, dict):
        raise ValueError("evaluation summary must be a JSON object")
    required = {
        "trial_metrics": report / "trial_metrics.csv",
        "case_metrics": report / "case_metrics.csv",
        "condition_metrics": report / "condition_metrics.csv",
        "efficiency_metrics": report / "efficiency_metrics.csv",
        "metric_denominators": report / "metric_denominators.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError(
            "rendered report artifacts are missing; run render_evaluation_report first: "
            + ", ".join(missing)
        )

    tables = {name: _read_csv(path) for name, path in required.items()}
    # ``metric_matrix.csv`` and ``metric_coverage.csv`` are report-only
    # projections introduced after the original v2 bundle.  Keep them
    # optional for legacy fixtures while copying them whenever present.
    metric_matrix_path = report / "metric_matrix.csv"
    metric_coverage_path = report / "metric_coverage.csv"
    metric_matrix_rows = _read_csv(metric_matrix_path) if metric_matrix_path.is_file() else []
    metric_coverage_rows = _read_csv(metric_coverage_path) if metric_coverage_path.is_file() else []
    reliability_path = report / "infrastructure_reliability.csv"
    reliability_rows = _read_csv(reliability_path) if reliability_path.is_file() else []
    collection_state_file = (
        None if collection_state_path is None else Path(collection_state_path).resolve()
    )
    case_manifest_file = (
        None if case_manifest_path is None else Path(case_manifest_path).resolve()
    )
    # Older collection roots persist ``collected_case_manifest.json`` while
    # current roots use ``case_manifest.json``.  Treat either as the same
    # collection authority when the caller supplied the canonical name.
    if case_manifest_file is not None and not case_manifest_file.is_file():
        alternatives = (
            case_manifest_file.with_name("collected_case_manifest.json"),
            case_manifest_file.with_name("case_manifest.json"),
        )
        case_manifest_file = next(
            (candidate for candidate in alternatives if candidate.is_file()),
            case_manifest_file,
        )
    collection_state = (
        None if collection_state_file is None else _read_json(collection_state_file)
    )
    case_manifest = None if case_manifest_file is None else _read_json(case_manifest_file)
    if collection_state is not None and not isinstance(collection_state, dict):
        raise ValueError("collection state must be a JSON object")
    if case_manifest is not None and not isinstance(case_manifest, dict):
        raise ValueError("case manifest must be a JSON object")
    status_counts: dict[str, int] = {}
    for row in tables["trial_metrics"]:
        status = str(row.get("status") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    requested_case_count = (
        int(case_manifest.get("case_count", 0))
        if case_manifest is not None
        else len(tables["trial_metrics"])
    )
    collection_observations = (
        list(collection_state.get("observations", ()))
        if collection_state is not None
        else []
    )
    observed_case_ids = {
        str(row.get("case_id"))
        for row in collection_observations
        if isinstance(row, dict) and row.get("case_id")
    }
    collection_case_status: list[dict[str, Any]] = []
    if case_manifest is not None:
        observations_by_case = {
            str(row.get("case_id")): row
            for row in collection_observations
            if isinstance(row, dict) and row.get("case_id")
        }
        for case in case_manifest.get("cases", ()):
            if not isinstance(case, dict):
                continue
            case_id = str(case.get("case_id") or "")
            observation = observations_by_case.get(case_id)
            collection_case_status.append(
                {
                    "case_id": case_id,
                    "dataset_id": case.get("dataset_id"),
                    "condition": case.get("condition"),
                    "trial_index": case.get("trial_index"),
                    "collection_status": (
                        observation.get("status")
                        if observation is not None
                        else "NOT_COLLECTED"
                    ),
                    "run_id": None if observation is None else observation.get("run_id"),
                    "efficiency": None
                    if observation is None
                    else observation.get("efficiency"),
                }
            )
    payload: dict[str, Any] = {
        "record_type": "FullDatasetN1MetricBundle",
        "schema_version": "full-dataset-n1-metric-bundle-v2",
        "model": model,
        "mode": summary.get("mode", "PILOT"),
        "status": summary.get("status"),
        "evaluation_root": str(evaluation),
        "summary_path": str(summary_file),
        "summary": summary,
        "evaluation_snapshot": _evaluation_snapshot(summary),
        "collection_state": collection_state,
        "case_manifest": case_manifest,
        "collection_case_status": collection_case_status,
        "trial_metrics": tables["trial_metrics"],
        "case_metrics": tables["case_metrics"],
        "condition_metrics": tables["condition_metrics"],
        "efficiency_metrics": tables["efficiency_metrics"],
        "metric_denominators": tables["metric_denominators"],
        "metric_matrix": metric_matrix_rows,
        "metric_coverage": metric_coverage_rows,
        "infrastructure_reliability": reliability_rows,
        "quality_efficiency": (
            _read_json(report / "quality_efficiency.json")
            if (report / "quality_efficiency.json").is_file() else None
        ),
        "coverage": {
            "requested_case_count": requested_case_count,
            "collected_observation_count": len(collection_observations),
            "not_collected_count": max(0, requested_case_count - len(observed_case_ids)),
            "evaluated_trial_count": len(tables["trial_metrics"]),
            "terminal_record_count": sum(
                value for key, value in status_counts.items() if key != "PENDING"
            ),
            "pending_count": status_counts.get("PENDING", 0),
            "infrastructure_invalid_count": status_counts.get(
                "INFRASTRUCTURE_INVALID", 0
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "scientific_metrics_are_copied": True,
            "pending_values_preserved_as_null": True,
        },
        "artifact_sha256": {
            "summary": _sha256(summary_file),
            **{name: _sha256(path) for name, path in required.items()},
            **(
                {"metric_matrix": _sha256(metric_matrix_path)}
                if metric_matrix_path.is_file()
                else {}
            ),
            **(
                {"metric_coverage": _sha256(metric_coverage_path)}
                if metric_coverage_path.is_file()
                else {}
            ),
            **(
                {"collection_state": _sha256(collection_state_file)}
                if collection_state_file is not None
                else {}
            ),
            **(
                {"case_manifest": _sha256(case_manifest_file)}
                if case_manifest_file is not None
                else {}
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--collection-state", type=Path)
    parser.add_argument("--case-manifest", type=Path)
    args = parser.parse_args()
    result = collect_n1_metrics_json(
        evaluation_root=args.evaluation_root,
        summary_path=args.summary,
        report_dir=args.report_dir,
        output_path=args.output,
        model=args.model,
        collection_state_path=args.collection_state,
        case_manifest_path=args.case_manifest,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "requested_case_count": result["coverage"]["requested_case_count"],
                "collected_observation_count": result["coverage"]["collected_observation_count"],
                "evaluated_trial_count": result["coverage"]["evaluated_trial_count"],
                "status_counts": result["coverage"]["status_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

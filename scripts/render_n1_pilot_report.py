"""Render descriptive N=1 pilot diagnostics from saved evaluations.

The script is intentionally a projection layer.  It reads finalized
``CaseEvaluationRecord``/pending artifacts and never invokes ``CaseEvaluator``
or recomputes a scientific judgment.  It does not create formal case or
condition aggregates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_evaluation_report import (  # noqa: E402
    TRIAL_COLUMNS,
    _artifact_trial_rows,
    _markdown_table,
    _write_csv,
)
from flowintentbench.evaluation_records import iter_current_evaluation_records


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"evaluation artifact must be an object: {path}")
    return value


def render_n1_pilot_report(
    *, evaluation_root: str | Path, output_dir: str | Path, model: str
) -> dict[str, Path]:
    root = Path(evaluation_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = _artifact_trial_rows(root, model)
    if any(row.get("trial") != 1 for row in rows):
        raise ValueError("N=1 pilot diagnostics require trial_index=1 only")
    trial_path = output / "pilot_trial_metrics.csv"
    _write_csv(trial_path, TRIAL_COLUMNS, rows)

    current_paths = list(iter_current_evaluation_records(root))
    final_paths = [path for path in current_paths if path.name == "case_evaluation_record.json"]
    pending_paths = [path for path in current_paths if path.name == "pending_case_evaluation.json"]
    status_counts = Counter(str(row.get("status")) for row in rows)
    novel_o = 0
    novel_finding = 0
    records: list[dict[str, Any]] = []
    finals: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for path in final_paths:
        payload = _read(path)
        run_id = str(payload.get("run_id") or path.parent)
        finals[run_id] = (path, payload)
    pending_candidates: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for path in pending_paths:
        payload = _read(path)
        run_id = str(payload.get("run_id") or path.parent)
        if run_id not in finals:
            pending_candidates[run_id] = (path, payload)

    # Mirror the report renderer's run-id join: one finalized record is
    # authoritative, and duplicate pending attempts for that run are only
    # historical diagnostics rather than additional trials.
    for path, payload in finals.values():
        if payload.get("novel_operationalization_adjudication") is not None:
            novel_o += 1
        adjudications = payload.get("novel_finding_adjudications")
        if isinstance(adjudications, Mapping) and adjudications:
            novel_finding += 1
        result = payload.get("result")
        result_status = (
            result.get("run_status")
            if isinstance(result, Mapping)
            else payload.get("run_status")
        )
        records.append({
            "case_id": result.get("case_id") if isinstance(result, Mapping) else payload.get("case_id"),
            "trial_index": payload.get("trial_index"),
            # Preserve the evaluator status verbatim.  In particular,
            # INFRASTRUCTURE_INVALID is a persisted terminal status, not a
            # completed scientific evaluation.
            "status": str(result_status or ("PENDING" if not isinstance(result, Mapping) else "UNKNOWN")),
            "source": str(path),
        })
    for path, payload in pending_candidates.values():
        records.append({
            "case_id": payload.get("case_id"),
            "trial_index": payload.get("trial_index"),
            "status": "PENDING",
            "source": str(path),
        })
    diagnostics = {
        "label": "N=1 PILOT DESCRIPTIVE ONLY / NOT FORMAL AGGREGATION",
        "model": model,
        "trial_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "pending_count": status_counts.get("PENDING", 0),
        "novel_operationalization_adjudication_incidence": novel_o,
        "novel_finding_adjudication_incidence": novel_finding,
        "scientific_judgments_recomputed": False,
        "records": sorted(records, key=lambda item: (str(item.get("case_id")), item.get("trial_index") or 0)),
    }
    diagnostics_path = output / "pilot_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# FlowIntentBench N=1 Pilot Diagnostics",
        "",
        "**N=1 PILOT DESCRIPTIVE ONLY — NOT FORMAL AGGREGATION**",
        "",
        f"- Model: `{model}`",
        f"- Trial records: `{len(rows)}`",
        f"- Status counts: `{dict(sorted(status_counts.items()))}`",
        f"- Pending incidence: `{status_counts.get('PENDING', 0)}`",
        f"- Novel-O adjudication incidence: `{novel_o}`",
        f"- Novel-Finding adjudication incidence: `{novel_finding}`",
        "- Scientific judgments recomputed by this renderer: `false`",
        "",
        "## Trial Metrics",
        "",
        _markdown_table(TRIAL_COLUMNS, rows),
        "",
    ]
    markdown_path = output / "pilot_diagnostics.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"pilot_trial_metrics": trial_path, "pilot_diagnostics": diagnostics_path, "pilot_diagnostics_markdown": markdown_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    render_n1_pilot_report(evaluation_root=args.evaluation_root, output_dir=args.output_dir, model=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

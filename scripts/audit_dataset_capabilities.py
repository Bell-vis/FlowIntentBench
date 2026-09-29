#!/usr/bin/env python3
"""Write the read-only seven-dataset capability audit for case reselection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.dataset_capability import audit_dataset_capabilities  # noqa: E402


def _markdown(report: dict) -> str:
    lines = [
        "# Dataset Capability Audit",
        "",
        "This read-only audit identifies inputs sufficient for candidate case authoring. It does not authorize formal release.",
        "",
        f"- Datasets checked: **{report['dataset_count']}**",
        f"- Data capability status: **{'PASS' if report['all_data_capability_pass'] else 'FAIL'}**",
        f"- Official release-ready cases: **{report['official_release_ready_count']}**",
        "",
        "| Dataset | Reader | Fields | Sources | Reference analyses | Capability | Candidate question |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in report["datasets"]:
        fields = ", ".join(row["vector_variables"] + row["scalar_variables"])
        question = row.get("recommended_question", "").replace("|", "\\|")
        lines.append(
            f"| `{row['dataset_id']}` | `{row['canonical_reader']}` | `{fields}` | "
            f"{row['source_url_count'] + row['construction_source_count']} | "
            f"{row['reference_analysis_count']} | **{row['data_capability_status']}** | {question} |"
        )
    lines.extend(
        [
            "",
            "Capability PASS means that a new case may be authored and tested. Evidence closure, Ground Truth semantic binding, SRAC when applicable, and accountable curator confirmation remain separate release gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/current/dataset_capability_audit")
    args = parser.parse_args()
    report = audit_dataset_capabilities(ROOT)
    output = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    (output / "capability_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "capability_audit.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "dataset_count": report["dataset_count"],
        "all_data_capability_pass": report["all_data_capability_pass"],
        "official_release_ready_count": report["official_release_ready_count"],
        "output_root": str(output),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["all_data_capability_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

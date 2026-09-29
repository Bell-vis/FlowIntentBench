"""Run the fail-closed FlowIntentBench next-stage readiness gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.next_stage import run_next_stage_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, default=ROOT / "artifacts/reference/full_dataset_n1_controlled_pilot")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/experiments/next_stage_gate")
    parser.add_argument("--judge-a-output", type=Path)
    parser.add_argument("--judge-b-output", type=Path)
    parser.add_argument("--human-reference", type=Path)
    parser.add_argument("--provider-compatibility-report", type=Path)
    parser.add_argument("--clean-pilot-root", type=Path)
    parser.add_argument("--primary-evaluator-manifest", type=Path)
    parser.add_argument("--o1-curator-audit", type=Path)
    parser.add_argument("--data-contract-curator-audit", type=Path)
    parser.add_argument("--archive-self-test", type=Path)
    parser.add_argument("--whole-system-efficiency", type=Path)
    parser.add_argument("--unchanged-task-efficiency", type=Path)
    parser.add_argument("--branch-evaluation-root", type=Path)
    args = parser.parse_args()
    result = run_next_stage_gate(
        repository_root=ROOT,
        collection_root=args.collection_root,
        output_root=args.output_root,
        judge_a_output=args.judge_a_output,
        judge_b_output=args.judge_b_output,
        human_reference=args.human_reference,
        provider_compatibility_report=args.provider_compatibility_report,
        clean_pilot_root=args.clean_pilot_root,
        primary_evaluator_manifest=args.primary_evaluator_manifest,
        o1_curator_audit=args.o1_curator_audit,
        data_contract_curator_audit=args.data_contract_curator_audit,
        archive_self_test=args.archive_self_test,
        whole_system_efficiency=args.whole_system_efficiency,
        unchanged_task_efficiency=args.unchanged_task_efficiency,
        branch_evaluation_root=args.branch_evaluation_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

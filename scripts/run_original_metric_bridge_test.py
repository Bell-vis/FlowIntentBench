"""Run the frozen original scorer on the one-answer bridge packet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.core_case_scoring import CoreCaseScorer, SCORER_IMPLEMENTATION_VERSION
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.ground_truth import GroundTruth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/original_metric_bridge_test")
    args = parser.parse_args()
    packet = json.loads((args.directory / "packet.json").read_text(encoding="utf-8"))
    judgment = json.loads((args.directory / "judgment.json").read_text(encoding="utf-8"))
    answer = packet["answer"]
    if hashlib.sha256(answer.encode()).hexdigest() != packet["answer_sha256"]:
        raise ValueError("answer identity changed")
    gt = GroundTruth.model_validate(packet["gt"])
    result = CoreCaseScorer().score(packet["case_payload"], answer, gt, judgment=judgment)
    output = {"protocol": "original-metric-bridge-test-v1", "scorer_version": SCORER_IMPLEMENTATION_VERSION,
              "run_id": packet["run_id"], "case_id": packet["case_id"], "prior_judgment_sha256": packet["prior_judgment_sha256"],
              "status": result["status"], "audit_status": result["audit_status"], "metrics": result["metrics"],
              "branch_metrics": result["branch_metrics"], "coverage": result["coverage"],
              "judgment_sha256": result["judgment_sha256"], "answer_sha256": result["answer_sha256"]}
    write_json(args.directory / "result.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

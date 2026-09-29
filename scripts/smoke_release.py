#!/usr/bin/env python3
"""Exercise frozen-reference preparation and answer ingestion with zero API calls."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="flowintentbench-smoke-") as directory:
        work = Path(directory)
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_evaluation_references.py"),
                        "--output", str(work / "references.json")], cwd=work, check=True)
        # A deliberately incomplete synthetic input, never a benchmark result.
        answers = work / "input.jsonl"
        answers.write_text(json.dumps({"model_id": "smoke-model", "case_id": "blunt_fin_o1_f1",
                                      "trial": 1, "answer": "No scientific analysis was performed in this packaging smoke test."}) + "\n")
        result = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_model_answers.py"),
            "--answers", str(answers), "--output", str(work / "evaluation"),
            "--offline", "--max-api-calls", "0"], cwd=ROOT)
        if result.returncode != 2:
            raise RuntimeError(f"Expected incomplete offline review (exit 2), got {result.returncode}")
        report = json.loads((work / "evaluation/reports/experiment_report.json").read_text())
        print(json.dumps({"smoke": "PASS", "evaluation_status": report.get("status"),
                          "expected_evaluator_exit": 2, "network_calls": 0,
                          "note": "Ingestion only; no scientific judgment or model score was generated."}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

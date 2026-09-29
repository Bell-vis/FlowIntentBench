#!/usr/bin/env python3
"""Snapshot and audit frozen references without solving tasks or calling an API."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.reference_packages import (VERSION, package, load_supplements, apply_supplement,
    materialized_support_record, reporting_precision_rules)
from flowintentbench.finding_requirements import FindingRequirementContract
from flowintentbench.evaluator import finding_verification_mode


def prepare(manifest, output, reference_package=None, support_output=None, reporting_precision=False):
    rows = json.loads(Path(manifest).read_text(encoding="utf-8"))["cases"]
    supplements = load_supplements(reference_package, ROOT) if reference_package else {}
    if set(supplements) - {r["case_id"] for r in rows}:
        raise ValueError("reference supplement contains a case outside this manifest")
    cases, errors, support_records = [], [], {}
    for row in rows:
        try:
            ci, meta, gt, material = load_development_case(ROOT, row)
            if support_output:
                record = materialized_support_record(ci, meta, gt, material)
                if reporting_precision:
                    record["reporting_policies"] = reporting_precision_rules(gt, record["supporting_findings"])
                    record["rationale"] += " Reporting uses a uniform three-significant-digit minimum and literal rounding compatibility; counts remain exact."
                support_records[row["case_id"]] = record
            gt, material = apply_supplement(ci, meta, gt, material, supplements.get(row["case_id"]))
            snapshot = package(ci, meta, gt, material)
            refs = {(b.operationalization_id, f.finding_id): f for b in gt.findings_by_operationalization for f in b.findings}
            missing = []
            for b in snapshot["branches"]:
                for ref in b["findings"]:
                    if not ref["host_policy"]:
                        missing.append([b["branch_id"], ref["finding_id"]])
                    else:
                        finding_verification_mode(refs[(b["branch_id"], ref["finding_id"])], ref["host_policy"])
            contract = FindingRequirementContract.from_mapping(material.get("finding_requirement_contract"))
            if row["condition"].endswith("F2") and (contract is None or contract.validate()):
                raise ValueError("F2 adequate-role contract is invalid")
            cases.append({"case_id": row["case_id"], "condition": row["condition"], **snapshot,
                "audit": {"missing_host_policy": missing,
                    "open_method_coverage": "NON_EXHAUSTIVE" if any(d["kind"] == "OPEN" for d in snapshot["dimensions"]) else "FIXED",
                    "reporting_precision_policy_count": sum(bool(f["reporting_policy"]) for b in snapshot["branches"] for f in b["findings"])}})
        except Exception as exc:
            errors.append({"case_id": row["case_id"], "error": f"{type(exc).__name__}: {exc}"})
    report = {"protocol": VERSION, "evaluation_boundary": "FROZEN_JSON_ONLY; NO_SOLVING_OR_API",
        "case_count": len(rows), "prepared_cases": len(cases), "errors": errors,
        "by_condition": dict(Counter(c["condition"] for c in cases)),
        "missing_policy_count": sum(len(c["audit"]["missing_host_policy"]) for c in cases),
        "coverage_note": "Structural preparation does not certify all possible alternative methods or extra claims. Scientific reference review is separate.",
        "cases": cases}
    write_json(Path(output), report)
    if support_output:
        from hashlib import sha256
        record_path = Path(support_output).with_name(Path(support_output).stem + "_records.json")
        if not record_path.resolve().is_relative_to(ROOT):
            raise ValueError("support records must be inside the repository")
        write_json(record_path, support_records)
        checksum = sha256(record_path.read_bytes()).hexdigest()
        write_json(Path(support_output), {"protocol": VERSION, "entries": [
            {"case_id": cid, "task_sha256": r["task_sha256"], "source": {
                "path": str(record_path.resolve().relative_to(ROOT)), "sha256": checksum,
                "pointer": "/" + cid.replace("~", "~0").replace("/", "~1")}}
            for cid, r in support_records.items()]})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    p.add_argument("--reference-package", type=Path)
    p.add_argument("--support-output", type=Path, help="Export typed, precomputed supporting references; never solve")
    p.add_argument("--reporting-precision", action="store_true", help="Author uniform three-significant-digit rounding policies")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation_v4_validation/reference_inventory.json")
    args = p.parse_args()
    if args.reporting_precision and not args.support_output:
        p.error("--reporting-precision requires --support-output")
    result = prepare(args.manifest, args.output, args.reference_package, args.support_output, args.reporting_precision)
    print(json.dumps({k: result[k] for k in ("case_count", "prepared_cases", "errors", "by_condition", "missing_policy_count")}, ensure_ascii=False))
    return 2 if result["errors"] or result["missing_policy_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

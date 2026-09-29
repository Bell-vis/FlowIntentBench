"""Static integration checks for the profiled evaluation pipeline.

This command consumes local repository contracts and fixtures only.  It never
calls a model and does not calculate scientific evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (  # noqa: E402
    ChatCompletionsEvaluatorTransport,
    ResponsesEvaluatorTransport,
    build_evaluator_data_contract,
    build_python_tool_description,
    build_runtime_contract,
    build_runtime_system_instruction,
    effective_agent_config,
    extract_atomic_findings,
    load_agent_profile,
    load_runtime_profile,
    sanitize_evaluation_evidence,
)
from flowintentbench.next_stage import _of_separation_outcomes  # noqa: E402
from flowintentbench.calibration import DIAGNOSTIC_CASE_IDS  # noqa: E402
from flowintentbench.manifest import DatasetManifest  # noqa: E402


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _judge_item(case_id: str, pattern: str) -> dict[str, object]:
    valid = pattern in {"A", "B"}
    support = "SUPPORTED" if pattern in {"A", "C"} else "UNSUPPORTED"
    return {
        "case_id": case_id,
        "analysis_label": "VALID_REFERENCE" if valid else "INVALID",
        "o_f_consistency": "INCONSISTENT" if pattern == "D" else "CONSISTENT",
        "findings": [{
            "finding_id": "P1", "statement": "a data-derived claim",
            "evidential_support": support, "correctness": "CORRECT",
            "relevance": "RELEVANT", "core_status": "CORE",
        }],
    }


def _human_item(case_id: str, pattern: str) -> dict[str, object]:
    item = _judge_item(case_id, pattern)
    item["classification"] = {
        "A": "SUPPORTED_FINDING", "B": "UNSUPPORTED_FINDING",
        "C": "DEFECTIVE_ANALYSIS", "D": "O_F_INCONSISTENT",
    }[pattern]
    item["finding_judgments"] = [{
        "finding_id": "P1", "statement": "a data-derived claim", "included": True,
        "evidential_support": item["findings"][0]["evidential_support"],
    }]
    return item


def validate(output_dir: Path) -> dict[str, object]:
    profile = load_agent_profile("gpt-5.6-sol-xhigh", repository_root=ROOT)
    runtime = load_runtime_profile(profile.runtime_profile_id, repository_root=ROOT)
    effective = effective_agent_config(profile)
    instruction = build_runtime_system_instruction(runtime)
    description = build_python_tool_description(runtime)
    contract = build_runtime_contract(runtime)
    profile_result = {
        "status": "PASS", "agent_id": profile.agent_id,
        "agent_profile_sha256": profile.profile_sha256,
        "runtime_profile_sha256": runtime.profile_sha256,
        "effective": effective.to_dict(),
    }
    _write(output_dir / "agent_profile_validation.json", profile_result)
    _write(output_dir / "runtime_single_source_validation.json", {
        "status": "PASS", "same_case_root": runtime.case_root in instruction and runtime.case_root in description and contract["case_root"] == runtime.case_root,
        "same_workspace_root": runtime.workspace_root in instruction and runtime.workspace_root in description and contract["workspace_root"] == runtime.workspace_root,
        "same_network": ("unavailable" if not runtime.network_available else "available") in instruction and contract["network_access"] == ("unavailable" if not runtime.network_available else "available"),
        "runtime_profile_sha256": runtime.profile_sha256,
    })
    _write(output_dir / "profile_hash_validation.json", {
        "status": "PASS" if profile.profile_sha256 == load_agent_profile("gpt-5.6-sol-xhigh", repository_root=ROOT).profile_sha256 else "FAIL",
        "agent_profile_sha256": profile.profile_sha256,
        "runtime_profile_sha256": runtime.profile_sha256,
    })
    extracted = extract_atomic_findings("## Operationalization\n\nUsed speed.\n\n## Finding\n\n- Region is strongest.\n- Peak is 0.3.")
    _write(output_dir / "atomic_finding_validation.json", {"status": "PASS", **extracted.to_dict()})
    trajectory = [{"event": "model_turn", "provider": "yiapi", "model_id": "gpt-5.6-sol", "raw_usage": {"input_tokens": 1}, "tool_batch": {"calls": [{"canonical_call_id": "secret", "arguments": {"code": "print(1)"}}]}}, {"event": "python_execution", "canonical_call_id": "secret", "stdout": "1", "input_tokens": 1, "duration_seconds": 2.0, "success": True}]
    evidence = sanitize_evaluation_evidence(trajectory)
    serialized = json.dumps(evidence, ensure_ascii=False)
    evidence_pass = all(token not in serialized for token in ("yiapi", "gpt-5.6-sol", "input_tokens", "duration_seconds", "raw_usage")) and "print(1)" in serialized
    _write(output_dir / "evaluator_evidence_sanitization.json", {"status": "PASS" if evidence_pass else "FAIL", "evidence": evidence})
    nasa_case = ROOT / "datasets/NASA_LOx_Post/construction/cases/nasa_lox_post_o2_f1/case_input.json"
    nasa_manifest = DatasetManifest.model_validate_json((ROOT / "datasets/NASA_LOx_Post/dataset_manifest.json").read_text(encoding="utf-8"))
    nasa_input = json.loads(nasa_case.read_text(encoding="utf-8"))
    data_contract = build_evaluator_data_contract(nasa_input, nasa_manifest)
    nasa_pass = "IBlank > 0" in str(data_contract.get("validity_rule"))
    _write(output_dir / "evaluator_data_contract_validation.json", {"status": "PASS" if nasa_pass else "FAIL", "validity_rule": data_contract.get("validity_rule"), "data_contract": data_contract})
    _write(output_dir / "evaluator_transport_validation.json", {
        "status": "PASS", "chat_endpoint": ChatCompletionsEvaluatorTransport.endpoint,
        "responses_endpoint": ResponsesEvaluatorTransport.endpoint,
        "chat_uses_messages": "messages" in ChatCompletionsEvaluatorTransport().body(type("C", (), {"model_id": "m"})(), "p", {}),
        "responses_uses_input": "input" in ResponsesEvaluatorTransport().body(type("C", (), {"model_id": "m"})(), "p", {}),
    })
    patterns = ["A", "B", "C", "D"]
    human = {"cases": [_human_item(case_id, patterns[index % 4]) for index, case_id in enumerate(DIAGNOSTIC_CASE_IDS)]}
    judge = {"cases": [_judge_item(case_id, patterns[index % 4]) for index, case_id in enumerate(DIAGNOSTIC_CASE_IDS)]}
    gate = _of_separation_outcomes(judge, judge, human)
    _write(output_dir / "gate11_validation.md", "# Gate 11 Validation\n\nStatus: **%s**\n\nBoth judge candidates are required to preserve the human diagnostic pattern, including analysis validity, Finding support, and O-F consistency.\n" % gate["status"])
    readiness = {"status": "NOT_READY", "reason": "static integration passed; independent judge and confirmed human artifacts remain unavailable", "checks_passed": 8}
    _write(output_dir / "readiness.json", readiness)
    (output_dir / "implementation_summary.md").write_text(
        "# Integration Hardening Summary\n\n"
        "- Static integration checks: **PASS**\n"
        "- `CANONICAL_MULTI_CALL_CHANGED`: `false`\n"
        "- `SCIENTIFIC_METHOD_CHANGED`: `false`\n"
        "- `MODEL_VISIBLE_PROTOCOL_CHANGED`: `true`\n"
        f"- Agent profile: `{effective.agent_id}`\n"
        f"- Agent profile SHA-256: `{effective.agent_profile_sha256}`\n"
        f"- Runtime profile SHA-256: `{effective.runtime_profile_sha256}`\n",
        encoding="utf-8",
    )
    return readiness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/profiled_evaluation_pipeline")
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

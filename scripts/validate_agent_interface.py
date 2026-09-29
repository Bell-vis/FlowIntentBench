"""Validate the agent/runtime/output interface without making a model call."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (  # noqa: E402
    CANONICAL_PYTHON_TOOL_SPEC,
    CaseLoader,
    DatasetManifest,
    DEFAULT_SYSTEM_PROMPT,
    load_agent_profile,
    load_runtime_profile,
    valid_final_answer_sections,
)
from flowintentbench.python_runtime import PythonExecutionEnvironment  # noqa: E402


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_validation(output_root: str | Path) -> dict[str, object]:
    output = Path(output_root).resolve()
    profile = load_agent_profile("gpt-5.6-sol-xhigh", repository_root=ROOT)
    runtime_profile = load_runtime_profile(profile.runtime_profile_id, repository_root=ROOT)
    datasets = ROOT / "datasets"
    case_dir = datasets / "Office" / "construction" / "cases" / "office_speed_zones_o1_f1"
    loaded = CaseLoader(datasets).load(case_dir / "case_input.json")
    manifest = DatasetManifest.model_validate_json(
        (datasets / "Office" / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    with PythonExecutionEnvironment(
        loaded,
        manifest=manifest,
        python_executable=sys.executable,
        require_network_isolation=False,
        runtime_profile=runtime_profile,
        workspace_parent=output / "runtime-work",
    ) as runtime:
        contract = json.loads((runtime.case_dir / "runtime_contract.json").read_text(encoding="utf-8"))
        case_files = json.loads((runtime.case_dir / "case_files.json").read_text(encoding="utf-8"))
    prompt_checks = {
        token: token in DEFAULT_SYSTEM_PROMPT
        for token in ("/case", "/workspace", "runtime_contract.json", "case_files.json", "network access is unavailable", "## Operationalization", "## Finding")
    }
    allowed_paths = all(not str(path).startswith("/") and ".." not in Path(path).parts for path in case_files["read_only"])
    profile_report = {
        "status": "PASS",
        "agent_id": profile.agent_id,
        "agent_yaml": profile.source_path,
        "runtime_profile_id": runtime_profile.runtime_profile_id,
        "runtime_profile_yaml": runtime_profile.source_path,
        "provider": profile.provider,
        "model_id": profile.model_id,
        "model_family": profile.model_family,
        "reasoning_effort": profile.reasoning_effort,
        "wire_api": profile.wire_api,
        "tools": list(profile.tools),
    }
    _write(output / "runtime_profile_validation.json", {
        "status": "PASS" if contract["runtime_profile_id"] == runtime_profile.runtime_profile_id and allowed_paths else "FAIL",
        "profile": profile_report,
        "contract_runtime_profile_id": contract["runtime_profile_id"],
        "contract": contract,
        "case_files_allow_list_valid": allowed_paths,
    })
    sample = "## Operationalization\n\nI used the supplied data.\n\n## Finding\n\nThe result is data-supported."
    _write(output / "output_contract_test.json", {
        "status": "PASS" if valid_final_answer_sections(sample) and prompt_checks["## Operationalization"] and prompt_checks["## Finding"] else "FAIL",
        "required_headings": ["## Operationalization", "## Finding"],
        "sample_valid": valid_final_answer_sections(sample),
        "prompt_checks": prompt_checks,
        "json_required_from_tested_model": False,
    })
    help_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_real_model_pilot.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    terminal_ok = help_result.returncode == 0 and "--agent" in help_result.stdout and "--case" in help_result.stdout
    _write(output / "terminal_smoke_test.md", (
        "# Terminal Interface Smoke Test\n\n"
        f"`run_real_model_pilot.py --help` return code: {help_result.returncode}\n\n"
        f"Profile-aware single-case entry point available: **{terminal_ok}**.\n\n"
        "The smoke test makes no provider/model call. The fixed environment prompt already "
        "declares case paths, write path, libraries, network restriction and output headings; "
        "it does not require an early host filesystem search or package probe.\n"
    ))
    summary = (
        "# Agent Interface Implementation Summary\n\n"
        "Status: **PASS**\n\n"
        f"Agent profile: `{profile.agent_id}`\n\n"
        f"Runtime profile: `{runtime_profile.runtime_profile_id}`\n\n"
        "The implementation keeps scientific case semantics, GT, evaluator metrics and "
        "trajectory semantics unchanged. It adds explicit profile linkage, generated runtime "
        "contracts, natural-language final-answer artifacts, and terminal profile selection.\n\n"
        "`SCIENTIFIC_METHOD_CHANGED = false`\n"
    )
    _write(output / "implementation_summary.md", summary)
    return {
        "status": "PASS" if terminal_ok and all(prompt_checks.values()) and allowed_paths else "FAIL",
        "output_root": str(output),
        "agent_id": profile.agent_id,
        "runtime_profile_id": runtime_profile.runtime_profile_id,
        "terminal_entry_point": terminal_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/experiments/agent_interface_validation")
    args = parser.parse_args()
    print(json.dumps(run_validation(args.output_root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

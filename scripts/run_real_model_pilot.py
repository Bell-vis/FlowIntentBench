"""Run an explicit, non-formal real-model pilot.

This helper deliberately accepts case IDs instead of discovering construction
directories.  It writes ordinary ``RunRecord``/trajectory artifacts and a
small marker stating that the output is a pilot, never a release.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (
    AgentProfile,
    BenchmarkRunner,
    CaseLoader,
    DatasetManifest,
    EvaluationTarget,
    OpenAIChatCompletionsAdapter,
    RunRecord,
    ThirdPartyResponsesAdapter,
    resolve_provider_configuration,
    resolve_api_key,
    evaluation_target_fingerprint,
    validate_reasoning_effort,
    load_agent_profile,
    effective_agent_config,
    reject_profile_identity_overrides,
)


DEFAULT_OFFICE_CASES = (
    "office_speed_zones_o1_f1",
    "office_speed_zones_o2_f1",
    "office_speed_zones_o3_f1",
    "office_speed_zones_o1_f2",
)


def import_collaboration_result(*, loaded_case, target: EvaluationTarget,
                                slot: dict, output_dir: Path,
                                answer: str | None, journal: list[dict],
                                elapsed_seconds: float, status: str,
                                failure_reason: str | None,
                                provenance: dict) -> RunRecord:
    """Import observed collaboration artifacts through the existing record contract.

    This is collection only; no scientific evaluator or model/API is invoked.
    Collaboration is a distinct transport, with unavailable telemetry left null.
    """
    from flowintentbench.model_runner import RunStatus

    runner = BenchmarkRunner(formal_mode=False)
    record = runner._base_record(
        loaded_case, target=target, case_id=slot["case_id"],
        trial_index=slot["trial_index"], ordering_seed=slot["ordering_seed"],
        case_execution_position=slot["case_execution_position"],
        run_id=slot["slot_id"], trajectory_path=output_dir / "trajectory.json",
        experiment_id=provenance["experiment_id"], benchmark_release_id=None,
    )
    record.run_status = RunStatus(status)
    record.final_response = answer
    record.failure_reason = failure_reason
    record.model_turn_count = None
    record.input_tokens = record.output_tokens = record.provider_reported_cost = None
    record.tool_batch_count = None
    record.tool_call_count = len(journal)
    record.python_execution_count = len(journal)
    record.python_error_count = sum(row.get("returncode") != 0 for row in journal)
    record.python_execution_total_time = sum(row["duration_seconds"] for row in journal)
    record.wall_clock_time = elapsed_seconds
    record.protocol_version = "COLLABORATION_STAGED_INSTRUCTIONS_V1"
    record.runtime_profile_id = "collaboration-staged-python-900s-60-v1"
    record.agent_id = target.model_id
    record.runtime_environment_fingerprint = provenance
    record.system_instruction_sha256 = provenance["prompt_sha256"]
    record.network_isolation_active = False
    record.trajectory = [{"event": "collaboration_provenance", **provenance}, *journal]
    record.write_trajectory(output_dir / "trajectory.json")
    record.write_json(output_dir / "run_record.json")
    return record


def parse_model_config_json(raw: str) -> dict[str, Any]:
    """Parse the generic provider/model configuration JSON object."""

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--model-config-json must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("--model-config-json must contain a JSON object")
    return value


def _collection_block_status(reason: object) -> str | None:
    text = str(reason or "").casefold()
    if "provider request timeout" in text or "timed out" in text:
        return "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_TIMEOUT"
    if "provider unavailable" in text or "connection refused" in text:
        return "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_UNAVAILABLE"
    if any(token in text for token in ("429", "quota", "rate limit", "insufficient_quota")):
        return "COLLECTION_BLOCKED_EXTERNAL_QUOTA"
    return None


def run_real_model_pilot(
    *,
    datasets_root: str | Path,
    dataset_id: str,
    case_ids: Sequence[str],
    target: EvaluationTarget,
    output_root: str | Path,
    adapter_factory: Callable[[EvaluationTarget], object],
    runner: BenchmarkRunner | None = None,
    trial_index: int = 1,
    agent_profile: AgentProfile | None = None,
    request_timeout_seconds: float | None = None,
    case_manifest_path: str | Path | None = None,
) -> list[RunRecord]:
    """Run exactly the explicit cases once in development/pilot mode."""

    selected = tuple(case_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("pilot case_ids must be a non-empty unique sequence")
    if trial_index < 1:
        raise ValueError("trial_index must be positive")
    if agent_profile is not None and (
        target.provider != agent_profile.provider or target.model_id != agent_profile.model_id
    ):
        raise ValueError("target provider/model does not match immutable agent profile")
    root = Path(datasets_root).resolve()
    output = Path(output_root).resolve()
    runner = runner or BenchmarkRunner(
        formal_mode=False,
        agent_profile=agent_profile,
    )
    if runner.formal_mode:
        raise ValueError("real-model pilot helper requires BenchmarkRunner(formal_mode=False)")

    inventory = None
    if case_manifest_path is not None:
        from flowintentbench.experiment_scope import case_inventory
        inventory = case_inventory(json.loads(Path(case_manifest_path).read_text()))
    records: list[RunRecord] = []
    target_fp = evaluation_target_fingerprint(target)
    for position, case_id in enumerate(selected):
        case_path = root / dataset_id / "construction" / "cases" / case_id / "case_input.json"
        if inventory is not None:
            import hashlib
            row = inventory[case_id]
            if row["dataset_id"] != dataset_id:
                raise ValueError("case manifest dataset mismatch")
            case_path = (root.parent / row["case_input_path"]).resolve()
            if not case_path.is_relative_to(root):
                raise ValueError("pilot case input must be under datasets root")
            if hashlib.sha256(case_path.read_bytes()).hexdigest() != row["case_input_sha256"]:
                raise ValueError("pilot case input checksum mismatch")
        loaded = CaseLoader(root).load(case_path)
        manifest_path = root / dataset_id / "dataset_manifest.json"
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        case_dir = output / dataset_id / case_id / target_fp / f"trial-{trial_index}"
        case_dir.mkdir(parents=True, exist_ok=False)
        record = runner.run_case(
            loaded,
            target=target,
            case_id=case_id,
            trial_index=trial_index,
            ordering_seed=None,
            case_execution_position=position,
            manifest=manifest,
            trajectory_path=case_dir / "trajectory.json",
            adapter_factory=adapter_factory,
            experiment_id="pilot-not-formal",
            benchmark_release_id=None,
        )
        record.write_json(case_dir / "run_record.json")
        if record.run_status.value == "INFRASTRUCTURE_INVALID" and record.failure_reason:
            block_status = _collection_block_status(record.failure_reason)
            if block_status:
                (case_dir / "incomplete_external_block.json").write_text(
                    json.dumps({"record_type": "IncompleteExternalBlock", "collection_status": block_status, "case_id": case_id, "blocked_case_id": case_id, "dataset_id": dataset_id, "blocked_dataset_id": dataset_id, "trial_index": trial_index, "blocked_trial_index": trial_index, "blocked_attempt_index": 1, "failure_message": record.failure_reason, "provider_error_message": record.failure_reason, "provider_failure_type": type(record.failure_reason).__name__, "resume_allowed": True, "agent_profile_sha256": record.agent_profile_sha256, "runtime_profile_sha256": record.runtime_profile_sha256, "protocol_version": record.protocol_version}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        records.append(record)

    blocked_records = [record for record in records if record.run_status.value == "INFRASTRUCTURE_INVALID" and _collection_block_status(record.failure_reason)]
    blocked_status = None if not blocked_records else _collection_block_status(blocked_records[-1].failure_reason)
    marker = {
        "status": "PILOT / NOT FORMAL BENCHMARK RESULT" if not blocked_records else blocked_status,
        "collection_status": blocked_status,
        "dataset_id": dataset_id,
        "case_ids": list(selected),
        "trial_count": 1,
        "target_fingerprint": target_fp,
        "agent_id": None if agent_profile is None else agent_profile.agent_id,
        "runtime_profile_id": None if agent_profile is None else agent_profile.runtime_profile_id,
        "protocol_version": records[0].protocol_version,
        "agent_profile_sha256": records[0].agent_profile_sha256,
        "runtime_profile_sha256": records[0].runtime_profile_sha256,
        "run_statuses": {record.case_id: record.run_status.value for record in records},
        "provider_timeout_configuration": {"connection_seconds": request_timeout_seconds, "read_seconds": request_timeout_seconds, "overall_request_seconds": request_timeout_seconds},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "pilot_manifest.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--dataset", default="Office")
    parser.add_argument("--case", dest="case_ids", action="append", required=True)
    parser.add_argument(
        "--agent",
        required=True,
        help="agent profile ID or path (for example gpt-5.6-sol-xhigh)",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--server-config", type=Path, default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", dest="model_id", default=None)
    parser.add_argument(
        "--model-config-json",
        default=None,
        help="JSON object passed unchanged to EvaluationTarget.model_configuration",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    try:
        profile = load_agent_profile(args.agent)
        model_configuration = (
            None if args.model_config_json is None else parse_model_config_json(args.model_config_json)
        )
        reject_profile_identity_overrides(
            provider=args.provider, model_id=args.model_id,
            model_configuration=model_configuration,
        )
        profile_configuration = profile.model_configuration
        profile_configuration.update(model_configuration or {})
        model_configuration = profile_configuration
        runtime = resolve_provider_configuration(
            role="model",
            config_path=args.server_config,
            provider=profile.provider,
            base_url=args.base_url,
            model_id=profile.model_id,
            model_configuration=model_configuration,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    target = EvaluationTarget(
        runtime.provider,
        runtime.model_id,
        model_configuration=runtime.model_configuration,
    )
    if runtime.provider != profile.provider or runtime.model_id != profile.model_id:
        parser.error("resolved provider/model does not match immutable agent profile")
    print(json.dumps({"effective_agent_config": effective_agent_config(profile).to_dict()}, indent=2))
    try:
        validate_reasoning_effort(target.model_configuration, label="evaluated model")
    except RuntimeError as exc:
        parser.error(str(exc))
    try:
        provider_api_key = resolve_api_key(runtime, explicit_api_key=args.api_key)
    except RuntimeError as exc:
        parser.error(str(exc))

    def factory(evaluation_target: EvaluationTarget):
        backend = str(runtime.model_configuration.get("execution_backend", "")).casefold()
        wire = str(runtime.model_configuration.get("wire_api", "")).casefold()
        if wire == "responses" or backend in {"third_party_responses", "responses"}:
            return ThirdPartyResponsesAdapter(
                evaluation_target,
                api_key=provider_api_key,
                api_key_env=runtime.api_key_env,
                base_url=runtime.base_url,
                formal_mode=False,
                request_timeout_seconds=args.request_timeout_seconds,
            )
        if wire in {"chat_completions", "chat_completion"}:
            return OpenAIChatCompletionsAdapter(
                evaluation_target,
                api_key=provider_api_key,
                api_key_env=runtime.api_key_env,
                base_url=runtime.base_url,
                formal_mode=False,
                request_timeout_seconds=args.request_timeout_seconds,
            )
        raise ValueError("server/provider configuration must declare a supported wire_api")

    records = run_real_model_pilot(
        datasets_root=args.datasets_root,
        dataset_id=args.dataset,
        case_ids=args.case_ids,
        target=target,
        output_root=args.output_root,
        adapter_factory=factory,
        agent_profile=profile,
        request_timeout_seconds=args.request_timeout_seconds,
        case_manifest_path=args.case_manifest,
    )
    for record in records:
        run_dir = Path(args.output_root).resolve() / args.dataset / record.case_id / evaluation_target_fingerprint(target) / f"trial-{record.trial_index}"
        print(f"Case: {record.case_id}")
        print(f"Agent: {profile.agent_id if profile is not None else target.model_id}")
        print(f"Status: {record.run_status.value}")
        if record.final_answer_available:
            print(f"Final answer: {run_dir / 'final_answer.md'}")
        print(f"Run record: {run_dir / 'run_record.json'}")
        print(f"Trajectory: {run_dir / 'trajectory.json'}")
        print(f"Wall time: {record.wall_clock_time:.3f}s")
        print(f"Input tokens: {record.input_tokens}")
        print(f"Output tokens: {record.output_tokens}")
        print(f"Python calls: {record.python_execution_count}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())

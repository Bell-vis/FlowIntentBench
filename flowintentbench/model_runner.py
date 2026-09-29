"""Provider-independent model evaluation runner infrastructure.

This module coordinates a model adapter with the existing case-scoped Python
runtime.  It deliberately contains no scientific evaluator and does not load
Ground Truth or construction metadata into the evaluated session.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeAlias

from .python_runtime import python_execution_result_payload as _result_to_payload
from uuid import uuid4

from .loader import LoadedCase
from .agent_profile import (
    AgentProfile,
    PROFILED_PROTOCOL_VERSION,
    RuntimeProfile,
    build_python_tool_description,
    build_runtime_system_instruction,
    effective_agent_config,
    load_runtime_profile,
)
from .manifest import DatasetManifest
from .python_runtime import (
    PythonExecutionEnvironment,
    PythonExecutionException,
    PythonExecutionResult,
    PythonRuntimeError,
    RuntimeUnavailableError,
)
from .release import (
    BenchmarkReleaseManifest,
    release_manifest_digest,
    validate_formal_release_case,
)
from .schema import BenchmarkCaseInput
from .provider_retry import ProviderRetryPolicy, failure_category


DEFAULT_SYSTEM_PROMPT_VERSION = "agent-run-v4"
DEFAULT_RUNNER_VERSION = "agent-runner-v1"
DEFAULT_CASE_PRESENTATION_VERSION = "case-input-v1"
PYTHON_TOOL_SPEC_VERSION = "python-tool-v1"
# Keep the provider policy identical for Terra/Luna and for both the CLI and
# direct BenchmarkRunner API.  Callers may still override these values for a
# deliberately pinned experiment, but the default route uses bounded
# exponential backoff rather than immediate reconnect storms.
DEFAULT_PROVIDER_RETRIES = 5
DEFAULT_PROVIDER_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_INFRASTRUCTURE_ATTEMPT_CAP = 3
DEFAULT_CASE_TIMEOUT_SECONDS = 900.0
MAX_TOOL_CALLS_PER_MODEL_TURN = 8
MAX_RETURNED_TOOL_TEXT_PER_BATCH = 48_000

_DEFAULT_RUNTIME_PROFILE = load_runtime_profile("flow-python-v1")
DEFAULT_SYSTEM_PROMPT = build_runtime_system_instruction(_DEFAULT_RUNTIME_PROFILE)


@dataclass(frozen=True)
class PythonToolSpec:
    """Canonical provider-neutral Python tool description."""

    version: str = PYTHON_TOOL_SPEC_VERSION
    name: str = "python"
    description: str = build_python_tool_description(_DEFAULT_RUNTIME_PROFILE)
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": json.loads(json.dumps(self.parameters)),
        }


CANONICAL_PYTHON_TOOL_SPEC = PythonToolSpec()


def valid_final_answer_sections(text: str) -> bool:
    """Return whether a new-format answer has two non-empty Markdown sections."""

    if not isinstance(text, str):
        return False
    lines = text.splitlines()
    headings = [index for index, line in enumerate(lines) if line.strip() in {
        "## Operationalization",
        "## Finding",
    }]
    if len(headings) != 2:
        return False
    first, second = headings
    if lines[first].strip() != "## Operationalization" or lines[second].strip() != "## Finding":
        return False
    operationalization = "\n".join(lines[first + 1:second]).strip()
    finding = "\n".join(lines[second + 1:]).strip()
    return bool(operationalization and finding)


def render_case_input(case_input: BenchmarkCaseInput) -> str:
    """Render one deterministic, provider-independent model user message."""

    payload = case_input.model_dump(mode="json")
    flow_data = json.dumps(payload["flow_data"], ensure_ascii=False, indent=2, sort_keys=True)
    case_context = json.dumps(payload["case_context"], ensure_ascii=False, indent=2, sort_keys=True)
    return (
        "Scientific question:\n"
        f"{payload['scientific_question']}\n\n"
        "Flow data:\n"
        f"{flow_data}\n\n"
        "Case context:\n"
        f"{case_context}"
    )


canonical_case_input_renderer = render_case_input


class RunStatus(str, Enum):
    """Top-level status of one evaluated model run."""

    COMPLETED = "COMPLETED"
    MODEL_NONCOMPLETION = "MODEL_NONCOMPLETION"
    INFRASTRUCTURE_INVALID = "INFRASTRUCTURE_INVALID"


class ModelAdapterError(RuntimeError):
    """Base class for provider/adapter failures owned by infrastructure."""


class ProviderTransportError(ModelAdapterError):
    """A provider failure whose request is known to be safe to replay."""


class ProviderRateLimitError(ProviderTransportError):
    """HTTP 429 provider rate-limit response with optional Retry-After."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.http_status = 429
        self.retry_after_seconds = retry_after_seconds


class ProviderQuotaError(ModelAdapterError):
    """Permanent provider quota/billing rejection (for example HTTP 403)."""

    def __init__(self, message: str, *, http_status: int = 403) -> None:
        super().__init__(message)
        self.http_status = http_status


class ProviderRequestTimeoutError(ProviderTransportError):
    """The external provider connection/read request exceeded its deadline."""


class ProviderUnavailableError(ProviderTransportError):
    """The external provider endpoint could not be reached."""


class ToolBoundaryError(ProviderTransportError):
    """A provider response cannot be represented by the canonical tool batch."""


class ToolBatchLimitError(ModelAdapterError):
    """A provider emitted more calls than the defensive per-turn safety bound."""


class AmbiguousProviderError(ModelAdapterError):
    """A provider failure with unknown delivery state; never replay automatically."""


class FormalExperimentIntegrityError(RuntimeError):
    """Formal execution could not produce a valid experimental observation set."""


@dataclass(frozen=True)
class PythonExecutionRequest:
    """A model request to execute one Python code block."""

    code: str


@dataclass(frozen=True)
class ToolCall:
    """Provider-independent identity and arguments for one Python call."""

    canonical_call_id: str
    provider_call_id: str
    ordinal: int
    tool_name: str
    arguments: Mapping[str, Any]
    arguments_hash: str

    def __post_init__(self) -> None:
        for name in ("canonical_call_id", "provider_call_id", "tool_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        if not isinstance(self.arguments, Mapping) or not isinstance(self.arguments.get("code"), str):
            raise ValueError("tool arguments must contain string code")
        if not isinstance(self.arguments_hash, str) or len(self.arguments_hash) != 64:
            raise ValueError("arguments_hash must be a SHA-256 digest")

    @property
    def code(self) -> str:
        return str(self.arguments["code"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_call_id": self.canonical_call_id,
            "provider_call_id": self.provider_call_id,
            "ordinal": self.ordinal,
            "tool_name": self.tool_name,
            "arguments": json.loads(json.dumps(dict(self.arguments), ensure_ascii=False)),
            "arguments_hash": self.arguments_hash,
        }


@dataclass(frozen=True)
class ToolBatch:
    """Provider-independent ordered batch of Python calls from one model turn."""

    model_turn_id: str
    provider_response_id: str
    calls: tuple[ToolCall, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_turn_id, str) or not self.model_turn_id.strip():
            raise ValueError("model_turn_id must be non-empty")
        if not isinstance(self.provider_response_id, str) or not self.provider_response_id.strip():
            raise ValueError("provider_response_id must be non-empty")
        if not self.calls:
            raise ValueError("ToolBatch must contain at least one call")
        if tuple(call.ordinal for call in self.calls) != tuple(range(len(self.calls))):
            raise ValueError("ToolBatch ordinals must be contiguous and emitted-order aligned")
        ids = [call.canonical_call_id for call in self.calls]
        if len(ids) != len(set(ids)):
            raise ValueError("ToolBatch canonical call IDs must be unique")

    @property
    def batch_id(self) -> str:
        return f"{self.provider_response_id}:{self.model_turn_id}"

    @property
    def code(self) -> str:
        """Backward-compatible access for a single-call batch."""

        if len(self.calls) != 1:
            raise AttributeError("a multi-call ToolBatch has no single code attribute")
        return self.calls[0].code

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_turn_id": self.model_turn_id,
            "provider_response_id": self.provider_response_id,
            "calls": [call.to_dict() for call in self.calls],
        }


@dataclass(frozen=True)
class FinalAnswer:
    """The model's one final natural-language answer."""

    text: str


ModelAction: TypeAlias = PythonExecutionRequest | ToolBatch | FinalAnswer


@dataclass(frozen=True)
class ModelResponse:
    """One completed model response returned by an adapter.

    ``message`` is optional assistant text accompanying a Python request.  It
    is retained in the process trajectory but is not required in a final
    answer.  Provider usage fields are nullable because not every provider
    exposes reliable token accounting. ``input_tokens`` and ``output_tokens``
    are usage deltas for this single completed inference, never cumulative
    conversation totals. Adapters must normalize cumulative provider usage
    before constructing a response.
    """

    action: ModelAction
    message: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    provider_reported_cost: float | None = None
    output_truncated: bool = False
    raw_usage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.provider_reported_cost is not None and self.provider_reported_cost < 0:
            raise ValueError("provider_reported_cost must be non-negative or None")


@dataclass(frozen=True)
class ModelStartRequest:
    """Canonical model-visible initial request with no case metadata."""

    system_prompt: str
    system_prompt_version: str
    user_message: str
    case_presentation_version: str
    python_tool_spec: Mapping[str, Any]
    python_tool_spec_version: str


@dataclass(frozen=True)
class PythonResultEvent:
    """The only model-visible event emitted after a Python request."""

    result: PythonExecutionResult | None = None
    runtime_state_reset: bool = False
    results: tuple[PythonExecutionResult, ...] = ()
    tool_batch: ToolBatch | None = None
    result_call_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.result is not None and self.results and self.results[0] != self.result:
            raise ValueError("result and results disagree")
        if not self.results and self.result is not None:
            object.__setattr__(self, "results", (self.result,))
        if not self.results:
            raise ValueError("PythonResultEvent requires at least one result")
        if self.result_call_ids:
            if len(self.result_call_ids) != len(self.results):
                raise ValueError("result_call_ids must align one-to-one with results")
            if len(set(self.result_call_ids)) != len(self.result_call_ids):
                raise ValueError("result_call_ids must be unique")
            if any(not isinstance(call_id, str) or not call_id.strip() for call_id in self.result_call_ids):
                raise ValueError("result_call_ids must contain non-empty strings")

    @property
    def ordered_results(self) -> tuple[PythonExecutionResult, ...]:
        return self.results

    def results_for_batch(self, batch: ToolBatch) -> tuple[PythonExecutionResult, ...]:
        """Return results in canonical emitted-call order.

        New runner events carry explicit canonical call identities.  Legacy
        single-call/adaptor fixtures may omit them and retain tuple-order
        semantics for backwards compatibility.
        """

        if not self.result_call_ids:
            return self.results
        by_id = dict(zip(self.result_call_ids, self.results))
        expected = tuple(call.canonical_call_id for call in batch.calls)
        if set(by_id) != set(expected):
            raise ValueError("tool result call IDs do not match canonical tool batch")
        return tuple(by_id[call_id] for call_id in expected)


class ModelAdapter(Protocol):
    """Minimal provider-independent adapter contract.

    A formal provider adapter must map every supplied ``timeout_seconds`` to
    an effective provider/API request timeout. Formal eligibility therefore
    requires empirical validation for each concrete provider integration.
    """

    def start_case(self, request: ModelStartRequest, *, timeout_seconds: float) -> ModelResponse:
        """Start a fresh model conversation for one case."""

    def continue_case(self, event: PythonResultEvent, *, timeout_seconds: float) -> ModelResponse:
        """Return the next response after one ordered Python result batch."""

    def close(self) -> None:
        """Release provider/session resources."""


@dataclass(frozen=True)
class EvaluationTarget:
    """Provider/model identity kept outside the scientific case input."""

    provider: str
    model_id: str
    model_configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model_id.strip():
            raise ValueError("provider and model_id must be non-empty")


def evaluation_target_fingerprint(target: EvaluationTarget) -> str:
    """Return a stable identity for one exact evaluated configuration."""

    payload = {
        "provider": target.provider,
        "model_id": target.model_id,
        "model_configuration": dict(target.model_configuration),
    }
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation target must be deterministically JSON serializable") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class RunRecord:
    """Serializable run-level record; it contains no Ground Truth fields."""

    run_id: str
    case_id: str
    trial_index: int
    ordering_seed: int | None
    case_execution_position: int | None
    provider: str
    model_id: str
    model_configuration: dict[str, Any]
    target_fingerprint: str
    experiment_id: str
    benchmark_release_id: str
    system_prompt_version: str
    case_presentation_version: str
    python_tool_spec_version: str
    runner_version: str
    runtime_environment_fingerprint: dict[str, Any] | None
    formal_mode: bool
    network_isolation_active: bool | None
    run_status: RunStatus
    final_response: str | None
    input_tokens: int | None
    output_tokens: int | None
    model_turn_count: int | None
    python_execution_count: int
    wall_clock_time: float
    python_execution_total_time: float | None = None
    python_error_count: int | None = None
    provider_reported_cost: float | None = None
    output_truncated: bool | None = None
    infrastructure_retry_count: int | None = None
    infrastructure_failed_request_time: float | None = None
    retry_backoff_time: float | None = None
    tool_batch_count: int | None = 0
    tool_call_count: int = 0
    trajectory_path: str | None = None
    agent_id: str | None = None
    runtime_profile_id: str | None = None
    protocol_version: str = "LEGACY_UNPROFILED"
    agent_profile_sha256: str | None = None
    runtime_profile_sha256: str | None = None
    system_instruction_sha256: str | None = None
    tool_schema_sha256: str | None = None
    final_answer_path: str | None = None
    final_answer_sha256: str | None = None
    final_answer_available: bool | None = None
    failure_reason: str | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int | None:
        """Derive total tokens only when both standardized components exist."""

        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("trajectory", None)
        payload["run_status"] = self.run_status.value
        payload["total_tokens"] = self.total_tokens
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunRecord":
        """Load a persisted run record without rehydrating its trajectory."""

        payload = dict(value)
        payload.pop("total_tokens", None)
        payload.pop("trajectory", None)
        payload["run_status"] = RunStatus(payload["run_status"])
        payload.setdefault("tool_batch_count", 0)
        # Historical records predate the explicit requested-tool counter. In
        # the frozen Python-only interface, every completed Python execution
        # corresponds to one requested tool call, which is the only sound
        # compatibility projection available for those records.
        payload.setdefault(
            "tool_call_count", payload.get("python_execution_count", 0)
        )
        payload.setdefault("protocol_version", "LEGACY_UNPROFILED")
        payload.setdefault("agent_profile_sha256", None)
        payload.setdefault("runtime_profile_sha256", None)
        payload.setdefault("system_instruction_sha256", None)
        payload.setdefault("tool_schema_sha256", None)
        payload.setdefault("infrastructure_failed_request_time", None)
        payload.setdefault("retry_backoff_time", None)
        return cls(**payload)

    @classmethod
    def load_json(cls, path: str | Path) -> "RunRecord":
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load run record: {source}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("run record JSON must be an object")
        return cls.from_dict(value)

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._write_final_answer(destination.parent)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_final_answer(self, directory: Path) -> None:
        """Persist the canonical human-readable answer for a completed run."""

        answer = directory / "final_answer.md"
        if self.run_status is RunStatus.COMPLETED and self.final_response:
            answer.write_text(self.final_response, encoding="utf-8")
            self.final_answer_path = "final_answer.md"
            self.final_answer_sha256 = hashlib.sha256(self.final_response.encode("utf-8")).hexdigest()
            self.final_answer_available = True
        else:
            if answer.exists():
                answer.unlink()
            self.final_answer_path = None
            self.final_answer_sha256 = None
            self.final_answer_available = False

    def write_trajectory(self, path: str | Path) -> None:
        """Persist process events separately from the run record."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.trajectory, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class TrialOrder:
    """One deterministic case order for a formal repetition."""

    trial_index: int
    ordering_seed: int
    case_ids: tuple[str, ...]


def seeded_case_order(
    case_ids: Sequence[str],
    ordering_seed: int,
    *,
    case_family_by_id: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return a deterministic shuffled order with optional family separation.

    The family adjustment is deliberately a small greedy preference.  It is
    not an optimization problem and never drops or duplicates a case.
    """

    values = [str(case_id) for case_id in case_ids]
    if len(values) != len(set(values)):
        raise ValueError("case_ids must be unique")
    if any(not value for value in values):
        raise ValueError("case_ids must be non-empty")
    result = list(values)
    random.Random(ordering_seed).shuffle(result)
    if case_family_by_id is None or len(result) < 2:
        return tuple(result)
    for index in range(1, len(result)):
        previous_family = case_family_by_id.get(result[index - 1])
        current_family = case_family_by_id.get(result[index])
        if previous_family is None or current_family is None or previous_family != current_family:
            continue
        replacement = next(
            (candidate for candidate in range(index + 1, len(result))
             if case_family_by_id.get(result[candidate]) != previous_family),
            None,
        )
        if replacement is not None:
            result[index], result[replacement] = result[replacement], result[index]
    return tuple(result)


def schedule_formal_trials(
    case_ids: Sequence[str],
    *,
    repetitions: int | None = None,
    base_seed: int = 0,
    case_family_by_id: Mapping[str, str] | None = None,
) -> tuple[TrialOrder, ...]:
    """Create the default independent trial schedule.

    Trial ``i`` uses ``base_seed + i - 1``.  The same schedule is therefore
    shared across evaluated model configurations when the same inputs are used.
    """

    from .experiment import FORMAL_TRIAL_COUNT

    repetitions = FORMAL_TRIAL_COUNT if repetitions is None else repetitions
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    return tuple(
        TrialOrder(
            trial_index=index,
            ordering_seed=base_seed + index - 1,
            case_ids=seeded_case_order(
                case_ids,
                base_seed + index - 1,
                case_family_by_id=case_family_by_id,
            ),
        )
        for index in range(1, repetitions + 1)
    )


RuntimeFactory: TypeAlias = Callable[..., PythonExecutionEnvironment]
ModelAdapterFactory: TypeAlias = Callable[[EvaluationTarget], ModelAdapter]


def run_formal_trials(
    runner: "BenchmarkRunner",
    *,
    case_ids: Sequence[str],
    loaded_cases: Mapping[str, LoadedCase],
    manifests: Mapping[str, DatasetManifest],
    target: EvaluationTarget,
    adapter_factory: ModelAdapterFactory,
    output_root: str | Path,
    release_manifest: BenchmarkReleaseManifest | Mapping[str, Any] | None = None,
    experiment_id: str = "formal-experiment",
    base_seed: int = 0,
    case_family_by_id: Mapping[str, str] | None = None,
    infrastructure_attempt_cap: int = DEFAULT_INFRASTRUCTURE_ATTEMPT_CAP,
    repository_root: str | Path | None = None,
) -> list[RunRecord]:
    """Run an explicit case set for the formal default of three trials.

    The helper never scans repository directories.  Callers supply the exact
    selected case IDs and their already-loaded case/manifest objects.
    """

    if not runner.formal_mode:
        raise ValueError("run_formal_trials requires BenchmarkRunner(formal_mode=True)")
    if runner.protocol_version != PROFILED_PROTOCOL_VERSION:
        raise FormalExperimentIntegrityError(
            "formal execution requires an explicit agent profile"
        )
    if infrastructure_attempt_cap <= 0:
        raise ValueError("infrastructure_attempt_cap must be positive")
    if release_manifest is None:
        raise ValueError("formal execution requires an explicit BenchmarkReleaseManifest")
    from .experiment import (
        EvaluatedModelConfiguration,
        ExperimentManifest,
        FORMAL_TRIAL_COUNT,
        code_data_version,
        save_experiment_manifest,
    )
    release = (
        release_manifest
        if isinstance(release_manifest, BenchmarkReleaseManifest)
        else BenchmarkReleaseManifest.model_validate(release_manifest)
    )
    selected = tuple(case_ids)
    if not selected:
        raise ValueError("formal case selection must not be empty")
    release_entries = [release.case_for_id(case_id) for case_id in selected]
    if case_family_by_id is None:
        case_family_by_id = {entry.case_id: entry.case_family_id for entry in release_entries}
    else:
        missing_families = sorted(case_id for case_id in selected if case_id not in case_family_by_id)
        if missing_families:
            raise ValueError(f"case_family_by_id is incomplete: {missing_families}")
        mismatches = sorted(
            entry.case_id
            for entry in release_entries
            if case_family_by_id[entry.case_id] != entry.case_family_id
        )
        if mismatches:
            raise ValueError(f"case_family_by_id disagrees with release manifest: {mismatches}")
    missing_cases = sorted(set(selected) - set(loaded_cases))
    missing_manifests = sorted(set(selected) - set(manifests))
    if missing_cases or missing_manifests:
        raise ValueError(
            f"formal case selection is incomplete: missing cases={missing_cases}, "
            f"missing manifests={missing_manifests}"
        )
    # Validate every released artifact and the underlying data identity before
    # creating any adapter, runtime, or formal output directory.
    for case_id, release_entry in zip(selected, release_entries):
        try:
            validate_formal_release_case(
                release_entry,
                loaded_cases[case_id],
                manifests[case_id],
            )
        except Exception as exc:
            raise FormalExperimentIntegrityError(
                f"formal release validation failed for {case_id!r}: {exc}"
            ) from exc
    output_root = Path(output_root)
    target_fingerprint = evaluation_target_fingerprint(target)
    target_root = (
        output_root
        / _safe_component(experiment_id)
        / _safe_component(target.provider)
        / _safe_component(target.model_id)
        / target_fingerprint
    )
    schedule = schedule_formal_trials(
        selected,
        base_seed=base_seed,
        case_family_by_id=case_family_by_id,
    )
    case_directories = {
        (trial.trial_index, case_id): (
            target_root / f"trial-{trial.trial_index}" / _safe_component(case_id)
        )
        for trial in schedule
        for case_id in trial.case_ids
    }
    if len(set(case_directories.values())) != len(case_directories):
        raise ValueError("formal case IDs collide after output-path normalization")
    collisions = sorted(str(path) for path in case_directories.values() if path.exists())
    manifest_path = target_root / "experiment_manifest.json"
    if manifest_path.exists():
        collisions.append(str(manifest_path))
    if collisions:
        raise FileExistsError(
            "formal result output already exists; refusing overwrite: " + ", ".join(collisions)
        )
    records: list[RunRecord] = []
    valid_records: list[RunRecord] = []
    for trial in schedule:
        for position, case_id in enumerate(trial.case_ids, start=1):
            case_dir = case_directories[(trial.trial_index, case_id)]
            attempt = 1
            while True:
                attempt_dir = (
                    case_dir
                    if attempt == 1
                    else case_dir / f"attempt-{attempt}"
                )
                attempt_dir.mkdir(parents=True, exist_ok=False)
                record = runner.run_case(
                    loaded_cases[case_id],
                    target=target,
                    case_id=case_id,
                    trial_index=trial.trial_index,
                    ordering_seed=trial.ordering_seed,
                    case_execution_position=position,
                    manifest=manifests[case_id],
                    trajectory_path=attempt_dir / "trajectory.json",
                    adapter_factory=adapter_factory,
                    experiment_id=experiment_id,
                    benchmark_release_id=release.release_id,
                )
                record.write_json(attempt_dir / "run_record.json")
                records.append(record)
                if record.run_status is not RunStatus.INFRASTRUCTURE_INVALID:
                    valid_records.append(record)
                    break
                if attempt >= infrastructure_attempt_cap:
                    raise FormalExperimentIntegrityError(
                        f"formal trial slot {case_id!r} trial {trial.trial_index} "
                        f"exhausted infrastructure attempt cap {infrastructure_attempt_cap} "
                        "after INFRASTRUCTURE_INVALID attempts: "
                        f"{record.failure_reason or 'unknown infrastructure failure'}"
                    )
                attempt += 1
    if len(valid_records) != len(selected) * FORMAL_TRIAL_COUNT:
        raise FormalExperimentIntegrityError(
            f"formal experiment did not produce exactly {FORMAL_TRIAL_COUNT} trials per case"
        )
    if any(
        record.run_status not in {RunStatus.COMPLETED, RunStatus.MODEL_NONCOMPLETION}
        for record in valid_records
    ):
        raise FormalExperimentIntegrityError("formal experiment contains an invalid observation")
    runtime_fingerprints = [record.runtime_environment_fingerprint for record in valid_records]
    if any(fingerprint is None for fingerprint in runtime_fingerprints):
        raise FormalExperimentIntegrityError(
            "formal experiment requires a non-null runtime fingerprint for every run"
        )
    runtime_fingerprint = runtime_fingerprints[0]
    if any(fingerprint != runtime_fingerprint for fingerprint in runtime_fingerprints[1:]):
        raise FormalExperimentIntegrityError(
            "formal experiment runtime fingerprints are not identical"
        )
    experiment_manifest = ExperimentManifest(
        experiment_id=experiment_id,
        benchmark_release_id=release.release_id,
        benchmark_release_digest=release_manifest_digest(release),
        evaluated_model_configuration=EvaluatedModelConfiguration(
            provider=target.provider,
            model_id=target.model_id,
            model_configuration=dict(target.model_configuration),
            target_fingerprint=target_fingerprint,
        ),
        system_prompt_version=runner.system_prompt_version,
        case_presentation_version=runner.case_presentation_version,
        python_tool_spec_version=runner.python_tool_spec.version,
        runner_version=runner.runner_version,
        runtime_environment_fingerprint=runtime_fingerprint,
        formal_trial_count=FORMAL_TRIAL_COUNT,
        trial_seeds=[trial.ordering_seed for trial in schedule],
        generated_case_order=[list(trial.case_ids) for trial in schedule],
        benchmark_code_data_version=code_data_version(
            Path(repository_root).resolve()
            if repository_root is not None
            else Path(__file__).resolve().parents[1]
        ),
        protocol_version=runner.protocol_version,
        agent_id=runner.agent_id,
        runtime_profile_id=runner.runtime_profile_id,
        agent_profile_sha256=runner.agent_profile_sha256,
        runtime_profile_sha256=runner.runtime_profile_sha256,
        system_instruction_sha256=runner.system_instruction_sha256,
        tool_schema_sha256=runner.tool_schema_sha256,
    )
    save_experiment_manifest(experiment_manifest, manifest_path)
    return records


def _safe_component(value: str) -> str:
    normalized = str(value).strip().replace("/", "_").replace("\\", "_")
    if normalized in {".", ".."}:
        normalized = normalized.replace(".", "_")
    return normalized or "unnamed"


def _result_from_payload(value: Mapping[str, Any]) -> PythonExecutionResult:
    exception = value.get("exception")
    error = None
    if isinstance(exception, Mapping):
        error = PythonExecutionException(
            type=str(exception.get("type", "RuntimeError")),
            message=str(exception.get("message", "")),
            traceback=str(exception.get("traceback", "")),
        )
    return PythonExecutionResult(
        success=bool(value.get("success")),
        stdout=str(value.get("stdout", "")),
        stderr=str(value.get("stderr", "")),
        exception=error,
        duration_seconds=float(value.get("duration_seconds", 0.0)),
        execution_index=int(value.get("execution_index", 0)),
        output_truncated=bool(value.get("output_truncated", False)),
        stdout_chars_total=value.get("stdout_chars_total"),
        stderr_chars_total=value.get("stderr_chars_total"),
        returned_stdout_chars=value.get("returned_stdout_chars"),
        returned_stderr_chars=value.get("returned_stderr_chars"),
        filesystem_discovery_blocked=bool(value.get("filesystem_discovery_blocked", False)),
    )


def _result_hash(result: PythonExecutionResult) -> str:
    return hashlib.sha256(
        json.dumps(_result_to_payload(result), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _legacy_request_batch(request: PythonExecutionRequest, *, model_turn_id: int) -> ToolBatch:
    arguments = {"code": request.code}
    arguments_hash = hashlib.sha256(
        json.dumps(arguments, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    call_id = f"legacy-turn-{model_turn_id}-call-0"
    call = ToolCall(
        canonical_call_id=call_id,
        provider_call_id=call_id,
        ordinal=0,
        tool_name="python",
        arguments=arguments,
        arguments_hash=arguments_hash,
    )
    return ToolBatch(
        model_turn_id=f"legacy-turn-{model_turn_id}",
        provider_response_id=f"legacy-response-{model_turn_id}",
        calls=(call,),
    )


class BenchmarkRunner:
    """Run one model adapter against one isolated benchmark case."""

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        system_prompt_version: str = DEFAULT_SYSTEM_PROMPT_VERSION,
        case_presentation_version: str = DEFAULT_CASE_PRESENTATION_VERSION,
        python_tool_spec: PythonToolSpec | None = None,
        runner_version: str = DEFAULT_RUNNER_VERSION,
        runtime_factory: RuntimeFactory = PythonExecutionEnvironment,
        runtime_kwargs: Mapping[str, Any] | None = None,
        provider_retries: int = DEFAULT_PROVIDER_RETRIES,
        provider_retry_backoff_seconds: float = DEFAULT_PROVIDER_RETRY_BACKOFF_SECONDS,
        provider_rate_limit_backoff_seconds: float = 15.0,
        provider_rate_limit_max_backoff_seconds: float = 60.0,
        case_timeout_seconds: float = DEFAULT_CASE_TIMEOUT_SECONDS,
        formal_mode: bool = True,
        agent_id: str | None = None,
        runtime_profile_id: str | None = None,
        require_final_answer_sections: bool | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> None:
        if agent_profile is not None:
            effective = effective_agent_config(agent_profile)
            if agent_id is not None and agent_id != effective.agent_id:
                raise ValueError("agent_id conflicts with agent profile")
            if runtime_profile_id is not None and runtime_profile_id != effective.runtime_profile_id:
                raise ValueError("runtime_profile_id conflicts with agent profile")
            supplied_runtime_kwargs = dict(runtime_kwargs or {})
            existing_profile = supplied_runtime_kwargs.get("runtime_profile")
            if existing_profile is not None and existing_profile != agent_profile.runtime_profile:
                raise ValueError("runtime profile override conflicts with agent profile")
            supplied_runtime_kwargs["runtime_profile"] = agent_profile.runtime_profile
            system_prompt = build_runtime_system_instruction(agent_profile.runtime_profile)
            python_tool_spec = PythonToolSpec(
                description=build_python_tool_description(agent_profile.runtime_profile)
            )
            agent_id = effective.agent_id
            runtime_profile_id = effective.runtime_profile_id
            if require_final_answer_sections is None:
                require_final_answer_sections = True
        else:
            effective = None
            supplied_runtime_kwargs = dict(runtime_kwargs or {})
            system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
            python_tool_spec = python_tool_spec or CANONICAL_PYTHON_TOOL_SPEC
            if require_final_answer_sections is None:
                require_final_answer_sections = False
        if not system_prompt.strip() or not system_prompt_version.strip() or not runner_version.strip():
            raise ValueError("system prompt and version values must be non-empty")
        if not case_presentation_version.strip() or not python_tool_spec.version.strip():
            raise ValueError("presentation and tool specification versions must be non-empty")
        if provider_retries < 0:
            raise ValueError("provider_retries must be non-negative")
        if provider_retry_backoff_seconds < 0:
            raise ValueError("provider_retry_backoff_seconds must be non-negative")
        if provider_rate_limit_backoff_seconds < 0:
            raise ValueError("provider_rate_limit_backoff_seconds must be non-negative")
        if provider_rate_limit_max_backoff_seconds < provider_rate_limit_backoff_seconds:
            raise ValueError(
                "provider_rate_limit_max_backoff_seconds must be at least provider_rate_limit_backoff_seconds"
            )
        if case_timeout_seconds <= 0:
            raise ValueError("case_timeout_seconds must be positive")
        if formal_mode:
            if supplied_runtime_kwargs.get("require_network_isolation") is False:
                raise ValueError("formal_mode requires network isolation")
            if supplied_runtime_kwargs.get("python_executable") is not None:
                raise ValueError("formal_mode requires the frozen benchmark Python environment")
        self.system_prompt = system_prompt
        self.system_prompt_version = system_prompt_version
        self.case_presentation_version = case_presentation_version
        self.python_tool_spec = python_tool_spec
        self.runner_version = runner_version
        self.runtime_factory = runtime_factory
        self.runtime_kwargs = supplied_runtime_kwargs
        self.provider_retries = provider_retries
        self.provider_retry_backoff_seconds = float(provider_retry_backoff_seconds)
        self.provider_rate_limit_backoff_seconds = float(provider_rate_limit_backoff_seconds)
        self.provider_rate_limit_max_backoff_seconds = float(provider_rate_limit_max_backoff_seconds)
        self.provider_retry_policy = ProviderRetryPolicy(
            transient_backoff_seconds=self.provider_retry_backoff_seconds,
            rate_limit_backoff_seconds=self.provider_rate_limit_backoff_seconds,
            rate_limit_max_backoff_seconds=self.provider_rate_limit_max_backoff_seconds,
        )
        self.case_timeout_seconds = case_timeout_seconds
        self.formal_mode = formal_mode
        self.agent_id = agent_id
        self.runtime_profile_id = runtime_profile_id
        self.require_final_answer_sections = require_final_answer_sections
        self.protocol_version = (
            PROFILED_PROTOCOL_VERSION if effective is not None else "LEGACY_UNPROFILED"
        )
        self.effective_agent_config = effective
        self.agent_profile_sha256 = None if effective is None else effective.agent_profile_sha256
        self.runtime_profile_sha256 = None if effective is None else effective.runtime_profile_sha256
        self.system_instruction_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        self.tool_schema_sha256 = hashlib.sha256(
            json.dumps(python_tool_spec.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def run_case(
        self,
        loaded_case: LoadedCase,
        adapter: ModelAdapter | None = None,
        *,
        target: EvaluationTarget,
        case_id: str,
        trial_index: int = 1,
        ordering_seed: int | None = None,
        case_execution_position: int | None = None,
        manifest: DatasetManifest | None = None,
        run_id: str | None = None,
        trajectory_path: str | Path | None = None,
        adapter_factory: ModelAdapterFactory | None = None,
        experiment_id: str | None = None,
        benchmark_release_id: str | None = None,
    ) -> RunRecord:
        """Execute one case and return a run record.

        ``case_id`` is stored only in evaluator-private metadata and is never
        included in ``ModelStartRequest`` or runtime model-visible files.
        """

        if not case_id.strip():
            raise ValueError("case_id must be non-empty")
        if trial_index <= 0:
            raise ValueError("trial_index must be positive")
        if adapter is not None and adapter_factory is not None:
            raise ValueError("provide adapter or adapter_factory, not both")
        if self.formal_mode and adapter is not None:
            raise ValueError("formal_mode requires adapter_factory for a fresh model session")
        if self.formal_mode and manifest is None:
            return self._invalid_record(
                loaded_case,
                target=target,
                case_id=case_id,
                trial_index=trial_index,
                ordering_seed=ordering_seed,
                case_execution_position=case_execution_position,
                run_id=run_id,
                trajectory_path=trajectory_path,
                experiment_id=experiment_id,
                benchmark_release_id=benchmark_release_id,
                reason="formal runner requires a DatasetManifest",
            )
        if manifest is not None:
            try:
                manifest.validate_against_case(loaded_case.case)
            except Exception as exc:
                return self._invalid_record(
                    loaded_case,
                    target=target,
                    case_id=case_id,
                    trial_index=trial_index,
                    ordering_seed=ordering_seed,
                    case_execution_position=case_execution_position,
                    run_id=run_id,
                    trajectory_path=trajectory_path,
                    experiment_id=experiment_id,
                    benchmark_release_id=benchmark_release_id,
                    reason=f"manifest validation failed: {exc}",
                )
        if adapter is None and adapter_factory is not None:
            try:
                adapter = adapter_factory(target)
            except Exception as exc:
                return self._invalid_record(
                    loaded_case,
                    target=target,
                    case_id=case_id,
                    trial_index=trial_index,
                    ordering_seed=ordering_seed,
                    case_execution_position=case_execution_position,
                    run_id=run_id,
                    trajectory_path=trajectory_path,
                    experiment_id=experiment_id,
                    benchmark_release_id=benchmark_release_id,
                    reason=f"adapter factory failed: {type(exc).__name__}: {exc}",
                )
        if adapter is None:
            raise ValueError("adapter or adapter_factory is required")

        record = self._base_record(
            loaded_case,
            target=target,
            case_id=case_id,
            trial_index=trial_index,
            ordering_seed=ordering_seed,
            case_execution_position=case_execution_position,
            run_id=run_id,
            trajectory_path=(str(trajectory_path) if trajectory_path is not None else None),
            experiment_id=experiment_id,
            benchmark_release_id=benchmark_release_id,
        )
        runtime: PythonExecutionEnvironment | None = None
        interaction_started: float | None = None
        input_total = 0
        output_total = 0
        input_complete = True
        output_complete = True
        cost_total = 0.0
        cost_complete = True
        execution_journal: dict[str, dict[str, Any]] = {}
        journal_path = (
            Path(trajectory_path).with_name("tool_execution_journal.json")
            if trajectory_path is not None
            else None
        )
        if journal_path is not None and journal_path.is_file():
            try:
                loaded_journal = json.loads(journal_path.read_text(encoding="utf-8"))
                if isinstance(loaded_journal, Mapping):
                    execution_journal = {
                        str(key): dict(value)
                        for key, value in loaded_journal.items()
                        if isinstance(value, Mapping)
                    }
            except (OSError, json.JSONDecodeError):
                execution_journal = {}

        def save_journal() -> None:
            if journal_path is None:
                return
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            journal_path.write_text(
                json.dumps(execution_journal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        try:
            runtime = self.runtime_factory(
                loaded_case,
                manifest=manifest,
                **self.runtime_kwargs,
            )
            if self.formal_mode:
                if not isinstance(runtime, PythonExecutionEnvironment):
                    raise RuntimeUnavailableError("formal_mode requires PythonExecutionEnvironment")
                if runtime.frozen_environment is None or not runtime.require_network_isolation:
                    raise RuntimeUnavailableError(
                        "formal_mode requires frozen environment and network isolation"
                    )
            runtime.start()
            interaction_started = time.perf_counter()
            response = self._adapter_call(
                adapter.start_case,
                ModelStartRequest(
                    system_prompt=self.system_prompt,
                    system_prompt_version=self.system_prompt_version,
                    user_message=render_case_input(loaded_case.case),
                    case_presentation_version=self.case_presentation_version,
                    python_tool_spec=self.python_tool_spec.to_dict(),
                    python_tool_spec_version=self.python_tool_spec.version,
                ),
                record,
                interaction_started,
            )
            while True:
                self._record_response(response, record)
                if response.input_tokens is None:
                    input_complete = False
                else:
                    input_total += response.input_tokens
                if response.output_tokens is None:
                    output_complete = False
                else:
                    output_total += response.output_tokens
                if response.provider_reported_cost is None:
                    cost_complete = False
                else:
                    cost_total += response.provider_reported_cost
                record.output_truncated = bool(record.output_truncated or response.output_truncated)
                if self._deadline_exceeded(interaction_started):
                    record.run_status = RunStatus.MODEL_NONCOMPLETION
                    record.failure_reason = "case safety timeout"
                    break
                if isinstance(response.action, FinalAnswer):
                    record.final_response = response.action.text
                    if response.output_truncated:
                        record.run_status = RunStatus.MODEL_NONCOMPLETION
                        record.failure_reason = "final response truncated by generation limit"
                    elif response.action.text.strip() and (
                        not self.require_final_answer_sections
                        or valid_final_answer_sections(response.action.text)
                    ):
                        record.run_status = RunStatus.COMPLETED
                    else:
                        record.run_status = RunStatus.MODEL_NONCOMPLETION
                        record.failure_reason = (
                            "empty final response"
                            if not response.action.text.strip()
                            else "final response missing required Operationalization and Finding sections"
                        )
                    break
                if isinstance(response.action, PythonExecutionRequest):
                    batch = _legacy_request_batch(response.action, model_turn_id=record.model_turn_count)
                elif isinstance(response.action, ToolBatch):
                    batch = response.action
                else:
                    raise ModelAdapterError("adapter returned an unsupported model action")
                if len(batch.calls) > MAX_TOOL_CALLS_PER_MODEL_TURN:
                    raise ToolBatchLimitError(
                        f"TOOL_BATCH_LIMIT: provider emitted {len(batch.calls)} calls; "
                        f"maximum is {MAX_TOOL_CALLS_PER_MODEL_TURN}"
                    )
                if self._deadline_exceeded(interaction_started):
                    record.run_status = RunStatus.MODEL_NONCOMPLETION
                    record.failure_reason = "case safety timeout before final response"
                    break
                record.tool_batch_count += 1
                record.tool_call_count += len(batch.calls)
                batch_started = time.perf_counter()
                results: list[PythonExecutionResult] = []
                batch_agent_errors = 0
                batch_infrastructure_errors = 0
                for call in batch.calls:
                    journal_entry = execution_journal.get(call.canonical_call_id)
                    journal_reused = bool(
                        journal_entry is not None
                        and journal_entry.get("status") == "COMPLETED"
                        and journal_entry.get("arguments_hash") == call.arguments_hash
                    )
                    if journal_reused:
                        result = _result_from_payload(journal_entry.get("result", {}))
                    else:
                        record.python_execution_count += 1
                        started_at = time.time()
                        result = runtime.execute(call.code)
                        completed_at = time.time()
                        classification = (
                            "AGENT_CODE_ERROR" if not result.success else None
                        )
                        execution_journal[call.canonical_call_id] = {
                            "canonical_call_id": call.canonical_call_id,
                            "provider_call_id": call.provider_call_id,
                            "model_turn_id": batch.model_turn_id,
                            "ordinal": call.ordinal,
                            "tool_name": call.tool_name,
                            "arguments_hash": call.arguments_hash,
                            "status": "COMPLETED",
                            "started_at": started_at,
                            "completed_at": completed_at,
                            "runtime_seconds": result.duration_seconds,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "stdout_original_chars": result.stdout_chars_total,
                            "stderr_original_chars": result.stderr_chars_total,
                            "stdout_truncated": result.output_truncated,
                            "stderr_truncated": result.output_truncated,
                            "error_classification": classification,
                            "result": _result_to_payload(result),
                            "result_hash": _result_hash(result),
                        }
                        save_journal()
                    results.append(result)
                    if not journal_reused:
                        record.python_execution_total_time = (
                            record.python_execution_total_time or 0.0
                        ) + result.duration_seconds
                    if not result.success:
                        if not journal_reused:
                            record.python_error_count = (record.python_error_count or 0) + 1
                        batch_agent_errors += 1
                    record.output_truncated = bool(
                        record.output_truncated or _execution_output_truncated(result)
                    )
                    record.trajectory.append(
                        {
                            "event": "python_execution",
                            "batch_id": batch.batch_id,
                            "canonical_call_id": call.canonical_call_id,
                            "provider_call_id": call.provider_call_id,
                            "ordinal": call.ordinal,
                            "execution_index": result.execution_index,
                            "success": result.success,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "duration_seconds": result.duration_seconds,
                            "output_truncated": result.output_truncated,
                            "stdout_chars_total": result.stdout_chars_total,
                            "stderr_chars_total": result.stderr_chars_total,
                            "returned_stdout_chars": result.returned_stdout_chars,
                            "returned_stderr_chars": result.returned_stderr_chars,
                            "filesystem_discovery_blocked": result.filesystem_discovery_blocked,
                            "journal_reused": journal_reused,
                            "exception": (
                                {
                                    "type": result.exception.type,
                                    "message": result.exception.message,
                                }
                                if result.exception is not None
                                else None
                            ),
                        }
                    )
                    if not runtime.started:
                        exception_type = result.exception.type if result.exception else ""
                        # A guarded code timeout is recoverable through the
                        # existing fresh interpreter reset.  Broken transport
                        # or process loss is not: the call's side effect is
                        # ambiguous and the observation must be invalidated.
                        if exception_type not in {"TimeoutError", "FilesystemDiscoveryError"}:
                            batch_infrastructure_errors += 1
                            raise RuntimeUnavailableError(
                                "Python runtime became unavailable during a tool batch; "
                                "execution integrity cannot be guaranteed"
                            )
                batch_runtime = time.perf_counter() - batch_started
                record.trajectory.append(
                    {
                        "event": "tool_batch",
                        "batch_id": batch.batch_id,
                        "model_turn_id": batch.model_turn_id,
                        "provider_response_id": batch.provider_response_id,
                        "batch_size": len(batch.calls),
                        "batch_runtime": batch_runtime,
                        "number_success": sum(result.success for result in results),
                        "number_agent_errors": batch_agent_errors,
                        "number_infrastructure_errors": batch_infrastructure_errors,
                    }
                )
                state_reset = False
                if not runtime.started:
                    runtime.reset()
                    state_reset = True
                if self._deadline_exceeded(interaction_started):
                    record.run_status = RunStatus.MODEL_NONCOMPLETION
                    record.failure_reason = "case safety timeout"
                    break
                response = self._adapter_call(
                    adapter.continue_case,
                    PythonResultEvent(
                        results=tuple(results),
                        result=results[0] if len(results) == 1 else None,
                        runtime_state_reset=state_reset,
                        tool_batch=batch,
                        result_call_ids=tuple(
                            call.canonical_call_id for call in batch.calls
                        ),
                    ),
                    record,
                    interaction_started,
                )
        except _ProviderDeadlineExceeded as exc:
            record.run_status = RunStatus.INFRASTRUCTURE_INVALID
            record.failure_reason = str(exc)
        except _CaseDeadlineExceeded as exc:
            record.run_status = RunStatus.MODEL_NONCOMPLETION
            record.failure_reason = str(exc)
        except _ExhaustedProviderError as exc:
            record.run_status = RunStatus.INFRASTRUCTURE_INVALID
            record.failure_reason = str(exc)
        except ToolBatchLimitError as exc:
            record.trajectory.append(
                {
                    "event": "tool_batch_limit",
                    "status": "INFRASTRUCTURE_INVALID",
                    "message": str(exc),
                    "max_calls": MAX_TOOL_CALLS_PER_MODEL_TURN,
                }
            )
            record.run_status = RunStatus.INFRASTRUCTURE_INVALID
            record.failure_reason = str(exc)
        except (RuntimeUnavailableError, PythonRuntimeError, ModelAdapterError, OSError) as exc:
            record.run_status = RunStatus.INFRASTRUCTURE_INVALID
            record.failure_reason = str(exc)
        except Exception as exc:  # runner/provider implementation failures are infrastructure failures
            record.run_status = RunStatus.INFRASTRUCTURE_INVALID
            record.failure_reason = f"runner infrastructure exception: {type(exc).__name__}: {exc}"
        finally:
            try:
                adapter.close()
            except Exception as exc:
                record.run_status = RunStatus.INFRASTRUCTURE_INVALID
                record.failure_reason = f"adapter close failed: {exc}"
            if runtime is not None:
                try:
                    record.runtime_environment_fingerprint = runtime.environment_fingerprint
                    if self.runtime_profile_id is None and hasattr(runtime, "runtime_profile"):
                        record.runtime_profile_id = runtime.runtime_profile.runtime_profile_id
                    record.network_isolation_active = (
                        runtime.network_isolation_active
                        if hasattr(runtime, "network_isolation_active")
                        else None
                    )
                finally:
                    runtime.close()

        record.input_tokens = input_total if input_complete else None
        record.output_tokens = output_total if output_complete else None
        record.provider_reported_cost = cost_total if cost_complete else None
        record.wall_clock_time = (
            max(0.0, time.perf_counter() - interaction_started)
            if interaction_started is not None
            else 0.0
        )
        if trajectory_path:
            record.write_trajectory(trajectory_path)
        return record

    def _adapter_call(
        self,
        method: Callable[..., ModelResponse],
        payload: Any,
        record: RunRecord,
        interaction_started: float,
    ) -> ModelResponse:
        retries = 0
        while True:
            remaining = self._remaining_case_time(interaction_started)
            if remaining <= 0:
                raise _CaseDeadlineExceeded("case safety deadline expired before provider request")
            request_started = time.perf_counter()
            try:
                response = method(payload, timeout_seconds=remaining)
                if not isinstance(response, ModelResponse):
                    raise ModelAdapterError("adapter must return ModelResponse")
                return response
            except ProviderTransportError as exc:
                record.infrastructure_failed_request_time = (
                    record.infrastructure_failed_request_time or 0.0
                ) + max(0.0, time.perf_counter() - request_started)
                remaining_after_failure = self._remaining_case_time(interaction_started)
                if remaining_after_failure <= 0:
                    # The case deadline was consumed while the provider
                    # request was failing. This is an infrastructure outcome,
                    # not a model noncompletion.
                    raise _ProviderDeadlineExceeded(
                        "case safety deadline expired during provider request"
                    ) from exc
                if retries >= self.provider_retries:
                    exhausted = _ExhaustedProviderError(
                        f"provider transport failed after {retries} retries: {exc}"
                    )
                    for name in ("http_status", "retry_after_seconds"):
                        if hasattr(exc, name):
                            setattr(exhausted, name, getattr(exc, name))
                    raise exhausted from exc
                retries += 1
                record.infrastructure_retry_count = (record.infrastructure_retry_count or 0) + 1
                record.trajectory.append(
                    {
                        "event": "infrastructure_retry",
                        "attempt": retries,
                        "category": (
                            "TOOL_INTERFACE_ERROR"
                            if isinstance(exc, ToolBoundaryError)
                            else "RATE_LIMIT"
                            if failure_category(exc) == "RATE_LIMIT"
                            else failure_category(exc)
                            if failure_category(exc) in {"TIMEOUT", "UNAVAILABLE"}
                            else "PROVIDER_TRANSPORT_ERROR"
                        ),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "http_status": getattr(exc, "http_status", None),
                        "retry_after_seconds": getattr(
                            exc, "retry_after_seconds", None
                        ),
                    }
                )
                if self.provider_retry_backoff_seconds or failure_category(exc) == "RATE_LIMIT":
                    # Retry the exact unacknowledged request after a bounded,
                    # deterministic delay.  Retry-After is honored for 429;
                    # otherwise transient and rate-limit schedules are frozen
                    # in the shared provider policy.
                    delay = self.provider_retry_policy.delay(retries - 1, exc)
                    sleep_for = min(delay, remaining_after_failure)
                    time.sleep(sleep_for)
                    record.retry_backoff_time = (record.retry_backoff_time or 0.0) + sleep_for
                    if sleep_for >= remaining_after_failure or self._remaining_case_time(interaction_started) <= 0:
                        raise _ProviderDeadlineExceeded(
                            "case safety deadline expired during provider retry backoff"
                        ) from exc

    def _record_response(self, response: ModelResponse, record: RunRecord) -> None:
        record.model_turn_count += 1
        event = {
            "event": "model_turn",
            "action": type(response.action).__name__,
            "message": response.message,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "raw_usage": json.loads(json.dumps(response.raw_usage)),
            "provider_reported_cost": response.provider_reported_cost,
            "output_truncated": response.output_truncated,
        }
        if isinstance(response.action, ToolBatch):
            event["tool_batch_count"] = 1
            event["number_of_tool_calls"] = len(response.action.calls)
            event["tool_batch"] = response.action.to_dict()
        elif isinstance(response.action, PythonExecutionRequest):
            event["code"] = response.action.code
            event["tool_batch_count"] = 1
            event["number_of_tool_calls"] = 1
        else:
            event["final_response"] = response.action.text
        record.trajectory.append(event)

    def _deadline_exceeded(self, interaction_started: float | None) -> bool:
        return interaction_started is not None and (
            time.perf_counter() - interaction_started >= self.case_timeout_seconds
        )

    def _remaining_case_time(self, interaction_started: float) -> float:
        return max(0.0, self.case_timeout_seconds - (time.perf_counter() - interaction_started))

    def _base_record(self, loaded_case: LoadedCase, *, target: EvaluationTarget, case_id: str, trial_index: int, ordering_seed: int | None, case_execution_position: int | None, run_id: str | None, trajectory_path: str | Path | None, experiment_id: str | None = None, benchmark_release_id: str | None = None) -> RunRecord:
        return RunRecord(
            run_id=run_id or uuid4().hex,
            case_id=case_id,
            trial_index=trial_index,
            ordering_seed=ordering_seed,
            case_execution_position=case_execution_position,
            provider=target.provider,
            model_id=target.model_id,
            model_configuration=dict(target.model_configuration),
            target_fingerprint=evaluation_target_fingerprint(target),
            experiment_id=experiment_id or "",
            benchmark_release_id=benchmark_release_id or "",
            system_prompt_version=self.system_prompt_version,
            case_presentation_version=self.case_presentation_version,
            python_tool_spec_version=self.python_tool_spec.version,
            runner_version=self.runner_version,
            runtime_environment_fingerprint=None,
            formal_mode=self.formal_mode,
            network_isolation_active=None,
            run_status=RunStatus.INFRASTRUCTURE_INVALID,
            final_response=None,
            input_tokens=None,
            output_tokens=None,
            model_turn_count=0,
            python_execution_count=0,
            wall_clock_time=0.0,
            python_execution_total_time=0.0,
            python_error_count=0,
            output_truncated=False,
            infrastructure_retry_count=0,
            infrastructure_failed_request_time=0.0,
            retry_backoff_time=0.0,
            trajectory_path=(str(trajectory_path) if trajectory_path is not None else None),
            agent_id=self.agent_id,
            runtime_profile_id=self.runtime_profile_id,
            protocol_version=self.protocol_version,
            agent_profile_sha256=self.agent_profile_sha256,
            runtime_profile_sha256=self.runtime_profile_sha256,
            system_instruction_sha256=self.system_instruction_sha256,
            tool_schema_sha256=self.tool_schema_sha256,
        )

    def _invalid_record(self, loaded_case: LoadedCase, *, target: EvaluationTarget, case_id: str, trial_index: int, ordering_seed: int | None, case_execution_position: int | None, run_id: str | None, trajectory_path: str | Path | None, reason: str, experiment_id: str | None = None, benchmark_release_id: str | None = None) -> RunRecord:
        record = self._base_record(
            loaded_case,
            target=target,
            case_id=case_id,
            trial_index=trial_index,
            ordering_seed=ordering_seed,
            case_execution_position=case_execution_position,
            run_id=run_id,
            trajectory_path=trajectory_path,
            experiment_id=experiment_id,
            benchmark_release_id=benchmark_release_id,
        )
        record.failure_reason = reason
        if trajectory_path:
            record.write_trajectory(trajectory_path)
        return record


class _ExhaustedProviderError(ModelAdapterError):
    """Internal marker for an exhausted retryable provider failure."""


class _ProviderDeadlineExceeded(ModelAdapterError):
    """Provider transport consumed the case deadline; classify as infra."""


class _CaseDeadlineExceeded(ModelAdapterError):
    """Internal marker for a total case deadline reached during adapter I/O."""


def _execution_output_truncated(result: PythonExecutionResult) -> bool:
    return bool(
        result.output_truncated
        or "[output truncated by runtime limit]" in result.stderr
    )

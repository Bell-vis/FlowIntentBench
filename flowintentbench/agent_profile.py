"""Explicit agent and benchmark runtime profiles.

Profiles are deliberately limited to stable execution information.  They do
not contain case answers, Ground Truth, evaluator configuration, or scientific
analysis hints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class AgentProfileError(ValueError):
    """Raised when an agent or runtime profile is missing or malformed."""


@dataclass(frozen=True)
class RuntimeProfile:
    runtime_profile_id: str
    case_root: str
    workspace_root: str
    case_root_read_only: bool
    network_available: bool
    tools: dict[str, dict[str, Any]]
    libraries: dict[str, str]
    root_recursive_search: str
    max_returned_text_chars_per_call: int
    source_path: str
    profile_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_profile_id": self.runtime_profile_id,
            "filesystem": {
                "case_root": self.case_root,
                "workspace_root": self.workspace_root,
                "case_root_read_only": self.case_root_read_only,
            },
            "network": {"available": self.network_available},
            "tools": self.tools,
            "libraries": self.libraries,
            "filesystem_policy": {"root_recursive_search": self.root_recursive_search},
            "output_policy": {
                "max_returned_text_chars_per_call": self.max_returned_text_chars_per_call,
            },
        }


@dataclass(frozen=True)
class AgentProfile:
    agent_id: str
    provider: str
    model_id: str
    model_family: str
    reasoning_effort: str
    wire_api: str
    runtime_profile_id: str
    tools: tuple[str, ...]
    runtime_profile: RuntimeProfile
    source_path: str
    profile_sha256: str

    @property
    def model_configuration(self) -> dict[str, Any]:
        backend = (
            "THIRD_PARTY_RESPONSES"
            if self.wire_api.casefold() == "responses"
            else "OPENAI_CHAT_COMPLETIONS"
        )
        return {
            "reasoning_effort": self.reasoning_effort,
            "wire_api": self.wire_api,
            "execution_backend": backend,
        }


@dataclass(frozen=True)
class EffectiveAgentConfig:
    """Immutable identity and protocol resolved before a model call."""

    agent_id: str
    provider: str
    model_id: str
    model_family: str
    reasoning_effort: str
    wire_api: str
    runtime_profile_id: str
    tools: tuple[str, ...]
    protocol_version: str
    agent_profile_sha256: str
    runtime_profile_sha256: str
    system_instruction_sha256: str
    tool_schema_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "reasoning_effort": self.reasoning_effort,
            "wire_api": self.wire_api,
            "runtime_profile_id": self.runtime_profile_id,
            "tools": list(self.tools),
            "protocol_version": self.protocol_version,
            "agent_profile_sha256": self.agent_profile_sha256,
            "runtime_profile_sha256": self.runtime_profile_sha256,
            "system_instruction_sha256": self.system_instruction_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
        }


PROFILED_PROTOCOL_VERSION = "profiled-natural-output-v1"
PROFILE_IDENTITY_KEYS = frozenset({
    "provider", "model", "model_id", "model_family", "reasoning_effort",
    "wire_api", "execution_backend", "runtime_profile", "runtime_profile_id", "tools",
})


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_runtime_system_instruction(profile: RuntimeProfile) -> str:
    if not profile.tools:
        return (
            "Review only the supplied model-visible scientific packet. No tools, Python runtime, "
            "filesystem access, web access, or network access are available. Return only the "
            "structured response required by the role instruction."
        )
    libraries = ", ".join(profile.libraries)
    network = "available" if profile.network_available else "unavailable"
    return (
        "Answer the supplied scientific question. Analyze the supplied flow data when needed; "
        "a persistent Python environment is available. When the question leaves scientific "
        "analysis choices unspecified, select appropriate choices yourself. Runtime paths are "
        f"fixed and non-scientific: read-only case files are under {profile.case_root}, "
        f"{profile.case_root}/case_input.json contains the model-visible case input, "
        f"{profile.case_root}/reader_metadata.json contains reader-level decoding metadata when "
        f"provided, {profile.case_root}/case_files.json lists the bounded visible file set, and "
        f"{profile.case_root}/runtime_contract.json lists runtime capabilities. {libraries} are "
        f"available; network access is {network}. Write scripts and intermediate artifacts only "
        f"under {profile.workspace_root}. Inspect those paths directly; do not search the host "
        "filesystem or use the network. Your final response must be natural language with these "
        "two sections and no JSON: '## Operationalization' describing the consequential analytical "
        "choices you actually used, followed by '## Finding' describing the result supported by "
        "the data."
    )


def build_python_tool_description(profile: RuntimeProfile) -> str:
    libraries = ", ".join(profile.libraries)
    return (
        "Execute Python in a persistent environment for the current case. Read case data from "
        f"{profile.case_root} and write scripts or artifacts under {profile.workspace_root}. "
        f"Network access is {'available' if profile.network_available else 'unavailable'}; "
        f"{libraries} are available. Normal Python errors are returned to the model. Recursive "
        f"discovery from filesystem root is {profile.root_recursive_search}."
    )


def build_runtime_contract(
    profile: RuntimeProfile,
    *,
    package_versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    versions = dict(package_versions or {})
    imports = {"NumPy": "numpy", "SciPy": "scipy", "Matplotlib": "matplotlib", "VTK": "vtk"}
    return {
        "contract_version": "python-runtime-contract-v1",
        "runtime_profile_id": profile.runtime_profile_id,
        "runtime_profile_sha256": profile.profile_sha256,
        "case_root": profile.case_root,
        "workspace_root": profile.workspace_root,
        "network_access": "available" if profile.network_available else "unavailable",
        "tools": profile.tools,
        "libraries": dict(profile.libraries),
        "supported_core_scientific_libraries": [
            {"name": name, "import_name": imports.get(name, name.casefold()),
             "version": versions.get(imports.get(name, name.casefold()).casefold().replace("-", "_"))}
            for name in profile.libraries
        ],
        "output_policy": {
            "max_returned_text_chars_per_call": profile.max_returned_text_chars_per_call,
            "max_returned_text_characters_per_stream": profile.max_returned_text_chars_per_call,
            "large_output": "head_tail_preview_with_explicit_telemetry",
        },
        "filesystem_policy": {
            "root_recursive_discovery": profile.root_recursive_search,
            "bounded_case_listing": f"{profile.case_root}/case_files.json",
            "case_root_read_only": profile.case_root_read_only,
        },
    }


def effective_agent_config(profile: AgentProfile) -> EffectiveAgentConfig:
    instruction = build_runtime_system_instruction(profile.runtime_profile)
    tool_schema = (
        {
            "name": "python",
            "description": build_python_tool_description(profile.runtime_profile),
            "parameters": {
                "type": "object", "properties": {"code": {"type": "string"}},
                "required": ["code"], "additionalProperties": False,
            },
        }
        if "python" in profile.tools
        else {"tools": []}
    )
    return EffectiveAgentConfig(
        agent_id=profile.agent_id, provider=profile.provider, model_id=profile.model_id,
        model_family=profile.model_family, reasoning_effort=profile.reasoning_effort,
        wire_api=profile.wire_api, runtime_profile_id=profile.runtime_profile_id,
        tools=profile.tools, protocol_version=PROFILED_PROTOCOL_VERSION,
        agent_profile_sha256=profile.profile_sha256,
        runtime_profile_sha256=profile.runtime_profile.profile_sha256,
        system_instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        tool_schema_sha256=_canonical_sha256(tool_schema),
    )


def reject_profile_identity_overrides(
    *,
    provider: str | None = None,
    model_id: str | None = None,
    model_configuration: Mapping[str, Any] | None = None,
) -> None:
    overrides = []
    if provider is not None:
        overrides.append("--provider")
    if model_id is not None:
        overrides.append("--model")
    overrides.extend(sorted(PROFILE_IDENTITY_KEYS.intersection(model_configuration or {})))
    if overrides:
        raise AgentProfileError(
            "agent profile identity is immutable; remove identity overrides: "
            + ", ".join(overrides)
        )


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgentProfileError(f"{label} must be an object")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentProfileError(f"{label} must be a non-empty string")
    return value.strip()


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AgentProfileError(f"failed to read profile: {path}") from exc
    return _object(value, str(path))


def _resolve_profile_path(value: str | Path, *, root: Path, kind: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        if kind == "agent" and candidate.name != "agent.yaml":
            candidate = root / "agents" / candidate / "agent.yaml"
        elif kind == "runtime" and candidate.suffix not in {".yaml", ".yml"}:
            candidate = root / "runtime_profiles" / f"{candidate}.yaml"
        else:
            candidate = root / candidate
    return candidate.resolve()


def load_runtime_profile(value: str | Path = "flow-python-v1", *, repository_root: str | Path | None = None) -> RuntimeProfile:
    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[1]
    path = _resolve_profile_path(value, root=root, kind="runtime")
    document = _read_yaml(path)
    profile_id = _required_string(document.get("runtime_profile_id"), "runtime_profile_id")
    filesystem = _object(document.get("filesystem"), "filesystem")
    network = _object(document.get("network"), "network")
    tools_raw = _object(document.get("tools"), "tools")
    libraries_raw = _object(document.get("libraries"), "libraries")
    policy = _object(document.get("filesystem_policy"), "filesystem_policy")
    output = _object(document.get("output_policy"), "output_policy")
    max_chars = output.get("max_returned_text_chars_per_call")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise AgentProfileError("output_policy.max_returned_text_chars_per_call must be positive")
    tools = {str(name): dict(_object(spec, f"tools.{name}")) for name, spec in tools_raw.items()}
    libraries = {str(name): _required_string(version, f"libraries.{name}") for name, version in libraries_raw.items()}
    if not isinstance(filesystem.get("case_root_read_only"), bool):
        raise AgentProfileError("filesystem.case_root_read_only must be boolean")
    if not isinstance(network.get("available"), bool):
        raise AgentProfileError("network.available must be boolean")
    return RuntimeProfile(
        runtime_profile_id=profile_id,
        case_root=_required_string(filesystem.get("case_root"), "filesystem.case_root"),
        workspace_root=_required_string(filesystem.get("workspace_root"), "filesystem.workspace_root"),
        case_root_read_only=filesystem["case_root_read_only"],
        network_available=network["available"],
        tools=tools,
        libraries=libraries,
        root_recursive_search=_required_string(policy.get("root_recursive_search"), "filesystem_policy.root_recursive_search"),
        max_returned_text_chars_per_call=max_chars,
        source_path=str(path),
        profile_sha256=_canonical_sha256(document),
    )


def load_agent_profile(value: str | Path, *, repository_root: str | Path | None = None) -> AgentProfile:
    root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[1]
    path = _resolve_profile_path(value, root=root, kind="agent")
    document = _read_yaml(path)
    agent_id = _required_string(document.get("agent_id"), "agent_id")
    model = _object(document.get("model"), "model")
    interface = _object(document.get("interface"), "interface")
    tools_raw = document.get("tools")
    if not isinstance(tools_raw, list) or any(not isinstance(item, str) or not item.strip() for item in tools_raw):
        raise AgentProfileError("tools must be a list of names")
    runtime_profile_id = _required_string(document.get("runtime_profile"), "runtime_profile")
    runtime_profile = load_runtime_profile(runtime_profile_id, repository_root=root)
    if runtime_profile.runtime_profile_id != runtime_profile_id:
        raise AgentProfileError("agent runtime_profile does not match the referenced profile")
    undeclared_tools = sorted(set(str(item) for item in tools_raw) - set(runtime_profile.tools))
    if undeclared_tools:
        raise AgentProfileError(
            "agent tools are not available in the runtime profile: "
            + ", ".join(undeclared_tools)
        )
    return AgentProfile(
        agent_id=agent_id,
        provider=_required_string(model.get("provider"), "model.provider"),
        model_id=_required_string(model.get("model_id"), "model.model_id"),
        model_family=_required_string(model.get("model_family"), "model.model_family"),
        reasoning_effort=_required_string(model.get("reasoning_effort"), "model.reasoning_effort"),
        wire_api=_required_string(interface.get("wire_api"), "interface.wire_api"),
        runtime_profile_id=runtime_profile_id,
        tools=tuple(str(item) for item in tools_raw),
        runtime_profile=runtime_profile,
        source_path=str(path),
        profile_sha256=_canonical_sha256(document),
    )


__all__ = [
    "AgentProfile", "AgentProfileError", "EffectiveAgentConfig", "RuntimeProfile",
    "PROFILED_PROTOCOL_VERSION", "PROFILE_IDENTITY_KEYS", "build_python_tool_description",
    "build_runtime_contract", "build_runtime_system_instruction", "effective_agent_config",
    "load_agent_profile", "load_runtime_profile", "reject_profile_identity_overrides",
]

"""Resolve provider/runtime settings from the benchmark server configuration.

This module intentionally owns deployment configuration only.  No scientific
method, dataset, or provider hostname is encoded here.  Resolution precedence
is explicit override, then the existing server/Codex configuration, then a
fail-closed error.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class RuntimeConfigurationError(RuntimeError):
    """The server/runtime provider configuration is missing or invalid."""


SUPPORTED_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})


def validate_reasoning_effort(
    model_configuration: Mapping[str, Any],
    *,
    label: str = "model",
) -> None:
    """Fail closed for reasoning values unsupported by the local benchmark route.

    The provider endpoint is not probed here.  This check only rejects known
    unsupported values (notably ``ultra`` on the current local route); the
    preflight/backend smoke remains responsible for provider-specific support.
    No automatic downgrade is performed.
    """

    value = model_configuration.get("reasoning_effort")
    if value is None:
        return
    if not isinstance(value, str) or value.casefold() not in SUPPORTED_REASONING_EFFORTS:
        supported = ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
        raise RuntimeConfigurationError(
            f"{label} reasoning_effort={value!r} is unsupported by the frozen local route; "
            f"choose one of: {supported}"
        )


@dataclass(frozen=True)
class ProviderRuntimeConfiguration:
    provider: str
    base_url: str
    model_id: str
    model_configuration: dict[str, Any]
    api_key_env: str = "OPENAI_API_KEY"
    source_path: str | None = None
    model_family: str | None = None
    api_key_file: str | None = None

    def without_secrets(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "model_configuration": dict(self.model_configuration),
            "api_key_env": self.api_key_env,
            "source_path": self.source_path,
            "api_key_file": self.api_key_file,
        }

    @property
    def configuration_digest(self) -> str:
        encoded = json.dumps(
            self.without_secrets(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def resolve_api_key(
    configuration: ProviderRuntimeConfiguration,
    *,
    explicit_api_key: str | None = None,
    auth_path: str | Path | None = None,
) -> str:
    """Resolve a credential without returning it in any experiment artifact."""

    if explicit_api_key:
        return explicit_api_key
    from_environment = os.environ.get(configuration.api_key_env)
    if from_environment:
        return from_environment
    if configuration.api_key_file:
        key_file = Path(configuration.api_key_file).expanduser()
        if key_file.is_file():
            key = key_file.read_text(encoding="utf-8").strip()
            if key:
                return key
    candidates: list[Path] = []
    if auth_path is not None:
        candidates.append(Path(auth_path).expanduser())
    if configuration.source_path:
        source = Path(configuration.source_path).expanduser()
        candidates.append(source.parent / "auth.json")
        # Project deployments commonly keep the ignored credential file next
        # to the repository's `config/` directory rather than inside it.
        candidates.append(source.parent.parent / "auth.json")
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        candidates.append(Path(codex_home).expanduser() / "auth.json")
    candidates.append(Path.home() / ".codex" / "auth.json")
    for candidate in candidates:
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping) and isinstance(value.get(configuration.api_key_env), str):
            key = value[configuration.api_key_env].strip()
            if key:
                return key
        if isinstance(value, Mapping) and configuration.api_key_env != "OPENAI_API_KEY":
            fallback = value.get("OPENAI_API_KEY")
            if isinstance(fallback, str) and fallback.strip():
                return fallback.strip()
    raise RuntimeConfigurationError(
        f"no credential available for configured provider; set {configuration.api_key_env}, "
        "provide an explicit API key, or configure the server auth file"
    )


def default_server_config_path() -> Path | None:
    """Return the current server/Codex config path without a URL fallback."""

    candidates: list[Path] = []
    explicit = os.environ.get("FLOWINTENTBENCH_SERVER_CONFIG")
    if explicit:
        # An explicit deployment path must never silently fall back.
        return Path(explicit).expanduser().resolve()
    candidates.append(Path(__file__).resolve().parents[1] / "config.toml")
    candidates.append(Path(__file__).resolve().parents[1] / "config/yiapi.toml")
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        candidates.append(Path(codex_home).expanduser() / "config.toml")
    candidates.append(Path.home() / ".codex" / "config.toml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeConfigurationError(f"server configuration {label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigurationError(f"server configuration {label} must be a non-empty string")
    return value.strip()


def _load_document(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeConfigurationError(f"failed to read server configuration: {path}") from exc
    return _mapping(value, "root")


def _role_values(
    document: Mapping[str, Any],
    role: str,
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> tuple[str, str, Mapping[str, Any]]:
    """Resolve a role identity and its provider block.

    An explicit role override may use a provider block that is declared under
    the model role when the server configuration has no separate evaluator
    role.  This is intentionally limited to the explicitly named provider;
    it reuses deployment endpoint/auth metadata only and never reuses the
    configured model identity.
    """
    role_key = role.strip().casefold()
    if role_key not in {"model", "evaluator"}:
        raise RuntimeConfigurationError("runtime configuration role must be model or evaluator")
    if role_key == "model":
        provider = document.get("model_provider")
        model = document.get("model")
        providers = document.get("model_providers", {})
    else:
        provider = document.get("evaluator_provider")
        model = document.get("evaluator_model")
        providers = document.get("evaluator_providers", {})
        evaluator = document.get("evaluator")
        if isinstance(evaluator, Mapping):
            provider = evaluator.get("provider", provider)
            model = evaluator.get("model", evaluator.get("model_id", model))
            providers = evaluator.get("providers", providers)
    # Explicit CLI/profile values are authoritative for identity.  Applying
    # them before the required-field check lets an evaluator be selected on a
    # server config that only declares the default model role.
    if provider_override is not None:
        provider = provider_override
    if model_override is not None:
        model = model_override
    if provider is None or model is None:
        raise RuntimeConfigurationError(
            f"server configuration has no designated {role_key} provider/model"
        )
    provider_name = _nonempty_string(provider, f"{role_key}_provider")
    model_id = _nonempty_string(model, f"{role_key}_model")
    provider_map = _mapping(providers, f"{role_key}_providers")
    provider_config = provider_map.get(provider_name)
    if (
        provider_config is None
        and role_key == "evaluator"
        and provider_override is not None
    ):
        # A local Codex-compatible config commonly declares one provider map
        # for the default model but no evaluator role.  An explicit evaluator
        # override may still select that exact provider block for its endpoint
        # and credential environment.  No evaluator identity is inferred.
        fallback_map = _mapping(
            document.get("model_providers", {}), "model_providers"
        )
        provider_config = fallback_map.get(provider_name)
    if provider_config is None:
        raise RuntimeConfigurationError(
            f"server configuration has no provider block for {provider_name!r}"
        )
    return provider_name, model_id, _mapping(provider_config, f"{role_key} provider {provider_name}")


def load_server_provider_configuration(
    path: str | Path | None = None,
    *,
    role: str = "model",
    provider_override: str | None = None,
    model_override: str | None = None,
) -> ProviderRuntimeConfiguration:
    """Load one provider role from the existing server/Codex TOML config."""

    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parents[1] / config_path
        config_path = config_path.resolve()
    else:
        config_path = default_server_config_path()
    if config_path is None or not config_path.is_file():
        raise RuntimeConfigurationError(
            "no server/provider configuration is available; provide an explicit runtime override "
            "or FLOWINTENTBENCH_SERVER_CONFIG"
        )
    document = _load_document(config_path)
    provider, model_id, provider_block = _role_values(
        document,
        role,
        provider_override=provider_override,
        model_override=model_override,
    )
    base_url = provider_block.get("base_url")
    if base_url is None:
        base_url = provider_block.get("endpoint")
    base_url = _nonempty_string(base_url, f"{role}.base_url")
    wire_api = provider_block.get("wire_api")
    model_configuration: dict[str, Any] = {}
    configured = provider_block.get("model_configuration")
    if configured is not None:
        model_configuration.update(dict(_mapping(configured, f"{role}.model_configuration")))
    if role.casefold() == "model":
        reasoning = document.get("model_reasoning_effort")
    else:
        reasoning = document.get("evaluator_reasoning_effort")
    if reasoning is not None:
        model_configuration.setdefault("reasoning_effort", reasoning)
    if wire_api is not None:
        model_configuration.setdefault("wire_api", wire_api)
        if str(wire_api).casefold().replace("-", "_") == "responses":
            model_configuration.setdefault("execution_backend", "THIRD_PARTY_RESPONSES")
        elif str(wire_api).casefold() in {"chat_completions", "chat_completion"}:
            model_configuration.setdefault("execution_backend", "OPENAI_CHAT_COMPLETIONS")
    api_key_env = provider_block.get("api_key_env", provider_block.get("env_key", "OPENAI_API_KEY"))
    api_key_env = _nonempty_string(api_key_env, f"{role}.api_key_env")
    family = provider_block.get("model_family")
    if family is None:
        family = document.get("model_family" if role.casefold() == "model" else "evaluator_model_family")
    if family is not None:
        family = _nonempty_string(family, f"{role}.model_family")
    return ProviderRuntimeConfiguration(
        provider=provider,
        base_url=base_url,
        model_id=model_id,
        model_configuration=model_configuration,
        model_family=family,
        api_key_env=api_key_env,
        source_path=str(config_path),
        api_key_file=provider_block.get("api_key_file"),
    )


def resolve_provider_configuration(
    *,
    role: str = "model",
    config_path: str | Path | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    model_id: str | None = None,
    model_configuration: Mapping[str, Any] | None = None,
) -> ProviderRuntimeConfiguration:
    """Resolve explicit overrides over the existing server configuration."""

    server: ProviderRuntimeConfiguration | None = None
    needs_server = any(value is None for value in (provider, base_url, model_id))
    if needs_server or config_path is not None:
        try:
            # When an endpoint is also explicit, an identity override should
            # not make an otherwise valid server config fail merely because it
            # has no provider block for that override (the explicit endpoint
            # is authoritative).  The fallback preserves the historical
            # behavior of loading the configured role for shared metadata such
            # as the credential environment.
            server = load_server_provider_configuration(
                config_path,
                role=role,
                provider_override=provider,
                model_override=model_id,
            )
        except RuntimeConfigurationError:
            if provider is None or model_id is None or base_url is None:
                raise
            server = load_server_provider_configuration(config_path, role=role)
    resolved_provider = _nonempty_string(provider if provider is not None else server.provider, "provider")
    resolved_base_url = _nonempty_string(base_url if base_url is not None else server.base_url, "base_url")
    resolved_model = _nonempty_string(model_id if model_id is not None else server.model_id, "model_id")
    merged = dict(server.model_configuration) if server is not None else {}
    if model_configuration is not None:
        merged.update(dict(model_configuration))
    api_key_env = server.api_key_env if server is not None else "OPENAI_API_KEY"
    return ProviderRuntimeConfiguration(
        provider=resolved_provider,
        base_url=resolved_base_url.rstrip("/"),
        model_id=resolved_model,
        model_configuration=merged,
        model_family=None if server is None else server.model_family,
        api_key_env=api_key_env,
        source_path=None if server is None else server.source_path,
        api_key_file=None if server is None else server.api_key_file,
    )


__all__ = [
    "ProviderRuntimeConfiguration",
    "RuntimeConfigurationError",
    "default_server_config_path",
    "load_server_provider_configuration",
    "resolve_api_key",
    "resolve_provider_configuration",
    "SUPPORTED_REASONING_EFFORTS",
    "validate_reasoning_effort",
]

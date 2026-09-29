"""Concrete provider adapters for the provider-neutral model runner.

These adapters do not alter the scientific question, select an
operationalization, inspect Ground Truth, or perform evaluation.  They only
translate the frozen runner contract to one explicitly selected wire API.
Chat Completions and Responses remain separate transports and are never used
as silent fallbacks for one another.
"""

from __future__ import annotations

import json
import hashlib
import os
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

from .python_runtime import python_execution_result_payload as _result_payload

from .model_runner import (
    AmbiguousProviderError,
    CANONICAL_PYTHON_TOOL_SPEC,
    EvaluationTarget,
    FinalAnswer,
    ModelAdapterError,
    ModelResponse,
    ModelStartRequest,
    ProviderTransportError,
    ProviderRateLimitError,
    ProviderQuotaError,
    ProviderRequestTimeoutError,
    ProviderUnavailableError,
    ToolBoundaryError,
    ToolBatch,
    ToolBatchLimitError,
    ToolCall,
    MAX_TOOL_CALLS_PER_MODEL_TURN,
    MAX_RETURNED_TOOL_TEXT_PER_BATCH,
    PythonExecutionRequest,
    PythonResultEvent,
)
from .provider_retry import parse_retry_after


Transport = Callable[[str, Mapping[str, str], bytes, float], bytes]

OPENAI_ALLOWED_MODEL_CONFIGURATION_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "reasoning_effort",
        "seed",
        "stream",
    }
)

RESPONSES_ALLOWED_MODEL_CONFIGURATION_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "max_output_tokens",
        "reasoning_effort",
        "seed",
    }
)


def _urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        message = payload.decode("utf-8", errors="replace")
        if exc.code in {403, 429} and any(code in message.casefold() for code in (
            "insufficient_quota", "billing_hard_limit_reached"
        )):
            raise ProviderQuotaError(f"provider HTTP {exc.code}: {message}") from exc
        if exc.code == 408:
            raise ProviderRequestTimeoutError(f"provider HTTP {exc.code}: {message}") from exc
        if exc.code == 429:
            retry_after = parse_retry_after(
                exc.headers.get("Retry-After") if exc.headers else None
            )
            raise ProviderRateLimitError(
                f"provider HTTP {exc.code}: {message}",
                retry_after_seconds=retry_after,
            ) from exc
        if exc.code == 403 and any(
            token in message.casefold() for token in ("quota", "insufficient", "balance", "billing", "credit")
        ):
            raise ProviderQuotaError(f"provider HTTP {exc.code}: {message}") from exc
        if exc.code >= 500:
            # A server-side response is an external provider outage.  The
            # request did not yield a usable model observation and may be
            # retried under the collection protocol.
            raise ProviderUnavailableError(f"provider HTTP {exc.code}: {message}") from exc
        raise ModelAdapterError(f"provider HTTP {exc.code}: {message}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise ProviderRequestTimeoutError(f"provider request timeout: {exc}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason).casefold():
            raise ProviderRequestTimeoutError(f"provider request timeout: {reason}") from exc
        raise ProviderUnavailableError(f"provider unavailable: {reason}") from exc
    except OSError as exc:
        if isinstance(exc, (socket.timeout, TimeoutError)) or "timed out" in str(exc).casefold():
            raise ProviderRequestTimeoutError(f"provider request timeout: {exc}") from exc
        raise ProviderUnavailableError(f"provider unavailable: {exc}") from exc


def _arguments_hash(arguments: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _canonical_call(
    *,
    provider_response_id: str,
    provider_call_id: str,
    ordinal: int,
    tool_name: str,
    arguments: Mapping[str, Any],
) -> ToolCall:
    return ToolCall(
        canonical_call_id=f"{provider_response_id}:{provider_call_id}",
        provider_call_id=provider_call_id,
        ordinal=ordinal,
        tool_name=tool_name,
        arguments=dict(arguments),
        arguments_hash=_arguments_hash(arguments),
    )


def _batch(
    *, provider_response_id: str, model_turn_id: str, calls: list[ToolCall]
) -> ToolBatch:
    if len(calls) > MAX_TOOL_CALLS_PER_MODEL_TURN:
        raise ToolBatchLimitError(
            f"TOOL_BATCH_LIMIT: provider emitted {len(calls)} calls; maximum is "
            f"{MAX_TOOL_CALLS_PER_MODEL_TURN}"
        )
    return ToolBatch(model_turn_id=model_turn_id, provider_response_id=provider_response_id, calls=tuple(calls))


def _legacy_batch_for_event(results: tuple[Any, ...], tool_name: str) -> ToolBatch:
    calls = []
    for ordinal, _result in enumerate(results):
        provider_call_id = f"legacy-python-call-{ordinal}"
        arguments = {"code": ""}
        calls.append(
            _canonical_call(
                provider_response_id="legacy-response",
                provider_call_id=provider_call_id,
                ordinal=ordinal,
                tool_name=tool_name,
                arguments=arguments,
            )
        )
    return _batch(
        provider_response_id="legacy-response",
        model_turn_id="legacy-turn",
        calls=calls,
    )


def _trim_preview(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n...[batch output truncated]...\n"
    if limit <= len(marker):
        return marker[:limit]
    available = max(0, limit - len(marker))
    head = available // 2
    return value[:head] + marker + value[-(available - head) :]


def _bounded_stream_limits(lengths: list[int], budget: int) -> list[int]:
    """Allocate a deterministic shared budget while preserving small streams."""

    limits = [0] * len(lengths)
    remaining = set(range(len(lengths)))
    available = max(0, budget)
    while remaining:
        fair = available // len(remaining)
        small = [index for index in remaining if lengths[index] <= fair]
        if small:
            for index in small:
                limits[index] = lengths[index]
                available -= lengths[index]
                remaining.remove(index)
            continue
        for offset, index in enumerate(sorted(remaining)):
            limits[index] = fair + int(offset < available % len(remaining))
        break
    return limits


def _result_payloads(
    event: PythonResultEvent,
    results: tuple[Any, ...] | None = None,
) -> list[dict[str, Any]]:
    payloads = [
        _result_payload(result)
        for result in (event.ordered_results if results is None else results)
    ]
    total = sum(len(str(item.get("stdout", ""))) + len(str(item.get("stderr", ""))) for item in payloads)
    if total <= MAX_RETURNED_TOOL_TEXT_PER_BATCH or not payloads:
        return payloads
    streams = [
        str(item.get(stream, ""))
        for item in payloads
        for stream in ("stdout", "stderr")
    ]
    limits = _bounded_stream_limits(
        [len(value) for value in streams], MAX_RETURNED_TOOL_TEXT_PER_BATCH
    )
    for item_index, item in enumerate(payloads):
        stdout = streams[item_index * 2]
        stderr = streams[item_index * 2 + 1]
        bounded_stdout = _trim_preview(stdout, limits[item_index * 2])
        bounded_stderr = _trim_preview(stderr, limits[item_index * 2 + 1])
        item["stdout"] = bounded_stdout
        item["stderr"] = bounded_stderr
        item["returned_stdout_chars"] = len(item["stdout"])
        item["returned_stderr_chars"] = len(item["stderr"])
        if bounded_stdout != stdout or bounded_stderr != stderr:
            item["batch_output_truncated"] = True
    return payloads


class OpenAIChatCompletionsAdapter:
    """One fresh, stateful adapter instance for one case conversation."""

    def __init__(
        self,
        target: EvaluationTarget,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str,
        organization: str | None = None,
        transport: Transport | None = None,
        formal_mode: bool = True,
        request_timeout_seconds: float | None = None,
    ) -> None:
        if not target.model_id.strip():
            raise ValueError("target.model_id must be non-empty")
        if not base_url.strip():
            raise ValueError("base_url must be non-empty")
        self.target = target
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"missing provider API key; set {api_key_env} or pass api_key")
        self.base_url = base_url.rstrip("/")
        self.organization = organization
        self.transport = transport or _urllib_transport
        self.formal_mode = formal_mode
        self.request_timeout_seconds = request_timeout_seconds
        self.validate_model_configuration(target.model_configuration, formal_mode=formal_mode)
        self._messages: list[dict[str, Any]] = []
        self._tool_name = "python"
        self._tool_spec: Mapping[str, Any] = {}
        self._model_turn_count = 0
        self._pending_tool_batch: ToolBatch | None = None
        self._closed = False

    @staticmethod
    def validate_model_configuration(
        configuration: Mapping[str, Any],
        *,
        formal_mode: bool = True,
    ) -> None:
        """Reject provider options that change the frozen capability boundary."""

        if not formal_mode:
            return
        unknown = sorted(set(configuration) - OPENAI_ALLOWED_MODEL_CONFIGURATION_KEYS)
        if unknown:
            raise ValueError(
                "formal provider configuration contains unknown or capability-changing "
                f"keys: {unknown}"
            )

    def start_case(self, request: ModelStartRequest, *, timeout_seconds: float) -> ModelResponse:
        self._ensure_open()
        if self.formal_mode and dict(request.python_tool_spec) != CANONICAL_PYTHON_TOOL_SPEC.to_dict():
            raise ModelAdapterError("formal provider request must use the canonical Python tool spec")
        self._messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_message},
        ]
        self._tool_name = str(request.python_tool_spec.get("name", "python"))
        self._tool_spec = json.loads(json.dumps(request.python_tool_spec, ensure_ascii=False))
        self._model_turn_count = 0
        self._pending_tool_batch = None
        return self._complete(request, timeout_seconds=timeout_seconds)

    def continue_case(self, event: PythonResultEvent, *, timeout_seconds: float) -> ModelResponse:
        self._ensure_open()
        batch = event.tool_batch or self._pending_tool_batch
        if batch is None:
            results = event.ordered_results
            batch = _legacy_batch_for_event(results, self._tool_name)
        else:
            if self._pending_tool_batch is not None and batch.calls != self._pending_tool_batch.calls:
                raise ToolBoundaryError("tool result batch identity does not match pending canonical batch")
            try:
                results = event.results_for_batch(batch)
            except ValueError as exc:
                raise ToolBoundaryError(str(exc)) from exc
        if len(batch.calls) != len(results):
            raise ToolBoundaryError("tool result count does not match canonical tool batch")
        result_payloads = _result_payloads(event, results)
        appended = []
        for call, payload in zip(batch.calls, result_payloads):
            message = {
                "role": "tool",
                "tool_call_id": call.provider_call_id,
                "name": self._tool_name,
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            }
            self._messages.append(message)
            appended.append(message)
        try:
            return self._complete(None, timeout_seconds=timeout_seconds)
        except ModelAdapterError:
            # A retryable transport/boundary failure must replay the exact same
            # request. Roll back the unacknowledged tool result so a retry does
            # not duplicate it in the provider conversation.
            del self._messages[-len(appended):]
            raise

    def close(self) -> None:
        self._closed = True
        self._messages.clear()
        self._pending_tool_batch = None

    def _complete(
        self,
        request: ModelStartRequest | None,
        *,
        timeout_seconds: float | None = None,
    ) -> ModelResponse:
        tool = request.python_tool_spec if request is not None else self._tool_spec
        payload: dict[str, Any] = {
            "model": self.target.model_id,
            "messages": json.loads(json.dumps(self._messages, ensure_ascii=False)),
        }
        # An empty tool specification is the explicit tool-free role mode.
        # Omit all tool-related request fields so the provider cannot return a
        # Python/tool call for evidence-conditioned advisory review.
        if tool:
            payload.update(
                {
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": tool.get("name", self._tool_name),
                                "description": tool.get("description", ""),
                                "parameters": tool.get("parameters", {}),
                            },
                        }
                    ],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
            )
        configuration = self.target.model_configuration
        for key in OPENAI_ALLOWED_MODEL_CONFIGURATION_KEYS:
            if key in configuration:
                value = configuration[key]
                payload[key] = value
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if configuration.get("stream") is True:
            headers["Accept"] = "text/event-stream"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        request_timeout = float(timeout_seconds if timeout_seconds is not None else 900.0)
        if self.request_timeout_seconds is not None:
            request_timeout = min(request_timeout, float(self.request_timeout_seconds))
        response_bytes = self.transport(
            self.base_url + "/chat/completions",
            headers,
            body,
            request_timeout,
        )
        try:
            response = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AmbiguousProviderError("provider returned invalid JSON") from exc
        if not isinstance(response, Mapping):
            raise AmbiguousProviderError("provider returned a non-object response")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise ModelAdapterError("provider response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise ModelAdapterError("provider response has no message")
        content = _message_text(message.get("content"))
        usage = response.get("usage")
        input_tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
        output_tokens = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
        for name, value in (("prompt_tokens", input_tokens), ("completion_tokens", output_tokens)):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ModelAdapterError(f"provider returned invalid {name}")
        raw_usage = dict(usage) if isinstance(usage, Mapping) else {}
        tool_calls = message.get("tool_calls")
        if tool_calls:
            if not isinstance(tool_calls, list) or not tool_calls:
                raise ToolBoundaryError("provider returned an invalid empty tool-call batch")
            response_id = response.get("id")
            if not isinstance(response_id, str) or not response_id:
                # Some OpenAI-compatible routes omit response IDs. Derive a
                # stable replay identity from the emitted tool-call payload,
                # not the mutable local turn counter.
                response_id = "chat-response-" + hashlib.sha256(
                    json.dumps(tool_calls, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()[:16]
            calls: list[ToolCall] = []
            for ordinal, call in enumerate(tool_calls):
                function = call.get("function") if isinstance(call, Mapping) else None
                if not isinstance(function, Mapping):
                    raise ModelAdapterError("provider requested an unknown tool")
                # The current gateway appends a deterministic ``__clove``
                # transport suffix to streamed tool names.  It is still the
                # exact Python tool declared in this request; accept only
                # that one alias and keep all other names fail-closed.
                function_name = function.get("name")
                if function_name == self._tool_name + "__clove":
                    function_name = self._tool_name
                if function_name != self._tool_name:
                    raise ModelAdapterError("provider requested an unknown tool")
                try:
                    arguments = json.loads(function.get("arguments", "{}"))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ModelAdapterError("provider returned invalid Python tool arguments") from exc
                if not isinstance(arguments, Mapping) or not isinstance(arguments.get("code"), str):
                    raise ModelAdapterError("Python tool arguments must contain string code")
                provider_call_id = str(call.get("id") or f"python-call-{ordinal}")
                calls.append(
                    _canonical_call(
                        provider_response_id=response_id,
                        provider_call_id=provider_call_id,
                        ordinal=ordinal,
                        tool_name=self._tool_name,
                        arguments=arguments,
                    )
                )
            batch = _batch(
                provider_response_id=response_id,
                model_turn_id=f"chat-turn-{self._model_turn_count + 1}",
                calls=calls,
            )
            self._messages.append(dict(message))
            self._pending_tool_batch = batch
            self._model_turn_count += 1
            return ModelResponse(
                action=batch,
                message=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_usage=raw_usage,
            )
        if not content:
            raise ModelAdapterError("provider returned neither a tool call nor final text")
        self._messages.append(dict(message))
        return ModelResponse(
            action=FinalAnswer(content),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_truncated=choices[0].get("finish_reason") in {"length", "max_tokens"},
            raw_usage=raw_usage,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ModelAdapterError("adapter is closed")


class ThirdPartyResponsesAdapter:
    """One fresh Responses-compatible third-party session for one case.

    ``base_url`` is mandatory and is supplied by runtime configuration.  The
    adapter does not encode or restrict provider hostnames; protocol
    compatibility is not provider identity.
    """

    def __init__(
        self,
        target: EvaluationTarget,
        *,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        transport: Transport | None = None,
        formal_mode: bool = True,
        request_timeout_seconds: float | None = None,
    ) -> None:
        if not target.model_id.strip():
            raise ValueError("target.model_id must be non-empty")
        if not base_url.strip():
            raise ValueError("base_url must be non-empty")
        normalized_base_url = base_url.rstrip("/")
        self.target = target
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(f"missing provider API key; set {api_key_env} or pass api_key")
        self.base_url = normalized_base_url
        self.transport = transport or _urllib_transport
        self.formal_mode = formal_mode
        self.request_timeout_seconds = request_timeout_seconds
        self.validate_model_configuration(target.model_configuration, formal_mode=formal_mode)
        self._tool_name = "python"
        self._tool_spec: Mapping[str, Any] = {}
        self._instructions = ""
        self._conversation_input: list[dict[str, Any]] = []
        self._previous_response_id: str | None = None
        self._pending_call_id: str | None = None
        self._pending_tool_batch: ToolBatch | None = None
        self._model_turn_count = 0
        self._closed = False

    @staticmethod
    def validate_model_configuration(
        configuration: Mapping[str, Any],
        *,
        formal_mode: bool = True,
    ) -> None:
        """Reject capability-changing options while permitting route identity keys."""

        if not formal_mode:
            return
        route_keys = {"execution_backend", "wire_api", "backend", "wire_protocol"}
        unknown = sorted(set(configuration) - RESPONSES_ALLOWED_MODEL_CONFIGURATION_KEYS - route_keys)
        if unknown:
            raise ValueError(
                "formal Responses configuration contains unknown or capability-changing "
                f"keys: {unknown}"
            )

    def start_case(self, request: ModelStartRequest, *, timeout_seconds: float) -> ModelResponse:
        self._ensure_open()
        if self._previous_response_id is not None:
            raise ModelAdapterError("Responses session has already started")
        if self.formal_mode and dict(request.python_tool_spec) != CANONICAL_PYTHON_TOOL_SPEC.to_dict():
            raise ModelAdapterError("formal provider request must use the canonical Python tool spec")
        self._tool_name = str(request.python_tool_spec.get("name", "python"))
        self._tool_spec = json.loads(json.dumps(request.python_tool_spec, ensure_ascii=False))
        self._instructions = request.system_prompt
        self._conversation_input = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": request.user_message}],
            }
        ]
        self._model_turn_count = 0
        self._pending_tool_batch = None
        payload = self._base_payload()
        payload["input"] = json.loads(json.dumps(self._conversation_input, ensure_ascii=False))
        return self._request(payload, timeout_seconds=timeout_seconds)

    def continue_case(self, event: PythonResultEvent, *, timeout_seconds: float) -> ModelResponse:
        self._ensure_open()
        if self._previous_response_id is None or self._pending_tool_batch is None:
            raise ModelAdapterError("Responses session has no pending Python tool call")
        batch = event.tool_batch or self._pending_tool_batch
        if self._pending_tool_batch is not None and batch.calls != self._pending_tool_batch.calls:
            raise ToolBoundaryError("tool result batch identity does not match pending canonical batch")
        try:
            results = event.results_for_batch(batch)
        except ValueError as exc:
            raise ToolBoundaryError(str(exc)) from exc
        if len(batch.calls) != len(results):
            raise ToolBoundaryError("tool result count does not match canonical tool batch")
        result_payloads = _result_payloads(event, results)
        tool_outputs = [
            {
                "type": "function_call_output",
                "call_id": call.provider_call_id,
                "output": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            }
            for call, payload in zip(batch.calls, result_payloads)
        ]
        previous_input = self._conversation_input
        previous_response_id = self._previous_response_id
        previous_batch = self._pending_tool_batch
        # _request commits the provider response items to the current input.
        # Use a candidate list so a transport or one-call-boundary failure can
        # be retried without duplicating the function_call_output.
        self._conversation_input = [*previous_input, *tool_outputs]
        payload = self._base_payload()
        payload["input"] = json.loads(json.dumps(self._conversation_input, ensure_ascii=False))
        try:
            return self._request(payload, timeout_seconds=timeout_seconds)
        except ModelAdapterError:
            self._conversation_input = previous_input
            self._previous_response_id = previous_response_id
            self._pending_tool_batch = previous_batch
            raise

    def close(self) -> None:
        self._closed = True
        self._tool_spec = {}
        self._instructions = ""
        self._conversation_input.clear()
        self._previous_response_id = None
        self._pending_call_id = None
        self._pending_tool_batch = None

    def _base_payload(self) -> dict[str, Any]:
        tool = self._tool_spec
        payload: dict[str, Any] = {
            "model": self.target.model_id,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        # Empty tool spec is an explicit no-tool request.  Keep the payload
        # free of tools/tool_choice so this role cannot access Python or web.
        if tool:
            payload.update(
                {
                    "tools": [
                        {
                            "type": "function",
                            "name": tool.get("name", self._tool_name),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        }
                    ],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
            )
        if self._instructions:
            payload["instructions"] = self._instructions
        configuration = self.target.model_configuration
        reasoning_effort = configuration.get("reasoning_effort")
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        for key in ("temperature", "top_p", "max_output_tokens", "seed"):
            if key in configuration:
                payload[key] = configuration[key]
        return payload

    def _request(self, payload: Mapping[str, Any], *, timeout_seconds: float) -> ModelResponse:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request_timeout = min(float(timeout_seconds), float(self.request_timeout_seconds)) if self.request_timeout_seconds is not None else float(timeout_seconds)
        response_bytes = self.transport(
            self.base_url + "/responses",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            body,
            request_timeout,
        )
        try:
            response = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AmbiguousProviderError("provider returned invalid JSON") from exc
        if not isinstance(response, Mapping):
            raise AmbiguousProviderError("provider returned a non-object response")
        if response.get("error") is not None:
            raise ModelAdapterError(f"provider returned an error response: {response['error']!r}")
        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise ModelAdapterError("Responses provider returned no response id")
        usage = response.get("usage")
        input_tokens = usage.get("input_tokens") if isinstance(usage, Mapping) else None
        output_tokens = usage.get("output_tokens") if isinstance(usage, Mapping) else None
        for name, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ModelAdapterError(f"provider returned invalid {name}")
        raw_usage = dict(usage) if isinstance(usage, Mapping) else {}

        output = response.get("output")
        if not isinstance(output, list):
            raise ModelAdapterError("Responses provider returned no output list")
        # Validate the canonical call batch before mutating conversation state so
        # a retry can safely replay the same request.
        text_parts: list[str] = []
        function_calls: list[Mapping[str, Any]] = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type == "function_call":
                function_calls.append(item)
            elif item_type == "message":
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if (
                            isinstance(part, Mapping)
                            and part.get("type") in {"output_text", "text"}
                            and isinstance(part.get("text"), str)
                        ):
                            text_parts.append(part["text"])
        content = "".join(text_parts)
        if function_calls:
            calls = []
            for ordinal, call in enumerate(function_calls):
                if call.get("name") != self._tool_name:
                    raise ModelAdapterError("provider requested an unknown tool")
                arguments_raw = call.get("arguments", "{}")
                try:
                    arguments = (
                        dict(arguments_raw)
                        if isinstance(arguments_raw, Mapping)
                        else json.loads(arguments_raw)
                    )
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ModelAdapterError("provider returned invalid Python tool arguments") from exc
                if not isinstance(arguments, Mapping) or not isinstance(arguments.get("code"), str):
                    raise ModelAdapterError("Python tool arguments must contain string code")
                call_id = call.get("call_id") or call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise ModelAdapterError("provider Python tool call has no call_id")
                calls.append(
                    _canonical_call(
                        provider_response_id=response_id,
                        provider_call_id=call_id,
                        ordinal=ordinal,
                        tool_name=self._tool_name,
                        arguments=arguments,
                    )
                )
            batch = _batch(
                provider_response_id=response_id,
                model_turn_id=f"responses-turn-{self._model_turn_count + 1}",
                calls=calls,
            )
            call_id = call.get("call_id") or call.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise ModelAdapterError("provider Python tool call has no call_id")
            self._previous_response_id = response_id
            self._conversation_input.extend(
                json.loads(json.dumps(item, ensure_ascii=False))
                for item in output
                if isinstance(item, Mapping)
            )
            self._pending_call_id = calls[0].provider_call_id
            self._pending_tool_batch = batch
            self._model_turn_count += 1
            return ModelResponse(
                action=batch,
                message=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_usage=raw_usage,
            )

        self._previous_response_id = response_id
        self._conversation_input.extend(
            json.loads(json.dumps(item, ensure_ascii=False))
            for item in output
            if isinstance(item, Mapping)
        )
        status = response.get("status")
        incomplete_details = response.get("incomplete_details")
        incomplete_reason = (
            incomplete_details.get("reason") if isinstance(incomplete_details, Mapping) else None
        )
        output_truncated = status == "incomplete" or incomplete_reason in {
            "max_output_tokens",
            "length",
        }
        if not content:
            output_types = [
                item.get("type")
                for item in output
                if isinstance(item, Mapping)
            ]
            raise ModelAdapterError(
                "provider returned neither a tool call nor final text; "
                f"status={status!r}, output_types={output_types!r}, "
                f"usage_present={isinstance(usage, Mapping)}"
            )
        return ModelResponse(
            action=FinalAnswer(content),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_truncated=output_truncated,
            raw_usage=raw_usage,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ModelAdapterError("adapter is closed")


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return ""


__all__ = [
    "OPENAI_ALLOWED_MODEL_CONFIGURATION_KEYS",
    "RESPONSES_ALLOWED_MODEL_CONFIGURATION_KEYS",
    "OpenAIChatCompletionsAdapter",
    "ThirdPartyResponsesAdapter",
]

"""Native provider thinking controls; preserve reasoning across Python turns."""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request

from flowintentbench.model_runner import (
    ModelAdapterError, ProviderQuotaError, ProviderRateLimitError,
    ProviderRequestTimeoutError, ProviderUnavailableError,
)
from flowintentbench.provider_retry import parse_retry_after
from scripts.third_party_benchmark_transport import network_configuration, network_opener

PROVIDERS = {
    "deepseek-flash": {
        "provider": "deepseek", "base_url": "https://api.deepseek.com",
        "key_field": "deepseek_API_KEY", "profile": "deepseek-v4.1-flash-max",
        "options": {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
        "response_models": ("deepseek-flash",),
    },
    "qwen3.8-max": {
        "provider": "qwen", "base_url": "https://ws-nptxbo6gj0b0h4ah.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "key_field": "qwen_API_KEY", "profile": "qwen3.8-max-xhigh",
        "options": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"},
        "response_models": ("qwen3.8-max", "qwen3.8-max-0902", "qwen3.8-max-2026-09-02"),
    },
}


def read_completion(response, deadline):
    """Assemble complete SSE messages, including fragmented tool arguments."""
    message = {"role": "assistant", "content": "", "reasoning_content": ""}
    calls, metadata, usage = {}, {}, None
    finish = None
    for raw in response:
        if time.monotonic() >= deadline:
            raise ProviderRequestTimeoutError("chat stream exceeded deadline")
        line = raw.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            raise ModelAdapterError("expected Chat Completions SSE response")
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("error"):
            raise ModelAdapterError("provider returned an error inside the stream")
        for key in ("id", "model", "system_fingerprint"):
            if event.get(key) is not None:
                metadata[key] = event[key]
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ModelAdapterError("unexpected multiple completion choices")
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            for key in ("content", "reasoning_content"):
                if isinstance(delta.get(key), str):
                    message[key] += delta[key]
            for call in delta.get("tool_calls") or []:
                current = calls.setdefault(call["index"], {"type": "function", "function": {"name": "", "arguments": ""}})
                if call.get("id"):
                    current["id"] = call["id"]
                for key in ("name", "arguments"):
                    fragment = (call.get("function") or {}).get(key)
                    if fragment:
                        current["function"][key] += fragment
    if finish is None:
        raise ProviderUnavailableError("chat stream ended without a terminal choice")
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return {**metadata, "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}


def chat_transport(url, headers, body, timeout):
    payload = json.loads(body)
    spec = PROVIDERS[payload["model"]]
    if url != spec["base_url"] + "/chat/completions":
        raise ModelAdapterError("provider endpoint does not match the configured model")
    payload.update(spec["options"], stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={**headers, "Accept": "text/event-stream"}, method="POST")
    started = time.monotonic()
    try:
        with network_opener(network_configuration())(request, timeout=timeout) as response:
            if "application/json" in response.headers.get("Content-Type", ""):
                value = json.load(response)
            else:
                value = read_completion(response, started + timeout)
        if value.get("model") not in spec["response_models"]:
            raise ModelAdapterError(f"unexpected response model: {value.get('model')}")
        if not value.get("choices") or not value["choices"][0].get("finish_reason"):
            raise ProviderUnavailableError("provider response has no terminal choice")
        return json.dumps(value).encode()
    except urllib.error.HTTPError as exc:
        # Persist only status/code, never raw headers, credentials, or bodies.
        code = "unknown"
        try:
            error = json.loads(exc.read()).get("error", {})
            code = str(error.get("code") or error.get("type") or "unknown")
        except (ValueError, AttributeError):
            pass
        detail = f"{spec['provider']} HTTP {exc.code}: {code}"
        if exc.code == 402 or code in {"insufficient_quota", "insufficient_balance", "Arrearage"}:
            raise ProviderQuotaError(detail) from None
        if exc.code == 429:
            raise ProviderRateLimitError(detail, retry_after_seconds=parse_retry_after(exc.headers.get("Retry-After"))) from None
        if exc.code == 408:
            raise ProviderRequestTimeoutError(detail) from None
        if exc.code >= 500:
            raise ProviderUnavailableError(detail) from None
        raise ModelAdapterError(detail) from None
    except (TimeoutError, urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise ProviderUnavailableError(f"provider connection failure: {type(exc).__name__}") from None
    except (UnicodeError, ValueError) as exc:
        raise ModelAdapterError(f"invalid provider response: {type(exc).__name__}") from None

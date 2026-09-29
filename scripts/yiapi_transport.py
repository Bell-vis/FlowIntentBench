"""Responses streaming transport shared by YiAPI reviewers and collectors."""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request

from flowintentbench.model_runner import (
    ModelAdapterError, ProviderRateLimitError, ProviderRequestTimeoutError,
    ProviderUnavailableError,
)
from flowintentbench.provider_retry import parse_retry_after
from scripts.third_party_benchmark_transport import network_configuration, network_opener


def read_response(response, *, deadline=None, require_completed=False, expected_model=None):
    """Consume SSE until response.completed; never accept partial text/tools.

    Some compatible gateways ignore stream=true and return ordinary JSON.
    Both representations must contain a complete Responses object.
    """
    data_lines = []
    json_lines = []
    mode = None
    response_id = None
    final_text = None
    stream_text_parts = []
    output_items = {}
    function_argument_parts = {}

    def event_index(event):
        index = event.get("output_index", event.get("index"))
        if index is not None:
            return index
        item_id = event.get("item_id")
        if item_id is not None:
            for key, item in output_items.items():
                if item.get("id") == item_id:
                    return key
            return item_id
        return None

    def remember_item(event, item=None):
        item = item if isinstance(item, dict) else event.get("item")
        if not isinstance(item, dict):
            return
        key = event_index(event)
        if key is None:
            key = item.get("id") or f"item-{len(output_items)}"
        existing = output_items.get(key, {})
        output_items[key] = {**existing, **item}

    def synthetic_text_response():
        text = final_text if isinstance(final_text, str) else "".join(stream_text_parts)
        if not text:
            return None
        return {
            "id": response_id or "yiapi-stream-text",
            "object": "response",
            "status": "completed",
            "output": [{"type": "message", "status": "completed", "content": [{"type": "output_text", "text": text}]}],
        }

    def completed_payload(payload):
        """Restore output items omitted by gateways in response.completed."""

        if not isinstance(payload, dict) or payload.get("output"):
            return payload
        reconstructed = list(output_items.values())
        if not reconstructed and stream_text_parts:
            text = "".join(stream_text_parts)
            reconstructed = [{
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }]
        if not reconstructed:
            return payload
        return {**payload, "output": reconstructed}

    def consume_event(raw_event):
        nonlocal response_id, final_text
        if raw_event == "[DONE]":
            return None
        event = json.loads(raw_event)
        kind = event.get("type")
        metadata = event.get("response")
        if expected_model is not None and isinstance(metadata, dict):
            returned_model = metadata.get("model")
            if returned_model is not None and returned_model != expected_model:
                raise ModelAdapterError(f"response model mismatch: expected {expected_model}, received {returned_model}")
        if isinstance(event.get("response"), dict) and event["response"].get("id"):
            response_id = event["response"]["id"]
        if kind == "response.completed":
            return validate_response(completed_payload(event["response"]))
        if kind == "response.output_item.added":
            remember_item(event)
            return None
        if kind == "response.output_item.done":
            remember_item(event)
            return None
        if kind == "response.function_call_arguments.delta":
            key = event_index(event)
            if key is None:
                key = event.get("item_id") or f"item-{len(output_items)}"
            function_argument_parts[key] = function_argument_parts.get(key, "") + str(event.get("delta", ""))
            item = output_items.setdefault(key, {"type": "function_call"})
            item["arguments"] = function_argument_parts[key]
            return None
        if kind == "response.function_call_arguments.done":
            key = event_index(event)
            if key is None:
                key = event.get("item_id") or f"item-{len(output_items)}"
            item = output_items.setdefault(key, {"type": "function_call"})
            item["arguments"] = str(event.get("arguments", function_argument_parts.get(key, "")))
            return None
        if kind == "response.output_text.delta" and isinstance(event.get("delta"), str):
            stream_text_parts.append(event["delta"])
            return None
        if kind == "response.output_text.done" and isinstance(event.get("text"), str):
            final_text = event["text"]
            return None if require_completed else synthetic_text_response()
        if kind in {"response.failed", "response.incomplete", "error"}:
            error = event.get('error') or (metadata.get('error') if isinstance(metadata, dict) else None) or {}
            code = error.get('code') if isinstance(error, dict) else None
            message = error.get('message') if isinstance(error, dict) else str(error)
            if expected_model is not None and code == 'rate_limit_exceeded':
                raise ProviderRateLimitError(f"Responses rate limit: {message}", retry_after_seconds=30)
            raise ProviderUnavailableError(f"Responses terminal event: {kind}; {code}: {message}")
        return None

    for line in response:
        if deadline is not None and time.monotonic() > deadline:
            completed = synthetic_text_response()
            if completed is not None and not require_completed:
                return completed
            raise TimeoutError("Responses stream exceeded request deadline")
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        # Some gateways send SSE keep-alives before choosing a JSON response
        # (including JSON errors with HTTP 200). Comments are not response data
        # and must not lock the parser into SSE or hide the provider error.
        if mode is None and (not line.strip() or line.lstrip().startswith(':')):
            continue
        if mode is None and line.strip():
            mode = "json" if line.lstrip().startswith("{") else "sse"
        if mode == "json":
            json_lines.append(line)
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
        if not line.strip() and data_lines:
            event = "\n".join(data_lines)
            data_lines.clear()
            completed = consume_event(event)
            if completed is not None:
                return completed
    if data_lines:
        completed = consume_event("\n".join(data_lines))
        if completed is not None:
            return completed
    if mode == "json":
        value = validate_response(json.loads("".join(json_lines)))
        if expected_model is not None and value.get("model") != expected_model:
            raise ModelAdapterError(f"response model mismatch: expected {expected_model}, received {value.get('model')}")
        return value
    raise ProviderUnavailableError("Responses stream ended before response.completed")


def validate_response(payload):
    if isinstance(payload, dict) and payload.get('error'):
        error = payload['error']
        code = (error.get('code') or error.get('type')) if isinstance(error, dict) else None
        message = error.get('message') if isinstance(error, dict) else str(error)
        if code == 'rate_limit_exceeded':
            raise ProviderRateLimitError(f'Responses rate limit: {message}', retry_after_seconds=30)
        raise ProviderUnavailableError(f'Responses provider error: {code}: {message}')
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise ProviderUnavailableError("Responses payload is not completed")
    if payload.get("error") or not isinstance(payload.get("output"), list):
        raise ModelAdapterError("Responses payload has an error or missing output")
    return payload


def responses_transport(url, headers, body, timeout):
    """Explicit project proxy policy, streaming, and typed provider failures."""
    payload = json.loads(body)
    payload["stream"] = True
    headers = {**headers, "Accept": "text/event-stream"}
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers, method="POST"
    )
    started = time.monotonic()
    try:
        with network_opener(network_configuration())(request, timeout=timeout) as response:
            value = read_response(response, deadline=started + timeout)
            return json.dumps(value, ensure_ascii=False).encode()
    except urllib.error.HTTPError as exc:
        detail = exc.read(2000).decode("utf-8", "replace")
        if exc.code == 429:
            raise ProviderRateLimitError(
                "YiAPI HTTP 429", retry_after_seconds=parse_retry_after(exc.headers.get("Retry-After"))
            ) from exc
        if exc.code == 408:
            raise ProviderRequestTimeoutError("YiAPI HTTP 408") from exc
        if exc.code >= 500:
            raise ProviderUnavailableError(f"YiAPI HTTP {exc.code}: {detail}") from exc
        raise ModelAdapterError(f"YiAPI HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise ProviderUnavailableError(f"YiAPI connection error: {type(exc).__name__}: {exc}") from exc


def validate_review(value, schema):
    # Reuse the repository's schema validator; JSON mode is never a bypass
    # of required fields, enums, finite numbers, or evidence structure.
    from scripts.run_codex_file_judgments import validate_schema
    validate_schema(value, schema)
    return value

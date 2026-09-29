"""Third-party transport for the existing staged collector and file reviewer.

Uses the project's Chat Completions adapter and journaled Python tool. It does
not claim equivalence to Codex's agent harness. No credentials enter artifacts.
"""
from __future__ import annotations

from email.utils import parsedate_to_datetime
from contextlib import nullcontext
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid

from flowintentbench.model_runner import (
    EvaluationTarget, FinalAnswer, ModelStartRequest, PythonResultEvent, PythonToolSpec, ToolBatch,
)
from flowintentbench.providers import OpenAIChatCompletionsAdapter
from flowintentbench.python_runtime import PythonExecutionResult
from flowintentbench.runtime_config import resolve_api_key, resolve_provider_configuration
from scripts.collect_subagent_runs import write, read, file_hash, PYTHON, HELPER


# The solver writes the durable answer file before its final chat acknowledgement.
# Once that file exists, asking for another model turn only repeats the same
# completion and resends the entire conversation history. The evaluator still
# validates the file and execution journal after transport completion.
ANSWER_FILE = 'answer.md'


class APITransportFailure(RuntimeError):
    def __init__(self, kind, *, ambiguous=False):
        self.kind, self.ambiguous = kind, ambiguous
        super().__init__(kind)


def retry_after(value, now=None):
    """Honor both Retry-After forms. Never truncate a server-requested delay."""
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - (time.time() if now is None else now))
        except (ValueError, TypeError, OverflowError):
            return 0.0


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise APITransportFailure('redirect_rejected')


NETWORK_CONFIG = Path(__file__).resolve().parents[1] / 'config/benchmark_network.toml'


def network_configuration(path=NETWORK_CONFIG):
    with Path(path).open('rb') as stream:
        config = tomllib.load(stream)
    mode = config.get('mode')
    if mode == 'direct':
        return {'mode': 'direct', 'proxy_url': None, 'environment_proxy_inheritance': False}
    proxy = config.get('proxy_url', '')
    parsed = urllib.parse.urlsplit(proxy)
    if (mode != 'proxy' or parsed.scheme != 'http' or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ('', '/')):
        raise ValueError('Network config requires direct mode or an explicit HTTP proxy without credentials')
    return {'mode': 'proxy', 'proxy_url': proxy.rstrip('/'), 'environment_proxy_inheritance': False}


class ExplicitProxyHandler(urllib.request.ProxyHandler):
    """Honor the recorded proxy even when a terminal exports NO_PROXY=*.

    Only credential-free HTTP proxies are accepted; HTTPS targets use CONNECT.
    No global environment is changed and there is no fallback to direct access.
    """
    def proxy_open(self, request, proxy, protocol):
        parsed = urllib.parse.urlsplit(proxy)
        request.set_proxy(parsed.netloc, parsed.scheme)
        return None


def network_opener(config):
    proxies = {} if config['mode'] == 'direct' else dict.fromkeys(('http', 'https'), config['proxy_url'])
    return urllib.request.build_opener(ExplicitProxyHandler(proxies), NoRedirect()).open


def response_body_kind(raw, content_type=''):
    text = raw.decode('utf-8', errors='replace').lstrip().lower()
    html = 'text/html' in content_type.lower() or text.startswith(('<!doctype html', '<html'))
    if html and 'only available in certain regions' in text:
        return 'region_restricted'
    return 'unexpected_html' if html else None


def _read_chat_completion_response(response, *, streaming=False, deadline=None):
    """Read a Chat Completions response, normalizing SSE to one JSON object.

    The gateway closes some long non-streaming responses after roughly one
    minute.  Streaming keeps the connection active while GPT-6 emits tokens;
    the evaluator still receives the same ordinary Chat Completions shape.
    Solver/tool calls stay on the original non-streaming path.
    """
    if not streaming:
        return response.read()
    # Preserve arbitrary chunk boundaries, but stop at the SSE terminator:
    # a gateway may keep the HTTP connection alive after the response is done.
    chunks = []
    line_buffer = b""
    for chunk in response:
        if deadline is not None and time.monotonic() >= deadline:
            raise APITransportFailure("stream_deadline_exceeded", ambiguous=True)
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        chunks.append(chunk)
        line_buffer += chunk
        done = False
        while b"\n" in line_buffer:
            line, line_buffer = line_buffer.split(b"\n", 1)
            if line.strip() in (b"data: [DONE]", b"data:[DONE]"):
                done = True
                break
        if done:
            break
    raw_body = b"".join(chunks)
    if raw_body.lstrip().startswith(b"{"):
        try:
            json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise APITransportFailure("invalid_stream_response", ambiguous=True)
        return raw_body
    content_parts = []
    tool_calls = {}
    response_id = response_model = system_fingerprint = None
    usage = None
    finish_reason = None
    saw_sse = False
    saw_done = False
    for line in raw_body.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.lower().startswith(b"data:"):
            # Ignore SSE comments/keep-alives, but retain unexpected plain
            # JSON so a compatible gateway can still be diagnosed cleanly.
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            saw_sse = True
            saw_done = True
            break
        try:
            event = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise APITransportFailure("invalid_stream_response", ambiguous=True)
        if not isinstance(event, dict):
            raise APITransportFailure("invalid_stream_response", ambiguous=True)
        saw_sse = True
        response_id = event.get("id", response_id)
        response_model = event.get("model", response_model)
        system_fingerprint = event.get("system_fingerprint", system_fingerprint)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        finish_reason = choice.get("finish_reason", finish_reason)
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        value = delta.get("content")
        if isinstance(value, str):
            content_parts.append(value)
        # Preserve streamed tool calls for future reviewer contracts.  The
        # evaluator currently requests text-only JSON, but dropping these
        # fields would silently corrupt a compatible response.
        for call in delta.get("tool_calls", ()) or ():
            if not isinstance(call, dict):
                continue
            index = call.get("index", len(tool_calls))
            current = tool_calls.setdefault(index, {"index": index})
            for key in ("id", "type"):
                if key in call:
                    current[key] = call[key]
            function = call.get("function")
            if isinstance(function, dict):
                target = current.setdefault("function", {})
                for key in ("name", "arguments"):
                    if isinstance(function.get(key), str):
                        target[key] = target.get(key, "") + function[key] if key == "arguments" else function[key]
    # A connection that ends after only the initial role event is an
    # incomplete provider outcome, even if HTTP returned 200.  Preserve it as
    # ambiguous so the scheduler can authorize a fresh request identity.
    # ``[DONE]`` is optional in several OpenAI-compatible proxies.  A complete
    # final choice with a finish_reason is sufficient to establish the model's
    # outcome; requiring the sentinel turned successful responses into
    # ambiguous retries.
    if not saw_sse or finish_reason is None:
        raise APITransportFailure("invalid_stream_response", ambiguous=True)
    choice = {"index": 0, "message": {"role": "assistant", "content": "".join(content_parts)},
              "finish_reason": finish_reason}
    if tool_calls:
        choice["message"]["tool_calls"] = [tool_calls[k] for k in sorted(tool_calls)]
    result = {"id": response_id, "object": "chat.completion", "model": response_model,
              "choices": [choice]}
    if system_fingerprint is not None:
        result["system_fingerprint"] = system_fingerprint
    if usage is not None:
        result["usage"] = usage
    return json.dumps(result, ensure_ascii=False).encode("utf-8")


def api_visible_prompt(prompt):
    """Translate only the CLI tool invocation paragraph; retain all case inputs."""
    start=prompt.find('All data analysis\nmust run through the supplied journal helper:')
    end=prompt.find('The helper records exact source, stdout, stderr, elapsed time, and failures.',start)
    if start < 0 or end < 0:
        return prompt
    return (prompt[:start] + 'All data analysis must use the provided python tool directly. '
            'Put the analysis Python code itself in the tool code argument. The host automatically '
            'runs that code through the supplied journal helper using the installed project Python. '
            'Do not start run_python.py yourself: doing so would acquire the same execution lock twice. '
            'Each tool call starts a fresh interpreter; persist intermediates only in your work directory.\n'
            + prompt[end:])


def nested_helper_reference(code):
    try:
        tree=ast.parse(code)
    except SyntaxError:
        return False  # Preserve normal Python syntax-error evidence.
    return any(isinstance(node,ast.Constant) and isinstance(node.value,str)
               and 'run_python.py' in node.value for node in ast.walk(tree))


class APIGovernor:
    """Bounded HTTP concurrency with endpoint-wide pacing and cooldown."""
    def __init__(self, status_path, *, min_interval=3.0, attempts=4, opener=None, network=None,
                 max_concurrency=1):
        if not 1 <= max_concurrency <= 8:
            raise ValueError('API HTTP concurrency must be between 1 and 8')
        self.status_path = Path(status_path)
        self.min_interval, self.attempts = min_interval, attempts
        self.max_concurrency = max_concurrency
        self.lock = threading.BoundedSemaphore(max_concurrency)
        self.state_lock = threading.RLock()
        self.health_lock = threading.Lock()
        self.next_request = 0.0
        self.wait_reason = 'pacing'
        self.disabled = False
        self.disabled_models = set()
        self.http_attempts = 0
        self.network = network or network_configuration()
        self.opener = opener or network_opener(self.network)
        self.health_url = None
        self.health_key = None
        self.health_ready = False
        self.health_http_attempts = 0
        self.failure_reason = None
        self.failure_revision = 0
        self.model_failures = {}
        self.shared_gate = None
        if self.status_path.exists():
            prior = read(self.status_path)
            self.next_request = float(prior.get('next_request_epoch', 0))
            self.wait_reason = prior.get('wait_reason', 'pacing')
            self.disabled = bool(prior.get('disabled', False))
            self.disabled_models.update(prior.get('disabled_models', []))
            self.http_attempts = int(prior.get('http_attempts', 0))
            self.health_http_attempts = int(prior.get('health_http_attempts', 0))
            self.failure_reason = prior.get('failure_reason')
            self.model_failures = dict(prior.get('model_failures', {}))

    def available(self, model=None):
        return (not self.disabled and model not in self.disabled_models and time.time() >= self.next_request
                and (self.health_url is None or self.health_ready)
                and (model not in self.model_failures if model is not None else not self.model_failures)
                and (self.shared_gate is None or self.shared_gate.ready()))

    def status(self, **extra):
        with self.state_lock:
            write(self.status_path, dict(updated_epoch=time.time(), http_attempts=self.http_attempts,
                  disabled=self.disabled, disabled_models=sorted(self.disabled_models),
                  network=self.network, health_ready=self.health_ready,
                  max_http_concurrency=self.max_concurrency,
                  shared_http_admission=(dict(limit=self.shared_gate.limit, directory=str(self.shared_gate.directory))
                                         if self.shared_gate is not None else None),
                  health_http_attempts=self.health_http_attempts, failure_reason=self.failure_reason,
                  model_failures=dict(self.model_failures),
                  next_request_epoch=self.next_request, wait_reason=self.wait_reason, **extra))

    def cooldown(self, seconds, reason):
        # A successful concurrent call must never shorten another call's 429
        # Retry-After or outage cooldown.
        with self.state_lock:
            until = time.time() + seconds
            if until >= self.next_request:
                self.next_request, self.wait_reason = until, reason

    def fail_model(self, model, reason):
        with self.state_lock:
            self.model_failures[model] = dict(reason=reason, retry_after_epoch=time.time()+120,
                                             failed_epoch=time.time())

    def admit_request(self, deadline, telemetry, model=None, *, allow_model_continuation=False):
        while True:
            with self.state_lock:
                if self.disabled:
                    raise APITransportFailure('api_circuit_open')
                if self.health_url is not None and not self.health_ready:
                    raise APITransportFailure('api_preflight_required')
                if model in self.model_failures and not allow_model_continuation:
                    raise APITransportFailure('api_model_preflight_required')
                delay = max(0.0, self.next_request - time.time())
                reason = self.wait_reason
                if delay >= deadline - time.monotonic():
                    raise APITransportFailure(reason + '_wait_exceeds_budget')
                if not delay:
                    self.next_request = time.time() + self.min_interval
                    self.wait_reason = 'pacing'
                    return
            waited = time.monotonic()
            time.sleep(min(delay, 1.0))
            telemetry[reason + '_wait_seconds'] += time.monotonic() - waited

    def configure_healthcheck(self, base_url, key):
        self.health_url = base_url.rstrip('/') + '/models'
        self.health_key = key
        self.health_ready = False

    def check_health(self, *, startup=False, timeout=15):
        """A read-only probe before trial admission, never a replacement answer.

        Startup rechecks a persistent circuit once. During a run, region/auth
        restrictions remain disabled; transient errors require a healthy probe.
        Explicit Retry-After is honored even across restart.
        """
        if self.health_url is None:
            return True
        with self.state_lock:
            due_models = {model: dict(failure) for model, failure in self.model_failures.items()
                          if time.time() >= failure['retry_after_epoch']}
        if self.health_ready and not startup and not due_models:
            return True
        if ((self.disabled and not startup) or
                (time.time() < self.next_request and (not startup or self.wait_reason == 'rate_limit'))):
            return False
        if not self.health_lock.acquire(blocking=False):
            return False
        if not self.lock.acquire(blocking=False):
            self.health_lock.release()
            return False
        try:
            # Another reviewer may have checked health while we waited.
            if self.health_ready and not startup and not due_models:
                return True
            folder = self.status_path.parent / 'api_preflight' / uuid.uuid4().hex
            folder.mkdir(parents=True)
            with self.state_lock:
                failure_revision = self.failure_revision
            receipt = dict(endpoint=self.health_url, method='GET', network=self.network,
                           started_epoch=time.time(), generation_requested=False)
            write(folder / 'receipt.json', dict(receipt, status='STARTED'))
            self.health_http_attempts += 1
            try:
                request = urllib.request.Request(self.health_url,
                    headers={'Authorization': 'Bearer ' + self.health_key, 'Accept': 'application/json'})
                try:
                    response = self.opener(request, timeout=timeout)
                except urllib.error.HTTPError as exc:
                    response = exc
                with response:
                    raw = response.read()
                    headers = getattr(response, 'headers', {})
                    code = getattr(response, 'status', 200)
                    content_type = headers.get('Content-Type', '')
                    receipt.update(http_status=code, content_type=content_type,
                                   response_sha256=hashlib.sha256(raw).hexdigest())
                (folder / 'response.body').write_bytes(raw)
                kind = response_body_kind(raw, content_type)
                if kind:
                    self.disabled = kind == 'region_restricted'
                    raise APITransportFailure(kind)
                if code != 200:
                    self.disabled = code in (401, 403, 404)
                    if code == 429:
                        self.next_request = time.time() + max(120, retry_after(headers.get('Retry-After')))
                        self.wait_reason = 'rate_limit'
                        if self.shared_gate is not None:
                            self.shared_gate.cooldown(max(120, retry_after(headers.get('Retry-After'))))
                    raise APITransportFailure('preflight_http_' + str(code))
                value = json.loads(raw)
                if not isinstance(value, dict) or not isinstance(value.get('data'), list):
                    raise APITransportFailure('invalid_model_catalog')
                ids = {item.get('id') for item in value['data'] if isinstance(item, dict)}
                if not ids:
                    raise APITransportFailure('empty_model_catalog')
                receipt.update(status='HEALTHY', catalog_model_ids=sorted(i for i in ids if isinstance(i,str)))
                with self.state_lock:
                    if self.failure_revision != failure_revision:
                        raise APITransportFailure('preflight_superseded_by_concurrent_failure')
                    self.disabled = False
                    self.health_ready = True
                    self.failure_reason = None
                    for model, previous in due_models.items():
                        if self.model_failures.get(model) == previous:
                            del self.model_failures[model]
                    self.cooldown(self.min_interval, 'pacing')
            except Exception as exc:
                if isinstance(exc, APITransportFailure):
                    kind = exc.kind
                elif isinstance(exc, urllib.error.URLError) and isinstance(getattr(exc, 'reason', None), OSError):
                    # A restricted execution namespace commonly reports an
                    # inaccessible host-local proxy as EPERM.  Preserve this
                    # distinction so callers can fail fast instead of
                    # treating it as DNS/TLS instability.
                    reason = exc.reason
                    kind = 'network_namespace_blocked' if getattr(reason, 'errno', None) == 1 else type(exc).__name__
                else:
                    kind = type(exc).__name__
                self.health_ready = False
                self.failure_reason = kind
                # Keep the normalized category for scheduling, but also retain
                # the underlying reason so URLError can be distinguished from
                # DNS, proxy, TLS, reset, and timeout failures.  No secrets or
                # request bodies are included here.
                detail = getattr(exc, 'reason', None) or str(exc)
                receipt.update(status='UNAVAILABLE', failure_reason=kind,
                               failure_detail=str(detail)[:500])
                if self.wait_reason != 'rate_limit' or self.next_request <= time.time():
                    self.cooldown(120, 'circuit')
            finally:
                receipt['finished_epoch'] = time.time()
                write(folder / 'receipt.json', receipt)
                self.status(last_preflight_path=str(folder / 'receipt.json'), last_status=receipt['status'])
            return self.health_ready
        finally:
            self.lock.release()
            self.health_lock.release()

    def transport(self, audit_dir, model):
        audit_dir = Path(audit_dir)
        telemetry = dict(client_queue_wait_seconds=0.0, rate_limit_wait_seconds=0.0,
                         pacing_wait_seconds=0.0, circuit_wait_seconds=0.0,
                         failed_request_seconds=0.0, successful_request_seconds=0.0,
                         http_failures=0)
        accepted_requests = 0
        def send(url, headers, body, timeout):
            nonlocal accepted_requests
            deadline = time.monotonic() + timeout
            queued = time.monotonic()
            if not self.lock.acquire(timeout=max(0.001, timeout)):
                telemetry['client_queue_wait_seconds'] += time.monotonic()-queued
                raise APITransportFailure('request_lane_budget_exhausted')
            telemetry['client_queue_wait_seconds'] += time.monotonic()-queued
            try:
                for attempt in range(self.attempts):
                    if self.disabled or model in self.disabled_models:
                        raise APITransportFailure('api_circuit_open')
                    if self.health_url is not None and not self.health_ready:
                        raise APITransportFailure('api_preflight_required')
                    # One lost conversation closes admission for NEW tasks,
                    # not healthy same-model conversations already in flight.
                    # Global auth, 429/shared cooldown and deadlines still apply.
                    self.admit_request(deadline, telemetry, model,
                                       allow_model_continuation=accepted_requests > 0)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise APITransportFailure('request_budget_exhausted')
                    shared = self.shared_gate.admit(deadline) if self.shared_gate is not None else nullcontext()
                    admission_started = time.monotonic()
                    try:
                        shared.__enter__()
                    except TimeoutError:
                        raise APITransportFailure('shared_admission_budget_exhausted', ambiguous=False) from None
                    finally:
                        telemetry['client_queue_wait_seconds'] += time.monotonic() - admission_started
                    try:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise APITransportFailure('shared_admission_budget_exhausted', ambiguous=False)
                        folder = audit_dir / ('request_' + uuid.uuid4().hex)
                        folder.mkdir(parents=True)
                        payload = json.loads(body)
                        write(folder / 'request.json', payload)
                        receipt = dict(model=model, endpoint=url, request_sha256=hashlib.sha256(body).hexdigest(),
                                       started_epoch=time.time(), status='STARTED', attempt=attempt + 1,
                                       active_conversation_continuation=accepted_requests > 0)
                        write(folder / 'http_receipt.json', receipt)
                        with self.state_lock:
                            self.http_attempts += 1
                        self.status(last_model=model, last_status='REQUEST_STARTED')
                        try:
                            request = urllib.request.Request(url, data=body, headers=headers, method='POST')
                            with self.opener(request, timeout=remaining) as response:
                                content_type = getattr(response, 'headers', {}).get('Content-Type', '')
                                # A few OpenAI-compatible test/deployment
                                # routes ignore ``stream`` and return JSON;
                                # accept that response without attempting to
                                # parse it as SSE.
                                # Parse according to the requested protocol,
                                # not a gateway's MIME label.  Some proxies
                                # send SSE with application/json; the parser
                                # also accepts a normal JSON fallback.
                                streaming = bool(payload.get('stream') is True)
                                raw = _read_chat_completion_response(response, streaming=streaming,
                                                                     deadline=deadline)
                                receipt.update(http_status=getattr(response, 'status', 200),
                                               content_type=content_type,
                                               streaming=streaming)
                            # Save exact body before parsing: ambiguous failures must be inspectable.
                            (folder / 'response.json').write_bytes(raw)
                            receipt['response_sha256'] = hashlib.sha256(raw).hexdigest()
                            kind = response_body_kind(raw, receipt['content_type'])
                            if kind:
                                receipt.update(status='SERVICE_UNAVAILABLE', failure_kind=kind)
                                self.disabled = kind == 'region_restricted'
                                self.cooldown(120, 'circuit')
                                raise APITransportFailure(kind, ambiguous=kind != 'region_restricted')
                            result = json.loads(raw)
                            returned = result.get('model')
                            receipt.update(status='RECEIVED', response_sha256=hashlib.sha256(raw).hexdigest(),
                                           response_id=result.get('id'), response_model=returned,
                                           system_fingerprint=result.get('system_fingerprint'), usage=result.get('usage'))
                            if returned != model:
                                self.disabled_models.add(model)
                                raise APITransportFailure('response_model_mismatch', ambiguous=True)
                            self.cooldown(self.min_interval, 'pacing')
                            receipt['status'] = 'ACCEPTED'
                            accepted_requests += 1
                            return raw
                        except urllib.error.HTTPError as exc:
                            receipt.update(status='HTTP_ERROR', http_status=exc.code)
                            if exc.code == 429:
                                wait = max(retry_after(exc.headers.get('Retry-After')),
                                           min(120.0, 10.0 * 2 ** attempt) + random.uniform(0, 3))
                                self.cooldown(wait, 'rate_limit')
                                if self.shared_gate is not None:
                                    self.shared_gate.cooldown(wait)
                                if attempt + 1 == self.attempts:
                                    raise APITransportFailure('rate_limit_retries_exhausted') from None
                                continue  # Same immutable request; server explicitly rejected it.
                            if exc.code in (401, 403):
                                self.disabled = True
                            elif exc.code in (400, 404, 422):
                                self.disabled_models.add(model)
                            else:
                                if self.max_concurrency > 1 and exc.code >= 500:
                                    self.fail_model(model, 'http_' + str(exc.code))
                                else:
                                    self.cooldown(120, 'circuit')
                            # A third-party 5xx may hide a completed upstream generation.
                            raise APITransportFailure('http_' + str(exc.code), ambiguous=exc.code >= 500) from None
                        except APITransportFailure as exc:
                            self.health_ready = False
                            self.failure_reason = exc.kind
                            raise
                        except Exception as exc:
                            receipt.update(status='UNKNOWN_OUTCOME', error_type=type(exc).__name__)
                            if self.max_concurrency > 1:
                                # A dropped Luna response is not evidence that an
                                # independently successful Terra session is broken.
                                self.fail_model(model, 'response_outcome_unknown')
                            else:
                                self.cooldown(120, 'circuit')
                                self.health_ready = False
                                self.failure_reason = 'response_outcome_unknown'
                            raise APITransportFailure('response_outcome_unknown', ambiguous=True) from None
                        finally:
                            receipt['finished_epoch'] = time.time()
                            elapsed = max(0.0,receipt['finished_epoch']-receipt['started_epoch'])
                            telemetry['successful_request_seconds' if receipt['status']=='ACCEPTED' else 'failed_request_seconds'] += elapsed
                            if receipt['status']!='ACCEPTED':
                                telemetry['http_failures'] += 1
                                with self.state_lock:
                                    model_local = (self.max_concurrency > 1 and model in self.model_failures
                                                   and (receipt['status'] == 'UNKNOWN_OUTCOME'
                                                        or (receipt.get('http_status') or 0) >= 500))
                                    if not model_local:
                                        self.failure_revision += 1
                                    if not model_local and not (receipt.get('http_status') == 429 and attempt + 1 < self.attempts):
                                        self.health_ready = False
                                        self.failure_reason = receipt.get('failure_kind') or receipt.get('error_type') or 'http_' + str(receipt.get('http_status'))
                            write(folder / 'http_receipt.json', receipt)
                            self.status(last_model=model, last_status=receipt['status'])
                    finally:
                        shared.__exit__(None, None, None)
            finally:
                self.lock.release()
        send.telemetry = telemetry
        return send


def compact_output_schema(schema):
    """Factor identical batch item schemas without weakening any constraint."""
    result = json.loads(json.dumps(schema))
    properties = result.get('properties', {})
    if len(properties) > 1:
        first = next(iter(properties.values()))
        if all(value == first for value in properties.values()):
            definitions = result.setdefault('$defs', {})
            key = '_shared_output_item'
            while key in definitions:
                key += '_'
            definitions[key] = first
            result['properties'] = {name: {'$ref': '#/$defs/' + key} for name in properties}
    return result


class ThirdPartyAgent:
    """Adapter-shaped runner; one fresh conversation, same local journal helper."""
    transport_name = 'third_party_chat_completions'

    def __init__(self, config_path, governor, *, structured_only=False):
        self.config_path, self.governor = Path(config_path), governor
        self.structured_only = structured_only
        self.runtime = resolve_provider_configuration(config_path=self.config_path)
        self.key = resolve_api_key(self.runtime)

    def __call__(self, prompt, model, workdir, output_dir, timeout_seconds=900, output_schema=None,
                 *, reasoning_effort='xhigh'):
        work, audit = Path(workdir), Path(output_dir)
        work.mkdir(parents=True, exist_ok=True)
        audit.mkdir(parents=True, exist_ok=True)
        started = time.time()
        deadline = time.monotonic() + timeout_seconds
        identity = uuid.uuid4().hex
        # A reviewer format-repair directory also contains execution_budget.
        # The explicit structured-output contract distinguishes that role.
        is_solver = output_schema is None and (work / 'execution_budget.json').exists()
        structured_only = getattr(self, 'structured_only', False) and not is_solver
        if is_solver and reasoning_effort != 'xhigh':
            raise ValueError('Solver effort is frozen at xhigh')
        if not is_solver and not structured_only:
            shutil.copyfile(HELPER, work / 'run_python.py')
            write(work / 'execution_budget.json', dict(started_epoch=started,
                  deadline_epoch=started + timeout_seconds, max_python_executions=60))
        budget = {} if structured_only else json.loads((work / 'execution_budget.json').read_text())
        max_python_executions = int(budget.get('max_python_executions', 60))
        budget_deadline = float(budget.get('deadline_epoch', started + timeout_seconds))
        deadline_seconds = int(max(1, budget_deadline - started))
        # This changes only the advertised transport capability, not the task or scientific inputs.
        instruction = ('You have one local python tool. Each call executes your code through the supplied '
                       'journal helper in a fresh interpreter. Do not invoke run_python.py from your code '
                       'and do not edit the helper, budget, or journal. Write files using Python. '
                       'The tool already uses the specified project Python. Read only the assigned '
                       'task and data paths. Do not use network, APIs, other agents, repository modules, '
                       'or other answers. Read the case once, batch related computations, reuse saved '
                       'intermediate results, keep stdout compact, and do not repeat identical tool calls. '
                       f'Finish within {deadline_seconds} seconds and at most {max_python_executions} Python calls. '
                       'For a solver task write answer.md as requested, then finish. For a reviewer '
                       'task return the requested JSON in your final message.')
        if structured_only:
            from scripts.grading_policy import STRUCTURED_INSTRUCTIONS
            instruction = STRUCTURED_INSTRUCTIONS
        if output_schema is not None:
            # API transport has no CLI --output-schema. Supply the actual
            # wrapper contract, not only the inner scientific response schema.
            # Local schema + typed/evidence validation remains authoritative.
            instruction += (' Final JSON must satisfy this output schema exactly. '
                            'For review envelopes, outer status is RESOLVED or PENDING; '
                            'ACCEPTED/REJECTED are scientific verdicts inside response, '
                            'never outer envelope status. '
                            + ('Use evidence_files: []. ' if structured_only else
                               'Only cite evidence files actually created and checked with the Python tool; '
                               'if supplied evidence suffices, use evidence_files: []. ')
                            + 'Schema: '
                            + json.dumps(compact_output_schema(output_schema), separators=(',', ':')))
        effective_prompt=api_visible_prompt(prompt)
        write(audit / 'invocation.json', dict(transport=self.transport_name,
              provider=self.runtime.provider, endpoint=self.runtime.base_url,
              requested_model=model, reasoning_effort=reasoning_effort, timeout_seconds=timeout_seconds,
              prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
              effective_prompt_sha256=hashlib.sha256(effective_prompt.encode()).hexdigest(),
              transport_instruction=instruction, config=self.runtime.without_secrets(),
              network=self.governor.network,
              source_sha256=file_hash(Path(__file__)), cli_equivalent=False))
        (audit / 'prompt.txt').write_text(prompt)
        (audit / 'api_prompt.txt').write_text(effective_prompt)
        # Reviewer generations use SSE so the gateway cannot terminate a
        # long xhigh JSON response while the connection is otherwise quiet.
        # Solver sessions retain the original non-streaming tool protocol.
        model_configuration = {'reasoning_effort': reasoning_effort}
        if not is_solver:
            model_configuration['stream'] = True
        target = EvaluationTarget(self.runtime.provider, model, model_configuration=model_configuration)
        http_transport = self.governor.transport(audit / 'http', model)
        adapter = OpenAIChatCompletionsAdapter(target, api_key=self.key, base_url=self.runtime.base_url,
                   formal_mode=False, transport=http_transport)
        spec = PythonToolSpec(description='Run Python locally in the assigned work directory; fresh interpreter, journaled execution.').to_dict()
        if structured_only:
            spec = {}
        request = ModelStartRequest(instruction, 'third-party-staged-v1', effective_prompt,
                  'unchanged-staged-prompt', spec,
                  'no-tools-structured-review-v1' if structured_only else 'journaled-python-v1')
        result = dict(completed=False, thread_id=identity, final_text='', usage={}, error=None,
                      reasoning_effort=reasoning_effort, task_role='solver' if is_solver else 'reviewer',
                      review_execution='structured_only' if structured_only else 'python_available',
                      effective_prompt_sha256=hashlib.sha256(effective_prompt.encode()).hexdigest(),
                      transport_instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                      network=self.governor.network,
                      returncode=None, timed_out=False, started_epoch=started, model=model,
                      transport=self.transport_name, events_path=str(audit / 'events.jsonl'))
        totals = {'input_tokens': 0, 'output_tokens': 0}
        action_count = 0
        def event(value):
            with (audit / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        try:
            response = adapter.start_case(request, timeout_seconds=max(.001, deadline-time.monotonic()))
            while True:
                action_count += 1
                for key in totals:
                    value = getattr(response, key)
                    totals[key] = None if totals[key] is None or value is None else totals[key] + value
                action = response.action
                event(dict(turn=action_count, message=response.message,
                           action=action.to_dict() if isinstance(action, ToolBatch) else {'final': action.text},
                           usage=dict(response.raw_usage)))
                if isinstance(action, FinalAnswer):
                    result.update(completed=not response.output_truncated, final_text=action.text,
                                  returncode=0, error='output_truncated' if response.output_truncated else None)
                    break
                if not isinstance(action, ToolBatch):
                    raise APITransportFailure('unsupported_tool_action', ambiguous=True)
                if structured_only:
                    # Defensive boundary even if a provider returns unsolicited calls.
                    raise APITransportFailure('tool_call_in_structured_review', ambiguous=False)
                outputs = []
                for call in action.calls:
                    journal = work / 'execution_journal.jsonl'
                    prior = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
                    if len(prior) >= max_python_executions or deadline <= time.monotonic():
                        result['timed_out'] = True
                        raise APITransportFailure('execution_budget_exhausted')
                    source = work / 'api_analysis.py'
                    code=call.code
                    if nested_helper_reference(code):
                        # Reject before execution, count the attempted tool use,
                        # and retain the unmodified model call in events.jsonl.
                        event(dict(event='nested_helper_rejected',call_id=call.canonical_call_id,
                                   original_code_sha256=hashlib.sha256(code.encode()).hexdigest()))
                        code="raise RuntimeError('The python tool is already journaled. Execute analysis code directly; do not read or invoke run_python.py. Read your analysis.py and exec its contents within this tool if needed.')\n"
                    source.write_text(code)
                    process = subprocess.Popen([str(PYTHON), str(work / 'run_python.py'), str(source)],
                              cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                    try:
                        process.wait(timeout=max(.001, deadline-time.monotonic()) + 2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=5)
                        raise APITransportFailure('python_helper_unfinished') from None
                    rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
                    if len(rows) != len(prior)+1 or rows[-1].get('finished_epoch') is None:
                        raise APITransportFailure('python_helper_invalid_journal')
                    row = rows[-1]
                    outputs.append(PythonExecutionResult(row['returncode']==0, row['stdout'], row['stderr'],
                                   None, row['duration_seconds'], row['execution_index']))
                    # `answer.md` is the authoritative solver artifact. The
                    # staged prompt asks for a final chat acknowledgement only
                    # after writing it, so do not spend one more API request on
                    # that acknowledgement. Reviewer work directories never
                    # contain this file.
                    if is_solver:
                        answer_path = work / ANSWER_FILE
                        if answer_path.is_file() and answer_path.stat().st_size > 0:
                            result.update(completed=True, final_text=f'{ANSWER_FILE} written',
                                          returncode=0, terminal_artifact=ANSWER_FILE)
                            break
                if result.get('completed') and result.get('terminal_artifact') == ANSWER_FILE:
                    break
                response = adapter.continue_case(PythonResultEvent(results=tuple(outputs), tool_batch=action,
                    result_call_ids=tuple(c.canonical_call_id for c in action.calls)),
                    timeout_seconds=max(.001, deadline-time.monotonic()))
        except Exception as exc:
            kind = exc.kind if isinstance(exc, APITransportFailure) else type(exc).__name__
            result.update(error=kind, ambiguous=getattr(exc, 'ambiguous', True))
            event(dict(error=kind, ambiguous=result['ambiguous']))
        finally:
            adapter.close()
            telemetry = dict(http_transport.telemetry)
            known_overhead = sum(telemetry[k] for k in ('client_queue_wait_seconds', 'rate_limit_wait_seconds',
                                  'pacing_wait_seconds','circuit_wait_seconds','failed_request_seconds'))
            telemetry.update(known_infrastructure_overhead_seconds=known_overhead,
                observed_time_excluding_known_overhead=max(0.0,time.time()-started-known_overhead),
                transport_affected=telemetry['http_failures']>0 or bool(result.get('ambiguous')),
                pure_model_inference_seconds=None,
                successful_request_time_scope='Includes unknown provider queueing and network; not pure model inference')
            result['observed_usage'] = dict(totals)
            if result.get('ambiguous'):
                totals = {key:None for key in totals}
            result.update(finished_epoch=time.time(), usage=totals, model_turn_count=action_count,
                          transport_timing=telemetry,
                          api_http_attempts=len(list((audit / 'http').glob('*/http_receipt.json'))))
            write(audit / 'receipt.json', result)
        return result

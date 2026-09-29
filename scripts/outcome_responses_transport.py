"""Audited, no-tool Responses reviewer for the completed-answer rubric."""
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from flowintentbench.external_file_evaluator import write_json
from flowintentbench.model_runner import ProviderRateLimitError, ProviderUnavailableError
from flowintentbench.runtime_config import resolve_api_key, resolve_provider_configuration
from scripts.shared_api_admission import SharedAPIAdmission
from scripts.third_party_benchmark_transport import network_configuration, network_opener
from scripts.yiapi_transport import read_response


class ResponsesOutcomeReviewer:
    def __init__(self, config_path, *, concurrency=4):
        self.runtime = resolve_provider_configuration(config_path=config_path)
        self.key = resolve_api_key(self.runtime)
        self.network = network_configuration()
        self.gate = SharedAPIAdmission(self.runtime.base_url, self.key, limit=concurrency,
                                       allow_limit_increase=True)

    def __call__(self, prompt, model, workdir, output_dir, timeout_seconds=420,
                 output_schema=None, *, reasoning_effort='medium', max_output_tokens=None,
                 structured_output=False, repeat_schema_in_instructions=False):
        audit = Path(output_dir)
        audit.mkdir(parents=True, exist_ok=True)
        payload = {'model': model, 'store': False, 'stream': True,
                   'reasoning': {'effort': reasoning_effort},
                   'instructions': 'Return only JSON matching the supplied schema. No tools.\n'
                                   + json.dumps(output_schema, ensure_ascii=False),
                   'input': [{'role': 'user', 'content': prompt}]}
        if structured_output:
            if not output_schema:
                raise ValueError('structured output requires an explicit schema')
            payload['text']={'format':{'type':'json_schema','name':'answer_evaluation',
                                      'schema':output_schema,'strict':True}}
            payload['instructions']='Return only JSON matching the supplied response schema. No tools. Use the required line citations; do not add commentary.'
            if repeat_schema_in_instructions:
                payload['instructions'] += '\n' + json.dumps(output_schema, ensure_ascii=False)
        if max_output_tokens is not None:
            if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens < 1:
                raise ValueError('max_output_tokens must be a positive integer')
            payload['max_output_tokens'] = max_output_tokens
        endpoint = self.runtime.base_url.rstrip('/') + '/responses'
        started = time.time()
        deadline = time.monotonic() + timeout_seconds
        receipt = {'completed': False, 'final_text': '', 'usage': {}, 'error': None,
                   'model': model, 'reasoning_effort': reasoning_effort,
                   'transport': 'responses', 'endpoint': endpoint, 'started_epoch': started,
                   'network': self.network, 'key_sha256_prefix': hashlib.sha256(self.key.encode()).hexdigest()[:12]}
        write_json(audit / 'request.json', payload)
        write_json(audit / 'receipt.json', receipt)
        try:
            with self.gate.admit(deadline=deadline):
                request = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                    headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json',
                             'Accept': 'text/event-stream'}, method='POST')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('shared admission exhausted request budget')
                with network_opener(self.network)(request, timeout=remaining) as response:
                    receipt['http_status'] = response.status
                    write_json(audit / 'receipt.json', receipt)
                    with (audit / 'response.sse').open('wb') as stream:
                        def audited_lines():
                            for line in response:
                                stream.write(line)
                                stream.flush()
                                yield line
                        value = read_response(audited_lines(), deadline=deadline, require_completed=True,
                                              expected_model=model)
                write_json(audit / 'response.json', value)
                receipt.update(response_model=value.get('model'), response_id=value.get('id'), usage=value.get('usage') or {})
                if value.get('model') != model:
                    raise ValueError('response model does not match requested reviewer')
                if any(item.get('type') == 'function_call' for item in value['output']):
                    raise ValueError('unexpected tool call from no-tool reviewer')
                final = '\n'.join(part['text'] for item in value['output'] if item.get('type') == 'message'
                                  for part in item.get('content', []) if part.get('type') == 'output_text'
                                  and isinstance(part.get('text'),str))
                if not final.strip():
                    raise ValueError('empty completed response')
                receipt.update(completed=True, final_text=final, usage=value.get('usage') or {},
                               response_id=value.get('id'), response_model=value.get('model'))
        except urllib.error.HTTPError as exc:
            detail = exc.read(3000).decode('utf-8', 'replace').replace(self.key, '[REDACTED]')
            receipt.update(http_status=exc.code, error=f'HTTP {exc.code}: {detail}')
            if exc.code == 429:
                self.gate.cooldown(30)
        except ProviderRateLimitError as exc:
            self.gate.cooldown(30)
            receipt['error'] = str(exc).replace(self.key, '[REDACTED]')
        except ProviderUnavailableError as exc:
            message = str(exc).replace(self.key, '[REDACTED]')
            receipt['error'] = f'{type(exc).__name__}: {message}'
            if any(word in message.lower() for word in ('overloaded', 'upstream_error', 'upstream failed')):
                # Shared backoff affects subsequent requests, not an invisible
                # retry of this paid attempt. Its receipt remains a failure.
                self.gate.cooldown(30)
                receipt['retry_cooldown_seconds'] = 30
        except Exception as exc:
            receipt['error'] = f'{type(exc).__name__}: {exc}'.replace(self.key, '[REDACTED]')
        receipt['finished_epoch'] = time.time()
        write_json(audit / 'receipt.json', receipt)
        return receipt

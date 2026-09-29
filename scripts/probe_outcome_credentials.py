"""Probe an explicit auth file without exposing credential contents."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.runtime_config import resolve_api_key, resolve_provider_configuration
from scripts.third_party_benchmark_transport import network_opener, network_configuration
from scripts.benchmark_runtime import require_benchmark_runtime


def main():
    require_benchmark_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument('--auth-path', type=Path, default=ROOT / 'auth.json')
    parser.add_argument('--api-config', type=Path, default=ROOT / 'config/yiapi.toml')
    args = parser.parse_args()
    runtime = resolve_provider_configuration(config_path=args.api_config)
    key = json.loads(args.auth_path.read_text(encoding='utf-8-sig'))['OPENAI_API_KEY'].strip()
    record = {'python': sys.executable, 'auth_path': str(args.auth_path.resolve()),
              'key_length': len(key), 'key_sha256_prefix': hashlib.sha256(key.encode()).hexdigest()[:12],
              'matches_default_resolution': resolve_api_key(runtime) == key,
              'endpoint': runtime.base_url}
    payload = {'model': 'gpt-6-astra', 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
               'max_tokens': 32, 'stream': False, 'reasoning_effort': 'medium'}
    if runtime.model_configuration.get('wire_api') == 'responses':
        from scripts.yiapi_transport import responses_transport
        payload = {'model': 'gpt-6-astra', 'input': [{'role': 'user', 'content': 'Reply OK.'}],
                   'store': False, 'reasoning': {'effort': 'medium'}}
        try:
            value = json.loads(responses_transport(runtime.base_url.rstrip('/') + '/responses',
                {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
                json.dumps(payload).encode(), 90))
            record.update(http_status=200, status=value.get('status'), model=value.get('model'),
                          output_received=bool(value.get('output')))
        except Exception as exc:
            record.update(error=type(exc).__name__, detail=str(exc).replace(key, '[REDACTED]'))
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    request = urllib.request.Request(runtime.base_url.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key,
                                                   'Content-Type': 'application/json'})
    try:
        with network_opener(network_configuration())(request, timeout=60) as response:
            raw = response.read().decode('utf-8', errors='replace')
            value = json.loads(raw)
            record.update(http_status=response.status, model=value.get('model'),
                          choices_received=bool(value.get('choices')), usage=value.get('usage'))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode('utf-8', errors='replace').replace(key, '[REDACTED]')
        record.update(http_status=exc.code, response_error=raw[:1500])
    except Exception as exc:
        record.update(error=type(exc).__name__, detail=str(exc).replace(key, '[REDACTED]'))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

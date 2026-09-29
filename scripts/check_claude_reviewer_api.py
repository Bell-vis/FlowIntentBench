"""One bounded GPT-6 xhigh generation probe using the evaluation transport."""

import argparse
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_runtime import require_benchmark_runtime


def main():
    require_benchmark_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/api_checks')
    args = parser.parse_args()
    if args.output.resolve().drive.upper() == 'C:':
        parser.error('API probe output on C: is disabled')
    from flowintentbench.external_file_evaluator import write_json
    from scripts.third_party_benchmark_transport import APIGovernor, ThirdPartyAgent

    audit = args.output.resolve() / ('gpt6_xhigh_' + str(time.time_ns()))
    governor = APIGovernor(audit / 'health.json', attempts=1, max_concurrency=1)
    agent = ThirdPartyAgent(ROOT / 'config/yiapi.toml', governor, structured_only=True)
    nonce = uuid.uuid4().hex
    schema = {'type': 'object', 'properties': {
        'ok': {'const': True}, 'sum': {'const': 13}, 'nonce': {'const': nonce}},
        'required': ['ok', 'sum', 'nonce'], 'additionalProperties': False}
    print(json.dumps(dict(event='api_probe_started', model='gpt-6-astra',
                         effort='xhigh', max_http_attempts=1, timeout_seconds=120,
                         output=str(audit))), flush=True)
    receipt = agent(
        'API availability test only. Compute 6 + 7 and return exactly one JSON '
        'object containing ok=true, sum, and nonce=' + nonce,
        'gpt-6-astra', audit / 'work', audit / 'transport',
        timeout_seconds=120, output_schema=schema, reasoning_effort='xhigh')
    passed = False
    error = receipt.get('error')
    if receipt.get('completed'):
        try:
            value = json.loads(receipt['final_text'])
            if (value != {'ok': True, 'sum': 13, 'nonce': nonce}
                    or value['ok'] is not True or type(value['sum']) is not int):
                raise ValueError('API probe did not satisfy the exact output contract')
            passed = receipt.get('api_http_attempts') == 1
        except (ValueError, TypeError, KeyError) as exc:
            error = type(exc).__name__
    result = dict(
        status='PASS' if passed else 'FAIL', requested_model='gpt-6-astra',
        reasoning_effort='xhigh', structured_output_valid=passed,
        api_http_attempts=receipt.get('api_http_attempts'), usage=receipt.get('usage'),
        elapsed_seconds=receipt['finished_epoch'] - receipt['started_epoch'],
        error=error, audit_directory=str(audit),
        efficiency_scope='Diagnostic only; excluded from benchmark solver and reviewer ledgers')
    write_json(audit / 'result.json', result)
    print(json.dumps(result), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

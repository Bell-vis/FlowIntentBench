#!/usr/bin/env python3
"""Call each configured API identity without a scientific case or Python tool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.runtime_config import resolve_provider_configuration, resolve_api_key
from scripts.run_full_dataset_n1_experiment import _atomic_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-config", type=Path, default=ROOT / "config.toml")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/provider_check.json")
    args = p.parse_args()
    results = []
    for model in ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"):
        runtime = resolve_provider_configuration(config_path=args.server_config, provider="yiapi", model_id=model)
        key = resolve_api_key(runtime)
        body = {"model": model, "messages": [{"role": "user", "content": 'Return exactly {"ok": true}.'}],
                "reasoning_effort": "xhigh" if model != "gpt-5.6-sol" else "low",
                "max_completion_tokens": 2048, "response_format": {"type": "json_object"}}
        request = urllib.request.Request(runtime.base_url + "/chat/completions",
                  data=json.dumps(body).encode(), method="POST",
                  headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        started = time.monotonic()
        result = {"model": model, "provider": runtime.provider, "endpoint": runtime.base_url,
                  "scientific_case": False, "tool_execution": False}
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.load(response)
            content = data["choices"][0]["message"]["content"]
            result.update({"status": "PASS" if json.loads(content).get("ok") is True else "FAIL",
                           "response_model": data.get("model"), "usage": data.get("usage"),
                           "response_id": data.get("id")})
        except urllib.error.HTTPError as exc:
            result.update({"status": "FAIL", "http_status": exc.code})
        except Exception as exc:
            # No request headers, credentials, or response bodies in failures.
            result.update({"status": "FAIL", "error_type": type(exc).__name__})
        result["seconds"] = time.monotonic() - started
        results.append(result)
        _atomic_json(args.output, {"results": results, "benchmark_observations": 0})
        print(json.dumps(result), flush=True)
    return 0 if all(r["status"] == "PASS" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

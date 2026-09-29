#!/usr/bin/env bash
set -euo pipefail
FIB_API_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$FIB_API_ROOT/run_benchmark_console.sh" all \
  --third-party-api --solver-transport api \
  --api-solver-workers 2 --api-solvers-per-model 2 --api-http-concurrency 4 \
  --api-shared-concurrency 4 --api-judge-workers 2 \
  --reviewer-model gpt-6-astra --judgment-workers 0 --judgment-limit 32 \
  --api-recovery-first-model gpt-5.6-luna --auto-recover-infrastructure "$@"

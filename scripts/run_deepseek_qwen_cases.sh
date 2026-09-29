#!/usr/bin/env bash
set -euo pipefail
FIB_DQ_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FIB_DQ_PYTHON="${FLOWINTENT_PYTHON:-python}"
cd -- "$FIB_DQ_ROOT"
exec "$FIB_DQ_PYTHON" scripts/run_deepseek_qwen_cases.py "$@"

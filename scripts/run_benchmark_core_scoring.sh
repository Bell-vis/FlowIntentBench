#!/usr/bin/env bash
# Primary fast scoring entrypoint for existing answers; never launches a solver.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${FLOWINTENT_PYTHON:-$(command -v python)}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "" && "${CONDA_DEFAULT_ENV}" != "benchmark" && "${CONDA_DEFAULT_ENV}" != "benchmark_py3.12" ]]; then
  echo "当前 Conda 环境不是 benchmark 或 benchmark_py3.12: ${CONDA_DEFAULT_ENV}" >&2
  exit 2
fi
exec "${PYTHON}" "${ROOT}/scripts/run_core_case_scoring.py" "$@"

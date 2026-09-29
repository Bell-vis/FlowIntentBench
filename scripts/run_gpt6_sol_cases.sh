#!/usr/bin/env bash
set -euo pipefail

FIB_GPT6_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FIB_GPT6_PYTHON="${FLOWINTENT_PYTHON:-}"
if [[ -z "${FIB_GPT6_PYTHON}" ]]; then
  if [[ ( "${CONDA_DEFAULT_ENV:-}" == "benchmark" || "${CONDA_DEFAULT_ENV:-}" == "benchmark_py3.12" ) && -x "${CONDA_PREFIX}/bin/python" ]]; then
    FIB_GPT6_PYTHON="${CONDA_PREFIX}/bin/python"
  else
    FIB_GPT6_PYTHON="$(command -v python)"
  fi
fi

cd -- "$FIB_GPT6_ROOT"
exec "$FIB_GPT6_PYTHON" scripts/run_gpt6_sol_cases.py "$@"

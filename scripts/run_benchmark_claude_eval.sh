#!/usr/bin/env bash
# Fixed runtime entrypoint for incremental evaluation of existing answers.
# It never launches a solver and always resumes the existing exchange.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${FLOWINTENT_PYTHON:-$(command -v python)}"
OUTPUT="${ROOT}/outputs/claude_resume_eval"

if [[ ! -x "${PYTHON}" ]]; then
  echo "项目 Python 不存在: ${PYTHON}" >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "" && "${CONDA_DEFAULT_ENV}" != "benchmark" && "${CONDA_DEFAULT_ENV}" != "benchmark_py3.12" ]]; then
  echo "当前 Conda 环境不是 benchmark 或 benchmark_py3.12: ${CONDA_DEFAULT_ENV}" >&2
  exit 2
fi

exec "${PYTHON}" "${ROOT}/scripts/evaluate_existing_claude_xhigh.py" \
  --resume --output "${OUTPUT}" --reviewer-model gpt-6-astra \
  --max-api-calls "${FLOWINTENTBENCH_MAX_API_CALLS:-8}" \
  --max-wall-seconds "${FLOWINTENTBENCH_MAX_WALL_SECONDS:-180}" \
  --judgment-limit "${FLOWINTENTBENCH_JUDGMENT_LIMIT:-16}" \
  --api-judge-workers "${FLOWINTENTBENCH_API_JUDGE_WORKERS:-2}" \
  --api-shared-concurrency "${FLOWINTENTBENCH_API_SHARED_CONCURRENCY:-4}" \
  "$@"

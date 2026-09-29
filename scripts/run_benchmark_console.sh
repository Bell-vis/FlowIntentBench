#!/usr/bin/env bash
set -euo pipefail
FIB_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FIB_PYTHON="${FLOWINTENT_PYTHON:-${CONDA_PREFIX:-}/bin/python}"
if [ ! -x "$FIB_PYTHON" ]; then
FIB_PYTHON="${FLOWINTENT_PYTHON:-$(command -v python)}"
fi
cd -- "$FIB_PROJECT_ROOT"
# Third-party network settings are loaded explicitly by the Python transport
# from config/benchmark_network.toml, independently of terminal proxy exports.
# `api-preflight` checks connectivity without collecting a benchmark answer.
exec "$FIB_PYTHON" scripts/run_codex_benchmark.py "$@"

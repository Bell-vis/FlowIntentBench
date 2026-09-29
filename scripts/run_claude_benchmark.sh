#!/usr/bin/env bash
set -euo pipefail
FIB_CLAUDE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FIB_CLAUDE_PYTHON="${FLOWINTENT_PYTHON:-${CONDA_PREFIX:-}/bin/python}"
if [ ! -x "$FIB_CLAUDE_PYTHON" ]; then
  FIB_CLAUDE_PYTHON="${FLOWINTENT_PYTHON:-$(command -v python)}"
fi
cd -- "$FIB_CLAUDE_ROOT"
exec "$FIB_CLAUDE_PYTHON" scripts/run_claude_benchmark.py "$@"

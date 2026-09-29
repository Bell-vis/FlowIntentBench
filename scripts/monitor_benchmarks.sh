#!/usr/bin/env bash
set -euo pipefail
FIB_MONITOR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FIB_PYTHON="${FLOWINTENT_PYTHON:-${CONDA_PREFIX:-}/bin/python}"
if [ ! -x "$FIB_PYTHON" ]; then
  FIB_PYTHON="${FLOWINTENT_PYTHON:-$(command -v python)}"
fi
exec "$FIB_PYTHON" "$FIB_MONITOR_ROOT/scripts/monitor_benchmarks.py" "$@"

#!/usr/bin/env bash
# Compatibility entrypoint: all solving and grading now use third-party APIs.
set -euo pipefail
FIB_SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$FIB_SCRIPT_ROOT/run_api_solvers_api_graders.sh" "$@"

"""Check the interpreter before importing benchmark dependencies."""

from pathlib import Path
import sys


SUPPORTED_ENVIRONMENTS = {"benchmark", "benchmark_py3.12"}


def require_benchmark_runtime() -> None:
    prefix = Path(sys.prefix).resolve()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "Runtime mismatch: use a Python 3.12 interpreter with the project dependencies installed. "
            f"Actual Python={sys.executable}, version={sys.version.split()[0]}, "
            f"prefix={prefix}. Activate the project environment before running this command."
        )


def main() -> int:
    require_benchmark_runtime()
    import json
    import pydantic
    import typing

    prefix = Path(sys.prefix).resolve()
    for module in (pydantic, typing):
        if not Path(module.__file__).resolve().is_relative_to(prefix):
            raise SystemExit(f"Foreign Python library detected: {module.__file__}")
    print(json.dumps({
        "runtime_check": "OK",
        "python": sys.executable,
        "version": sys.version.split()[0],
        "prefix": str(prefix),
        "pydantic": pydantic.__file__,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

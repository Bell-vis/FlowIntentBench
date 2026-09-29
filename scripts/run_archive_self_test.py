"""Run the fresh-extraction release archive self-test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_handoff_archive import run_archive_self_test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--file-hash-manifest", type=Path, required=True)
    parser.add_argument("--pytest-timeout", type=int, default=300)
    args = parser.parse_args()
    result = run_archive_self_test(
        repository_root=args.root,
        archive_path=args.archive,
        report_path=args.report,
        file_hash_manifest=args.file_hash_manifest,
        pytest_timeout=args.pytest_timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

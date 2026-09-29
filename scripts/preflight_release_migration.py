"""Archive the historical pilot and report whether GT retirement is allowed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.release_migration import (
    archive_legacy_cases,
    preflight_release_migration,
    render_preflight_report,
)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-legacy", action="store_true", help="copy active pilot cases to the immutable archive")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/current/release_migration")
    args = parser.parse_args()
    if args.archive_legacy:
        archive_legacy_cases(ROOT)
    result = preflight_release_migration(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "preflight.md").write_text(render_preflight_report(result), encoding="utf-8")
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

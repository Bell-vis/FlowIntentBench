#!/usr/bin/env python3
"""Archive stale question manifests and derive indexes from frozen v7 bytes."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.reference_authority import (  # noqa: E402
    ARCHIVE_RELATIVE_PATH,
    DERIVED_REVIEW_RELATIVE_PATH,
    audit_frozen_question_consistency,
    sync_frozen_question_manifests,
)


def main() -> int:
    stale = ROOT / "outputs/current/scientific_questions"
    archive = ROOT / ARCHIVE_RELATIVE_PATH / "outputs_current_scientific_questions"
    stale_payloads = (
        [path for path in stale.iterdir() if path.name != "README.md"]
        if stale.is_dir()
        else []
    )
    if stale_payloads:
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            shutil.rmtree(archive)
        shutil.copytree(stale, archive)
    result = sync_frozen_question_manifests(ROOT)
    # Keep the former path as an explicit non-authoritative pointer so an old
    # shell command fails loudly rather than silently reading stale JSON.
    if stale.exists() and stale.resolve() != (ROOT / DERIVED_REVIEW_RELATIVE_PATH).resolve():
        shutil.rmtree(stale)
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "README.md").write_text(
        "Deprecated compatibility path. Use `artifacts/review/current/` for "
        "frozen derived indexes; authoring history is archived under "
        "`artifacts/question_authoring/archive/pre_v7`.\n",
        encoding="utf-8",
    )
    audit = audit_frozen_question_consistency(ROOT)
    print(json.dumps({"sync": result, "consistency": audit}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

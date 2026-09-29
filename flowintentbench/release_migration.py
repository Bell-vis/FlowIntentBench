"""Fail-closed migration helpers for replacing the historical 28-case pilot.

The historical case directories are useful regression material, but they are
not silently promoted to the formal benchmark.  This module only performs
mechanical inventory/archiving and evaluates the existing release counters;
it does not create scientific evidence, Ground Truth, curator decisions, or
new release criteria.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping


REQUIRED_ZERO_COUNTERS = (
    "TOTAL_PENDING_CASE_IDENTITY",
    "TOTAL_PENDING_EVIDENCE",
    "TOTAL_PENDING_SRAC",
    "TOTAL_PENDING_GT",
    "TOTAL_PENDING_CURATOR",
    "TOTAL_NEW_CASE_REQUIRED",
)


@dataclass(frozen=True)
class MigrationPreflight:
    """Auditable state used to decide whether legacy GT may be retired."""

    total_active_cases: int
    active_ground_truth_files: int
    phase2_total_cases: int | None
    phase2_gt_bound: int | None
    phase2_official_release_ready: int | None
    counters: dict[str, int | None]
    legacy_archive_present: bool
    new_portfolio_complete: bool
    old_gt_delete_authorized: bool
    blockers: tuple[str, ...]
    # Optional reporting projection.  Migration authorization intentionally
    # remains based on the established Phase-2 counters and archive checks;
    # formal manifest authority is lifecycle-gated separately.
    formal_release_eligible: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_case_dirs(root: str | Path) -> list[Path]:
    """Return active case directories with an explicit GT file."""

    base = Path(root) / "datasets"
    return sorted(
        path.parent
        for path in base.glob("*/construction/cases/*/ground_truth.json")
        if path.is_file()
    )


def archive_legacy_cases(
    root: str | Path,
    archive_root: str | Path | None = None,
    *,
    refuse_overwrite: bool = True,
) -> dict[str, Any]:
    """Copy active pilot cases to an explicit archive and record file hashes.

    This operation is intentionally a copy, never a delete.  The destination
    is refused when it already contains a case unless ``refuse_overwrite`` is
    disabled, preventing accidental mutation of the historical snapshot.
    """

    repo = Path(root)
    destination = Path(archive_root) if archive_root else repo / "artifacts/archive/controlled_pilot_legacy_cases"
    cases = active_case_dirs(repo)
    records: list[dict[str, Any]] = []
    for case_dir in cases:
        relative = case_dir.relative_to(repo / "datasets")
        target = destination / relative
        if target.exists() and refuse_overwrite:
            raise FileExistsError(f"legacy archive case already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(case_dir, target)
        files = {
            str(path.relative_to(target)): _sha256(path)
            for path in sorted(target.rglob("*"))
            if path.is_file()
        }
        records.append(
            {
                "dataset_id": relative.parts[0],
                "case_id": relative.parts[-1],
                "relative_case_path": str(relative),
                "files": files,
                "classification": "CONTROLLED_PILOT_ONLY",
            }
        )
    manifest = {
        "schema_version": "controlled-pilot-legacy-archive-v1",
        "classification": "CONTROLLED_PILOT_ONLY",
        "source": "active datasets/*/construction/cases/*",
        "case_count": len(records),
        "cases": records,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "archive_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def preflight_release_migration(root: str | Path) -> MigrationPreflight:
    """Evaluate the existing Phase-2 counters without changing them."""

    repo = Path(root)
    active = active_case_dirs(repo)
    closure = _read_json(repo / "outputs/current/qualified_scientific_cases/phase2_closure.json")
    closure = closure if isinstance(closure, Mapping) else {}
    counters: dict[str, int | None] = {}
    for name in REQUIRED_ZERO_COUNTERS:
        value = closure.get(name)
        counters[name] = int(value) if isinstance(value, int) else None
    total_cases = closure.get("TOTAL_CASES")
    gt_bound = closure.get("TOTAL_GT_BOUND")
    official = closure.get("OFFICIAL_RELEASE_READY")
    formal_eligible = closure.get("FORMAL_RELEASE_ELIGIBLE")
    archive = repo / "artifacts/archive/controlled_pilot_legacy_cases/archive_manifest.json"
    archive_doc = _read_json(archive)
    blockers: list[str] = []
    if len(active) != 28:
        blockers.append(f"ACTIVE_CASE_COUNT:{len(active)} (expected 28)")
    if len([case for case in active if (case / "ground_truth.json").is_file()]) != 28:
        blockers.append("ACTIVE_GROUND_TRUTH_COUNT_NOT_28")
    if total_cases != 28:
        blockers.append(f"PHASE2_TOTAL_CASES:{total_cases!r}")
    for name, value in counters.items():
        if value != 0:
            blockers.append(f"{name}:{value!r}")
    if gt_bound != 28:
        blockers.append(f"TOTAL_GT_BOUND:{gt_bound!r}")
    if official != 28:
        blockers.append(f"OFFICIAL_RELEASE_READY:{official!r}")
    # ``OFFICIAL_RELEASE_READY`` remains the established Phase-2/migration
    # counter.  Formal manifest authority is reported separately and is not
    # folded into this legacy-GT retirement predicate.
    if not archive.is_file():
        blockers.append("LEGACY_ARCHIVE_MISSING")
    elif not isinstance(archive_doc, Mapping) or archive_doc.get("classification") != "CONTROLLED_PILOT_ONLY":
        blockers.append("LEGACY_ARCHIVE_CLASSIFICATION_INVALID")
    elif archive_doc.get("case_count") != 28 or len(archive_doc.get("cases", ())) != 28:
        blockers.append(f"LEGACY_ARCHIVE_CASE_COUNT:{archive_doc.get('case_count')!r}")
    # This is intentionally the Phase-2 migration completeness predicate.
    # It authorizes legacy-GT retirement only after the established counter
    # and archive checks; formal release membership remains the authority of
    # the explicit benchmark manifest.
    complete = not blockers
    return MigrationPreflight(
        total_active_cases=len(active),
        active_ground_truth_files=len([case for case in active if (case / "ground_truth.json").is_file()]),
        phase2_total_cases=int(total_cases) if isinstance(total_cases, int) else None,
        phase2_gt_bound=int(gt_bound) if isinstance(gt_bound, int) else None,
        phase2_official_release_ready=int(official) if isinstance(official, int) else None,
        formal_release_eligible=int(formal_eligible) if isinstance(formal_eligible, int) else None,
        counters=counters,
        legacy_archive_present=archive.is_file(),
        new_portfolio_complete=complete,
        old_gt_delete_authorized=complete,
        blockers=tuple(blockers),
    )


def render_preflight_report(result: MigrationPreflight) -> str:
    lines = [
        "# Release migration preflight",
        "",
        "This report is mechanical lifecycle state. It does not create or infer scientific evidence.",
        "",
        f"- Active cases with Ground Truth: **{result.active_ground_truth_files}/{result.total_active_cases}**",
        f"- Phase-2 total cases: **{result.phase2_total_cases}**",
        f"- GT bound: **{result.phase2_gt_bound}**",
        f"- Phase-2 `OFFICIAL_RELEASE_READY` counter: **{result.phase2_official_release_ready}**",
        f"- Formal release authority (`RELEASE_ELIGIBLE`): **{result.formal_release_eligible}**",
        f"- Legacy archive present: **{result.legacy_archive_present}**",
        f"- New portfolio complete: **{result.new_portfolio_complete}**",
        f"- Old GT delete authorized: **{result.old_gt_delete_authorized}**",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in result.blockers) if result.blockers else lines.append("- None")
    lines.extend(
        [
            "",
            "Legacy controlled-pilot cases remain immutable regression material until a complete new portfolio is validated.",
        ]
    )
    return "\n".join(lines) + "\n"

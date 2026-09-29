#!/usr/bin/env python3
"""Build the complete seven-dataset human-review question portfolio.

The command consumes the existing scientific-question candidate matrix by
default.  It only creates review artifacts; it does not select questions,
mutate cases, or run downstream benchmark gates.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.human_review_portfolio import (  # noqa: E402
    _candidate_flow_expert_review_status,
    _combustor_target_selection_is_qualified,
    render_human_review_markdown,
    run_human_review_scientific_question_expansion,
)


CANONICAL_OUTPUT_ROOT = ROOT / "artifacts/review/current"
MATRIX_OUTPUT_ROOT = ROOT / "outputs/current/scientific_question_matrix"
MATRIX_MANIFEST_NAME = "scientific_question_manifest.json"
PORTFOLIO_ARTIFACT_TYPES = {
    "HUMAN_REVIEW_SCIENTIFIC_QUESTION_PORTFOLIO",
    "SCIENTIFIC_QUESTION_HUMAN_REVIEW_PORTFOLIO",
}
DETERMINISTIC_FALLBACK_SOURCES = {
    "DETERMINISTIC_REVIEW_FALLBACK",
    "DETERMINISTIC_FALLBACK",
}
_ABSOLUTE_HOME_PATH = re.compile(r"/home/[^\s\"']+")


def _json_object(path: Path) -> Mapping[str, Any] | None:
    """Read a manifest header without making a candidate-science judgment."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, Mapping) else None


def _is_portfolio_manifest(path: Path) -> bool:
    value = _json_object(path)
    if value is None:
        return False
    artifact_type = str(value.get("artifact_type", "")).strip()
    schema_version = str(value.get("schema_version", "")).strip()
    return artifact_type in PORTFOLIO_ARTIFACT_TYPES or schema_version.startswith(
        "human-review-scientific-question-expansion-"
    )


def _discover_matrix_manifest(root: Path, explicit: Path | None) -> Path | None:
    """Find a source matrix while refusing to consume a final portfolio.

    The old repository stored the matrix under ``scientific_questions``.  The
    final portfolio now uses that directory too, so the staging directory is
    preferred and portfolio-shaped manifests are always ignored.
    """

    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"matrix manifest not found: {candidate}")
        if _is_portfolio_manifest(candidate):
            raise ValueError(
                "matrix manifest points to a final human-review portfolio; "
                "provide the source candidate matrix or use --rebuild-matrix"
            )
        return candidate

    candidates = (
        root / "outputs/current/scientific_question_matrix",
        root / "outputs/current/scientific_question_source",
        root / "outputs/current/scientific_question_review",
        root / "outputs/current/scientific_questions",
    )
    seen: set[Path] = set()
    for directory in candidates:
        path = (directory / MATRIX_MANIFEST_NAME).resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file() and not _is_portfolio_manifest(path):
            return path
    return None


def _stage_matrix_if_overwritten(path: Path | None, output_root: Path, root: Path) -> Path | None:
    """Preserve an old in-place matrix before writing the canonical portfolio."""

    if path is None or path.parent.resolve() != output_root.resolve():
        return path
    staged = (root / "outputs/current/scientific_question_matrix" / path.name).resolve()
    if staged != path.resolve():
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, staged)
    return staged


def _portable_value(value: Any, root: Path) -> Any:
    """Remove working-tree absolute paths from review artifacts.

    This is intentionally limited to path-like strings.  Scientific wording is
    otherwise preserved byte-for-byte.
    """

    if isinstance(value, Mapping):
        return {str(key): _portable_value(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_value(item, root) for item in value]
    if isinstance(value, tuple):
        return [_portable_value(item, root) for item in value]
    if not isinstance(value, str):
        return value
    if "/home/" not in value:
        return value
    root_text = str(root.resolve())
    if value == root_text or value.startswith(root_text + "/"):
        return value.replace(root_text + "/", "", 1)
    # An external temporary path has no stable identity.  Keep a portable
    # artifact marker instead of leaking the local working tree.
    return _ABSOLUTE_HOME_PATH.sub("<repository-relative-path>", value)


def _relative_identifier(path: Path | None, root: Path) -> str | None:
    """Return a stable repository-relative artifact identifier."""

    if path is None:
        return None
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        # Caller-supplied external manifests may be useful for a local run,
        # but their working-tree prefix must never become canonical metadata.
        return resolved.name


def _candidate_is_genuine(candidate: Mapping[str, Any]) -> bool:
    """Return whether an authored candidate may enter genuine-question counts."""

    source = str(candidate.get("candidate_source", "")).strip().upper()
    if source in DETERMINISTIC_FALLBACK_SOURCES or "FALLBACK" in source:
        return False
    if source in {"COMBUSTOR_TARGET_REDESIGN_REVIEW", "TARGET_PROPOSAL"}:
        return False
    if str(candidate.get("candidate_role", "")).strip().upper() == "TARGET_PROPOSAL":
        return False
    # The authoring path should identify itself explicitly.  Missing source is
    # treated as unknown rather than silently promoted to a successful count.
    if not source:
        return False
    question = str(
        candidate.get("scientific_question")
        or candidate.get("question_text")
        or candidate.get("model_visible_text")
        or ""
    ).strip()
    return bool(question)


def _candidate_authoring_invocation_verified(candidate: Mapping[str, Any]) -> bool:
    """Require an explicit successful author call for canonical counts.

    The legacy helper above remains permissive for old, hand-shaped fixtures;
    canonical portfolio artifacts carry ``authoring_provenance`` and must pass
    this stricter check.  A status label alone is not a call record.
    """

    # ``authoring_provenance.invocation_verified`` is a derived assertion, not
    # an invocation record.  Never accept that flag by itself: a hand-edited
    # canonical manifest could otherwise turn an unrun/fallback candidate into
    # a genuine authored question.  We require the concrete envelope below.
    for key in ("question_authoring_invocation", "authoring_invocation", "authoring_call"):
        record = candidate.get(key)
        if not isinstance(record, Mapping):
            continue
        status = str(record.get("invocation_status", "")).strip().upper()
        mode = str(record.get("execution_mode", "")).strip().upper()
        if status in {"SUCCESS", "PASS", "COMPLETED", "COMPLETE"} and not (
            mode.startswith("NOT_RUN") or "DETERMINISTIC" in mode
        ):
            return True
    return False


def _strict_candidate_is_genuine(candidate: Mapping[str, Any]) -> bool:
    # Canonical portfolio artifacts carry the producer's explicit provenance
    # decision.  A source label and successful transport envelope alone are
    # insufficient: callers must not be able to inflate the 28-question
    # counters by hand-editing metadata around an unverified/partial record.
    if candidate.get("genuine_authored_question") is not True:
        return False
    if str(candidate.get("authoring_status", "")).strip().upper() not in {
        "PASS",
    }:
        return False
    fixed = candidate.get("fixed_operationalization")
    opened = candidate.get("open_operationalization")
    finding = candidate.get("finding_responsibility")
    target = candidate.get("scientific_target")
    if (
        not str(target or "").strip()
        or not isinstance(fixed, list)
        or not isinstance(opened, list)
        or finding in (None, "")
    ):
        return False
    return _candidate_is_genuine(candidate) and _candidate_authoring_invocation_verified(
        candidate
    )


def _strict_count_mode(manifest: Mapping[str, Any]) -> bool:
    """Identify canonical portfolio artifacts (as opposed to old fixtures)."""

    artifact_type = str(manifest.get("artifact_type", "")).strip().upper()
    schema = str(manifest.get("schema_version", "")).strip().lower()
    return artifact_type in PORTFOLIO_ARTIFACT_TYPES or schema.startswith(
        "human-review-scientific-question-expansion-"
    )


def _count_condition_candidates(
    rows: list[Mapping[str, Any]],
    predicate,
) -> int:
    count = 0
    for row in rows:
        candidates = row.get("question_candidates")
        if not isinstance(candidates, list):
            continue
        if any(isinstance(item, Mapping) and predicate(item, row) for item in candidates):
            count += 1
    return count


def _portfolio_counters(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Normalize final counters without treating fallback prose as authored."""

    rows = [item for item in manifest.get("conditions", ()) if isinstance(item, Mapping)]
    slots = len(rows)
    # Canonical portfolio artifacts use strict provenance.  Keep permissive
    # behavior only for pre-portfolio unit fixtures that intentionally omit an
    # artifact type and invocation records.
    strict_mode = _strict_count_mode(manifest)
    genuine_predicate = (
        _strict_candidate_is_genuine if strict_mode else _candidate_is_genuine
    )
    genuine = _count_condition_candidates(rows, lambda candidate, _row: genuine_predicate(candidate))

    # Explicit candidate-level statuses are preferred.  A missing status is
    # deliberately zero, because reviewability must not be inferred from text.
    reviewed = _count_condition_candidates(
        rows,
        lambda candidate, _row: genuine_predicate(candidate)
        and (
            _candidate_flow_expert_review_status(candidate)[0] == "PASS"
            if strict_mode
            else str(candidate.get("flow_expert_review_status", "")).upper() == "PASS"
        ),
    )
    fidelity = _count_condition_candidates(
        rows,
        lambda candidate, _row: genuine_predicate(candidate)
        and str(candidate.get("semantic_fidelity_status", "")).upper() == "PASS",
    )
    hccq = _count_condition_candidates(
        rows,
        lambda candidate, _row: genuine_predicate(candidate)
        and str(candidate.get("hccq_status", "")).upper() == "PASS",
    )
    leakage = _count_condition_candidates(
        rows,
        lambda candidate, _row: genuine_predicate(candidate)
        and str(candidate.get("family_template_leakage_status", "")).upper() == "PASS",
    )
    if strict_mode:
        # Recompute this gate from candidate records rather than trusting a
        # stale row-level boolean.  Every quality gate, including the
        # dedicated final Flow Expert review, must pass.
        human_ready = sum(
            any(
                isinstance(candidate, Mapping)
                and genuine_predicate(candidate)
                and _candidate_flow_expert_review_status(candidate)[0] == "PASS"
                and str(candidate.get("semantic_fidelity_status", "")).upper() == "PASS"
                and str(candidate.get("hccq_status", "")).upper() == "PASS"
                and str(candidate.get("family_template_leakage_status", "")).upper() == "PASS"
                for candidate in (row.get("question_candidates") or [])
            )
            for row in rows
        )
    else:
        human_ready = sum(
            bool(row["genuine_question_ready"])
            if "genuine_question_ready" in row
            else (
                str(row.get("QUESTION_CANDIDATE_STATUS", "")).upper()
                == "READY_FOR_HUMAN_REVIEW"
                and any(
                    isinstance(candidate, Mapping) and _candidate_is_genuine(candidate)
                    for candidate in (row.get("question_candidates") or [])
                )
            )
            for row in rows
        )
    # Derive every count from the condition records.  Carrying a stale source
    # matrix counter into the final artifact could turn fallback prose into a
    # successful result or hide a newly omitted slot.
    counters = {
        "TOTAL_QUESTION_SLOTS": slots,
        "TOTAL_GENUINE_AUTHORED_QUESTIONS": genuine,
        "TOTAL_FLOW_EXPERT_REVIEWED": reviewed,
        "TOTAL_SEMANTIC_FIDELITY_PASS": fidelity,
        "TOTAL_HCCQ_PASS": hccq,
        "TOTAL_FAMILY_TEMPLATE_LEAKAGE_PASS": leakage,
        "TOTAL_HUMAN_REVIEW_READY": human_ready,
        "TOTAL_RELEASE_READY": sum(
            str(row.get("SCIENTIFIC_RELEASE_STATUS", "")).upper() == "ELIGIBLE"
            for row in rows
        ),
        "TOTAL_PENDING_EVIDENCE": sum(
            str(row.get("SCIENTIFIC_RELEASE_STATUS", "")).upper()
            == "BLOCKED_PENDING_EVIDENCE"
            for row in rows
        ),
        "TOTAL_PENDING_O_SPACE": sum(
            str(row.get("SCIENTIFIC_RELEASE_STATUS", "")).upper()
            == "BLOCKED_PENDING_O_SPACE"
            for row in rows
        ),
        "TOTAL_PENDING_SRAC": sum(
            str(row.get("SCIENTIFIC_RELEASE_STATUS", "")).upper()
            in {"BLOCKED_PENDING_SRAC", "PENDING_SRAC"}
            or "SRAC" in str(row.get("GT_SRAC_IMPACT", "")).upper()
            and "REQUIRED" in str(row.get("GT_SRAC_IMPACT", "")).upper()
            for row in rows
        ),
        # Canonical portfolio artifacts derive this family-level count only
        # from an explicit recommended provisional target.  Alternatives and
        # stale summary integers are audit material, not a recommendation.
        "TOTAL_PROVISIONAL_TARGET": (
            sum(
                _combustor_target_selection_is_qualified(candidate)
                for candidate in (
                    manifest.get("combustor_redesign_target_candidates") or []
                )
                if isinstance(candidate, Mapping)
            )
            if strict_mode
            else (
                int(manifest["TOTAL_PROVISIONAL_TARGET"])
                if isinstance(manifest.get("TOTAL_PROVISIONAL_TARGET"), int)
                else sum(
                    str(row.get("SCIENTIFIC_RELEASE_STATUS", "")).upper()
                    in {"PROVISIONAL_TARGET_PENDING_HUMAN_APPROVAL", "PROVISIONAL_TARGET"}
                    or any(
                        isinstance(candidate, Mapping)
                        and str(candidate.get("target_status", "")).upper()
                        == "PROVISIONAL_TARGET_PENDING_HUMAN_APPROVAL"
                        for candidate in (row.get("question_candidates") or [])
                    )
                    for row in rows
                )
            )
        ),
        "TOTAL_OMITTED": sum(
            not any(
                isinstance(candidate, Mapping)
                and genuine_predicate(candidate)
                for candidate in (row.get("question_candidates") or [])
            )
            for row in rows
        ),
    }
    return counters


def _summary_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    """Expose structural validation and scientific completion separately.

    ``status`` is retained as the backwards-compatible structural validator
    result.  The explicit ``portfolio_status`` field prevents a structurally
    valid but incomplete authored set from being reported as a completed
    28-question portfolio.
    """

    validation = result.get("validation")
    validation_status = (
        str(validation.get("status", "NOT_REPORTED"))
        if isinstance(validation, Mapping)
        else "NOT_REPORTED"
    )
    portfolio_status = str(
        result.get("HUMAN_REVIEW_PORTFOLIO_STATUS", "NOT_REPORTED")
    )
    return {
        # Compatibility field: this is only the JSON/artifact structure check.
        "status": validation_status,
        "validation_status": validation_status,
        "portfolio_status": portfolio_status,
        "portfolio_complete": portfolio_status == "COMPLETE",
        "schema_version": result.get("schema_version"),
        **_portfolio_counters(result),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CANONICAL_OUTPUT_ROOT,
        help="Canonical authoring/review artifact directory (defaults to artifacts/review/current).",
    )
    parser.add_argument(
        "--matrix-manifest",
        type=Path,
        default=None,
        help="Existing source-matrix scientific_question_manifest.json; defaults to the staged matrix directory.",
    )
    parser.add_argument(
        "--rebuild-matrix",
        action="store_true",
        help="Rebuild the source matrix before creating review artifacts (offline unless --live-matrix is set).",
    )
    parser.add_argument(
        "--live-matrix",
        action="store_true",
        help="When rebuilding, run the configured live Flow Expert matrix.",
    )
    parser.add_argument("--matrix-output-root", type=Path, default=None)
    parser.add_argument("--server-config", type=Path, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-workers", type=int, default=7)
    parser.add_argument("--candidate-count", type=int, choices=(2, 3), default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    try:
        explicit_matrix = (
            args.matrix_manifest.expanduser().resolve()
            if args.matrix_manifest is not None
            else None
        )
        matrix_manifest = _discover_matrix_manifest(repository_root, explicit_matrix)
        # If no source matrix is available, force a temporary rebuild.  This
        # prevents the canonical output manifest from being mistaken for its
        # own source on a subsequent invocation.
        refresh_matrix = bool(args.rebuild_matrix or matrix_manifest is None)
        matrix_manifest = _stage_matrix_if_overwritten(
            matrix_manifest, output_root, repository_root
        )
        matrix_output_root = (
            args.matrix_output_root.expanduser().resolve()
            if args.matrix_output_root is not None
            else None
        )
        result = run_human_review_scientific_question_expansion(
            repository_root,
            output_root,
            matrix_manifest_path=matrix_manifest,
            matrix_output_root=matrix_output_root,
            refresh_matrix=refresh_matrix,
            run_live_matrix=args.live_matrix,
            config_path=args.server_config,
            api_key=args.api_key,
            timeout=args.timeout,
            max_workers=args.max_workers,
            candidate_count=args.candidate_count,
        )
        if matrix_manifest is not None:
            result["source_matrix_manifest"] = _relative_identifier(
                matrix_manifest, repository_root
            )
        # The library API remains usable with arbitrary temporary output
        # roots.  The CLI is the canonical artifact boundary, so rewrite only
        # path metadata and preserve all scientific content.
        portable_result = _portable_value(result, repository_root)
        if isinstance(portable_result, Mapping):
            result = dict(portable_result)
        counters = _portfolio_counters(result)
        result.update(counters)
        manifest_path = output_root / MATRIX_MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        review_path = output_root / "scientific_question_review.md"
        review_path.write_text(
            _ABSOLUTE_HOME_PATH.sub(
                "<repository-relative-path>",
                render_human_review_markdown(result),
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - exercised by CLI integration
        print(
            json.dumps(
                {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    summary = _summary_payload(result)
    # Preserve the historical all-caps spelling in CLI output while exposing
    # the clearer lower-case alias used by new callers.
    summary["HUMAN_REVIEW_PORTFOLIO_STATUS"] = summary["portfolio_status"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result["validation"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

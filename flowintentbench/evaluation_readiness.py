"""Offline readiness audit for the frozen 28-case evaluation portfolio.

The audit is deliberately read-only.  It validates reference truth,
construction/evaluator boundaries, saved-response reuse, and deterministic
metric replay without invoking an evaluated model or an evaluator model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluator import (
    CaseEvaluationRecord,
    EvaluatorComponentIdentity,
    PendingCaseEvaluationRecord,
    load_case_evaluation_record,
)
from .experiment import code_data_version
from .evaluator_backend import StructuredEvaluatorBackend
from .evaluation_records import iter_current_evaluation_records
from .evaluation_contract import FrozenEvaluationCase
from .model_response_reuse import require_reusable_model_responses
from .pending_adjudication import (
    PENDING_OWNER_BY_TYPE,
    REQUIRED_PENDING_OWNERS,
    PendingContinuationRegistry,
    load_pending_owner_manifest,
    route_pending_adjudication,
    continuation_execution_manifest_digest,
)
from .reference_freeze import (
    REFERENCE_FREEZE_V4_SCHEMA_VERSION,
    REFERENCE_FREEZE_V5_SCHEMA_VERSION,
    REFERENCE_FREEZE_V6_SCHEMA_VERSION,
    REFERENCE_FREEZE_V7_SCHEMA_VERSION,
    verify_reference_portfolio_freeze,
    verify_reference_portfolio_freeze_v4,
    verify_reference_portfolio_freeze_v5,
    verify_reference_portfolio_freeze_v6,
    verify_reference_portfolio_freeze_v7,
)


READINESS_AUDIT_VERSION = "offline-evaluation-readiness-v1"
FRESH_N1_READINESS_VERSION = "fresh-n1-model-test-readiness-v1"


def _verify_offline_evaluator_manifest_content(value: Mapping[str, Any]) -> None:
    """Re-derive the local offline evaluator identity without a model call.

    The persisted evaluator manifest is an input freeze, not merely a label.
    For the repository's offline contract manifest we can reconstruct the
    exact backend from its declared fields and reject stale hashes before a
    live N=1 collection starts.  Provider-specific live manifests remain
    authoritative artifacts and are checked for their required digest fields
    without guessing their transport implementation.
    """

    backend_value = value.get("evaluator_backend")
    if not isinstance(backend_value, Mapping):
        return
    if (
        backend_value.get("implementation_id") != "flowintentbench-structured-evaluator"
        or backend_value.get("version") != "offline-contract"
    ):
        return
    hashes = value.get("deterministic_contract_hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("offline evaluator manifest lacks deterministic contract hashes")
    repairs = hashes.get("max_format_repairs", 1)
    if isinstance(repairs, bool) or not isinstance(repairs, int) or repairs not in {0, 1}:
        raise ValueError("offline evaluator manifest has invalid max_format_repairs")
    identity = EvaluatorComponentIdentity(
        implementation_id=str(backend_value.get("implementation_id")),
        version=str(backend_value.get("version")),
        provider=str(backend_value.get("provider", "")),
        model=str(backend_value.get("model", "")),
        model_configuration=backend_value.get("model_configuration", {}),
        # The persisted prompt marker is derived by ``manifest``.  Supplying
        # it here would duplicate the marker during re-derivation.
        prompt_version=None,
    )
    backend = StructuredEvaluatorBackend(
        completion=lambda _operation, _payload: {"result": "MATCH"},
        identity=identity,
        extraction_spec_id=str(value.get("extraction_spec_id", "")),
        eligibility_spec_id=str(value.get("eligibility_spec_id", "")),
        semantic_match_spec_id=str(value.get("semantic_match_spec_id", "")),
        max_format_repairs=repairs,
    )
    expected = backend.manifest(
        unit_converter_version=value.get("unit_converter_version")
    ).to_dict()
    # Continuation execution identity is an orthogonal runtime-installed
    # extension. It is checked against the supplied registry by the caller;
    # the deterministic offline backend must not pretend to own it.
    comparable = dict(value)
    comparable.pop("continuation_execution_manifest", None)
    if expected != comparable:
        raise ValueError("offline evaluator manifest is stale relative to current evaluator semantics")


def _run_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("run_record.json"))


def _final_record_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in iter_current_evaluation_records(root)
        if path.name == "case_evaluation_record.json"
    )


def _pending_record_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in iter_current_evaluation_records(root)
        if path.name == "pending_case_evaluation.json"
    )


def audit_offline_evaluation_readiness(
    repository_root: str | Path,
    *,
    reference_freeze_manifest: str | Path | None = None,
    saved_run_root: str | Path | None = None,
    collected_case_manifest: str | Path | Mapping[str, Any] | None = None,
    candidate_case_manifest: str | Path | Mapping[str, Any] | None = None,
    existing_evaluation_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit whether scientific metrics can be computed after adjudication.

    ``scientific_metrics_computable`` describes the implementation and frozen
    input contracts.  It is distinct from ``current_metrics_available``:
    existing responses may still be pending or infrastructure-invalid even
    when the metric pipeline itself is sound.
    """

    root = Path(repository_root).resolve()
    checks: dict[str, Any] = {}
    blockers: list[str] = []

    freeze_manifest_path: Path | None = None
    freeze_schema: str | None = None
    try:
        # Dispatch to the verifier for the selected immutable schema.  The
        # historical generic v2 verifier cannot reproduce v4/v5/v6 payloads
        # because those snapshots add evaluator-facing artifacts and semantic
        # alignment checks; using it here created a false readiness failure for
        # the canonical v6 N=1 inputs.
        manifest_candidate = (
            Path(reference_freeze_manifest).resolve()
            if reference_freeze_manifest is not None
            else root / "artifacts/archive/reference/portfolio_v2/reference_science_baseline_manifest.json"
        )
        manifest_schema = ""
        if manifest_candidate.is_file():
            try:
                manifest_schema = str(
                    json.loads(manifest_candidate.read_text(encoding="utf-8"))
                    .get("schema_version", "")
                )
            except (OSError, json.JSONDecodeError, AttributeError):
                manifest_schema = ""
        verifier = {
            REFERENCE_FREEZE_V4_SCHEMA_VERSION: verify_reference_portfolio_freeze_v4,
            REFERENCE_FREEZE_V5_SCHEMA_VERSION: verify_reference_portfolio_freeze_v5,
            REFERENCE_FREEZE_V6_SCHEMA_VERSION: verify_reference_portfolio_freeze_v6,
            REFERENCE_FREEZE_V7_SCHEMA_VERSION: verify_reference_portfolio_freeze_v7,
        }.get(manifest_schema, verify_reference_portfolio_freeze)
        freeze = verifier(root, reference_freeze_manifest)
        checks["reference_portfolio"] = freeze
        # The selected immutable reference snapshot is the only valid source
        # for evaluation-case consumability.  Falling back to the mutable
        # construction workspace here used to let stale questions pass the
        # readiness gate while the N=1 manifest pointed at a newer snapshot.
        manifest_value = freeze.get("manifest_path")
        if isinstance(manifest_value, str) and manifest_value.strip():
            freeze_manifest_path = Path(manifest_value).resolve()
            try:
                freeze_schema = str(
                    json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
                    .get("schema_version", "")
                )
            except (OSError, json.JSONDecodeError, AttributeError):
                freeze_schema = None
    except Exception as exc:
        checks["reference_portfolio"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        blockers.append("REFERENCE_PORTFOLIO_FREEZE_INVALID")

    case_rows: list[dict[str, Any]] = []
    frozen_cases: dict[str, FrozenEvaluationCase] = {}
    # v2 is a legacy freeze that intentionally predates SEC artifacts; its
    # snapshot cannot be consumed by the current FrozenEvaluationCase adapter.
    # v3+ snapshots contain the complete evaluator-facing case bundle and are
    # therefore the sole source for readiness checks.  This keeps compatibility
    # replay working without reopening the v6 mutable-workspace split-brain.
    use_immutable_case_root = freeze_manifest_path is not None and freeze_schema in {
        "reference-science-baseline-v3",
        REFERENCE_FREEZE_V4_SCHEMA_VERSION,
        REFERENCE_FREEZE_V5_SCHEMA_VERSION,
        REFERENCE_FREEZE_V6_SCHEMA_VERSION,
        REFERENCE_FREEZE_V7_SCHEMA_VERSION,
    }
    case_root = (
        freeze_manifest_path.parent / "cases"
        if use_immutable_case_root
        else root / "outputs/current/qualified_scientific_cases/agent_ready_cases"
    )
    if case_root.is_dir():
        for case_dir in sorted(
            path
            for dataset_dir in case_root.iterdir()
            if dataset_dir.is_dir()
            for path in dataset_dir.iterdir()
            if path.is_dir()
        ):
            try:
                frozen_case = FrozenEvaluationCase.from_paths(
                    case_dir, formal_frozen=False
                )
                case_rows.append(
                    {
                        "case_id": frozen_case.case_metadata.case_id,
                        "dataset_id": frozen_case.case_metadata.dataset_id,
                        "status": "PASS",
                    }
                )
                frozen_cases[frozen_case.case_metadata.case_id] = frozen_case
            except Exception as exc:
                case_rows.append(
                    {
                        "case_id": case_dir.name,
                        "dataset_id": case_dir.parent.name,
                        "status": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    checks["evaluation_case_consumability"] = {
        "status": (
            "PASS"
            if len(case_rows) == 28 and all(row["status"] == "PASS" for row in case_rows)
            else "FAIL"
        ),
        "case_count": len(case_rows),
        "consumable_case_count": sum(row["status"] == "PASS" for row in case_rows),
        "case_source": (
            "immutable_reference_snapshot"
            if use_immutable_case_root
            else "mutable_agent_ready_construction_workspace"
        ),
        "case_root": str(case_root),
        "rows": case_rows,
    }
    if checks["evaluation_case_consumability"]["status"] != "PASS":
        blockers.append("EVALUATION_CASE_CONSUMABILITY_FAILED")

    if saved_run_root is not None:
        if collected_case_manifest is None or candidate_case_manifest is None:
            checks["saved_response_reuse"] = {
                "status": "FAIL",
                "reason": "case manifests are required for reuse audit",
            }
            blockers.append("SAVED_RESPONSE_REUSE_INPUT_MISSING")
        else:
            try:
                reuse = require_reusable_model_responses(
                    run_records=_run_paths(Path(saved_run_root)),
                    collected_case_manifest=collected_case_manifest,
                    candidate_case_manifest=candidate_case_manifest,
                )
                checks["saved_response_reuse"] = reuse
            except Exception as exc:
                checks["saved_response_reuse"] = {
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                blockers.append("SAVED_RESPONSE_REUSE_NOT_ESTABLISHED")
    else:
        checks["saved_response_reuse"] = {"status": "NOT_REQUESTED"}

    replay_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    if existing_evaluation_root is not None:
        for path in _final_record_paths(Path(existing_evaluation_root)):
            try:
                record = load_case_evaluation_record(path)
                if not isinstance(record, CaseEvaluationRecord):
                    raise ValueError("record is not terminal")
                record.replay()
                replay_rows.append(
                    {
                        "path": str(path),
                        "status": "PASS",
                        "case_id": record.result.case_id,
                        "run_status": record.result.run_status.value,
                        "scientific_metrics_present": record.result.metrics is not None,
                        "eligible_for_scientific_aggregation": record.result.eligible_for_scientific_aggregation,
                    }
                )
            except Exception as exc:
                replay_rows.append(
                    {
                        "path": str(path),
                        "status": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        for path in _pending_record_paths(Path(existing_evaluation_root)):
            try:
                record = load_case_evaluation_record(path)
                if not isinstance(record, PendingCaseEvaluationRecord):
                    raise ValueError("record is not pending")
                frozen_case = frozen_cases.get(record.case_id)
                if frozen_case is None:
                    raise ValueError("pending record has no frozen case")
                route = route_pending_adjudication(
                    record,
                    frozen_case.case_input,
                    frozen_case.case_metadata,
                )
                packet = route["packet"]
                forbidden = set(packet["forbidden_fields"])
                leaked = sorted(forbidden.intersection(packet))
                if leaked:
                    raise ValueError(
                        "pending continuation packet leaks forbidden fields: "
                        + ", ".join(leaked)
                    )
                pending_rows.append(
                    {
                        "path": str(path),
                        "status": "PASS",
                        "case_id": record.case_id,
                        "pending_type": record.pending_type,
                        "next_action": packet["next_action"],
                        "pending_owner": route["pending_owner"],
                        "scientific_score_blocked": route["scientific_score_blocked"],
                    }
                )
            except Exception as exc:
                pending_rows.append(
                    {
                        "path": str(path),
                        "status": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if any(row["status"] != "PASS" for row in pending_rows):
            blockers.append("PENDING_CONTINUATION_ROUTING_FAILED")
        if any(row["status"] != "PASS" for row in replay_rows):
            blockers.append("DETERMINISTIC_METRIC_REPLAY_FAILED")
        scientific_rows = [
            row
            for row in replay_rows
            if row.get("status") == "PASS"
            and row.get("scientific_metrics_present") is True
            and row.get("eligible_for_scientific_aggregation") is True
        ]
        checks["deterministic_metric_replay"] = {
            "status": "PASS" if replay_rows and not any(row["status"] != "PASS" for row in replay_rows) else "NOT_AVAILABLE" if not replay_rows else "FAIL",
            "terminal_record_count": len(replay_rows),
            "replayed_record_count": sum(row["status"] == "PASS" for row in replay_rows),
            "scientifically_finalized_record_count": len(scientific_rows),
            "rows": replay_rows,
        }
        checks["pending_continuation_routing"] = {
            "status": (
                "PASS"
                if pending_rows and all(row["status"] == "PASS" for row in pending_rows)
                else "NOT_AVAILABLE"
                if not pending_rows
                else "FAIL"
            ),
            "pending_record_count": len(pending_rows),
            "routable_pending_record_count": sum(
                row["status"] == "PASS" for row in pending_rows
            ),
            "rows": pending_rows,
        }
    else:
        checks["deterministic_metric_replay"] = {"status": "NOT_REQUESTED"}
        checks["pending_continuation_routing"] = {"status": "NOT_REQUESTED"}

    reference_ready = checks["reference_portfolio"].get("status") == "PASS"
    reuse_ready = checks["saved_response_reuse"].get("status") in {
        "REUSABLE",
        "NOT_REQUESTED",
    }
    replay_status = checks["deterministic_metric_replay"].get("status")
    replay_ready = replay_status in {"PASS", "NOT_REQUESTED", "NOT_AVAILABLE"}
    pending_route_ready = checks["pending_continuation_routing"].get("status") in {
        "PASS",
        "NOT_REQUESTED",
        "NOT_AVAILABLE",
    }
    cases_ready = checks["evaluation_case_consumability"].get("status") == "PASS"
    implementation_ready = (
        reference_ready
        and cases_ready
        and reuse_ready
        and replay_ready
        and pending_route_ready
        and not blockers
    )
    scientific_finalized = checks["deterministic_metric_replay"].get(
        "scientifically_finalized_record_count", 0
    )
    complete_case_count = len(
        {
            str(row.get("case_id"))
            for row in checks["deterministic_metric_replay"].get("rows", ())
            if row.get("status") == "PASS"
            and row.get("scientific_metrics_present") is True
            and row.get("eligible_for_scientific_aggregation") is True
            and row.get("case_id")
        }
    )
    expected_case_count = int(
        checks["evaluation_case_consumability"].get("case_count", 0)
    )
    return {
        "record_type": "OfflineEvaluationReadinessAudit",
        "schema_version": READINESS_AUDIT_VERSION,
        "no_model_calls": True,
        "status": "PASS" if implementation_ready else "BLOCKED",
        "scientific_metrics_computable": implementation_ready,
        "current_metrics_available": scientific_finalized > 0,
        "current_metrics_are_complete_28_case_results": (
            expected_case_count == 28 and complete_case_count == 28
        ),
        "checks": checks,
        "blockers": blockers,
        "notes": [
            "Computable means finalized scientific judgments can be scored deterministically.",
            "Pending adjudications are not converted to zero.",
            "Existing infrastructure-invalid records are not scientific results.",
            "Routable pending records remain unscored until their owner completes adjudication.",
            "This audit does not claim curator confirmation or formal release readiness.",
        ],
    }


def save_offline_evaluation_readiness(
    report: Mapping[str, Any], path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_fresh_n1_model_test_readiness(
    repository_root: str | Path,
    *,
    reference_manifest: str | Path,
    n1_manifest: str | Path,
    evaluator_manifest: str | Path | None = None,
    pending_continuation_registry: PendingContinuationRegistry | None = None,
    required_test_command: Sequence[str] | None = None,
    test_report: str | Path | Mapping[str, Any] | None = None,
    pending_owner_manifest: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the no-model gate immediately before a fresh N=1 test.

    This is stricter than the historical replay audit: the N=1 manifest must
    be explicitly bound to the selected immutable reference snapshot, all case
    digests must match, and the evaluator identity must be present.  The
    function never creates a provider client or invokes a model.  The current
    authority is v7; v4/v5/v6 remain accepted only for compatibility replay.
    """

    root = Path(repository_root).resolve()
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    reference_path = Path(reference_manifest).resolve()
    reference_schema: str | None = None
    try:
        frozen = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_schema = str(frozen.get("schema_version", ""))
        if reference_schema == REFERENCE_FREEZE_V6_SCHEMA_VERSION:
            checks["reference_portfolio_v6"] = verify_reference_portfolio_freeze_v6(root, reference_path)
        elif reference_schema == REFERENCE_FREEZE_V7_SCHEMA_VERSION:
            checks["reference_portfolio_v7"] = verify_reference_portfolio_freeze_v7(root, reference_path)
        elif reference_schema == REFERENCE_FREEZE_V5_SCHEMA_VERSION:
            checks["reference_portfolio_v5"] = verify_reference_portfolio_freeze_v5(root, reference_path)
        elif reference_schema == REFERENCE_FREEZE_V4_SCHEMA_VERSION:
            checks["reference_portfolio_v4"] = verify_reference_portfolio_freeze_v4(root, reference_path)
        else:
            raise ValueError("reference manifest is neither v4, v5, v6, nor v7")
    except Exception as exc:
        checks["reference_portfolio"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        blockers.append("REFERENCE_PORTFOLIO_INVALID")

    n1_path = Path(n1_manifest).resolve()
    try:
        n1 = json.loads(n1_path.read_text(encoding="utf-8"))
        rows = n1.get("cases")
        if (
            n1.get("manifest_version") != "full-dataset-n1-v1"
            or n1.get("N") != 1
            or n1.get("formal") is not False
            or n1.get("case_count") != 28
            or not isinstance(rows, list)
            or len(rows) != 28
            or n1.get("case_artifact_authority") != {
                REFERENCE_FREEZE_V4_SCHEMA_VERSION: "IMMUTABLE_REFERENCE_V4",
                REFERENCE_FREEZE_V5_SCHEMA_VERSION: "IMMUTABLE_REFERENCE_V5",
                REFERENCE_FREEZE_V6_SCHEMA_VERSION: "IMMUTABLE_REFERENCE_V6",
                REFERENCE_FREEZE_V7_SCHEMA_VERSION: "IMMUTABLE_REFERENCE_V7",
            }.get(reference_schema)
            or n1.get("reference_schema_version") != reference_schema
        ):
            raise ValueError("N=1 manifest is not bound to the selected immutable reference")
        expected_reference_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        if n1.get("reference_manifest_sha256") != expected_reference_digest:
            raise ValueError("N=1 manifest reference digest does not match the selected reference")
        bad_rows: list[str] = []
        reference_root = reference_path.parent / "cases"
        for row in rows:
            case_id = str(row.get("case_id", ""))
            expected_authority = {
                REFERENCE_FREEZE_V4_SCHEMA_VERSION: "IMMUTABLE_REFERENCE_V4",
                REFERENCE_FREEZE_V5_SCHEMA_VERSION: "IMMUTABLE_REFERENCE_V5",
                REFERENCE_FREEZE_V6_SCHEMA_VERSION: "IMMUTABLE_REFERENCE_V6",
                REFERENCE_FREEZE_V7_SCHEMA_VERSION: "IMMUTABLE_REFERENCE_V7",
            }.get(reference_schema)
            if row.get("case_artifact_authority") != expected_authority:
                bad_rows.append(f"{case_id}:authority")
                continue
            raw_case_input = Path(str(row.get("case_input_path", "")))
            case_input = (raw_case_input if raw_case_input.is_absolute() else root / raw_case_input).resolve()
            # Historical v4/v5 manifests were archived without rewriting
            # their repository-relative paths.  Resolve that compatibility
            # alias only when the declared path is absent; current v7 paths
            # remain strictly bound to the selected immutable reference.
            if not case_input.is_file() and str(raw_case_input).startswith("artifacts/reference/"):
                archived = root / "artifacts/archive/reference" / Path(raw_case_input).relative_to(Path("artifacts/reference"))
                if archived.is_file():
                    case_input = archived.resolve()
            try:
                case_input.relative_to(reference_root)
            except ValueError:
                # The archived compatibility alias is accepted when it is
                # exactly the archive mirror of the declared v5/v4 root.
                if not (reference_schema in {"reference-science-baseline-v4", "reference-science-baseline-v5"} and "artifacts/archive/reference" in str(case_input)):
                    bad_rows.append(f"{case_id}:path")
                    continue
            if not case_input.is_file():
                bad_rows.append(f"{case_id}:missing")
                continue
            expected_hash = str(row.get("case_input_sha256", ""))
            if hashlib.sha256(case_input.read_bytes()).hexdigest() != expected_hash:
                bad_rows.append(f"{case_id}:digest")
        if bad_rows:
            raise ValueError("reference-bound N=1 rows invalid: " + ", ".join(bad_rows))
        checks["reference_bound_n1_manifest"] = {
            "status": "PASS",
            "case_count": 28,
            "manifest_path": str(n1_path),
            "manifest_sha256": hashlib.sha256(n1_path.read_bytes()).hexdigest(),
        }
    except Exception as exc:
        checks["reference_bound_n1_manifest"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        blockers.append("REFERENCE_BOUND_N1_MANIFEST_INVALID")

    if evaluator_manifest is None:
        checks["evaluator_manifest"] = {"status": "NOT_CONFIGURED"}
        blockers.append("EVALUATOR_MANIFEST_MISSING")
    else:
        evaluator_path = Path(evaluator_manifest).resolve()
        try:
            value = json.loads(evaluator_path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise ValueError("evaluator manifest must be an object")
            hashes = value.get("deterministic_contract_hashes")
            if not isinstance(hashes, Mapping) or not hashes.get("evaluator_contract_sha256"):
                raise ValueError("evaluator manifest lacks deterministic contract digest")
            _verify_offline_evaluator_manifest_content(value)
            if pending_continuation_registry is not None:
                expected_continuation = pending_continuation_registry.execution_manifest
                declared_continuation = value.get("continuation_execution_manifest")
                if not isinstance(declared_continuation, Mapping):
                    raise ValueError(
                        "evaluator manifest lacks the installed continuation execution manifest"
                    )
                if dict(declared_continuation) != expected_continuation:
                    raise ValueError(
                        "evaluator manifest continuation execution identity does not match installed handlers"
                    )
            checks["evaluator_manifest"] = {
                "status": "PASS",
                "path": str(evaluator_path),
                "sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
                "evaluator_contract_sha256": hashes["evaluator_contract_sha256"],
            }
        except Exception as exc:
            checks["evaluator_manifest"] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
            blockers.append("EVALUATOR_MANIFEST_INVALID")

    invalid_owner_bindings = sorted(
        key for key, owner in PENDING_OWNER_BY_TYPE.items() if not str(owner).strip()
    )
    installed_owners = (
        set(pending_continuation_registry.installed_owners)
        if pending_continuation_registry is not None
        else set()
    )
    missing_runtime_owners = sorted(REQUIRED_PENDING_OWNERS - installed_owners)
    checks["pending_owner_contract"] = {
        "status": "PASS" if not invalid_owner_bindings else "FAIL",
        "pending_type_count": len(PENDING_OWNER_BY_TYPE),
        "owners": sorted(set(PENDING_OWNER_BY_TYPE.values())),
        "materialization_first": True,
        "runtime_handlers": "NOT_REQUIRED_FOR_STATIC_GATE"
        if pending_continuation_registry is None
        else "INSTALLED",
        "installed_owners": sorted(installed_owners),
        "missing_runtime_owners": missing_runtime_owners,
    }
    if invalid_owner_bindings:
        blockers.append("PENDING_OWNER_CONTRACT_INVALID")
    if pending_owner_manifest is None:
        checks["pending_owner_manifest"] = {
            "status": "NOT_CONFIGURED",
            "implementation_declarations_only": True,
        }
    else:
        try:
            owner_manifest = (
                dict(pending_owner_manifest)
                if isinstance(pending_owner_manifest, Mapping)
                else load_pending_owner_manifest(pending_owner_manifest)
            )
            checks["pending_owner_manifest"] = {
                "status": "PASS",
                "owner_count": owner_manifest["owner_count"],
                "implementation_declarations_only": True,
                "path": None
                if isinstance(pending_owner_manifest, Mapping)
                else str(Path(pending_owner_manifest).resolve()),
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            checks["pending_owner_manifest"] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
            blockers.append("PENDING_OWNER_MANIFEST_INVALID")

    capability_issues = (
        []
        if pending_continuation_registry is None
        else list(pending_continuation_registry.production_capability_issues)
    )
    production_status = (
        "NOT_CONFIGURED"
        if pending_continuation_registry is None
        else "INCOMPLETE"
        if missing_runtime_owners
        else "PROTOCOL_ONLY"
        if capability_issues
        else "READY"
    )
    checks["production_continuation"] = {
        "status": production_status,
        "installed_owner_count": len(installed_owners),
        "required_owner_count": len(REQUIRED_PENDING_OWNERS),
        "missing_runtime_owners": missing_runtime_owners,
        "capability_issues": capability_issues,
        "scientific_judgments_not_inferred": True,
    }
    if pending_continuation_registry is not None:
        installed_manifest = pending_continuation_registry.execution_manifest
        manifest_status = (
            "PASS"
            if not missing_runtime_owners and not capability_issues
            else "INCOMPLETE"
            if missing_runtime_owners
            else "PROTOCOL_ONLY"
        )
        try:
            continuation_execution_manifest_digest(installed_manifest)
        except ValueError as exc:
            manifest_status = "FAIL"
            blockers.append("CONTINUATION_EXECUTION_MANIFEST_INVALID")
            checks["production_continuation"]["manifest_error"] = str(exc)
        checks["continuation_execution_manifest"] = {
            "status": manifest_status,
            "manifest_sha256": installed_manifest["manifest_sha256"],
            "owner_count": installed_manifest["owner_count"],
            "required_owner_count": installed_manifest["required_owner_count"],
            "capability_issues": list(capability_issues),
        }
    else:
        checks["continuation_execution_manifest"] = {
            "status": "NOT_CONFIGURED",
            "note": "No executable continuation registry was supplied to this no-model audit.",
        }

    # The command is only a static/no-model gate.  A caller may optionally
    # record the exact frozen test command without making it a model call.
    checks["no_model_test_gate"] = {
        "status": "PASS",
        "no_model_calls": True,
        "required_test_command": list(required_test_command or ("pytest", "-q")),
    }
    if test_report is None:
        checks["local_test_report"] = {
            "status": "NOT_PROVIDED",
            "required": False,
            "note": "Pass --test-report to bind this gate to a completed no-model test run.",
        }
    else:
        try:
            if isinstance(test_report, Mapping):
                report_value = dict(test_report)
                report_path = None
            else:
                report_path_obj = Path(test_report).resolve()
                report_value = json.loads(report_path_obj.read_text(encoding="utf-8"))
                report_path = str(report_path_obj)
            if not isinstance(report_value, Mapping):
                raise ValueError("test report must be an object")
            returncode = report_value.get("returncode", report_value.get("exit_code"))
            failed = report_value.get("failed_count", report_value.get("failed", 0))
            passed = report_value.get("passed_count", report_value.get("passed", 0))
            if returncode != 0 or int(failed) != 0 or int(passed) <= 0:
                raise ValueError("test report does not prove a successful test run")
            reported_code_version = report_value.get("benchmark_code_data_version")
            if reported_code_version is not None and str(reported_code_version) != code_data_version(root):
                raise ValueError("test report is stale relative to the current source tree")
            checks["local_test_report"] = {
                "status": "PASS",
                "path": report_path,
                "returncode": int(returncode),
                "passed_count": int(passed),
                "failed_count": int(failed),
                "benchmark_code_data_version": reported_code_version,
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            checks["local_test_report"] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
            blockers.append("LOCAL_TEST_REPORT_INVALID")
    status = "READY_FOR_N1_MODEL_TEST" if not blockers else "BLOCKED"
    production_status = checks["production_continuation"]["status"]
    # Keep collection readiness separate from metric-bearing evaluation.  A
    # frozen, locally validated input surface is sufficient to open the N=1
    # collection gate, but it is not a live scientific result and it does not
    # imply that pending continuation handlers or a model run exist.
    # Collection readiness and metric-bearing readiness are separate.  The
    # v5 path makes the remaining external dependency explicit: open O/F
    # routes need installed owner handlers before a live run can yield all
    # scientific metrics.  We do not pretend that a missing handler is a
    # model result, and we do not add it to the static collection blockers.
    if status != "READY_FOR_N1_MODEL_TEST":
        metric_status = "BLOCKED"
    elif reference_schema in {REFERENCE_FREEZE_V5_SCHEMA_VERSION, REFERENCE_FREEZE_V6_SCHEMA_VERSION, REFERENCE_FREEZE_V7_SCHEMA_VERSION} and (
        missing_runtime_owners
        or checks["production_continuation"].get("capability_issues")
    ):
        metric_status = "BLOCKED_PRODUCTION_CONTINUATION"
    else:
        metric_status = "PENDING_MODEL_RUN"
    return {
        "record_type": "FreshN1ModelTestReadiness",
        "schema_version": FRESH_N1_READINESS_VERSION,
        "status": status,
        "collection_readiness_status": status,
        "metric_bearing_evaluation_status": metric_status,
        "metric_bearing_evaluation_blockers": (
            [
                "PENDING_CONTINUATION_HANDLERS_MISSING"
                if missing_runtime_owners
                else "PENDING_CONTINUATION_CAPABILITY_INSUFFICIENT"
            ]
            if metric_status == "BLOCKED_PRODUCTION_CONTINUATION"
            else []
        ),
        "production_continuation_status": production_status,
        "no_model_calls": True,
        "scientific_metrics_computable": not blockers,
        # Keep deterministic scorer computability separate from whether the
        # live evaluator can close every pending scientific route.  The latter
        # is false when owner handlers are absent, even though a finalized
        # record would still be scoreable offline.
        "metric_bearing_scientific_metrics_computable": (
            metric_status not in {"BLOCKED", "BLOCKED_PRODUCTION_CONTINUATION"}
        ),
        "model_test_not_run": True,
        "reference_schema_version": reference_schema,
        "entrypoint": {
            "script": "scripts/run_full_dataset_n1_pilot.py",
            "manifest": (
                str(n1_path.relative_to(root))
                if n1_path.is_relative_to(root)
                else str(n1_path)
            ),
            "model_calls_executed": False,
            "outputs": [
                "preflight_report.json",
                "collection_state.json",
                "runs/*/run_record.json",
                "runs/*/trajectory.jsonl",
                "pilot_manifest.json",
            ],
        },
        "checks": checks,
        "blockers": blockers,
        "notes": [
            "This gate proves the evaluator can consume frozen N=1 inputs; it is not a model result.",
            "Pending evaluations remain explicit and are never converted to zero.",
            "Installed protocol fixtures do not satisfy production continuation readiness.",
        ],
    }


__all__ = [
    "READINESS_AUDIT_VERSION",
    "FRESH_N1_READINESS_VERSION",
    "audit_offline_evaluation_readiness",
    "audit_fresh_n1_model_test_readiness",
    "save_offline_evaluation_readiness",
]

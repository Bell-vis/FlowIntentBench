"""Benchmark release membership and structural artifact checks.

Release membership is deliberately separate from case construction metadata and
from model-facing case input.  This module does not decide whether a case is
scientifically publishable; those gates remain curator review decisions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReleaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BenchmarkReleaseCase(ReleaseModel):
    case_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    case_family_id: str = Field(min_length=1)
    case_input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ground_truth_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    semantic_contract_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    scientific_evaluation_contract_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    scientific_artifact_binding_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def validate_semantic_binding_fields(self) -> "BenchmarkReleaseCase":
        values = (
            self.semantic_contract_sha256,
            self.scientific_evaluation_contract_sha256,
            self.scientific_artifact_binding_sha256,
        )
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError(
                "semantic-bound release cases require semantic contract, scientific "
                "evaluation contract, and scientific artifact binding digests together"
            )
        return self


class BenchmarkReleaseManifest(ReleaseModel):
    """The complete, explicit set of cases in one immutable benchmark release."""

    release_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    cases: list[BenchmarkReleaseCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_membership(self) -> "BenchmarkReleaseManifest":
        case_ids = [item.case_id for item in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("release cases must have globally unique case_id values")
        return self

    def case_for_id(self, case_id: str) -> BenchmarkReleaseCase:
        matches = [item for item in self.cases if item.case_id == case_id]
        if not matches:
            raise ValueError(f"case {case_id!r} is not in release {self.release_id!r}")
        if len(matches) > 1:
            raise ValueError(
                f"case_id {case_id!r} is ambiguous in release; use unique case IDs"
            )
        return matches[0]

    def contains(self, dataset_id: str, case_id: str) -> bool:
        return any(item.dataset_id == dataset_id and item.case_id == case_id for item in self.cases)


def canonical_release_json(manifest: BenchmarkReleaseManifest | Mapping[str, Any]) -> bytes:
    """Return canonical bytes used for release identity and reproducibility."""

    validated = manifest if isinstance(manifest, BenchmarkReleaseManifest) else BenchmarkReleaseManifest.model_validate(manifest)
    return json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def release_manifest_digest(manifest: BenchmarkReleaseManifest | Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_release_json(manifest)).hexdigest()


def load_release_manifest(path: str | Path) -> BenchmarkReleaseManifest:
    source = Path(path)
    try:
        return BenchmarkReleaseManifest.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load benchmark release manifest: {source}") from exc


def save_release_manifest(
    manifest: BenchmarkReleaseManifest | Mapping[str, Any],
    path: str | Path,
    *,
    refuse_overwrite: bool = True,
) -> BenchmarkReleaseManifest:
    validated = manifest if isinstance(manifest, BenchmarkReleaseManifest) else BenchmarkReleaseManifest.model_validate(manifest)
    destination = Path(path)
    if refuse_overwrite and destination.exists():
        raise FileExistsError(f"refusing to overwrite release manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(validated.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return validated


def resolve_case_artifact(case: BenchmarkReleaseCase, datasets_root: str | Path) -> dict[str, Path]:
    case_dir = Path(datasets_root) / case.dataset_id / "construction" / "cases" / case.case_id
    return {
        "case_input": case_dir / "case_input.json",
        "ground_truth": case_dir / "ground_truth.json",
        "semantic_contract": case_dir / "scientific_semantic_projection.json",
        "scientific_evaluation_contract": case_dir
        / "scientific_evaluation_contract.json",
        "scientific_artifact_binding": case_dir
        / "scientific_artifact_binding.json",
        "presentation": case_dir / "scientific_question.json",
    }


def _validate_semantic_bound_release_artifacts(
    case: BenchmarkReleaseCase,
    artifacts: Mapping[str, Path],
) -> None:
    if case.semantic_contract_sha256 is None:
        return
    from .scientific_artifact_binding import (
        ScientificArtifactBinding,
        artifact_sha256,
        require_formal_scientific_artifact_binding,
    )

    required = (
        "semantic_contract",
        "scientific_evaluation_contract",
        "scientific_artifact_binding",
        "presentation",
        "ground_truth",
    )
    for name in required:
        if not artifacts[name].is_file():
            raise FileNotFoundError(artifacts[name])
    try:
        semantic_contract = json.loads(
            artifacts["semantic_contract"].read_text(encoding="utf-8")
        )
        scientific_evaluation_contract = json.loads(
            artifacts["scientific_evaluation_contract"].read_text(encoding="utf-8")
        )
        binding = ScientificArtifactBinding.from_mapping(
            json.loads(
                artifacts["scientific_artifact_binding"].read_text(
                    encoding="utf-8"
                )
            )
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"STALE_SCIENTIFIC_ARTIFACT: invalid semantic binding for {case.case_id}"
        ) from exc
    actual_semantic = artifact_sha256(semantic_contract)
    actual_contract = hashlib.sha256(
        artifacts["scientific_evaluation_contract"].read_bytes()
    ).hexdigest()
    actual_binding = hashlib.sha256(
        artifacts["scientific_artifact_binding"].read_bytes()
    ).hexdigest()
    if actual_semantic != case.semantic_contract_sha256:
        raise ValueError(
            f"STALE_SCIENTIFIC_ARTIFACT: semantic contract digest mismatch for {case.case_id}"
        )
    if actual_contract != case.scientific_evaluation_contract_sha256:
        raise ValueError(
            "STALE_SCIENTIFIC_ARTIFACT: scientific evaluation contract digest "
            f"mismatch for {case.case_id}"
        )
    if actual_binding != case.scientific_artifact_binding_sha256:
        raise ValueError(
            f"STALE_SCIENTIFIC_ARTIFACT: binding digest mismatch for {case.case_id}"
        )
    require_formal_scientific_artifact_binding(
        binding,
        case_id=case.case_id,
        semantic_contract=actual_semantic,
        ground_truth=artifacts["ground_truth"],
        scientific_evaluation_contract=scientific_evaluation_contract,
        presentation=artifacts["presentation"],
    )


def validate_release_artifacts(
    manifest: BenchmarkReleaseManifest | Mapping[str, Any],
    datasets_root: str | Path,
    *,
    require_artifacts: bool = True,
) -> BenchmarkReleaseManifest:
    """Check release membership artifacts and optional digests only.

    This is intentionally structural.  It does not run Ground Truth validators
    or make any scientific release decision.  Callers must obtain the explicit
    ``LifecycleState.RELEASE_ELIGIBLE`` decision (and any required closure
    report) separately before selecting a case in a formal manifest; membership
    here is only the manifest's explicit selection, never an inferred lifecycle
    promotion.
    """

    validated = manifest if isinstance(manifest, BenchmarkReleaseManifest) else BenchmarkReleaseManifest.model_validate(manifest)
    for case in validated.cases:
        artifacts = resolve_case_artifact(case, datasets_root)
        for name in ("case_input", "ground_truth"):
            path = artifacts[name]
            if not path.is_file():
                if require_artifacts:
                    raise FileNotFoundError(path)
                continue
            expected = case.case_input_sha256 if name == "case_input" else case.ground_truth_sha256
            if expected is not None:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"{name} digest mismatch for {case.dataset_id}/{case.case_id}")
        _validate_semantic_bound_release_artifacts(case, artifacts)
    return validated


def validate_formal_release_case(
    release_case: BenchmarkReleaseCase,
    loaded_case: Any,
    dataset_manifest: Any,
) -> None:
    """Validate one released case before any model/provider session starts.

    Ground Truth is treated as an opaque released artifact: only its bytes are
    hashed here, never parsed or used to guide scientific execution.
    """

    if not release_case.case_input_sha256 or not release_case.ground_truth_sha256:
        raise ValueError(
            f"formal release case {release_case.case_id!r} requires case_input_sha256 "
            "and ground_truth_sha256"
        )
    if dataset_manifest.dataset_id != release_case.dataset_id:
        raise ValueError(
            f"release dataset_id {release_case.dataset_id!r} does not match "
            f"DatasetManifest {dataset_manifest.dataset_id!r}"
        )
    artifacts = resolve_case_artifact(release_case, loaded_case.case_root)
    case_input_path = artifacts["case_input"]
    ground_truth_path = artifacts["ground_truth"]
    loaded_case_path = getattr(loaded_case, "case_path", None)
    if loaded_case_path is not None and loaded_case_path.resolve() != case_input_path.resolve():
        raise ValueError(
            f"loaded case path does not match released case artifact for {release_case.case_id}"
        )
    if not case_input_path.is_file():
        raise FileNotFoundError(case_input_path)
    if not ground_truth_path.is_file():
        raise FileNotFoundError(ground_truth_path)
    actual_case_input = hashlib.sha256(case_input_path.read_bytes()).hexdigest()
    if actual_case_input != release_case.case_input_sha256:
        raise ValueError(f"case_input digest mismatch for {release_case.case_id}")
    actual_ground_truth = hashlib.sha256(ground_truth_path.read_bytes()).hexdigest()
    if actual_ground_truth != release_case.ground_truth_sha256:
        raise ValueError(f"ground_truth digest mismatch for {release_case.case_id}")
    _validate_semantic_bound_release_artifacts(release_case, artifacts)
    dataset_manifest.validate_files_against_loaded_case(loaded_case)


def released_case_ids(manifest: BenchmarkReleaseManifest | Mapping[str, Any]) -> tuple[str, ...]:
    validated = manifest if isinstance(manifest, BenchmarkReleaseManifest) else BenchmarkReleaseManifest.model_validate(manifest)
    return tuple(item.case_id for item in validated.cases)


def build_release_portfolio_summary(
    manifest: BenchmarkReleaseManifest | Mapping[str, Any],
    *,
    metadata_by_case_id: Mapping[str, Any] | None = None,
    datasets_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Produce a curator-facing capability summary without applying quotas."""

    validated = manifest if isinstance(manifest, BenchmarkReleaseManifest) else BenchmarkReleaseManifest.model_validate(manifest)
    metadata_by_case_id = dict(metadata_by_case_id or {})
    if datasets_root is not None:
        from .case_design import CaseConstructionMetadata

        for case in validated.cases:
            if case.case_id in metadata_by_case_id:
                continue
            metadata_path = (
                Path(datasets_root)
                / case.dataset_id
                / "construction"
                / "cases"
                / case.case_id
                / "case_construction_metadata.json"
            )
            if metadata_path.is_file():
                metadata_by_case_id[case.case_id] = CaseConstructionMetadata.model_validate_json(
                    metadata_path.read_text(encoding="utf-8")
                )
    summary: list[dict[str, Any]] = []
    for case in validated.cases:
        metadata = metadata_by_case_id.get(case.case_id)
        get = (lambda name, default=None: getattr(metadata, name, default)) if metadata is not None else (lambda name, default=None: default)
        case_type = case.case_family_id
        if metadata is not None:
            try:
                from .case_design import primary_case_type

                case_type = primary_case_type(metadata) or case_type
            except (ImportError, TypeError, AttributeError):
                case_type = get("case_family_id", case_type)
        summary.append(
            {
                "dataset_id": case.dataset_id,
                "case_id": case.case_id,
                "case_family_id": case.case_family_id,
                "case_type": case_type,
                "scientific_target": get("scientific_target"),
                "principal_operationalization_dimensions": [
                    getattr(item, "value", item)
                    for item in (get("principal_operationalization_dimensions", []) or [])
                ],
                "unresolved_operationalization_dimensions": [
                    getattr(item, "value", item)
                    for item in (get("unresolved_operationalization_dimensions", []) or [])
                ],
                "finding_categories": [
                    getattr(getattr(item, "category", item), "value", getattr(item, "category", item))
                    for item in (get("explicit_finding_requirements", []) or [])
                ],
            }
        )
    return summary


__all__ = [
    "BenchmarkReleaseCase",
    "BenchmarkReleaseManifest",
    "build_release_portfolio_summary",
    "canonical_release_json",
    "load_release_manifest",
    "release_manifest_digest",
    "released_case_ids",
    "resolve_case_artifact",
    "save_release_manifest",
    "validate_formal_release_case",
    "validate_release_artifacts",
]

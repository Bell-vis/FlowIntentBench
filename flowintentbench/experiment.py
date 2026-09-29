"""Immutable experiment manifests for formal model evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


FORMAL_TRIAL_COUNT = 3


class ExperimentModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        protected_namespaces=(),
    )


class EvaluatedModelConfiguration(ExperimentModel):
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    target_fingerprint: str = Field(pattern=r"^[0-9a-f]{16}$")


class ExperimentManifest(ExperimentModel):
    """All non-scientific inputs needed to reproduce one formal run."""

    experiment_id: str = Field(min_length=1)
    benchmark_release_id: str = Field(min_length=1)
    benchmark_release_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluated_model_configuration: EvaluatedModelConfiguration
    system_prompt_version: str = Field(min_length=1)
    case_presentation_version: str = Field(min_length=1)
    python_tool_spec_version: str = Field(min_length=1)
    runner_version: str = Field(min_length=1)
    runtime_environment_fingerprint: dict[str, Any]
    formal_trial_count: int = Field(default=FORMAL_TRIAL_COUNT, ge=1)
    trial_seeds: list[int] = Field(min_length=1)
    generated_case_order: list[list[str]] = Field(min_length=1)
    benchmark_code_data_version: str = Field(min_length=1)
    # Optional only so historical manifests remain readable. New formal runs
    # are admitted exclusively through the profile-aware runner and populate all fields.
    protocol_version: str | None = None
    agent_id: str | None = None
    runtime_profile_id: str | None = None
    agent_profile_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime_profile_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    system_instruction_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tool_schema_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_trial_count_and_identity(self) -> "ExperimentManifest":
        if self.formal_trial_count != FORMAL_TRIAL_COUNT:
            raise ValueError(
                f"formal_trial_count is frozen at {FORMAL_TRIAL_COUNT}"
            )
        if len(self.trial_seeds) != self.formal_trial_count:
            raise ValueError("trial_seeds must contain exactly formal_trial_count entries")
        if len(self.generated_case_order) != self.formal_trial_count:
            raise ValueError("generated_case_order must contain exactly formal_trial_count entries")
        if len(set(self.trial_seeds)) != len(self.trial_seeds):
            raise ValueError("trial_seeds must be unique")
        if any(not order or len(order) != len(set(order)) for order in self.generated_case_order):
            raise ValueError("each generated case order must contain unique non-empty case IDs")
        expected_cases = set(self.generated_case_order[0])
        if any(set(order) != expected_cases for order in self.generated_case_order[1:]):
            raise ValueError("generated case orders must contain the same selected case set")
        return self


def canonical_experiment_json(manifest: ExperimentManifest | Mapping[str, Any]) -> bytes:
    validated = manifest if isinstance(manifest, ExperimentManifest) else ExperimentManifest.model_validate(manifest)
    return json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def experiment_manifest_digest(manifest: ExperimentManifest | Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_experiment_json(manifest)).hexdigest()


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    source = Path(path)
    try:
        return ExperimentManifest.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load experiment manifest: {source}") from exc


def save_experiment_manifest(
    manifest: ExperimentManifest | Mapping[str, Any],
    path: str | Path,
    *,
    refuse_overwrite: bool = True,
) -> ExperimentManifest:
    validated = manifest if isinstance(manifest, ExperimentManifest) else ExperimentManifest.model_validate(manifest)
    destination = Path(path)
    if refuse_overwrite and destination.exists():
        raise FileExistsError(f"refusing to overwrite experiment manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(validated.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return validated


def code_data_version(root: str | Path = ".") -> str:
    """Return a stable repository/data identity without machine telemetry.

    A Git commit is preferred.  Source-tree hashing is the deterministic
    fallback used by source distributions and workspaces without ``.git``.
    """

    root_path = Path(root).resolve()
    def git_command(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root_path), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout

    commit = git_command("rev-parse", "HEAD")
    if commit and commit.strip():
        status = git_command("status", "--porcelain", "--untracked-files=all")
        if status is not None and not status.strip():
            return "git:" + commit.strip()
        return "git:" + commit.strip() + "+dirty:" + _source_tree_digest(root_path)
    return "tree:" + _source_tree_digest(root_path)


def _source_tree_digest(root_path: Path) -> str:
    """Hash source/config bytes in a path-stable order."""

    digest = hashlib.sha256()
    relative_paths: set[Path] = set()
    for pattern in ("flowintentbench/**/*.py", "scripts/**/*.py", "tests/**/*.py"):
        relative_paths.update(
            path.relative_to(root_path)
            for path in root_path.glob(pattern)
            if path.is_file()
        )
    for name in ("pyproject.toml", "runtime-requirements.txt", "method.md", "FlowIntentBench.md"):
        candidate = root_path / name
        if candidate.is_file():
            relative_paths.add(candidate.relative_to(root_path))
    for relative in sorted(relative_paths, key=lambda value: value.as_posix()):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((root_path / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "FORMAL_TRIAL_COUNT",
    "EvaluatedModelConfiguration",
    "ExperimentManifest",
    "canonical_experiment_json",
    "code_data_version",
    "experiment_manifest_digest",
    "load_experiment_manifest",
    "save_experiment_manifest",
]

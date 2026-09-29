"""Load and resolve a model-facing benchmark case."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import BenchmarkCaseInput, DataFile, GeometryAsset


class CaseLoadError(ValueError):
    """Raised when a syntactically valid case cannot be resolved in its root."""


@dataclass(frozen=True)
class ResolvedDataFile:
    specification: DataFile
    path: Path


@dataclass(frozen=True)
class ResolvedGeometryAsset:
    specification: GeometryAsset
    path: Path


@dataclass(frozen=True)
class LoadedCase:
    case: BenchmarkCaseInput
    case_root: Path
    data_files: tuple[ResolvedDataFile, ...]
    geometry_assets: tuple[ResolvedGeometryAsset, ...]
    case_path: Path | None = None


class CaseLoader:
    """Parse a case and optionally verify all referenced files exist."""

    def __init__(self, case_root: str | Path, *, verify_files: bool = True) -> None:
        self.case_root = Path(case_root).expanduser().resolve()
        self.verify_files = verify_files

    def load(self, case_file: str | Path) -> LoadedCase:
        path = Path(case_file)
        if not path.is_absolute():
            path = self.case_root / path
        path = path.resolve()
        if not path.is_file():
            raise CaseLoadError(f"case file does not exist: {path}")
        if path.suffix.casefold() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.casefold() in {".yaml", ".yml"}:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise CaseLoadError("case files must use .json, .yaml, or .yml")
        loaded = self.load_mapping(payload)
        return LoadedCase(
            loaded.case,
            loaded.case_root,
            loaded.data_files,
            loaded.geometry_assets,
            case_path=path,
        )

    def load_mapping(self, payload: Any) -> LoadedCase:
        if not isinstance(payload, dict):
            raise CaseLoadError("case input must be a mapping")
        case = BenchmarkCaseInput.model_validate(payload)
        data_files = tuple(
            ResolvedDataFile(item, self._resolve(item.path, "data file"))
            for item in case.flow_data.data_files
        )
        geometry_assets = tuple(
            ResolvedGeometryAsset(item, self._resolve(item.path, "geometry asset"))
            for item in case.case_context.geometry_assets
        )
        data_paths = {item.specification.path for item in data_files}
        geometry_paths = {item.specification.path for item in geometry_assets}
        overlap = data_paths & geometry_paths
        if overlap:
            raise CaseLoadError(f"geometry assets must not also be data files: {sorted(overlap)}")
        return LoadedCase(case, self.case_root, data_files, geometry_assets)

    def _resolve(self, relative_path: str, kind: str) -> Path:
        candidate = (self.case_root / relative_path).resolve()
        try:
            candidate.relative_to(self.case_root)
        except ValueError as exc:
            raise CaseLoadError(f"{kind} escapes the case root: {relative_path}") from exc
        if self.verify_files and not candidate.is_file():
            raise CaseLoadError(f"{kind} does not exist: {relative_path}")
        return candidate

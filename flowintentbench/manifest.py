"""Dataset-level resources kept outside the model-facing input contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schema import DataFileRole, FormatType, _relative_path


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ManifestResource(ManifestModel):
    size_bytes: int = Field(ge=0)
    checksum: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    checksum_algorithm: str = "sha256"
    source_url: str | None = None

    @field_validator("checksum_algorithm")
    @classmethod
    def validate_algorithm(cls, value: str) -> str:
        if value.casefold() != "sha256":
            raise ValueError("only sha256 is supported")
        return "sha256"


class ManifestFile(ManifestResource):
    path: str
    role: DataFileRole

    _validate_path = field_validator("path")(_relative_path)


class AuxiliaryAsset(ManifestResource):
    path: str
    role: Literal["geometry"]

    _validate_path = field_validator("path")(_relative_path)


class VariableMapping(ManifestModel):
    output_name: str = Field(min_length=1)
    source_file_role: DataFileRole
    convention: str | None = None
    raw_components: dict[str, str] = Field(default_factory=dict)
    calculation: str | None = None


class GridBlankingMapping(ManifestModel):
    name: str = Field(min_length=1)
    type: Literal["scalar", "vector", "tensor"]
    components: int = Field(ge=1)
    association: Literal["point", "cell"]
    source: Literal["reader_derived"]
    # Explicit validity semantics are model-visible through reader_metadata,
    # but excluded from legacy manifest projections that compare only the
    # reader array identity.
    validity_rule: str | None = Field(default=None, exclude=True)


class ReaderConfiguration(ManifestModel):
    format: FormatType
    canonical_reader: str = Field(min_length=1)
    reader_configuration: dict[str, Any] = Field(default_factory=dict)
    plot3d_convention: str | None = None
    variable_mappings: list[VariableMapping] = Field(default_factory=list)
    grid_blanking: list[GridBlankingMapping] = Field(default_factory=list)


class DatasetManifest(ManifestModel):
    dataset_id: str = Field(min_length=1)
    file_root: str = Field(default="datasets", min_length=1)
    files: list[ManifestFile] = Field(min_length=1)
    auxiliary_assets: list[AuxiliaryAsset] = Field(default_factory=list)
    reader: ReaderConfiguration
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> "DatasetManifest":
        paths = [item.path for item in self.files]
        paths.extend(item.path for item in self.auxiliary_assets)
        if len(paths) != len(set(paths)):
            raise ValueError("manifest file paths must be unique")
        if self.reader.format == "PLOT3D" and not self.reader.plot3d_convention:
            raise ValueError("PLOT3D manifests require plot3d_convention")
        return self

    def validate_against_case(self, case: Any) -> None:
        """Ensure internal resources describe exactly the files exposed to a case."""
        case_paths = {(item.path, item.role) for item in case.flow_data.data_files}
        manifest_paths = {(item.path, item.role) for item in self.files}
        if case_paths != manifest_paths:
            raise ValueError("DatasetManifest files do not match BenchmarkCaseInput data_files")
        if self.reader.format != case.flow_data.data_metadata.format.type:
            raise ValueError("manifest reader format does not match case format")

    def validate_files_against_loaded_case(self, loaded_case: Any) -> None:
        """Validate manifest sizes/checksums against already resolved case files."""

        self.validate_against_case(loaded_case.case)
        resolved: dict[str, Path] = {}
        for item in (*loaded_case.data_files, *loaded_case.geometry_assets):
            resolved[item.specification.path] = item.path
        resources = [*self.files, *self.auxiliary_assets]
        for resource in resources:
            path = resolved.get(resource.path)
            if path is None:
                raise ValueError(f"manifest resource is not resolved by case: {resource.path}")
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != resource.size_bytes:
                raise ValueError(f"manifest size mismatch for {resource.path}")
            actual = sha256_file(path)
            if actual.casefold() != resource.checksum.casefold():
                raise ValueError(f"manifest checksum mismatch for {resource.path}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file checksum without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_file(path: Path, relative_path: str, role: DataFileRole, source_url: str | None = None) -> ManifestFile:
    """Create a manifest file record from an existing file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return ManifestFile(
        path=relative_path,
        role=role,
        size_bytes=path.stat().st_size,
        checksum=sha256_file(path),
        source_url=source_url,
    )


def manifest_auxiliary_asset(
    path: Path,
    relative_path: str,
    role: Literal["geometry"],
    source_url: str | None = None,
) -> AuxiliaryAsset:
    """Create an internal auxiliary asset record from an existing file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return AuxiliaryAsset(
        path=relative_path,
        role=role,
        size_bytes=path.stat().st_size,
        checksum=sha256_file(path),
        source_url=source_url,
    )

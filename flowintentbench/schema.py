"""Strict model-facing input contract for FlowIntentBench.

This module intentionally contains no provenance, checksum, source, or ground
truth fields. Those belong to :mod:`flowintentbench.manifest`.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DataFileRole = Literal["grid", "solution", "flow_field"]
FormatType = Literal["VTK", "VTU", "PLOT3D", "RAW"]
GridType = Literal["structured", "unstructured", "rectilinear", "other"]
VariableType = Literal["scalar", "vector", "tensor"]
Association = Literal["point", "cell"]
VariableSource = Literal["stored", "reader_derived"]

_FORBIDDEN_READER_DERIVED = {
    "q",
    "q-criterion",
    "q criterion",
    "lambda-2",
    "lambda2",
    "vorticity",
    "swirling strength",
}


class ContractModel(BaseModel):
    """Base model with a closed schema at every contract boundary."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    if "\x00" in value:
        raise ValueError("path must not contain NUL")
    # Check both POSIX and Windows syntax even when validation runs on Linux.
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError("path must be relative to the case root")
    if PureWindowsPath(value).drive:
        raise ValueError("path must not contain a drive prefix")
    parts = value.replace("\\", "/").split("/")
    if any(part in {"", ".."} for part in parts):
        raise ValueError("path must be normalized and must not escape the case root")
    return value


def _three_numbers(value: list[float] | None, field_name: str) -> list[float] | None:
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    return value


class DataFile(ContractModel):
    path: str
    role: DataFileRole

    _validate_path = field_validator("path")(_relative_path)


class FormatMetadata(ContractModel):
    type: FormatType
    details: dict[str, Any] = Field(default_factory=dict)


class GridMetadata(ContractModel):
    type: GridType
    dimensions: list[int] | None = None
    origin: list[float] | None = None
    spacing: list[float] | None = None

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and (len(value) != 3 or any(item <= 0 for item in value)):
            raise ValueError("dimensions must contain three positive integers")
        return value

    @field_validator("origin", "spacing")
    @classmethod
    def validate_vectors(cls, value: list[float] | None, info: Any) -> list[float] | None:
        return _three_numbers(value, info.field_name)


class VariableMetadata(ContractModel):
    name: str = Field(min_length=1)
    physical_quantity: str | None = None
    type: VariableType
    components: int = Field(ge=1)
    association: Association
    unit: str | None = None
    source: VariableSource

    @model_validator(mode="after")
    def validate_source_semantics(self) -> "VariableMetadata":
        if self.source == "reader_derived" and self.physical_quantity:
            normalized = " ".join(self.physical_quantity.casefold().replace("_", " ").split())
            if normalized in _FORBIDDEN_READER_DERIVED:
                raise ValueError(
                    "analysis quantities such as Q-criterion, Lambda-2, vorticity, "
                    "and swirling strength must not be reader_derived"
                )
        return self


class CoordinateAxisMeaning(ContractModel):
    x: str | None = None
    y: str | None = None
    z: str | None = None


class CoordinateSystemMetadata(ContractModel):
    type: Literal["cartesian", "cylindrical", "other"]
    axis_meaning: CoordinateAxisMeaning
    unit: str | None = None


class TemporalMetadata(ContractModel):
    type: Literal["single_snapshot"]


class DataMetadata(ContractModel):
    format: FormatMetadata
    grid: GridMetadata
    variables: list[VariableMetadata] = Field(min_length=1)
    coordinate_system: CoordinateSystemMetadata
    temporal: TemporalMetadata

    @model_validator(mode="after")
    def validate_format_requirements(self) -> "DataMetadata":
        details = self.format.details
        if self.format.type == "RAW":
            required = {"dtype", "endianness", "memory_order", "header_bytes"}
            missing = sorted(key for key in required if key not in details)
            if missing:
                raise ValueError(f"RAW metadata is missing required fields: {', '.join(missing)}")
            if details["endianness"] not in {"little", "big"}:
                raise ValueError("RAW endianness must be 'little' or 'big'")
            if details["memory_order"] not in {"C", "Fortran"}:
                raise ValueError("RAW memory_order must be 'C' or 'Fortran'")
            if not isinstance(details["header_bytes"], int) or details["header_bytes"] < 0:
                raise ValueError("RAW header_bytes must be a non-negative integer")
            if self.grid.dimensions is None:
                raise ValueError("RAW grid.dimensions must be explicitly provided")
        elif self.format.type == "PLOT3D":
            required = {"encoding", "precision", "endianness", "has_byte_count"}
            missing = sorted(key for key in required if key not in details)
            if missing:
                raise ValueError(
                    f"PLOT3D metadata is missing required fields: {', '.join(missing)}"
                )
            if details.get("encoding") not in {"binary", "ascii"}:
                raise ValueError("PLOT3D encoding must be 'binary' or 'ascii'")
            if details["precision"] not in {None, "float32", "float64"}:
                raise ValueError("PLOT3D precision must be float32, float64, or null")
            if details["endianness"] not in {None, "little", "big"}:
                raise ValueError("PLOT3D endianness must be little, big, or null")
            if details["has_byte_count"] not in {None, True, False}:
                raise ValueError("PLOT3D has_byte_count must be true, false, or null")
        return self


class ObjectContext(ContractModel):
    name: str = Field(min_length=1)
    description: str | None = None


class BoundaryContext(ContractModel):
    name: str = Field(min_length=1)
    type: Literal["wall", "inlet", "outlet", "symmetry", "other"]
    description: str | None = None


class RegionContext(ContractModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ReferenceDirectionContext(ContractModel):
    name: str = Field(min_length=1)
    vector: list[float] | None = None
    description: str | None = None

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, value: list[float] | None) -> list[float] | None:
        value = _three_numbers(value, "vector")
        if value is not None and all(component == 0 for component in value):
            raise ValueError("vector must not be the zero vector")
        return value


class OperatingConditionContext(ContractModel):
    name: str = Field(min_length=1)
    value: float | int | str
    unit: str | None = None


class GeometryAsset(ContractModel):
    path: str
    format: str = Field(min_length=1)
    description: str | None = None

    _validate_path = field_validator("path")(_relative_path)


class CaseContext(ContractModel):
    physical_setting: str | None = None
    objects: list[ObjectContext] = Field(default_factory=list)
    boundaries: list[BoundaryContext] = Field(default_factory=list)
    regions: list[RegionContext] = Field(default_factory=list)
    reference_directions: list[ReferenceDirectionContext] = Field(default_factory=list)
    operating_conditions: list[OperatingConditionContext] = Field(default_factory=list)
    geometry_assets: list[GeometryAsset] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_context_identity(self) -> "CaseContext":
        named_collections = {
            "objects": self.objects,
            "boundaries": self.boundaries,
            "regions": self.regions,
            "reference_directions": self.reference_directions,
            "operating_conditions": self.operating_conditions,
        }
        for field_name, items in named_collections.items():
            names = [item.name for item in items]
            if len(names) != len(set(names)):
                raise ValueError(f"{field_name} names must be unique")
        paths = [item.path for item in self.geometry_assets]
        if len(paths) != len(set(paths)):
            raise ValueError("geometry_assets paths must be unique")
        return self


class FlowData(ContractModel):
    data_files: list[DataFile] = Field(min_length=1)
    data_metadata: DataMetadata

    @model_validator(mode="after")
    def validate_file_roles(self) -> "FlowData":
        paths = [item.path for item in self.data_files]
        if len(paths) != len(set(paths)):
            raise ValueError("data_files paths must be unique")
        if self.data_metadata.format.type == "PLOT3D":
            roles = {item.role for item in self.data_files}
            if not {"grid", "solution"}.issubset(roles):
                raise ValueError("PLOT3D cases require both grid and solution data files")
        return self


class BenchmarkCaseInput(ContractModel):
    scientific_question: str = Field(min_length=1)
    flow_data: FlowData
    case_context: CaseContext


def validate_model_input(payload: Any) -> BenchmarkCaseInput:
    """Validate a model-facing payload without accepting internal resources."""
    return BenchmarkCaseInput.model_validate(payload)


def model_input_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for the frozen model-facing contract."""
    return BenchmarkCaseInput.model_json_schema()

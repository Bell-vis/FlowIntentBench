"""Shared VTK reader configuration used by generation and execution."""

from __future__ import annotations

from typing import Any, Mapping


def canonical_vtk_reader_configuration(format_type: str) -> dict[str, bool]:
    """Return the frozen reader configuration for a VTK-family format."""
    if format_type == "VTK":
        return {"read_all_scalars": True, "read_all_vectors": True}
    if format_type == "VTU":
        # XML unstructured-grid reader has no corresponding legacy toggles.
        return {"read_all_scalars": False, "read_all_vectors": False}
    raise ValueError(f"unsupported VTK-family format: {format_type}")


def apply_legacy_vtk_reader_configuration(
    reader: Any, configuration: Mapping[str, Any] | None = None
) -> None:
    """Apply the canonical scalar/vector loading switches to a legacy reader."""
    config = dict(configuration or canonical_vtk_reader_configuration("VTK"))
    required = ("read_all_scalars", "read_all_vectors")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"VTK reader configuration is missing: {', '.join(missing)}")
    for key, on_method, off_method in (
        ("read_all_scalars", "ReadAllScalarsOn", "ReadAllScalarsOff"),
        ("read_all_vectors", "ReadAllVectorsOn", "ReadAllVectorsOff"),
    ):
        value = config[key]
        if not isinstance(value, bool):
            raise ValueError(f"VTK reader configuration {key} must be boolean")
        getattr(reader, on_method if value else off_method)()

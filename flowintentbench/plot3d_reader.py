"""Shared canonical PLOT3D reader configuration."""

from __future__ import annotations

from typing import Any, Mapping


PLOT3D_READER_NAME = "vtkMultiBlockPLOT3DReader"


def canonical_plot3d_reader_configuration() -> dict[str, Any]:
    """Return the frozen v1 configuration used by generation and execution."""
    return {
        "auto_detect_format": True,
        "scalar_function_number": 153,
        "vector_function_number": 200,
        "preserve_intermediate_functions": True,
        "require_single_block": True,
    }


def apply_plot3d_reader_configuration(
    reader: Any, configuration: Mapping[str, Any] | None = None
) -> bool:
    """Apply the v1 configuration and return its single-block requirement."""
    expected = canonical_plot3d_reader_configuration()
    config = dict(configuration or expected)
    missing = [key for key in expected if key not in config]
    if missing:
        raise ValueError(f"PLOT3D reader configuration is missing: {', '.join(missing)}")
    mismatches = [
        key for key, value in expected.items()
        if config[key] != value or type(config[key]) is not type(value)
    ]
    if mismatches:
        raise ValueError(
            "PLOT3D reader configuration does not match canonical v1 values: "
            + ", ".join(mismatches)
        )

    reader.AutoDetectFormatOn()
    reader.SetScalarFunctionNumber(expected["scalar_function_number"])
    reader.SetVectorFunctionNumber(expected["vector_function_number"])
    reader.PreserveIntermediateFunctionsOn()
    return expected["require_single_block"]

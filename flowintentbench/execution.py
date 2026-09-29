"""Format adapters for executing a loaded case.

The adapter exposes only reader-level data. It deliberately has no vortex,
Q-criterion, Lambda-2, or other case-specific analysis operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .loader import LoadedCase, ResolvedDataFile
from .manifest import DatasetManifest
from .plot3d_reader import (
    PLOT3D_READER_NAME,
    apply_plot3d_reader_configuration,
)
from .vtk_reader import apply_legacy_vtk_reader_configuration


class ReaderConfigurationError(RuntimeError):
    """Raised when a case has no deterministic reader configuration."""


@dataclass(frozen=True)
class ExecutionResult:
    format: str
    data: dict[str, Any]


Plot3DReader = Callable[[LoadedCase, DatasetManifest], ExecutionResult]


class ExecutionAdapter:
    """Dispatch a loaded case to a canonical format reader."""

    def __init__(self) -> None:
        self._plot3d_readers: dict[str, Plot3DReader] = {
            PLOT3D_READER_NAME: self._read_plot3d_vtk,
        }

    def register_plot3d_reader(self, name: str, reader: Plot3DReader) -> None:
        if not name.strip():
            raise ValueError("reader name must be non-empty")
        self._plot3d_readers[name] = reader

    def read(self, loaded_case: LoadedCase, manifest: DatasetManifest | None = None) -> ExecutionResult:
        case = loaded_case.case
        format_type = case.flow_data.data_metadata.format.type
        if manifest is not None:
            manifest.validate_against_case(case)
        if format_type == "RAW":
            return self._read_raw(loaded_case, manifest)
        if format_type == "VTK":
            return self._read_vtk(loaded_case, manifest)
        if format_type == "VTU":
            return self._read_vtu(loaded_case)
        if format_type == "PLOT3D":
            return self._read_plot3d(loaded_case, manifest)
        raise ReaderConfigurationError(f"unsupported format: {format_type}")

    def _read_raw(
        self, loaded_case: LoadedCase, manifest: DatasetManifest | None
    ) -> ExecutionResult:
        metadata = loaded_case.case.flow_data.data_metadata
        details = metadata.format.details
        dimensions = tuple(metadata.grid.dimensions or ())
        stored = [variable for variable in metadata.variables if variable.source == "stored"]
        if len(loaded_case.data_files) != 1 or len(stored) != 1:
            raise ReaderConfigurationError(
                "RAW adapter requires one data file and one stored variable; "
                "use an internal manifest mapping for multi-array RAW data"
            )
        variable = stored[0]
        if variable.components != 1:
            layout = (
                manifest.reader.reader_configuration.get("component_layout")
                if manifest is not None
                else None
            )
            if layout != "interleaved":
                raise ReaderConfigurationError(
                    "RAW vector layout is not self-describing; an internal manifest "
                    "must declare component_layout=interleaved"
                )
        dtype = np.dtype(details["dtype"])
        dtype = dtype.newbyteorder("<" if details["endianness"] == "little" else ">")
        expected_items = int(np.prod(dimensions)) * variable.components
        path = loaded_case.data_files[0].path
        expected_bytes = int(details["header_bytes"]) + expected_items * dtype.itemsize
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ReaderConfigurationError(
                f"RAW size mismatch for {path.name}: expected {expected_bytes}, got {actual_bytes}"
            )
        array = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=int(details["header_bytes"]),
            shape=(expected_items,),
        )
        if variable.components == 1:
            shape = dimensions
        else:
            shape = dimensions + (variable.components,)
        order = "C" if details["memory_order"] == "C" else "F"
        return ExecutionResult("RAW", {loaded_case.data_files[0].specification.path: array.reshape(shape, order=order)})

    def _read_vtk(
        self, loaded_case: LoadedCase, manifest: DatasetManifest | None
    ) -> ExecutionResult:
        item = self._single_flow_file(loaded_case)
        configuration = manifest.reader.reader_configuration if manifest is not None else None
        reader = self._make_vtk_reader(item.path, configuration)
        return ExecutionResult("VTK", {item.specification.path: reader.GetOutput()})

    def _read_vtu(self, loaded_case: LoadedCase) -> ExecutionResult:
        try:
            from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ReaderConfigurationError("VTK Python bindings are required for VTU files") from exc
        item = self._single_flow_file(loaded_case)
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(item.path))
        reader.Update()
        output = reader.GetOutput()
        if output is None:
            raise ReaderConfigurationError(f"VTU reader returned no dataset: {item.path}")
        return ExecutionResult("VTU", {item.specification.path: output})

    def _make_vtk_reader(
        self, path: Path, configuration: dict[str, Any] | None = None
    ) -> Any:
        try:
            from vtkmodules.vtkIOLegacy import vtkDataSetReader
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ReaderConfigurationError("VTK Python bindings are required for VTK files") from exc
        reader = vtkDataSetReader()
        try:
            apply_legacy_vtk_reader_configuration(reader, configuration)
        except (AttributeError, ValueError) as exc:
            raise ReaderConfigurationError(
                f"invalid VTK reader configuration for {path.name}: {exc}"
            ) from exc
        reader.SetFileName(str(path))
        reader.Update()
        if reader.GetOutput() is None:
            raise ReaderConfigurationError(f"VTK reader returned no dataset: {path}")
        return reader

    def _read_plot3d(
        self, loaded_case: LoadedCase, manifest: DatasetManifest | None
    ) -> ExecutionResult:
        if manifest is None:
            raise ReaderConfigurationError("PLOT3D execution requires an internal DatasetManifest")
        reader_name = manifest.reader.canonical_reader
        reader = self._plot3d_readers.get(reader_name)
        if reader is None:
            raise ReaderConfigurationError(f"no registered canonical PLOT3D reader: {reader_name}")
        return reader(loaded_case, manifest)

    def _read_plot3d_vtk(
        self, loaded_case: LoadedCase, manifest: DatasetManifest
    ) -> ExecutionResult:
        try:
            from vtkmodules.vtkIOParallel import vtkMultiBlockPLOT3DReader
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ReaderConfigurationError(
                "VTK Python bindings with vtkMultiBlockPLOT3DReader are required"
            ) from exc

        grid_files = [
            item for item in loaded_case.data_files
            if item.specification.role == "grid"
        ]
        solution_files = [
            item for item in loaded_case.data_files
            if item.specification.role == "solution"
        ]
        if len(grid_files) != 1 or len(solution_files) != 1:
            raise ReaderConfigurationError(
                "canonical PLOT3D execution requires exactly one grid and one solution file"
            )

        reader = vtkMultiBlockPLOT3DReader()
        try:
            require_single_block = apply_plot3d_reader_configuration(
                reader, manifest.reader.reader_configuration
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ReaderConfigurationError(f"invalid PLOT3D reader configuration: {exc}") from exc
        reader.SetXYZFileName(str(grid_files[0].path))
        reader.SetQFileName(str(solution_files[0].path))
        reader.Update()

        output = reader.GetOutput()
        if output is None:
            raise ReaderConfigurationError("vtkMultiBlockPLOT3DReader returned no output")
        block_count = output.GetNumberOfBlocks()
        if require_single_block and block_count != 1:
            raise ReaderConfigurationError(
                f"canonical PLOT3D execution requires exactly one block; got {block_count}"
            )
        if block_count == 0 or output.GetBlock(0) is None:
            raise ReaderConfigurationError("vtkMultiBlockPLOT3DReader returned an empty output")

        # The Q/solution path identifies the combined PLOT3D reader output.
        return ExecutionResult(
            "PLOT3D",
            {solution_files[0].specification.path: output},
        )

    @staticmethod
    def _single_flow_file(loaded_case: LoadedCase) -> ResolvedDataFile:
        if len(loaded_case.data_files) != 1:
            raise ReaderConfigurationError("VTK-family adapter expects one flow_field data file")
        return loaded_case.data_files[0]

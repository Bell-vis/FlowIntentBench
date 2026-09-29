"""Pinned source conversion used by build_context_construction; no admission decisions."""

import json, hashlib, zipfile
from pathlib import Path
import numpy as np, h5py, vtk
from vtk.util.numpy_support import numpy_to_vtk
from flowintentbench.source_acquisition import HTTPRangeFile


def _well(d, spec):
    name = d.name
    s = json.loads((d / "sources/selected_asset.json").read_text())
    raw = d / "raw"
    raw.mkdir(exist_ok=True)
    with HTTPRangeFile(s["url"], s["size"], raw / "ranges") as stream:
        with h5py.File(stream, "r") as f:
            frame = int(spec["frame"])
            trajectory = int(spec["trajectory"])
            coords = [f["dimensions/" + axis][:] for axis in "xyz"]
            grid = vtk.vtkRectilinearGrid()
            grid.SetDimensions(*[len(x) for x in coords])
            for axis, array in zip("XYZ", coords):
                getattr(grid, "Set" + axis + "Coordinates")(
                    numpy_to_vtk(array, deep=True)
                )
            arrays = {}
            meta = {}
            for group in ["t0_fields", "t1_fields"]:
                for field in f[group]:
                    node = f[group + "/" + field]
                    print(name, "READ", field, flush=True)
                    values = node[trajectory, frame]
                    arrays[field] = values
                    # The Well uses (x,y,z[,components]); VTK's point index varies fastest in x.
                    ordered = values.transpose(
                        (2, 1, 0, 3) if values.ndim == 4 else (2, 1, 0)
                    )
                    flat = (
                        ordered.reshape(-1, values.shape[-1])
                        if values.ndim == 4
                        else ordered.ravel()
                    )
                    a = numpy_to_vtk(np.ascontiguousarray(flat), deep=True)
                    a.SetName(field)
                    grid.GetPointData().AddArray(a)
                    meta[field] = {
                        "shape": list(values.shape),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "finite": bool(np.isfinite(values).all()),
                    }
            time = float(f["dimensions/time"][frame])
            attrs = {
                k: (
                    v.tolist()
                    if isinstance(v, np.ndarray)
                    else v.item()
                    if isinstance(v, np.generic)
                    else v
                )
                for k, v in f.attrs.items()
            }
            np.savez_compressed(
                raw / "snapshot.npz", **arrays, **dict(zip("xyz", coords)), time=time
            )
            w = vtk.vtkRectilinearGridWriter()
            w.SetFileName(str(d / "snapshot.vtk"))
            w.SetInputData(grid)
            w.SetFileTypeToBinary()
            w.Write()
            report = {
                "source": s,
                "whole_remote_object_checksum_verified": False,
                "trajectory": trajectory,
                "frame": frame,
                "time": time,
                "field_metadata": meta,
                "attrs": attrs,
                "sample_geometry": "Stored coordinate samples represented as VTK points; the analysis domain is their rectilinear span. No extrapolation to unstored boundary faces.",
                "snapshot_sha256": hashlib.sha256(
                    (d / "snapshot.vtk").read_bytes()
                ).hexdigest(),
                "ranges": stream.chunks,
            }
            (d / "sources/snapshot_provenance.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(name, "DONE", meta, flush=True)


def _electro(d, spec):
    a = spec["source"]
    prefix = spec["member_prefix"]
    with HTTPRangeFile(a["url"], a["size_bytes"], d / "raw/ranges") as stream:
        with zipfile.ZipFile(stream) as z:
            rows = []
            for info in z.infolist():
                if not info.filename.startswith(prefix) or info.is_dir():
                    continue
                if Path(info.filename).name.startswith("omz"):
                    continue
                output = d / "raw/selected" / Path(info.filename).name
                output.parent.mkdir(exist_ok=True, parents=True)
                if not output.exists():
                    print("EXTRACT", info.filename, flush=True)
                    output.write_bytes(z.read(info))
                import zlib

                if (
                    output.stat().st_size != info.file_size
                    or zlib.crc32(output.read_bytes()) != info.CRC
                ):
                    raise ValueError("ZIP member CRC/size mismatch: " + info.filename)
                rows.append(
                    {
                        "member": info.filename,
                        "path": str(output.relative_to(d)),
                        "size": info.file_size,
                        "crc32": info.CRC,
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    }
                )
            (d / "sources/subset_provenance.json").write_text(
                json.dumps(
                    {
                        "source": a,
                        "whole_remote_object_checksum_verified": False,
                        "members": rows,
                        "ranges": stream.chunks,
                    },
                    indent=2,
                )
                + "\n"
            )
    print("ELECTRO DONE", flush=True)


def _openfoam(d, spec):
    archive = d / spec["archive"]
    with zipfile.ZipFile(archive) as z:
        out = d / "raw/extracted"
        out.mkdir(exist_ok=True, parents=True)
        for member in z.infolist():
            target = (out / member.filename).resolve()
            if not target.is_relative_to(out.resolve()):
                raise ValueError("escaping archive member")
        z.extractall(out)
    reader = vtk.vtkOpenFOAMReader()
    reader.SetFileName(str(out / "case.foam"))
    reader.UpdateInformation()
    reader.SetTimeValue(float(spec["time"]))
    reader.EnableAllCellArrays()
    reader.EnableAllPatchArrays()
    reader.Update()
    mesh = reader.GetOutput().GetBlock(0)
    if mesh is None or mesh.GetNumberOfCells() == 0:
        raise ValueError("empty OpenFOAM internal mesh")
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(d / "flow.vtu"))
    writer.SetInputData(mesh)
    if writer.Write() != 1:
        raise ValueError("VTU write failed")
    (d / "sources/conversion_provenance.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "time": spec["time"],
                "block": "internalMesh",
                "output_sha256": hashlib.sha256(
                    (d / "flow.vtu").read_bytes()
                ).hexdigest(),
                "vtk_version": vtk.vtkVersion.GetVTKVersion(),
            },
            indent=2,
        )
        + "\n"
    )


def _electro_grid(d, spec):
    import xml.etree.ElementTree as ET

    raw = d / "raw/selected"
    grid = None
    arrays = {}
    for name in ["p", "vf", "vx", "vy", "vz"]:
        paths = list(raw.glob(name + "_*.xmf"))
        if len(paths) != 1:
            raise ValueError("expected one XMF for " + name)
        root = ET.parse(paths[0]).getroot()
        topology = root.find(".//Topology")
        dims = tuple(map(int, topology.attrib["Dimensions"].split()))[::-1]
        geometry = root.find(".//Geometry")
        items = geometry.findall("DataItem")
        origin = tuple(map(float, items[0].text.split()))
        spacing = tuple(map(float, items[1].text.split()))
        if grid is None:
            grid = vtk.vtkImageData()
            grid.SetDimensions(*dims)
            grid.SetOrigin(*origin)
            grid.SetSpacing(*spacing)
        elif (dims, origin, spacing) != (
            grid.GetDimensions(),
            grid.GetOrigin(),
            grid.GetSpacing(),
        ):
            raise ValueError("incompatible XMF grids")
        binary = next(
            item
            for item in root.iter("DataItem")
            if item.attrib.get("Format") == "Binary"
        )
        path = (raw / binary.text.strip()).resolve()
        if not path.is_relative_to(raw.resolve()):
            raise ValueError("escaping RAW path")
        if (
            binary.attrib.get("Precision") != "8"
            or binary.attrib.get("Seek", "0") != "0"
        ):
            raise ValueError("unsupported RAW layout")
        endian = binary.attrib.get("Endian", spec["undeclared_endian"])
        values = np.fromfile(
            path,
            dtype="<f8"
            if endian == "Little"
            else ">f8"
            if endian == "Big"
            else "INVALID",
        )
        if len(values) != grid.GetNumberOfCells() or not np.isfinite(values).all():
            raise ValueError("invalid RAW field")
        arrays[name] = values
        if name in ["p", "vf"]:
            array = numpy_to_vtk(values, deep=True)
            array.SetName(name)
            grid.GetCellData().AddArray(array)
    vector = numpy_to_vtk(
        np.column_stack([arrays[n] for n in ["vx", "vy", "vz"]]), deep=True
    )
    vector.SetName("velocity")
    grid.GetCellData().AddArray(vector)
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(str(d / "flow.vtk"))
    writer.SetFileTypeToBinary()
    writer.SetInputData(grid)
    if writer.Write() != 1:
        raise ValueError("flow write failed")
    (d / "sources/grid_conversion.json").write_text(
        json.dumps(
            {
                "undeclared_endian_assumption": spec["undeclared_endian"],
                "layout": "XMF cell arrays; x index fastest; no interpolation",
                "field_ranges": {
                    k: [float(v.min()), float(v.max())] for k, v in arrays.items()
                },
                "output_sha256": hashlib.sha256(
                    (d / "flow.vtk").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


def _electric_components(d, spec):
    source = (d.parent / spec["source_path"]).resolve()
    if not source.is_relative_to(d.parent.resolve()):
        raise ValueError("escaping electrical source path")
    if hashlib.sha256(source.read_bytes()).hexdigest() != spec["source_sha256"]:
        raise ValueError("electrical source checksum mismatch")
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(str(source))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    mesh = reader.GetOutput()
    from vtk.util.numpy_support import vtk_to_numpy

    values = np.column_stack(
        [vtk_to_numpy(mesh.GetPointData().GetArray(n)) for n in ["jx", "jy", "jz"]]
    )
    if not np.isfinite(values).all():
        raise ValueError("nonfinite current components")
    array = numpy_to_vtk(values, deep=True)
    array.SetName("current_density")
    mesh.GetPointData().AddArray(array)
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(str(d / "electric.vtk"))
    writer.SetFileTypeToBinary()
    writer.SetInputData(mesh)
    if writer.Write() != 1:
        raise ValueError("electrical field write failed")


def prepare_source_data(dataset_dir: Path):
    """Execute the dataset's recorded import, retaining source/cache provenance."""
    dataset_dir = dataset_dir.resolve()
    spec = json.loads((dataset_dir / "construction/source_import.json").read_text())
    steps = spec.get("steps", [spec])
    handlers = {
        "well_snapshot": _well,
        "zip_subset": _electro,
        "electro_grid": _electro_grid,
        "openfoam": _openfoam,
        "electric_components": _electric_components,
    }
    for step in steps:
        if step["kind"] not in handlers:
            raise ValueError("unsupported source import " + step["kind"])
        handlers[step["kind"]](dataset_dir, step)
    return {
        "dataset_id": dataset_dir.name,
        "status": "PREPARED",
        "steps": [step["kind"] for step in steps],
    }

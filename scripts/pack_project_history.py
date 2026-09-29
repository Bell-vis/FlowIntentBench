"""Pack explicitly retired trees; verify every byte before pruning originals."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "superseded_20260920": "archive/outputs/superseded_20260920",
    "superseded_20260921": "archive/outputs/superseded_20260921",
    "legacy_docs_20260922": "archive/docs",
    "legacy_scripts_20260922": "archive/scripts",
}


def inventory(source: Path) -> dict[str, tuple[int, int, int]]:
    """Never traverse links/junctions, including the source itself."""
    result = {}
    stack = [source]
    while stack:
        path = stack.pop()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
            raise RuntimeError(f"Refusing reparse point: {path}")
        key = path.relative_to(source).as_posix()
        if stat.S_ISDIR(info.st_mode):
            result[key + "/"] = (0, info.st_mtime_ns, info.st_ino)
            stack.extend(path.iterdir())
        elif stat.S_ISREG(info.st_mode):
            result[key] = (info.st_size, info.st_mtime_ns, info.st_ino)
        else:
            raise RuntimeError(f"Refusing special file: {path}")
    return result


def stream_hash(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, expected: dict[str, str]) -> None:
    with zipfile.ZipFile(path) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise RuntimeError("Archive entries differ from source inventory")
        for name, digest in expected.items():
            with archive.open(name) as stream:
                if stream_hash(stream) != digest:
                    raise RuntimeError(f"Archive content mismatch: {name}")


def save_report(root: Path, row: dict) -> None:
    path = root / "archive/PACKED_HISTORY.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"packs": {}}
    data["packs"][row["name"]] = row
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def pack(root: Path, name: str) -> dict:
    root = root.resolve()
    source = root / TARGETS[name]
    # Validate the exact parent chain before any rename/removal.
    for parent in (source, *source.parents):
        if parent == root:
            break
        if parent.is_symlink() or getattr(parent.lstat(), "st_file_attributes", 0) & 1024:
            raise RuntimeError(f"Refusing reparse point: {parent}")
    source.resolve().relative_to(root / "archive")
    destination = root / "archive" / f"{name}.zip"
    temporary = destination.with_suffix(".zip.partial")
    quarantine = source.with_name(source.name + ".verified-packing")
    if destination.exists() or temporary.exists() or quarantine.exists():
        raise RuntimeError(f"Existing pack or interrupted operation for {name}; inspect before retrying")
    before = inventory(source)
    prefix = source.relative_to(root).as_posix()
    digests = {}
    print(f"Packing {name}: {len(before)} entries", flush=True)
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, strict_timestamps=False) as archive:
        for index, key in enumerate(sorted(before)):
            if key.endswith("/"):
                archive.writestr(prefix + "/" + ("" if key == "./" else key), b"")
                continue
            arcname = prefix + "/" + key
            digest = hashlib.sha256()
            info = zipfile.ZipInfo.from_file(source / key, arcname, strict_timestamps=False)
            info.compress_type = zipfile.ZIP_DEFLATED
            with (source / key).open("rb") as reader, archive.open(info, "w", force_zip64=True) as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    digest.update(chunk)
                    writer.write(chunk)
            digests[arcname] = digest.hexdigest()
            if index and index % 5000 == 0:
                print(f"  packed {index}/{len(before)} entries", flush=True)
        manifest = (json.dumps({"source": prefix, "sha256": digests}, sort_keys=True) + "\n").encode()
        archive.writestr("PACK_MANIFEST.json", manifest)
        digests["PACK_MANIFEST.json"] = hashlib.sha256(manifest).hexdigest()
    print(f"Verifying {name}: all file contents", flush=True)
    verify_archive(temporary, digests)
    if inventory(source) != before:
        raise RuntimeError("Source changed during packing; original retained")
    temporary.replace(destination)
    with destination.open("rb") as stream:
        archive_hash = stream_hash(stream)
    row = {
        "name": name, "source": prefix,
        "archive": destination.relative_to(root).as_posix(),
        "files_before": sum(not key.endswith("/") for key in before),
        "directories_before": sum(key.endswith("/") for key in before),
        "bytes_before": sum(value[0] for value in before.values()),
        "bytes_after": destination.stat().st_size,
        "sha256": archive_hash, "verified_utc": datetime.now(timezone.utc).isoformat(),
        "state": "verified_original_retained",
        "quarantine": quarantine.relative_to(root).as_posix(),
    }
    save_report(root, row)
    # Detach only the retired tree; readers of active output paths are untouched.
    source.rename(quarantine)
    after = inventory(quarantine)
    # A directory rename may update the root's timestamp on some filesystems.
    before.pop("./", None)
    after.pop("./", None)
    if before != after:
        quarantine.rename(source)
        raise RuntimeError("Source changed before pruning; original restored")
    row["state"] = "verified_pruning"
    save_report(root, row)
    print(f"Removing verified expanded tree {prefix}", flush=True)
    shutil.rmtree(quarantine)
    row["state"] = "packed"
    save_report(root, row)
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=list(TARGETS))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for name in args.names:
        if args.apply:
            pack(ROOT, name)
        else:
            entries = inventory(ROOT / TARGETS[name])
            print(json.dumps({"name": name, "source": TARGETS[name],
                              "files": sum(not key.endswith("/") for key in entries),
                              "bytes": sum(row[0] for row in entries.values())}))


if __name__ == "__main__":
    main()

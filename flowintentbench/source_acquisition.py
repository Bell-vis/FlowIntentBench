"""Acquire published source assets for the existing dataset construction step."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


class HTTPRangeFile(io.RawIOBase):
    """Bounded, cached reads from a pinned HTTP object for HDF5/ZIP import.

    Every response must establish the exact requested byte range. Cached
    chunks carry local hashes; this is subset provenance, not verification of
    the publisher's checksum for the entire remote object.
    """

    def __init__(
        self, url: str, size: int, cache_dir: Path, block_size: int = 4 * 1024 * 1024
    ):
        if urlparse(url).scheme != "https" or size <= 0 or block_size <= 0:
            raise ValueError("invalid pinned HTTP source")
        self.url, self.size, self.block_size, self.position = url, size, block_size, 0
        self.cache_dir = cache_dir / hashlib.sha256(url.encode()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.size + offset
            if whence == 2
            else -1
        )
        if position < 0:
            raise ValueError("invalid seek")
        self.position = position
        return position

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def read(self, count=-1):
        end = self.size if count < 0 else min(self.position + count, self.size)
        parts = []
        while self.position < end:
            start = self.position // self.block_size * self.block_size
            stop = min(start + self.block_size, self.size) - 1
            path = self.cache_dir / f"{start}-{stop}.bin"
            hash_path = path.with_suffix(".sha256")
            if path.is_file() and hash_path.is_file():
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != hash_path.read_text().strip():
                    raise ValueError("corrupt cached source range")
            else:
                separator = "&" if "?" in self.url else "?"
                # Different range URLs avoid intermediary caches that ignore
                # Range in their cache key. Content-Range is still verified.
                url = f"{self.url}{separator}range_start={start}"
                with tempfile.TemporaryDirectory() as temporary:
                    headers = Path(temporary) / "headers"
                    result = subprocess.run(
                        [
                            "curl",
                            "--fail",
                            "--location",
                            "--silent",
                            "--show-error",
                            "--proto",
                            "=https",
                            "--proto-redir",
                            "=https",
                            "--connect-timeout",
                            "15",
                            "--max-time",
                            "180",
                            "--retry",
                            "2",
                            "--max-filesize",
                            str(stop - start + 1),
                            "--range",
                            f"{start}-{stop}",
                            "--dump-header",
                            str(headers),
                            url,
                        ],
                        capture_output=True,
                        check=True,
                    )
                    matches = re.findall(
                        r"content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)",
                        headers.read_text().lower(),
                    )
                    if not matches or tuple(map(int, matches[-1])) != (
                        start,
                        stop,
                        self.size,
                    ):
                        raise ValueError(
                            "server did not establish the requested byte range"
                        )
                    content = result.stdout
                if len(content) != stop - start + 1:
                    raise ValueError("incomplete source byte range")
                path.write_bytes(content)
                hash_path.write_text(hashlib.sha256(content).hexdigest() + "\n")
            if len(content) != stop - start + 1:
                raise ValueError("cached range length mismatch")
            self.chunks[str(start)] = {
                "stop": stop,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            length = min(end - self.position, stop - self.position + 1)
            parts.append(
                content[self.position - start : self.position - start + length]
            )
            self.position += length
        return b"".join(parts)


def acquire_source_assets(dataset_dir: Path, *, timeout: int = 600) -> dict:
    """Download pinned assets atomically; only checksum-verified bytes are usable.

    Input is construction/source_assets.json. This does not create evidence,
    scientific qualification, or a dataset manifest on behalf of a curator.
    """
    dataset_dir = dataset_dir.resolve()
    specification = json.loads(
        (dataset_dir / "construction/source_assets.json").read_text()
    )
    assets = specification["assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError("source_assets requires a non-empty assets list")
    validated = []
    seen = set()
    for asset in assets:
        path = Path(asset["path"])
        target = (dataset_dir / path).resolve()
        if (
            path.is_absolute()
            or not target.is_relative_to(dataset_dir)
            or target in seen
        ):
            raise ValueError(f"duplicate or escaping source asset path: {path}")
        seen.add(target)
        if urlparse(asset["url"]).scheme not in {"http", "https"}:
            raise ValueError("source asset URL must use HTTP(S)")
        algorithm, digest = asset["checksum"].split(":", 1)
        if algorithm not in {"md5", "sha256", "git-sha1"} or len(digest) != {
            "md5": 32,
            "sha256": 64,
            "git-sha1": 40,
        }.get(algorithm):
            raise ValueError(
                "source assets require a published md5, sha256 or git-sha1 checksum"
            )
        int(digest, 16)
        if type(asset["size_bytes"]) is not int or asset["size_bytes"] <= 0:
            raise ValueError("source asset size_bytes must be a positive integer")
        validated.append((asset, target, algorithm, digest.lower()))
    report = {"dataset_id": dataset_dir.name, "status": "INCOMPLETE", "assets": []}
    report_path = dataset_dir / "construction/source_acquisition.json"

    def persist():
        report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    for asset, target, algorithm, digest in validated:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".download")
        row = {
            "path": asset["path"],
            "source_url": asset["url"],
            "published_checksum": asset["checksum"],
        }
        report["assets"].append(row)
        try:
            cached = target.is_file()
            if not cached:
                command = [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--proto",
                    "=http,https",
                    "--proto-redir",
                    "=http,https",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    str(timeout),
                    "--continue-at",
                    "-",
                    "--max-filesize",
                    str(asset["size_bytes"]),
                    "--output",
                    str(partial),
                    asset["url"],
                ]
                # Start a new curl process for each retry so --continue-at
                # re-reads the updated partial size instead of restarting at
                # the offset from the first failed request.
                for attempt in range(3):
                    try:
                        subprocess.run(command, check=True, capture_output=True)
                        break
                    except subprocess.CalledProcessError:
                        if attempt == 2:
                            raise
            candidate = target if cached else partial
            source_hash = hashlib.new("sha1" if algorithm == "git-sha1" else algorithm)
            if algorithm == "git-sha1":
                source_hash.update(f"blob {candidate.stat().st_size}\0".encode())
            local_hash = hashlib.sha256()
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    source_hash.update(chunk)
                    local_hash.update(chunk)
            if (
                candidate.stat().st_size != asset["size_bytes"]
                or source_hash.hexdigest() != digest
            ):
                # Keep existing complete files untouched; reject corrupt
                # temporary downloads so a rerun can start from zero.
                if not cached:
                    partial.unlink()
                raise ValueError("published asset size/checksum mismatch")
            if not cached:
                partial.replace(target)
            row.update(
                status="VERIFIED",
                sha256=local_hash.hexdigest(),
                size_bytes=target.stat().st_size,
                cached=cached,
            )
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            row.update(status="FAILED", error=type(exc).__name__)
            persist()
            raise
        persist()
    report["status"] = "VERIFIED"
    persist()
    return report

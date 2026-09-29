"""Frozen Python environment construction for evaluated execution.

The environment is intentionally small: a dedicated venv is created from the
single repository lock file and receives only the locked distributions and
their non-extra dependency closure.  Host site-packages are never mounted into
the evaluated sandbox.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import sys
import tempfile
import threading
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement


class FrozenEnvironmentError(RuntimeError):
    """Raised when the locked runtime environment cannot be built or verified."""


_BUILD_LOCK = threading.Lock()
_ENVIRONMENT_BUILDER_VERSION = "3"


def _default_requirements_path() -> Path:
    return Path(__file__).resolve().parents[1] / "runtime-requirements.txt"


def _default_cache_parent() -> Path:
    """Use the OS temp volume unless it is too small for VTK and SciPy."""
    temporary = Path(tempfile.gettempdir()) / "flowintentbench-runtime-env"
    try:
        if shutil.disk_usage(temporary.parent).free >= 2 * 1024**3:
            return temporary
    except OSError:
        pass
    return Path(__file__).resolve().parents[1] / ".runtime-cache"


def _requirements(requirements_path: Path) -> dict[str, Requirement]:
    if not requirements_path.is_file():
        raise FrozenEnvironmentError(f"runtime requirements file does not exist: {requirements_path}")
    result: dict[str, Requirement] = {}
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            requirement = Requirement(stripped)
        except Exception as exc:
            raise FrozenEnvironmentError(f"invalid runtime requirement: {stripped}") from exc
        if not requirement.specifier:
            raise FrozenEnvironmentError(
                f"runtime requirement must pin an exact version: {stripped}"
            )
        if len(requirement.specifier) != 1 or not all(
            item.operator == "==" for item in requirement.specifier
        ):
            raise FrozenEnvironmentError(
                f"runtime requirement must use one exact == pin: {stripped}"
            )
        result[requirement.name.casefold().replace("-", "_")] = requirement
    if not result:
        raise FrozenEnvironmentError("runtime requirements file contains no packages")
    return result


def _distribution_for(name: str) -> metadata.Distribution:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError as exc:
        raise FrozenEnvironmentError(f"required runtime package is not installed: {name}") from exc


def _dependency_closure(
    direct: dict[str, Requirement],
) -> dict[str, metadata.Distribution]:
    pending = list(direct)
    distributions: dict[str, metadata.Distribution] = {}
    while pending:
        name = pending.pop()
        normalized = name.casefold().replace("-", "_")
        if normalized in distributions:
            continue
        distribution = _distribution_for(name)
        requirement = direct.get(normalized)
        if requirement is not None and not requirement.specifier.contains(
            distribution.version,
            prereleases=True,
        ):
            raise FrozenEnvironmentError(
                f"installed {distribution.metadata['Name']}=={distribution.version} does not satisfy "
                f"{requirement}"
            )
        distributions[normalized] = distribution
        for raw_dependency in distribution.requires or ():
            dependency = Requirement(raw_dependency)
            environment = dict(os.environ)
            environment["extra"] = ""
            if dependency.marker is not None and not dependency.marker.evaluate(environment):
                continue
            pending.append(dependency.name)
    return distributions


def _python_executable(root: Path) -> Path:
    candidates = (
        (root / "Scripts" / "python.exe", root / "Lib" / "site-packages"),
        (root / "bin" / "python", root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"),
    )
    for executable, _ in candidates:
        if executable.is_file():
            return executable
    return candidates[0][0] if os.name == "nt" else candidates[1][0]


def _site_packages(root: Path) -> Path:
    candidates = (
        root / "Lib" / "site-packages",
        root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FrozenEnvironmentError(
        "dedicated venv site-packages is missing: " + ", ".join(str(path) for path in candidates)
    )


def _copy_distribution(
    distribution: metadata.Distribution,
    destination: Path,
    environment_root: Path,
) -> None:
    files = distribution.files
    if files is None:
        raise FrozenEnvironmentError(
            f"package has no install file record and cannot be copied safely: "
            f"{distribution.metadata['Name']}"
        )
    source_root = Path(distribution.locate_file(""))

    def copy_file(relative_path: Path, source: Path) -> None:
        if relative_path.is_absolute():
            raise FrozenEnvironmentError(
                f"package file escapes its site root: {distribution.metadata['Name']} / {relative_path}"
            )
        if not source.is_file():
            return
        target = (destination / relative_path).resolve()
        try:
            target.relative_to(environment_root.resolve())
        except ValueError as exc:
            raise FrozenEnvironmentError(
                f"package file escapes dedicated environment: {distribution.metadata['Name']} / {relative_path}"
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for relative in files:
        relative_path = Path(str(relative))
        source = Path(distribution.locate_file(relative))
        copy_file(relative_path, source)

    # Some legacy egg-info distributions omit native package files from their
    # file record.  Copy only the declared top-level package directories to
    # keep the dedicated environment complete without exposing host paths.
    top_level = distribution.read_text("top_level.txt")
    if top_level:
        for name in top_level.splitlines():
            package_name = name.strip()
            if not package_name or package_name.startswith("#"):
                continue
            package_root = source_root / package_name
            if package_root.is_dir():
                for source in package_root.rglob("*"):
                    if source.is_file():
                        copy_file(source.relative_to(source_root), source)


def _fingerprint(
    requirements_path: Path,
    distributions: dict[str, metadata.Distribution],
) -> tuple[str, dict[str, Any]]:
    requirement_bytes = requirements_path.read_bytes()
    package_versions = {
        name: distribution.version
        for name, distribution in sorted(distributions.items())
    }
    digest = hashlib.sha256()
    digest.update(requirement_bytes)
    digest.update(sys.version.encode("utf-8"))
    digest.update(json.dumps(package_versions, sort_keys=True).encode("utf-8"))
    digest.update(_ENVIRONMENT_BUILDER_VERSION.encode("utf-8"))
    identifier = digest.hexdigest()[:20]
    return identifier, {
        "environment_id": f"flowintentbench-v1-{identifier}",
        "builder_version": _ENVIRONMENT_BUILDER_VERSION,
        "python_version": sys.version.split()[0],
        "requirements_sha256": hashlib.sha256(requirement_bytes).hexdigest(),
        "packages": package_versions,
    }


@dataclass(frozen=True)
class FrozenRuntimeEnvironment:
    """A dedicated interpreter and its locked package fingerprint."""

    root: Path
    python_executable: Path
    site_packages: Path
    base_prefix: Path
    fingerprint: dict[str, Any]


def ensure_frozen_environment(
    requirements_path: str | Path | None = None,
    *,
    cache_parent: str | Path | None = None,
) -> FrozenRuntimeEnvironment:
    """Create or reuse a dedicated venv from the pinned runtime requirements."""

    requirements = Path(requirements_path or _default_requirements_path()).resolve()
    direct = _requirements(requirements)
    distributions = _dependency_closure(direct)
    identifier, fingerprint = _fingerprint(requirements, distributions)
    parent = Path(cache_parent or _default_cache_parent())
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / identifier
    metadata_path = root / "flowintentbench-environment.json"
    with _BUILD_LOCK:
        if metadata_path.is_file():
            recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if recorded == fingerprint and _python_executable(root).is_file():
                return _environment_from_root(root, recorded)

        temporary_root = Path(tempfile.mkdtemp(prefix=f".{identifier}-", dir=parent))
        try:
            venv.EnvBuilder(with_pip=False, clear=True, symlinks=False).create(temporary_root)
            destination = _site_packages(temporary_root)
            for distribution in distributions.values():
                _copy_distribution(distribution, destination, temporary_root)
            (temporary_root / "flowintentbench-environment.json").write_text(
                json.dumps(fingerprint, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if root.exists():
                shutil.rmtree(root)
            os.replace(temporary_root, root)
        except Exception as exc:
            shutil.rmtree(temporary_root, ignore_errors=True)
            raise FrozenEnvironmentError(
                f"could not build dedicated runtime environment {identifier}: {exc}"
            ) from exc
    return _environment_from_root(root, fingerprint)


def _environment_from_root(root: Path, fingerprint: dict[str, Any]) -> FrozenRuntimeEnvironment:
    python_executable = _python_executable(root)
    if not python_executable.is_file():
        raise FrozenEnvironmentError(f"dedicated runtime Python executable is missing: {python_executable}")
    return FrozenRuntimeEnvironment(
        root=root,
        python_executable=python_executable,
        site_packages=_site_packages(root),
        base_prefix=Path(sys.base_prefix).resolve(),
        fingerprint=fingerprint,
    )

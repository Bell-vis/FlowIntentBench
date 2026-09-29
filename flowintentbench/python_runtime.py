"""Minimal case-scoped Python execution for model-controlled analysis.

This module is deliberately separate from :mod:`flowintentbench.execution`.
The latter reads benchmark files for infrastructure validation; this module
hosts model-written Python and exposes no flow-analysis operations.

The default Linux sandbox uses bubblewrap to provide a read-only model-visible
case directory, a writable workspace, and an isolated network namespace.  A
runtime cannot silently fall back to an unsandboxed process when the primary
no-web condition is requested.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import selectors
import secrets
import socket
import shutil
import subprocess
import tempfile
import textwrap
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Mapping

from .loader import LoadedCase
from .manifest import DatasetManifest
from .runtime_environment import FrozenEnvironmentError, FrozenRuntimeEnvironment, ensure_frozen_environment
from .agent_profile import RuntimeProfile, build_runtime_contract, load_runtime_profile


class PythonRuntimeError(RuntimeError):
    """Base error for runtime lifecycle and staging failures."""


class RuntimeUnavailableError(PythonRuntimeError):
    """Raised when the requested isolation mechanism is unavailable."""


class PythonRuntimeClosedError(PythonRuntimeError):
    """Raised when execution is attempted after a runtime was closed."""


class FilesystemDiscoveryError(PythonRuntimeError):
    """Model code attempted an unrestricted recursive scan from filesystem root."""


@dataclass(frozen=True)
class PythonExecutionException:
    """Serializable information about an exception raised by model code."""

    type: str
    message: str
    traceback: str


@dataclass(frozen=True)
class PythonExecutionResult:
    """Observable result of one evaluated-agent Python invocation."""

    success: bool
    stdout: str
    stderr: str
    exception: PythonExecutionException | None
    duration_seconds: float
    execution_index: int
    output_truncated: bool = False
    stdout_chars_total: int | None = None
    stderr_chars_total: int | None = None
    returned_stdout_chars: int | None = None
    returned_stderr_chars: int | None = None
    filesystem_discovery_blocked: bool = False

    @property
    def duration_ms(self) -> float:
        """Return duration in milliseconds for simple telemetry consumers."""

        return self.duration_seconds * 1000.0


def python_execution_result_payload(result: PythonExecutionResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exception": (
            {
                "type": result.exception.type,
                "message": result.exception.message,
                "traceback": result.exception.traceback,
            }
            if result.exception is not None
            else None
        ),
        "duration_seconds": result.duration_seconds,
        "execution_index": result.execution_index,
        "output_truncated": result.output_truncated,
        "stdout_chars_total": result.stdout_chars_total,
        "stderr_chars_total": result.stderr_chars_total,
        "returned_stdout_chars": result.returned_stdout_chars,
        "returned_stderr_chars": result.returned_stderr_chars,
        "filesystem_discovery_blocked": result.filesystem_discovery_blocked,
    }


_CHILD_SCRIPT = textwrap.dedent(
    r"""
    import codecs
    import contextlib
    import io
    import json
    import os
    import socket
    import sys
    import threading
    import time
    import traceback


    class _ByteCapture:
        def __init__(self, limit):
            self.limit = limit
            full_marker = b"\n...[runtime output truncated; tail preview follows]...\n"
            self.marker = full_marker[:limit]
            preview_limit = max(0, limit - len(self.marker))
            self.head_limit = preview_limit // 2
            self.tail_limit = preview_limit - self.head_limit
            self.head = bytearray()
            self.tail = bytearray()
            self.total_bytes = 0
            self.total_chars = 0
            self.truncated = False
            self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            self.lock = threading.Lock()

        def append(self, chunk):
            with self.lock:
                self.total_bytes += len(chunk)
                self.total_chars += len(self.decoder.decode(chunk, final=False))
                if not self.truncated and len(self.head) + len(chunk) <= self.limit:
                    self.head.extend(chunk)
                    return
                if not self.truncated:
                    self.truncated = True
                    combined = bytes(self.head) + chunk
                    self.head = bytearray(combined[: self.head_limit])
                    self.tail = bytearray(
                        combined[-self.tail_limit :] if self.tail_limit else b""
                    )
                    return
                self.tail.extend(chunk)
                if len(self.tail) > self.tail_limit:
                    del self.tail[: len(self.tail) - self.tail_limit]

        def text(self):
            if not self.truncated:
                return bytes(self.head).decode("utf-8", errors="replace")
            return (
                bytes(self.head).decode("utf-8", errors="replace")
                + self.marker.decode("ascii")
                + bytes(self.tail).decode("utf-8", errors="replace")
            )

        def finish_character_count(self):
            self.total_chars += len(self.decoder.decode(b"", final=True))


    class _BoundedTextIO(io.TextIOBase):
        def __init__(self, stream):
            self.stream = stream

        @property
        def buffer(self):
            return self.stream.buffer

        def write(self, value):
            return self.stream.write(str(value))

        def flush(self):
            return self.stream.flush()

        def fileno(self):
            return self.stream.fileno()

        def isatty(self):
            return False

        @property
        def encoding(self):
            return getattr(self.stream, "encoding", "utf-8")


    def _drain_capture(read_fd, capture):
        try:
            while True:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    return
                capture.append(chunk)
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass


    def _start_capture(fd, limit):
        read_fd, write_fd = os.pipe()
        saved_fd = os.dup(fd)
        os.dup2(write_fd, fd)
        os.close(write_fd)
        capture = _ByteCapture(limit)
        thread = threading.Thread(target=_drain_capture, args=(read_fd, capture), daemon=True)
        thread.start()
        return saved_fd, capture, thread


    def _finish_capture(fd, saved_fd, capture, thread):
        try:
            os.dup2(saved_fd, fd)
        finally:
            os.close(saved_fd)
        thread.join(timeout=2)
        return capture


    control_fd = os.environ.get("FLOWINTENTBENCH_CONTROL_FD")
    if control_fd is not None:
        control = socket.socket(fileno=int(control_fd))
    else:
        control = socket.create_connection(
            (
                os.environ["FLOWINTENTBENCH_CONTROL_HOST"],
                int(os.environ["FLOWINTENTBENCH_CONTROL_PORT"]),
            ),
            timeout=10,
        )
        control.sendall(os.environ["FLOWINTENTBENCH_CONTROL_TOKEN"].encode("ascii"))
        # The timeout only protects the startup handshake.  The persistent
        # protocol must wait indefinitely between model turns while the parent
        # performs the next provider request.
        control.settimeout(None)
    protocol = control.makefile("rwb", buffering=0)
    namespace = {"__name__": "__main__", "__builtins__": __builtins__}
    empty_stdin = io.StringIO()
    protocol.write(b'{"runtime_ready": true}\n')
    protocol.flush()

    for raw_line in protocol:
        try:
            request = json.loads(raw_line)
            code = request["code"]
            output_limit = int(request["output_limit"])
        except Exception as exc:
            response = {
                "success": False,
                "stdout": "",
                "stderr": "",
                "exception": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "duration_seconds": 0.0,
            }
            protocol.write(json.dumps(response).encode("utf-8") + b"\n")
            protocol.flush()
            continue

        started = time.perf_counter()
        stdout_capture = None
        stderr_capture = None
        stdout_saved = None
        stderr_saved = None
        stdout_thread = None
        stderr_thread = None
        failure = None
        try:
            compiled = compile(code, "<flowintentbench-python>", "exec")
            stdout_saved, stdout_capture, stdout_thread = _start_capture(1, output_limit)
            stderr_saved, stderr_capture, stderr_thread = _start_capture(2, output_limit)
            with contextlib.redirect_stdout(_BoundedTextIO(sys.__stdout__)):
                with contextlib.redirect_stderr(_BoundedTextIO(sys.__stderr__)):
                    original_stdin = sys.stdin
                    sys.stdin = empty_stdin
                    try:
                        exec(compiled, namespace, namespace)
                    finally:
                        sys.stdin = original_stdin
        except BaseException as exc:
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
                try:
                    stream.flush()
                except Exception:
                    pass
            if stdout_saved is not None:
                stdout_capture = _finish_capture(1, stdout_saved, stdout_capture, stdout_thread)
            if stderr_saved is not None:
                stderr_capture = _finish_capture(2, stderr_saved, stderr_capture, stderr_thread)

        stdout_text = stdout_capture.text() if stdout_capture is not None else ""
        stderr_text = stderr_capture.text() if stderr_capture is not None else ""
        if stdout_capture is not None:
            stdout_capture.finish_character_count()
        if stderr_capture is not None:
            stderr_capture.finish_character_count()
        if (stdout_capture is not None and stdout_capture.truncated) or (
            stderr_capture is not None and stderr_capture.truncated
        ):
            marker = "\n[output truncated by runtime limit]"
            remaining = max(0, output_limit - len(stderr_text))
            stderr_text += marker[:remaining]
        response = {
            "success": failure is None,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "exception": failure,
            "duration_seconds": time.perf_counter() - started,
            "output_truncated": bool(
                (stdout_capture is not None and stdout_capture.truncated)
                or (stderr_capture is not None and stderr_capture.truncated)
            ),
            "stdout_chars_total": stdout_capture.total_chars if stdout_capture is not None else 0,
            "stderr_chars_total": stderr_capture.total_chars if stderr_capture is not None else 0,
            "returned_stdout_chars": len(stdout_text),
            "returned_stderr_chars": len(stderr_text),
        }
        protocol.write(json.dumps(response).encode("utf-8") + b"\n")
        protocol.flush()
    """
).strip()


def reader_metadata_for_manifest(manifest: DatasetManifest) -> dict[str, Any]:
    """Build model-visible reader metadata without construction provenance."""

    reader = manifest.reader
    grid_blanking = []
    for item in reader.grid_blanking:
        payload = item.model_dump(mode="json")
        if item.validity_rule is not None:
            payload["validity_rule"] = item.validity_rule
        grid_blanking.append(payload)
    return {
        "format": reader.format,
        "canonical_reader": reader.canonical_reader,
        "reader_configuration": reader.reader_configuration,
        "plot3d_convention": reader.plot3d_convention,
        "variable_mappings": [item.model_dump(mode="json") for item in reader.variable_mappings],
        "grid_blanking": grid_blanking,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _runtime_contract(
    environment: FrozenRuntimeEnvironment | None,
    *,
    max_output_chars: int,
    runtime_profile: RuntimeProfile,
) -> dict[str, Any]:
    package_versions = (
        dict(environment.fingerprint.get("packages", {}))
        if environment is not None
        else {}
    )
    contract = build_runtime_contract(runtime_profile, package_versions=package_versions)
    # Low-level development callers may supply a stricter stream cap; the
    # profile remains the advertised/default policy and formal runners use it.
    contract["output_policy"]["max_returned_text_characters_per_stream"] = max_output_chars
    return contract


def _copy_read_only(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(0o444)


def _safe_destination(root: Path, relative_path: str) -> Path:
    destination = (root / relative_path).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise PythonRuntimeError(f"model-visible path escapes staging root: {relative_path}") from exc
    return destination


_ROOT_SCAN_MESSAGE = (
    "Unrestricted recursive discovery from filesystem root is disabled. "
    "Read /case/case_files.json and inspect /case directly; write only under /workspace."
)


def _literal_string(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _is_filesystem_root(value: str | None) -> bool:
    """Recognize POSIX roots and obvious Windows drive/UNC roots."""

    if value is None:
        return False
    return bool(
        value == "/"
        or re.fullmatch(r"[A-Za-z]:[\\/]*", value)
        or value.startswith("\\\\")
    )


def _is_path_root_call(node: ast.AST, root_names: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in root_names
    if not isinstance(node, ast.Call) or not node.args:
        return False
    function = node.func
    function_name = (
        function.id
        if isinstance(function, ast.Name)
        else function.attr if isinstance(function, ast.Attribute) else ""
    )
    return function_name == "Path" and _is_filesystem_root(_literal_string(node.args[0]))


def _root_recursive_discovery_reason(code: str) -> str | None:
    """Return a concise reason for obvious unrestricted root-recursive scans.

    This is a narrow engineering guard, not a general code policy. It blocks
    only literal filesystem-root recursion while leaving `/case` traversal and
    ordinary file access untouched.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    root_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None and (
                _is_filesystem_root(_literal_string(value)) or _is_path_root_call(value, set())
            ):
                root_names.update(
                    target.id for target in targets if isinstance(target, ast.Name)
                )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            method = function.attr
            receiver = function.value
            if method in {"rglob", "walk"} and _is_path_root_call(receiver, root_names):
                return _ROOT_SCAN_MESSAGE
            if method == "glob" and _is_path_root_call(receiver, root_names):
                pattern = _literal_string(node.args[0]) if node.args else None
                if pattern is not None and "**" in pattern:
                    return _ROOT_SCAN_MESSAGE
            if method == "walk" and node.args and _is_filesystem_root(_literal_string(node.args[0])):
                return _ROOT_SCAN_MESSAGE
            if method in {"glob", "iglob"} and node.args:
                pattern = _literal_string(node.args[0])
                recursive = any(
                    keyword.arg == "recursive"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                )
                if recursive and pattern is not None and (
                    pattern.startswith("/")
                    or bool(re.match(r"^[A-Za-z]:[\\/]", pattern))
                    or pattern.startswith("\\\\")
                ) and "**" in pattern:
                    return _ROOT_SCAN_MESSAGE
        elif isinstance(function, ast.Name) and function.id in {"glob", "iglob", "walk"}:
            first = _literal_string(node.args[0]) if node.args else None
            if function.id == "walk" and _is_filesystem_root(first):
                return _ROOT_SCAN_MESSAGE
            recursive = any(
                keyword.arg == "recursive"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            if recursive and first is not None and (
                first.startswith("/")
                or bool(re.match(r"^[A-Za-z]:[\\/]", first))
                or first.startswith("\\\\")
            ) and "**" in first:
                return _ROOT_SCAN_MESSAGE
    # Cover literal shell/subprocess forms without attempting general command
    # interpretation. This intentionally does not block `find /case`.
    if re.search(r"(?:^|[;&|\s])find\s+/(?:\s|$)", code):
        return _ROOT_SCAN_MESSAGE
    return None


class PythonExecutionEnvironment:
    """Run model-written Python in one isolated, persistent case session.

    ``LoadedCase`` supplies only the model-facing case contract and resolved
    permitted files.  The low-level API keeps ``DatasetManifest`` optional for
    core/runtime tests, while a formal benchmark runner must load, validate,
    and pass the dataset manifest for every formal case so the sanitized
    reader metadata sidecar is staged.  The manifest itself is never staged.
    """

    def __init__(
        self,
        loaded_case: LoadedCase,
        *,
        manifest: DatasetManifest | None = None,
        python_executable: str | Path | None = None,
        workspace_parent: str | Path | None = None,
        timeout_seconds: float = 120.0,
        max_output_chars: int | None = None,
        require_network_isolation: bool | None = None,
        runtime_profile: RuntimeProfile | str | Path | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if manifest is not None:
            manifest.validate_against_case(loaded_case.case)
        self.loaded_case = loaded_case
        self.manifest = manifest
        if isinstance(runtime_profile, RuntimeProfile):
            self.runtime_profile = runtime_profile
        else:
            self.runtime_profile = load_runtime_profile(runtime_profile or "flow-python-v1")
        if require_network_isolation is None:
            require_network_isolation = not self.runtime_profile.network_available
        if max_output_chars is None:
            max_output_chars = self.runtime_profile.max_returned_text_chars_per_call
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")
        if self.runtime_profile.case_root != "/case" or self.runtime_profile.workspace_root != "/workspace":
            if self.runtime_profile.runtime_profile_id != "flow-python-windows-local-v1" or (
                self.runtime_profile.case_root != "case"
                or self.runtime_profile.workspace_root != "workspace"
            ):
                raise RuntimeUnavailableError(
                    "the frozen runtime requires case_root=/case and workspace_root=/workspace, "
                    "except for the native Windows local profile case/workspace roots"
                )
        if self.runtime_profile.runtime_profile_id == "flow-python-windows-local-v1" and os.name != "nt":
            raise RuntimeUnavailableError(
                "flow-python-windows-local-v1 is only available on native Windows"
            )
        if self.runtime_profile.network_available and require_network_isolation:
            raise RuntimeUnavailableError("runtime profile permits network access but no-web isolation was requested")
        if "python" not in self.runtime_profile.tools or not self.runtime_profile.tools["python"].get("enabled", False):
            raise RuntimeUnavailableError("runtime profile must enable the Python tool")
        self.frozen_environment: FrozenRuntimeEnvironment | None = None
        if python_executable is None:
            try:
                self.frozen_environment = ensure_frozen_environment()
            except FrozenEnvironmentError as exc:
                raise RuntimeUnavailableError(str(exc)) from exc
            self.python_executable = str(self.frozen_environment.python_executable)
        else:
            self.python_executable = str(python_executable)
        if not Path(self.python_executable).is_file():
            raise RuntimeUnavailableError(f"Python executable does not exist: {self.python_executable}")
        self.workspace_parent = Path(workspace_parent).expanduser().resolve() if workspace_parent else None
        if self.workspace_parent is not None:
            self.workspace_parent.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.require_network_isolation = require_network_isolation
        self._staged_root: Path | None = None
        self._case_dir: Path | None = None
        self._workspace_dir: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._owns_process_group = False
        self._control_parent: socket.socket | None = None
        self._control_fd: int | None = None
        self._control_listener: socket.socket | None = None
        self._control_token: bytes | None = None
        self._control_buffer = bytearray()
        self._stdout_lines: deque[str] = deque(maxlen=64)
        self._stderr_lines: deque[str] = deque(maxlen=64)
        self._stderr_lock = Lock()
        self._stdout_thread: Thread | None = None
        self._stderr_thread: Thread | None = None
        self._started = False
        self._closed = False
        self._execution_index = 0

    @property
    def case_dir(self) -> Path:
        if self._case_dir is None:
            raise PythonRuntimeClosedError("runtime has not been started")
        return self._case_dir

    @property
    def workspace_dir(self) -> Path:
        if self._workspace_dir is None:
            raise PythonRuntimeClosedError("runtime has not been started")
        return self._workspace_dir

    @property
    def started(self) -> bool:
        return self._started and self._process is not None and self._process.poll() is None

    @property
    def environment_fingerprint(self) -> dict[str, Any] | None:
        """Return the frozen runtime fingerprint used for evaluated code."""

        return self.frozen_environment.fingerprint if self.frozen_environment else None

    @property
    def network_isolation_active(self) -> bool:
        """Report whether the primary bubblewrap network namespace is active."""

        return bool(
            self.started
            and self.require_network_isolation
            and shutil.which("bwrap") is not None
            and self._process is not None
            and self._process.poll() is None
        )

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        """Return non-scientific metadata needed to audit this runtime run."""

        if self.runtime_profile.runtime_profile_id == "flow-python-windows-local-v1":
            execution_mode = "WINDOWS_LOCAL_PROCESS"
            filesystem_enforcement = "staged_paths_best_effort"
        elif self.runtime_profile.network_available:
            execution_mode = "HOST_NETWORK_PROOT"
            filesystem_enforcement = "staged_file_permissions_and_userspace_paths"
        elif self.require_network_isolation:
            execution_mode = "BUBBLEWRAP"
            filesystem_enforcement = "kernel_mounts"
        else:
            execution_mode = "LOCAL_PROCESS"
            filesystem_enforcement = "none"
        return {
            "runtime_profile_id": self.runtime_profile.runtime_profile_id,
            "environment_fingerprint": self.environment_fingerprint,
            "network_isolation_required": self.require_network_isolation,
            "network_isolation_active": self.network_isolation_active,
            "execution_mode": execution_mode,
            "filesystem_enforcement": filesystem_enforcement,
            "timeout_seconds": self.timeout_seconds,
            "max_output_chars": self.max_output_chars,
        }

    def __enter__(self) -> "PythonExecutionEnvironment":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback_value: Any) -> None:
        self.close()

    def start(self) -> None:
        """Stage the case and start its persistent Python interpreter."""

        if self._closed:
            raise PythonRuntimeClosedError("runtime was closed; call reset() to create a new run")
        if self._started:
            return
        windows_local = self.runtime_profile.runtime_profile_id == "flow-python-windows-local-v1"
        bwrap = shutil.which("bwrap") if self.require_network_isolation else None
        if self.require_network_isolation and bwrap is None:
            raise RuntimeUnavailableError(
                "bubblewrap is required to enforce the primary no-web execution condition"
            )

        staging_parent = str(self.workspace_parent) if self.workspace_parent else None
        self._staged_root = Path(tempfile.mkdtemp(prefix="flowintentbench-case-", dir=staging_parent))
        self._case_dir = self._staged_root / "case"
        self._workspace_dir = self._staged_root / "workspace"
        self._case_dir.mkdir()
        self._workspace_dir.mkdir()
        control_parent: socket.socket | None = None
        control_child: socket.socket | None = None
        control_listener: socket.socket | None = None
        if windows_local:
            control_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            control_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            control_listener.bind(("127.0.0.1", 0))
            control_listener.listen(1)
            # Native Windows Conda/Python startup can be slow when the frozen
            # environment is first materialized or antivirus scans its DLLs.
            control_listener.settimeout(120)
            self._control_listener = control_listener
            self._control_token = secrets.token_bytes(32)
        else:
            if not hasattr(socket, "AF_UNIX"):
                self._cleanup_staging()
                raise RuntimeUnavailableError(
                    "isolated Python runtime failed to start: this platform lacks AF_UNIX socket support"
                )
            control_parent, control_child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            self._control_parent = control_parent
            self._control_fd = control_child.fileno()
        try:
            self._stage_case()
            command = self._sandbox_command(bwrap)
        except Exception:
            if control_parent is not None:
                control_parent.close()
            if control_child is not None:
                control_child.close()
            if control_listener is not None:
                control_listener.close()
            self._control_parent = None
            self._control_fd = None
            self._control_listener = None
            self._control_token = None
            self._cleanup_staging()
            raise
        if windows_local:
            assert control_listener is not None
            host, port = control_listener.getsockname()
            assert self._control_token is not None
            control_environment = {
                "FLOWINTENTBENCH_CONTROL_HOST": str(host),
                "FLOWINTENTBENCH_CONTROL_PORT": str(port),
                "FLOWINTENTBENCH_CONTROL_TOKEN": self._control_token.hex(),
            }
        else:
            control_environment = {"FLOWINTENTBENCH_CONTROL_FD": str(self._control_fd)}
        if os.name == "nt":
            runtime_path = str(Path(self.python_executable).parent)
            system_root = os.environ.get("SystemRoot")
            if system_root:
                runtime_path += os.pathsep + str(Path(system_root) / "System32")
        else:
            runtime_path = (
                "/runtime-python/bin:/usr/bin:/bin"
                if self.frozen_environment is not None and self.require_network_isolation
                else f"{Path(self.python_executable).parent}:/usr/bin:/bin"
            )
        environment = {
            "PATH": runtime_path,
            "HOME": str(self._staged_root if windows_local else "/tmp"),
            "TMPDIR": str(self._workspace_dir if windows_local else "/tmp"),
            "LANG": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            **control_environment,
        }
        if os.name == "nt":
            # Python's Winsock provider lookup needs the Windows system
            # directory variables even though the child receives a sanitized
            # environment rather than inheriting the host environment.
            for name in ("SystemRoot", "WINDIR", "TEMP", "TMP", "COMSPEC", "PATHEXT"):
                value = os.environ.get(name)
                if value:
                    environment[name] = value
        if self.frozen_environment is not None and os.name != "nt":
            environment["LD_LIBRARY_PATH"] = str(self.frozen_environment.base_prefix / "lib")
        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": environment,
            "cwd": str(self._staged_root if windows_local else self._workspace_dir),
        }
        if control_child is not None:
            popen_kwargs["pass_fds"] = (control_child.fileno(),)
        if os.name != "nt" and self.runtime_profile.network_available:
            # PRoot can leave its traced Python alive when its leader exits.
            # Own a separate group so timeout/cleanup reaches every case child.
            popen_kwargs["start_new_session"] = True
            self._owns_process_group = True
        try:
            self._process = subprocess.Popen(command, **popen_kwargs)
        except OSError as exc:
            if control_parent is not None:
                control_parent.close()
            if control_child is not None:
                control_child.close()
            if control_listener is not None:
                control_listener.close()
            self._control_parent = None
            self._control_fd = None
            self._control_listener = None
            self._control_token = None
            self._cleanup_staging()
            raise RuntimeUnavailableError(f"could not start isolated Python runtime: {exc}") from exc
        if control_child is not None:
            control_child.close()
        if windows_local:
            assert control_listener is not None
            assert self._control_token is not None
            try:
                control_parent, _ = control_listener.accept()
                control_parent.settimeout(None)
                expected = self._control_token.hex().encode("ascii")
                received = bytearray()
                while len(received) < len(expected):
                    chunk = control_parent.recv(len(expected) - len(received))
                    if not chunk:
                        raise OSError("Windows runtime control channel closed during handshake")
                    received.extend(chunk)
                if bytes(received) != expected:
                    raise OSError("Windows runtime control channel authentication failed")
                self._control_parent = control_parent
            except (OSError, TimeoutError) as exc:
                if control_parent is not None:
                    control_parent.close()
                self._stop_process()
                self._cleanup_staging()
                raise RuntimeUnavailableError(
                    f"Windows local Python runtime failed to connect: {exc}"
                ) from exc
            finally:
                control_listener.close()
                self._control_listener = None
                self._control_token = None
        self._stdout_thread = Thread(
            target=self._drain_process_stream,
            args=(self._process, "stdout"),
            daemon=True,
        )
        self._stderr_thread = Thread(
            target=self._drain_process_stream,
            args=(self._process, "stderr"),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            ready = self._read_response(120 if windows_local else 10)
            if ready != {"runtime_ready": True}:
                raise OSError("invalid Python runtime startup acknowledgement")
        except (OSError, TimeoutError) as exc:
            self._stop_process()
            self._cleanup_staging()
            raise RuntimeUnavailableError(f"isolated Python runtime failed to start: {exc}") from exc
        self._started = True

    def execute(self, code: str) -> PythonExecutionResult:
        """Execute one code block and preserve its interpreter state."""

        if not isinstance(code, str):
            raise TypeError("code must be a string")
        if not self.started:
            diagnostics = self._process_diagnostics()
            process_status = self._process.poll() if self._process is not None else None
            detail = f"runtime is not running (process_status={process_status})"
            if diagnostics:
                detail += f"; {diagnostics}"
            raise PythonRuntimeClosedError(detail)
        assert self._process is not None
        assert self._control_parent is not None
        self._execution_index += 1
        index = self._execution_index
        started = time.perf_counter()
        discovery_error = _root_recursive_discovery_reason(code)
        if discovery_error is not None:
            return PythonExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exception=PythonExecutionException(
                    "FilesystemDiscoveryError",
                    discovery_error,
                    "",
                ),
                duration_seconds=time.perf_counter() - started,
                execution_index=index,
                output_truncated=False,
                stdout_chars_total=0,
                stderr_chars_total=0,
                returned_stdout_chars=0,
                returned_stderr_chars=0,
                filesystem_discovery_blocked=True,
            )
        request = json.dumps(
            {"code": code, "output_limit": self.max_output_chars},
            ensure_ascii=False,
        ).encode("utf-8") + b"\n"
        try:
            self._control_parent.sendall(request)
            payload = self._read_response(self.timeout_seconds)
        except (BrokenPipeError, OSError) as exc:
            diagnostics = self._process_diagnostics()
            self._terminate_process()
            return PythonExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exception=PythonExecutionException(
                    type(exc).__name__,
                    str(exc) + (f"; infrastructure diagnostics: {diagnostics}" if diagnostics else ""),
                    "",
                ),
                duration_seconds=time.perf_counter() - started,
                execution_index=index,
            )
        except TimeoutError as exc:
            self._terminate_process()
            return PythonExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exception=PythonExecutionException("TimeoutError", str(exc), ""),
                duration_seconds=time.perf_counter() - started,
                execution_index=index,
            )

        exception_payload = payload.get("exception")
        exception = (
            PythonExecutionException(
                type=str(exception_payload.get("type", "RuntimeError")),
                message=str(exception_payload.get("message", "")),
                traceback=str(exception_payload.get("traceback", "")),
            )
            if isinstance(exception_payload, dict)
            else None
        )
        return PythonExecutionResult(
            success=bool(payload.get("success", False)),
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
            exception=exception,
            duration_seconds=float(payload.get("duration_seconds", time.perf_counter() - started)),
            execution_index=index,
            output_truncated=bool(payload.get("output_truncated", False)),
            stdout_chars_total=(
                int(payload["stdout_chars_total"])
                if payload.get("stdout_chars_total") is not None
                else None
            ),
            stderr_chars_total=(
                int(payload["stderr_chars_total"])
                if payload.get("stderr_chars_total") is not None
                else None
            ),
            returned_stdout_chars=(
                int(payload["returned_stdout_chars"])
                if payload.get("returned_stdout_chars") is not None
                else len(str(payload.get("stdout", "")))
            ),
            returned_stderr_chars=(
                int(payload["returned_stderr_chars"])
                if payload.get("returned_stderr_chars") is not None
                else len(str(payload.get("stderr", "")))
            ),
        )

    def reset(self) -> None:
        """Destroy the current process and staging, then start a clean case run."""

        self._stop_process()
        self._cleanup_staging()
        self._started = False
        self._closed = False
        self._execution_index = 0
        self.start()

    def close(self) -> None:
        """Destroy the interpreter and remove staged model-visible resources."""

        if self._closed:
            return
        self._stop_process()
        self._cleanup_staging()
        self._started = False
        self._closed = True

    def _stage_case(self) -> None:
        assert self._case_dir is not None
        case_payload = self.loaded_case.case.model_dump(mode="json")
        _write_json(self._case_dir / "case_input.json", case_payload)
        (self._case_dir / "case_input.json").chmod(0o444)

        for item in self.loaded_case.data_files:
            _copy_read_only(item.path, _safe_destination(self._case_dir, item.specification.path))
        for item in self.loaded_case.geometry_assets:
            _copy_read_only(item.path, _safe_destination(self._case_dir, item.specification.path))
        if self.manifest is not None:
            metadata_path = self._case_dir / "reader_metadata.json"
            _write_json(metadata_path, reader_metadata_for_manifest(self.manifest))
            metadata_path.chmod(0o444)
        runtime_contract_path = self._case_dir / "runtime_contract.json"
        _write_json(
            runtime_contract_path,
            _runtime_contract(
                self.frozen_environment,
                max_output_chars=self.max_output_chars,
                runtime_profile=self.runtime_profile,
            ),
        )
        runtime_contract_path.chmod(0o444)
        visible_files = {
            "case_root": "/case",
            "workspace_root": "/workspace",
            "read_only": [
                "case_input.json",
                "case_files.json",
                *[item.specification.path for item in self.loaded_case.data_files],
                *[item.specification.path for item in self.loaded_case.geometry_assets],
                *(["reader_metadata.json"] if self.manifest is not None else []),
                "runtime_contract.json",
            ],
            "notes": (
                "This bounded index lists only model-visible case files; "
                "construction and GT resources are hidden."
            ),
        }
        index_path = self._case_dir / "case_files.json"
        _write_json(index_path, visible_files)
        index_path.chmod(0o444)

    def _host_network_command(self) -> list[str]:
        """Keep virtual case paths, using the host network and no namespaces.

        PRoot is a userspace path mapper, not a kernel security boundary.
        Only runtime libraries and staged case/workspace paths are mapped;
        the repository, provider credentials and scoring inputs are omitted.
        """
        proot = shutil.which("proot")
        if proot is None:
            raise RuntimeUnavailableError("host-network profile requires proot in the active environment")
        assert self._staged_root is not None
        guest = self._staged_root / "guest"
        guest.mkdir()
        for name in ("tmp", "proc", "dev", "etc", "workspace", "case"):
            (guest / name).mkdir()
        command = [proot, "-r", str(guest), "-w", "/workspace"]
        paths = ["/usr", "/bin", "/lib", "/lib64", "/etc/resolv.conf", "/etc/hosts",
                 "/etc/nsswitch.conf", "/etc/ssl", "/etc/ld.so.cache",
                 "/dev/null", "/dev/zero", "/dev/urandom", "/dev/random",
                 "/proc/cpuinfo", "/proc/meminfo"]
        for value in paths:
            if Path(value).exists():
                command.extend(["-b", value])
        command.extend(["-b", f"{self.case_dir}:/case", "-b", f"{self.workspace_dir}:/workspace"])
        if self.frozen_environment is not None:
            command.extend(["-b", str(self.frozen_environment.root), "-b", str(self.frozen_environment.base_prefix / "lib")])
        else:
            command.extend(["-b", str(Path(self.python_executable).resolve().parent.parent)])
        command.extend([self.python_executable, "-I", "-u", "-c", _CHILD_SCRIPT])
        return command

    def _sandbox_command(self, bwrap: str | None) -> list[str]:
        if self.runtime_profile.runtime_profile_id == "flow-python-windows-local-v1":
            return [self.python_executable, "-I", "-u", "-c", _CHILD_SCRIPT]
        if self.runtime_profile.network_available:
            return self._host_network_command()
        if not self.require_network_isolation:
            return [self.python_executable, "-I", "-u", "-c", _CHILD_SCRIPT]
        assert bwrap is not None
        command = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--setenv",
            "PATH",
            "/runtime-python/bin:/usr/bin:/bin"
            if self.frozen_environment is not None
            else f"{Path(self.python_executable).parent}:/usr/bin:/bin",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "FLOWINTENTBENCH_CONTROL_FD",
            str(self._control_fd) if self._control_fd is not None else "3",
        ]
        if self.frozen_environment is not None:
            command.extend(
                [
                    "--setenv",
                    "LD_LIBRARY_PATH",
                    str(self.frozen_environment.base_prefix / "lib"),
                ]
            )
        for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"):
            if Path(path).exists():
                command.extend(["--ro-bind", path, path])
        if self.frozen_environment is not None:
            base_prefix = self.frozen_environment.base_prefix
            command.extend(
                [
                    "--dir",
                    str(base_prefix.parent.parent),
                    "--dir",
                    str(base_prefix.parent),
                    "--dir",
                    str(base_prefix),
                    "--ro-bind",
                    str(self.frozen_environment.root),
                    "/runtime-python",
                    "--ro-bind",
                    str(base_prefix / "lib"),
                    str(base_prefix / "lib"),
                ]
            )
            base_site = (
                base_prefix
                / "lib"
                / f"python{self.environment_fingerprint['python_version'].rsplit('.', 1)[0]}"
                / "site-packages"
            )
            if base_site.is_dir():
                command.extend(["--tmpfs", str(base_site)])
            runtime_python = "/runtime-python/bin/python"
        else:
            prefix = Path(self.python_executable).resolve().parent
            while prefix.parent != prefix and prefix.name not in {"bin", "Scripts"}:
                prefix = prefix.parent
            if prefix.name in {"bin", "Scripts"}:
                prefix = prefix.parent
            if not str(Path(self.python_executable).resolve()).startswith("/usr/"):
                command.extend(["--ro-bind", str(prefix), str(prefix)])
            runtime_python = self.python_executable
        command.extend(
            [
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--ro-bind",
                str(self.case_dir),
                "/case",
                "--bind",
                str(self.workspace_dir),
                "/workspace",
                "--chdir",
                "/workspace",
                runtime_python,
                "-I",
                "-u",
                "-c",
                _CHILD_SCRIPT,
            ]
        )
        return command

    def _read_response(self, timeout: float) -> dict[str, Any]:
        assert self._control_parent is not None
        selector = selectors.DefaultSelector()
        selector.register(self._control_parent, selectors.EVENT_READ)
        deadline = time.perf_counter() + timeout
        try:
            while True:
                newline = self._control_buffer.find(b"\n")
                if newline >= 0:
                    line = bytes(self._control_buffer[:newline])
                    del self._control_buffer[: newline + 1]
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f"Python execution exceeded {timeout:g} seconds")
                events = selector.select(remaining)
                if not events:
                    raise TimeoutError(f"Python execution exceeded {timeout:g} seconds")
                chunk = self._control_parent.recv(65536)
                if not chunk:
                    raise OSError(
                        f"isolated Python process exited with code {self._process.poll() if self._process else None}; "
                        f"{self._process_diagnostics()}"
                    )
                self._control_buffer.extend(chunk)
        finally:
            selector.close()
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OSError("isolated Python returned malformed execution result") from exc
        if not isinstance(payload, dict):
            raise OSError("isolated Python returned a non-object execution result")
        return payload

    def _drain_process_stream(
        self,
        process: subprocess.Popen[bytes],
        stream_name: str,
    ) -> None:
        stream = getattr(process, stream_name)
        if stream is None:
            return
        for line in stream:
            with self._stderr_lock:
                target = self._stdout_lines if stream_name == "stdout" else self._stderr_lines
                target.append(line.decode("utf-8", errors="replace").rstrip())

    def _process_stderr(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_lines)

    def _process_diagnostics(self) -> str:
        with self._stderr_lock:
            diagnostics = []
            if self._stdout_lines:
                diagnostics.append("runtime stdout: " + "\n".join(self._stdout_lines))
            if self._stderr_lines:
                diagnostics.append("runtime stderr: " + "\n".join(self._stderr_lines))
            return "\n".join(diagnostics)

    def _terminate_process(self) -> None:
        self._stop_process()
        self._started = False

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        control = self._control_parent
        self._control_parent = None
        self._control_fd = None
        listener = self._control_listener
        self._control_listener = None
        self._control_token = None
        if control is not None:
            control.close()
        if listener is not None:
            listener.close()
        if process is None:
            return
        if self._owns_process_group:
            import signal
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self._owns_process_group:
            import signal
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._owns_process_group = False
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=1)
        # BufferedReader.close acquires the same lock as a blocked reader.
        # Never wait indefinitely here if a detached descendant kept the pipe.
        if process.stdout is not None and (self._stdout_thread is None or not self._stdout_thread.is_alive()):
            process.stdout.close()
        if process.stderr is not None and (self._stderr_thread is None or not self._stderr_thread.is_alive()):
            process.stderr.close()
        self._stdout_thread = None
        self._stderr_thread = None

    def _cleanup_staging(self) -> None:
        if self._staged_root is not None:
            shutil.rmtree(self._staged_root, ignore_errors=True)
        self._staged_root = None
        self._case_dir = None
        self._workspace_dir = None

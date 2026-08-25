"""Bounded parent-side lifecycle for the private Morris authority child."""

from __future__ import annotations

import http.client
import json
import logging
import os
import secrets
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from .contracts import MORRIS_JOB_SCHEMA_ID, MORRIS_REQUEST_SCHEMA_ID

AUTHORITY_TOKEN_ENV = "ROC_MORRIS_AUTHORITY_CHILD_TOKEN"
API_PREFIX = "/api/rate-of-closure/v1"
CAPABILITY_PATH = f"{API_PREFIX}/morris/capabilities"
_CONTROL_PATH = "/_control/shutdown"
_EXPECTED_CAPABILITY = {
    "schema_id": "rate-of-closure/morris-authority-capability",
    "schema_version": 1,
    "available": True,
    "api_prefix": API_PREFIX,
    "request_schema_id": MORRIS_REQUEST_SCHEMA_ID,
    "job_schema_id": MORRIS_JOB_SCHEMA_ID,
}
_POLL_INTERVAL_S = 0.05
_CLOSE_TIMEOUT_S = 5.0
_MAX_CAPABILITY_BYTES = 1_024
_DIAGNOSTIC_TAIL_CHARS = 2_048
logger = logging.getLogger(__name__)


@dataclass
class MorrisAuthorityRuntime:
    """One authenticated child and its server-only Vite proxy environment."""

    process: subprocess.Popen[str]
    base_url: str
    token: str = field(repr=False)
    _closed: bool = False

    @classmethod
    def start(
        cls,
        source_root: Path | None = None,
        startup_timeout_s: float = 10.0,
    ) -> MorrisAuthorityRuntime:
        """Spawn the child and return only after authenticated capability proof."""
        timeout = _startup_timeout(startup_timeout_s)
        root = _source_root(source_root)
        token = secrets.token_urlsafe(32)
        process, diagnostics = _spawn_child(root, token)
        deadline = time.monotonic() + timeout
        try:
            port = _ready_port(process, diagnostics, timeout)
            base_url = f"http://127.0.0.1:{port}"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"Morris authority startup timed out: "
                    f"{_startup_diagnosis(process, diagnostics)}"
                )
            _wait_authenticated(process, port, token, remaining, diagnostics)
            return cls(process, base_url, token)
        except BaseException:
            try:
                _force_reap(process)
            except BaseException:
                logger.warning("Morris authority startup cleanup failed")
            raise
        finally:
            diagnostics.close()

    @property
    def authorization_headers(self) -> dict[str, str]:
        """Return detached private request headers for parent-only control."""
        return {"Authorization": f"Bearer {self.token}"}

    @property
    def vite_env(self) -> dict[str, str]:
        """Return detached server-only proxy variables for the Vite process."""
        return {
            "ROC_MORRIS_AUTHORITY_URL": self.base_url,
            "ROC_MORRIS_AUTHORITY_TOKEN": self.token,
        }

    def close(self) -> None:
        """Idempotently request graceful shutdown, then reap with bounded fallback."""
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            _request_shutdown(self.process, self.base_url, self.authorization_headers)
        _force_reap(self.process, allow_grace=True)

    def __enter__(self) -> MorrisAuthorityRuntime:
        return self

    def __exit__(self, *_details: object) -> None:
        self.close()


def _startup_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("startup_timeout_s must be within (0, 60]")
    timeout = float(value)
    if not 0 < timeout <= 60:
        raise ValueError("startup_timeout_s must be within (0, 60]")
    return timeout


def _source_root(value: Path | None) -> Path:
    root = Path(__file__).resolve().parents[3] if value is None else Path(value)
    root = root.resolve()
    if not root.is_dir() or not (root / "rate_of_closure").is_dir():
        raise ValueError("source_root must contain the rate_of_closure package")
    return root


def _spawn_child(root: Path, token: str) -> tuple[subprocess.Popen[str], IO[str]]:
    # Deliberately NOT `.resolve()`. Inside a virtualenv on POSIX, `bin/python`
    # is a symlink to the base interpreter, so resolving it hands the child the
    # *base* interpreter and silently drops the venv's site-packages — the child
    # then dies on `import uvicorn`. Windows venvs copy the executable instead,
    # which is why this only ever failed on Linux CI. `sys.executable` is
    # already absolute; keep the venv identity it encodes.
    interpreter = Path(sys.executable)
    if not interpreter.is_absolute() or not interpreter.is_file():
        raise RuntimeError("current Python interpreter path is unavailable")
    environment = os.environ.copy()
    environment[AUTHORITY_TOKEN_ENV] = token
    environment["PYTHONUNBUFFERED"] = "1"
    prior_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(root) if not prior_path else os.pathsep.join((str(root), prior_path))
    )
    # Capture stderr to a temporary file rather than DEVNULL. A child that dies
    # before announcing its port is otherwise indistinguishable from one that
    # announced a malformed port, and the reason is gone. A file (not a PIPE)
    # keeps that diagnosis without risking a deadlock on a full pipe buffer,
    # since nothing drains the child's stderr while we wait on stdout.
    diagnostics = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    return (
        subprocess.Popen(
            [str(interpreter), "-u", "-m", "rate_of_closure.application.morris.child"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=diagnostics,
            text=True,
            bufsize=1,
            shell=False,
        ),
        diagnostics,
    )


def _startup_diagnosis(
    process: subprocess.Popen[str],
    diagnostics: IO[str] | None,
    stdout_preview: str = "",
) -> str:
    """Summarise why a child never reached readiness, for the raised error."""
    exit_code = process.poll()
    stderr_captured = ""
    if diagnostics is not None:
        try:
            diagnostics.seek(0)
            stderr_captured = diagnostics.read(_DIAGNOSTIC_TAIL_CHARS).strip()
        except OSError:  # pragma: no cover - diagnosis must never mask the failure
            stderr_captured = ""
    stdout_captured = stdout_preview.strip()
    if not stdout_captured and exit_code is not None and process.stdout is not None:
        try:
            stdout_captured = process.stdout.read(_DIAGNOSTIC_TAIL_CHARS).strip()
        except OSError:
            stdout_captured = ""
    state = "still running" if exit_code is None else f"exited with {exit_code}"
    return (
        f"child {state}; "
        f"stdout: {stdout_captured or '<empty>'}; "
        f"stderr: {stderr_captured or '<empty>'}"
    )


def _ready_port(
    process: subprocess.Popen[str], diagnostics: IO[str], timeout_s: float
) -> int:
    if process.stdout is None:
        raise RuntimeError("authority readiness channel is unavailable")
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="morris-ready")
    future = pool.submit(process.stdout.readline)
    try:
        line = future.result(timeout=timeout_s)
    except TimeoutError as exc:
        raise RuntimeError(
            f"Morris authority child readiness timed out: "
            f"{_startup_diagnosis(process, diagnostics)}"
        ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if not line:
        # EOF, not a malformed announcement: the child closed stdout without
        # ever printing a port. Reporting this as "invalid readiness" sent
        # every such failure to the wrong diagnosis.
        raise RuntimeError(
            f"Morris authority child closed stdout before announcing a port: "
            f"{_startup_diagnosis(process, diagnostics)}"
        )
    if not line.endswith("\n") or not line[:-1].isdigit() or len(line) > 6:
        raise RuntimeError(
            f"invalid Morris authority child readiness: "
            f"{_startup_diagnosis(process, diagnostics, stdout_preview=line)}"
        )
    port = int(line[:-1])
    if not 1 <= port <= 65535 or line != f"{port}\n":
        raise RuntimeError(
            f"invalid Morris authority child readiness: "
            f"{_startup_diagnosis(process, diagnostics, stdout_preview=line)}"
        )
    return port


def _wait_authenticated(
    process: subprocess.Popen[str],
    port: int,
    token: str,
    timeout_s: float,
    diagnostics: IO[str] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    headers = {"Authorization": f"Bearer {token}"}
    while time.monotonic() < deadline:
        if process.poll() is not None:
            diag = _startup_diagnosis(process, diagnostics)
            raise RuntimeError(
                f"Morris authority child exited before readiness: {diag}"
            )
        try:
            status, media_type, document = _direct_request(
                port, "GET", CAPABILITY_PATH, headers
            )
            if (
                status == 200
                and media_type == "application/json"
                and document == _EXPECTED_CAPABILITY
            ):
                return
        except (OSError, http.client.HTTPException, ValueError):
            pass
        time.sleep(_POLL_INTERVAL_S)
    diag = _startup_diagnosis(process, diagnostics)
    raise RuntimeError(f"Morris authority authenticated readiness timed out: {diag}")


def _direct_request(
    port: int, method: str, path: str, headers: dict[str, str]
) -> tuple[int, str, object]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
    try:
        connection.request(method, path, headers=headers)
        response = connection.getresponse()
        body = response.read(_MAX_CAPABILITY_BYTES + 1)
        if len(body) > _MAX_CAPABILITY_BYTES:
            raise ValueError("authority response exceeds the readiness bound")
        media_type = response.getheader("Content-Type", "").split(";", 1)[0].lower()
        document: Any = json.loads(body.decode("utf-8", errors="strict"))
        return response.status, media_type, document
    finally:
        connection.close()


def _request_shutdown(
    process: subprocess.Popen[str], base_url: str, headers: dict[str, str]
) -> None:
    port = int(base_url.rsplit(":", 1)[1])
    try:
        _direct_request(port, "POST", _CONTROL_PATH, headers)
        process.wait(timeout=_CLOSE_TIMEOUT_S)
    except (OSError, ValueError, http.client.HTTPException, subprocess.TimeoutExpired):
        return


def _force_reap(process: subprocess.Popen[str], allow_grace: bool = False) -> None:
    reap_error: BaseException | None = None
    try:
        if process.poll() is not None:
            process.wait()
        elif allow_grace:
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_CLOSE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=_CLOSE_TIMEOUT_S)
    except BaseException as error:
        reap_error = error
    try:
        stream = process.stdout
        if stream is not None and not stream.closed:
            stream.close()
    except BaseException:
        if reap_error is None:
            raise
        logger.warning("Morris authority pipe cleanup failed")
    if reap_error is not None:
        raise reap_error.with_traceback(reap_error.__traceback__)


__all__ = ["MorrisAuthorityRuntime"]

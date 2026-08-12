"""Isolated process lifecycle for the loopback regional-ground authority."""

from __future__ import annotations

import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.client import HTTPConnection
from pathlib import Path
from types import MappingProxyType
from typing import Final

import platformdirs

from .api import CAPABILITY_PATH
from .capability import AuthorityCapability

LOOPBACK_HOST: Final = "127.0.0.1"
AUTHORITY_URL_ENV: Final = "ROC_AUTHORITY_URL"
AUTHORITY_TOKEN_ENV: Final = "ROC_AUTHORITY_TOKEN"
AUTHORITY_STATE_ROOT_ENV: Final = "ROC_AUTHORITY_STATE_ROOT"
AUTHORITY_PORT_ENV: Final = "ROC_AUTHORITY_PORT"
AUTHORITY_APP_FACTORY_ENV: Final = "ROC_AUTHORITY_APP_FACTORY"
_READINESS_TIMEOUT_S: Final = 15.0
_READINESS_INTERVAL_S: Final = 0.05
_SHUTDOWN_TIMEOUT_S: Final = 10.0
_PORT_REPORT_TIMEOUT_S: Final = 15.0
DEFAULT_AUTHORITY_APP_FACTORY: Final = (
    "rate_of_closure.web_authority.server:create_app_from_environment"
)


@dataclass(frozen=True, slots=True)
class AuthorityProcessSpec:
    """Command and private environment for one isolated authority process."""

    command: tuple[str, ...]
    environment: Mapping[str, str] = field(repr=False)
    port: int


@dataclass(slots=True)
class AuthorityRuntime:
    """Owned process and Vite proxy environment for one launcher session."""

    process: subprocess.Popen[bytes]
    token: str = field(repr=False)
    port: int

    @property
    def vite_environment(self) -> dict[str, str]:
        """Return the private Vite dev-server proxy configuration."""
        return {
            AUTHORITY_URL_ENV: f"http://{LOOPBACK_HOST}:{self.port}",
            AUTHORITY_TOKEN_ENV: self.token,
        }

    def close(self) -> None:
        """Terminate and reap the isolated authority child process."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        if self.process.stdout is not None:
            self.process.stdout.close()


def _app_factory(value: str) -> str:
    """Validate one import-only Uvicorn application-factory identity."""
    if type(value) is not str or value != value.strip() or len(value) > 240:
        raise ValueError("authority app_factory must be bounded trimmed text")
    module, separator, function = value.partition(":")
    valid_module = module and all(part.isidentifier() for part in module.split("."))
    if separator != ":" or not valid_module or not function.isidentifier():
        raise ValueError("authority app_factory must be module.path:function")
    return value


def _authority_command() -> tuple[str, ...]:
    """Build the fixed child command without exposing token or selected port."""
    return (
        sys.executable,
        "-m",
        "rate_of_closure.web_authority.child",
    )


def build_authority_process_spec(
    *,
    token: str,
    port: int,
    source_root: Path,
    state_root: Path | None = None,
    app_factory: str = DEFAULT_AUTHORITY_APP_FACTORY,
) -> AuthorityProcessSpec:
    """Build a loopback-only process spec with its token outside the command."""
    if not token or token != token.strip():
        raise ValueError("authority token must be nonempty and trimmed")
    if port < 0 or port > 65_535:
        raise ValueError("authority port must lie within [0, 65535]")
    environment = os.environ.copy()
    inherited_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), inherited_path) if part
    )
    environment[AUTHORITY_TOKEN_ENV] = token
    environment[AUTHORITY_PORT_ENV] = str(port)
    environment[AUTHORITY_APP_FACTORY_ENV] = _app_factory(app_factory)
    if state_root is not None:
        if not state_root.is_absolute() or not state_root.is_dir():
            raise ValueError(
                "authority state_root must be an existing absolute directory"
            )
        if state_root.is_symlink():
            raise ValueError("authority state_root must not be a symbolic link")
        environment[AUTHORITY_STATE_ROOT_ENV] = str(state_root)
    return AuthorityProcessSpec(
        command=_authority_command(),
        environment=MappingProxyType(environment),
        port=port,
    )


def _read_child_port(process: subprocess.Popen[bytes]) -> int:
    """Receive the child-owned listener port over one private bounded pipe."""
    if process.stdout is None:
        raise RuntimeError("authority child port pipe is unavailable")
    stdout = process.stdout
    reports: queue.Queue[bytes] = queue.Queue(maxsize=1)

    def read_report() -> None:
        reports.put(stdout.readline(16))

    threading.Thread(target=read_report, daemon=True).start()
    try:
        report = reports.get(timeout=_PORT_REPORT_TIMEOUT_S)
    except queue.Empty as exc:
        raise RuntimeError("authority child did not report its listener") from exc
    try:
        source = report.decode("ascii")
        port = int(source.removesuffix("\n"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("authority child reported an invalid listener") from exc
    if not source.endswith("\n") or not 1 <= port <= 65_535:
        raise RuntimeError("authority child reported an invalid listener")
    return port


def _is_ready(runtime: AuthorityRuntime) -> bool:
    """Probe the authenticated capability route on the fixed loopback host."""
    connection = HTTPConnection(
        LOOPBACK_HOST,
        runtime.port,
        timeout=_READINESS_INTERVAL_S,
    )
    try:
        connection.request(
            "GET",
            CAPABILITY_PATH,
            headers={"Authorization": f"Bearer {runtime.token}"},
        )
        response = connection.getresponse()
        if response.status != 200:
            return False
        if response.headers.get_content_type() != "application/json":
            return False
        capability = AuthorityCapability.from_json(response.read(4_097).decode("utf-8"))
        return bool(capability.available and capability.regional_ground_execution)
    except (OSError, TimeoutError, UnicodeDecodeError, TypeError, ValueError):
        return False
    finally:
        connection.close()


def _wait_until_ready(runtime: AuthorityRuntime) -> None:
    """Wait a fixed bound for authenticated child readiness."""
    deadline = time.monotonic() + _READINESS_TIMEOUT_S
    while time.monotonic() < deadline:
        if runtime.process.poll() is not None:
            raise RuntimeError("local Python authority exited before readiness")
        if _is_ready(runtime):
            return
        time.sleep(_READINESS_INTERVAL_S)
    raise RuntimeError("local Python authority did not become ready")


def start_authority(
    *,
    source_root: Path,
    app_factory: str = DEFAULT_AUTHORITY_APP_FACTORY,
    state_root: Path | None = None,
) -> AuthorityRuntime:
    """Start and authenticate one isolated loopback authority process."""
    token = secrets.token_urlsafe(32)
    root = state_root or (
        platformdirs.user_state_path("rate-of-closure", appauthor=False)
        / "regional-ground-authority-v1"
    )
    root.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        root.chmod(0o700)
    spec = build_authority_process_spec(
        token=token,
        port=0,
        source_root=source_root,
        state_root=root,
        app_factory=app_factory,
    )
    process = subprocess.Popen(
        spec.command,
        env=dict(spec.environment),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        port = _read_child_port(process)
        runtime = AuthorityRuntime(process=process, token=token, port=port)
        _wait_until_ready(runtime)
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        raise
    return runtime

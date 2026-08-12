"""Isolated process lifecycle for the loopback regional-ground authority."""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from http.client import HTTPConnection
from pathlib import Path
from types import MappingProxyType
from typing import Final

from .api import CAPABILITY_PATH

LOOPBACK_HOST: Final = "127.0.0.1"
AUTHORITY_URL_ENV: Final = "ROC_AUTHORITY_URL"
AUTHORITY_TOKEN_ENV: Final = "ROC_AUTHORITY_TOKEN"
_READINESS_TIMEOUT_S: Final = 15.0
_READINESS_INTERVAL_S: Final = 0.05
_SHUTDOWN_TIMEOUT_S: Final = 5.0
DEFAULT_AUTHORITY_APP_FACTORY: Final = (
    "rate_of_closure.web_authority.server:create_app_from_environment"
)


@dataclass(frozen=True, slots=True)
class AuthorityProcessSpec:
    """Command and private environment for one isolated authority process."""

    command: tuple[str, ...]
    environment: Mapping[str, str]
    port: int


@dataclass(slots=True)
class AuthorityRuntime:
    """Owned process and Vite proxy environment for one launcher session."""

    process: subprocess.Popen[bytes]
    token: str
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
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=_SHUTDOWN_TIMEOUT_S)


def _app_factory(value: str) -> str:
    """Validate one import-only Uvicorn application-factory identity."""
    if type(value) is not str or value != value.strip() or len(value) > 240:
        raise ValueError("authority app_factory must be bounded trimmed text")
    module, separator, function = value.partition(":")
    valid_module = module and all(part.isidentifier() for part in module.split("."))
    if separator != ":" or not valid_module or not function.isidentifier():
        raise ValueError("authority app_factory must be module.path:function")
    return value


def _authority_command(port: int, app_factory: str) -> tuple[str, ...]:
    """Build the fixed Uvicorn child-process command."""
    return (
        sys.executable,
        "-m",
        "uvicorn",
        _app_factory(app_factory),
        "--factory",
        "--host",
        LOOPBACK_HOST,
        "--port",
        str(port),
        "--no-access-log",
        "--log-level=warning",
    )


def build_authority_process_spec(
    *,
    token: str,
    port: int,
    source_root: Path,
    app_factory: str = DEFAULT_AUTHORITY_APP_FACTORY,
) -> AuthorityProcessSpec:
    """Build a loopback-only process spec with its token outside the command."""
    if not token or token != token.strip():
        raise ValueError("authority token must be nonempty and trimmed")
    if port < 1 or port > 65_535:
        raise ValueError("authority port must lie within [1, 65535]")
    environment = os.environ.copy()
    inherited_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), inherited_path) if part
    )
    environment[AUTHORITY_TOKEN_ENV] = token
    return AuthorityProcessSpec(
        command=_authority_command(port, app_factory),
        environment=MappingProxyType(environment),
        port=port,
    )


def _reserve_loopback_port() -> int:
    """Reserve and release one ephemeral loopback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK_HOST, 0))
        return int(listener.getsockname()[1])


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
        return connection.getresponse().status == 200
    except (OSError, TimeoutError):
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
) -> AuthorityRuntime:
    """Start and authenticate one isolated loopback authority process."""
    token = secrets.token_urlsafe(32)
    spec = build_authority_process_spec(
        token=token,
        port=_reserve_loopback_port(),
        source_root=source_root,
        app_factory=app_factory,
    )
    process = subprocess.Popen(
        spec.command,
        env=dict(spec.environment),
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    runtime = AuthorityRuntime(process=process, token=token, port=spec.port)
    try:
        _wait_until_ready(runtime)
    except Exception:
        runtime.close()
        raise
    return runtime

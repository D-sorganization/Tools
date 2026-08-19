"""Owned loopback lifecycle for the source production web companion."""

from __future__ import annotations

import re
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.client import HTTPConnection
from pathlib import Path
from typing import Final

import uvicorn

from rate_of_closure.web_authority.runtime import (
    DEFAULT_AUTHORITY_APP_FACTORY,
    LOOPBACK_HOST,
    AuthorityRuntime,
)
from rate_of_closure.web_distribution.package_assets import (
    resolve_packaged_web_assets,
)

from .app import create_companion_app
from .bundle import CompanionWebBundle, build_companion_bundle
from .supervisor import AuthoritySupervisor

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STARTUP_TIMEOUT_S: Final = 15.0
_REQUEST_TIMEOUT_S: Final = 30.0
_SHUTDOWN_TIMEOUT_S: Final = 15


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            listener.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        listener.bind((LOOPBACK_HOST, 0))
        listener.listen(socket.SOMAXCONN)
        return listener
    except Exception:
        listener.close()
        raise


def _wait_until_ready(runtime: CompanionRuntime) -> None:
    deadline = time.monotonic() + _STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if not runtime.thread.is_alive():
            raise RuntimeError("production companion exited before readiness")
        connection = HTTPConnection(LOOPBACK_HOST, runtime.port, timeout=0.2)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            source = response.read(16)
            if response.status == 200 and source:
                return
        except (OSError, TimeoutError):
            pass
        finally:
            connection.close()
        time.sleep(0.05)
    raise RuntimeError("production companion did not become ready")


@dataclass(slots=True)
class CompanionRuntime:
    """One gateway listener, server thread, transport, and authority child."""

    supervisor: AuthoritySupervisor = field(repr=False)
    server: uvicorn.Server = field(repr=False)
    listener: socket.socket = field(repr=False)
    thread: threading.Thread = field(repr=False)
    port: int
    _closed: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @property
    def authority(self) -> AuthorityRuntime:
        """Return the supervisor's exact current authority runtime."""
        return self.supervisor.authority

    @property
    def url(self) -> str:
        """Return the only public loopback origin for this session."""
        return f"http://{LOOPBACK_HOST}:{self.port}/"

    def wait(self) -> None:
        """Block in the foreground while preserving KeyboardInterrupt handling."""
        while self.thread.is_alive():
            self.thread.join(0.5)

    def close(self) -> None:
        """Stop admission and reap gateway and authority within fixed bounds."""
        with self._lock:
            if self._closed:
                return
            self.server.should_exit = True
            self.thread.join(_SHUTDOWN_TIMEOUT_S)
            if self.thread.is_alive():
                self.server.force_exit = True
                self.thread.join(1.0)
            self.listener.close()
            self.supervisor.close()
            if self.thread.is_alive():
                raise RuntimeError("production companion server did not stop")
            self._closed = True


def _qualified_bundle(bundle: CompanionWebBundle | None) -> CompanionWebBundle:
    selected = (
        build_companion_bundle(resolve_packaged_web_assets())
        if bundle is None
        else bundle
    )
    if _COMMIT.fullmatch(selected.release_revision) is None:
        raise ValueError("production companion requires an exact release revision")
    return selected


def start_companion(
    *,
    bundle: CompanionWebBundle | None = None,
    state_root: Path | None = None,
    open_browser: bool = True,
    authority_app_factory: str = DEFAULT_AUTHORITY_APP_FACTORY,
) -> CompanionRuntime:
    """Start one exact-revision Python-only browser companion."""
    selected = _qualified_bundle(bundle)
    supervisor: AuthoritySupervisor | None = None
    listener: socket.socket | None = None
    runtime: CompanionRuntime | None = None
    try:
        supervisor = AuthoritySupervisor(
            source_root=_source_root(),
            state_root=state_root,
            timeout_s=_REQUEST_TIMEOUT_S,
            app_factory=authority_app_factory,
        )
        listener = _listener()
        port = int(listener.getsockname()[1])
        expected_host = f"{LOOPBACK_HOST}:{port}"
        app = create_companion_app(
            bundle=selected, transport=supervisor, expected_host=expected_host
        )
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                access_log=False,
                log_level="warning",
                server_header=False,
                timeout_graceful_shutdown=_SHUTDOWN_TIMEOUT_S,
            )
        )
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [listener]},
            name="rate-of-closure-web-companion",
            daemon=False,
        )
        runtime = CompanionRuntime(supervisor, server, listener, thread, port)
        thread.start()
        _wait_until_ready(runtime)
        if open_browser:
            webbrowser.open(runtime.url, new=1, autoraise=True)
        return runtime
    except Exception:
        if runtime is not None and runtime.thread.ident is not None:
            runtime.close()
        else:
            if listener is not None:
                listener.close()
            if supervisor is not None:
                supervisor.close()
        raise

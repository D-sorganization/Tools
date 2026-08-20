"""Single-flight authority replacement without automatic request replay."""

from __future__ import annotations

from pathlib import Path
from threading import RLock

from rate_of_closure.application.regional_ground_authority_transport import (
    AuthorityHttpResponse,
    LoopbackAuthorityHttpTransport,
)
from rate_of_closure.web_authority.runtime import (
    DEFAULT_AUTHORITY_APP_FACTORY,
    AuthorityRuntime,
    start_authority,
)


class AuthoritySupervisor:
    """Own one restartable child and serialize its short-lived HTTP requests."""

    def __init__(
        self,
        *,
        source_root: Path,
        state_root: Path | None,
        timeout_s: float,
        app_factory: str = DEFAULT_AUTHORITY_APP_FACTORY,
    ) -> None:
        self._source_root = source_root
        self._state_root = state_root
        self._timeout_s = timeout_s
        self._app_factory = app_factory
        self._lock = RLock()
        self._closed = False
        self._authority = self._start()
        self._transport = self._new_transport(self._authority)

    @property
    def authority(self) -> AuthorityRuntime:
        """Return the current owned runtime for lifecycle inspection."""
        with self._lock:
            return self._authority

    def _start(self) -> AuthorityRuntime:
        return start_authority(
            source_root=self._source_root,
            state_root=self._state_root,
            app_factory=self._app_factory,
        )

    def _new_transport(
        self, runtime: AuthorityRuntime
    ) -> LoopbackAuthorityHttpTransport:
        return LoopbackAuthorityHttpTransport(runtime, timeout_s=self._timeout_s)

    def _replace_dead_authority(self) -> None:
        self._transport.close()
        self._authority.close()
        replacement = self._start()
        self._authority = replacement
        self._transport = self._new_transport(replacement)

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None,
        maximum_bytes: int,
    ) -> AuthorityHttpResponse:
        """Restart only before a request; never retry after dispatch."""
        with self._lock:
            if self._closed:
                raise RuntimeError("authority supervisor is closed")
            if self._authority.process.poll() is not None:
                self._replace_dead_authority()
            return self._transport.request(method, path, body, maximum_bytes)

    def close(self) -> None:
        """Reject new work and reap the exact current child."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._transport.close()
            self._authority.close()

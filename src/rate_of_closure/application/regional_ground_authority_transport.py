"""Bounded authenticated HTTP transport for the loopback Python authority."""

from __future__ import annotations

from dataclasses import dataclass
from http.client import HTTPConnection, HTTPResponse
from threading import RLock
from typing import Final, Protocol

from rate_of_closure.web_authority.runtime import LOOPBACK_HOST, AuthorityRuntime

_JSON_MEDIA_TYPE: Final = "application/json"


@dataclass(frozen=True, slots=True)
class AuthorityHttpResponse:
    """Bounded response detached from its short-lived HTTP connection."""

    status: int
    media_type: str
    body: bytes


class RegionalGroundAuthorityTransport(Protocol):
    """Injectable transport boundary used by the UI-neutral submitter."""

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None,
        maximum_bytes: int,
    ) -> AuthorityHttpResponse:
        """Issue one bounded authority request."""

    def close(self) -> None:
        """Reject future requests and release transport resources."""


class LoopbackAuthorityHttpTransport:
    """Short-lived fixed-host connections authenticated by the runtime token."""

    def __init__(self, runtime: AuthorityRuntime, *, timeout_s: float) -> None:
        """Bind an exact owned runtime without taking lifecycle ownership."""
        if type(runtime) is not AuthorityRuntime:
            raise TypeError("runtime must be an exact AuthorityRuntime")
        if type(timeout_s) not in (int, float) or not 0.0 < timeout_s <= 30.0:
            raise ValueError("timeout_s must lie within (0, 30]")
        self._runtime = runtime
        self._timeout_s = float(timeout_s)
        self._closed = False
        self._lock = RLock()

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None,
        maximum_bytes: int,
    ) -> AuthorityHttpResponse:
        """Issue one authenticated request and enforce response wire bounds."""
        self._validate_request(method, path, body, maximum_bytes)
        with self._lock:
            if self._closed:
                raise RuntimeError("authority transport is closed")
            connection = HTTPConnection(
                LOOPBACK_HOST, self._runtime.port, timeout=self._timeout_s
            )
            try:
                connection.request(
                    method,
                    path,
                    body=body,
                    headers=self._headers(body),
                )
                response = connection.getresponse()
                self._validate_response_headers(response, maximum_bytes)
                payload = response.read(maximum_bytes + 1)
                if len(payload) > maximum_bytes:
                    raise ValueError("authority response exceeds maximum wire size")
                media_type = response.getheader("Content-Type", "")
                return AuthorityHttpResponse(
                    response.status,
                    media_type.split(";", 1)[0].strip().lower(),
                    payload,
                )
            finally:
                connection.close()

    def close(self) -> None:
        """Atomically reject future connections without closing the runtime."""
        with self._lock:
            self._closed = True

    def _headers(self, body: bytes | None) -> dict[str, str]:
        """Build fixed non-caching JSON and bearer headers."""
        headers = {
            "Accept": _JSON_MEDIA_TYPE,
            "Authorization": f"Bearer {self._runtime.token}",
            "Cache-Control": "no-store",
        }
        if body is not None:
            headers["Content-Type"] = _JSON_MEDIA_TYPE
        return headers

    @staticmethod
    def _validate_request(
        method: str, path: str, body: bytes | None, maximum_bytes: int
    ) -> None:
        """Reject requests outside the fixed authority surface."""
        if method not in ("GET", "POST"):
            raise ValueError("authority method must be GET or POST")
        unsafe = ("?", "#", "\\", "..")
        if (
            not path.startswith("/api/rate-of-closure/v1/")
            or any(part in path for part in unsafe)
            or any(ord(character) < 0x20 for character in path)
        ):
            raise ValueError("authority path is outside the fixed API root")
        if body is not None and type(body) is not bytes:
            raise TypeError("authority body must be bytes or None")
        if type(maximum_bytes) is not int or maximum_bytes < 1:
            raise ValueError("maximum_bytes must be a positive integer")

    @staticmethod
    def _validate_response_headers(response: HTTPResponse, maximum_bytes: int) -> None:
        """Reject encoded or declared-oversize authority payloads."""
        encoding = str(response.getheader("Content-Encoding", "identity")).lower()
        if encoding != "identity":
            raise ValueError("encoded authority responses are unsupported")
        declared = response.getheader("Content-Length")
        if declared is None:
            return
        try:
            length = int(declared)
        except ValueError as error:
            raise ValueError("authority Content-Length must be an integer") from error
        if str(length) != declared or length < 0 or length > maximum_bytes:
            raise ValueError("authority response exceeds maximum wire size")


__all__ = [
    "AuthorityHttpResponse",
    "LoopbackAuthorityHttpTransport",
    "RegionalGroundAuthorityTransport",
]

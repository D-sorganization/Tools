"""Strict direct-loopback HTTP client for the private Morris authority."""

from __future__ import annotations

import http.client
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeVar
from urllib.parse import urlsplit

from rate_of_closure.application._workspace_validation import unique_json_object

from .contracts import MorrisAuthorityRequest, parse_morris_request
from .response_contract import (
    MorrisCapability,
    MorrisResponseJob,
    parse_morris_capability,
    parse_morris_job,
)

_CAPABILITY_PATH = "/api/rate-of-closure/v1/morris/capabilities"
_JOBS_PATH = "/api/rate-of-closure/v1/morris/jobs"
_MAX_ERROR_BYTES = 8_192
# Ten Rate factors by seventeen targets produce 170 aggregated estimates. A
# 16 MiB cap leaves ample schema overhead while never buffering raw observations.
_MAX_SUCCESS_BYTES = 16 * 1024 * 1024
_ResponseT = TypeVar("_ResponseT")


class MorrisAuthorityHttpError(RuntimeError):
    """Sanitized authority response failure carrying bounded public data."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Morris authority request failed ({status}): {message}")
        self.status = status


@dataclass(frozen=True)
class MorrisAuthorityClient:
    """Authenticated no-proxy HTTP client restricted to IPv4 loopback."""

    base_url: str
    headers: Mapping[str, str] = field(repr=False)
    timeout_s: float = 5.0
    _port: int = field(init=False, repr=False)
    _secret: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        exact_origin = (
            parsed.scheme == "http"
            and parsed.hostname == "127.0.0.1"
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
            and parsed.port is not None
        )
        if not exact_origin:
            raise ValueError("base_url must be an exact numeric IPv4 loopback origin")
        authorization = self.headers.get("Authorization")
        if (
            not isinstance(authorization, str)
            or not all(32 <= ord(character) < 127 for character in authorization)
            or not authorization.startswith("Bearer ")
            or len(authorization) < 15
        ):
            raise ValueError("headers must contain one visible bearer authorization")
        if set(self.headers) != {"Authorization"}:
            raise ValueError("only the copied Authorization header is accepted")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not 0 < float(self.timeout_s) <= 60
        ):
            raise ValueError("timeout_s must be within (0, 60]")
        assert parsed.port is not None and authorization is not None
        detached = MappingProxyType({"Authorization": authorization})
        object.__setattr__(self, "base_url", f"http://127.0.0.1:{parsed.port}")
        object.__setattr__(self, "headers", detached)
        object.__setattr__(self, "_port", parsed.port)
        object.__setattr__(self, "_secret", authorization.removeprefix("Bearer "))

    def capability(self) -> MorrisCapability:
        """Fetch and parse the exact capability document."""
        document = self._request("GET", _CAPABILITY_PATH, None, 200)
        return _validated_success(parse_morris_capability, document)

    def create(self, request: MorrisAuthorityRequest | object) -> MorrisResponseJob:
        """Submit one canonical validated request."""
        parsed = (
            request
            if isinstance(request, MorrisAuthorityRequest)
            else parse_morris_request(request)
        )
        document = self._request(
            "POST", _JOBS_PATH, parsed.to_json_dict(), expected=202
        )
        return _validated_success(parse_morris_job, document)

    def status(self, job_id: str) -> MorrisResponseJob:
        """Fetch one job state."""
        document = self._request("GET", self._job_path(job_id), None, 200)
        return _validated_success(parse_morris_job, document)

    def cancel(self, job_id: str) -> MorrisResponseJob:
        """Request idempotent cancellation."""
        document = self._request(
            "DELETE", self._job_path(job_id), None, expected=(200, 202)
        )
        return _validated_success(parse_morris_job, document)

    def request_document(
        self,
        method: str,
        path: str,
        document: object | None,
        expected: int | tuple[int, ...],
    ) -> object:
        """Send a bounded request for another contract on the same authority."""
        return self._request(method, path, document, expected)

    def _job_path(self, job_id: str) -> str:
        valid = (
            isinstance(job_id, str)
            and bool(job_id)
            and job_id == job_id.strip()
            and all(
                character
                in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-"
                for character in job_id
            )
        )
        if not valid:
            raise ValueError("job_id must be a nonempty portable identifier")
        return f"{_JOBS_PATH}/{job_id}"

    def _request(
        self,
        method: str,
        path: str,
        document: object | None,
        expected: int | tuple[int, ...],
    ) -> object:
        body = _request_body(document)
        headers = dict(self.headers)
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection = http.client.HTTPConnection(
            "127.0.0.1", self._port, timeout=float(self.timeout_s)
        )
        try:
            connection.request(method, path, body=body, headers=headers)
            return self._response(connection.getresponse(), expected)
        except MorrisAuthorityHttpError:
            raise
        except (
            OSError,
            http.client.HTTPException,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise MorrisAuthorityHttpError(
                0, "authority transport or response validation failed"
            ) from exc
        finally:
            connection.close()

    def _response(
        self, response: http.client.HTTPResponse, expected: int | tuple[int, ...]
    ) -> object:
        statuses = (expected,) if isinstance(expected, int) else expected
        limit = _MAX_SUCCESS_BYTES if response.status in statuses else _MAX_ERROR_BYTES
        raw = _bounded_body(response, limit)
        media_type = (
            response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
        )
        if media_type != "application/json":
            raise MorrisAuthorityHttpError(
                response.status, "invalid response media type"
            )
        value = _strict_json(raw)
        if response.status not in statuses:
            message = _public_error(value).replace(self._secret, "[redacted]")
            raise MorrisAuthorityHttpError(response.status, message)
        return value


def _request_body(document: object | None) -> bytes | None:
    if document is None:
        return None
    return json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _validated_success(
    parser: Callable[[object], _ResponseT], document: object
) -> _ResponseT:
    try:
        return parser(document)
    except (TypeError, ValueError) as exc:
        raise MorrisAuthorityHttpError(
            0, "authority success response failed validation"
        ) from exc


def _bounded_body(response: http.client.HTTPResponse, limit: int) -> bytes:
    length = response.getheader("Content-Length")
    if length is not None and (not length.isdigit() or int(length) > limit):
        raise MorrisAuthorityHttpError(
            response.status, "response body exceeds the contract bound"
        )
    body = response.read(limit + 1)
    if len(body) > limit:
        raise MorrisAuthorityHttpError(
            response.status, "response body exceeds the contract bound"
        )
    return body


def _strict_json(raw: bytes) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(value)

    return json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=unique_json_object,
        parse_constant=reject_constant,
    )


def _public_error(value: object) -> str:
    if (
        not isinstance(value, dict)
        or set(value) != {"error"}
        or not isinstance(value["error"], str)
    ):
        return "authority rejected the request"
    message = value["error"]
    if (
        not message
        or message != message.strip()
        or any(
            ord(character) < 32 or 127 <= ord(character) <= 159 for character in message
        )
    ):
        return "authority rejected the request"
    return message[:512]


__all__ = ["MorrisAuthorityClient", "MorrisAuthorityHttpError"]

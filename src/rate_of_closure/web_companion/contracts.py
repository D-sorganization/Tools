"""Pure fail-closed request routing for the same-origin companion."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final

from rate_of_closure.application.regional_ground_authority_status import (
    MAX_AUTHORITY_JOB_STATUS_BYTES,
)
from rate_of_closure.application.regional_ground_execution_job import (
    MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
)
from rate_of_closure.application.regional_ground_execution_result import (
    MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES,
)
from rate_of_closure.application.regional_ground_job_preparation_request import (
    MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES,
)
from rate_of_closure.web_authority.api import (
    CAPABILITY_PATH,
    JOB_COLLECTION_PATH,
    JOB_PREPARATION_PATH,
)

_MAX_RAW_PATH_BYTES: Final = 512
_MAX_ERROR_BYTES: Final = 4_096
_PORTABLE_ASSET = re.compile(r"^assets/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STABLE_ID = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
_STATUS_PATH = re.compile(rf"^{re.escape(JOB_COLLECTION_PATH)}/({_STABLE_ID})$")
_CANCEL_PATH = re.compile(rf"^{re.escape(JOB_COLLECTION_PATH)}/({_STABLE_ID})/cancel$")
_RESULT_PATH = re.compile(rf"^{re.escape(JOB_COLLECTION_PATH)}/({_STABLE_ID})/result$")
_FORBIDDEN_HEADERS: Final = frozenset(
    {
        "authorization",
        "cookie",
        "forwarded",
        "proxy-authorization",
        "via",
        "access-control-request-headers",
        "access-control-request-method",
    }
)


class CompanionRouteKind(Enum):
    """Closed public route categories served by the companion."""

    INDEX = "index"
    ASSET = "asset"
    API = "api"


class CompanionApiOperation(Enum):
    """Exact authority operation selected by a public API route."""

    CAPABILITY = "capability"
    PREPARE = "prepare"
    SUBMIT = "submit"
    STATUS = "status"
    CANCEL = "cancel"
    RESULT = "result"


class CompanionRequestRejected(ValueError):
    """Bounded public rejection without internal or secret context."""

    def __init__(self, status_code: int, code: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True, slots=True)
class CompanionRequest:
    """Transport-neutral request metadata inspected before any body is read."""

    method: str
    raw_path: bytes
    query_string: bytes
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CompanionRoute:
    """One classified public route and its byte bounds."""

    kind: CompanionRouteKind
    asset_path: str | None = None
    upstream_path: str | None = None
    request_limit: int = 0
    response_limit: int = _MAX_ERROR_BYTES
    operation: CompanionApiOperation | None = None
    response_statuses: frozenset[int] = frozenset()


def _reject(status_code: int, code: str) -> CompanionRequestRejected:
    return CompanionRequestRejected(status_code, code)


def _normalized_headers(source: Mapping[str, str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in source.items():
        folded = key.lower()
        if folded in headers or not folded.isascii() or not isinstance(value, str):
            raise _reject(400, "invalid_headers")
        headers[folded] = value
    return headers


def _canonical_path(request: CompanionRequest) -> str:
    if request.query_string:
        raise _reject(400, "query_unsupported")
    source = request.raw_path
    if not source or len(source) > _MAX_RAW_PATH_BYTES:
        raise _reject(400, "invalid_path")
    try:
        path = source.decode("ascii")
    except UnicodeDecodeError as exc:
        raise _reject(400, "invalid_path") from exc
    if (
        not path.startswith("/")
        or "%" in path
        or "\\" in path
        or "//" in path
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        raise _reject(400, "invalid_path")
    return path


def _require_public_boundary(headers: Mapping[str, str], expected_host: str) -> None:
    if headers.get("host") != expected_host:
        raise _reject(403, "invalid_host")
    names = frozenset(headers)
    if names & _FORBIDDEN_HEADERS or any(
        name.startswith("x-forwarded-") for name in names
    ):
        raise _reject(403, "credentials_or_forwarding_rejected")


def _require_state_origin(headers: Mapping[str, str], expected_host: str) -> None:
    if headers.get("origin") != f"http://{expected_host}":
        raise _reject(403, "invalid_origin")
    if headers.get("sec-fetch-site") != "same-origin":
        raise _reject(403, "invalid_fetch_site")
    if headers.get("sec-fetch-mode") != "cors":
        raise _reject(403, "invalid_fetch_mode")
    if headers.get("sec-fetch-dest") != "empty":
        raise _reject(403, "invalid_fetch_destination")


def _api_route(method: str, path: str) -> CompanionRoute | None:
    if method == "GET" and path == CAPABILITY_PATH:
        return CompanionRoute(
            CompanionRouteKind.API,
            upstream_path=path,
            operation=CompanionApiOperation.CAPABILITY,
            response_statuses=frozenset({200}),
        )
    if method == "POST" and path == JOB_COLLECTION_PATH:
        return CompanionRoute(
            CompanionRouteKind.API,
            upstream_path=path,
            request_limit=MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
            response_limit=MAX_AUTHORITY_JOB_STATUS_BYTES,
            operation=CompanionApiOperation.SUBMIT,
            response_statuses=frozenset({202, 400, 409, 413, 415, 503}),
        )
    if method == "POST" and path == JOB_PREPARATION_PATH:
        return CompanionRoute(
            CompanionRouteKind.API,
            upstream_path=path,
            request_limit=MAX_REGIONAL_GROUND_JOB_PREPARATION_REQUEST_BYTES,
            response_limit=MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
            operation=CompanionApiOperation.PREPARE,
            response_statuses=frozenset({200, 400, 413, 415, 422, 503}),
        )
    if method == "GET" and _STATUS_PATH.fullmatch(path):
        return CompanionRoute(
            CompanionRouteKind.API,
            upstream_path=path,
            response_limit=MAX_AUTHORITY_JOB_STATUS_BYTES,
            operation=CompanionApiOperation.STATUS,
            response_statuses=frozenset({200, 404}),
        )
    if method == "POST" and _CANCEL_PATH.fullmatch(path):
        return CompanionRoute(
            CompanionRouteKind.API,
            upstream_path=path,
            response_limit=MAX_AUTHORITY_JOB_STATUS_BYTES,
            operation=CompanionApiOperation.CANCEL,
            response_statuses=frozenset({202, 404, 503}),
        )
    if method == "GET" and _RESULT_PATH.fullmatch(path):
        return CompanionRoute(
            CompanionRouteKind.API,
            upstream_path=path,
            response_limit=MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES,
            operation=CompanionApiOperation.RESULT,
            response_statuses=frozenset({200, 404, 409}),
        )
    return None


def classify_companion_request(
    request: CompanionRequest, *, expected_host: str
) -> CompanionRoute:
    """Classify one exact request or reject before reading its body."""
    if type(request) is not CompanionRequest or not expected_host:
        raise TypeError("request and expected_host must be exact and nonempty")
    headers = _normalized_headers(request.headers)
    _require_public_boundary(headers, expected_host)
    path = _canonical_path(request)
    method = request.method
    if method == "GET" and path in {"/", "/index.html"}:
        return CompanionRoute(CompanionRouteKind.INDEX)
    if method == "GET" and _PORTABLE_ASSET.fullmatch(path.removeprefix("/")):
        return CompanionRoute(
            CompanionRouteKind.ASSET, asset_path=path.removeprefix("/")
        )
    route = _api_route(method, path)
    if route is None:
        raise _reject(405 if method not in {"GET", "POST"} else 404, "not_found")
    if method == "POST":
        _require_state_origin(headers, expected_host)
    return route

"""FastAPI boundary for the same-origin Rate of Closure companion."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from http.client import HTTPException as HttpClientException
from typing import Final

from fastapi import FastAPI, Request, Response
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint

from rate_of_closure.application.regional_ground_authority_transport import (
    AuthorityHttpResponse,
    RegionalGroundAuthorityTransport,
)

from .bundle import CompanionWebBundle
from .contracts import (
    CompanionRequest,
    CompanionRequestRejected,
    CompanionRoute,
    CompanionRouteKind,
    classify_companion_request,
)
from .response_contract import validate_authority_response

_JSON_MEDIA_TYPE: Final = "application/json"
_MAX_DECLARED_BODY_BYTES: Final = 2_147_483_647
_CSP: Final = (
    "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; "
    "worker-src 'self'; object-src 'none'; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'none'"
)
_SECURITY_HEADERS: Final = {
    "Content-Security-Policy": _CSP,
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "microphone=(), payment=(), usb=()"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


def _public_headers(*, cache_control: str) -> dict[str, str]:
    return {**_SECURITY_HEADERS, "Cache-Control": cache_control}


def _error(status_code: int, code: str) -> Response:
    source = json.dumps({"code": code}, separators=(",", ":")).encode("ascii")
    return Response(
        source,
        status_code=status_code,
        media_type=_JSON_MEDIA_TYPE,
        headers=_public_headers(cache_control="no-store"),
    )


def _request_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key_bytes, value_bytes in request.scope["headers"]:
        try:
            key = key_bytes.decode("ascii").lower()
            value = value_bytes.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise CompanionRequestRejected(400, "invalid_headers") from exc
        if key in headers:
            raise CompanionRequestRejected(400, "duplicate_headers")
        headers[key] = value
    return headers


def _request_contract(request: Request) -> CompanionRequest:
    raw_path = request.scope.get("raw_path")
    query = request.scope.get("query_string")
    if type(raw_path) is not bytes or type(query) is not bytes:
        raise CompanionRequestRejected(400, "invalid_path")
    return CompanionRequest(
        method=request.method,
        raw_path=raw_path,
        query_string=query,
        headers=_request_headers(request),
    )


def _declared_length(headers: dict[str, str], maximum: int) -> int | None:
    source = headers.get("content-length")
    if source is None:
        return None
    try:
        length = int(source)
    except ValueError as exc:
        raise CompanionRequestRejected(400, "invalid_content_length") from exc
    if length < 0 or str(length) != source:
        raise CompanionRequestRejected(400, "invalid_content_length")
    if length > maximum:
        raise CompanionRequestRejected(413, "body_too_large")
    return length


def _validate_body_headers(
    headers: dict[str, str], route: CompanionRoute
) -> int | None:
    if "transfer-encoding" in headers:
        raise CompanionRequestRejected(415, "encoded_body_unsupported")
    encoding = headers.get("content-encoding", "identity").lower()
    if encoding != "identity":
        raise CompanionRequestRejected(415, "encoded_body_unsupported")
    if route.request_limit == 0:
        declared = _declared_length(headers, _MAX_DECLARED_BODY_BYTES)
        if declared not in {None, 0}:
            raise CompanionRequestRejected(400, "body_unsupported")
        return declared
    media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != _JSON_MEDIA_TYPE:
        raise CompanionRequestRejected(415, "unsupported_media_type")
    return _declared_length(headers, route.request_limit)


async def _read_body(request: Request, route: CompanionRoute) -> bytes | None:
    headers = _request_headers(request)
    declared = _validate_body_headers(headers, route)
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > route.request_limit:
            raise CompanionRequestRejected(413, "body_too_large")
        body.extend(chunk)
    if declared is not None and declared != len(body):
        raise CompanionRequestRejected(400, "content_length_mismatch")
    if route.request_limit == 0:
        if body:
            raise CompanionRequestRejected(400, "body_unsupported")
        return None
    if not body:
        raise CompanionRequestRejected(400, "body_required")
    return bytes(body)


def _upstream_response(
    route: CompanionRoute, response: AuthorityHttpResponse
) -> Response:
    try:
        body = validate_authority_response(
            route,
            status=response.status,
            media_type=response.media_type,
            body=response.body,
        )
    except (TypeError, ValueError):
        return _error(502, "authority_unavailable")
    return Response(
        body,
        status_code=response.status,
        media_type=_JSON_MEDIA_TYPE,
        headers=_public_headers(cache_control="no-store"),
    )


def _static_response(bundle: CompanionWebBundle, route: CompanionRoute) -> Response:
    if route.kind is CompanionRouteKind.INDEX:
        asset = bundle.index
        cache = "no-store"
    else:
        try:
            asset = bundle.asset(route.asset_path or "")
        except ValueError:
            return _error(404, "not_found")
        cache = "public, max-age=31536000, immutable"
    return Response(
        asset.source,
        media_type=asset.media_type,
        headers=_public_headers(cache_control=cache),
    )


def _lifespan(
    transport: RegionalGroundAuthorityTransport,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def close_transport(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            transport.close()

    return close_transport


def create_companion_app(
    *,
    bundle: CompanionWebBundle,
    transport: RegionalGroundAuthorityTransport,
    expected_host: str,
) -> FastAPI:
    """Create a closed no-CORS gateway over immutable assets and one authority."""
    if type(bundle) is not CompanionWebBundle or not expected_host:
        raise TypeError("bundle and expected_host must be exact and nonempty")

    app = FastAPI(
        title="Rate of Closure local production companion",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan(transport),
    )

    @app.middleware("http")
    async def secure_every_response(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method not in {
            "GET",
            "POST",
            "OPTIONS",
            "PUT",
            "PATCH",
            "DELETE",
            "HEAD",
        }:
            return _error(405, "not_found")
        response = await call_next(request)
        for name, value in _public_headers(cache_control="no-store").items():
            if name not in response.headers:
                response.headers[name] = value
        return response

    @app.api_route(
        "/{public_path:path}",
        methods=["GET", "POST", "OPTIONS", "PUT", "PATCH", "DELETE", "HEAD"],
    )
    async def public_gateway(request: Request, public_path: str) -> Response:
        del public_path
        try:
            route = classify_companion_request(
                _request_contract(request), expected_host=expected_host
            )
            if route.kind is not CompanionRouteKind.API:
                return _static_response(bundle, route)
            body = await _read_body(request, route)
            response = await run_in_threadpool(
                transport.request,
                request.method,
                route.upstream_path or "",
                body,
                route.response_limit,
            )
            return _upstream_response(route, response)
        except CompanionRequestRejected as error:
            return _error(error.status_code, error.code)
        except (
            HttpClientException,
            OSError,
            TimeoutError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return _error(502, "authority_unavailable")

    return app

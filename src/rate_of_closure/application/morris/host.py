"""Authenticated loopback-only FastAPI host for the Morris authority router."""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from rate_of_closure.application.durable_ensemble.contracts import (
    DURABLE_ENSEMBLE_JOB_SCHEMA_ID,
    DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID,
    DURABLE_ENSEMBLE_SCOPE,
)
from rate_of_closure.application.durable_ensemble.registry import (
    DurableEnsembleJobRegistry,
)
from rate_of_closure.application.durable_ensemble.router import (
    create_durable_ensemble_router,
)

from .contracts import MORRIS_JOB_SCHEMA_ID, MORRIS_REQUEST_SCHEMA_ID
from .router import MorrisJobRegistry, create_morris_router

API_PREFIX = "/api/rate-of-closure/v1"
CAPABILITY_PATH = f"{API_PREFIX}/morris/capabilities"
DURABLE_ENSEMBLE_CAPABILITY_PATH = f"{API_PREFIX}/durable-ensembles/capabilities"
_CAPABILITY = {
    "schema_id": "rate-of-closure/morris-authority-capability",
    "schema_version": 1,
    "available": True,
    "api_prefix": API_PREFIX,
    "request_schema_id": MORRIS_REQUEST_SCHEMA_ID,
    "job_schema_id": MORRIS_JOB_SCHEMA_ID,
}
_DURABLE_CAPABILITY = {
    "schema_id": "rate-of-closure/durable-ensemble-authority-capability",
    "schema_version": 1,
    "available": True,
    "api_prefix": API_PREFIX,
    "scope": DURABLE_ENSEMBLE_SCOPE,
    "request_schema_id": DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID,
    "job_schema_id": DURABLE_ENSEMBLE_JOB_SCHEMA_ID,
}


def _token(value: object) -> str:
    if not isinstance(value, str) or len(value) < 8 or value != value.strip():
        raise ValueError("authority token must be a nonempty private token")
    if any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise ValueError("authority token must contain visible ASCII only")
    return value


def create_morris_authority_app(
    token: str,
    registry: MorrisJobRegistry,
    shutdown: Callable[[], None] | None = None,
    *,
    lifespan_started: Callable[[], None] | None = None,
    durable_ensemble_registry: DurableEnsembleJobRegistry | None = None,
) -> FastAPI:
    """Build a no-CORS, bearer-authenticated mountable authority app."""
    secret = _token(token)
    if not isinstance(registry, MorrisJobRegistry):
        raise TypeError("registry must be a MorrisJobRegistry")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if lifespan_started is not None:
            lifespan_started()
        try:
            yield
        finally:
            try:
                registry.close()
            finally:
                if durable_ensemble_registry is not None:
                    durable_ensemble_registry.close()

    app = FastAPI(
        title="Rate Morris Authority",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    def secured(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(Exception)
    async def internal_error(_request: Request, _error: Exception) -> Response:
        return secured(
            JSONResponse({"error": "internal server error"}, status_code=500)
        )

    @app.middleware("http")
    async def secure(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        supplied = request.headers.get("authorization", "")
        expected = f"Bearer {secret}"
        response: Response
        if not hmac.compare_digest(supplied, expected):
            response = JSONResponse(
                {"error": "authentication required"}, status_code=401
            )
            response.headers["WWW-Authenticate"] = "Bearer"
        else:
            response = await call_next(request)
        return secured(response)

    @app.get(CAPABILITY_PATH)
    async def capability() -> dict[str, object]:
        return dict(_CAPABILITY)

    @app.get(DURABLE_ENSEMBLE_CAPABILITY_PATH)
    async def durable_capability() -> dict[str, object]:
        return {
            **_DURABLE_CAPABILITY,
            "available": durable_ensemble_registry is not None,
        }

    @app.post("/_control/shutdown")
    async def stop_child() -> dict[str, str]:
        if shutdown is None:
            return {"status": "unavailable"}
        shutdown()
        return {"status": "stopping"}

    app.include_router(create_morris_router(registry), prefix=API_PREFIX)
    if durable_ensemble_registry is not None:
        app.include_router(
            create_durable_ensemble_router(durable_ensemble_registry),
            prefix=API_PREFIX,
        )
    return app


__all__ = [
    "API_PREFIX",
    "CAPABILITY_PATH",
    "DURABLE_ENSEMBLE_CAPABILITY_PATH",
    "create_morris_authority_app",
]

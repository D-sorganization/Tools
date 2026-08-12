"""Authenticated FastAPI application for the loopback model authority."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rate_of_closure.application.regional_ground_execution_job import (
    MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
    regional_ground_execution_job_from_json,
)
from rate_of_closure.application.regional_ground_execution_result import (
    regional_ground_execution_result_to_json,
)

from .capability import DEFAULT_UNAVAILABLE_CAPABILITY, AuthorityCapability
from .jobs import (
    AuthorityExecutionUnavailable,
    AuthorityJobConflict,
    AuthorityJobManager,
    AuthorityJobResultUnavailable,
)

CAPABILITY_PATH = "/api/rate-of-closure/v1/capabilities"
JOB_COLLECTION_PATH = "/api/rate-of-closure/v1/regional-ground/jobs"
_BEARER = HTTPBearer(auto_error=False)


class _RequestBodyTooLarge(ValueError):
    """Internal signal for a request that exceeds its exact wire bound."""


class _UnsupportedMediaType(TypeError):
    """Internal signal for content the authority does not decode."""


def _error(code: str, detail: str, status_code: int) -> JSONResponse:
    """Build one non-cacheable bounded API error."""
    return JSONResponse(
        {"code": code, "detail": detail},
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )


def _validate_content_headers(request: Request) -> int | None:
    """Reject encoded, mistyped, malformed, or declared-oversize bodies."""
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise _UnsupportedMediaType("Content-Type must be application/json")
    if request.headers.get("content-encoding", "identity").lower() != "identity":
        raise _UnsupportedMediaType("encoded request bodies are unsupported")
    declared = request.headers.get("content-length")
    if declared is None:
        return None
    try:
        length = int(declared)
    except ValueError as exc:
        raise ValueError("Content-Length must be an integer") from exc
    if length < 0 or str(length) != declared:
        raise ValueError("Content-Length must be canonical and nonnegative")
    if length > MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES:
        raise _RequestBodyTooLarge("request body exceeds maximum wire size")
    return length


async def _read_job_text(request: Request) -> str:
    """Read one UTF-8 job document without buffering beyond its wire bound."""
    declared = _validate_content_headers(request)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES:
            raise _RequestBodyTooLarge("request body exceeds maximum wire size")
    if declared is not None and declared != len(body):
        raise ValueError("Content-Length does not match the request body")
    if not body:
        raise ValueError("request body must be nonempty")
    try:
        return bytes(body).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("request body must be valid UTF-8") from exc


def _require_token(token: str) -> str:
    """Validate the injected ephemeral authority token."""
    if not token or token != token.strip():
        raise ValueError("authority token must be nonempty and trimmed")
    return token


def create_authority_app(
    *,
    token: str,
    capability: AuthorityCapability = DEFAULT_UNAVAILABLE_CAPABILITY,
    job_manager: AuthorityJobManager | None = None,
) -> FastAPI:
    """Create an authenticated, non-cacheable loopback authority application."""
    expected_token = _require_token(token)
    manager = AuthorityJobManager() if job_manager is None else job_manager
    if type(manager) is not AuthorityJobManager:
        raise TypeError("job_manager must be an exact AuthorityJobManager")
    app = FastAPI(
        title="Rate of Closure local model authority",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def authorize(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_BEARER),
        ],
    ) -> None:
        """Require the exact ephemeral bearer token without leaking comparisons."""
        supplied = credentials.credentials if credentials is not None else ""
        scheme = credentials.scheme if credentials is not None else ""
        valid = scheme.lower() == "bearer" and secrets.compare_digest(
            supplied, expected_token
        )
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Local authority authentication required.",
                headers={
                    "WWW-Authenticate": "Bearer",
                    "Cache-Control": "no-store",
                },
            )

    @app.get(CAPABILITY_PATH, dependencies=[Depends(authorize)])
    def read_capability(response: Response) -> dict[str, object]:
        """Return the immutable capability state with caching disabled."""
        response.headers["Cache-Control"] = "no-store"
        return dict(capability.to_wire())

    @app.post(JOB_COLLECTION_PATH, dependencies=[Depends(authorize)])
    async def submit_job(request: Request) -> Response:
        """Validate and enqueue one bounded canonical execution job."""
        try:
            text = await _read_job_text(request)
            job = regional_ground_execution_job_from_json(text)
            snapshot = manager.submit(job)
        except _RequestBodyTooLarge as error:
            return _error(
                "body_too_large", str(error), status.HTTP_413_CONTENT_TOO_LARGE
            )
        except _UnsupportedMediaType as error:
            return _error(
                "unsupported_media_type",
                str(error),
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            )
        except (TypeError, ValueError):
            return _error(
                "invalid_job",
                "Regional-ground execution job is invalid.",
                status.HTTP_400_BAD_REQUEST,
            )
        except AuthorityExecutionUnavailable:
            return _error(
                "execution_unavailable",
                "Qualified regional-ground execution is unavailable.",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except AuthorityJobConflict:
            return _error(
                "job_conflict",
                "Another job is active or this identity is retained.",
                status.HTTP_409_CONFLICT,
            )
        return JSONResponse(
            snapshot.to_wire(),
            status_code=status.HTTP_202_ACCEPTED,
            headers={"Cache-Control": "no-store"},
        )

    @app.get(f"{JOB_COLLECTION_PATH}/{{job_id}}", dependencies=[Depends(authorize)])
    def read_job_status(job_id: str) -> Response:
        """Return one retained job status without caching."""
        try:
            snapshot = manager.status(job_id)
        except KeyError:
            return _error(
                "job_not_found", "Job was not found.", status.HTTP_404_NOT_FOUND
            )
        return JSONResponse(snapshot.to_wire(), headers={"Cache-Control": "no-store"})

    @app.post(
        f"{JOB_COLLECTION_PATH}/{{job_id}}/cancel",
        dependencies=[Depends(authorize)],
    )
    def cancel_job(job_id: str) -> Response:
        """Idempotently request cooperative cancellation for one job."""
        try:
            snapshot = manager.cancel(job_id)
        except KeyError:
            return _error(
                "job_not_found", "Job was not found.", status.HTTP_404_NOT_FOUND
            )
        return JSONResponse(
            snapshot.to_wire(),
            status_code=status.HTTP_202_ACCEPTED,
            headers={"Cache-Control": "no-store"},
        )

    @app.get(
        f"{JOB_COLLECTION_PATH}/{{job_id}}/result",
        dependencies=[Depends(authorize)],
    )
    def read_job_result(job_id: str) -> Response:
        """Return only a complete validated canonical result document."""
        try:
            result = manager.result(job_id)
        except KeyError:
            return _error(
                "job_not_found", "Job was not found.", status.HTTP_404_NOT_FOUND
            )
        except AuthorityJobResultUnavailable:
            return _error(
                "result_unavailable",
                "Complete result is unavailable.",
                status.HTTP_409_CONFLICT,
            )
        return Response(
            regional_ground_execution_result_to_json(result),
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )

    return app

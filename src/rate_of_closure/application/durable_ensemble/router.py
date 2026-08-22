"""Mountable HTTP transport for durable ensemble lifecycle controls."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rate_of_closure.application._strict_http_json import (
    StrictHttpFailure,
    strict_json_document,
)

from .contracts import DurableEnsembleJobEnvelope, parse_durable_ensemble_request
from .registry import DurableEnsembleJobRegistry

_TERMINAL = frozenset({"completed", "cancelled", "failed"})


def _response(envelope: DurableEnsembleJobEnvelope, status: int = 200) -> JSONResponse:
    return JSONResponse(envelope.to_json_dict(), status_code=status)


def create_durable_ensemble_router(
    registry: DurableEnsembleJobRegistry,
) -> APIRouter:
    """Create progress, cancellation, resume, and inspection routes."""
    if not isinstance(registry, DurableEnsembleJobRegistry):
        raise TypeError("registry must be a DurableEnsembleJobRegistry")
    router = APIRouter()

    @router.post("/durable-ensembles/jobs")
    async def create_job(request: Request) -> JSONResponse:
        try:
            document = await strict_json_document(request, registry.max_body_bytes)
            return _response(
                registry.create(parse_durable_ensemble_request(document)), 202
            )
        except StrictHttpFailure as exc:
            return JSONResponse({"error": exc.message}, status_code=exc.status)
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except FileExistsError:
            return JSONResponse({"error": "archive already has an active writer"}, 409)
        except OverflowError:
            return JSONResponse({"error": "ensemble authority is at capacity"}, 429)

    @router.get("/durable-ensembles/jobs/{job_id}")
    async def get_job(job_id: str) -> JSONResponse:
        try:
            return _response(registry.status(job_id))
        except KeyError:
            return JSONResponse({"error": "unknown durable ensemble job"}, 404)

    @router.delete("/durable-ensembles/jobs/{job_id}")
    async def cancel_job(job_id: str) -> JSONResponse:
        try:
            envelope = registry.cancel(job_id)
            return _response(envelope, 200 if envelope.status in _TERMINAL else 202)
        except KeyError:
            return JSONResponse({"error": "unknown durable ensemble job"}, 404)

    return router


__all__ = ["create_durable_ensemble_router"]

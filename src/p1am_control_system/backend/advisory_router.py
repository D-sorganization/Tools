"""REST review surface for synthetic non-authoritative advisories."""

from __future__ import annotations

from collections.abc import Callable

from advisory_workspace import (
    AdvisoryDisposition,
    AdvisoryResult,
    AdvisoryService,
    DispositionRecord,
    representative_advisory_request,
)
from fastapi import APIRouter, Depends, HTTPException
from identity import Principal


def create_advisory_router(
    service: AdvisoryService,
    operator_dependency: Callable[..., Principal],
) -> APIRouter:
    """Create review-only routes; no authoritative command route is defined."""
    if not isinstance(service, AdvisoryService):
        raise TypeError("service must be an AdvisoryService")
    if not callable(operator_dependency):
        raise TypeError("operator_dependency must be callable")
    router = APIRouter(prefix="/api/operator/advisories", tags=["advisories"])

    @router.get("/representative")
    async def representative_advisory() -> AdvisoryResult:
        return service.evaluate(representative_advisory_request())

    @router.post("/{advisory_id}/dispositions")
    async def record_disposition(
        advisory_id: str,
        body: AdvisoryDisposition,
        principal: Principal = Depends(operator_dependency),  # noqa: B008
    ) -> DispositionRecord:
        try:
            return service.record_disposition(advisory_id, body, principal)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return router

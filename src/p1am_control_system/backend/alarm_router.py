"""Supervisory professional alarm-management REST API."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import cast

from alarm_lifecycle import AlarmPerformanceReport, AlarmSnapshot
from alarm_service import AlarmService
from fastapi import APIRouter, Depends, HTTPException
from identity import Principal
from pydantic import BaseModel, Field


class ShelfRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)
    duration_seconds: int = Field(ge=1, le=86_400)


class SuppressionRequest(BaseModel):
    rule: str = Field(min_length=1, max_length=200)
    active: bool


def _domain_call(operation: Callable[[], AlarmSnapshot]) -> AlarmSnapshot:
    try:
        return operation()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def create_alarm_router(
    service: AlarmService,
    operator_dependency: Callable[..., Principal],
    engineer_dependency: Callable[..., Principal],
) -> APIRouter:
    """Build a role-aware router over one alarm application service."""
    if not isinstance(service, AlarmService):
        raise TypeError("service must be an AlarmService")
    if not callable(operator_dependency) or not callable(engineer_dependency):
        raise TypeError("alarm authorization dependencies must be callable")
    router = APIRouter(prefix="/api/alarm-management", tags=["alarm-management"])

    @router.get("/active")
    async def active() -> list[AlarmSnapshot]:
        return cast(list[AlarmSnapshot], service.active())

    @router.post("/{tag}/acknowledge")
    async def acknowledge(
        tag: str,
        principal: Principal = Depends(operator_dependency),  # noqa: B008
    ) -> AlarmSnapshot:
        return _domain_call(lambda: service.acknowledge(tag, principal))

    @router.post("/{tag}/shelf")
    async def shelve(
        tag: str,
        request: ShelfRequest,
        principal: Principal = Depends(operator_dependency),  # noqa: B008
    ) -> AlarmSnapshot:
        return _domain_call(
            lambda: service.shelve(
                tag,
                principal,
                request.reason,
                timedelta(seconds=request.duration_seconds),
            )
        )

    @router.delete("/{tag}/shelf")
    async def unshelve(
        tag: str,
        principal: Principal = Depends(operator_dependency),  # noqa: B008
    ) -> AlarmSnapshot:
        return _domain_call(lambda: service.unshelve(tag, principal))

    @router.post("/{tag}/suppression")
    async def suppress(
        tag: str,
        request: SuppressionRequest,
        _principal: Principal = Depends(engineer_dependency),  # noqa: B008
    ) -> AlarmSnapshot:
        return _domain_call(lambda: service.suppress(tag, request.rule, request.active))

    @router.get("/performance")
    async def performance(
        _principal: Principal = Depends(engineer_dependency),  # noqa: B008
    ) -> AlarmPerformanceReport:
        return service.performance()

    return router

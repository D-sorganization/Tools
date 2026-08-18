"""REST adapter for the non-confidential synthetic operator workspace."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from identity import Principal
from process_overview import ProcessOverview, synthetic_process_overview
from protection_management import (
    BypassRequest,
    ManagedBypass,
    ProtectionDefinition,
    ProtectionService,
    TripRecord,
)
from pydantic import BaseModel, ConfigDict, Field


class TripRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    group_id: str = Field(min_length=1, max_length=100)


class BypassBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str = Field(min_length=8, max_length=500)
    expires_at: datetime


class ProtectionSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    definitions: list[ProtectionDefinition]
    trips: list[TripRecord]
    active_bypasses: list[ManagedBypass]


def _translate_domain(
    operation: Callable[[], TripRecord | ManagedBypass],
) -> TripRecord | ManagedBypass:
    try:
        return operation()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def create_operator_router(
    protections: ProtectionService,
    engineer_dependency: Callable[..., Principal],
    read_dependency: Callable[..., object],
) -> APIRouter:
    """Build the bounded representative operator API."""
    if not isinstance(protections, ProtectionService):
        raise TypeError("protections must be a ProtectionService")
    if not callable(engineer_dependency):
        raise TypeError("engineer_dependency must be callable")
    if not callable(read_dependency):
        raise TypeError("read_dependency must be callable")
    router = APIRouter(prefix="/api/operator", tags=["operator"])

    @router.get("/overview", dependencies=[Depends(read_dependency)])
    async def overview() -> ProcessOverview:
        return synthetic_process_overview()

    @router.get("/protections", dependencies=[Depends(read_dependency)])
    async def protection_snapshot() -> ProtectionSnapshot:
        return ProtectionSnapshot(
            definitions=protections.definitions(),
            trips=protections.trips(),
            active_bypasses=protections.active_bypasses(),
        )

    @router.post("/protections/{protection_id}/trips")
    async def trip(
        protection_id: str,
        request: TripRequest,
        _principal: Principal = Depends(engineer_dependency),  # noqa: B008
    ) -> TripRecord:
        return _translate_domain(
            lambda: protections.trip(protection_id, group_id=request.group_id)
        )

    @router.post("/protections/{protection_id}/bypasses")
    async def bypass(
        protection_id: str,
        request: BypassBody,
        principal: Principal = Depends(engineer_dependency),  # noqa: B008
    ) -> ManagedBypass:
        return _translate_domain(
            lambda: protections.request_bypass(
                BypassRequest(
                    protection_id=protection_id,
                    reason=request.reason,
                    expires_at=request.expires_at,
                ),
                principal,
            )
        )

    return router

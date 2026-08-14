"""REST adapter for investigations, asset advisories, and shift handover."""

from __future__ import annotations

import io
from collections.abc import Callable

from asset_health import AssetHealthReport
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from identity import Principal
from pydantic import BaseModel, ConfigDict, Field
from saved_investigation import (
    InvestigationService,
    InvestigationSpec,
    SavedInvestigation,
)
from shift_log import (
    HandoverAcknowledgment,
    ShiftEntry,
    ShiftEntryDraft,
    ShiftLogService,
    ShiftSignoff,
)


class HandoverBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    note: str = Field(min_length=1, max_length=1000)


def _domain_call(operation: Callable[[], object]) -> object:
    try:
        return operation()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def create_operations_router(
    investigations: InvestigationService,
    shifts: ShiftLogService,
    asset_report_provider: Callable[[], AssetHealthReport],
    operator_dependency: Callable[..., Principal],
) -> APIRouter:
    if not isinstance(investigations, InvestigationService):
        raise TypeError("investigations must be an InvestigationService")
    if not isinstance(shifts, ShiftLogService):
        raise TypeError("shifts must be a ShiftLogService")
    if not callable(asset_report_provider) or not callable(operator_dependency):
        raise TypeError("operations providers and dependencies must be callable")
    router = APIRouter(prefix="/api/operator", tags=["operator-operations"])

    @router.post("/investigations")
    async def save_investigation(
        spec: InvestigationSpec,
        principal: Principal = Depends(operator_dependency),  # noqa: B008
    ) -> SavedInvestigation:
        result = _domain_call(lambda: investigations.save(spec, principal))
        assert isinstance(result, SavedInvestigation)
        return result

    @router.get("/investigations/{investigation_id}")
    async def get_investigation(investigation_id: str) -> SavedInvestigation:
        result = _domain_call(lambda: investigations.get(investigation_id))
        assert isinstance(result, SavedInvestigation)
        return result

    @router.get("/investigations/{investigation_id}/export")
    async def export_investigation(investigation_id: str) -> StreamingResponse:
        try:
            artifact = investigations.export(investigation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return StreamingResponse(
            io.BytesIO(artifact.payload),
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{investigation_id}-investigation.zip"'
                ),
                "X-Artifact-SHA256": artifact.sha256,
                "X-Investigation-ID": investigation_id,
            },
        )

    @router.get("/assets/health/representative")
    async def representative_asset_health() -> AssetHealthReport:
        return asset_report_provider()

    @router.post("/shift-log")
    async def append_shift_entry(
        draft: ShiftEntryDraft,
        principal: Principal = Depends(operator_dependency),  # noqa: B008
    ) -> ShiftEntry:
        result = _domain_call(lambda: shifts.append(draft, principal))
        assert isinstance(result, ShiftEntry)
        return result

    @router.get("/shift-log")
    async def search_shift_entries(
        query: str = Query(default="", max_length=200),
    ) -> list[ShiftEntry]:
        entries: list[ShiftEntry] = shifts.search(query)
        return entries

    @router.post("/shift-log/{entry_id}/signoff")
    async def sign_off_shift_entry(
        entry_id: str,
        principal: Principal = Depends(operator_dependency),  # noqa: B008
    ) -> ShiftSignoff:
        result = _domain_call(lambda: shifts.sign_off(entry_id, principal))
        assert isinstance(result, ShiftSignoff)
        return result

    @router.post("/shift-log/{entry_id}/handover")
    async def acknowledge_handover(
        entry_id: str,
        body: HandoverBody,
        principal: Principal = Depends(operator_dependency),  # noqa: B008
    ) -> HandoverAcknowledgment:
        result = _domain_call(
            lambda: shifts.acknowledge_handover(entry_id, principal, body.note)
        )
        assert isinstance(result, HandoverAcknowledgment)
        return result

    return router

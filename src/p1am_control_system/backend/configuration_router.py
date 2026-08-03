"""Role-aware REST adapter for protected configuration revisions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from configuration_workflow import (
    ConfigurationDiff,
    ConfigurationRevision,
    ConfigurationWorkflow,
)
from fastapi import APIRouter, Depends, HTTPException, Query
from identity import Principal
from models import RoutingConfig
from pydantic import BaseModel, Field


class DraftRequest(BaseModel):
    payload: RoutingConfig
    reason: str = Field(min_length=1, max_length=500)


class ReasonRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)


def _domain_call(
    operation: Callable[[], ConfigurationRevision],
) -> ConfigurationRevision:
    try:
        return operation()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


async def _async_domain_call(
    operation: Callable[[], Awaitable[ConfigurationRevision]],
) -> ConfigurationRevision:
    try:
        return await operation()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def create_configuration_router(
    workflow: ConfigurationWorkflow,
    engineer_dependency: Callable[..., Principal],
    admin_dependency: Callable[..., Principal],
) -> APIRouter:
    """Build the only public mutation path for protected configuration."""
    if not isinstance(workflow, ConfigurationWorkflow):
        raise TypeError("workflow must be a ConfigurationWorkflow")
    if not callable(engineer_dependency) or not callable(admin_dependency):
        raise TypeError("configuration authorization dependencies must be callable")
    router = APIRouter(prefix="/api/configurations", tags=["configuration"])

    @router.get("")
    async def revisions() -> list[ConfigurationRevision]:
        return workflow.list()

    @router.get("/active")
    async def active() -> ConfigurationRevision | None:
        return workflow.active()

    @router.post("/drafts")
    async def create_draft(
        request: DraftRequest,
        principal: Principal = Depends(engineer_dependency),  # noqa: B008
    ) -> ConfigurationRevision:
        return _domain_call(
            lambda: workflow.create_draft(request.payload, principal, request.reason)
        )

    @router.get("/{revision_id}")
    async def get_revision(revision_id: str) -> ConfigurationRevision:
        return _domain_call(lambda: workflow.get(revision_id))

    @router.get("/{revision_id}/diff")
    async def diff(
        revision_id: str,
        base_revision_id: str | None = Query(default=None),
    ) -> list[ConfigurationDiff]:
        try:
            return workflow.diff(revision_id, base_revision_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/{revision_id}/validate")
    async def validate(
        revision_id: str,
        principal: Principal = Depends(engineer_dependency),  # noqa: B008
    ) -> ConfigurationRevision:
        return _domain_call(lambda: workflow.validate(revision_id, principal))

    @router.post("/{revision_id}/review")
    async def review(
        revision_id: str,
        principal: Principal = Depends(engineer_dependency),  # noqa: B008
    ) -> ConfigurationRevision:
        return _domain_call(lambda: workflow.submit_for_review(revision_id, principal))

    @router.post("/{revision_id}/approve")
    async def approve(
        revision_id: str,
        request: ReasonRequest,
        principal: Principal = Depends(engineer_dependency),  # noqa: B008
    ) -> ConfigurationRevision:
        return _domain_call(
            lambda: workflow.approve(revision_id, principal, request.reason)
        )

    @router.post("/{revision_id}/activate")
    async def activate(
        revision_id: str,
        principal: Principal = Depends(admin_dependency),  # noqa: B008
    ) -> ConfigurationRevision:
        return await _async_domain_call(
            lambda: workflow.activate(revision_id, principal)
        )

    @router.post("/{revision_id}/rollback")
    async def rollback(
        revision_id: str,
        request: ReasonRequest,
        principal: Principal = Depends(admin_dependency),  # noqa: B008
    ) -> ConfigurationRevision:
        return await _async_domain_call(
            lambda: workflow.rollback(revision_id, principal, request.reason)
        )

    return router

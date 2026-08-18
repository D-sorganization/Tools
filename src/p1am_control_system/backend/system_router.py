"""REST surface for recovery packages, identity, and system health."""

from __future__ import annotations

from collections.abc import Callable

from configuration_workflow import ConfigurationRevision
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from identity import Principal
from recovery_package import RecoveryPackageService
from system_health import DeploymentIdentity, SystemHealthReport, SystemHealthService


def create_system_router(
    recovery: RecoveryPackageService,
    health: SystemHealthService,
    engineer_dependency: Callable[..., Principal],
    admin_dependency: Callable[..., Principal],
    read_dependency: Callable[..., object],
) -> APIRouter:
    """Build recovery endpoints over narrow application services."""
    if not isinstance(recovery, RecoveryPackageService):
        raise TypeError("recovery must be a RecoveryPackageService")
    if not isinstance(health, SystemHealthService):
        raise TypeError("health must be a SystemHealthService")
    if (
        not callable(engineer_dependency)
        or not callable(admin_dependency)
        or not callable(read_dependency)
    ):
        raise TypeError("system authorization dependencies must be callable")
    router = APIRouter(prefix="/api/system", tags=["system-health"])

    @router.get("/identity", dependencies=[Depends(read_dependency)])
    async def identity() -> DeploymentIdentity:
        return health.identity()

    @router.get("/health", dependencies=[Depends(read_dependency)])
    async def report() -> SystemHealthReport:
        return health.report()

    @router.post("/backups")
    async def backup(
        _principal: Principal = Depends(admin_dependency),  # noqa: B008
    ) -> Response:
        try:
            artifact = recovery.create()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(
            content=artifact.payload,
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    "attachment; filename=p1am-configuration-recovery.zip"
                ),
                "X-Artifact-SHA256": artifact.sha256,
                "X-Configuration-Revision": artifact.manifest.configuration_revision,
                "X-Energized-State-Included": "false",
            },
        )

    @router.post("/restores")
    async def restore(
        request: Request,
        principal: Principal = Depends(engineer_dependency),  # noqa: B008
        artifact_sha256: str | None = Header(default=None, alias="X-Artifact-SHA256"),
        change_reason: str = Header(alias="X-Change-Reason"),
    ) -> ConfigurationRevision:
        payload = await request.body()
        try:
            return recovery.restore_as_draft(
                payload,
                principal,
                change_reason,
                artifact_sha256,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router

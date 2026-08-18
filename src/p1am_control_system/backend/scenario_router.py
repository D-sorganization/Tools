"""REST adapter for isolated synthetic acceptance scenarios."""

from __future__ import annotations

from collections.abc import Callable

from evidence_package import EvidencePackageService
from fastapi import APIRouter, Depends, HTTPException, Response
from identity import Principal
from scenario_evidence import (
    RepresentativeScenarioAdapter,
    ScenarioDefinition,
    ScenarioRunner,
    ScenarioStep,
)

IdentityProvider = Callable[[], tuple[str, str]]


def representative_scenario() -> ScenarioDefinition:
    """Return a generic fixture with no plant names, addresses, or control logic."""
    return ScenarioDefinition(
        name="Representative transport and quality recovery",
        data_classification="synthetic",
        not_for_live_control=True,
        steps=[
            ScenarioStep(
                step_id="disconnect-transport",
                action="transport_disconnect",
                target="SYNTHETIC.TRANSPORT",
                parameters={},
                expected={"connected": False},
                timing_window_ms=100,
            ),
            ScenarioStep(
                step_id="mark-stale",
                action="set_quality",
                target="SYNTHETIC.SIGNAL_0",
                parameters={"quality": "stale"},
                expected={"quality": "stale"},
                timing_window_ms=100,
            ),
            ScenarioStep(
                step_id="recover-transport",
                action="transport_recover",
                target="SYNTHETIC.TRANSPORT",
                parameters={},
                expected={"connected": True},
                timing_window_ms=100,
            ),
            ScenarioStep(
                step_id="restore-quality",
                action="set_quality",
                target="SYNTHETIC.SIGNAL_0",
                parameters={"quality": "good"},
                expected={"quality": "good"},
                timing_window_ms=100,
            ),
        ],
    )


def create_scenario_router(
    identity_provider: IdentityProvider,
    admin_dependency: Callable[..., Principal],
    read_dependency: Callable[..., object],
) -> APIRouter:
    """Build a runner that can only instantiate the isolated representative adapter."""
    if (
        not callable(identity_provider)
        or not callable(admin_dependency)
        or not callable(read_dependency)
    ):
        raise TypeError("scenario providers must be callable")
    router = APIRouter(prefix="/api/acceptance/scenarios", tags=["acceptance"])

    @router.get("/representative", dependencies=[Depends(read_dependency)])
    async def representative() -> ScenarioDefinition:
        return representative_scenario()

    @router.post("/run")
    async def run(
        scenario: ScenarioDefinition,
        _principal: Principal = Depends(admin_dependency),  # noqa: B008
    ) -> Response:
        try:
            software_revision, configuration_revision = identity_provider()
            runner = ScenarioRunner(
                RepresentativeScenarioAdapter(),
                software_revision=software_revision,
                configuration_revision=configuration_revision,
            )
            evidence = await runner.run(scenario)
            artifact = EvidencePackageService().create(scenario, evidence)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(
            content=artifact.payload,
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    "attachment; filename=p1am-acceptance-evidence.zip"
                ),
                "X-Artifact-SHA256": artifact.sha256,
                "X-Evidence-ID": evidence.evidence_id,
                "X-Evidence-Passed": str(evidence.passed).lower(),
                "X-Data-Classification": "synthetic",
                "X-Not-For-Live-Control": "true",
            },
        )

    return router

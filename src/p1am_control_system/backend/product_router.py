"""REST adapter for the reusable synthetic control-product contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from availability import AvailabilityHealth, AvailabilityService
from connector_plugins import (
    ConnectorDiagnostic,
    ConnectorManager,
    ConnectorSample,
)
from fastapi import APIRouter, Depends, HTTPException
from identity import Principal
from notification_policy import (
    NotificationAudit,
    NotificationPolicy,
    NotificationService,
)
from pydantic import BaseModel, ConfigDict, Field
from synthetic_procedure import (
    ProcedureCommand,
    ProcedureEvent,
    ProcedureState,
    SyntheticProcedure,
)


class ProcedureCommandBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str = Field(min_length=1, max_length=500)


class ProductStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    procedure_state: ProcedureState
    procedure_events: list[ProcedureEvent]
    connectors: list[ConnectorDiagnostic]
    samples: dict[str, ConnectorSample]
    notification_policy: NotificationPolicy
    notification_audit: list[NotificationAudit]
    availability: AvailabilityHealth
    data_classification: Literal["synthetic"] = "synthetic"
    not_for_live_control: Literal[True] = True


def create_product_router(
    procedure: SyntheticProcedure,
    connectors: ConnectorManager,
    notifications: NotificationService,
    availability: AvailabilityService,
    operator_dependency: Callable[..., Principal],
) -> APIRouter:
    if not all(
        (
            isinstance(procedure, SyntheticProcedure),
            isinstance(connectors, ConnectorManager),
            isinstance(notifications, NotificationService),
            isinstance(availability, AvailabilityService),
            callable(operator_dependency),
        )
    ):
        raise TypeError("product router dependencies do not satisfy their contracts")
    router = APIRouter(prefix="/api/operator", tags=["control-product"])

    @router.get("/product-status")
    async def product_status() -> ProductStatus:
        return ProductStatus(
            procedure_state=procedure.state,
            procedure_events=procedure.events(),
            connectors=connectors.diagnostics(),
            samples=connectors.poll(),
            notification_policy=notifications.policy,
            notification_audit=notifications.audit(),
            availability=availability.health(),
        )

    @router.post("/procedure/commands/{command}")
    async def procedure_command(
        command: ProcedureCommand,
        body: ProcedureCommandBody,
        principal: Principal = Depends(operator_dependency),  # noqa: B008
    ) -> ProcedureEvent:
        try:
            return procedure.dispatch(command, principal, body.reason)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return router

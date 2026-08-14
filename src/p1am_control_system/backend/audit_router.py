"""Role-protected, paginated read API for the append-only audit trail."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any

from audit_log import AuditLog, AuditOutcome
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel import Session, col, select


class AuditItem(BaseModel):
    """Structured public representation of one immutable audit row."""

    id: int
    actor_subject: str
    actor_display_name: str
    actor_role: str
    action: str
    target: str
    reason: str
    outcome: AuditOutcome
    before: Any
    after: Any
    source: str
    configuration_revision: str
    correlation_id: str
    error_code: str | None
    timestamp: datetime


class AuditPage(BaseModel):
    """Bounded audit result page with continuation metadata."""

    items: list[AuditItem]
    limit: int
    offset: int
    has_more: bool


def _item(row: AuditLog) -> AuditItem:
    if row.id is None:
        raise ValueError("persisted audit row must have an id")
    return AuditItem(
        id=row.id,
        actor_subject=row.actor_subject,
        actor_display_name=row.actor_display_name,
        actor_role=row.actor_role,
        action=row.action,
        target=row.target,
        reason=row.reason,
        outcome=AuditOutcome(row.outcome),
        before=json.loads(row.before_json),
        after=json.loads(row.after_json),
        source=row.source,
        configuration_revision=row.configuration_revision,
        correlation_id=row.correlation_id,
        error_code=row.error_code,
        timestamp=row.timestamp,
    )


def create_audit_router(
    get_session_dep: Callable[..., Session],
    audit_auth_dep: Callable[..., object],
) -> APIRouter:
    """Create the audit query router from injected persistence/auth boundaries."""
    if not callable(get_session_dep) or not callable(audit_auth_dep):
        raise TypeError("audit router dependencies must be callable")
    router = APIRouter(
        prefix="/api/audit",
        tags=["audit"],
        dependencies=[Depends(audit_auth_dep)],
    )

    @router.get("")
    async def query_audit(
        session: Session = Depends(get_session_dep),  # noqa: B008
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
        actor_subject: Annotated[str | None, Query(min_length=1)] = None,
        outcome: AuditOutcome | None = None,
        correlation_id: Annotated[str | None, Query(min_length=1)] = None,
    ) -> AuditPage:
        statement = select(AuditLog)
        if actor_subject is not None:
            statement = statement.where(AuditLog.actor_subject == actor_subject)
        if outcome is not None:
            statement = statement.where(AuditLog.outcome == outcome.value)
        if correlation_id is not None:
            statement = statement.where(AuditLog.correlation_id == correlation_id)
        rows = list(
            session.exec(
                statement.order_by(
                    col(AuditLog.timestamp).desc(), col(AuditLog.id).desc()
                )
                .offset(offset)
                .limit(limit + 1)
            )
        )
        return AuditPage(
            items=[_item(row) for row in rows[:limit]],
            limit=limit,
            offset=offset,
            has_more=len(rows) > limit,
        )

    return router

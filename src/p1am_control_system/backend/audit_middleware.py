"""Automatic append-only audit capture for every SCADA API mutation attempt."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable

from audit_log import AuditEvent, AuditOutcome, append_audit_event
from identity import Principal, Role
from sqlalchemy import Engine
from sqlmodel import Session
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger("dcs_backend.audit")

MUTATION_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
DEFAULT_MAX_PAYLOAD_BYTES = 65_536
_anonymous = Principal(
    subject="unauthenticated.api",
    display_name="Unauthenticated API Client",
    role=Role.VIEWER,
)

PrincipalResolver = Callable[[Request], Principal | None]
RevisionResolver = Callable[[], str]


def _is_mutation(request: Request) -> bool:
    return request.method in MUTATION_METHODS and request.url.path.startswith("/api/")


def _request_payload(request: Request, body: bytes, maximum: int) -> object:
    media_type = request.headers.get("content-type", "unknown").split(";", 1)[0]
    if not body:
        return {}
    if len(body) > maximum:
        return {"body_bytes": len(body), "media_type": media_type, "truncated": True}
    if media_type == "application/json":
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"body_bytes": len(body), "media_type": media_type, "invalid": True}
    return {"body_bytes": len(body), "media_type": media_type}


class MutationAuditMiddleware(BaseHTTPMiddleware):
    """Persist attributed, redacted audit rows without blocking plant controls."""

    def __init__(
        self,
        app: ASGIApp,
        engine: Engine,
        principal_resolver: PrincipalResolver,
        configuration_revision: RevisionResolver,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        super().__init__(app)
        if not isinstance(engine, Engine):
            raise TypeError("engine must be a SQLAlchemy Engine")
        if not callable(principal_resolver) or not callable(configuration_revision):
            raise TypeError("audit resolvers must be callable")
        if not isinstance(max_payload_bytes, int) or max_payload_bytes < 1:
            raise ValueError("max_payload_bytes must be a positive integer")
        self._engine = engine
        self._principal_resolver = principal_resolver
        self._configuration_revision = configuration_revision
        self._max_payload_bytes = max_payload_bytes

    def _principal(self, request: Request) -> Principal:
        try:
            return self._principal_resolver(request) or _anonymous
        except Exception as exc:  # noqa: BLE001 - invalid auth must still be audited
            logger.warning("Audit attribution failed closed: %s", type(exc).__name__)
            return _anonymous

    def _revision(self) -> str:
        try:
            revision = self._configuration_revision().strip()
        except Exception as exc:  # noqa: BLE001 - audit must not block control
            logger.warning("Audit revision lookup failed: %s", type(exc).__name__)
            return "unknown"
        return revision or "unknown"

    def _persist(
        self,
        request: Request,
        body: bytes,
        outcome: AuditOutcome,
        error_code: str | None,
    ) -> None:
        client = request.client.host if request.client else "unknown"
        event = AuditEvent(
            principal=self._principal(request),
            action=f"{request.method.lower()} {request.url.path}",
            target=request.url.path,
            reason=request.headers.get("X-Change-Reason") or "API mutation",
            outcome=outcome,
            before={},
            after={
                "request": _request_payload(
                    request,
                    body,
                    self._max_payload_bytes,
                )
            },
            source=f"api:{client}",
            configuration_revision=self._revision(),
            correlation_id=(
                request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
            ),
            error_code=error_code,
        )
        try:
            with Session(self._engine) as session:
                append_audit_event(session, event)
                session.commit()
        except Exception as exc:  # noqa: BLE001 - never obstruct a control action
            logger.error("Audit persistence failed: %s", type(exc).__name__)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if not _is_mutation(request):
            return await call_next(request)
        body = await request.body()
        try:
            response = await call_next(request)
        except Exception:
            self._persist(request, body, AuditOutcome.FAILED, "EXCEPTION")
            raise
        outcome = (
            AuditOutcome.SUCCEEDED
            if response.status_code < 400
            else AuditOutcome.FAILED
        )
        error_code = (
            None
            if outcome is AuditOutcome.SUCCEEDED
            else f"HTTP_{response.status_code}"
        )
        self._persist(request, body, outcome, error_code)
        return response

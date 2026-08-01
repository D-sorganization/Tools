"""Append-only audit trail for every state-changing request (issue #4029).

Why a separate table
--------------------
``EventLog`` is not an audit trail. It is written by alarm processing, by alarm
acknowledgement, and — critically — by the *client-supplied* ``POST
/api/events``, so any holder of an operator key can forge history. It is also
deleted wholesale by ``POST /api/capture/clear {"include_events": true}``. A
record that the subject of the record can write and erase proves nothing.

``AuditEvent`` lives in its own table that:

- no request handler writes directly (only this module does),
- ``data_capture.clear_capture`` does not know about, so the historian
  maintenance path cannot erase it,
- carries the resolved credential tier, a non-reversible credential
  fingerprint, and the client IP — the actor fields ``EventLog`` never had.

Why middleware
--------------
The real risk is the *next* endpoint, not the current ones. Instrumenting each
handler means every future mutating route starts un-audited until somebody
remembers. A pure-ASGI middleware inverts that default: a new route is audited
the moment it exists, and skipping it requires an explicit entry in
:data:`AUDIT_EXEMPT_PREFIXES`.

Best-effort by contract
-----------------------
Auditing must never fail a control action. Every persistence path here is
wrapped: if the sink is unreachable the request still completes and the failure
is logged. The row is also mirrored to the ``dcs_backend.audit`` logger, which
systemd captures into journald — so a second copy exists off the SQLite file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from auth_config import (
    admin_key_configured,
    is_dev_no_auth,
    verify_admin_key,
    verify_operator_key,
)
from models import utc_now
from sqlmodel import Field, Session, SQLModel

__all__ = [
    "AUDITED_METHODS",
    "AUDIT_EXEMPT_PREFIXES",
    "Actor",
    "AuditEvent",
    "AuditMiddleware",
    "credential_fingerprint",
    "redact_payload",
    "resolve_actor",
]

logger = logging.getLogger("dcs_backend.audit")

#: Methods that can change plant state and therefore must be recorded.
AUDITED_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Read-only analysis routes that happen to use POST for their request body.
#: Recording them would bury real control actions under HMI chatter.
AUDIT_EXEMPT_PREFIXES: tuple[str, ...] = ("/api/explorer",)

#: Payload keys whose values are masked before persistence.
_SENSITIVE_KEY_MARKERS = ("key", "token", "secret", "password", "credential")

#: Upper bound on a captured payload. Larger bodies (project-import archives)
#: are recorded by shape only — the audit trail is not a data store.
_MAX_CAPTURED_BODY_BYTES = 4096

_CREDENTIAL_HEADER = "x-api-" + "key"  # pragma: allowlist secret
_REDACTED = "<redacted>"


class AuditEvent(SQLModel, table=True):  # type: ignore[call-arg]
    """One state-changing request, recorded after the response was produced.

    This table is append-only by construction: nothing in the request-handling
    code path issues a DELETE or UPDATE against it, and the historian clear
    routine does not reference it.
    """

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=utc_now, index=True)
    route: str = Field(index=True)
    method: str
    status_code: int = Field(default=0)
    #: ``admin`` | ``operator`` | ``invalid`` | ``anonymous`` | ``dev-no-auth``
    actor_tier: str = Field(default="anonymous", index=True)
    #: Truncated SHA-256 of the presented credential. Identifies *which* key was
    #: used across requests without storing anything that can be replayed.
    actor_fingerprint: str | None = Field(default=None)
    client_ip: str | None = Field(default=None)
    #: Redacted JSON request body, or a shape description for large/binary ones.
    payload: str | None = Field(default=None)


@dataclass(frozen=True)
class Actor:
    """The resolved identity behind a request. Never carries the credential."""

    tier: str
    fingerprint: str | None


def credential_fingerprint(key: str | None) -> str | None:
    """Return a short, non-reversible fingerprint of a credential.

    Args:
        key: The presented credential, or None/empty when absent.

    Returns:
        The first 16 hex characters of the SHA-256 digest, or None.
    """
    if not key:
        return None
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def resolve_actor(api_key: str | None) -> Actor:
    """Classify the credential presented with a request.

    Distinguishes an admin credential from an operator one, an invalid key from
    no key at all, and flags a bench deployment running with authentication
    disabled so those rows are never mistaken for authenticated actions.
    """
    if is_dev_no_auth():
        return Actor("dev-no-auth", credential_fingerprint(api_key))

    fingerprint = credential_fingerprint(api_key)
    if not api_key:
        return Actor("anonymous", None)
    if verify_admin_key(api_key):
        return Actor("admin", fingerprint)
    if verify_operator_key(api_key):
        # With no distinct admin key the operator credential spans both tiers,
        # so record it as admin — that is the authority it actually carries.
        return Actor("operator" if admin_key_configured() else "admin", fingerprint)
    return Actor("invalid", fingerprint)


def redact_payload(payload: Any) -> Any:
    """Recursively mask credential-like values in a decoded request body.

    A tag write carries ``{"value": 12.5}`` — exactly what an auditor needs —
    but a body could also carry a key. Masking is by key name so the structure
    (and therefore the audit value) survives.
    """
    if isinstance(payload, dict):
        return {
            key: (
                _REDACTED
                if any(marker in str(key).lower() for marker in _SENSITIVE_KEY_MARKERS)
                else redact_payload(value)
            )
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    return payload


def _summarize_body(body: bytes, content_type: str) -> str | None:
    """Render a request body as a redacted, bounded string for storage."""
    if not body:
        return None
    if len(body) > _MAX_CAPTURED_BODY_BYTES:
        return f"<{len(body)} bytes, {content_type or 'unknown type'}, not captured>"
    if not content_type.startswith("application/json"):
        return f"<{len(body)} bytes, {content_type or 'unknown type'}>"
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return f"<{len(body)} bytes, malformed JSON>"
    return json.dumps(redact_payload(decoded), default=str)[:_MAX_CAPTURED_BODY_BYTES]


def _should_audit(
    method: str, path: str, exempt_prefixes: tuple[str, ...] = AUDIT_EXEMPT_PREFIXES
) -> bool:
    """True when this request must produce an audit row."""
    if method.upper() not in AUDITED_METHODS:
        return False
    return not path.startswith(exempt_prefixes)


def record_audit_event(
    session_factory: Callable[[], Session],
    event: AuditEvent,
) -> None:
    """Persist one audit row and mirror it to journald. Never raises.

    Postcondition: a failure to persist is logged at ERROR and the caller
    proceeds — auditing must not be able to stop the plant.
    """
    logger.info(
        "AUDIT %s %s -> %s actor=%s key=%s ip=%s payload=%s",
        event.method,
        event.route,
        event.status_code,
        event.actor_tier,
        event.actor_fingerprint or "-",
        event.client_ip or "-",
        event.payload or "-",
    )
    try:
        with session_factory() as session:
            session.add(event)
            session.commit()
    except Exception as exc:  # noqa: BLE001 - auditing is best-effort by contract
        logger.error("Failed to persist audit row for %s: %s", event.route, exc)


class AuditMiddleware:
    """Pure-ASGI middleware recording every state-changing request.

    Written against the raw ASGI interface rather than ``BaseHTTPMiddleware``
    so the request body can be buffered and replayed deterministically without
    depending on Starlette's internal request caching.
    """

    def __init__(
        self,
        app: Any,
        *,
        session_factory: Callable[[], Session],
        exempt_prefixes: Sequence[str] = AUDIT_EXEMPT_PREFIXES,
    ) -> None:
        self.app = app
        self.session_factory = session_factory
        self.exempt_prefixes = tuple(exempt_prefixes)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        method = scope.get("method", "GET")
        path = scope.get("path", "")
        if scope.get("type") != "http" or not _should_audit(
            method, path, self.exempt_prefixes
        ):
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        body, receive = await self._buffer_body(receive, headers)

        status_holder: dict[str, int] = {"status": 0}

        async def send_wrapper(message: Any) -> None:
            if message.get("type") == "http.response.start":
                status_holder["status"] = int(message.get("status", 0))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self._record(scope, headers, body, status_holder["status"])

    async def _buffer_body(
        self, receive: Any, headers: dict[str, str]
    ) -> tuple[bytes, Any]:
        """Read the body when it is small enough, then hand back a replay."""
        content_type = headers.get("content-type", "")
        try:
            declared = int(headers.get("content-length", "0"))
        except ValueError:
            declared = 0
        if not content_type.startswith("application/json") or declared > (
            _MAX_CAPTURED_BODY_BYTES
        ):
            return b"", receive

        messages: list[Any] = []
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            body += message.get("body", b"")
            more_body = bool(message.get("more_body", False))

        index = 0

        async def replay() -> Any:
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return await receive()

        return body, replay

    def _record(
        self, scope: Any, headers: dict[str, str], body: bytes, status: int
    ) -> None:
        try:
            actor = resolve_actor(headers.get(_CREDENTIAL_HEADER))
            client = scope.get("client")
            query = scope.get("query_string", b"").decode("latin-1")
            route = scope.get("path", "")
            if query:
                route = f"{route}?{query}"
            event = AuditEvent(
                route=route,
                method=scope.get("method", ""),
                status_code=status,
                actor_tier=actor.tier,
                actor_fingerprint=actor.fingerprint,
                client_ip=client[0] if client else None,
                payload=_summarize_body(body, headers.get("content-type", "")),
            )
        except Exception as exc:  # noqa: BLE001 - never break the control path
            logger.error("Failed to build audit row: %s", exc)
            return
        record_audit_event(self.session_factory, event)

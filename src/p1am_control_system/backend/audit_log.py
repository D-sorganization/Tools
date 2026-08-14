"""Append-only, secret-redacting audit domain and SQLite persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from identity import Principal
from models import utc_now
from sqlalchemy import Engine, text
from sqlmodel import Field, Session, SQLModel

if TYPE_CHECKING:
    # Type checkers must see the real 3.11 symbol; TYPE_CHECKING is always
    # true for them and always false at runtime, so this needs no version
    # test and never degrades StrEnum members to bare `str`.
    from enum import StrEnum
else:
    from enum_compat import StrEnum

try:
    from datetime import UTC
except ImportError:  # Python 3.10 support
    UTC = timezone.utc  # noqa: UP017

REDACTED = "[REDACTED]"
_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "session_token",
    "token",
)


class AuditOutcome(StrEnum):
    """Result of an attempted state-changing operation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _optional_text(value: object | None, field_name: str) -> str | None:
    return None if value is None else _required_text(value, field_name)


def _is_secret_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS)


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if _is_secret_key(key) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_redact(item) for item in value]
    return value


def _json_payload(value: object) -> str:
    try:
        return json.dumps(_redact(value), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("audit payload must be JSON-serializable") from exc


@dataclass(frozen=True)
class AuditEvent:
    """Complete attribution contract for one attempted mutation."""

    principal: Principal
    action: str
    target: str
    reason: str
    outcome: AuditOutcome
    before: object
    after: object
    source: str
    configuration_revision: str
    correlation_id: str
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.principal, Principal):
            raise TypeError("principal must be a Principal")
        for field_name in (
            "action",
            "target",
            "reason",
            "source",
            "configuration_revision",
            "correlation_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        if not isinstance(self.outcome, AuditOutcome):
            raise TypeError("outcome must be an AuditOutcome")
        object.__setattr__(
            self,
            "error_code",
            _optional_text(self.error_code, "error_code"),
        )
        _json_payload(self.before)
        _json_payload(self.after)


class AuditLog(SQLModel, table=True):  # type: ignore[call-arg]
    """Immutable persisted representation of :class:`AuditEvent`."""

    id: int | None = Field(default=None, primary_key=True)
    actor_subject: str = Field(index=True)
    actor_display_name: str
    actor_role: str = Field(index=True)
    action: str = Field(index=True)
    target: str = Field(index=True)
    reason: str
    outcome: str = Field(index=True)
    before_json: str
    after_json: str
    source: str
    configuration_revision: str = Field(index=True)
    correlation_id: str = Field(index=True)
    error_code: str | None = Field(default=None)
    timestamp: datetime = Field(default_factory=utc_now, index=True)


def append_audit_event(session: Session, event: AuditEvent) -> AuditLog:
    """Append one audit row; the caller owns the surrounding transaction."""
    if not isinstance(session, Session):
        raise TypeError("session must be a SQLModel Session")
    if not isinstance(event, AuditEvent):
        raise TypeError("event must be an AuditEvent")
    row = AuditLog(
        actor_subject=event.principal.subject,
        actor_display_name=event.principal.display_name,
        actor_role=event.principal.role.value,
        action=event.action,
        target=event.target,
        reason=event.reason,
        outcome=event.outcome.value,
        before_json=_json_payload(event.before),
        after_json=_json_payload(event.after),
        source=event.source,
        configuration_revision=event.configuration_revision,
        correlation_id=event.correlation_id,
        error_code=event.error_code,
        timestamp=datetime.now(UTC),
    )
    session.add(row)
    session.flush()
    return row


def install_append_only_guards(engine: Engine) -> None:
    """Install idempotent database guards that reject audit mutation."""
    if not isinstance(engine, Engine):
        raise TypeError("engine must be a SQLAlchemy Engine")
    statements = (
        "CREATE TRIGGER IF NOT EXISTS auditlog_no_update "
        "BEFORE UPDATE ON auditlog BEGIN "
        "SELECT RAISE(ABORT, 'audit log is append-only'); END",
        "CREATE TRIGGER IF NOT EXISTS auditlog_no_delete "
        "BEFORE DELETE ON auditlog BEGIN "
        "SELECT RAISE(ABORT, 'audit log is append-only'); END",
    )
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

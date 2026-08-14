"""Durable attributable shift entries, sign-off, and handover acknowledgment."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Literal, Protocol

from identity import Principal, Role
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlmodel import Field as SqlField
from sqlmodel import SQLModel

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def _synthetic_id(value: str) -> str:
    normalized = value.strip()
    if not normalized.startswith("SYNTHETIC."):
        raise ValueError("linked identifiers must begin with SYNTHETIC.")
    return normalized


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a UTC offset")
    return value


def _restore_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _required_text(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} is required")
    return normalized


def _canonical_bytes(model: BaseModel) -> bytes:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


class EventReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    occurred_at: datetime

    _event_is_synthetic = field_validator("event_id")(_synthetic_id)
    _timestamp_is_aware = field_validator("occurred_at")(_aware)


class TrendReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    investigation_id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    _investigation_is_synthetic = field_validator("investigation_id")(_synthetic_id)


class ShiftEntryDraft(BaseModel):
    model_config = ConfigDict(frozen=True)

    shift_id: str
    run_id: str
    summary: str = Field(min_length=1, max_length=4000)
    unresolved_actions: tuple[str, ...] = ()
    event_references: tuple[EventReference, ...] = ()
    trend_references: tuple[TrendReference, ...] = ()

    _shift_is_synthetic = field_validator("shift_id")(_synthetic_id)
    _run_is_synthetic = field_validator("run_id")(_synthetic_id)

    @field_validator("summary")
    @classmethod
    def _summary_required(cls, value: str) -> str:
        return _required_text(value, "summary")

    @field_validator("unresolved_actions")
    @classmethod
    def _actions_required(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_required_text(value, "unresolved action") for value in values)


class ShiftEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry_id: str
    shift_id: str
    run_id: str
    summary: str
    unresolved_actions: tuple[str, ...]
    event_references: tuple[EventReference, ...]
    trend_references: tuple[TrendReference, ...]
    created_by: str
    created_at: datetime
    data_classification: Literal["synthetic"] = "synthetic"


class ShiftSignoff(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry_id: str
    signed_by: str
    signed_at: datetime
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HandoverAcknowledgment(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry_id: str
    acknowledged_by: str
    acknowledged_at: datetime
    note: str


class ShiftEntryRecord(SQLModel, table=True):  # type: ignore[call-arg]
    entry_id: str = SqlField(primary_key=True)
    shift_id: str = SqlField(index=True)
    run_id: str = SqlField(index=True)
    summary: str
    unresolved_actions_json: str
    event_references_json: str
    trend_references_json: str
    created_by: str = SqlField(index=True)
    created_at: datetime = SqlField(index=True)


class ShiftSignoffRecord(SQLModel, table=True):  # type: ignore[call-arg]
    entry_id: str = SqlField(primary_key=True, foreign_key="shiftentryrecord.entry_id")
    signed_by: str
    signed_at: datetime
    content_sha256: str


class HandoverAcknowledgmentRecord(SQLModel, table=True):  # type: ignore[call-arg]
    entry_id: str = SqlField(primary_key=True, foreign_key="shiftentryrecord.entry_id")
    acknowledged_by: str
    acknowledged_at: datetime
    note: str


class ShiftLogRepository(Protocol):
    def append(self, entry: ShiftEntry) -> None: ...

    def get(self, entry_id: str) -> ShiftEntry: ...

    def search(self, query: str) -> list[ShiftEntry]: ...

    def sign_off(self, signoff: ShiftSignoff) -> None: ...

    def signoff(self, entry_id: str) -> ShiftSignoff | None: ...

    def acknowledge(self, acknowledgment: HandoverAcknowledgment) -> None: ...

    def handover(self, entry_id: str) -> HandoverAcknowledgment | None: ...


class ShiftLogService:
    def __init__(
        self,
        repository: ShiftLogRepository,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._now = now or (lambda: datetime.now(UTC))

    @staticmethod
    def _authorize(principal: Principal) -> None:
        if principal.role is Role.VIEWER:
            raise PermissionError("operator, engineer, or admin role required")

    def append(self, draft: ShiftEntryDraft, principal: Principal) -> ShiftEntry:
        self._authorize(principal)
        entry = ShiftEntry(
            entry_id=f"shift-entry-{uuid.uuid4().hex}",
            **draft.model_dump(),
            created_by=principal.subject,
            created_at=_aware(self._now()),
        )
        self._repository.append(entry)
        return entry

    def search(self, query: str) -> list[ShiftEntry]:
        return self._repository.search(query)

    def sign_off(self, entry_id: str, principal: Principal) -> ShiftSignoff:
        self._authorize(principal)
        if self._repository.signoff(entry_id) is not None:
            raise ValueError("shift entry is already signed off")
        entry = self._repository.get(entry_id)
        signoff = ShiftSignoff(
            entry_id=entry_id,
            signed_by=principal.subject,
            signed_at=_aware(self._now()),
            content_sha256=hashlib.sha256(_canonical_bytes(entry)).hexdigest(),
        )
        self._repository.sign_off(signoff)
        return signoff

    def acknowledge_handover(
        self,
        entry_id: str,
        principal: Principal,
        note: str,
    ) -> HandoverAcknowledgment:
        self._authorize(principal)
        if self._repository.signoff(entry_id) is None:
            raise ValueError("shift entry must be signed off before handover")
        if self._repository.handover(entry_id) is not None:
            raise ValueError("handover is already acknowledged")
        acknowledgment = HandoverAcknowledgment(
            entry_id=entry_id,
            acknowledged_by=principal.subject,
            acknowledged_at=_aware(self._now()),
            note=_required_text(note, "handover note"),
        )
        self._repository.acknowledge(acknowledgment)
        return acknowledgment

    def handover(self, entry_id: str) -> HandoverAcknowledgment | None:
        return self._repository.handover(entry_id)

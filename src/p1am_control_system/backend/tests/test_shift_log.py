"""F13 attributable shift-log and handover contracts."""

from __future__ import annotations

from datetime import datetime, timezone

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

import pytest
from identity import Principal, Role
from shift_log import (
    EventReference,
    ShiftEntryDraft,
    ShiftLogService,
    TrendReference,
)
from shift_log_repository import SqliteShiftLogRepository
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine


def _fixture() -> tuple[ShiftLogService, object, callable]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def factory() -> Session:
        return Session(engine)

    service = ShiftLogService(
        SqliteShiftLogRepository(factory),
        now=lambda: datetime(2026, 8, 3, 20, 0, tzinfo=UTC),
    )
    return service, engine, factory


def _draft() -> ShiftEntryDraft:
    return ShiftEntryDraft(
        shift_id="SYNTHETIC.SHIFT.2026-08-03-NIGHT",
        run_id="SYNTHETIC.RUN.0042",
        summary="Representative reactor temperature excursion reviewed.",
        unresolved_actions=("Verify synthetic temperature calibration",),
        event_references=(
            EventReference(
                event_id="SYNTHETIC.EVENT.0001",
                occurred_at=datetime(2026, 8, 3, 19, 50, tzinfo=UTC),
            ),
        ),
        trend_references=(
            TrendReference(
                investigation_id="SYNTHETIC.INVESTIGATION.0001",
                content_sha256="a" * 64,
            ),
        ),
    )


def _principal(subject: str) -> Principal:
    return Principal(subject, subject.title(), Role.OPERATOR)


def test_entry_is_attributable_searchable_and_exactly_linked() -> None:
    service, _, _ = _fixture()

    entry = service.append(_draft(), _principal("operator.one"))
    results = service.search("temperature")

    assert results == [entry]
    assert entry.created_by == "operator.one"
    assert entry.event_references[0].event_id == "SYNTHETIC.EVENT.0001"
    assert entry.trend_references[0].content_sha256 == "a" * 64
    assert entry.unresolved_actions == ("Verify synthetic temperature calibration",)


def test_signoff_makes_entry_append_only_even_below_service_layer() -> None:
    service, _, factory = _fixture()
    entry = service.append(_draft(), _principal("operator.one"))

    signoff = service.sign_off(entry.entry_id, _principal("operator.one"))

    assert len(signoff.content_sha256) == 64
    with factory() as session:
        with pytest.raises(DatabaseError, match="signed shift entries are append-only"):
            session.exec(
                text(
                    "UPDATE shiftentryrecord SET summary='tampered' WHERE entry_id=:id"
                ),
                params={"id": entry.entry_id},
            )
            session.commit()


def test_handover_acknowledgment_is_explicit_and_attributable() -> None:
    service, _, _ = _fixture()
    entry = service.append(_draft(), _principal("operator.one"))
    service.sign_off(entry.entry_id, _principal("operator.one"))

    acknowledgment = service.acknowledge_handover(
        entry.entry_id,
        _principal("operator.two"),
        "Unresolved calibration check accepted",
    )

    assert acknowledgment.acknowledged_by == "operator.two"
    assert acknowledgment.note == "Unresolved calibration check accepted"
    assert service.handover(entry.entry_id) == acknowledgment


def test_unsigned_entry_cannot_be_acknowledged() -> None:
    service, _, _ = _fixture()
    entry = service.append(_draft(), _principal("operator.one"))

    with pytest.raises(ValueError, match="signed off"):
        service.acknowledge_handover(
            entry.entry_id,
            _principal("operator.two"),
            "Premature acknowledgment",
        )

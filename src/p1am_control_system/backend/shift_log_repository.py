"""SQLite persistence and database guards for the shift-log domain."""

from __future__ import annotations

import json
from collections.abc import Callable

from shift_log import (
    EventReference,
    HandoverAcknowledgment,
    HandoverAcknowledgmentRecord,
    ShiftEntry,
    ShiftEntryRecord,
    ShiftSignoff,
    ShiftSignoffRecord,
    TrendReference,
    _restore_utc,
)
from sqlalchemy import text
from sqlmodel import Session, col, select

_GUARDS = (
    """CREATE TRIGGER IF NOT EXISTS signed_shift_entry_no_update
    BEFORE UPDATE ON shiftentryrecord
    WHEN EXISTS (SELECT 1 FROM shiftsignoffrecord WHERE entry_id = OLD.entry_id)
    BEGIN SELECT RAISE(ABORT, 'signed shift entries are append-only'); END""",
    """CREATE TRIGGER IF NOT EXISTS signed_shift_entry_no_delete
    BEFORE DELETE ON shiftentryrecord
    WHEN EXISTS (SELECT 1 FROM shiftsignoffrecord WHERE entry_id = OLD.entry_id)
    BEGIN SELECT RAISE(ABORT, 'signed shift entries are append-only'); END""",
    """CREATE TRIGGER IF NOT EXISTS shift_signoff_no_update
    BEFORE UPDATE ON shiftsignoffrecord
    BEGIN SELECT RAISE(ABORT, 'shift signoffs are append-only'); END""",
    """CREATE TRIGGER IF NOT EXISTS shift_signoff_no_delete
    BEFORE DELETE ON shiftsignoffrecord
    BEGIN SELECT RAISE(ABORT, 'shift signoffs are append-only'); END""",
)


class SqliteShiftLogRepository:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _ensure_guards(session: Session) -> None:
        for statement in _GUARDS:
            session.execute(text(statement))

    @staticmethod
    def _entry(record: ShiftEntryRecord) -> ShiftEntry:
        return ShiftEntry(
            entry_id=record.entry_id,
            shift_id=record.shift_id,
            run_id=record.run_id,
            summary=record.summary,
            unresolved_actions=tuple(json.loads(record.unresolved_actions_json)),
            event_references=tuple(
                EventReference.model_validate(item)
                for item in json.loads(record.event_references_json)
            ),
            trend_references=tuple(
                TrendReference.model_validate(item)
                for item in json.loads(record.trend_references_json)
            ),
            created_by=record.created_by,
            created_at=_restore_utc(record.created_at),
        )

    def append(self, entry: ShiftEntry) -> None:
        record = ShiftEntryRecord(
            entry_id=entry.entry_id,
            shift_id=entry.shift_id,
            run_id=entry.run_id,
            summary=entry.summary,
            unresolved_actions_json=json.dumps(entry.unresolved_actions),
            event_references_json=json.dumps(
                [item.model_dump(mode="json") for item in entry.event_references]
            ),
            trend_references_json=json.dumps(
                [item.model_dump(mode="json") for item in entry.trend_references]
            ),
            created_by=entry.created_by,
            created_at=entry.created_at,
        )
        with self._session_factory() as session:
            self._ensure_guards(session)
            session.add(record)
            session.commit()

    def get(self, entry_id: str) -> ShiftEntry:
        with self._session_factory() as session:
            record = session.get(ShiftEntryRecord, entry_id)
            if record is None:
                raise KeyError(f"unknown shift entry: {entry_id}")
            return self._entry(record)

    def search(self, query: str) -> list[ShiftEntry]:
        needle = query.strip().casefold()
        with self._session_factory() as session:
            records = session.exec(
                select(ShiftEntryRecord).order_by(
                    col(ShiftEntryRecord.created_at).desc()
                )
            ).all()
        entries = [self._entry(record) for record in records]
        if not needle:
            return entries
        return [
            entry
            for entry in entries
            if needle
            in " ".join(
                (entry.summary, entry.shift_id, entry.run_id, *entry.unresolved_actions)
            ).casefold()
        ]

    def sign_off(self, signoff: ShiftSignoff) -> None:
        with self._session_factory() as session:
            self._ensure_guards(session)
            session.add(ShiftSignoffRecord(**signoff.model_dump()))
            session.commit()

    def signoff(self, entry_id: str) -> ShiftSignoff | None:
        with self._session_factory() as session:
            record = session.get(ShiftSignoffRecord, entry_id)
            if record is None:
                return None
            return ShiftSignoff(
                entry_id=record.entry_id,
                signed_by=record.signed_by,
                signed_at=_restore_utc(record.signed_at),
                content_sha256=record.content_sha256,
            )

    def acknowledge(self, acknowledgment: HandoverAcknowledgment) -> None:
        with self._session_factory() as session:
            session.add(HandoverAcknowledgmentRecord(**acknowledgment.model_dump()))
            session.commit()

    def handover(self, entry_id: str) -> HandoverAcknowledgment | None:
        with self._session_factory() as session:
            record = session.get(HandoverAcknowledgmentRecord, entry_id)
            if record is None:
                return None
            return HandoverAcknowledgment(
                entry_id=record.entry_id,
                acknowledged_by=record.acknowledged_by,
                acknowledged_at=_restore_utc(record.acknowledged_at),
                note=record.note,
            )

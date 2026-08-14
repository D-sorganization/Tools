"""SQLite adapter for immutable configuration revision documents."""

from __future__ import annotations

from collections.abc import Callable

from configuration_workflow import ConfigurationRevision, ConfigurationState
from sqlalchemy import func
from sqlmodel import Field, Session, SQLModel, col, select


class ConfigurationRevisionRecord(SQLModel, table=True):
    """Durable revision envelope; the JSON document is canonically validated."""

    revision_id: str = Field(primary_key=True)
    version: int = Field(index=True, unique=True)
    state: str = Field(index=True)
    payload_sha256: str = Field(index=True)
    document_json: str


class SqliteRevisionRepository:
    """Persist revision transitions without permitting payload identity rewrites."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory

    @staticmethod
    def _record(revision: ConfigurationRevision) -> ConfigurationRevisionRecord:
        return ConfigurationRevisionRecord(
            revision_id=revision.revision_id,
            version=revision.version,
            state=revision.state.value,
            payload_sha256=revision.payload_sha256,
            document_json=revision.model_dump_json(),
        )

    @staticmethod
    def _revision(record: ConfigurationRevisionRecord) -> ConfigurationRevision:
        return ConfigurationRevision.model_validate_json(record.document_json)

    def next_version(self) -> int:
        with self._session_factory() as session:
            highest = session.exec(
                select(func.max(ConfigurationRevisionRecord.version))
            ).one()
            return int(highest or 0) + 1

    def save(self, revision: ConfigurationRevision) -> None:
        if not isinstance(revision, ConfigurationRevision):
            raise TypeError("revision must be a ConfigurationRevision")
        with self._session_factory() as session:
            existing = session.get(ConfigurationRevisionRecord, revision.revision_id)
            if existing is not None:
                current = self._revision(existing)
                if (
                    current.payload_sha256 != revision.payload_sha256
                    or current.payload != revision.payload
                    or current.version != revision.version
                ):
                    raise ValueError(
                        "configuration revision payload identity is immutable"
                    )
                existing.state = revision.state.value
                existing.document_json = revision.model_dump_json()
                session.add(existing)
            else:
                session.add(self._record(revision))
            session.commit()

    def get(self, revision_id: str) -> ConfigurationRevision:
        if not isinstance(revision_id, str) or not revision_id:
            raise ValueError("revision_id must be a non-empty string")
        with self._session_factory() as session:
            record = session.get(ConfigurationRevisionRecord, revision_id)
            if record is None:
                raise KeyError(f"unknown configuration revision {revision_id!r}")
            return self._revision(record)

    def list(self) -> list[ConfigurationRevision]:
        with self._session_factory() as session:
            records = session.exec(
                select(ConfigurationRevisionRecord).order_by(
                    col(ConfigurationRevisionRecord.version)
                )
            ).all()
            return [self._revision(record) for record in records]

    def activate(self, revision: ConfigurationRevision) -> ConfigurationRevision:
        if revision.state is not ConfigurationState.ACTIVE:
            raise ValueError("activated revision must have active state")
        with self._session_factory() as session:
            target = session.get(ConfigurationRevisionRecord, revision.revision_id)
            if target is None:
                raise KeyError(
                    f"unknown configuration revision {revision.revision_id!r}"
                )
            current_target = self._revision(target)
            if (
                current_target.payload_sha256 != revision.payload_sha256
                or current_target.payload != revision.payload
            ):
                raise ValueError("configuration revision payload identity is immutable")
            active_records = session.exec(
                select(ConfigurationRevisionRecord).where(
                    ConfigurationRevisionRecord.state == ConfigurationState.ACTIVE.value
                )
            ).all()
            for record in active_records:
                current = self._revision(record)
                superseded = current.model_copy(
                    update={"state": ConfigurationState.SUPERSEDED}
                )
                record.state = superseded.state.value
                record.document_json = superseded.model_dump_json()
                session.add(record)
            target.state = revision.state.value
            target.document_json = revision.model_dump_json()
            session.add(target)
            session.commit()
        return revision

"""Durable, reproducible historian investigations with explicit bad-data policy."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal, Protocol

from identity import Principal, Role
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlmodel import Field as SqlField
from sqlmodel import Session, SQLModel

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def _synthetic_tag(value: str) -> str:
    normalized = value.strip()
    if not normalized.startswith("SYNTHETIC."):
        raise ValueError("tags and linked records must begin with SYNTHETIC.")
    return normalized


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a UTC offset")
    return value


def _canonical_bytes(model: BaseModel) -> bytes:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


class BadDataPolicy(StrEnum):
    PRESERVE = "preserve"
    EXCLUDE = "exclude"


class InvestigationQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    tags: tuple[str, ...] = Field(min_length=1, max_length=64)
    start: datetime
    end: datetime
    max_points: int = Field(ge=10, le=100_000)

    @field_validator("tags")
    @classmethod
    def _tags_are_synthetic(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_synthetic_tag(value) for value in values)
        if len(set(normalized)) != len(normalized):
            raise ValueError("query tags must be unique")
        return normalized

    @field_validator("start", "end")
    @classmethod
    def _timestamps_are_aware(cls, value: datetime) -> datetime:
        return _aware(value)

    @model_validator(mode="after")
    def _ordered_window(self) -> InvestigationQuery:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class TagMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    tag: str
    description: str = Field(min_length=1, max_length=300)
    unit: str = Field(min_length=1, max_length=24)
    source: str = Field(min_length=1, max_length=100)

    _tag_is_synthetic = field_validator("tag")(_synthetic_tag)


class Transformation(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation: Literal["moving_average", "difference", "scale", "offset"]
    parameters: dict[str, float | int]


class ChartDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    chart_id: str = Field(min_length=1, max_length=100)
    kind: Literal["trend", "scatter", "histogram"]
    tags: tuple[str, ...] = Field(min_length=1)

    @field_validator("tags")
    @classmethod
    def _chart_tags_are_synthetic(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_synthetic_tag(value) for value in values)


class InvestigationSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: Literal["p1am.synthetic-investigation/v1"] = (
        "p1am.synthetic-investigation/v1"
    )
    title: str = Field(min_length=1, max_length=200)
    query: InvestigationQuery
    tag_metadata: tuple[TagMetadata, ...] = Field(min_length=1)
    transformations: tuple[Transformation, ...] = ()
    charts: tuple[ChartDefinition, ...] = Field(min_length=1)
    annotations: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()
    bad_data_policy: BadDataPolicy
    context: str = Field(min_length=1, max_length=2000)
    data_classification: Literal["synthetic"] = "synthetic"
    not_for_live_control: Literal[True] = True

    @field_validator("event_ids")
    @classmethod
    def _event_ids_are_synthetic(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_synthetic_tag(value) for value in values)

    @model_validator(mode="after")
    def _metadata_covers_query(self) -> InvestigationSpec:
        metadata_tags = {item.tag for item in self.tag_metadata}
        if not set(self.query.tags).issubset(metadata_tags):
            raise ValueError("tag_metadata must cover every query tag")
        query_tags = set(self.query.tags)
        if any(not set(chart.tags).issubset(query_tags) for chart in self.charts):
            raise ValueError("chart tags must be present in the query")
        return self


class SavedInvestigation(BaseModel):
    model_config = ConfigDict(frozen=True)

    investigation_id: str
    version: int = Field(gt=0)
    spec: InvestigationSpec
    created_by: str
    created_at: datetime
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class InvestigationRecord(SQLModel, table=True):  # type: ignore[call-arg]
    investigation_id: str = SqlField(primary_key=True)
    created_at: datetime = SqlField(index=True)
    created_by: str
    content_sha256: str
    document_json: str


class InvestigationRepository(Protocol):
    def save(self, investigation: SavedInvestigation) -> None: ...

    def get(self, investigation_id: str) -> SavedInvestigation: ...


class SqliteInvestigationRepository:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def save(self, investigation: SavedInvestigation) -> None:
        record = InvestigationRecord(
            investigation_id=investigation.investigation_id,
            created_at=investigation.created_at,
            created_by=investigation.created_by,
            content_sha256=investigation.content_sha256,
            document_json=_canonical_bytes(investigation).decode(),
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()

    def get(self, investigation_id: str) -> SavedInvestigation:
        with self._session_factory() as session:
            record = session.get(InvestigationRecord, investigation_id)
            if record is None:
                raise KeyError(f"unknown investigation: {investigation_id}")
            # Annotated local: this package uses flat intra-package imports, which
            # mypy resolves only when invoked from this directory. CI invokes it
            # from the repo root with --follow-imports=skip, where the model
            # becomes Any. Pinning the type keeps the check honest either way.
            loaded: SavedInvestigation = SavedInvestigation.model_validate_json(
                record.document_json
            )
            return loaded


class InvestigationExportManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: Literal["p1am.synthetic-investigation-package/v1"] = (
        "p1am.synthetic-investigation-package/v1"
    )
    investigation_id: str
    entries: dict[str, str]
    data_classification: Literal["synthetic"] = "synthetic"


@dataclass(frozen=True)
class InvestigationArtifact:
    payload: bytes = field(repr=False)
    sha256: str
    manifest: InvestigationExportManifest


def _zip_entry(name: str, payload: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info, payload


class InvestigationService:
    def __init__(
        self,
        repository: InvestigationRepository,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._now = now or (lambda: datetime.now(UTC))

    def save(self, spec: InvestigationSpec, principal: Principal) -> SavedInvestigation:
        if principal.role is Role.VIEWER:
            raise PermissionError("operator, engineer, or admin role required")
        created_at = _aware(self._now())
        content_sha256 = hashlib.sha256(_canonical_bytes(spec)).hexdigest()
        saved = SavedInvestigation(
            investigation_id=f"inv-{uuid.uuid4().hex}",
            version=1,
            spec=spec,
            created_by=principal.subject,
            created_at=created_at,
            content_sha256=content_sha256,
        )
        self._repository.save(saved)
        return saved

    def get(self, investigation_id: str) -> SavedInvestigation:
        return self._repository.get(investigation_id)

    def export(self, investigation_id: str) -> InvestigationArtifact:
        investigation = self.get(investigation_id)
        investigation_bytes = _canonical_bytes(investigation)
        manifest = InvestigationExportManifest(
            investigation_id=investigation_id,
            entries={
                "investigation.json": hashlib.sha256(investigation_bytes).hexdigest()
            },
        )
        manifest_bytes = _canonical_bytes(manifest)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(*_zip_entry("manifest.json", manifest_bytes))
            archive.writestr(*_zip_entry("investigation.json", investigation_bytes))
        payload = buffer.getvalue()
        return InvestigationArtifact(
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            manifest=manifest,
        )

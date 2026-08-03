"""F08 reproducible historian-investigation contracts."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime, timedelta

import pytest
from identity import Principal, Role
from saved_investigation import (
    BadDataPolicy,
    ChartDefinition,
    InvestigationQuery,
    InvestigationService,
    InvestigationSpec,
    SqliteInvestigationRepository,
    TagMetadata,
    Transformation,
)
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine


def _service() -> InvestigationService:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return InvestigationService(SqliteInvestigationRepository(lambda: Session(engine)))


def _spec() -> InvestigationSpec:
    start = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    return InvestigationSpec(
        title="Synthetic temperature excursion review",
        query=InvestigationQuery(
            tags=("SYNTHETIC.REACTOR.TEMPERATURE", "SYNTHETIC.REACTOR.SETPOINT"),
            start=start,
            end=start + timedelta(hours=1),
            max_points=4000,
        ),
        tag_metadata=(
            TagMetadata(
                tag="SYNTHETIC.REACTOR.TEMPERATURE",
                description="Representative reactor temperature",
                unit="°C",
                source="synthetic_driver",
            ),
            TagMetadata(
                tag="SYNTHETIC.REACTOR.SETPOINT",
                description="Representative target",
                unit="°C",
                source="synthetic_driver",
            ),
        ),
        transformations=(
            Transformation(operation="moving_average", parameters={"window": 5}),
        ),
        charts=(
            ChartDefinition(
                chart_id="temperature-context",
                kind="trend",
                tags=("SYNTHETIC.REACTOR.TEMPERATURE",),
            ),
        ),
        annotations=("Synthetic trip at 20:23 UTC",),
        event_ids=("SYNTHETIC.EVENT.0001",),
        bad_data_policy=BadDataPolicy.PRESERVE,
        context="Representative demonstration; no plant records.",
    )


def test_saved_investigation_round_trip_reproduces_complete_context() -> None:
    service = _service()
    principal = Principal("analyst", "Analyst", Role.ENGINEER)

    saved = service.save(_spec(), principal)
    restored = service.get(saved.investigation_id)

    assert restored == saved
    assert restored.created_by == "analyst"
    assert restored.spec.query.tags == _spec().query.tags
    assert restored.spec.tag_metadata == _spec().tag_metadata
    assert restored.spec.transformations == _spec().transformations
    assert restored.spec.charts == _spec().charts
    assert restored.spec.annotations == _spec().annotations
    assert restored.spec.event_ids == _spec().event_ids
    assert restored.spec.bad_data_policy is BadDataPolicy.PRESERVE
    assert len(restored.content_sha256) == 64


def test_export_package_has_reproducible_checksums() -> None:
    service = _service()
    saved = service.save(_spec(), Principal("analyst", "Analyst", Role.ENGINEER))

    artifact = service.export(saved.investigation_id)

    assert hashlib.sha256(artifact.payload).hexdigest() == artifact.sha256
    with zipfile.ZipFile(io.BytesIO(artifact.payload)) as archive:
        assert set(archive.namelist()) == {"manifest.json", "investigation.json"}
        investigation_bytes = archive.read("investigation.json")
        assert (
            hashlib.sha256(investigation_bytes).hexdigest()
            == artifact.manifest.entries["investigation.json"]
        )
        assert b"Synthetic temperature excursion review" in investigation_bytes


def test_bad_data_cannot_be_silently_interpolated() -> None:
    payload = _spec().model_dump()
    payload["bad_data_policy"] = "interpolate"

    with pytest.raises(ValueError):
        InvestigationSpec.model_validate(payload)


def test_query_rejects_non_synthetic_tags_and_inverted_time() -> None:
    start = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="SYNTHETIC"):
        InvestigationQuery(
            tags=("REAL.PLANT.TAG",),
            start=start,
            end=start + timedelta(minutes=1),
            max_points=100,
        )

    with pytest.raises(ValueError, match="after start"):
        InvestigationQuery(
            tags=("SYNTHETIC.TAG",),
            start=start,
            end=start,
            max_points=100,
        )

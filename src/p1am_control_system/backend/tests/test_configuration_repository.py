"""SQLite persistence contracts for configuration revision identity."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration_repository import SqliteRevisionRepository  # noqa: E402
from configuration_workflow import (  # noqa: E402
    ConfigurationRevision,
    ConfigurationState,
)
from models import InterlockConfig, RoutingConfig  # noqa: E402

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def _revision(
    revision_id: str = "cfg-000001-aaaaaaaaaaaa",
    state: ConfigurationState = ConfigurationState.DRAFT,
) -> ConfigurationRevision:
    payload = RoutingConfig(
        input_routing=["TAG_0"],
        output_routing=[],
        pids=[],
        interlocks={
            "TAG_0": InterlockConfig(
                lolo_limit=0,
                low_limit=10,
                high_limit=90,
                hihi_limit=100,
            )
        },
    )
    return ConfigurationRevision(
        revision_id=revision_id,
        version=int(revision_id[4:10]),
        state=state,
        payload=payload,
        payload_sha256="a" * 64,
        reason="Synthetic test revision",
        created_by="engineer",
        created_at=datetime(2026, 8, 3, tzinfo=UTC),
    )


@pytest.fixture
def repository() -> SqliteRevisionRepository:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return SqliteRevisionRepository(lambda: Session(engine))


def test_repository_round_trips_revision_and_monotonic_version(repository) -> None:
    repository.save(_revision())

    restored = repository.get("cfg-000001-aaaaaaaaaaaa")
    assert restored.payload.interlocks["TAG_0"].high_limit == 90
    assert restored.state is ConfigurationState.DRAFT
    assert repository.next_version() == 2


def test_repository_rejects_payload_rewrite_under_existing_identity(repository) -> None:
    original = _revision()
    repository.save(original)
    changed_payload = original.payload.model_copy(deep=True)
    changed_payload.interlocks["TAG_0"].high_limit = 80
    rewritten = original.model_copy(update={"payload": changed_payload})

    with pytest.raises(ValueError, match="immutable"):
        repository.save(rewritten)


def test_activation_supersedes_prior_revision_atomically(repository) -> None:
    first = _revision(state=ConfigurationState.ACTIVE)
    second = _revision("cfg-000002-bbbbbbbbbbbb", ConfigurationState.APPROVED)
    repository.save(first)
    repository.save(second)

    repository.activate(second.model_copy(update={"state": ConfigurationState.ACTIVE}))

    assert repository.get(first.revision_id).state is ConfigurationState.SUPERSEDED
    assert repository.get(second.revision_id).state is ConfigurationState.ACTIVE

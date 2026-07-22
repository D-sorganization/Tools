"""Tests for the durable operator-configuration store (config_store)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

import config_store  # noqa: E402
from config_store import (  # noqa: E402
    PersistedConfig,
    load_config,
    load_model,
    save_config,
    save_model,
)
from pydantic import BaseModel, Field  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


class _Sample(BaseModel):
    setpoint_c: float = Field(ge=0.0)
    label: str


def test_save_and_load_roundtrip(session: Session) -> None:
    save_config(session, "routing", {"a": 1, "b": [1, 2, 3]})
    assert load_config(session, "routing") == {"a": 1, "b": [1, 2, 3]}


def test_load_missing_returns_none(session: Session) -> None:
    assert load_config(session, "nope") is None


def test_save_is_upsert(session: Session) -> None:
    save_config(session, "k", {"v": 1})
    save_config(session, "k", {"v": 2})
    assert load_config(session, "k") == {"v": 2}
    # Exactly one row for the key.
    assert session.get(PersistedConfig, "k") is not None


def test_save_model_and_load_model_roundtrip(session: Session) -> None:
    save_model(session, "temperature", _Sample(setpoint_c=250.0, label="K"))
    restored = load_model(session, "temperature", _Sample)
    assert restored is not None
    assert restored.setpoint_c == 250.0
    assert restored.label == "K"


def test_load_model_missing_returns_none(session: Session) -> None:
    assert load_model(session, "absent", _Sample) is None


def test_load_model_schema_drift_returns_none(session: Session) -> None:
    # Stored blob no longer fits the model (missing required field) -> default.
    save_config(session, "temperature", {"label": "K"})  # no setpoint_c
    assert load_model(session, "temperature", _Sample) is None


def test_load_config_corrupt_row_returns_none(session: Session) -> None:
    session.add(PersistedConfig(key="bad", value_json="{not json"))
    session.commit()
    assert load_config(session, "bad") is None


def test_dbc_guards(session: Session) -> None:
    with pytest.raises(TypeError):
        save_config(session, 123, {"v": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        save_config(session, "", {"v": 1})
    with pytest.raises(TypeError):
        save_config(session, "k", ["not", "a", "dict"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        save_model(session, "k", {"not": "a model"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        load_config(session, 5)  # type: ignore[arg-type]


def test_module_exports_table(session: Session) -> None:
    assert hasattr(config_store, "PersistedConfig")

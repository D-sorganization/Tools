"""Persistence tests for :class:`TemperatureService`.

Covers the durable-settings behaviour wired through the config store:
    - Changing config / setpoint via the service persists it, and a fresh
      service.restore_persisted() recalls both.
    - Restoring persisted settings leaves the controller IDLE (never
      armed/running/energized) — restoring settings must not resume the heater.
    - A stopped-setpoint no-op (controller in IDLE) is NOT persisted, so a later
      restore does not resurrect a value the operator never actually ran.
    - The recalled last setpoint surfaces in status().last_setpoint_c.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("sqlmodel")

from sqlmodel import Session, SQLModel, create_engine  # noqa: E402
from sqlmodel.pool import StaticPool  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config_store  # noqa: E402,F401  (registers PersistedConfig on the metadata)
from temperature_integration import TemperatureService  # noqa: E402
from temperature_models import (  # noqa: E402
    TcType,
    TemperatureState,
)


class _FakePLC:
    """Minimal PLC double — only the coil seam TemperatureService touches."""

    def __init__(self) -> None:
        self.connected = True
        self.write_coil = AsyncMock(return_value=True)


@pytest.fixture()
def engine() -> Any:
    """A fresh in-memory SQLite engine with the config table created."""
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    return eng


def _service(engine: Any) -> TemperatureService:
    return TemperatureService(
        _FakePLC(),
        logging.getLogger("test"),
        session_factory=lambda: Session(engine),
    )


def _arm(service: TemperatureService) -> None:
    """Drive the controller IDLE -> ARMED so a setpoint actually applies."""
    service.controller.set_permissive(True)
    assert service.controller.state == TemperatureState.ARMED


# --------------------------------------------------------------------------
# config persistence + recall
# --------------------------------------------------------------------------


def test_update_config_persists_and_restores(engine: Any) -> None:
    svc = _service(engine)
    new_cfg = svc.controller.config.model_copy(update={"deadband_c": 12.5})
    svc.update_config(new_cfg)

    fresh = _service(engine)
    assert fresh.controller.config.deadband_c != 12.5  # default before restore
    with Session(engine) as s:
        fresh.restore_persisted(s)
    assert fresh.controller.config.deadband_c == 12.5


def test_set_active_tc_type_persists(engine: Any) -> None:
    svc = _service(engine)
    assert svc.controller.config.active_tc_type == TcType.TYPE_K
    svc.set_active_tc_type(TcType.TYPE_R)

    fresh = _service(engine)
    with Session(engine) as s:
        fresh.restore_persisted(s)
    assert fresh.controller.config.active_tc_type == TcType.TYPE_R


# --------------------------------------------------------------------------
# setpoint persistence + recall
# --------------------------------------------------------------------------


def test_applied_setpoint_persists_and_recalls_last_setpoint(engine: Any) -> None:
    svc = _service(engine)
    _arm(svc)
    applied = svc.set_setpoint(300.0)
    assert applied == 300.0

    fresh = _service(engine)
    assert fresh.status().last_setpoint_c is None  # nothing recalled yet
    with Session(engine) as s:
        fresh.restore_persisted(s)
    assert fresh.status().last_setpoint_c == 300.0


def test_stopped_setpoint_noop_is_not_persisted(engine: Any) -> None:
    svc = _service(engine)
    # Controller is IDLE (not armed): set_setpoint is a no-op that must not persist.
    assert svc.controller.state == TemperatureState.IDLE
    svc.set_setpoint(400.0)
    assert svc._last_setpoint_c is None

    fresh = _service(engine)
    with Session(engine) as s:
        fresh.restore_persisted(s)
    assert fresh._last_setpoint_c is None
    assert fresh.status().last_setpoint_c is None


# --------------------------------------------------------------------------
# SAFETY: restore never arms/energizes the heater
# --------------------------------------------------------------------------


def test_restore_leaves_controller_idle(engine: Any) -> None:
    # Persist a running setpoint + config from an armed session.
    svc = _service(engine)
    _arm(svc)
    svc.set_setpoint(500.0)
    assert svc.controller.state == TemperatureState.RUNNING

    fresh = _service(engine)
    with Session(engine) as s:
        fresh.restore_persisted(s)

    # The recalled target is surfaced for the HMI, but the controller stays IDLE:
    # restoring settings must never arm or energize the heater.
    assert fresh.status().last_setpoint_c == 500.0
    assert fresh.controller.state == TemperatureState.IDLE
    assert fresh.controller.status().setpoint_c == 0.0
    assert fresh.controller.status().relay_on is False


def test_restore_bad_blob_does_not_raise(engine: Any) -> None:
    # A corrupt persisted config blob must be skipped, not block boot.
    with Session(engine) as s:
        config_store.save_config(
            s, "temperature_config", {"deadband_c": "not-a-number"}
        )
        config_store.save_config(s, "temperature_setpoint", {"nope": 1})

    fresh = _service(engine)
    with Session(engine) as s:
        fresh.restore_persisted(s)  # must not raise
    # Falls back to defaults; nothing recalled.
    assert fresh.status().last_setpoint_c is None


def test_persistence_disabled_when_no_session_factory(engine: Any) -> None:
    # No session_factory -> service simply skips persistence (existing behaviour).
    svc = TemperatureService(_FakePLC(), logging.getLogger("test"))
    svc.controller.set_permissive(True)
    svc.set_setpoint(250.0)  # must not raise even though nothing is persisted
    with Session(engine) as s:
        assert config_store.load_config(s, "temperature_setpoint") is None

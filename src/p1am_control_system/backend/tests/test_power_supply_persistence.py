"""Durable-settings persistence tests for :class:`PowerSupplyService`.

Mirrors the temperature-persistence contract for the power supply:

    - persist-on-change: an operator config / setpoint change is written to the
      durable store under ``"power_config"`` / ``"power_setpoint"``.
    - restore recall: a fresh service restores the persisted config to the
      controller and surfaces the recalled last setpoint on the status.
    - restore-stays-idle (SAFETY): restoring persisted settings never arms or
      energizes the output — the controller stays IDLE after restore.
    - no-op / no-factory: when ``session_factory`` is ``None`` the service skips
      persistence entirely (existing behaviour is preserved).

These tests exercise the real config store against an in-memory SQLite DB, so
they require SQLModel; the module is skipped when it is unavailable.
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

import config_store  # noqa: E402,F401  (ensures PersistedConfig is registered)
from config_store import load_model, save_config  # noqa: E402
from power_supply import (  # noqa: E402
    PowerSupplyConfig,
    PowerSupplyMode,
    PowerSupplyState,
)
from power_supply_integration import PowerSupplyService  # noqa: E402
from power_supply_models import PowerSupplyLastSetpoint  # noqa: E402

_CONFIG_KEY = "power_config"
_SETPOINT_KEY = "power_setpoint"


class _FakePLC:
    """Minimal PLC stub — only the public write seam the service touches."""

    def __init__(self) -> None:
        self.write_pid_setpoint = AsyncMock(return_value=True)


def _make_engine() -> Any:
    """A shared in-memory SQLite engine with the config table created."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


def _make_service(engine: Any | None) -> PowerSupplyService:
    factory = (lambda: Session(engine)) if engine is not None else None
    return PowerSupplyService(
        plc_client=_FakePLC(),
        logger=logging.getLogger("test.power.persistence"),
        session_factory=factory,
    )


# --------------------------------------------------------------------------
# Constructor guard (DbC)
# --------------------------------------------------------------------------


def test_non_callable_session_factory_rejected() -> None:
    with pytest.raises(TypeError):
        PowerSupplyService(
            plc_client=_FakePLC(),
            logger=logging.getLogger("test.power.persistence"),
            session_factory=object(),  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------
# persist-on-change
# --------------------------------------------------------------------------


def test_update_config_persists() -> None:
    engine = _make_engine()
    service = _make_service(engine)

    new_cfg = PowerSupplyConfig(output_clamp_percent=12.5)
    service.update_config(new_cfg)

    with Session(engine) as s:
        loaded = load_model(s, _CONFIG_KEY, PowerSupplyConfig)
    assert loaded is not None
    assert loaded.output_clamp_percent == 12.5


def test_set_current_setpoint_persists() -> None:
    engine = _make_engine()
    service = _make_service(engine)
    # Arm so the controller actually applies the setpoint (persistence records
    # the operator intent regardless, but arming keeps the test realistic).
    service.set_permissive(True)

    service.set_current_setpoint(42.0)

    with Session(engine) as s:
        loaded = load_model(s, _SETPOINT_KEY, PowerSupplyLastSetpoint)
    assert loaded is not None
    assert loaded.mode == PowerSupplyMode.CURRENT
    assert loaded.value_a == 42.0
    assert service.last_setpoint is not None
    assert service.last_setpoint.value_a == 42.0


def test_set_power_setpoint_persists() -> None:
    engine = _make_engine()
    service = _make_service(engine)

    service.set_power_setpoint(1500.0)

    with Session(engine) as s:
        loaded = load_model(s, _SETPOINT_KEY, PowerSupplyLastSetpoint)
    assert loaded is not None
    assert loaded.mode == PowerSupplyMode.POWER
    assert loaded.value_w == 1500.0


# --------------------------------------------------------------------------
# restore recall
# --------------------------------------------------------------------------


def test_restore_recalls_config_and_setpoint() -> None:
    engine = _make_engine()
    writer = _make_service(engine)
    writer.update_config(PowerSupplyConfig(output_clamp_percent=17.0))
    writer.set_permissive(True)
    writer.set_current_setpoint(33.0)

    fresh = _make_service(engine)
    # Sanity: a fresh service starts at defaults with no recalled setpoint.
    assert fresh.controller.config.output_clamp_percent == 20.0
    assert fresh.status().last_setpoint is None

    with Session(engine) as s:
        fresh.restore_persisted(s)

    assert fresh.controller.config.output_clamp_percent == 17.0
    status = fresh.status()
    assert status.last_setpoint is not None
    assert status.last_setpoint.mode == PowerSupplyMode.CURRENT
    assert status.last_setpoint.value_a == 33.0


# --------------------------------------------------------------------------
# restore-stays-idle (SAFETY)
# --------------------------------------------------------------------------


def test_restore_never_arms_or_energizes() -> None:
    engine = _make_engine()
    writer = _make_service(engine)
    writer.update_config(PowerSupplyConfig(output_clamp_percent=15.0))
    # Drive the writer into a RUNNING, energized state before persisting.
    writer.set_permissive(True)
    writer.set_current_setpoint(50.0)
    assert writer.controller.status().state == PowerSupplyState.RUNNING

    fresh = _make_service(engine)
    with Session(engine) as s:
        fresh.restore_persisted(s)

    status = fresh.status()
    # SAFETY: the controller must come back IDLE, not armed and not running,
    # with a zero live setpoint — only the *recalled* setpoint is surfaced.
    assert status.state == PowerSupplyState.IDLE
    assert status.permissive is False
    assert status.setpoint_a == 0.0
    assert status.commanded_output_percent == 0.0
    # The recalled last setpoint is informational only.
    assert status.last_setpoint is not None
    assert status.last_setpoint.value_a == 50.0


def test_restore_is_best_effort_on_corrupt_blob() -> None:
    engine = _make_engine()
    # Write a garbage config blob directly, then confirm restore doesn't raise
    # and leaves the controller at defaults.
    with Session(engine) as s:
        save_config(s, _CONFIG_KEY, {"output_clamp_percent": "not-a-number"})

    fresh = _make_service(engine)
    with Session(engine) as s:
        fresh.restore_persisted(s)  # must not raise

    assert fresh.controller.config.output_clamp_percent == 20.0
    assert fresh.status().last_setpoint is None


# --------------------------------------------------------------------------
# no-op: session_factory=None skips persistence
# --------------------------------------------------------------------------


def test_no_session_factory_skips_persistence() -> None:
    service = _make_service(None)
    service.set_permissive(True)
    service.set_current_setpoint(10.0)
    service.update_config(PowerSupplyConfig(output_clamp_percent=11.0))

    # Nothing to persist to; the in-memory last setpoint is still tracked so the
    # HMI pre-fill works within a live session, but no DB write is attempted.
    assert service.last_setpoint is not None
    assert service.last_setpoint.value_a == 10.0


def test_persist_failure_does_not_break_command() -> None:
    """A failing session factory must not propagate out of an operator command."""

    def _boom() -> Any:
        raise RuntimeError("db down")

    service = PowerSupplyService(
        plc_client=_FakePLC(),
        logger=logging.getLogger("test.power.persistence"),
        session_factory=_boom,
    )
    service.set_permissive(True)
    # Should swallow the persistence error and still apply/track the setpoint.
    service.set_current_setpoint(7.0)
    assert service.last_setpoint is not None
    assert service.last_setpoint.value_a == 7.0

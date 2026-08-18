"""Endpoint-level guarantees for the commanded E-stop and the PID tuner step.

* #4000 ``POST /api/estop`` reported unqualified success. It must only report
  success once the PLC acknowledged the writes that actually de-energize —
  above all the heater relay coil.
* #4015 the tuner's open-loop step wrote through ``write_tag``, which resolves
  to a register the firmware republishes and never reads. It fitted FOPDT gains
  to a step the plant never saw and returned them as ``status="success"``. The
  step must go through ``write_pid_setpoint``, the seam that reaches the device.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("PLC_DRIVER", "modbus")

pytest.importorskip("sqlmodel")
pytest.importorskip("httpx")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402
from main import app, backup_simulator, control_context, modbus_manager  # noqa: E402
from models import PIDConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _bench_no_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-establish the bench auth bypass for EVERY test in this module.

    This must NOT be a bare ``os.environ`` assignment at import time: ``settings``
    is a module-level singleton read once at first import, so whether the
    variable lands before or after that import depends on collection order and
    xdist worker assignment. A sibling suite that clears the variable at *its*
    import time (``tests/p1am_control_system/test_backend_security.py``) then
    silently disables the bypass for this whole module and these tests fail with
    503 — or pass, under a lucky ordering, which is the dangerous direction over
    an E-stop write path (#4061). A per-test ``monkeypatch`` is immune to that
    and unwinds cleanly afterwards.
    """
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")


# The HMI marker header: cors_config.RequestGuardMiddleware (#4037) refuses a
# state-changing request carrying no preflight-forcing signal, because a bodyless
# control POST is otherwise a CORS-"simple" request any page can make. Inert
# today, set here so this file matches the fleet convention.
client = TestClient(app, headers={"X-Requested-With": "p1am-hmi"})


@pytest.fixture(autouse=True)
def restore_state() -> Generator[None, None, None]:
    import main as backend_main

    original_config = copy.deepcopy(control_context.active_config)
    backend_main.control_context.clear_estop()
    try:
        yield
    finally:
        backend_main.control_context.clear_estop()
        backend_main.power_supply_service.clear_estop()
        backend_main.temperature_service.clear_estop()
        control_context.apply_config(original_config, modbus_manager, backup_simulator)
        control_context.tuning_sessions.clear()


class TestEstopEndpoint:
    def test_success_requires_the_plc_to_acknowledge(self) -> None:
        estop = AsyncMock(return_value=True)
        with (
            patch.object(modbus_manager, "_connected", True),
            patch.object(modbus_manager, "trigger_estop", estop),
        ):
            response = client.post("/api/estop")

        assert response.status_code == 200
        estop.assert_awaited_once()

    def test_unacknowledged_de_energize_is_not_reported_as_success(self) -> None:
        """A False from trigger_estop now means the relay may still be closed."""
        estop = AsyncMock(return_value=False)
        with (
            patch.object(modbus_manager, "_connected", True),
            patch.object(modbus_manager, "trigger_estop", estop),
        ):
            response = client.post("/api/estop")

        assert response.status_code == 502
        detail = response.json()["detail"].lower()
        assert "heater" in detail or "de-energ" in detail


class TestTunerStepWritePath:
    def _arm_session(self) -> None:
        config = copy.deepcopy(control_context.active_config)
        config.pids[1] = PIDConfig(
            pv_tag="TAG_3", cv_tag="TAG_4", setpoint=30.0, kp=1.5, ki=0.2, kd=0.05
        )
        control_context.apply_config(config, modbus_manager, backup_simulator)
        control_context.latest_tags["TAG_3"] = 20.0
        control_context.latest_tags["TAG_4"] = 10.0
        control_context.tuning_sessions[1] = {
            "start_time": 0.0,
            "history": [],
            "step_triggered": False,
        }

    def test_step_reaches_the_device_through_write_pid_setpoint(self) -> None:
        self._arm_session()
        setpoint_write = AsyncMock(return_value=True)
        tag_write = AsyncMock(return_value=True)

        with (
            patch.object(modbus_manager, "_connected", True),
            patch.object(modbus_manager, "write_pid_setpoint", setpoint_write),
            patch.object(modbus_manager, "write_tag", tag_write),
        ):
            response = client.post("/api/pid/1/tuning/step", json={"step_value": 75.0})

        assert response.status_code == 200
        setpoint_write.assert_awaited_once_with(1, 75.0)
        # The dead seam must not be used for the physical step any more.
        tag_write.assert_not_awaited()

    def test_step_failure_is_not_reported_as_success(self) -> None:
        self._arm_session()
        with (
            patch.object(modbus_manager, "_connected", True),
            patch.object(
                modbus_manager, "write_pid_setpoint", AsyncMock(return_value=False)
            ),
        ):
            response = client.post("/api/pid/1/tuning/step", json={"step_value": 75.0})

        assert response.status_code == 502
        assert control_context.tuning_sessions[1]["step_triggered"] is False

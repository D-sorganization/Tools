import copy
import os
from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest

os.environ["PLC_DRIVER"] = "modbus"

pytest.importorskip("httpx")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient
from main import app, backup_simulator, control_context, modbus_manager
from models import PIDConfig

# The HMI marker header: cors_config.RequestGuardMiddleware refuses a
# state-changing request that carries no preflight-forcing signal, because a
# bodyless control POST is otherwise a CORS-"simple" request any page can make
# (#4037). Set it once on the client so every request below is HMI-shaped.


@pytest.fixture(autouse=True)
def _bench_no_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-establish the bench auth bypass for EVERY test in this module.

    This used to be a bare ``os.environ`` assignment at import time, which is
    order-dependent: a sibling suite that clears the variable at *its* import
    time silently disables the bypass for this whole module, and the tests then
    fail with 503 ("no credential configured") depending only on collection
    order and xdist worker assignment (#4061). A per-test ``monkeypatch`` is
    immune to that and unwinds cleanly afterwards.
    """
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")


client = TestClient(app, headers={"X-Requested-With": "p1am-hmi"})


@pytest.fixture(autouse=True)
def restore_control_context() -> Generator[None, None, None]:
    original_config = control_context.active_config
    original_sessions = dict(control_context.tuning_sessions)
    try:
        yield
    finally:
        control_context.apply_config(original_config, modbus_manager, backup_simulator)
        control_context.tuning_sessions.clear()
        control_context.tuning_sessions.update(original_sessions)


def test_pid_tuning_start_rejects_unmapped_pv_tag() -> None:
    """Starting tuning with an unmapped PV tag returns a controlled 4xx."""
    config = copy.deepcopy(control_context.active_config)
    config.pids[1] = PIDConfig(
        pv_tag="TAG_255",
        cv_tag="TAG_4",
        setpoint=30.0,
        kp=1.5,
        ki=0.2,
        kd=0.05,
    )
    control_context.apply_config(config, modbus_manager, backup_simulator)

    response = client.post("/api/pid/1/tuning/start")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "PID loop 1 PV tag 'TAG_255'" in detail
    assert "latest tag values" in detail
    assert 1 not in control_context.tuning_sessions


def test_pid_tuning_step_rejects_unmapped_cv_tag_without_write() -> None:
    """Stepping tuning with an unmapped CV tag must not write a physical output."""
    config = copy.deepcopy(control_context.active_config)
    config.pids[1] = PIDConfig(
        pv_tag="TAG_3",
        cv_tag="TAG_255",
        setpoint=30.0,
        kp=1.5,
        ki=0.2,
        kd=0.05,
    )
    control_context.apply_config(config, modbus_manager, backup_simulator)
    control_context.tuning_sessions[1] = {
        "start_time": 0.0,
        "history": [],
        "step_triggered": False,
    }
    plc_write = AsyncMock(return_value=True)
    simulator_write = AsyncMock(return_value=True)

    with (
        patch.object(modbus_manager, "write_tag", plc_write),
        patch.object(backup_simulator, "write_tag", simulator_write),
    ):
        response = client.post(
            "/api/pid/1/tuning/step",
            json={"step_value": 75.0},
        )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "PID loop 1 CV tag 'TAG_255'" in detail
    plc_write.assert_not_awaited()
    simulator_write.assert_not_awaited()
    assert control_context.tuning_sessions[1]["step_triggered"] is False

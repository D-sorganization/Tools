import logging
import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["PLC_DRIVER"] = "modbus"
os.environ["P1AM_DEV_NO_AUTH"] = "1"

pytest.importorskip("sqlmodel")
pytest.importorskip("httpx")
pytest.importorskip("fastapi.testclient")

import main as backend_main
from defaults import default_routing_config
from fastapi.testclient import TestClient
from main import app, get_session, modbus_manager
from models import PIDConfig
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
client = TestClient(app)


def override_get_session() -> Generator[object, None, None]:
    with backend_main.Session(test_engine) as session:
        yield session


def reset_control_state() -> None:
    backend_main.control_context.apply_config(
        default_routing_config(),
        backend_main.plc_client,
        backend_main.backup_simulator,
    )
    backend_main.control_context.reset_tag_values()
    backend_main.control_context.clear_estop()
    backend_main.control_context.active_alarms.clear()
    backend_main.control_context.tuning_sessions.clear()
    backend_main.latest_frame = {}
    backend_main.shutdown_event.clear()


@pytest.fixture(autouse=True)
def isolated_backend_state() -> Generator[None, None, None]:
    prev_no_auth = os.environ.get("P1AM_DEV_NO_AUTH")
    os.environ["P1AM_DEV_NO_AUTH"] = "1"
    prev_override = app.dependency_overrides.get(get_session)
    app.dependency_overrides[get_session] = override_get_session
    reset_control_state()
    SQLModel.metadata.create_all(test_engine)
    try:
        yield
    finally:
        SQLModel.metadata.drop_all(test_engine)
        reset_control_state()
        if prev_override is None:
            app.dependency_overrides.pop(get_session, None)
        else:
            app.dependency_overrides[get_session] = prev_override
        if prev_no_auth is None:
            os.environ.pop("P1AM_DEV_NO_AUTH", None)
        else:
            os.environ["P1AM_DEV_NO_AUTH"] = prev_no_auth


def test_write_tag_value_rejects_estop_without_write() -> None:
    backend_main.control_context.engage_estop()
    plc_write = AsyncMock(return_value=True)
    simulator_write = AsyncMock(return_value=True)

    with (
        patch.object(modbus_manager, "_connected", True),
        patch.object(modbus_manager, "write_tag", plc_write),
        patch.object(backend_main.backup_simulator, "write_tag", simulator_write),
    ):
        response = client.post("/api/tags/4", json={"value": 12.5})

    assert response.status_code == 409
    assert "E-stop active" in response.json()["detail"]
    plc_write.assert_not_called()
    simulator_write.assert_not_called()


def test_acknowledge_alarm_reports_audit_commit_failure() -> None:
    class FailingCommitSession:
        def __init__(self) -> None:
            self.rollback = MagicMock()

        def add(self, event: object) -> None:
            self.event = event

        def commit(self) -> None:
            raise RuntimeError("audit unavailable")

    failing_session = FailingCommitSession()
    backend_main.control_context.active_alarms["TAG_7"] = {
        "state": "High",
        "acknowledged": False,
    }

    app.dependency_overrides[get_session] = lambda: failing_session
    response = client.post("/api/alarms/TAG_7/acknowledge")

    assert response.status_code == 500
    assert "Failed to persist acknowledgment" in response.json()["detail"]
    assert backend_main.control_context.active_alarms["TAG_7"]["acknowledged"] is False
    failing_session.rollback.assert_called_once()


def test_acknowledge_alarm_reports_failed_ack_result() -> None:
    backend_main.control_context.active_alarms["TAG_8"] = {
        "state": "High",
        "acknowledged": False,
    }

    with patch.object(
        backend_main.control_context, "acknowledge_alarm", return_value=False
    ) as acknowledge:
        response = client.post("/api/alarms/TAG_8/acknowledge")

    assert response.status_code == 409
    assert "could not be acknowledged" in response.json()["detail"]
    # The operator identity is forwarded so the alarm engine can record it
    # in acknowledged_by (issue #4034).
    acknowledge.assert_called_once_with("TAG_8", user=None)


def test_pid_tuning_start_rejects_unmapped_tags() -> None:
    config = default_routing_config()
    config.pids[1] = PIDConfig(
        pv_tag="TAG_255",
        cv_tag="TAG_4",
        setpoint=30.0,
        kp=1.5,
        ki=0.2,
        kd=0.05,
    )
    backend_main.control_context.apply_config(
        config,
        backend_main.plc_client,
        backend_main.backup_simulator,
    )

    response = client.post("/api/pid/1/tuning/start")

    assert response.status_code == 409
    assert "TAG_255" in response.json()["detail"]
    assert 1 not in backend_main.control_context.tuning_sessions


def test_pid_tuning_step_rejects_unmapped_cv_without_write() -> None:
    config = default_routing_config()
    config.pids[1] = PIDConfig(
        pv_tag="TAG_3",
        cv_tag="TAG_255",
        setpoint=30.0,
        kp=1.5,
        ki=0.2,
        kd=0.05,
    )
    backend_main.control_context.apply_config(
        config,
        backend_main.plc_client,
        backend_main.backup_simulator,
    )
    session = {
        "start_time": 100.0,
        "history": [],
        "step_triggered": False,
        "step_time": 0.0,
        "initial_cv": 0.0,
        "initial_pv": 0.0,
        "final_cv": 0.0,
        "final_pv": 0.0,
    }
    backend_main.control_context.tuning_sessions[1] = session
    plc_write = AsyncMock(return_value=True)
    simulator_write = AsyncMock(return_value=True)

    with (
        patch.object(modbus_manager, "write_tag", plc_write),
        patch.object(backend_main.backup_simulator, "write_tag", simulator_write),
    ):
        response = client.post("/api/pid/1/tuning/step", json={"step_value": 75.0})

    assert response.status_code == 409
    assert "TAG_255" in response.json()["detail"]
    assert session["step_triggered"] is False
    plc_write.assert_not_called()
    simulator_write.assert_not_called()


def test_pid_tuning_step_rejects_estop_without_write() -> None:
    response = client.post("/api/pid/1/tuning/start")
    assert response.status_code == 200
    backend_main.control_context.engage_estop()
    plc_write = AsyncMock(return_value=True)
    simulator_write = AsyncMock(return_value=True)

    with (
        patch.object(modbus_manager, "write_tag", plc_write),
        patch.object(backend_main.backup_simulator, "write_tag", simulator_write),
    ):
        response = client.post("/api/pid/1/tuning/step", json={"step_value": 75.0})

    assert response.status_code == 409
    assert "E-stop active" in response.json()["detail"]
    assert backend_main.control_context.tuning_sessions[1]["step_triggered"] is False
    plc_write.assert_not_called()
    simulator_write.assert_not_called()


def test_pid_tuning_stop_without_step_returns_warning() -> None:
    response = client.post("/api/pid/1/tuning/start")
    assert response.status_code == 200

    response = client.post("/api/pid/1/tuning/stop")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "warning"
    assert data["parameters"] == {"kp": 0.0, "tau": 0.0, "theta": 0.0}
    assert data["recommended_pid"] == {"kp": 0.0, "ki": 0.0, "kd": 0.0}


def test_pid_tuning_stop_recommends_expected_pid_for_fixed_history() -> None:
    backend_main.control_context.tuning_sessions[1] = {
        "start_time": 0.0,
        "history": [
            (0.5, 20.0, 11.0),
            (1.5, 20.0, 16.32),
            (2.0, 20.0, 20.0),
            (3.0, 20.0, 20.0),
            (4.0, 20.0, 20.0),
            (5.0, 20.0, 20.0),
            (6.0, 20.0, 20.0),
            (7.0, 20.0, 20.0),
            (8.0, 20.0, 20.0),
            (9.0, 20.0, 20.0),
            (10.0, 20.0, 20.0),
            (11.0, 20.0, 20.0),
        ],
        "step_triggered": True,
        "step_time": 0.0,
        "initial_cv": 10.0,
        "initial_pv": 10.0,
        "final_cv": 20.0,
        "final_pv": 20.0,
    }

    response = client.post("/api/pid/1/tuning/stop")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["parameters"] == {"kp": 1.0, "tau": 1.0, "theta": 0.5}
    assert data["recommended_pid"] == {"kp": 2.916, "ki": 2.833, "kd": 0.486}


@pytest.mark.asyncio
async def test_poll_plc_loop_backs_off_and_surfaces_degraded_status(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sleeps: list[float] = []

    async def failing_poll_once(**_: object) -> dict[str, object]:
        raise RuntimeError("boom")

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        if len(sleeps) == 4:
            backend_main.shutdown_event.set()

    monkeypatch.setattr(backend_main, "_poll_once", failing_poll_once)
    monkeypatch.setattr(backend_main.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(backend_main.settings, "poll_interval_s", 0.1)

    with caplog.at_level(logging.WARNING):
        await backend_main.poll_plc_loop()

    assert sleeps == pytest.approx([0.1, 0.2, 0.4, 0.8])
    assert backend_main.latest_frame["polling_status"]["status"] == "degraded"
    assert backend_main.latest_frame["polling_status"]["consecutive_failures"] == 4
    assert any("PLC polling loop degraded" in rec.message for rec in caplog.records)

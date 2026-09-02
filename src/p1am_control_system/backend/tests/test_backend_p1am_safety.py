import logging
import math
import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["PLC_DRIVER"] = "modbus"

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


# The fixed reference plant for the tuning-route tests: Kp=1, tau=1 s,
# theta=0.5 s, stepped from CV 10% to CV 20%.
REF_PLANT_GAIN = 1.0
REF_PLANT_TAU = 1.0
REF_PLANT_THETA = 0.5
REF_INITIAL_PV = 10.0
REF_INITIAL_CV = 10.0
REF_FINAL_CV = 20.0

# The under-sampled recording of that same plant this file used before the
# 28.3%/63.2% identification correction. Kept as a rejection fixture.
UNDERSAMPLED_HISTORY: list[tuple[float, float, float]] = [
    (0.5, 20.0, 11.0),
    (1.5, 20.0, 16.32),
    *((float(t), 20.0, 20.0) for t in range(2, 12)),
]


def _fopdt_step_history(
    *, dt: float, duration: float
) -> list[tuple[float, float, float]]:
    """Sample a noise-free step response of the reference plant.

    Returns ``(time_offset, cv, pv)`` triples with the step applied at t=0.
    """
    delta_cv = REF_FINAL_CV - REF_INITIAL_CV
    history: list[tuple[float, float, float]] = []
    for i in range(int(round(duration / dt)) + 1):
        t = i * dt
        if t < REF_PLANT_THETA:
            pv = REF_INITIAL_PV
        else:
            pv = REF_INITIAL_PV + REF_PLANT_GAIN * delta_cv * (
                1.0 - math.exp(-(t - REF_PLANT_THETA) / REF_PLANT_TAU)
            )
        history.append((round(t, 4), REF_FINAL_CV, pv))
    return history


def _tuning_session(history: list[tuple[float, float, float]]) -> dict[str, object]:
    return {
        "start_time": 0.0,
        "history": history,
        "step_triggered": True,
        "step_time": 0.0,
        "initial_cv": REF_INITIAL_CV,
        "initial_pv": REF_INITIAL_PV,
        "final_cv": REF_FINAL_CV,
        "final_pv": 20.0,
    }


def test_pid_tuning_stop_recommends_expected_pid_for_fixed_history() -> None:
    """A properly sampled step response yields the expected recommendation.

    This test previously hard-coded a 12-sample history whose PV values 11.0
    and 16.32 were exactly the 10% and 63.2% thresholds of the *old* two-point
    identification, recorded at a 1 s sample interval with only two points on
    the transient. That fixture encoded the biased 10%/63.2% method: under the
    corrected 28.3%/63.2% identification the same data is unresolvable,
    because the first threshold crossing lands within two sample intervals of
    the step. The guard rejecting it is the correct behaviour and is pinned by
    ``test_pid_tuning_stop_rejects_undersampled_step`` below.

    The plant is unchanged (Kp=1, tau=1 s, theta=0.5 s); only the sampling is
    now fine enough to identify it. The residual tau/theta error against the
    true plant is sample quantisation: first-crossing detection snaps the
    28.3% crossing up to the next 0.1 s sample boundary.
    """
    backend_main.control_context.tuning_sessions[1] = _tuning_session(
        _fopdt_step_history(dt=0.1, duration=15.0)
    )

    response = client.post("/api/pid/1/tuning/stop")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["parameters"] == {"kp": 1.0, "tau": 0.9, "theta": 0.6}
    # Kc for this identification is 2.24950, which sits directly on the
    # 3-decimal rounding boundary the response applies, so it is pinned with a
    # tolerance rather than exact equality. 1e-3 is still tight enough to
    # catch any Cohen-Coon coefficient typo, which moves the gains by percent.
    assert data["recommended_pid"] == pytest.approx(
        {"kp": 2.2495, "ki": 1.9093, "kd": 0.4377}, abs=1e-3
    )


def test_pid_tuning_stop_rejects_undersampled_step() -> None:
    """An under-sampled step response must not produce recommended gains.

    Because Kc is proportional to tau/theta, a dead time that cannot be
    resolved at the recorded sample rate inflates the recommendation without
    bound -- this fixture used to yield Kc=2.916 for a plant whose properly
    sampled identification gives 2.249. The route must refuse to emit gains
    rather than hand an unresolvable identification to the operator.
    """
    backend_main.control_context.tuning_sessions[1] = _tuning_session(
        UNDERSAMPLED_HISTORY
    )

    response = client.post("/api/pid/1/tuning/stop")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] != "success"
    assert data["status"] == "warning"
    assert data["recommended_pid"] == {"kp": 0.0, "ki": 0.0, "kd": 0.0}
    assert "sample interval" in data["message"].lower()


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

    monkeypatch.setattr(backend_main, "poll_once", failing_poll_once)
    monkeypatch.setattr(backend_main.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(backend_main.settings, "poll_interval_s", 0.1)

    with caplog.at_level(logging.WARNING):
        await backend_main.poll_plc_loop()

    assert sleeps == pytest.approx([0.1, 0.2, 0.4, 0.8])
    assert backend_main.latest_frame["polling_status"]["status"] == "degraded"
    assert backend_main.latest_frame["polling_status"]["consecutive_failures"] == 4
    assert any("PLC polling loop degraded" in rec.message for rec in caplog.records)

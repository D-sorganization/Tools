import csv
import io
import os
from collections.abc import Generator
from datetime import datetime, timezone

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["PLC_DRIVER"] = "modbus"
# These functional tests exercise endpoint behavior, not the auth gate.
# Opt out of auth here.

pytest.importorskip("sqlmodel")
pytest.importorskip("httpx")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient
from main import app, control_context, get_session, modbus_manager
from models import InterlockConfig, PIDConfig, RoutingConfig, TagLog
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

# Use in-memory SQLite with StaticPool for testing
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


def override_get_session() -> Generator[Session, None, None]:
    with Session(test_engine) as session:
        yield session


app.dependency_overrides[get_session] = override_get_session


@pytest.fixture(autouse=True)
def _bench_no_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the endpoint tests on the explicit bench auth posture (#4061)."""
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")


client = TestClient(app, headers={"X-Requested-With": "p1am-hmi"})


@pytest.fixture(autouse=True)
def setup_db() -> Generator[None, None, None]:
    """Recreate test database schemas before each test case runs.

    Also re-asserts the auth opt-out env var for every test. These functional
    tests exercise endpoint behavior, not the auth gate, so they run with
    ``P1AM_DEV_NO_AUTH=1``. The var is set at module import, but sibling test
    modules (e.g. ``tests/p1am_control_system/test_backend_security.py``) pop
    it from ``os.environ`` at *their* import time during collection, which
    would otherwise leak a 503 auth gate into these tests. Setting it per-test
    makes this suite robust to cross-module collection order (#3289/#3292).
    """
    prev_no_auth = os.environ.get("P1AM_DEV_NO_AUTH")
    os.environ["P1AM_DEV_NO_AUTH"] = "1"
    # Re-assert the session dependency override per-test. ``app`` is a process
    # singleton and ``dependency_overrides`` is shared module state; sibling
    # test modules that import ``from main import app`` and register their own
    # ``get_session`` override can clobber ours during collection, routing the
    # ``/api/export`` and ``/api/import`` endpoints at a different (table-less)
    # engine and yielding ``no such table`` errors. Pinning the override here
    # makes this suite robust to cross-module collection order (#3289/#3292).
    prev_override = app.dependency_overrides.get(get_session)
    app.dependency_overrides[get_session] = override_get_session
    SQLModel.metadata.create_all(test_engine)
    try:
        yield
    finally:
        SQLModel.metadata.drop_all(test_engine)
        if prev_override is None:
            app.dependency_overrides.pop(get_session, None)
        else:
            app.dependency_overrides[get_session] = prev_override
        if prev_no_auth is None:
            os.environ.pop("P1AM_DEV_NO_AUTH", None)
        else:
            os.environ["P1AM_DEV_NO_AUTH"] = prev_no_auth


@pytest.fixture
def sample_routing_config() -> RoutingConfig:
    """Fixture providing a valid dynamic routing config."""
    pids = [
        PIDConfig(pv_tag="TAG_1", cv_tag="TAG_2", setpoint=50.0, kp=1.0, ki=0.5, kd=0.1)
        for _ in range(4)
    ]
    interlocks = {
        f"TAG_{i}": InterlockConfig(
            hihi_limit=105.0, high_limit=95.0, low_limit=10.0, lolo_limit=5.0
        )
        for i in range(32)
    }
    return RoutingConfig(
        input_routing=[f"TAG_{i}" for i in range(6)],
        output_routing=["TAG_10", "TAG_11"],
        pids=pids,
        interlocks=interlocks,
    )


@pytest.mark.asyncio
async def test_get_routing_disconnected() -> None:
    """Ensure GET /api/routing falls back to simulated config.

    This happens when the PLC client is disconnected.
    """
    with patch.object(modbus_manager, "_connected", False):
        response = client.get("/api/routing")
        assert response.status_code == 200
        data = response.json()
        assert "input_routing" in data
        assert len(data["pids"]) == 4


@pytest.mark.asyncio
async def test_get_routing_success() -> None:
    """Verify GET /api/routing successfully reads and decodes Modbus registers."""
    # 1. Setup mock responses from the Modbus client read requests
    mock_input = MagicMock()
    mock_input.isError.return_value = False
    mock_input.registers = [10, 11, 12, 13, 14, 15]

    mock_output = MagicMock()
    mock_output.isError.return_value = False
    mock_output.registers = [20, 21]

    # Create dummy registers for PID configs (4 loops * 10 registers = 40 registers)
    # Register 200 & 201 are integers, 202-209 are
    # packed float words (4 pairs of registers)
    dummy_pid_regs = []
    for _ in range(4):
        dummy_pid_regs.extend([5, 6])  # pv=5, cv=6
        dummy_pid_regs.extend([0, 16256])  # setpoint = 50.0
        dummy_pid_regs.extend([0, 16256])  # kp = 1.0
        dummy_pid_regs.extend([0, 16256])  # ki = 0.5 (or generic packed value)
        dummy_pid_regs.extend([0, 16256])  # kd = 0.1

    mock_pid = MagicMock()
    mock_pid.isError.return_value = False
    mock_pid.registers = dummy_pid_regs

    # Create dummy registers for interlocks (32 tags * 8 registers = 256 registers).
    # read_routing chunks this into four 64-register reads so we mock the same.
    dummy_interlock_regs = []
    for _ in range(32):
        dummy_interlock_regs.extend([0, 16256])  # lolo limit
        dummy_interlock_regs.extend([0, 16256])  # low limit
        dummy_interlock_regs.extend([0, 16256])  # high limit
        dummy_interlock_regs.extend([0, 16256])  # hihi limit

    mock_interlock_chunks = []
    for offset in (0, 64, 128, 192):
        chunk = MagicMock()
        chunk.isError.return_value = False
        chunk.registers = dummy_interlock_regs[offset : offset + 64]
        mock_interlock_chunks.append(chunk)

    # Create AsyncMock to mock client holding register reads
    async_read_mock = AsyncMock()
    async_read_mock.side_effect = [
        mock_input,
        mock_output,
        mock_pid,
        *mock_interlock_chunks,
    ]

    with (
        patch.object(modbus_manager, "_connected", True),
        patch.object(modbus_manager, "client", MagicMock()) as mock_client,
    ):
        mock_client.read_holding_registers = async_read_mock

        response = client.get("/api/routing")
        assert response.status_code == 200
        data = response.json()
        assert data["input_routing"] == [
            "TAG_10",
            "TAG_11",
            "TAG_12",
            "TAG_13",
            "TAG_14",
            "TAG_15",
        ]
        assert data["output_routing"] == ["TAG_20", "TAG_21"]
        assert len(data["pids"]) == 4
        assert len(data["interlocks"]) == 32


@pytest.mark.asyncio
async def test_update_routing_success(
    sample_routing_config: RoutingConfig,
) -> None:
    """Verify POST /api/routing writes configs and triggers Save to Flash coil."""
    mock_write_routing = AsyncMock(return_value=True)
    mock_save_flash = AsyncMock(return_value=True)

    with (
        patch.object(modbus_manager, "_connected", True),
        patch.object(modbus_manager, "write_routing", mock_write_routing),
        patch.object(modbus_manager, "save_to_flash", mock_save_flash),
    ):
        payload = sample_routing_config.model_dump()
        response = client.post("/api/routing", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_write_routing.assert_called_once()
        mock_save_flash.assert_called_once()


@pytest.mark.asyncio
async def test_estop_trigger() -> None:
    """Verify POST /api/estop triggers the client shutdown registers."""
    mock_estop = AsyncMock(return_value=True)

    with (
        patch.object(modbus_manager, "_connected", True),
        patch.object(modbus_manager, "trigger_estop", mock_estop),
    ):
        try:
            response = client.post("/api/estop")
            assert response.status_code == 200
            assert "E-stop triggered" in response.json()["message"]
            mock_estop.assert_called_once()
        finally:
            # `control_context` is a module-level singleton shared by every
            # test in this package. The clear used to be the last statement of
            # the body, so any assertion above failing left E-stop latched for
            # whatever ran next on the same xdist worker -- which surfaced as
            # test_pid_tuning_tag_guards failing with "E-stop active; output
            # writes are inhibited." instead of its own expected message.
            control_context.clear_estop()


def test_export_data() -> None:
    """Verify historical tag states log queries.

    Ensure they are formatted correctly in CSV output.
    """
    db_session = next(override_get_session())

    # Add mock data
    log1 = TagLog(
        tag_name="TAG_2",
        value=35.5,
        timestamp=datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC),
    )
    log2 = TagLog(
        tag_name="TAG_2",
        value=36.0,
        timestamp=datetime(2026, 5, 20, 12, 5, 0, tzinfo=UTC),
    )
    log3 = TagLog(
        tag_name="TAG_5",
        value=78.2,
        timestamp=datetime(2026, 5, 20, 12, 2, 0, tzinfo=UTC),
    )

    db_session.add(log1)
    db_session.add(log2)
    db_session.add(log3)
    db_session.commit()

    # Query CSV
    params = {
        "tag_ids": "2,5",
        "start_time": "2026-05-20T11:59:00Z",
        "end_time": "2026-05-20T12:06:00Z",
    }
    response = client.get("/api/export", params=params)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"

    csv_data = response.text
    reader = csv.reader(io.StringIO(csv_data))
    rows = list(reader)

    # 1 Header row + 3 Data rows
    assert len(rows) == 4
    assert rows[0] == ["Timestamp", "Tag Name", "Value"]
    # Sorted by timestamp asc
    assert rows[1][1] == "TAG_2"
    assert rows[1][2] == "35.5"
    assert rows[2][1] == "TAG_5"
    assert rows[2][2] == "78.2"
    assert rows[3][1] == "TAG_2"
    assert rows[3][2] == "36.0"


def test_websocket_connection() -> None:
    """Verify WebSocket stream connects and allows message cycle loops."""
    # Ensure WebSocket client can connect and read broadcasts
    with client.websocket_connect("/api/stream") as websocket:
        # Just close immediately or read standard loop
        # We can send a test ping to keep connection alive
        websocket.send_text("ping")
        # Ensure it doesn't crash on standard inputs


@pytest.mark.asyncio
async def test_write_tag_success() -> None:
    """Verify manual override POST /api/tags/{tag_id} writes to registers."""
    mock_write_tag = AsyncMock(return_value=True)

    with (
        patch.object(modbus_manager, "_connected", True),
        patch.object(modbus_manager, "write_tag", mock_write_tag),
    ):
        response = client.post("/api/tags/5", json={"value": 42.5})
        assert response.status_code == 200
        assert "Successfully wrote 42.5" in response.json()["message"]
        mock_write_tag.assert_called_once_with("TAG_5", 42.5)


def test_write_tag_invalid_id() -> None:
    """Verify manual override returns error for out-of-bound tag IDs."""
    response = client.post("/api/tags/35", json={"value": 10.0})
    assert response.status_code == 400
    assert "Tag ID must be between 0 and 31" in response.json()["detail"]


def test_write_tag_disconnected() -> None:
    """Verify manual override writes to simulated tags if the PLC is offline."""
    with patch.object(modbus_manager, "_connected", False):
        response = client.post("/api/tags/3", json={"value": 12.0})
        assert response.status_code == 200
        assert "forced simulated tag" in response.json()["message"]


@pytest.fixture
def simulated_alicats(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a bench-simulator MFC registry for the gas endpoints.

    This module runs the app with ``PLC_DRIVER=modbus``, and issue #4031 makes
    "real PLC + simulated gas control" an unreachable combination, so the app's
    own registry is deliberately empty here (see
    ``test_gas_control_is_absent_when_mock_mfcs_are_refused``).
    """
    import main
    from alicat_manager import create_default_manager

    manager = create_default_manager(connection_type="mock", plc_driver="simulator")
    monkeypatch.setattr(main, "alicat_manager", manager)
    return manager


def test_gas_control_is_absent_when_mock_mfcs_are_refused() -> None:
    """Issue #4031: simulated gas must never be served against a real PLC."""
    from main import alicat_manager

    assert alicat_manager.devices == {}
    assert alicat_manager.registration_error is not None

    response = client.get("/api/alicats")
    assert response.status_code == 200
    assert response.json() == []

    response = client.post("/api/alicats/A/setpoint", json={"setpoint": 25.5})
    assert response.status_code == 404


def test_get_alicats_success(simulated_alicats: Any) -> None:
    """Verify GET /api/alicats successfully returns default MFCs."""
    response = client.get("/api/alicats")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    # Check default keys
    mfc_a = next(m for m in data if m["device_id"] == "A")
    assert mfc_a["name"] == "Oxygen MFC"
    assert mfc_a["gas"] == "O2"
    assert mfc_a["max_flow"] == 50.0
    assert mfc_a["connection_state"] == "simulated"


def test_update_alicat_setpoint(simulated_alicats: Any) -> None:
    """Verify POST /api/alicats/{id}/setpoint updates target setpoint."""
    # Test valid MFC update
    response = client.post("/api/alicats/A/setpoint", json={"setpoint": 25.5})
    assert response.status_code == 200
    assert "Setpoint for MFC 'A' set to 25.5" in response.json()["message"]

    # Test setpoint updates on manager
    assert simulated_alicats.devices["A"].setpoint == 25.5

    # Test invalid MFC
    response = client.post("/api/alicats/Z/setpoint", json={"setpoint": 10.0})
    assert response.status_code == 404
    assert "Z" in response.json()["detail"]


def test_update_alicat_gas(simulated_alicats: Any) -> None:
    """Verify POST /api/alicats/{id}/gas updates gas selection calibration."""
    # Test valid gas species change
    response = client.post("/api/alicats/A/gas", json={"gas": "He"})
    assert response.status_code == 200
    assert "Gas species for MFC 'A' set to He" in response.json()["message"]

    assert simulated_alicats.devices["A"].gas == "He"

    # Test invalid gas species
    response = client.post("/api/alicats/A/gas", json={"gas": "Unobtainium"})
    assert response.status_code == 404

    # Test invalid MFC
    response = client.post("/api/alicats/Z/gas", json={"gas": "O2"})
    assert response.status_code == 404


def test_pid_tuning_workflow() -> None:
    """Test start, step, and stop PID tuning endpoints."""
    # 1. Start tuning for PID loop 1
    response = client.post("/api/pid/1/tuning/start")
    assert response.status_code == 200
    assert "Tuning mode started" in response.json()["message"]

    # 2. Apply a step change
    response = client.post("/api/pid/1/tuning/step", json={"step_value": 75.0})
    assert response.status_code == 200
    assert "Step change applied" in response.json()["message"]

    # 3. Stop tuning and verify parameter identification and recommendation
    response = client.post("/api/pid/1/tuning/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("success", "warning")
    assert "parameters" in data
    assert "recommended_pid" in data


def test_mpc_simulation() -> None:
    """Verify that MPC simulation works and yields expected trajectories."""
    payload = {
        "prediction_horizon": 15,
        "control_horizon": 4,
        "setpoint": 65.0,
        "rho": 0.2,
        "process_gain": 1.5,
        "process_tau": 6.0,
        "process_delay": 1.2,
    }
    response = client.post("/api/mpc/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "time" in data
    assert "pid" in data
    assert "mpc" in data
    assert len(data["time"]) == 50
    assert len(data["pid"]["pv"]) == 50
    assert len(data["mpc"]["pv"]) == 50


def test_project_import_and_hierarchy() -> None:
    """Test importing a project zip file, building hierarchy and retrieving it."""
    import io
    import zipfile

    # 1. Create a dummy zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        tagl_content = """{
            "tags": [
                {
                    "name": "Line1_Boiler_Reactor_Temp",
                    "type": "Real",
                    "description": "Reactor Core Temp",
                    "external_availability": "Enabled"
                },
                {
                    "name": "Line1_Boiler_Reactor_Valve",
                    "type": "Boolean",
                    "description": "Inlet Valve Control",
                    "external_availability": "Enabled"
                }
            ]
        }"""
        sdv_content = (
            "Line1_Boiler_Reactor_Temp\tANY\tY:102:B\t2\tANY\t2.5\n"
            "Line1_Boiler_Reactor_Valve\tANY\tC:50:LB\t2\tANY\t1.0\n"
        )

        zip_file.writestr("tagl.json", tagl_content.encode("utf-16"))
        zip_file.writestr("plc_driver_map.SDV", sdv_content.encode("utf-16"))

    zip_buffer.seek(0)

    # 2. Upload the zip
    response = client.post(
        "/api/project/import",
        files={"file": ("project.zip", zip_buffer, "application/zip")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["tags_imported"] == 2
    assert data["mapped_registers"] == 2

    # 3. Query the ladder-explorer endpoint
    response = client.get("/api/project/ladder-explorer")
    assert response.status_code == 200
    explorer_data = response.json()
    assert len(explorer_data) == 2
    temp_tag = next(
        t for t in explorer_data if t["name"] == "Line1_Boiler_Reactor_Temp"
    )
    assert temp_tag["register_type"] == "Y"
    assert temp_tag["register_num"] == 102
    assert temp_tag["scale_factor"] == 2.5
    assert temp_tag["area"] == "Line1"
    assert temp_tag["unit"] == "Boiler"
    assert temp_tag["equipment"] == "Reactor"

    # 4. Query the plant hierarchy layout endpoint
    response = client.get("/api/plant")
    assert response.status_code == 200
    plant_data = response.json()
    assert "Line1" in plant_data["areas"]
    assert "Boiler" in plant_data["areas"]["Line1"]["units"]
    assert "Reactor" in plant_data["areas"]["Line1"]["units"]["Boiler"]["equipment"]
    reactor_tags = plant_data["areas"]["Line1"]["units"]["Boiler"]["equipment"][
        "Reactor"
    ]["tags"]
    assert "Line1_Boiler_Reactor_Temp" in reactor_tags

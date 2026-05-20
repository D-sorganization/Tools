import csv
import io
import os
from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["PLC_DRIVER"] = "modbus"
from fastapi.testclient import TestClient
from main import app, get_session, modbus_manager
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

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db() -> Generator[None, None, None]:
    """Recreate test database schemas before each test case runs."""
    SQLModel.metadata.create_all(test_engine)
    yield
    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture
def sample_routing_config() -> RoutingConfig:
    """Fixture providing a valid 32-channel routing config."""
    pids = [
        PIDConfig(pv_tag_id=1, cv_tag_id=2, setpoint=50.0, kp=1.0, ki=0.5, kd=0.1)
        for _ in range(4)
    ]
    interlocks = [
        InterlockConfig(
            hihi_limit=105.0, high_limit=95.0, low_limit=10.0, lolo_limit=5.0
        )
        for _ in range(32)
    ]
    return RoutingConfig(
        input_routing=[0, 1, 2, 3, 4, 5],
        output_routing=[10, 11],
        pids=pids,
        interlocks=interlocks,
    )


@pytest.mark.asyncio
async def test_get_routing_disconnected() -> None:
    """Ensure GET /api/routing falls back to simulated config when the PLC client is disconnected."""
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

    # Create dummy registers for interlocks (32 tags * 8 registers = 256 registers)
    dummy_interlock_regs = []
    for _ in range(32):
        dummy_interlock_regs.extend([0, 16256])  # lolo limit
        dummy_interlock_regs.extend([0, 16256])  # low limit
        dummy_interlock_regs.extend([0, 16256])  # high limit
        dummy_interlock_regs.extend([0, 16256])  # hihi limit

    mock_interlock = MagicMock()
    mock_interlock.isError.return_value = False
    mock_interlock.registers = dummy_interlock_regs

    # Create AsyncMock to mock client holding register reads
    async_read_mock = AsyncMock()
    async_read_mock.side_effect = [
        mock_input,
        mock_output,
        mock_pid,
        mock_interlock,
    ]

    with (
        patch.object(modbus_manager, "_connected", True),
        patch.object(modbus_manager, "client", MagicMock()) as mock_client,
    ):
        mock_client.read_holding_registers = async_read_mock

        response = client.get("/api/routing")
        assert response.status_code == 200
        data = response.json()
        assert data["input_routing"] == [10, 11, 12, 13, 14, 15]
        assert data["output_routing"] == [20, 21]
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
        response = client.post("/api/estop")
        assert response.status_code == 200
        assert "E-stop triggered" in response.json()["message"]
        mock_estop.assert_called_once()


def test_export_data() -> None:
    """Verify historical tag states log queries.

    Ensure they are formatted correctly in CSV output.
    """
    db_session = next(override_get_session())

    # Add mock data
    log1 = TagLog(
        tag_id=2,
        value=35.5,
        timestamp=datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC),
    )
    log2 = TagLog(
        tag_id=2,
        value=36.0,
        timestamp=datetime(2026, 5, 20, 12, 5, 0, tzinfo=UTC),
    )
    log3 = TagLog(
        tag_id=5,
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
    assert rows[0] == ["Timestamp", "Tag ID", "Value"]
    # Sorted by timestamp asc
    assert rows[1][1] == "2"
    assert rows[1][2] == "35.5"
    assert rows[2][1] == "5"
    assert rows[2][2] == "78.2"
    assert rows[3][1] == "2"
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
        mock_write_tag.assert_called_once_with(5, 42.5)


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


def test_get_alicats_success() -> None:
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


def test_update_alicat_setpoint() -> None:
    """Verify POST /api/alicats/{id}/setpoint updates target setpoint."""
    # Test valid MFC update
    response = client.post("/api/alicats/A/setpoint", json={"setpoint": 25.5})
    assert response.status_code == 200
    assert "Setpoint for MFC 'A' set to 25.5" in response.json()["message"]

    # Test setpoint updates on manager
    from main import alicat_manager

    assert alicat_manager.devices["A"].setpoint == 25.5

    # Test invalid MFC
    response = client.post("/api/alicats/Z/setpoint", json={"setpoint": 10.0})
    assert response.status_code == 404
    assert "Z" in response.json()["detail"]


def test_update_alicat_gas() -> None:
    """Verify POST /api/alicats/{id}/gas updates gas selection calibration."""
    # Test valid gas species change
    response = client.post("/api/alicats/A/gas", json={"gas": "He"})
    assert response.status_code == 200
    assert "Gas species for MFC 'A' set to He" in response.json()["message"]

    from main import alicat_manager

    assert alicat_manager.devices["A"].gas == "He"

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

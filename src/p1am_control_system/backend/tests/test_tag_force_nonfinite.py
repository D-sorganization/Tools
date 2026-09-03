"""NaN/Inf are refused at the tag-force boundary with a typed error (#3974).

Before: ``POST /api/tags/{id}`` with ``{"value": Infinity}`` passed pydantic,
reached ``AsyncModbusManager.write_tag``, and the codec's ``ValueError`` was
swallowed by the blanket ``except Exception`` whose contract is "any I/O
failure drops the connection" -- an input-validation error flagged the live
PLC client disconnected, and subsequent writes silently went to the simulator.

After: the payload model rejects it (422), every write seam raises
``hardware.NonFiniteValueError`` *before* touching the socket, and
``_connected`` is untouched.
"""

from __future__ import annotations

import asyncio
import math
from unittest.mock import MagicMock

import hardware
import pytest

pytest.importorskip("sqlmodel")

from modbus_client import AsyncModbusManager  # noqa: E402
from simulator_client import SimulatedPLCClient  # noqa: E402
from state import SystemState  # noqa: E402
from tuning_router import TagWritePayload  # noqa: E402

_NON_FINITE = [float("nan"), float("inf"), float("-inf")]


class _ExplodingClient:
    """A pymodbus stand-in that must never be reached for a bad value."""

    async def write_registers(self, address: int, values: list[int]) -> MagicMock:
        raise AssertionError("socket touched for a non-finite value")


def _manager() -> AsyncModbusManager:
    m = AsyncModbusManager(host="127.0.0.1")
    m._connected = True
    m._get_client = lambda: _ExplodingClient()
    m.tag_map = {"PUMP_CMD": MagicMock(register_type="V", register_num=1000)}
    return m


def test_error_type_is_a_distinct_value_error() -> None:
    assert issubclass(hardware.NonFiniteValueError, ValueError)
    assert not issubclass(hardware.NonFiniteValueError, OSError)
    with pytest.raises(hardware.NonFiniteValueError):
        hardware.require_finite_value(math.nan)
    with pytest.raises(TypeError):
        hardware.require_finite_value("1.0")
    with pytest.raises(TypeError):
        hardware.require_finite_value(True)
    assert hardware.require_finite_value(3) == 3.0


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_payload_model_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError):
        TagWritePayload(value=bad)


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_modbus_write_tag_raises_typed_error_and_keeps_connection(bad: float) -> None:
    async def go() -> None:
        m = _manager()
        with pytest.raises(hardware.NonFiniteValueError):
            await m.write_tag("PUMP_CMD", bad)
        assert m.connected is True, "a precondition failure is not an I/O failure"

    asyncio.run(go())


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_modbus_write_pid_setpoint_raises_typed_error_and_keeps_connection(
    bad: float,
) -> None:
    async def go() -> None:
        m = _manager()
        with pytest.raises(hardware.NonFiniteValueError):
            await m.write_pid_setpoint(0, bad)
        assert m.connected is True

    asyncio.run(go())


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_simulator_write_tag_raises_typed_error(bad: float) -> None:
    async def go() -> None:
        sim = SimulatedPLCClient()
        with pytest.raises(hardware.NonFiniteValueError):
            await sim.write_tag("TAG_5", bad)
        assert sim.simulated_tags["TAG_5"] == 0.0

    asyncio.run(go())


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_system_state_write_tag_raises_typed_error(bad: float) -> None:
    state = SystemState(alarm_engine_factory=lambda _cfg: MagicMock())
    with pytest.raises(hardware.NonFiniteValueError):
        state.write_tag("TAG_5", bad)
    assert state.latest_tags["TAG_5"] == 0.0


def test_http_route_answers_422_not_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through FastAPI: a NaN body is a client error (422, not the
    500 FastAPI's stock handler produced when echoing a NaN ``input``), and
    the PLC client's connection state is exactly what it was before."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    pytest.importorskip("httpx")
    from sqlalchemy.pool import StaticPool
    from sqlmodel import SQLModel, create_engine

    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    import main as backend_main

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def _session():
        with backend_main.Session(engine) as session:
            yield session

    monkeypatch.setitem(
        backend_main.app.dependency_overrides, backend_main.get_session, _session
    )

    client = fastapi_testclient.TestClient(
        backend_main.app, headers={"X-Requested-With": "p1am-hmi"}
    )
    connected_before = backend_main.plc_client.connected
    if hasattr(backend_main.plc_client, "_get_client"):
        monkeypatch.setattr(
            backend_main.plc_client, "_get_client", lambda: _ExplodingClient()
        )

    # Raw JSON bodies: pydantic parses the bare NaN/Infinity tokens.
    for token in ("NaN", "Infinity", "-Infinity"):
        response = client.post(
            "/api/tags/PUMP_CMD",
            content=f'{{"value": {token}}}',
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422, response.text
        assert "finite" in response.text
        assert backend_main.plc_client.connected is connected_before

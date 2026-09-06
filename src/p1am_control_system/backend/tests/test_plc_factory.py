"""Tests for ``PLCFactory`` env-driven driver selection (issue #3537)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pymodbus")
pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from modbus_client import AsyncModbusManager  # noqa: E402
from plc_factory import PLCFactory  # noqa: E402
from simulator_client import SimulatedPLCClient  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("PLC_DRIVER", "PLC_IP", "PLC_PORT"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    ("driver", "expected_type"),
    [
        ("simulator", SimulatedPLCClient),
        ("p1am", AsyncModbusManager),
        ("modbus", AsyncModbusManager),
        ("does-not-exist", SimulatedPLCClient),  # unknown -> safe simulator
        ("neural", SimulatedPLCClient),  # withdrawn (#4950) -> safe simulator
    ],
)
def test_driver_selection(
    monkeypatch: pytest.MonkeyPatch,
    driver: str,
    expected_type: type,
) -> None:
    monkeypatch.setenv("PLC_DRIVER", driver)
    client = PLCFactory.create_client()
    assert isinstance(client, expected_type)


def test_default_driver_is_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    # No PLC_DRIVER set -> default "simulated" -> simulator.
    client = PLCFactory.create_client()
    assert isinstance(client, SimulatedPLCClient)


def test_modbus_uses_configured_host_and_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PLC_DRIVER", "modbus")
    monkeypatch.setenv("PLC_IP", "10.0.0.5")
    monkeypatch.setenv("PLC_PORT", "1502")
    client = PLCFactory.create_client()
    assert isinstance(client, AsyncModbusManager)
    assert client.host == "10.0.0.5"
    assert client.port == 1502


def test_modbus_bad_port_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PLC_DRIVER", "modbus")
    monkeypatch.setenv("PLC_PORT", "not-a-number")
    client = PLCFactory.create_client()
    assert isinstance(client, AsyncModbusManager)
    assert client.port == 502

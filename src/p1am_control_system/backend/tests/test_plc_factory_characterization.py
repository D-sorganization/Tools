"""Characterization of ``PLCFactory`` driver resolution (issue #5532).

Pins every driver name that resolved before the registry refactor to the
same class with the same constructor arguments, so the refactor is provably
behaviour-preserving.
"""

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


@pytest.mark.parametrize("driver", ["p1am", "modbus"])
def test_real_drivers_resolve_to_modbus_with_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch, driver: str
) -> None:
    monkeypatch.setenv("PLC_DRIVER", driver)
    monkeypatch.setenv("PLC_IP", "10.1.2.3")
    monkeypatch.setenv("PLC_PORT", "1503")
    client = PLCFactory.create_client()
    assert type(client) is AsyncModbusManager
    assert (client.host, client.port) == ("10.1.2.3", 1503)


@pytest.mark.parametrize(
    "driver", ["simulator", "simulated", "does-not-exist", "neural", ""]
)
def test_simulated_and_unknown_names_resolve_to_simulator(
    monkeypatch: pytest.MonkeyPatch, driver: str
) -> None:
    monkeypatch.setenv("PLC_DRIVER", driver)
    assert type(PLCFactory.create_client()) is SimulatedPLCClient

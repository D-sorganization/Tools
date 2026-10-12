"""Tests for the PLC driver registry (issue #5532)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pymodbus")
pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

import plc_factory  # noqa: E402
from modbus_client import AsyncModbusManager  # noqa: E402
from plc_factory import (  # noqa: E402
    DriverCapabilities,
    PLCDriverError,
    PLCFactory,
    register_driver,
    registered_drivers,
)
from settings import REAL_PLC_DRIVERS, P1AMSettings  # noqa: E402
from simulator_client import SimulatedPLCClient  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("PLC_DRIVER", "PLC_IP", "PLC_PORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(plc_factory, "_DRIVERS", dict(plc_factory._DRIVERS))


def test_builtin_drivers_are_registered() -> None:
    assert {"p1am", "modbus", "simulator", "simulated"} <= set(registered_drivers())


def test_real_hardware_capability_matches_settings_allow_list() -> None:
    real = {n for n, c in registered_drivers().items() if c.real_hardware}
    assert real == set(REAL_PLC_DRIVERS)


def test_runtime_registered_driver_is_instantiated_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = SimulatedPLCClient()
    seen: list[P1AMSettings] = []

    def factory(settings: P1AMSettings) -> SimulatedPLCClient:
        seen.append(settings)
        return sentinel

    register_driver("bench-rig", factory, DriverCapabilities(real_hardware=False))
    monkeypatch.setenv("PLC_DRIVER", "bench-rig")
    assert PLCFactory.create_client() is sentinel
    assert len(seen) == 1


def test_duplicate_registration_raises() -> None:
    with pytest.raises(PLCDriverError, match="modbus"):
        register_driver(
            "modbus", lambda s: SimulatedPLCClient(), DriverCapabilities(True)
        )


@pytest.mark.parametrize("name", ["", "   ", None, 3])
def test_invalid_name_rejected(name: object) -> None:
    with pytest.raises((PLCDriverError, TypeError)):
        register_driver(
            name,  # type: ignore[arg-type]
            lambda s: SimulatedPLCClient(),
            DriverCapabilities(False),
        )


def test_non_callable_factory_rejected() -> None:
    with pytest.raises(TypeError):
        register_driver("x", "nope", DriverCapabilities(False))  # type: ignore[arg-type]


def test_unknown_name_still_falls_back_to_simulator(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PLC_DRIVER", "no-such-driver")
    with caplog.at_level("WARNING", logger="dcs_backend.plc_factory"):
        client = PLCFactory.create_client()
    assert type(client) is SimulatedPLCClient
    assert "FELL BACK" in caplog.text


def test_lookup_of_unknown_driver_lists_registered_names() -> None:
    with pytest.raises(PLCDriverError) as exc:
        plc_factory.get_driver("no-such-driver")
    for name in ("p1am", "modbus", "simulator"):
        assert name in str(exc.value)


def test_registered_simulator_gets_banner_real_driver_does_not(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    register_driver(
        "bench-sim", lambda s: SimulatedPLCClient(), DriverCapabilities(False)
    )
    monkeypatch.setenv("PLC_DRIVER", "bench-sim")
    with caplog.at_level("WARNING", logger="dcs_backend.plc_factory"):
        PLCFactory.create_client()
    assert "SIMULATED PLC" in caplog.text
    caplog.clear()
    monkeypatch.setenv("PLC_DRIVER", "modbus")
    with caplog.at_level("WARNING", logger="dcs_backend.plc_factory"):
        PLCFactory.create_client()
    assert "SIMULATED PLC" not in caplog.text


def test_broken_real_driver_raises_and_never_falls_back_to_simulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(settings: P1AMSettings) -> AsyncModbusManager:
        raise RuntimeError("socket exploded")

    register_driver("flaky-plc", boom, DriverCapabilities(real_hardware=True))
    monkeypatch.setenv("PLC_DRIVER", "flaky-plc")
    with pytest.raises(PLCDriverError, match="flaky-plc") as exc:
        PLCFactory.create_client()
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_broken_builtin_modbus_does_not_become_simulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("no route")

    monkeypatch.setattr("modbus_client.AsyncModbusManager.__init__", boom)
    monkeypatch.setenv("PLC_DRIVER", "modbus")
    with pytest.raises(PLCDriverError, match="modbus"):
        PLCFactory.create_client()


def test_registration_is_not_reachable_from_http_surface() -> None:
    main_src = (Path(__file__).parent.parent / "main.py").read_text(encoding="utf-8")
    assert "register_driver" not in main_src

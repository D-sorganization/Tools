from __future__ import annotations

import asyncio
from typing import Any

import pytest
from alicat_manager import (
    DEFAULT_MFC_SPECS,
    MOCK_CONNECTION_TYPE,
    AlicatManager,
    AlicatMFC,
    create_default_manager,
)


@pytest.mark.parametrize(
    ("connection_type", "port_or_ip"),
    [
        ("serial", "COM7"),
        ("tcp", "192.0.2.10"),
    ],
)
def test_physical_setpoint_update_fails_closed(
    connection_type: str, port_or_ip: str
) -> None:
    manager = AlicatManager()
    mfc = AlicatMFC(
        device_id="A",
        name="Physical MFC",
        connection_type=connection_type,
        port_or_ip=port_or_ip,
    )
    manager.add_device(mfc)

    assert manager.update_mfc_setpoint("A", 5.0) is False
    assert mfc.connection_state == "unsupported"
    assert mfc.setpoint == 0.0
    assert mfc.mass_flow == 0.0
    assert mfc._target_setpoint == 0.0


@pytest.mark.parametrize(
    ("connection_type", "port_or_ip"),
    [
        ("serial", "COM7"),
        ("tcp", "192.0.2.10"),
    ],
)
def test_physical_polling_marks_unsupported_without_simulating(
    connection_type: str, port_or_ip: str
) -> None:
    mfc = AlicatMFC(
        device_id="A",
        name="Physical MFC",
        connection_type=connection_type,
        port_or_ip=port_or_ip,
    )

    asyncio.run(mfc.poll_hardware())

    assert mfc.connection_state == "unsupported"
    assert mfc.setpoint == 0.0
    assert mfc.mass_flow == 0.0
    assert mfc.volumetric_flow == 0.0
    assert mfc._target_setpoint == 0.0


def test_mock_setpoint_behavior_is_preserved() -> None:
    manager = AlicatManager()
    mfc = AlicatMFC(device_id="A", name="Mock MFC", connection_type="mock")
    manager.add_device(mfc)

    assert manager.update_mfc_setpoint("A", 5.0) is True
    assert mfc.connection_state == "simulated"
    assert mfc.setpoint == 5.0
    assert mfc._target_setpoint == 5.0

    asyncio.run(mfc.poll_hardware())

    assert mfc.connection_state == "simulated"
    assert mfc.mass_flow > 0.0


# --- issue #4031: simulated gas control must not be reachable in production ---


def test_mock_device_is_refused_when_a_real_plc_driver_is_configured() -> None:
    """A real PLC with simulated MFCs is the exact purge-failure trap (#4031)."""
    manager = AlicatManager(plc_driver="p1am")

    with pytest.raises(ValueError, match="P1AM_ALICAT_CONNECTION_TYPE"):
        manager.add_device(
            AlicatMFC(device_id="A", name="Nitrogen MFC", connection_type="mock")
        )

    assert manager.devices == {}


@pytest.mark.parametrize("driver", ["simulator", "simulated", "SIMULATOR"])
def test_mock_device_is_allowed_against_a_simulated_plc(driver: str) -> None:
    manager = AlicatManager(plc_driver=driver)
    manager.add_device(
        AlicatMFC(device_id="A", name="Nitrogen MFC", connection_type="mock")
    )

    assert manager.devices["A"].connection_state == "simulated"


def test_physical_device_is_allowed_against_a_real_plc_driver() -> None:
    manager = AlicatManager(plc_driver="p1am")
    manager.add_device(
        AlicatMFC(
            device_id="A",
            name="Nitrogen MFC",
            connection_type="serial",
            port_or_ip="/dev/ttyUSB0",
        )
    )

    assert manager.devices["A"].connection_state == "disconnected"


def test_unknown_connection_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="connection_type"):
        AlicatMFC(device_id="A", name="MFC", connection_type="carrier-pigeon")


def test_non_string_connection_type_is_rejected() -> None:
    not_a_transport: Any = 7
    with pytest.raises(TypeError):
        AlicatMFC(device_id="A", name="MFC", connection_type=not_a_transport)


@pytest.mark.parametrize("connection_type", ["serial", "tcp"])
def test_physical_device_requires_a_port_or_ip(connection_type: str) -> None:
    with pytest.raises(ValueError, match="port_or_ip"):
        AlicatMFC(device_id="A", name="MFC", connection_type=connection_type)


def test_create_default_manager_uses_the_configured_connection_type() -> None:
    manager = create_default_manager(
        connection_type="tcp", port_or_ip="192.0.2.10", plc_driver="p1am"
    )

    assert set(manager.devices) == {spec["device_id"] for spec in DEFAULT_MFC_SPECS}
    for device in manager.devices.values():
        assert device.connection_type == "tcp"
        assert device.port_or_ip == "192.0.2.10"
        assert device.connection_state == "disconnected"


def test_create_default_manager_refuses_mock_gas_on_a_real_plc() -> None:
    """Gas control must be plainly absent, never silently simulated (#4031)."""
    manager = create_default_manager(
        connection_type=MOCK_CONNECTION_TYPE, plc_driver="modbus"
    )

    assert manager.devices == {}
    assert manager.registration_error is not None
    assert "P1AM_ALICAT_CONNECTION_TYPE" in manager.registration_error
    assert manager.get_devices_data() == []
    assert manager.update_mfc_setpoint("B", 20.0) is False


def test_create_default_manager_refuses_a_physical_transport_without_a_port() -> None:
    manager = create_default_manager(connection_type="serial", plc_driver="p1am")

    assert manager.devices == {}
    assert manager.registration_error is not None
    assert "port_or_ip" in manager.registration_error


def test_create_default_manager_rejects_an_unknown_transport() -> None:
    with pytest.raises(ValueError, match="connection_type"):
        create_default_manager(connection_type="carrier-pigeon", plc_driver="p1am")


def test_create_default_manager_keeps_the_bench_simulator_working() -> None:
    manager = create_default_manager(
        connection_type=MOCK_CONNECTION_TYPE, plc_driver="simulator"
    )

    assert [d.gas for d in manager.devices.values()] == ["O2", "N2", "CO2"]
    assert all(d.connection_state == "simulated" for d in manager.devices.values())


# --- issue #4031: parse_ascii_response bypassed the VALID_GASES check ---


def test_parse_ascii_response_rejects_a_gas_outside_valid_gases() -> None:
    mfc = AlicatMFC(device_id="A", name="MFC", gas="N2")

    mfc.parse_ascii_response("A 14.65 24.8 12.35 12.35 15.00 Kryptonite")

    assert mfc.gas == "N2", "an unsupported gas must not be written to the device state"
    assert mfc.mass_flow == 12.35
    assert mfc.setpoint == 15.00


def test_parse_ascii_response_accepts_a_valid_gas() -> None:
    mfc = AlicatMFC(device_id="A", name="MFC", gas="N2")

    mfc.parse_ascii_response("A 14.65 24.8 12.35 12.35 15.00 Air")

    assert mfc.gas == "Air"
    assert mfc.pressure == 14.65
    assert mfc.temperature == 24.8


def test_parse_ascii_response_ignores_a_foreign_address() -> None:
    mfc = AlicatMFC(device_id="A", name="MFC", gas="N2")

    mfc.parse_ascii_response("B 14.65 24.8 12.35 12.35 15.00 Air")

    assert mfc.mass_flow == 0.0
    assert mfc.gas == "N2"


def test_setpoint_endpoint_reports_physical_io_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")

    try:
        import main
    except Exception as exc:
        pytest.skip(f"P1AM backend not importable in this environment: {exc}")

    physical_manager = AlicatManager()
    physical_manager.add_device(
        AlicatMFC(
            device_id="A",
            name="Physical MFC",
            connection_type="tcp",
            port_or_ip="192.0.2.10",
        )
    )
    monkeypatch.setattr(main, "alicat_manager", physical_manager)

    payload = main.AlicatSetpointPayload(setpoint=5.0)
    with pytest.raises(main.HTTPException) as exc_info:
        asyncio.run(main.update_alicat_setpoint("A", payload))

    assert exc_info.value.status_code == 503
    assert "physical IO is unsupported" in exc_info.value.detail
    assert physical_manager.devices["A"].connection_state == "unsupported"

from __future__ import annotations

import asyncio

import pytest
from alicat_manager import AlicatManager, AlicatMFC


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

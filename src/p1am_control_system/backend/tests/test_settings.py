"""Settings and timestamp defaults for the P1AM backend (#3541)."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("pydantic_settings")
pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import EventLog, TagLog  # noqa: E402
from settings import P1AMSettings  # noqa: E402


def test_settings_accept_legacy_plc_env_aliases() -> None:
    settings = P1AMSettings(PLC_DRIVER="MODBUS", PLC_IP="10.1.2.3", PLC_PORT="1502")

    assert settings.plc_driver == "modbus"
    assert settings.plc_ip == "10.1.2.3"
    assert settings.plc_port == 1502


def test_settings_accept_p1am_scoped_runtime_tunables() -> None:
    settings = P1AMSettings(
        P1AM_POLL_INTERVAL_S="0.25",
        P1AM_CONNECT_RETRY_INTERVAL_S="2.5",
        P1AM_HISTORIAN_MAX_BYTES="123456",
        P1AM_HISTORIAN_RETENTION_INTERVAL_S="60",
        P1AM_SQLITE_SYNCHRONOUS="full",
    )

    assert settings.poll_interval_s == 0.25
    assert settings.connect_retry_interval_s == 2.5
    assert settings.historian_max_bytes == 123456
    assert settings.historian_retention_interval_s == 60.0
    assert settings.sqlite_synchronous == "FULL"


def test_settings_fall_back_for_bad_legacy_port_and_sqlite_mode() -> None:
    settings = P1AMSettings(PLC_PORT="not-a-number", P1AM_SQLITE_SYNCHRONOUS="bad")

    assert settings.plc_port == 502
    assert settings.sqlite_synchronous == "NORMAL"


def test_model_timestamp_defaults_are_aware_utc() -> None:
    tag = TagLog(tag_name="TAG_0", value=1.0)
    event = EventLog(event_type="SYSTEM", description="startup")

    assert tag.timestamp.tzinfo is not None
    assert event.timestamp.tzinfo is not None
    assert tag.timestamp.utcoffset() == timedelta(0)
    assert event.timestamp.utcoffset() == timedelta(0)


def test_simulated_driver_classification_is_by_real_hardware_allowlist() -> None:
    """#4004: only a named hardware driver counts as real; the rest simulate."""
    from settings import is_simulated_driver

    assert P1AMSettings(PLC_DRIVER="modbus").plc_driver_is_simulated is False
    assert P1AMSettings(PLC_DRIVER="p1am").plc_driver_is_simulated is False
    assert P1AMSettings(PLC_DRIVER="simulator").plc_driver_is_simulated is True
    assert P1AMSettings(PLC_DRIVER="neural").plc_driver_is_simulated is True
    # An unrecognised driver resolves to a simulator in PLCFactory, so it must
    # classify as one here too — never as trustworthy hardware.
    assert P1AMSettings(PLC_DRIVER="typo").plc_driver_is_simulated is True
    assert is_simulated_driver("  MODBUS ") is False


def test_simulated_driver_rejects_a_non_string() -> None:
    from settings import is_simulated_driver

    with pytest.raises(TypeError):
        is_simulated_driver(None)


def test_modbus_timeout_is_sized_to_the_scan_period() -> None:
    """#4009: pymodbus's 3 s default must not stretch a 0.1 s control period."""
    from settings import MIN_MODBUS_TIMEOUT_S

    fast = P1AMSettings(P1AM_POLL_INTERVAL_S="0.1")
    assert fast.resolved_modbus_timeout_s == MIN_MODBUS_TIMEOUT_S

    slow = P1AMSettings(P1AM_POLL_INTERVAL_S="1.5")
    assert slow.resolved_modbus_timeout_s == 1.5

    explicit = P1AMSettings(P1AM_MODBUS_TIMEOUT_S="0.05")
    assert explicit.resolved_modbus_timeout_s == 0.05
def test_alicat_connection_type_defaults_to_mock() -> None:
    settings = P1AMSettings()

    assert settings.alicat_connection_type == "mock"
    assert settings.alicat_port_or_ip is None


def test_alicat_connection_type_is_driven_by_the_environment() -> None:
    """Issue #4031: the MFC transport must not be hardcoded in main.py."""
    settings = P1AMSettings(
        P1AM_ALICAT_CONNECTION_TYPE="TCP",
        P1AM_ALICAT_PORT_OR_IP="192.0.2.10",
    )

    assert settings.alicat_connection_type == "tcp"
    assert settings.alicat_port_or_ip == "192.0.2.10"


def test_alicat_connection_type_rejects_an_unknown_transport() -> None:
    with pytest.raises(ValueError, match="alicat_connection_type"):
        P1AMSettings(P1AM_ALICAT_CONNECTION_TYPE="carrier-pigeon")

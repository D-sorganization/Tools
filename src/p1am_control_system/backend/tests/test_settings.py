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

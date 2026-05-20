"""Unit tests for the P1AM HMI Control System Desktop Auth and Logging.

Verifies role authentication, password verification, event logging, and filtering.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from p1am_control_system.desktop.auth import AuthManager, Role
from p1am_control_system.desktop.event_logger import EventLogger


def test_auth_manager_default_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify standard AuthManager behavior with default password 'Vitro95'."""
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    am = AuthManager()
    assert am.current_role is None
    assert am.is_authenticated() is False

    # Test operator login (should not require password)
    assert am.login(Role.OPERATOR) is True
    assert am.current_role == Role.OPERATOR
    assert am.is_authenticated() is True
    assert am.is_admin() is False

    # Test admin login with wrong password (should fail and maintain operator)
    assert am.login(Role.ADMIN, "wrong_password") is False
    assert am.current_role == Role.OPERATOR
    assert am.is_admin() is False

    # Test admin login with correct password
    assert am.login(Role.ADMIN, "Vitro95") is True
    assert am.current_role == Role.ADMIN
    assert am.is_admin() is True

    # Test logout
    am.logout()
    assert am.current_role is None
    assert am.is_authenticated() is False
    assert am.is_admin() is False


def test_auth_manager_env_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify AuthManager uses environment variable password when present."""
    monkeypatch.setenv("ADMIN_PASSWORD", "SuperSecureEnvPass1!")
    am = AuthManager()

    # Vitro95 should fail
    assert am.login(Role.ADMIN, "Vitro95") is False

    # Environment variable password should succeed
    assert am.login(Role.ADMIN, "SuperSecureEnvPass1!") is True
    assert am.current_role == Role.ADMIN


def test_event_logger_basic(tmp_path: Path) -> None:
    """Verify EventLogger database creation and basic event insertion."""
    db_file = tmp_path / "test_events.db"
    logger = EventLogger(str(db_file))

    # Log several events of different kinds
    logger.log_event("button_click", "INFO", "Operator", "Start Motor")
    logger.log_setpoint_modification("Admin", "TempSetpoint", 45.0, 55.5)
    logger.log_alarm_trip("Temperature High", "Temp > 90C", "CRITICAL")
    logger.log_alarm_acknowledgment("Operator", "Temperature High")

    # Fetch and check
    logs = logger.fetch_logs()
    assert len(logs) == 4

    # Ensure sorting is newest first
    assert logs[0][2] == "alarm_acknowledgment"
    assert logs[0][4] == "Operator"
    assert "Temperature High" in logs[0][5]

    assert logs[1][2] == "alarm_trip"
    assert logs[1][3] == "CRITICAL"

    assert logs[2][2] == "setpoint_modification"
    assert "TempSetpoint" in logs[2][5]
    assert logs[2][6] == "old_value=45.0, new_value=55.5"

    assert logs[3][2] == "button_click"


def test_event_logger_filters(tmp_path: Path) -> None:
    """Verify EventLogger query filtering (by type, severity, keyword, date)."""
    db_file = tmp_path / "test_events.db"
    logger = EventLogger(str(db_file))

    now = datetime.now()

    # Log events with staggered timestamps
    logger.log_event(
        "button_click",
        "INFO",
        "Operator",
        "Open Valve",
        timestamp=now - timedelta(days=2),
    )
    logger.log_event(
        "alarm_trip",
        "WARNING",
        "None",
        "Low pressure warning",
        timestamp=now - timedelta(days=1),
    )
    logger.log_event(
        "operator_login",
        "INFO",
        "Admin",
        "Administrator login",
        timestamp=now,
    )

    # Test Severity Filter
    warn_logs = logger.fetch_logs(severity="WARNING")
    assert len(warn_logs) == 1
    assert warn_logs[0][2] == "alarm_trip"

    # Test Event Type Filter
    btn_logs = logger.fetch_logs(event_type="button_click")
    assert len(btn_logs) == 1
    assert btn_logs[0][5] == "Open Valve"

    # Test Keyword Search Filter
    valve_logs = logger.fetch_logs(keyword="Valve")
    assert len(valve_logs) == 1
    assert valve_logs[0][2] == "button_click"

    admin_logs = logger.fetch_logs(keyword="Admin")
    assert len(admin_logs) == 1
    assert admin_logs[0][2] == "operator_login"

    # Test Date Range Filter
    recent_logs = logger.fetch_logs(start_date=now - timedelta(hours=12))
    assert len(recent_logs) == 1
    assert recent_logs[0][2] == "operator_login"

    old_logs = logger.fetch_logs(end_date=now - timedelta(days=1, hours=12))
    assert len(old_logs) == 1
    assert old_logs[0][2] == "button_click"

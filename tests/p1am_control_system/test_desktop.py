"""Unit tests for the P1AM HMI Control System Desktop Auth and Logging.

Verifies role authentication, password verification, event logging, and filtering.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from p1am_control_system.desktop.auth import (
    AuthManager,
    Role,
    admin_credential_configured,
    hash_admin_password,
)
from p1am_control_system.desktop.event_logger import EventLogger


def test_auth_manager_fails_closed_without_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No credential configured -> Admin login fails closed (regression for #3288)."""
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD_HASH", raising=False)
    am = AuthManager()
    assert am.current_role is None
    assert am.is_authenticated() is False
    assert admin_credential_configured() is False
    assert am.admin_credential_configured() is False

    # Operator login still works (no password required).
    assert am.login(Role.OPERATOR) is True
    assert am.current_role == Role.OPERATOR
    assert am.is_admin() is False

    # The historical hardcoded default 'Vitro95' must NOT grant Admin anymore.
    assert am.login(Role.ADMIN, "Vitro95") is False
    assert am.login(Role.ADMIN, "anything") is False
    assert am.current_role == Role.OPERATOR
    assert am.is_admin() is False


def test_auth_manager_rejects_legacy_hardcoded_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two removed hardcoded fallback hashes must not be accepted (#3288)."""
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD_HASH", raising=False)
    am = AuthManager()
    # Empty password and the old default both fail.
    assert am.verify_admin_password("") is False
    assert am.verify_admin_password("Vitro95") is False


def test_auth_manager_env_plaintext_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AuthManager honors the ADMIN_PASSWORD convenience variable via KDF."""
    monkeypatch.delenv("ADMIN_PASSWORD_HASH", raising=False)
    monkeypatch.setenv("ADMIN_PASSWORD", "SuperSecureEnvPass1!")
    am = AuthManager()
    assert admin_credential_configured() is True

    # Wrong passwords (incl. the old default) fail.
    assert am.login(Role.ADMIN, "Vitro95") is False
    assert am.login(Role.ADMIN, "") is False

    # Correct password succeeds.
    assert am.login(Role.ADMIN, "SuperSecureEnvPass1!") is True
    assert am.current_role == Role.ADMIN

    am.logout()
    assert am.is_authenticated() is False


def test_auth_manager_env_password_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """AuthManager verifies against a salted ADMIN_PASSWORD_HASH (preferred)."""
    encoded = hash_admin_password("CorrectHorseBatteryStaple")
    assert encoded.startswith("pbkdf2_sha256$")
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", encoded)
    am = AuthManager()
    assert admin_credential_configured() is True

    assert am.login(Role.ADMIN, "wrong") is False
    assert am.login(Role.ADMIN, "CorrectHorseBatteryStaple") is True
    assert am.is_admin() is True


def test_hash_admin_password_is_salted_and_kdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hashing uses a salted KDF (distinct salts) and round-trips correctly."""
    h1 = hash_admin_password("same-password")
    h2 = hash_admin_password("same-password")
    # Random salt -> two hashes of the same password differ.
    assert h1 != h2
    # Not bare SHA-256 of the plaintext.
    import hashlib

    assert hashlib.sha256(b"same-password").hexdigest() not in h1

    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", h1)
    am = AuthManager()
    assert am.verify_admin_password("same-password") is True
    assert am.verify_admin_password("different") is False

    with pytest.raises(ValueError):
        hash_admin_password("")


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


def test_hmi_main_window_restyles_when_shared_theme_changes(
    qapp, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HMI main window registers with ThemeManager and follows Light/Dark changes."""

    class _FakeSignal:
        def __init__(self) -> None:
            self._slots = []

        def connect(self, slot) -> None:
            self._slots.append(slot)

    class _FakeWebSocketClientThread:
        def __init__(self, uri: str) -> None:
            self.uri = uri
            self.messageReceived = _FakeSignal()
            self.connectionStatusChanged = _FakeSignal()

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def wait(self) -> None:
            pass

    class _FakeUnifiedToolsSidebar:
        def __init__(self, parent=None) -> None:
            self.parent = parent

        def install_as_dock(self, main_window, area: str = "right") -> None:
            self.main_window = main_window
            self.area = area

        def register_shortcuts(self, main_window) -> None:
            self.shortcut_parent = main_window

    from theme.theme_manager import ThemeManager, get_theme_manager

    from p1am_control_system.desktop import header as hmi_header

    monkeypatch.setitem(sys.modules, "dotenv", None)
    monkeypatch.setitem(sys.modules, "requests", None)
    monkeypatch.setitem(sys.modules, "websockets", None)

    from p1am_control_system.desktop import main_window as hmi_main_window

    settings_app = f"HMIMainWindowThemeTest-{tmp_path.name}"

    def _test_theme_manager(window=None):
        return get_theme_manager(window, settings_app=settings_app)

    monkeypatch.setenv("EVENT_LOG_DB_PATH", str(tmp_path / "events.db"))
    monkeypatch.setattr(hmi_header, "get_theme_manager", _test_theme_manager)
    monkeypatch.setattr(hmi_main_window, "get_theme_manager", _test_theme_manager)
    monkeypatch.setattr(
        hmi_main_window.HMIMainWindow, "_load_routing_config", lambda self: None
    )
    monkeypatch.setattr(
        hmi_main_window, "WebSocketClientThread", _FakeWebSocketClientThread
    )
    monkeypatch.setattr(
        hmi_main_window, "UnifiedToolsSidebar", _FakeUnifiedToolsSidebar
    )

    ThemeManager.reset_instance()
    window = hmi_main_window.HMIMainWindow()
    manager = window.theme_manager
    previous_theme = manager.get_theme_preference()
    try:
        manager.change_theme("Dark")
        dark_stylesheet = window.styleSheet()
        manager.change_theme("Light")
        qapp.processEvents()

        assert window.styleSheet() != dark_stylesheet
        assert window.header.theme_btn.text() == "Theme: Light"
        assert "#1a1d23" not in window.log_list.styleSheet().lower()
    finally:
        if previous_theme in manager.get_available_themes():
            manager.change_theme(previous_theme)
        window.close()
        ThemeManager.reset_instance()

"""Regression tests for shared PyQt font management helpers."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

import shared.python.theme.font_manager as font_manager_module
from shared.python.theme.font_manager import FontManager, get_font_manager


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


@pytest.fixture(autouse=True)
def reset_font_manager_singleton() -> None:
    FontManager._instance = None
    yield
    FontManager._instance = None


def _settings_scope() -> tuple[str, str]:
    return "D-sorganization-tests", f"FontManager-{uuid4().hex}"


def test_loads_default_font_from_empty_settings() -> None:
    org, app = _settings_scope()
    settings = QSettings(org, app)
    settings.clear()

    manager = FontManager(settings_org=org, settings_app=app)

    assert manager.get_current_font() == "Inter"


def test_loads_persisted_font_from_context_group() -> None:
    org, app = _settings_scope()
    settings = QSettings(org, app)
    settings.clear()
    settings.beginGroup("Font_Modeling")
    settings.setValue("font_family", "Roboto")
    settings.endGroup()

    manager = FontManager(
        app_context="Modeling",
        settings_org=org,
        settings_app=app,
    )

    assert manager.get_current_font() == "Roboto"


def test_available_fonts_always_include_system_default(qapp: QApplication) -> None:
    org, app = _settings_scope()
    manager = FontManager(settings_org=org, settings_app=app)

    assert "System Default" in manager.get_available_fonts()


def test_change_font_persists_applies_and_emits(qapp: QApplication) -> None:
    org, app = _settings_scope()
    manager = FontManager(settings_org=org, settings_app=app)
    emitted: list[str] = []
    manager.fontChanged.connect(emitted.append)

    manager.change_font("Arial")

    assert manager.get_current_font() == "Arial"
    assert qapp.font().family() == "Arial"
    assert emitted == ["Arial"]

    settings = QSettings(org, app)
    settings.beginGroup("Font_Global")
    assert settings.value("font_family", type=str) == "Arial"


def test_change_font_noops_when_font_is_current(qapp: QApplication) -> None:
    org, app = _settings_scope()
    manager = FontManager(settings_org=org, settings_app=app)
    emitted: list[str] = []
    manager.fontChanged.connect(emitted.append)

    manager.change_font(manager.get_current_font())

    assert emitted == []


def test_apply_font_handles_system_default(qapp: QApplication) -> None:
    org, app = _settings_scope()
    manager = FontManager(settings_org=org, settings_app=app)

    manager.change_font("System Default")

    assert manager.get_current_font() == "System Default"
    assert qapp.font().family()


def test_apply_font_without_qapplication_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    org, app = _settings_scope()
    manager = FontManager(settings_org=org, settings_app=app)
    monkeypatch.setattr(
        font_manager_module.QApplication,
        "instance",
        staticmethod(lambda: None),
    )

    manager.apply_font()

    assert "No QApplication instance available" in caplog.text


def test_get_font_manager_returns_singleton() -> None:
    org, app = _settings_scope()

    first = get_font_manager(
        app_context="Shared",
        settings_org=org,
        settings_app=app,
    )
    second = get_font_manager(
        app_context="Ignored",
        settings_org=org,
        settings_app=app,
    )

    assert first is second
    assert first.app_context == "Shared"

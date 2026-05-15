"""Regression tests for shared application zoom behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication

from shared.python.theme.zoom import (
    ApplicationZoomController,
    ZoomConfig,
    ZoomTokenSet,
    install_application_zoom,
    scale_px,
)


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


def _wheel_event(modifiers: Qt.KeyboardModifier) -> QWheelEvent:
    return QWheelEvent(
        QPointF(1, 1),
        QPointF(1, 1),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _key_event(key: int, modifiers: Qt.KeyboardModifier) -> QKeyEvent:
    return QKeyEvent(QKeyEvent.Type.KeyPress, key, modifiers)


def test_zoom_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="minimum_percent"):
        ZoomConfig(minimum_percent=0)
    with pytest.raises(ValueError, match="default_percent"):
        ZoomConfig(default_percent=500)


def test_set_zoom_clamps_and_persists(qapp: QApplication) -> None:
    settings = MagicMock()
    settings.value.return_value = 100
    controller = ApplicationZoomController(qapp, settings=settings)

    controller.set_zoom_percent(999)

    assert controller.zoom_percent == 200
    settings.setValue.assert_called_with("percent", 200)


def test_zoom_changes_application_font_size(qapp: QApplication) -> None:
    settings = MagicMock()
    settings.value.return_value = 100
    controller = ApplicationZoomController(qapp, settings=settings)
    base_size = controller.base_point_size

    controller.set_zoom_percent(150)

    assert qapp.font().pointSizeF() == pytest.approx(base_size * 1.5)


def test_ctrl_wheel_is_handled_but_plain_wheel_is_not(qapp: QApplication) -> None:
    settings = MagicMock()
    settings.value.return_value = 100
    controller = ApplicationZoomController(qapp, settings=settings)

    plain_wheel = _wheel_event(Qt.KeyboardModifier.NoModifier)
    ctrl_wheel = _wheel_event(Qt.KeyboardModifier.ControlModifier)

    assert controller.eventFilter(None, plain_wheel) is False
    assert controller.eventFilter(None, ctrl_wheel) is True
    assert controller.zoom_percent == 110


def test_ctrl_shortcuts_zoom_and_reset(qapp: QApplication) -> None:
    settings = MagicMock()
    settings.value.return_value = 100
    controller = ApplicationZoomController(qapp, settings=settings)

    ctrl = Qt.KeyboardModifier.ControlModifier

    assert controller.eventFilter(None, _key_event(Qt.Key.Key_Plus, ctrl))
    assert controller.zoom_percent == 110
    assert controller.eventFilter(None, _key_event(Qt.Key.Key_Minus, ctrl))
    assert controller.zoom_percent == 100
    assert controller.eventFilter(None, _key_event(Qt.Key.Key_0, ctrl))
    assert controller.zoom_percent == 100


def test_scale_px_uses_zoom_percent_contract() -> None:
    assert scale_px(10, 150) == 15
    assert scale_px(10, 50) == 5
    with pytest.raises(ValueError, match="zoom_percent"):
        scale_px(10, 0)


def test_zoom_tokens_scale_common_ui_dimensions() -> None:
    tokens = ZoomTokenSet.from_percent(150)

    assert tokens.font_px == 18
    assert tokens.padding_px == 12
    assert tokens.icon_px == 24
    assert tokens.minimum_control_px == 120


def test_install_application_zoom_installs_event_filter(qapp: QApplication) -> None:
    settings = MagicMock()
    settings.value.return_value = 100

    controller = install_application_zoom(qapp, settings=settings)

    try:
        assert isinstance(controller, ApplicationZoomController)
        ctrl_wheel = _wheel_event(Qt.KeyboardModifier.ControlModifier)
        assert controller.eventFilter(None, ctrl_wheel)
    finally:
        controller.uninstall()


def test_theme_package_exports_zoom_and_responsive_helpers() -> None:
    from shared.python import theme

    assert theme.ApplicationZoomController is ApplicationZoomController
    assert theme.ZoomTokenSet is ZoomTokenSet
    assert callable(theme.set_text_minimum_width)

"""Regression tests for fleet application zoom helpers."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QFont, QKeyEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication

from src.shared.python.theme.zoom import (
    ApplicationZoomController,
    ZoomConfig,
    ZoomTokenSet,
    _coerce_percent,
    _point_size,
    install_application_zoom,
    scale_px,
)


class MemorySettings:
    """Small QSettings stand-in that records persisted values."""

    def __init__(self, initial: object | None = None) -> None:
        self.initial = initial
        self.values: dict[str, object] = {}

    def value(
        self,
        key: str,
        defaultValue: object | None = None,
        **kwargs: object,
    ) -> object:
        if key in self.values:
            return self.values[key]
        return self.initial if self.initial is not None else defaultValue

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        self.values[key] = value


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


@pytest.fixture(autouse=True)
def restore_app_font(qapp: QApplication) -> None:
    original = QFont(qapp.font())
    yield
    qapp.setFont(original)


def test_zoom_config_validates_bounds() -> None:
    assert ZoomConfig(minimum_percent=75, maximum_percent=150).default_percent == 100

    invalid_configs: list[tuple[str, dict[str, Any]]] = [
        ("minimum_percent", {"minimum_percent": 0}),
        ("maximum_percent", {"maximum_percent": 0}),
        ("step_percent", {"step_percent": 0}),
        (
            "minimum_percent must not exceed",
            {"minimum_percent": 200, "maximum_percent": 100},
        ),
        ("default_percent", {"minimum_percent": 75, "default_percent": 50}),
    ]
    for message, kwargs in invalid_configs:
        with pytest.raises(ValueError, match=message):
            ZoomConfig(**kwargs)


def test_scale_px_and_token_set_validate_scaling_contract() -> None:
    assert scale_px(12, 125) == 15
    assert scale_px(1, 50) == 1
    assert ZoomTokenSet.from_percent(150) == ZoomTokenSet(
        font_px=18,
        label_font_px=16,
        padding_px=12,
        spacing_px=9,
        icon_px=24,
        minimum_control_px=120,
    )

    with pytest.raises(ValueError, match="zoom_percent"):
        scale_px(12, 0)
    with pytest.raises(ValueError, match="value"):
        scale_px(0, 100)


def test_coerce_percent_accepts_int_and_string_defaults_other_values() -> None:
    assert _coerce_percent(125, 100) == 125
    assert _coerce_percent("150", 100) == 150
    assert _coerce_percent(None, 100) == 100


@pytest.mark.parametrize("stored", ["", "  ", "abc", "1.5", "120%", "0x7b"])
def test_coerce_percent_falls_back_when_stored_string_is_malformed(
    stored: str,
) -> None:
    """A corrupt persisted setting must not crash application start-up."""
    assert _coerce_percent(stored, 100) == 100


def test_controller_recovers_from_malformed_persisted_zoom(
    qapp: QApplication,
) -> None:
    settings = MemorySettings("not-a-number")

    controller = ApplicationZoomController(qapp, settings=settings)

    assert controller.zoom_percent == ZoomConfig().default_percent


def test_point_size_uses_float_size_when_available() -> None:
    font = QFont()
    font.setPointSizeF(11.5)

    assert _point_size(font) == pytest.approx(11.5)


def test_controller_loads_clamped_zoom_and_exposes_scaled_tokens(
    qapp: QApplication,
) -> None:
    qapp.setFont(QFont("Arial", 10))
    settings = MemorySettings("250")
    config = ZoomConfig(minimum_percent=75, maximum_percent=175)

    controller = ApplicationZoomController(qapp, config=config, settings=settings)

    assert controller.zoom_percent == 175
    assert controller.base_point_size == pytest.approx(10)
    assert controller.tokens.font_px == 21
    assert qapp.font().pointSizeF() == pytest.approx(17.5)


def test_controller_persists_zoom_and_emits_only_on_change(qapp: QApplication) -> None:
    qapp.setFont(QFont("Arial", 10))
    settings = MemorySettings()
    controller = ApplicationZoomController(qapp, settings=settings)
    observed: list[int] = []
    controller.zoomChanged.connect(observed.append)

    controller.set_zoom_percent(130)
    controller.set_zoom_percent(130)
    controller.set_zoom_percent(500)

    assert observed == [130, 200]
    assert settings.values == {"percent": 200}
    assert qapp.font().pointSizeF() == pytest.approx(20)


def test_controller_step_helpers_respect_bounds(qapp: QApplication) -> None:
    settings = MemorySettings(100)
    controller = ApplicationZoomController(
        qapp,
        config=ZoomConfig(minimum_percent=90, maximum_percent=110, step_percent=10),
        settings=settings,
    )

    controller.zoom_in()
    controller.zoom_in()
    assert controller.zoom_percent == 110

    controller.zoom_out()
    controller.zoom_out()
    controller.zoom_out()
    assert controller.zoom_percent == 90

    controller.reset_zoom()
    assert controller.zoom_percent == 100


def test_install_application_zoom_registers_controller(qapp: QApplication) -> None:
    controller = install_application_zoom(qapp, settings=MemorySettings())

    assert isinstance(controller, ApplicationZoomController)

    controller.uninstall()


def test_event_filter_handles_supported_key_shortcuts(qapp: QApplication) -> None:
    controller = ApplicationZoomController(qapp, settings=MemorySettings())

    assert controller.eventFilter(None, _key_event(Qt.Key.Key_Plus)) is True
    assert controller.zoom_percent == 110
    assert controller.eventFilter(None, _key_event(Qt.Key.Key_Minus)) is True
    assert controller.zoom_percent == 100
    assert controller.eventFilter(None, _key_event(Qt.Key.Key_0)) is True
    assert controller.zoom_percent == 100


def test_event_filter_ignores_non_zoom_keys_and_missing_control(
    qapp: QApplication,
) -> None:
    controller = ApplicationZoomController(qapp, settings=MemorySettings())

    assert controller.eventFilter(None, None) is False
    assert controller.eventFilter(None, _key_event(Qt.Key.Key_A)) is False
    assert (
        controller.eventFilter(
            None,
            _key_event(Qt.Key.Key_Plus, Qt.KeyboardModifier.NoModifier),
        )
        is False
    )
    assert controller.zoom_percent == 100


def test_event_filter_handles_control_wheel_zoom(qapp: QApplication) -> None:
    controller = ApplicationZoomController(qapp, settings=MemorySettings())

    assert controller.eventFilter(None, _wheel_event(120)) is True
    assert controller.zoom_percent == 110
    assert controller.eventFilter(None, _wheel_event(-120)) is True
    assert controller.zoom_percent == 100
    assert (
        controller.eventFilter(
            None,
            _wheel_event(120, Qt.KeyboardModifier.NoModifier),
        )
        is False
    )


def _key_event(
    key: Qt.Key,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.ControlModifier,
) -> QKeyEvent:
    return QKeyEvent(QKeyEvent.Type.KeyPress, int(key), modifiers)


def _wheel_event(
    delta_y: int,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.ControlModifier,
) -> QWheelEvent:
    return QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )

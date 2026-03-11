"""GUI control-panel contract tests.

These tests stay lightweight: they instantiate the widgets offscreen and
exercise input parsing without driving the full event loop.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app() -> Any:
    qtwidgets = pytest.importorskip("PyQt6.QtWidgets")
    QApplication = qtwidgets.QApplication

    qt_app = QApplication.instance()
    if qt_app is None:
        qt_app = QApplication([])
    return qt_app


class TestControlsWidgetValidation:
    def test_double_widget_rejects_negative_mass(self, app: Any) -> None:
        from double_pendulum_golf.gui.controls_widget import ControlsWidget

        widget = ControlsWidget()
        # UnitAwareInput.set_value expects a float; LabeledInput expects str
        try:
            widget.inp_m1.set_value(-1.0, is_si=True)
        except TypeError:
            widget.inp_m1.set_value("-1")
        with pytest.raises(ValueError, match="m1 must be positive"):
            widget.get_params()
        widget.deleteLater()

    def test_triple_widget_rejects_negative_length(self, app: Any) -> None:
        from double_pendulum_golf.gui.controls_widget_triple import ControlsWidgetTriple

        widget = ControlsWidgetTriple()
        widget.inp_L2.set_value("-0.5")
        with pytest.raises(ValueError, match="L2 must be positive"):
            widget.get_params()
        widget.deleteLater()

    def test_golfer_widget_rejects_invalid_grip_location(self, app: Any) -> None:
        from double_pendulum_golf.gui.controls_widget_golfer import ControlsWidgetGolfer

        widget = ControlsWidgetGolfer()
        widget.inp_grip_right.set_value("2.0")
        with pytest.raises(ValueError, match="grip_right must be ≤ L_club"):
            widget.get_params()
        widget.deleteLater()

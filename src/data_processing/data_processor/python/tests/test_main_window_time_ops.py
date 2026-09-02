"""Tests for TimeOpsMixin._calculate_trendline's X-range validation.

Regression coverage for issue #3979: an invalid X-min/X-max value used to be
silently swallowed (`except ValueError: pass`), and if only the SECOND field
failed to parse, the FIRST field's value had already been applied -- a
half-applied range with no error shown to the user. The fix validates each
field independently and aborts with a clear message before any range is
applied.

A minimal fake host is used instead of a real QWidget: TimeOpsMixin doesn't
inherit from QObject/QWidget (same segfault-avoidance reasoning as
CalculatorStateMixin), and the method only touches a handful of duck-typed
attributes, so a SimpleNamespace-based double is sufficient and avoids
needing a full main-window fixture.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pandas as pd
from data_processor.ui.pyqt6.main_window_time_ops import TimeOpsMixin


class _FixedText:
    """Duck-types a QLineEdit's `.text()` for a fixed value."""

    def __init__(self, value: str) -> None:
        self._value = value

    def text(self) -> str:
        return self._value


class _FixedCurrentText:
    """Duck-types a QComboBox's `.currentText()` for a fixed value."""

    def __init__(self, value: str) -> None:
        self._value = value

    def currentText(self) -> str:  # noqa: N802 (Qt method naming)
        return self._value


def _make_host(x_min_text: str, x_max_text: str) -> Any:
    host = SimpleNamespace()
    host.current_data = pd.DataFrame({"t": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    host.x_axis_combo = _FixedCurrentText("t")
    host.signal_list = SimpleNamespace(get_selected_signals=lambda: ["y"])
    host.trendline_type_combo = _FixedCurrentText("Linear")
    host.poly_degree_spin = SimpleNamespace(value=lambda: 1)
    host.trend_x_min_edit = _FixedText(x_min_text)
    host.trend_x_max_edit = _FixedText(x_max_text)
    host.trendline_results = SimpleNamespace(setText=lambda _text: None)
    host.status_bar = SimpleNamespace(set_status=lambda _status: None)
    return host


def test_invalid_x_min_aborts_with_no_partial_apply() -> None:
    host = _make_host("not-a-number", "10.0")
    with (
        patch("data_processor.core.signal_processing.calculate_trendline") as mock_calc,
        patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warning,
    ):
        TimeOpsMixin._calculate_trendline(host)
        mock_warning.assert_called_once()
        assert "not-a-number" in mock_warning.call_args[0][2]
        mock_calc.assert_not_called()


def test_invalid_x_max_aborts_and_does_not_apply_valid_x_min() -> None:
    """The half-applied-range case the issue describes: x_min parses fine,
    x_max does not -- the whole operation must abort, not silently proceed
    with only x_min applied."""
    host = _make_host("0.0", "not-a-number")
    with (
        patch("data_processor.core.signal_processing.calculate_trendline") as mock_calc,
        patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warning,
    ):
        TimeOpsMixin._calculate_trendline(host)
        mock_warning.assert_called_once()
        assert "not-a-number" in mock_warning.call_args[0][2]
        mock_calc.assert_not_called()


def test_valid_x_range_proceeds_to_calculation() -> None:
    host = _make_host("0.0", "10.0")
    with (
        patch("data_processor.core.signal_processing.calculate_trendline") as mock_calc,
        patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warning,
    ):
        mock_calc.return_value = {
            "equation": "y = x",
            "r_squared": 1.0,
            "slope": 1.0,
        }
        TimeOpsMixin._calculate_trendline(host)
        mock_warning.assert_not_called()
        mock_calc.assert_called_once()
        _, kwargs = mock_calc.call_args
        assert kwargs["x_min"] == 0.0
        assert kwargs["x_max"] == 10.0


def test_blank_x_range_fields_proceed_with_none() -> None:
    """Empty fields are not errors -- they mean 'no bound', unchanged from
    before this fix."""
    host = _make_host("", "")
    with (
        patch("data_processor.core.signal_processing.calculate_trendline") as mock_calc,
        patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warning,
    ):
        mock_calc.return_value = {
            "equation": "y = x",
            "r_squared": 1.0,
            "slope": 1.0,
        }
        TimeOpsMixin._calculate_trendline(host)
        mock_warning.assert_not_called()
        mock_calc.assert_called_once()
        _, kwargs = mock_calc.call_args
        assert kwargs["x_min"] is None
        assert kwargs["x_max"] is None

from typing import Any

"""Tests for __main__.py and gui.__init__.py in double_pendulum_golf.

This covers the application entry point, logging configuration, and
global event filters like Ctrl+Wheel UI zooming.
"""


import logging
import sys
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QEvent, Qt

from double_pendulum_golf import __main__
from double_pendulum_golf.gui import __getattr__ as gui_getattr

# ---------------------------------------------------------------------------
# gui.__init__.py test
# ---------------------------------------------------------------------------


def test_gui_init_getattr_main_window() -> Any:
    """Verify lazy-loading of MainWindow via __getattr__."""
    mw = gui_getattr("MainWindow")
    assert mw.__name__ == "MainWindow"


def test_gui_init_getattr_unknown() -> Any:
    """Verify AttributeError for unknown attributes."""
    with pytest.raises(AttributeError, match="has no attribute 'DoesNotExist'"):
        gui_getattr("DoesNotExist")


# ---------------------------------------------------------------------------
# __main__.py : _WheelBlockFilter
# ---------------------------------------------------------------------------


class TestWheelBlockFilter:
    def test_event_filter_not_wheel(self) -> Any:
        f = __main__._WheelBlockFilter()
        event = MagicMock()
        event.type.return_value = QEvent.Type.User
        assert f.eventFilter(None, event) is False

    @patch("double_pendulum_golf.__main__.QApplication.instance")
    def test_event_filter_ctrl_wheel(self, mock_instance) -> Any:
        # Because _WheelBlockFilter does isinstance(app, QApplication), we need a real or subclassed app.
        # However, mocking the instance might fail the isinstance check unless we patch QApplication itself.
        pass

    def test_event_filter_ctrl_wheel_actual_app(self, qapp) -> Any:
        # qapp is provided by pytest-qt, giving a real QApplication
        f = __main__._WheelBlockFilter()

        event = MagicMock()
        event.type.return_value = QEvent.Type.Wheel
        event.modifiers.return_value = Qt.KeyboardModifier.ControlModifier
        event.angleDelta().y.return_value = 120  # Zoom in

        # Capture old size
        old_size = qapp.font().pointSize()

        handled = f.eventFilter(None, event)

        assert handled is True
        event.accept.assert_called_once()
        new_size = qapp.font().pointSize()
        assert new_size == old_size + 1

    def test_event_filter_blocks_spinbox_wheel(self, qapp) -> Any:
        from PyQt6.QtWidgets import QSpinBox

        f = __main__._WheelBlockFilter()
        spin = QSpinBox()

        event = MagicMock()
        event.type.return_value = QEvent.Type.Wheel
        event.modifiers.return_value = Qt.KeyboardModifier.NoModifier

        handled = f.eventFilter(spin, event)
        assert handled is True
        event.ignore.assert_called_once()

    def test_reset_font(self, qapp) -> Any:
        f = __main__._WheelBlockFilter()
        f._default_font_pt = 12
        f.reset_font()
        assert qapp.font().pointSize() == 12


# ---------------------------------------------------------------------------
# __main__.py : Core execution flow
# ---------------------------------------------------------------------------


class TestMainFunctions:
    @patch("double_pendulum_golf.__main__.logging.basicConfig")
    def test_configure_logging(self, mock_basic) -> Any:
        __main__._configure_logging()
        mock_basic.assert_called_once()
        kwargs = mock_basic.call_args[1]
        assert kwargs["level"] == logging.INFO
        assert len(kwargs["handlers"]) == 2

    def test_main_version(self, capsys) -> Any:
        with patch.object(sys, "argv", ["prog", "--version"]):
            with pytest.raises(SystemExit) as exit_exc:
                __main__.main()
            assert exit_exc.value.code == 0

        out, _ = capsys.readouterr()
        assert "pendulum-simulator" in out

    @patch("double_pendulum_golf.__main__._WheelBlockFilter")
    @patch("double_pendulum_golf.__main__.QApplication")
    @patch("double_pendulum_golf.__main__.MainWindow.show")
    def test_main_execution(self, mock_show, mock_qapp, mock_wbf) -> Any:
        mock_exec = mock_qapp.return_value.exec
        mock_exec.return_value = 0

        with patch.object(sys, "argv", ["prog"]):
            with patch("double_pendulum_golf.__main__.Path.exists", return_value=True):
                with pytest.raises(SystemExit) as exit_exc:
                    __main__.main()
                assert exit_exc.value.code == 0

        mock_show.assert_called_once()
        mock_exec.assert_called_once()

    @patch("double_pendulum_golf.__main__._WheelBlockFilter")
    @patch("double_pendulum_golf.__main__.QApplication")
    @patch("double_pendulum_golf.__main__.MainWindow.show")
    def test_main_execution_no_icon(self, mock_show, mock_qapp, mock_wbf) -> Any:
        mock_exec = mock_qapp.return_value.exec
        mock_exec.return_value = 0
        with patch.object(sys, "argv", ["prog"]):
            with patch("double_pendulum_golf.__main__.Path.exists", return_value=False):
                with pytest.raises(SystemExit):
                    __main__.main()

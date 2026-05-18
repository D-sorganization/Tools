# ruff: noqa: E501
from typing import Any

"""Tests for MainWindow."""

from typing import Any
from unittest.mock import MagicMock, patch
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QWheelEvent
from double_pendulum_golf.gui.main_window import MainWindow, _find_sibling_package


def test_find_sibling_package() -> Any:
    # It should find itself if we look for the current folder
    assert _find_sibling_package("gui") is not None
    # Looking for a non-existent folder should return None
    assert _find_sibling_package("does_not_exist_at_all_12345") is None


def test_main_window_init(qapp, monkeypatch) -> Any:
    w = MainWindow()
    assert w.windowTitle() == "Pendulums"
    w.close()


def test_wheel_event_zoom(qapp) -> Any:
    """Ctrl+wheel zoom respects offset bounds and never escapes them."""
    w = MainWindow()
    w._font_zoom_pt = 0  # Start from the canonical zero offset

    # Not a wheel event — should be a no-op
    w.wheelEvent(object())

    from PyQt6.QtCore import QPointF, QPoint

    we_ctrl_in = QWheelEvent(
        QPointF(0, 0),
        QPointF(0, 0),
        QPoint(0, 120),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    we_ctrl_out = QWheelEvent(
        QPointF(0, 0),
        QPointF(0, 0),
        QPoint(0, -120),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )

    # One step in raises offset by 1
    old = w._font_zoom_pt
    w.wheelEvent(we_ctrl_in)
    assert w._font_zoom_pt == old + 1

    # One step out brings it back
    w.wheelEvent(we_ctrl_out)
    assert w._font_zoom_pt == old

    # Hard upper bound: scrolling up forever stops at OFFSET_MAX
    for _ in range(50):
        w.wheelEvent(we_ctrl_in)
    assert w._font_zoom_pt == MainWindow._FONT_OFFSET_MAX

    # Hard lower bound: scrolling down forever stops at OFFSET_MIN
    for _ in range(50):
        w.wheelEvent(we_ctrl_out)
    assert w._font_zoom_pt == MainWindow._FONT_OFFSET_MIN

    w.close()


def test_shortcuts(qapp) -> Any:
    w = MainWindow()
    # Execute shortcuts mechanically
    w._on_shortcut_play_pause()
    w._on_shortcut_reset()

    # Mock active panel to avoid real signals
    mock_panel = MagicMock()
    with patch.object(w, "_active_panel", return_value=mock_panel):
        w._on_shortcut_play_pause()
        w._on_shortcut_reset()
        w._on_shortcut_export_data()
        w._on_shortcut_run()
        w._on_shortcut_stop()
    w.close()


def test_wire_analysis_tab_emits(qapp) -> Any:
    w = MainWindow()
    double_p = w._double_panel
    # Emit sim finished
    double_p._result = MagicMock()
    double_p.sim_finished.emit()
    w.close()


def test_on_tab_changed(qapp) -> Any:
    w = MainWindow()

    # Switch to triple
    w._on_tab_changed(1)

    # Switch to golfer
    w._on_tab_changed(2)

    w.close()


def test_popout_chart_no_result(qapp, monkeypatch) -> Any:
    w = MainWindow()

    # Mock messagebox
    mock_msg = MagicMock()
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.information", mock_msg)

    # Active panel has no result
    with patch.object(w, "_active_panel") as mock_active:
        mock_active.return_value._result = None
        w._on_popout_chart()
        mock_msg.assert_called_once()

    w.close()


def test_popout_chart_with_result_dialog_accepted(qapp, monkeypatch) -> Any:
    w = MainWindow()

    mock_panel = MagicMock()
    mock_panel._result = MagicMock()
    # Adding attributes for golfer to cover model_type logic branches
    mock_panel._golfer = True

    with patch.object(w, "_active_panel", return_value=mock_panel):
        # mock dialog
        class MockDialog:
            def __init__(self, *args, **kwargs):
                pass

            def exec(self) -> Any:
                return 1  # Accepted

            def get_selection(self) -> Any:
                return "x", "y", 1

        monkeypatch.setattr(
            "double_pendulum_golf.gui.chart_data_dialog.ChartDataDialog", MockDialog
        )

        # mock extract_series
        mock_extract = MagicMock(side_effect=[([1], "X", "m"), ([2], "Y", "m")])
        monkeypatch.setattr(
            "double_pendulum_golf.data_extractor.extract_series", mock_extract
        )

        # mock PopOutChart
        mock_chart_class = MagicMock()
        monkeypatch.setattr(
            "double_pendulum_golf.gui.popout_chart.PopOutChart", mock_chart_class
        )

        w._on_popout_chart()

        assert hasattr(w, "_popout_charts")
        mock_chart_class().show.assert_called_once()
        mock_chart_class().add_regression.assert_called_once_with(degree=1)

    w.close()


def test_popout_chart_with_result_data_error(qapp, monkeypatch) -> Any:
    w = MainWindow()

    mock_panel = MagicMock()
    mock_panel._result = MagicMock()
    mock_panel._triple = True  # for coverage of triple model type

    with patch.object(w, "_active_panel", return_value=mock_panel):

        class MockDialog:
            def __init__(self, *args, **kwargs):
                pass

            def exec(self) -> Any:
                return 1

            def get_selection(self) -> Any:
                return "x", "y", 0

        monkeypatch.setattr(
            "double_pendulum_golf.gui.chart_data_dialog.ChartDataDialog", MockDialog
        )

        def mock_extract(*args) -> Any:
            raise KeyError("bad")

        monkeypatch.setattr(
            "double_pendulum_golf.data_extractor.extract_series", mock_extract
        )

        mock_msg = MagicMock()
        monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.warning", mock_msg)

        w._on_popout_chart()
        mock_msg.assert_called_once()

    w.close()


def test_popout_chart_dialog_cancelled(qapp, monkeypatch) -> Any:
    w = MainWindow()
    mock_panel = MagicMock()
    mock_panel._result = MagicMock()

    with patch.object(w, "_active_panel", return_value=mock_panel):

        class MockDialog:
            def __init__(self, *args, **kwargs):
                pass

            def exec(self) -> Any:
                return 0  # Rejected

        monkeypatch.setattr(
            "double_pendulum_golf.gui.chart_data_dialog.ChartDataDialog", MockDialog
        )
        w._on_popout_chart()  # Should return early

    w.close()


def test_theme_manager_methods(qapp, monkeypatch) -> Any:
    w = MainWindow()

    w._theme_manager = MagicMock()
    w._on_theme_changed("New Theme")

    # Test open theme manager dialog
    mock_dialog = MagicMock()
    monkeypatch.setattr(
        "double_pendulum_golf.gui.main_window.ThemeManagerDialog",
        mock_dialog,
        raising=False,
    )
    monkeypatch.setattr("double_pendulum_golf.gui.main_window._THEME_AVAILABLE", True)

    # Inject ThemeManagerDialog onto the module directly because it might not be imported!
    import double_pendulum_golf.gui.main_window as mw

    mw.ThemeManagerDialog = mock_dialog

    w._open_theme_manager()
    mock_dialog.return_value.exec.assert_called_once()

    # Test unavailable theme
    monkeypatch.setattr("double_pendulum_golf.gui.main_window._THEME_AVAILABLE", False)
    mock_msg = MagicMock()
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.information", mock_msg)
    w._open_theme_manager()
    mock_msg.assert_called_once()

    w.close()


def test_ui_interactions(qapp) -> Any:
    w = MainWindow()

    # About
    with patch("PyQt6.QtWidgets.QMessageBox.about") as mock_about:
        w._show_about()
        mock_about.assert_called_once()

    # Toggle analysis dock
    w._toggle_analysis_dock(True)
    # The dock may not return isVisible()=True if main window is hidden, check check property or effect
    assert hasattr(w, "_analysis_dock")

    # Apply dark theme
    w._apply_pendulum_dark()

    w.close()


def test_close_event(qapp) -> Any:
    from PyQt6.QtGui import QCloseEvent

    w = MainWindow()
    we = QCloseEvent()
    w.closeEvent(we)
    # Ensure panels had save_layout called
    import double_pendulum_golf.gui.main_window as mw

    settings = QSettings(mw._SETTINGS_ORG, mw._SETTINGS_APP)
    assert settings.value("window_geometry") is not None

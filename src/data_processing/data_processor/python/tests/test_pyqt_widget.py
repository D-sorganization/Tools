"""Tests for the PyQt6 DataProcessorWidget.

The widget loads and processes asynchronously: user actions spawn ``QThread``
workers and results come back via queued signals. Tests must therefore wait
for the observable outcome AND for the worker to be reaped before asserting.
Asserting immediately races the worker, and letting a worker outlive its test
aborts the interpreter when the ``QThread`` wrapper is garbage-collected --
under xdist that is the "node down: Not properly terminated" crash, and in a
serial run it hangs the session at exit, after the last test, where no
pytest-level timeout is armed.

``QApplication`` is supplied by pytest-qt via ``qtbot``; creating one at
import time put global Qt state into every process that so much as collected
this module.
"""

from unittest.mock import patch

import pandas as pd
import pytest
from data_processor.pyqt_widget import DataProcessorWidget
from PyQt6.QtCore import Qt


@pytest.fixture
def widget(qtbot):
    """Fixture to create the widget with mocked backend."""
    # Patch the backend classes before instantiating the widget
    with (
        patch("data_processor.pyqt_widget.DataLoader") as MockDataLoader,
        patch("data_processor.pyqt_widget.SignalProcessor") as MockSignalProcessor,
    ):
        mock_loader = MockDataLoader.return_value
        mock_processor = MockSignalProcessor.return_value

        # Setup default mock behaviors
        mock_loader.get_numeric_signals.return_value = ["A", "B"]

        widget = DataProcessorWidget()
        qtbot.addWidget(widget)

        # Attach mocks to widget for verification in tests
        widget.mock_loader = mock_loader
        widget.mock_processor = mock_processor

        yield widget


def test_widget_initialization(widget):
    """Test that the widget initializes with correct UI elements."""
    assert widget.load_btn is not None
    assert widget.signal_list is not None
    assert widget.filter_combo is not None
    assert widget.process_btn is not None
    assert widget.result_table is not None


def test_load_file_cancel(widget):
    """Test loading file cancellation."""
    with patch("PyQt6.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")):
        widget.load_file()
        assert widget.current_df is None
        assert widget.file_label.text() == "No file loaded"


def test_load_file_success(widget, qtbot, tmp_path):
    """Test successful file loading."""
    # Create a dummy DataFrame to return
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    widget.mock_loader.load_csv_file.return_value = df
    widget.mock_loader.get_numeric_signals.return_value = ["A", "B"]

    csv_path = str(tmp_path / "test.csv")

    with patch(
        "PyQt6.QtWidgets.QFileDialog.getOpenFileName",
        return_value=(csv_path, "CSV Files (*.csv)"),
    ):
        widget.load_file()

        # Loading happens on a DataLoadWorker thread; wait for the result to
        # land and for the widget to reap the worker before asserting.
        qtbot.waitUntil(lambda: widget.current_df is df, timeout=5000)
        qtbot.waitUntil(lambda: widget._load_worker is None, timeout=5000)

    # Verify loader was called
    widget.mock_loader.load_csv_file.assert_called_with(csv_path)

    # Verify UI updates
    assert widget.file_label.text() == "test.csv"
    assert widget.signal_list.count() == 2
    assert widget.result_table.rowCount() == 3


def test_process_no_file(widget):
    """Test processing without loading file."""
    with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warning:
        widget.process_data()
        mock_warning.assert_called_once()
        assert "load a file" in mock_warning.call_args[0][2]


def test_process_no_selection(widget, tmp_path):
    """Test processing without selecting signals."""
    # Simulate loaded file
    widget.current_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    widget.mock_loader.get_numeric_signals.return_value = ["A", "B"]
    widget._populate_signals()  # Helper to populate list

    with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warning:
        widget.process_data()
        mock_warning.assert_called_once()
        assert "select at least one signal" in mock_warning.call_args[0][2]


def test_process_success(widget, qtbot):
    """Test successful processing."""
    # Simulate loaded file
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    widget.current_df = df
    widget.mock_loader.get_numeric_signals.return_value = ["A", "B"]
    widget._populate_signals()

    # Mock filtered result
    filtered_df = pd.DataFrame({"A": [1.1, 2.1, 3.1]})
    widget.mock_processor.apply_filter.return_value = filtered_df

    # Select signal 'A'
    item = widget.signal_list.findItems("A", Qt.MatchFlag.MatchExactly)[0]
    item.setSelected(True)

    # Select filter
    widget.filter_combo.setCurrentText("Moving Average")

    # The success dialog fires from a queued signal after the worker
    # finishes, so the patch must stay active until the worker is reaped --
    # otherwise the real modal QMessageBox opens during teardown and blocks
    # the session with nobody to dismiss it.
    with patch("PyQt6.QtWidgets.QMessageBox.information") as mock_info:
        widget.process_data()

        qtbot.waitUntil(lambda: widget.processed_df is filtered_df, timeout=5000)
        qtbot.waitUntil(lambda: widget._process_worker is None, timeout=5000)

        # Verify processor called
        widget.mock_processor.apply_filter.assert_called_once()

        # Verify success message
        mock_info.assert_called_once()

    # Verify UI updates
    assert "A" in widget.processed_df.columns
    assert widget.result_table.rowCount() == 3
    # Check that table shows filtered values
    assert widget.result_table.item(0, 0).text() == "1.1"

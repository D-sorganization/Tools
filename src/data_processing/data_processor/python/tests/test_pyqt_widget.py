from typing import Any

"""Tests for the PyQt6 DataProcessorWidget."""

import sys
from unittest.mock import patch

import pandas as pd
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

# Ensure QApplication exists
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from data_processor.pyqt_widget import DataProcessorWidget  # noqa: E402


@pytest.fixture
def widget(qtbot) -> Any:
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


def test_widget_initialization(widget) -> Any:
    """Test that the widget initializes with correct UI elements."""
    assert widget.load_btn is not None
    assert widget.signal_list is not None
    assert widget.filter_combo is not None
    assert widget.process_btn is not None
    assert widget.result_table is not None


def test_load_file_cancel(widget) -> Any:
    """Test loading file cancellation."""
    with patch("PyQt6.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")):
        widget.load_file()
        assert widget.current_df is None
        assert widget.file_label.text() == "No file loaded"


def test_load_file_success(widget, tmp_path) -> Any:
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

        # Verify loader was called
        widget.mock_loader.load_csv_file.assert_called_with(csv_path)

        # Verify UI updates
        assert widget.current_df is df
        assert widget.file_label.text() == "test.csv"
        assert widget.signal_list.count() == 2
        assert widget.result_table.rowCount() == 3


def test_process_no_file(widget) -> Any:
    """Test processing without loading file."""
    with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warning:
        widget.process_data()
        mock_warning.assert_called_once()
        assert "load a file" in mock_warning.call_args[0][2]


def test_process_no_selection(widget, tmp_path) -> Any:
    """Test processing without selecting signals."""
    # Simulate loaded file
    widget.current_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    widget.mock_loader.get_numeric_signals.return_value = ["A", "B"]
    widget._populate_signals()  # Helper to populate list

    with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warning:
        widget.process_data()
        mock_warning.assert_called_once()
        assert "select at least one signal" in mock_warning.call_args[0][2]


def test_process_success(widget) -> Any:
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

    # Mock MessageBox to avoid blocking
    with patch("PyQt6.QtWidgets.QMessageBox.information") as mock_info:
        widget.process_data()

        # Verify processor called
        widget.mock_processor.apply_filter.assert_called_once()

        # Verify success message
        mock_info.assert_called_once()

        # Verify UI updates
        assert widget.processed_df is filtered_df
        assert "A" in widget.processed_df.columns
        assert widget.result_table.rowCount() == 3
        # Check that table shows filtered values
        assert widget.result_table.item(0, 0).text() == "1.1"

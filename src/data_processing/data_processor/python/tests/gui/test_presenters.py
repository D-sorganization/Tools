"""Tests for PyQt6 GUI presenters (TDD - RED phase)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication

# Ensure QApplication exists
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


class TestDataPresenter:
    """Tests for the DataPresenter."""

    @pytest.fixture
    def presenter(self):
        """Create DataPresenter with mocked dependencies."""
        from data_processor.gui.presenters.data_presenter import DataPresenter

        presenter = DataPresenter()
        presenter.data_loader = MagicMock()
        return presenter

    def test_load_files_calls_loader(self, presenter) -> None:
        """Loading files calls the data loader."""
        presenter.data_loader.load_csv_file.return_value = pd.DataFrame({"a": [1, 2]})
        presenter.load_files(["/path/to/file.csv"])
        presenter.data_loader.load_csv_file.assert_called_once()

    def test_load_files_emits_data_loaded(self, presenter, qtbot) -> None:
        """Loading files emits data_loaded signal."""
        presenter.data_loader.load_csv_file.return_value = pd.DataFrame({"a": [1, 2]})
        with qtbot.waitSignal(presenter.data_loaded, timeout=1000):
            presenter.load_files(["/path/to/file.csv"])

    def test_get_signals_returns_numeric_columns(self, presenter) -> None:
        """get_signals returns list of numeric column names."""
        presenter.current_data = pd.DataFrame({
            "numeric_a": [1.0, 2.0],
            "numeric_b": [3.0, 4.0],
            "text_c": ["a", "b"],
        })
        signals = presenter.get_signals()
        assert "numeric_a" in signals
        assert "numeric_b" in signals
        assert "text_c" not in signals

    def test_get_data_returns_dataframe(self, presenter) -> None:
        """get_data returns the current dataframe."""
        df = pd.DataFrame({"a": [1, 2]})
        presenter.current_data = df
        assert presenter.get_data() is df


class TestFilterPresenter:
    """Tests for the FilterPresenter."""

    @pytest.fixture
    def presenter(self):
        """Create FilterPresenter with mocked dependencies."""
        from data_processor.gui.presenters.filter_presenter import FilterPresenter

        presenter = FilterPresenter()
        presenter.signal_processor = MagicMock()
        return presenter

    def test_apply_filter_calls_processor(self, presenter) -> None:
        """Applying filter calls the signal processor."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        presenter.signal_processor.apply_filter.return_value = df

        config = {"filter_type": "Moving Average", "parameters": {"ma_window": 5}}
        presenter.apply_filter(df, ["a"], config)

        presenter.signal_processor.apply_filter.assert_called_once()

    def test_apply_filter_emits_filter_applied(self, presenter, qtbot) -> None:
        """Applying filter emits filter_applied signal."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        presenter.signal_processor.apply_filter.return_value = df

        config = {"filter_type": "Moving Average", "parameters": {"ma_window": 5}}

        with qtbot.waitSignal(presenter.filter_applied, timeout=1000):
            presenter.apply_filter(df, ["a"], config)

    def test_get_filter_types_returns_list(self, presenter) -> None:
        """get_filter_types returns list of available filters."""
        filter_types = presenter.get_filter_types()
        assert isinstance(filter_types, list)
        assert len(filter_types) > 0
        assert "Moving Average" in filter_types


class TestExportPresenter:
    """Tests for the ExportPresenter."""

    @pytest.fixture
    def presenter(self):
        """Create ExportPresenter with mocked dependencies."""
        from data_processor.gui.presenters.export_presenter import ExportPresenter

        presenter = ExportPresenter()
        presenter.data_loader = MagicMock()
        return presenter

    def test_export_calls_save_dataframe(self, presenter, tmp_path) -> None:
        """Exporting calls save_dataframe on loader."""
        df = pd.DataFrame({"a": [1, 2]})
        output_path = str(tmp_path / "output.csv")
        presenter.data_loader.save_dataframe.return_value = True

        presenter.export_data(df, output_path, "csv")

        presenter.data_loader.save_dataframe.assert_called_once()

    def test_export_emits_export_completed(self, presenter, qtbot, tmp_path) -> None:
        """Exporting emits export_completed signal."""
        df = pd.DataFrame({"a": [1, 2]})
        output_path = str(tmp_path / "output.csv")
        presenter.data_loader.save_dataframe.return_value = True

        with qtbot.waitSignal(presenter.export_completed, timeout=1000):
            presenter.export_data(df, output_path, "csv")

    def test_get_export_formats_returns_list(self, presenter) -> None:
        """get_export_formats returns list of available formats."""
        formats = presenter.get_export_formats()
        assert isinstance(formats, list)
        assert "csv" in formats
        assert "parquet" in formats

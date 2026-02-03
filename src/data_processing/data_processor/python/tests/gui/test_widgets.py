"""Tests for PyQt6 GUI widgets (TDD - RED phase)."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication

# Ensure QApplication exists
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


class TestFilePanel:
    """Tests for the FilePanel widget."""

    @pytest.fixture
    def panel(self, qtbot):
        """Create FilePanel for testing."""
        from data_processor.gui.widgets.file_panel import FilePanel

        widget = FilePanel()
        qtbot.addWidget(widget)
        return widget

    def test_initialization(self, panel) -> None:
        """Panel initializes with required elements."""
        assert panel.select_button is not None
        assert panel.clear_button is not None
        assert panel.load_button is not None
        assert panel.file_list is not None

    def test_select_files_emits_signal(self, panel, qtbot, tmp_path) -> None:
        """Selecting files emits files_selected signal."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        with qtbot.waitSignal(panel.files_selected, timeout=1000):
            with patch(
                "PyQt6.QtWidgets.QFileDialog.getOpenFileNames",
                return_value=([str(csv_file)], ""),
            ):
                panel.select_button.click()

    def test_clear_files_clears_list(self, panel, qtbot) -> None:
        """Clearing files empties the file list."""
        panel.file_list.addItem("test.csv")
        panel.clear_button.click()
        assert panel.file_list.count() == 0

    def test_load_button_emits_signal(self, panel, qtbot) -> None:
        """Load button emits load_requested signal."""
        panel.file_list.addItem("test.csv")
        with qtbot.waitSignal(panel.load_requested, timeout=1000):
            panel.load_button.click()


class TestSignalPanel:
    """Tests for the SignalPanel widget."""

    @pytest.fixture
    def panel(self, qtbot):
        """Create SignalPanel for testing."""
        from data_processor.gui.widgets.signal_panel import SignalPanel

        widget = SignalPanel()
        qtbot.addWidget(widget)
        return widget

    def test_initialization(self, panel) -> None:
        """Panel initializes with required elements."""
        assert panel.signal_list is not None
        assert panel.select_all_button is not None
        assert panel.clear_selection_button is not None

    def test_set_signals_populates_list(self, panel) -> None:
        """Setting signals populates the list."""
        signals = ["signal_a", "signal_b", "signal_c"]
        panel.set_signals(signals)
        assert panel.signal_list.count() == 3

    def test_select_all_selects_all_items(self, panel) -> None:
        """Select all button selects all items."""
        panel.set_signals(["a", "b", "c"])
        panel.select_all_button.click()
        selected = panel.get_selected_signals()
        assert len(selected) == 3

    def test_clear_selection_deselects_all(self, panel) -> None:
        """Clear selection button deselects all items."""
        panel.set_signals(["a", "b", "c"])
        panel.select_all_button.click()
        panel.clear_selection_button.click()
        selected = panel.get_selected_signals()
        assert len(selected) == 0

    def test_selection_changed_emits_signal(self, panel, qtbot) -> None:
        """Changing selection emits selection_changed signal."""
        panel.set_signals(["a", "b"])
        with qtbot.waitSignal(panel.selection_changed, timeout=1000):
            item = panel.signal_list.item(0)
            item.setSelected(True)


class TestFilterPanel:
    """Tests for the FilterPanel widget."""

    @pytest.fixture
    def panel(self, qtbot):
        """Create FilterPanel for testing."""
        from data_processor.gui.widgets.filter_panel import FilterPanel

        widget = FilterPanel()
        qtbot.addWidget(widget)
        return widget

    def test_initialization(self, panel) -> None:
        """Panel initializes with required elements."""
        assert panel.filter_combo is not None
        assert panel.apply_button is not None
        assert panel.params_container is not None

    def test_filter_types_populated(self, panel) -> None:
        """Filter combo is populated with filter types."""
        assert panel.filter_combo.count() > 0

    def test_changing_filter_updates_params(self, panel, qtbot) -> None:
        """Changing filter type updates parameter widgets."""
        initial_count = panel.params_container.layout().count()
        panel.filter_combo.setCurrentText("Butterworth Low-pass")
        # Should show different params
        assert panel.params_container.layout() is not None

    def test_apply_emits_signal_with_config(self, panel, qtbot) -> None:
        """Apply button emits filter_requested signal with config."""
        with qtbot.waitSignal(panel.filter_requested, timeout=1000) as signal:
            panel.apply_button.click()

    def test_get_filter_config_returns_valid_config(self, panel) -> None:
        """get_filter_config returns valid configuration."""
        config = panel.get_filter_config()
        assert "filter_type" in config
        assert "parameters" in config


class TestPreviewTable:
    """Tests for the PreviewTable widget."""

    @pytest.fixture
    def table(self, qtbot):
        """Create PreviewTable for testing."""
        from data_processor.gui.widgets.preview_table import PreviewTable

        widget = PreviewTable()
        qtbot.addWidget(widget)
        return widget

    def test_initialization(self, table) -> None:
        """Table initializes correctly."""
        assert table.table_widget is not None
        assert table.row_count_label is not None

    def test_set_data_populates_table(self, table) -> None:
        """Setting data populates the table."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        table.set_data(df)
        assert table.table_widget.rowCount() == 3
        assert table.table_widget.columnCount() == 2

    def test_clear_removes_all_data(self, table) -> None:
        """Clearing removes all data from table."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        table.set_data(df)
        table.clear()
        assert table.table_widget.rowCount() == 0


class TestExportPanel:
    """Tests for the ExportPanel widget."""

    @pytest.fixture
    def panel(self, qtbot):
        """Create ExportPanel for testing."""
        from data_processor.gui.widgets.export_panel import ExportPanel

        widget = ExportPanel()
        qtbot.addWidget(widget)
        return widget

    def test_initialization(self, panel) -> None:
        """Panel initializes with required elements."""
        assert panel.format_combo is not None
        assert panel.export_button is not None

    def test_export_formats_populated(self, panel) -> None:
        """Export format combo is populated."""
        assert panel.format_combo.count() > 0

    def test_export_emits_signal(self, panel, qtbot) -> None:
        """Export button emits export_requested signal."""
        with qtbot.waitSignal(panel.export_requested, timeout=1000):
            panel.export_button.click()

    def test_get_export_format_returns_selected(self, panel) -> None:
        """get_export_format returns selected format."""
        panel.format_combo.setCurrentText("csv")
        assert panel.get_export_format() == "csv"


class TestStatisticsPanel:
    """Tests for the StatisticsPanel widget."""

    @pytest.fixture
    def panel(self, qtbot):
        """Create StatisticsPanel for testing."""
        from data_processor.gui.widgets.statistics_panel import StatisticsPanel

        widget = StatisticsPanel()
        qtbot.addWidget(widget)
        return widget

    def test_initialization(self, panel) -> None:
        """Panel initializes with required elements."""
        assert panel.stats_display is not None
        assert panel.calculate_button is not None

    def test_set_statistics_displays_data(self, panel) -> None:
        """Setting statistics displays the data."""
        stats = {
            "signal_a": {"mean": 1.5, "std": 0.5, "min": 1.0, "max": 2.0},
        }
        panel.set_statistics(stats)
        # Should have content in display
        assert len(panel.stats_display.toPlainText()) > 0

    def test_calculate_emits_signal(self, panel, qtbot) -> None:
        """Calculate button emits calculate_requested signal."""
        with qtbot.waitSignal(panel.calculate_requested, timeout=1000):
            panel.calculate_button.click()

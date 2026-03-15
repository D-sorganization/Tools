"""Tests for psa_gui.py."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication
from upstream_drift_tools.process_calculators.psa_package.psa_gui import (
    InputPanel,
    PSAMainWindow,
    ResultsPanel,
    SensitivityPlotWidget,
)


@pytest.fixture
def dummy_qapp():
    """Create a dummy QApplication for UI testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_input_panel_initialization(dummy_qapp):
    """Test that the input panel initializes correctly."""
    panel = InputPanel()
    assert panel.feed_input.text() == "1100"

    # Check default values
    feed, s2, prod, components = panel.get_parameters()
    assert feed == 1100.0
    assert s2 == 1.0
    assert prod == 0.0
    assert len(components) == 7

    # Test reset defaults
    panel.feed_input.setText("500")
    panel._reset_defaults()
    assert panel.feed_input.text() == "1100"


def test_results_panel_initialization(dummy_qapp):
    """Test that the results panel initializes correctly."""
    panel = ResultsPanel()
    assert panel.h2_recovery_label.text() == "--"

    # Mock PSAResults
    mock_results = MagicMock()
    mock_results.h2_recovery_pct = 85.5
    mock_results.h2_purity_pct = 99.99
    mock_results.total_net_product_scfm = 500.0
    mock_results.total_exhaust_scfm = 200.0
    mock_results.mass_balance_error = 0.0001
    mock_results.s2_tail_h2_pct = 5.0
    mock_results.s2_tail_o2_pct = 1.0
    mock_results.component_names = ["H2", "CH4"]

    # Mock flows
    mock_flows = MagicMock()
    mock_flows.fresh_feed = [100.0, 50.0]
    mock_flows.mixed_feed = [110.0, 55.0]
    mock_flows.exhaust = [10.0, 5.0]
    mock_flows.interstage = [90.0, 45.0]
    mock_flows.s2_tail = [5.0, 2.5]
    mock_flows.s2_tail_recycle = [5.0, 2.5]
    mock_flows.gross_product = [85.0, 42.5]
    mock_flows.net_product = [80.0, 40.0]
    mock_results.flows = mock_flows
    mock_results.total_feed_scfm = 150.0

    # Mock compositions
    mock_comps = MagicMock()
    mock_comps.fresh_feed = [66.6, 33.3]
    mock_comps.mixed_feed = [66.6, 33.3]
    mock_comps.exhaust = [66.6, 33.3]
    mock_comps.interstage = [66.6, 33.3]
    mock_comps.s2_tail = [66.6, 33.3]
    mock_comps.net_product = [66.6, 33.3]
    mock_results.compositions = mock_comps

    panel.update_results(mock_results)
    assert panel.h2_recovery_label.text() == "85.50%"


@patch(
    "upstream_drift_tools.process_calculators.psa_package.psa_gui.calculate_sensitivity"
)
def test_sensitivity_plot_widget(mock_calc, dummy_qapp):
    """Test the sensitivity plot widget."""
    import numpy as np

    mock_sensitivity = {
        "h2_recovery": np.ones((51, 3)) * np.array([70.0, 75.0, 80.0]),
        "net_product": np.ones((51, 3)) * np.array([400.0, 450.0, 500.0]),
    }
    mock_calc.return_value = mock_sensitivity

    widget = SensitivityPlotWidget()

    # Update plot
    widget.plot_type_combo.setCurrentText("H2 Recovery vs Recycle")
    widget._update_plot()
    mock_calc.assert_called()


@patch("PyQt6.QtWidgets.QMainWindow.show")
def test_psa_main_window_initialization(mock_show, dummy_qapp):
    """Test PSA main window initialization."""
    with patch(
        "upstream_drift_tools.process_calculators.psa_package.psa_gui.PSAModel"
    ) as mock_model:
        mock_results = MagicMock()
        mock_results.h2_recovery_pct = 85.5
        mock_results.h2_purity_pct = 99.99
        mock_results.total_net_product_scfm = 500.0
        mock_results.total_exhaust_scfm = 200.0
        mock_results.mass_balance_error = 0.0001
        mock_results.s2_tail_h2_pct = 5.0
        mock_results.s2_tail_o2_pct = 1.0
        mock_results.component_names = ["H2", "CH4"]

        mock_flows = MagicMock()
        mock_flows.fresh_feed = [100.0, 50.0]
        mock_flows.mixed_feed = [110.0, 55.0]
        mock_flows.exhaust = [10.0, 5.0]
        mock_flows.interstage = [90.0, 45.0]
        mock_flows.s2_tail = [5.0, 2.5]
        mock_flows.s2_tail_recycle = [5.0, 2.5]
        mock_flows.gross_product = [85.0, 42.5]
        mock_flows.net_product = [80.0, 40.0]
        mock_results.flows = mock_flows
        mock_results.total_feed_scfm = 150.0

        # Mock compositions
        mock_comps = MagicMock()
        mock_comps.fresh_feed = [66.6, 33.3]
        mock_comps.mixed_feed = [66.6, 33.3]
        mock_comps.exhaust = [66.6, 33.3]
        mock_comps.interstage = [66.6, 33.3]
        mock_comps.s2_tail = [66.6, 33.3]
        mock_comps.net_product = [66.6, 33.3]
        mock_results.compositions = mock_comps

        mock_instance = MagicMock()
        mock_instance.calculate.return_value = mock_results
        mock_model.return_value = mock_instance
        window = PSAMainWindow()
        assert window.windowTitle() == "Two-Stage PSA System Analysis"

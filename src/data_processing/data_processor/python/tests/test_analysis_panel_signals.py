"""Tests for AnalysisPanel aggregate signal forwarding.

These guard the orthogonality contract: consumers connect to the panel's own
signals, and the panel forwards them from its internal child widgets. Renaming
or restructuring a child widget must not require touching the consumer.
"""

import sys

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from data_processor.ui.pyqt6.analysis_widgets import AnalysisPanel  # noqa: E402


@pytest.fixture
def panel(qtbot):
    widget = AnalysisPanel()
    qtbot.addWidget(widget)
    return widget


def test_panel_exposes_aggregate_signals(panel):
    """The panel exposes a request signal for each analysis type."""
    for name in (
        "pca_requested",
        "anova_requested",
        "regression_requested",
        "surface_requested",
        "nn_train_requested",
    ):
        assert hasattr(panel, name), f"AnalysisPanel missing signal {name}"


@pytest.mark.parametrize(
    ("child_attr", "child_signal", "panel_signal"),
    [
        ("pca_widget", "analysis_requested", "pca_requested"),
        ("anova_widget", "analysis_requested", "anova_requested"),
        ("regression_widget", "analysis_requested", "regression_requested"),
        ("surface_widget", "plot_requested", "surface_requested"),
        ("nn_widget", "train_requested", "nn_train_requested"),
    ],
)
def test_child_signal_forwards_to_panel_signal(
    panel, child_attr, child_signal, panel_signal
):
    """Emitting a child signal re-emits the matching panel-level signal."""
    received: list[dict] = []
    getattr(panel, panel_signal).connect(received.append)

    payload = {"marker": panel_signal}
    getattr(getattr(panel, child_attr), child_signal).emit(payload)

    assert received == [payload]

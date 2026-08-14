"""Desktop presentation tests for the Launch Monitor Analytics tab."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.ui.pyqt6.launch_monitor_analytics_tab import (  # noqa: E402
    LaunchMonitorAnalyticsTab,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_demo_analysis_populates_results_and_traceability(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)

    result = tab.run_analysis()

    assert result.dataset.row_count == 120
    assert tab.result_table.rowCount() >= 3
    assert result.dataset.fingerprint_sha256 in tab.details.toPlainText()
    assert tab.export_result_button.isEnabled()


def test_every_interactive_control_has_accessible_help(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    controls = (
        tab.import_button,
        tab.demo_button,
        tab.export_data_button,
        tab.export_result_button,
        tab.convention_combo,
        tab.outcome_combo,
        tab.predictor_list,
        tab.mode_combo,
        tab.method_combo,
        tab.missing_combo,
        tab.group_combo,
        tab.confidence_spin,
        tab.min_samples_spin,
        tab.run_button,
    )
    assert all(control.accessibleName() for control in controls)
    assert all(control.toolTip() for control in controls)

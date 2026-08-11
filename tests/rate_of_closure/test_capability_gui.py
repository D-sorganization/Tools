"""PyQt capability optimizer workflow and diagnostics."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.application.capability_workflow import (  # noqa: E402
    CapabilityWorkflowInputs,
    build_capability_workflow,
)
from rate_of_closure.ui.pyqt6.capability_controls import (
    CapabilityControls,  # noqa: E402
)
from rate_of_closure.ui.pyqt6.capability_tab import (  # noqa: E402
    CapabilityOptimizationTab,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_capability_controls_round_trip_integration_settings(qtbot) -> None:  # type: ignore[no-untyped-def]
    controls = CapabilityControls()
    qtbot.addWidget(controls)
    source = CapabilityWorkflowInputs(max_time_s=7.0, trajectory_sample_interval_s=0.02)

    controls.set_inputs(source)

    assert controls.inputs() == source
    assert controls._numeric["max_time_s"].isVisibleTo(controls)
    assert controls._numeric["trajectory_sample_interval_s"].isVisibleTo(controls)


def test_capability_tab_runs_and_exposes_complete_diagnostics(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = CapabilityOptimizationTab()
    qtbot.addWidget(tab)
    tab.controls.set_inputs(
        CapabilityWorkflowInputs(
            candidate_budget=1, ensemble_size=2, alternatives_count=1
        )
    )

    tab.run()
    qtbot.waitUntil(lambda: tab.status.text().startswith("Completed"), timeout=20_000)

    assert tab.results.isVisibleTo(tab)
    assert tab.results.alternatives.rowCount() == 1
    headers = [
        tab.results.alternatives.horizontalHeaderItem(index).text()
        for index in range(tab.results.alternatives.columnCount())
    ]
    expected_diagnostics = {
        "Score",
        "Miss CVaR",
        "Downside carry",
        "Outcomes",
        "Failure rate",
        "Evidence",
        "Pareto",
    }
    assert expected_diagnostics <= set(headers)
    assert tab.results.raw_rows.rowCount() == 2
    assert tab.results.x_axis.count() > 10
    assert tab.results.minimumWidth() >= 520
    axis_labels = [
        tab.results.x_axis.itemText(index)
        for index in range(tab.results.x_axis.count())
    ]
    assert len(axis_labels) == len(set(axis_labels))
    assert tab.results.plot.toolbar() is not None
    assert tab.csv_button.isEnabled()
    assert tab.json_button.isEnabled()
    assert tab.result_csv_button.isEnabled()
    assert tab.result_json_button.isEnabled()
    tab.stop()


def test_capability_tab_rejects_oversized_interactive_workload(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = CapabilityOptimizationTab()
    qtbot.addWidget(tab)
    tab.controls._integers["candidate_budget"].setValue(1000)
    tab.controls._integers["ensemble_size"].setValue(1000)

    tab.run()

    assert "100000" in tab.status.text()
    assert tab._worker is None


def test_capability_workspace_apply_replaces_inputs_and_invalidates_results(
    qtbot,  # type: ignore[no-untyped-def]
) -> None:
    tab = CapabilityOptimizationTab()
    qtbot.addWidget(tab)
    tab._document = build_capability_workflow(CapabilityWorkflowInputs())
    tab.results.setVisible(True)
    requested = build_capability_workflow(
        CapabilityWorkflowInputs(
            profile_id="loaded-profile",
            target_distance_m=198.0,
            spin_axis_tilt_deg=-8.0,
        )
    )

    tab.apply_capability_workspace_document(requested)

    assert tab.capability_workspace_document() == requested
    assert tab._document is None
    assert not tab.results.isVisibleTo(tab)

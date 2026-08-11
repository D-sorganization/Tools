"""PyQt capability optimizer workflow and diagnostics."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.application.capability_workflow import (  # noqa: E402
    CapabilityWorkflowInputs,
    build_capability_workflow,
    capability_workflow_from_json,
    capability_workflow_json,
)
from rate_of_closure.ui.pyqt6.capability_controls import (
    CapabilityControls,  # noqa: E402
)
from rate_of_closure.ui.pyqt6.capability_tab import (  # noqa: E402
    CapabilityOptimizationTab,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _custom_workflow():  # type: ignore[no-untyped-def]
    payload = json.loads(
        capability_workflow_json(build_capability_workflow(CapabilityWorkflowInputs()))
    )
    profile = payload["profile"]
    club = profile["clubs"][0]
    profile.update(provenance="measured/session-42", confidence=0.71)
    club.update(provenance="fit/driver-42", confidence=0.63)
    club["matrix"] = [[1.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.0]]
    club["parameters"][0].update(
        bias=0.4,
        lower_bound=10.0,
        upper_bound=95.0,
        evidence_lower_bound=30.0,
        evidence_upper_bound=85.0,
    )
    request = payload["request"]
    request.update(
        problem_id="custom-problem-42",
        cvar_alpha=0.83,
        minimum_success_fraction=0.64,
    )
    request["target"].update(kind="fairway", band_half_length_m=21.0, half_width_m=8.0)
    payload["evaluator_config"]["spin_defaults"][0]["provenance"] = "measured/spin-42"
    return capability_workflow_from_json(json.dumps(payload))


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
    requested = _custom_workflow()

    tab.apply_capability_workspace_document(requested)

    assert tab.capability_workspace_document() == requested
    assert tab._document is None
    assert not tab.results.isVisibleTo(tab)


def test_capability_workspace_rejects_stale_worker_success_after_replacement(
    qtbot,
) -> None:  # type: ignore[no-untyped-def]
    tab = CapabilityOptimizationTab()
    qtbot.addWidget(tab)

    class StaleWorker:
        def isRunning(self) -> bool:
            return False

        def cancel(self) -> None:
            pass

    stale_worker = StaleWorker()
    tab._worker = stale_worker  # type: ignore[assignment]
    stale_generation = tab.worker_generation()

    tab.apply_capability_workspace_document(_custom_workflow())
    tab.accept_worker_success(stale_worker, stale_generation, object(), object())

    assert tab._document is None
    assert tab._dataset is None
    assert tab._result is None
    assert not tab.results.isVisibleTo(tab)
    assert not tab.csv_button.isEnabled()
    assert not tab.result_json_button.isEnabled()

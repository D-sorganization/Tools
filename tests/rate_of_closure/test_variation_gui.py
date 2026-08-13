"""PyQt Variation-tab construction and source-context tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.club import get_club  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.simulation import SimulationConfig  # noqa: E402
from rate_of_closure.ui.pyqt6.variation_tab import VariationTab  # noqa: E402
from rate_of_closure.ui.pyqt6.variation_worker import VariationWorker  # noqa: E402
from shared.python.swing_sim.variation import (  # noqa: E402
    CATEGORY_LAUNCH,
    CATEGORY_SWING,
    NoiseSpec,
    VariationPlan,
    keys_for_mode,
    run_variation,
)

_LOCALIZED_FIXTURE = (
    Path(__file__).parents[2]
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "localized_torque_authoring_v1.json"
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_BALL = f"{CATEGORY_LAUNCH}.ball_speed_mph"
_SHOULDER_TORQUE = f"{CATEGORY_SWING}.shoulder_commanded_torque_offset_nm"
_WRIST_TORQUE = f"{CATEGORY_SWING}.wrist_commanded_torque_offset_nm"


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = VariationTab()
    qtbot.addWidget(widget)
    yield widget
    widget.stop()


def _fast_launch_plan(n_runs: int = 12) -> VariationPlan:
    return VariationPlan(
        mode="launch",
        noise=(NoiseSpec(_BALL, scale=1.0),),
        n_runs=n_runs,
        seed=1,
    )


class TestConstruction:
    def test_every_input_carries_hover_guidance(self, tab: VariationTab) -> None:
        widgets = [
            tab._mode_combo,
            tab._base_combo,
            tab._flight_combo,
            tab._runs_spin,
            tab._seed_spin,
            tab._sens_check,
            tab._run_button,
            tab._cancel_button,
            tab._export_csv,
            tab._export_json,
        ]
        for row in tab._rows:
            widgets.extend([row.variable, row.distribution, row.scale, row.clip])
        for widget in widgets:
            assert widget.toolTip(), f"missing tooltip on {widget}"

    def test_variable_picker_covers_the_mode_registry(self, tab: VariationTab) -> None:
        row = tab._rows[0]
        keys = tuple(row.variable.itemData(i) for i in range(row.variable.count()))
        assert keys == keys_for_mode("delivery")

    def test_mode_switch_repopulates_rows(self, tab: VariationTab) -> None:
        tab._mode_combo.setCurrentIndex(2)
        row = tab._rows[0]
        keys = tuple(row.variable.itemData(i) for i in range(row.variable.count()))
        assert keys == keys_for_mode("launch")

    def test_swing_picker_authors_exact_contextual_joint_locus(
        self, tab: VariationTab
    ) -> None:
        tab._mode_combo.setCurrentIndex(1)
        row = tab._rows[0]
        keys = tuple(row.variable.itemData(i) for i in range(row.variable.count()))

        assert _SHOULDER_TORQUE in keys
        assert _WRIST_TORQUE in keys
        row.variable.setCurrentIndex(row.variable.findData(_SHOULDER_TORQUE))
        assert not row.locus_widget.isHidden()
        assert row.joint_selector.currentData() == "joint.shoulder"
        assert row.joint_selector.count() == 1
        assert "topological" in row.joint_selector.toolTip().lower()
        assert "half-open" in row.window_start.toolTip().lower()

        row.window_start.setValue(0.125)
        row.window_end.setValue(0.375)
        spec = row.to_spec()
        assert spec.time_window_s == (0.125, 0.375)
        assert spec.point_ids == ("joint.shoulder",)

        row.variable.setCurrentIndex(row.variable.findData(_WRIST_TORQUE))
        changed = row.to_spec()
        assert changed.point_ids == ("joint.wrist",)
        assert changed.time_window_s != (0.125, 0.375)

    def test_python_reads_the_shared_localized_authoring_fixture(self) -> None:
        plan = VariationPlan.loads(_LOCALIZED_FIXTURE.read_text(encoding="utf-8"))
        assert json.loads(plan.dumps()) == json.loads(
            _LOCALIZED_FIXTURE.read_text(encoding="utf-8")
        )

    def test_incompatible_source_hides_localized_variables(
        self, tab: VariationTab
    ) -> None:
        tab._mode_combo.setCurrentIndex(1)
        tab.set_simulation_config(
            SimulationConfig(
                scenario=ImpactScenario(clubhead_speed_mph=100.0),
                club=get_club("Driver 10.5°"),
                source_kind="manual",
            )
        )

        keys = tuple(
            tab._rows[0].variable.itemData(index)
            for index in range(tab._rows[0].variable.count())
        )
        assert _SHOULDER_TORQUE not in keys
        assert _WRIST_TORQUE not in keys

    def test_add_and_remove_rows_keeps_at_least_one(self, tab: VariationTab) -> None:
        tab._add_row()
        assert len(tab._rows) == 2
        tab._remove_row(tab._rows[1])
        assert len(tab._rows) == 1
        tab._remove_row(tab._rows[0])
        assert len(tab._rows) == 1

    def test_base_change_clears_results_and_ignores_stale_callbacks(
        self, tab: VariationTab
    ) -> None:
        dataset = run_variation(_fast_launch_plan(4), n_workers=1)
        tab._dataset = dataset
        tab._populate_results()
        assert tab._summary_table.rowCount() > 0
        old_generation = tab._generation
        replacement = SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=110.0),
            club=get_club("Driver 10.5°"),
        )

        tab.set_simulation_config(replacement)

        assert tab._dataset is None
        assert tab._summary_table.rowCount() == 0
        expected_status = tab._status.text()
        tab._accept_succeeded(old_generation, dataset, None)
        tab._accept_failed(old_generation, "stale failure")
        assert tab._dataset is None
        assert tab._summary_table.rowCount() == 0
        assert tab._status.text() == expected_status

    def test_old_finished_signal_cannot_unlock_a_new_worker(
        self, tab: VariationTab
    ) -> None:
        old_worker = VariationWorker(_fast_launch_plan(4), compute_sensitivity=False)
        new_worker = VariationWorker(_fast_launch_plan(4), compute_sensitivity=False)
        tab._generation = 2
        tab._worker = new_worker
        tab._set_running(True)

        tab._accept_finished(1, old_worker)

        assert tab._worker is new_worker
        assert not tab._run_button.isEnabled()
        assert tab._cancel_button.isEnabled()

    def test_same_valid_base_recovers_from_explicit_invalid_state(
        self, tab: VariationTab
    ) -> None:
        base = tab._base_simulation_config
        tab.set_simulation_unavailable("Current Simulation inputs are invalid.")
        assert not tab._run_button.isEnabled()

        tab.set_simulation_config(base)

        assert tab._run_button.isEnabled()
        assert tab._status.text() == "Ready with the current Simulation inputs."

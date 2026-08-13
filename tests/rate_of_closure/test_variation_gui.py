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
from rate_of_closure.ui.pyqt6.variation_tab_results import (  # noqa: E402
    VariationTabResultsMixin,
)
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


def _install_accepted(tab: VariationTab, plan: VariationPlan) -> object:
    tab._active_plan = plan
    tab._active_compute_sensitivity = False
    tab._active_authority_identity = tab._current_authority_identity(plan)
    tab._on_succeeded(run_variation(plan, n_workers=1), None)
    tab._set_running(False)
    assert tab.dataset() is not None
    return tab._accepted_authority_identity


class TestConstruction:
    def test_result_methods_resolve_to_results_mixin(self) -> None:
        assert (
            VariationTab._clear_result_widgets
            is VariationTabResultsMixin._clear_result_widgets
        )
        assert (
            VariationTab._apply_prepared_result
            is VariationTabResultsMixin._apply_prepared_result
        )

    def test_visual_state_retains_only_same_authority_accepted_result(
        self, tab: VariationTab
    ) -> None:
        plan = _fast_launch_plan(4)
        identity = _install_accepted(tab, plan)
        assert tab._visual_frame.property("visualPhase") == "result"
        assert tab._summary_table.rowCount() > 0
        landing = tab._landing

        tab.build_plan = lambda: plan  # type: ignore[method-assign]
        tab._on_run()
        assert tab._active_authority_identity == identity
        assert tab._landing is landing
        assert tab._summary_table.rowCount() > 0
        assert tab._visual_frame.property("visualOrigin") == "prior-accepted"
        assert "prior accepted" in tab._visual_frame.accessibleName().lower()
        assert tab._worker is not None
        tab._worker.cancel()
        tab._worker.wait(10_000)

        tab._runs_spin.setValue(tab._runs_spin.value() + 1)
        assert tab._summary_table.rowCount() == 0
        assert tab._visual_frame.property("visualPhase") == "empty"

    def test_failed_same_authority_rerun_keeps_exact_visual_and_export_source(
        self, tab: VariationTab
    ) -> None:
        plan = _fast_launch_plan(4)
        _install_accepted(tab, plan)
        accepted = tab.dataset()
        landing = tab._landing

        tab._active_authority_identity = tab._accepted_authority_identity
        tab._on_failed("diagnostic failure")

        assert tab.dataset() is accepted
        assert tab._landing is landing
        assert tab._visual_frame.property("visualPhase") == "error"
        assert tab._visual_frame.property("visualOrigin") == "prior-accepted"
        assert tab._export_json.isEnabled()

    def test_reserved_state_strip_never_overlaps_visual_content(
        self, tab: VariationTab
    ) -> None:
        tab.resize(1200, 800)
        tab.show()
        tab._active_authority_identity = object()
        tab._accepted_authority_identity = tab._active_authority_identity
        tab._on_failed("diagnostic failure")

        strip = tab._visual_frame._state_strip
        strip_rect = strip.geometry()
        content_rect = tab._visual_frame.content.geometry()
        assert strip.isVisible()
        assert not strip_rect.intersects(content_rect)
        assert content_rect.height() >= 240

    def test_cancel_advances_generation_and_blocks_all_late_partial_callbacks(
        self, tab: VariationTab
    ) -> None:
        plan = _fast_launch_plan(4)
        _install_accepted(tab, plan)
        accepted = tab.dataset()
        tab.build_plan = lambda: plan  # type: ignore[method-assign]
        tab._on_run()
        assert tab._worker is not None
        old_generation = tab._generation

        tab._on_cancel()
        tab._accept_succeeded(old_generation, run_variation(plan, n_workers=1), None)
        tab._accept_failed(old_generation, "late failure")

        assert tab._generation == old_generation + 1
        assert tab.dataset() is accepted
        assert tab._visual_frame.property("visualOrigin") == "prior-accepted"
        assert tab._status.text() == (
            "Cancelled: no partial variation result was accepted."
        )
        assert tab._active_authority_identity is None
        assert "late failure" not in tab._status.text()

    @pytest.mark.parametrize(
        "edit",
        [
            lambda tab: tab._runs_spin.setValue(tab._runs_spin.value() + 1),
            lambda tab: tab._seed_spin.setValue(tab._seed_spin.value() + 1),
            lambda tab: tab._flight_combo.setCurrentIndex(1),
            lambda tab: tab._base_combo.setCurrentIndex(1),
            lambda tab: tab._sens_check.setChecked(not tab._sens_check.isChecked()),
            lambda tab: tab._rows[0].scale.setValue(tab._rows[0].scale.value() + 1),
        ],
    )
    def test_every_editor_family_invalidates_accepted_visual_authority(
        self, tab: VariationTab, edit
    ) -> None:  # type: ignore[no-untyped-def]
        plan = _fast_launch_plan(4)
        _install_accepted(tab, plan)
        assert tab._summary_table.rowCount() > 0

        edit(tab)

        assert tab._summary_table.rowCount() == 0
        assert tab._visual_frame.property("visualPhase") == "empty"

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
        plan = _fast_launch_plan(4)
        _install_accepted(tab, plan)
        dataset = tab.dataset()
        assert dataset is not None
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

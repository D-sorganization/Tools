"""PyQt Variation worker execution, result-view, and export tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
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
    SensitivityResult,
    VariationPlan,
    one_at_a_time_sensitivity,
    run_variation,
)
from shared.python.swing_sim.variation.dataset_io import read_json  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_BALL = f"{CATEGORY_LAUNCH}.ball_speed_mph"


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


class TestRunAndResults:
    def test_individual_only_study_publishes_sensitivity_without_joint_dataset(
        self, qtbot, tab: VariationTab
    ) -> None:  # type: ignore[no-untyped-def]
        tab.load_plan(_fast_launch_plan(8))
        tab._analysis_combo.setCurrentIndex(tab._analysis_combo.findData("individual"))

        with _wait_done(qtbot, tab):
            pass

        assert tab.dataset() is None
        assert tab._sensitivity is not None
        assert tab._summary_table.rowCount() == 0
        assert tab._sensitivity_table.rowCount() == 1
        assert tab._spearman_table.rowCount() == 0
        assert not tab._export_json.isEnabled()
        assert tab._visual_frame.property("visualPhase") == "result"

    def test_long_failure_is_bounded_before_state_strip_publication(
        self, tab: VariationTab
    ) -> None:
        plan = _fast_launch_plan(4)
        tab._active_plan = plan
        tab._active_authority_identity = tab._current_authority_identity(plan)
        tab._on_failed("x" * 100_000)

        assert len(tab._status.text()) < 530
        assert tab._visual_frame.property("visualOrigin") == "empty-preview"
        assert tab._summary_table.rowCount() == 0

    @pytest.mark.parametrize("with_prior", [False, True])
    def test_malformed_typed_sensitivity_never_partially_publishes(
        self, tab: VariationTab, with_prior: bool
    ) -> None:
        plan = _fast_launch_plan(4)
        dataset = run_variation(plan, n_workers=1)
        tab._analysis_combo.setCurrentIndex(tab._analysis_combo.findData("both"))
        if with_prior:
            valid = one_at_a_time_sensitivity(plan, n_workers=1)
            tab._active_plan = plan
            tab._active_analysis_execution = "both"
            tab._active_authority_identity = tab._current_authority_identity(plan)
            tab._on_succeeded(dataset, valid)
        prior = tab.dataset()
        prior_rows = tab._summary_table.rowCount()
        tab._active_plan = plan
        tab._active_analysis_execution = "both"
        tab._active_authority_identity = tab._current_authority_identity(plan)
        malformed = SensitivityResult(
            input_keys=tuple(spec.variable_key for spec in plan.noise),
            output_names=dataset.output_names,
            matrix=np.empty((0, 0)),
            normalized=np.empty((0, 0)),
        )

        tab._on_succeeded(dataset, malformed)

        assert tab.dataset() is prior
        assert tab._summary_table.rowCount() == prior_rows
        assert tab._active_authority_identity is None
        expected_origin = "prior-accepted" if with_prior else "empty-preview"
        assert tab._visual_frame.property("visualOrigin") == expected_origin
        assert tab._visual_frame.property("visualPhase") == "error"
        assert "shape" in tab._status.text()

    def test_forged_same_generation_plan_is_rejected_before_result_mutation(
        self, tab: VariationTab
    ) -> None:
        accepted_plan = _fast_launch_plan(4)
        tab._active_plan = accepted_plan
        tab._active_authority_identity = tab._current_authority_identity(accepted_plan)

        forged_plan = VariationPlan(
            mode=accepted_plan.mode,
            noise=accepted_plan.noise,
            n_runs=accepted_plan.n_runs,
            seed=accepted_plan.seed + 1,
        )
        forged = run_variation(forged_plan, n_workers=1)
        tab._on_succeeded(forged, None)

        assert tab.dataset() is None
        assert "does not match" in tab._status.text()
        assert tab._visual_frame.property("visualPhase") == "error"

    def test_plot_derivation_failure_preserves_prior_complete_bundle(
        self, qtbot, tab: VariationTab, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        yaw = f"{CATEGORY_SWING}.yaw_deg"
        plan = VariationPlan(
            mode="swing",
            noise=(NoiseSpec(yaw, distribution="uniform", scale=0.2),),
            n_runs=3,
            seed=2,
        )
        tab.load_plan(plan)
        tab.set_simulation_config(
            SimulationConfig(
                scenario=ImpactScenario(clubhead_speed_mph=30.0),
                club=get_club("Sand Wedge"),
                source_kind="double_pendulum",
                swing_duration_s=0.05,
            )
        )
        with _wait_done(qtbot, tab):
            pass
        accepted_dataset = tab.dataset()
        accepted_ensemble = tab.ensemble_result()
        assert accepted_dataset is not None and accepted_ensemble is not None
        monkeypatch.setattr(
            "rate_of_closure.ui.pyqt6.variation_tab_run.build_ensemble_plot_dataset",
            lambda _result: (_ for _ in ()).throw(
                RuntimeError("planted render failure")
            ),
        )

        with _wait_done(qtbot, tab):
            pass

        assert tab.dataset() is accepted_dataset
        assert tab.ensemble_result() is accepted_ensemble
        assert tab._visual_frame.property("visualOrigin") == "prior-accepted"
        assert "planted render failure" in tab._status.text()

    def test_swing_callback_rejects_same_plan_different_dataset_bundle(
        self, qtbot, tab: VariationTab
    ) -> None:  # type: ignore[no-untyped-def]
        yaw = f"{CATEGORY_SWING}.yaw_deg"
        plan = VariationPlan(
            mode="swing",
            noise=(NoiseSpec(yaw, distribution="uniform", scale=0.2),),
            n_runs=3,
            seed=2,
        )
        tab.load_plan(plan)
        tab._analysis_combo.setCurrentIndex(
            tab._analysis_combo.findData("all_together")
        )
        tab.set_simulation_config(
            SimulationConfig(
                scenario=ImpactScenario(clubhead_speed_mph=30.0),
                club=get_club("Sand Wedge"),
                source_kind="double_pendulum",
                swing_duration_s=0.05,
            )
        )
        with _wait_done(qtbot, tab):
            pass
        accepted_dataset = tab.dataset()
        accepted_ensemble = tab.ensemble_result()
        accepted_identity = tab._accepted_authority_identity
        assert accepted_dataset is not None and accepted_ensemble is not None
        assert accepted_identity is not None
        tab._active_plan = plan
        tab._active_authority_identity = accepted_identity
        tab._active_analysis_execution = "all_together"
        tab._pending_ensemble_result = accepted_ensemble
        forged_dataset = replace(
            accepted_dataset,
            elapsed_s=accepted_dataset.elapsed_s + 0.001,
        )

        tab._on_succeeded(forged_dataset, None)

        assert tab.dataset() is accepted_dataset
        assert tab.ensemble_result() is accepted_ensemble
        assert tab._active_authority_identity is None
        assert tab._visual_frame.property("visualOrigin") == "prior-accepted"
        assert "does not match" in tab._status.text()

    def test_post_derivation_publication_failure_rolls_back_exact_prior_bundle(
        self, qtbot, tab: VariationTab, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        plan = _fast_launch_plan(8)
        tab.load_plan(plan)
        with _wait_done(qtbot, tab):
            pass
        accepted_dataset = tab.dataset()
        accepted_sensitivity = tab._sensitivity
        row_count = tab._summary_table.rowCount()
        assert accepted_dataset is not None and accepted_sensitivity is not None
        from rate_of_closure.ui.pyqt6 import variation_tab_results

        populate = variation_tab_results.populate_result_views
        calls = 0

        def fail_once(*args):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("planted publication failure")
            return populate(*args)

        monkeypatch.setattr(variation_tab_results, "populate_result_views", fail_once)

        with _wait_done(qtbot, tab):
            pass

        assert tab.dataset() is accepted_dataset
        assert tab._sensitivity is accepted_sensitivity
        assert tab._summary_table.rowCount() == row_count
        assert tab._accepted_authority_identity is not None
        assert tab._active_authority_identity is None
        assert tab._visual_frame.property("visualOrigin") == "prior-accepted"
        assert tab._export_json.isEnabled()
        assert "planted publication failure" in tab._status.text()

    def test_full_study_populates_every_results_view(
        self, qtbot, tab: VariationTab
    ) -> None:  # type: ignore[no-untyped-def]
        tab.load_plan(_fast_launch_plan())
        with _wait_done(qtbot, tab):
            pass
        dataset = tab.dataset()
        assert dataset is not None and dataset.n_success == 12
        assert tab._summary_table.rowCount() == len(dataset.output_names)
        assert tab._sensitivity_table.rowCount() == 1
        assert tab._spearman_table.columnCount() == len(dataset.output_names)
        assert tab._ensemble_scatter._canvas.axes.collections
        scatter_keys = {
            tab._ensemble_scatter._y_combo.itemData(index)
            for index in range(tab._ensemble_scatter._y_combo.count())
        }
        assert "output:carry_m" in scatter_keys

    def test_cancel_before_start_reports_cancelled(
        self, qtbot, tab: VariationTab
    ) -> None:  # type: ignore[no-untyped-def]
        worker = VariationWorker(
            _fast_launch_plan(200), analysis_execution="all_together"
        )
        worker.cancel()
        with qtbot.waitSignal(worker.cancelled, timeout=15_000):
            worker.start()
        worker.wait(10_000)

    def test_worker_progress_counts_joint_and_individual_evaluations(
        self, qtbot
    ) -> None:  # type: ignore[no-untyped-def]
        worker = VariationWorker(
            _fast_launch_plan(8), analysis_execution="both", n_workers=1
        )
        reports = []
        worker.progressed.connect(reports.append)

        with qtbot.waitSignal(worker.finished, timeout=30_000):
            worker.start()
        worker.wait(10_000)

        assert worker.total_runs == 16
        assert reports[-1].iteration == 16

    def test_swing_study_populates_trace_scatter_and_arc_views(
        self, qtbot, tab: VariationTab
    ) -> None:  # type: ignore[no-untyped-def]
        yaw = f"{CATEGORY_SWING}.yaw_deg"
        tab.load_plan(
            VariationPlan(
                mode="swing",
                noise=(NoiseSpec(yaw, distribution="uniform", scale=0.2),),
                n_runs=3,
                seed=2,
            )
        )
        tab.set_simulation_config(
            SimulationConfig(
                scenario=ImpactScenario(clubhead_speed_mph=30.0),
                club=get_club("Sand Wedge"),
                source_kind="double_pendulum",
                swing_duration_s=0.05,
            )
        )

        with _wait_done(qtbot, tab):
            pass

        assert tab.ensemble_result() is not None
        assert tab._sensitivity_table.rowCount() == 1
        assert tab._ensemble_scatter._canvas.axes.collections
        assert len(tab._arc_overlay._canvas.axes.lines) >= 4
        assert tab._arc_overlay._variability_canvas.axes.lines
        assert "3/3 trials" in tab._arc_overlay._status.text()
        assert "Hits 3 · No Impact 0 · Failures 0 · Landings 3" in (
            tab._landing._axes.get_title()
        )
        assert tab._export_trace_csv.isEnabled()
        assert tab._export_ensemble_json.isEnabled()

    def test_export_json_round_trips_from_the_tab(
        self, qtbot, tab: VariationTab, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        tab.load_plan(_fast_launch_plan())
        with _wait_done(qtbot, tab):
            pass
        target = tmp_path / "study.json"
        monkeypatch.setattr(
            "rate_of_closure.ui.pyqt6.variation_tab.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: (str(target), "JSON (*.json)")),
        )
        tab._on_export_json()
        loaded = read_json(target)
        assert loaded.plan == tab.dataset().plan
        document = json.loads(target.read_text(encoding="utf-8"))
        assert document["schema_version"] == 2
        assert document["plan_document"]["schema_version"] == 3


class _wait_done:
    """Click Run and wait for the worker to finish."""

    def __init__(self, qtbot, tab: VariationTab) -> None:  # type: ignore[no-untyped-def]
        self._qtbot = qtbot
        self._tab = tab

    def __enter__(self) -> None:
        tab = self._tab
        tab._on_run()
        assert tab._worker is not None
        with self._qtbot.waitSignal(tab._worker.finished, timeout=60_000):
            pass

    def __exit__(self, *exc: object) -> None:
        return None

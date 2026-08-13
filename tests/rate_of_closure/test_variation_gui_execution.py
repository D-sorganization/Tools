"""PyQt Variation worker execution, result-view, and export tests."""

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
        worker = VariationWorker(_fast_launch_plan(200), compute_sensitivity=False)
        worker.cancel()
        with qtbot.waitSignal(worker.cancelled, timeout=15_000):
            worker.start()
        worker.wait(10_000)

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
        assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == 1


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

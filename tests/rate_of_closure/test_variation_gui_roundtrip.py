"""PyQt Variation-plan authoring, persistence, and atomic-import tests."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.ui.pyqt6.variation_tab import VariationTab  # noqa: E402
from shared.python.contracts import ContractViolationError  # noqa: E402
from shared.python.swing_sim.variation import (  # noqa: E402
    CATEGORY_DELIVERY,
    CATEGORY_LAUNCH,
    CATEGORY_SWING,
    NoiseSpec,
    PerturbationGroup,
    VariationPlan,
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


class TestPlanRoundTrip:
    def test_build_plan_reflects_the_editors(self, tab: VariationTab) -> None:
        tab._runs_spin.setValue(33)
        tab._seed_spin.setValue(7)
        row = tab._rows[0]
        index = row.variable.findData(f"{CATEGORY_DELIVERY}.face_angle_deg")
        row.variable.setCurrentIndex(index)
        row.distribution.setCurrentText("uniform")
        row.scale.setValue(2.5)
        plan = tab.build_plan()
        assert plan.mode == "delivery"
        assert plan.n_runs == 33 and plan.seed == 7
        spec = plan.noise[0]
        assert spec.variable_key == f"{CATEGORY_DELIVERY}.face_angle_deg"
        assert spec.distribution == "uniform" and spec.scale == 2.5

    def test_load_plan_round_trips_including_base_and_truncation(
        self, tab: VariationTab
    ) -> None:
        plan = VariationPlan(
            mode="launch",
            base_variables={_BALL: 155.0},
            noise=(
                NoiseSpec(_BALL, scale=2.0, lower=140.0, upper=170.0),
                NoiseSpec(
                    f"{CATEGORY_LAUNCH}.spin_rpm",
                    distribution="triangular",
                    scale=150.0,
                ),
            ),
            n_runs=44,
            seed=9,
            flight_model="waterloo_penner",
        )
        tab.load_plan(plan)
        assert tab.build_plan() == plan

    def test_load_contextual_plan_round_trips_locus_groups_and_precision(
        self, tab: VariationTab
    ) -> None:
        plan = VariationPlan(
            mode="swing",
            noise=(
                NoiseSpec(
                    _SHOULDER_TORQUE,
                    scale=1.123456789,
                    spec_id="shoulder-window",
                    time_window_s=(0.123456789, 0.456789123),
                    point_ids=("joint.shoulder",),
                ),
                NoiseSpec(
                    _WRIST_TORQUE,
                    scale=0.987654321,
                    spec_id="wrist-window",
                    time_window_s=(0.2, 0.6),
                    point_ids=("joint.wrist",),
                ),
            ),
            groups=(
                PerturbationGroup(
                    group_id="joint-torque-group",
                    spec_ids=("shoulder-window", "wrist-window"),
                    matrix=((1.0, 0.25), (0.25, 1.0)),
                ),
            ),
            n_runs=4,
        )

        tab.load_plan(plan)

        assert tab.build_plan() == plan
        tab._seed_spin.setValue(9)
        rebuilt = tab.build_plan()
        assert rebuilt.seed == 9
        assert rebuilt.noise[0].scale == 1.123456789
        assert rebuilt.noise[0].time_window_s == (0.123456789, 0.456789123)
        assert rebuilt.noise[0].point_ids == ("joint.shoulder",)
        assert rebuilt.groups == plan.groups

    @pytest.mark.parametrize("edited_field", ["start", "end"])
    def test_locus_edit_preserves_unedited_high_precision_endpoint(
        self, tab: VariationTab, edited_field: str
    ) -> None:
        exact_window = (0.123456789123, 0.456789123456)
        spec = NoiseSpec(
            _SHOULDER_TORQUE,
            scale=1.0,
            time_window_s=exact_window,
            point_ids=("joint.shoulder",),
        )
        tab.load_plan(VariationPlan(mode="swing", noise=(spec,), n_runs=4))
        row = tab._rows[0]

        if edited_field == "start":
            row.window_start.setValue(0.2)
        else:
            row.window_end.setValue(0.6)
        rebuilt = tab.build_plan().noise[0]

        expected = (
            (0.2, exact_window[1])
            if edited_field == "start"
            else (exact_window[0], 0.6)
        )
        assert rebuilt.time_window_s == expected

    @pytest.mark.parametrize(
        ("window", "points", "message"),
        [
            (None, ("joint.shoulder",), "time window"),
            ((0.4, 0.2), ("joint.shoulder",), "start < end"),
            ((0.2, 1.6), ("joint.shoulder",), "duration"),
            ((0.2, 0.4), ("swing.wrist",), "topological joint"),
        ],
    )
    def test_invalid_contextual_plan_fails_before_editor_mutation(
        self,
        tab: VariationTab,
        window: tuple[float, float] | None,
        points: tuple[str, ...],
        message: str,
    ) -> None:
        original = tab.build_plan()
        with pytest.raises((ContractViolationError, ValueError), match=message):
            spec = NoiseSpec(
                _SHOULDER_TORQUE,
                scale=1.0,
                time_window_s=window,
                point_ids=points,
            )
            tab.load_plan(VariationPlan(mode="swing", noise=(spec,), n_runs=4))
        assert tab.build_plan() == original

    def test_reversed_authored_window_blocks_run_with_visible_status(
        self, tab: VariationTab
    ) -> None:
        tab._mode_combo.setCurrentIndex(1)
        row = tab._rows[0]
        row.variable.setCurrentIndex(row.variable.findData(_SHOULDER_TORQUE))
        row.window_start.setValue(0.4)
        row.window_end.setValue(0.2)

        tab._on_run()

        assert "Cannot run" in tab._status.text()
        assert "start < end" in tab._status.text()
        assert tab.dataset() is None

    def test_v2_plan_round_trips_custom_ids_loci_groups_and_save(
        self, tab: VariationTab, tmp_path: Path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        spin = f"{CATEGORY_LAUNCH}.spin_rpm"
        plan = VariationPlan(
            mode="launch",
            base_variables={_BALL: 155.25, spin: 2_450.0},
            noise=(
                NoiseSpec(
                    _BALL,
                    scale=2.125,
                    spec_id="launch-speed-window",
                    time_window_s=(0.01, 0.025),
                    point_ids=("ball.center",),
                ),
                NoiseSpec(
                    spin,
                    scale=175.5,
                    spec_id="launch-spin-window",
                    time_window_s=(0.01, 0.025),
                    point_ids=("ball.center",),
                ),
            ),
            groups=(
                PerturbationGroup(
                    group_id="speed-spin-correlation",
                    spec_ids=("launch-speed-window", "launch-spin-window"),
                    matrix=((1.0, -0.35), (-0.35, 1.0)),
                ),
            ),
            n_runs=44,
            seed=19,
        )

        tab.load_plan(plan)
        assert tab.build_plan() == plan
        target = tmp_path / "variation-plan-v2.json"
        monkeypatch.setattr(
            "rate_of_closure.ui.pyqt6.variation_tab.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: (str(target), "JSON (*.json)")),
        )
        tab._on_save_plan()
        assert VariationPlan.loads(target.read_text(encoding="utf-8")) == plan

        tab._rows[0].scale.setValue(2.5)
        edited = tab.build_plan()
        assert edited.noise[0].scale == 2.5
        assert edited.noise[0].spec_id == "launch-speed-window"
        assert edited.noise[0].time_window_s == (0.01, 0.025)
        assert edited.noise[0].point_ids == ("ball.center",)
        assert edited.groups == plan.groups

    def test_unrelated_edit_preserves_each_unedited_numeric_authority(
        self, tab: VariationTab
    ) -> None:
        precise = NoiseSpec(
            _BALL,
            distribution="normal",
            scale=2.123456789,
            lower=140.123456789,
            upper=170.987654321,
            spec_id="precise-launch-speed",
        )
        plan = VariationPlan(mode="launch", noise=(precise,), n_runs=12, seed=5)
        tab.load_plan(plan)

        tab._rows[0].distribution.setCurrentText("uniform")
        rebuilt = tab.build_plan().noise[0]

        assert rebuilt.distribution == "uniform"
        assert rebuilt.scale == precise.scale
        assert rebuilt.lower == precise.lower
        assert rebuilt.upper == precise.upper

    @pytest.mark.parametrize("load_from_file", [False, True])
    def test_unrepresentable_plan_is_rejected_before_editor_mutation(
        self,
        tab: VariationTab,
        tmp_path: Path,
        monkeypatch,
        load_from_file: bool,
    ) -> None:  # type: ignore[no-untyped-def]
        original = _fast_launch_plan(12)
        tab.load_plan(original)
        before = tab.build_plan()
        unsupported = VariationPlan(
            mode="launch",
            noise=(NoiseSpec(_BALL, scale=1.0),),
            n_runs=tab._runs_spin.maximum() + 1,
            seed=91,
        )

        if load_from_file:
            target = tmp_path / "unsupported-plan.json"
            target.write_text(unsupported.dumps(), encoding="utf-8")
            monkeypatch.setattr(
                "rate_of_closure.ui.pyqt6.variation_tab.QFileDialog.getOpenFileName",
                staticmethod(lambda *a, **k: (str(target), "JSON (*.json)")),
            )
            tab._on_load_plan()
            assert tab._status.text().startswith("Cannot load plan: plan n_runs")
        else:
            with pytest.raises(ValueError, match="n_runs"):
                tab.load_plan(unsupported)

        assert tab.build_plan() == before

    def test_explorer_scenario_base_carries_speed_and_offsets(
        self, tab: VariationTab
    ) -> None:
        tab._base_combo.setCurrentIndex(1)
        plan = tab.build_plan()
        key = f"{CATEGORY_DELIVERY}.clubhead_speed_mps"
        assert plan.base_variables[key] == pytest.approx(113.0 * 0.44704)

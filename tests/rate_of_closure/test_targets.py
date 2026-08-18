"""App-side target-region facade + Variation hold-% tests (#4125 H7b)."""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.simulation.targets import (
    TargetRegion,
    hold_fraction,
    hold_stats,
    layout_for_region,
)
from shared.python.swing_sim.variation import (
    NoiseSpec,
    VariationDataset,
    VariationPlan,
    outputs_for_mode,
)

pytestmark = [pytest.mark.unit, pytest.mark.physics]

_GREEN = TargetRegion(kind="green", distance_m=200.0, radius_m=10.0)


class TestHoldStats:
    def test_hand_counted_scatter(self) -> None:
        """Hand-counted fixture: exactly 3 of 5 finite shots hold."""
        carries = np.array([200.0, 205.0, 209.0, 215.0, 200.0, np.nan])
        laterals = np.array([0.0, 0.0, 0.0, 0.0, 11.0, 0.0])
        # In: (200,0) d=-10, (205,0) d=-5, (209,0) d=-1.
        # Out: (215,0) d=+5, (200,11) d=+1. NaN excluded entirely.
        held, total = hold_stats(carries, laterals, _GREEN)
        assert (held, total) == (3, 5)

    def test_boundary_point_counts_as_holding(self) -> None:
        held, total = hold_stats(np.array([210.0]), np.array([0.0]), _GREEN)
        assert (held, total) == (1, 1)

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="matching"):
            hold_stats(np.zeros(3), np.zeros(2), _GREEN)


class TestHoldFraction:
    def _dataset(self, carries: list[float], laterals: list[float]) -> VariationDataset:
        names = outputs_for_mode("launch")
        n = len(carries)
        outputs = np.full((n, len(names)), 1.0)
        outputs[:, names.index("carry_m")] = carries
        outputs[:, names.index("lateral_m")] = laterals
        plan = VariationPlan(
            mode="launch",
            noise=(
                NoiseSpec(
                    variable_key="swing_sim.flight.launch.ball_speed_mph", scale=1.0
                ),
            ),
            n_runs=n,
        )
        return VariationDataset(
            plan=plan,
            input_names=("swing_sim.flight.launch.ball_speed_mph",),
            inputs=np.zeros((n, 1)),
            output_names=names,
            outputs=outputs,
            success=np.ones(n, dtype=bool),
        )

    def test_matches_the_hand_counted_fixture(self) -> None:
        dataset = self._dataset(
            [200.0, 205.0, 209.0, 215.0, 200.0], [0.0, 0.0, 0.0, 0.0, 11.0]
        )
        assert hold_fraction(dataset, _GREEN) == pytest.approx(3.0 / 5.0)

    def test_all_out_and_all_in(self) -> None:
        assert hold_fraction(self._dataset([300.0], [0.0]), _GREEN) == 0.0
        assert hold_fraction(self._dataset([200.0], [0.0]), _GREEN) == 1.0


class TestLayoutBridge:
    def test_green_target_moves_the_course_green(self) -> None:
        layout = layout_for_region(_GREEN)
        assert layout.green_distance_m == pytest.approx(200.0)
        assert layout.green_radius_m == pytest.approx(10.0)

    def test_fairway_target_sets_the_strip_width(self) -> None:
        fw = TargetRegion(
            kind="fairway",
            distance_m=230.0,
            band_half_length_m=20.0,
            half_width_m=22.0,
        )
        layout = layout_for_region(fw)
        assert layout.fairway_half_width_m == pytest.approx(22.0)
        assert layout.green_distance_m == pytest.approx(250.0)

"""Dispersion / sensitivity analysis behaviour (#4120 V3)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from shared.python.swing_sim.variation import (
    CATEGORY_DELIVERY,
    CATEGORY_LAUNCH,
    NoiseSpec,
    PerturbationGroup,
    VariationDataset,
    VariationPlan,
    dispersion_ellipse,
    one_at_a_time_sensitivity,
    outputs_for_mode,
    run_variation,
    spearman_matrix,
    summary_stats,
)

pytestmark = pytest.mark.physics

_FACE = f"{CATEGORY_DELIVERY}.face_angle_deg"
_SPEED = f"{CATEGORY_DELIVERY}.clubhead_speed_mps"


def _planted_plan(n_runs: int = 32) -> VariationPlan:
    """3-degree face-angle noise vs 0.1 mph (0.045 m/s) speed noise."""
    return VariationPlan(
        mode="delivery",
        noise=(
            NoiseSpec(_FACE, scale=3.0),
            NoiseSpec(_SPEED, scale=0.045),
        ),
        n_runs=n_runs,
        seed=9,
    )


def _synthetic_launch_dataset(points: np.ndarray) -> VariationDataset:
    """Wrap known (carry, lateral) landing points in a launch dataset."""
    n = points.shape[0]
    plan = VariationPlan(
        mode="launch",
        noise=(NoiseSpec(f"{CATEGORY_LAUNCH}.ball_speed_mph", scale=1.0),),
        n_runs=n,
        seed=0,
    )
    names = outputs_for_mode("launch")
    outputs = np.zeros((n, len(names)))
    outputs[:, names.index("carry_m")] = points[:, 0]
    outputs[:, names.index("lateral_m")] = points[:, 1]
    return VariationDataset(
        plan=plan,
        input_names=(f"{CATEGORY_LAUNCH}.ball_speed_mph",),
        inputs=np.linspace(148.0, 152.0, n).reshape(-1, 1),
        output_names=names,
        outputs=outputs,
        success=np.ones(n, dtype=bool),
    )


class TestSensitivity:
    def test_planted_dominant_variable_is_identified(self) -> None:
        """Face-angle noise must dominate lateral landing; speed carry."""
        result = one_at_a_time_sensitivity(_planted_plan(), n_workers=4)
        assert result.dominant_input("lateral_m") == _FACE
        assert result.dominant_input("launch_azimuth_deg") == _FACE
        lat = result.output_names.index("lateral_m")
        face_row = result.input_keys.index(_FACE)
        speed_row = result.input_keys.index(_SPEED)
        assert result.matrix[face_row, lat] > 10.0 * result.matrix[speed_row, lat]
        # Normalization: the dominant input scores 1.0 per output column.
        assert result.normalized[face_row, lat] == pytest.approx(1.0)

    def test_grouped_plan_oat_runs_each_spec_as_an_independent_intervention(
        self,
    ) -> None:
        plan = _planted_plan(n_runs=8)
        grouped = dataclasses.replace(
            plan,
            noise=tuple(
                dataclasses.replace(spec, spec_id=f"spec-{index}")
                for index, spec in enumerate(plan.noise)
            ),
            groups=(
                PerturbationGroup(
                    group_id="delivery-correlation",
                    spec_ids=("spec-0", "spec-1"),
                    matrix=((1.0, 0.25), (0.25, 1.0)),
                ),
            ),
        )

        result = one_at_a_time_sensitivity(grouped, n_workers=1)

        assert result.input_keys == tuple(spec.variable_key for spec in grouped.noise)
        assert result.matrix.shape == (
            len(grouped.noise),
            len(result.output_names),
        )

    def test_spearman_corroborates_the_planted_dominance(self) -> None:
        dataset = run_variation(_planted_plan(), n_workers=4)
        rho = spearman_matrix(dataset)
        lat = dataset.output_names.index("lateral_m")
        face_row = dataset.input_names.index(_FACE)
        speed_row = dataset.input_names.index(_SPEED)
        assert abs(rho[face_row, lat]) > 0.9
        assert abs(rho[face_row, lat]) > abs(rho[speed_row, lat])

    def test_spearman_is_plus_one_for_a_monotonic_relation(self) -> None:
        points = np.column_stack(
            [np.linspace(200.0, 220.0, 12), np.linspace(-3.0, 3.0, 12)]
        )
        dataset = _synthetic_launch_dataset(points)
        # inputs ascend with rows and carry ascends with rows -> rho = +1.
        rho = spearman_matrix(dataset)
        carry = dataset.output_names.index("carry_m")
        assert rho[0, carry] == pytest.approx(1.0)


class TestSummaryStats:
    def test_stats_match_numpy_on_a_real_dataset(self) -> None:
        dataset = run_variation(_planted_plan(n_runs=24), n_workers=4)
        stats = {s.name: s for s in summary_stats(dataset)}
        carry = dataset.output_column("carry_m")
        assert stats["carry_m"].mean == pytest.approx(float(np.mean(carry)))
        assert stats["carry_m"].std == pytest.approx(float(np.std(carry, ddof=1)))
        assert stats["carry_m"].n == 24

    def test_empty_column_reports_nan(self) -> None:
        dataset = _synthetic_launch_dataset(np.zeros((4, 2)))
        object.__setattr__(dataset, "success", np.zeros(4, dtype=bool))
        stats = summary_stats(dataset)
        assert all(math.isnan(s.mean) and s.n == 0 for s in stats)


class TestDispersionEllipse:
    def test_axis_aligned_gaussian_recovers_its_sigmas(self) -> None:
        rng = np.random.default_rng(0)
        points = np.column_stack(
            [rng.normal(230.0, 6.0, 4000), rng.normal(2.0, 1.5, 4000)]
        )
        ellipse = dispersion_ellipse(_synthetic_launch_dataset(points))
        assert ellipse.center_carry_m == pytest.approx(230.0, abs=0.5)
        assert ellipse.center_lateral_m == pytest.approx(2.0, abs=0.2)
        assert ellipse.semi_major_m == pytest.approx(2.0 * 6.0, rel=0.1)
        assert ellipse.semi_minor_m == pytest.approx(2.0 * 1.5, rel=0.1)
        # Principal axis along carry (angle ~ 0 or 180 degrees).
        assert min(abs(ellipse.angle_deg), abs(abs(ellipse.angle_deg) - 180.0)) < 10.0

    def test_requires_two_successful_runs(self) -> None:
        dataset = _synthetic_launch_dataset(np.zeros((4, 2)))
        object.__setattr__(dataset, "success", np.array([True, False, False, False]))
        with pytest.raises(Exception, match="successful runs"):
            dispersion_ellipse(dataset)

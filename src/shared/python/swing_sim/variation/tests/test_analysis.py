"""Dispersion / sensitivity analysis behaviour (#4120 V3)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    CATEGORY_DELIVERY,
    CATEGORY_LAUNCH,
    NoiseSpec,
    PerturbationGroup,
    SensitivityResult,
    VariationDataset,
    VariationPlan,
    dispersion_ellipse,
    one_at_a_time_sensitivity,
    outputs_for_mode,
    run_variation,
    spearman_matrix,
    summary_stats,
)
from shared.python.swing_sim.variation import analysis as variation_analysis

pytestmark = pytest.mark.physics

_FACE = f"{CATEGORY_DELIVERY}.face_angle_deg"
_SPEED = f"{CATEGORY_DELIVERY}.clubhead_speed_mps"
_PAIRWISE_FIXTURE = (
    Path(__file__).parents[4] / "fixtures" / "variation_spearman_pairwise_finite.json"
)


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


def _pairwise_fixture_dataset() -> tuple[VariationDataset, np.ndarray]:
    """Load the cross-runtime missing-value Spearman fixture."""
    payload = json.loads(_PAIRWISE_FIXTURE.read_text(encoding="utf-8"))
    inputs = np.asarray(
        [
            [math.nan if value is None else value for value in row]
            for row in payload["inputs"]
        ],
        dtype=float,
    )
    outputs = np.asarray(
        [
            [math.nan if value is None else value for value in row]
            for row in payload["outputs"]
        ],
        dtype=float,
    )
    expected = np.asarray(
        [
            [math.nan if value is None else value for value in row]
            for row in payload["expected"]
        ],
        dtype=float,
    )
    plan = VariationPlan(
        mode="launch",
        noise=(
            NoiseSpec(f"{CATEGORY_LAUNCH}.ball_speed_mph", scale=1.0),
            NoiseSpec(f"{CATEGORY_LAUNCH}.spin_rpm", scale=1.0),
            NoiseSpec(f"{CATEGORY_LAUNCH}.launch_angle_deg", scale=1.0),
        ),
        n_runs=len(payload["success"]),
        seed=0,
    )
    dataset = VariationDataset(
        plan=plan,
        input_names=tuple(payload["input_names"]),
        inputs=inputs,
        output_names=tuple(payload["output_names"]),
        outputs=outputs,
        success=np.asarray(payload["success"], dtype=bool),
    )
    return dataset, expected


class TestSensitivity:
    def test_dominant_input_ignores_unavailable_cells_and_rejects_empty_column(
        self,
    ) -> None:
        matrix = np.asarray([[math.nan, math.nan], [2.0, math.nan]])
        result = SensitivityResult(
            ("unavailable", "measured"),
            ("partial", "empty"),
            matrix,
            matrix.copy(),
        )
        assert result.dominant_input("partial") == "measured"
        with pytest.raises(ContractViolationError, match="no available"):
            result.dominant_input("empty")

    def test_normalized_policy_preserves_unavailable_and_finite_zero_columns(
        self,
    ) -> None:
        matrix = np.asarray([[math.nan, 0.0, math.nan], [math.nan, 0.0, 2.0]])
        normalized = variation_analysis._normalize_sensitivity_matrix(matrix)
        assert np.all(np.isnan(normalized[:, 0]))
        assert np.array_equal(normalized[:, 1], np.asarray([0.0, 0.0]))
        assert np.isnan(normalized[0, 2])
        assert normalized[1, 2] == 1.0

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

    def test_spearman_uses_pairwise_finite_rows_with_cross_runtime_parity(
        self,
    ) -> None:
        dataset, expected = _pairwise_fixture_dataset()

        rho = spearman_matrix(dataset)

        np.testing.assert_allclose(rho, expected, equal_nan=True)

    def test_oat_uses_each_outputs_finite_evaluated_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = _planted_plan(n_runs=4)

        def fake_run_variation(
            sub_plan: VariationPlan, **_kwargs: object
        ) -> VariationDataset:
            is_face = sub_plan.noise[0].variable_key == _FACE
            outputs = (
                np.array(
                    ((1.0, 4.0), (math.nan, math.nan), (3.0, math.nan), (999.0, 999.0))
                )
                if is_face
                else np.array(((2.0, 5.0), (4.0, 7.0), (6.0, 9.0), (999.0, 999.0)))
            )
            return VariationDataset(
                plan=sub_plan,
                input_names=(sub_plan.noise[0].variable_key,),
                inputs=np.zeros((4, 1)),
                output_names=("partially_available", "below_minimum"),
                outputs=outputs,
                success=np.array((True, True, True, False)),
            )

        monkeypatch.setattr(variation_analysis, "run_variation", fake_run_variation)

        result = one_at_a_time_sensitivity(plan, n_workers=1)

        assert result.matrix[0, 0] == pytest.approx(math.sqrt(2.0))
        assert math.isnan(result.matrix[0, 1])
        assert result.matrix[1, 0] == pytest.approx(2.0)
        assert result.matrix[1, 1] == pytest.approx(2.0)


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
        with pytest.raises(Exception, match="paired finite landing rows"):
            dispersion_ellipse(dataset)

    def test_uses_only_paired_finite_landing_rows(self) -> None:
        dataset = _synthetic_launch_dataset(
            np.array(((100.0, 10.0), (1000.0, np.nan), (np.nan, 100.0), (300.0, 30.0)))
        )

        ellipse = dispersion_ellipse(dataset)

        assert ellipse.n == 2
        assert ellipse.center_carry_m == pytest.approx(200.0)
        assert ellipse.center_lateral_m == pytest.approx(20.0)

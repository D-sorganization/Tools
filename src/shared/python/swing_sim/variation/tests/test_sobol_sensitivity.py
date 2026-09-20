"""Tests for Saltelli sampling and Sobol sensitivity indices (#4253 item a)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from shared.python.swing_sim.variation.sobol_sensitivity import (
    SOBOL_REPORT_SCHEMA_ID,
    SOBOL_REPORT_SCHEMA_VERSION,
    SobolFactor,
    SobolObservations,
    SobolOutput,
    analyze_sobol,
    generate_saltelli_design,
    sobol_run_count_guidance,
)

pytestmark = pytest.mark.physics


def _factors() -> tuple[SobolFactor, ...]:
    return (
        SobolFactor(
            spec_id="x1",
            variable_key="swing_sim.impact.delivery.face_angle_deg",
            lower=0.0,
            upper=1.0,
            unit="deg",
        ),
        SobolFactor(
            spec_id="x2",
            variable_key="swing_sim.impact.delivery.clubhead_speed_mps",
            lower=0.0,
            upper=1.0,
            unit="m/s",
        ),
    )


def _observations(
    response: Callable[[np.ndarray], float],
    *,
    base_samples: int = 256,
    seed: int = 42,
    factors: tuple[SobolFactor, ...] | None = None,
) -> SobolObservations:
    factor_list = factors if factors is not None else _factors()
    design = generate_saltelli_design(factor_list, base_samples=base_samples, seed=seed)
    total_samples = design.total_samples
    values = np.empty((total_samples, 1), dtype=float)
    for i in range(total_samples):
        values[i, 0] = response(design.physical_points[i])
    success = np.ones(total_samples, dtype=bool)
    return SobolObservations(
        design=design,
        outputs=(
            SobolOutput(
                name="test_output",
                unit="m",
            ),
        ),
        values=values,
        success=success,
    )


class TestSobolGuidance:
    def test_run_count_guidance_calculates_total_samples(self) -> None:
        guidance = sobol_run_count_guidance(factor_count=3, base_samples=512)
        assert guidance.factor_count == 3
        assert guidance.base_samples == 512
        assert guidance.total_runs == 512 * (3 + 2)  # 2560
        assert guidance.adequacy == "adequate"
        assert "2560" in guidance.guidance_message

    def test_guidance_flags_low_sample_counts(self) -> None:
        guidance = sobol_run_count_guidance(factor_count=2, base_samples=32)
        assert guidance.adequacy == "inadequate"

        marginal = sobol_run_count_guidance(factor_count=2, base_samples=128)
        assert marginal.adequacy == "marginal"


class TestSaltelliDesign:
    def test_design_shape_and_bounds(self) -> None:
        factors = _factors()
        design = generate_saltelli_design(factors, base_samples=64, seed=123)
        # N * (k + 2) = 64 * 4 = 256
        assert design.total_samples == 256
        assert design.base_samples == 64
        assert design.factors == factors
        assert design.physical_points.shape == (256, 2)
        assert np.all(design.physical_points >= 0.0)
        assert np.all(design.physical_points <= 1.0)

    def test_reproducible_from_seed(self) -> None:
        factors = _factors()
        d1 = generate_saltelli_design(factors, base_samples=32, seed=99)
        d2 = generate_saltelli_design(factors, base_samples=32, seed=99)
        np.testing.assert_array_equal(d1.physical_points, d2.physical_points)


class TestSobolAnalysis:
    def test_additive_model_recovers_analytical_indices(self) -> None:
        # f(x1, x2) = 2*x1 + 3*x2 on [0, 1]^2
        # Var(2*x1) = 4/12 = 1/3, Var(3*x2) = 9/12 = 3/4
        # Total Var = 13/12
        # S1 = 4/13 ~ 0.308, S2 = 9/13 ~ 0.692
        # ST1 = S1, ST2 = S2 (no interaction)
        obs = _observations(
            lambda pt: 2.0 * pt[0] + 3.0 * pt[1],
            base_samples=512,
            seed=42,
        )
        report = analyze_sobol(obs)
        assert report.method == "saltelli-sobol"
        est1 = report.estimate("x1", "test_output")
        est2 = report.estimate("x2", "test_output")

        expected_s1 = 4.0 / 13.0
        expected_s2 = 9.0 / 13.0

        assert est1.s1 == pytest.approx(expected_s1, abs=0.06)
        assert est2.s1 == pytest.approx(expected_s2, abs=0.06)
        assert est1.st == pytest.approx(expected_s1, abs=0.06)
        assert est2.st == pytest.approx(expected_s2, abs=0.06)
        assert est1.availability == "available"

    def test_interaction_model_shows_total_greater_than_first_order(self) -> None:
        # f(x1, x2) = x1 * x2
        # S1 < ST1 and S2 < ST2 because of the cross-term interaction
        obs = _observations(
            lambda pt: pt[0] * pt[1],
            base_samples=512,
            seed=101,
        )
        report = analyze_sobol(obs)
        est1 = report.estimate("x1", "test_output")
        est2 = report.estimate("x2", "test_output")
        assert est1.st > est1.s1
        assert est2.st > est2.s1

    def test_constant_output_handled_safely(self) -> None:
        obs = _observations(
            lambda _pt: 42.0,
            base_samples=64,
            seed=7,
        )
        report = analyze_sobol(obs)
        est = report.estimate("x1", "test_output")
        assert est.availability == "constant-output"
        assert est.s1 == 0.0
        assert est.st == 0.0

    def test_insufficient_samples_flagged(self) -> None:
        obs = _observations(
            lambda pt: pt[0],
            base_samples=8,
            seed=7,
        )
        report = analyze_sobol(obs, min_base_samples=16)
        est = report.estimate("x1", "test_output")
        assert est.availability == "insufficient-data"
        assert math.isnan(est.s1)
        assert math.isnan(est.st)

    def test_report_json_serialization(self) -> None:
        obs = _observations(
            lambda pt: 2.0 * pt[0] + pt[1],
            base_samples=64,
            seed=42,
        )
        report = analyze_sobol(obs)
        payload = report.to_json_dict()
        assert payload["schema_id"] == SOBOL_REPORT_SCHEMA_ID
        assert payload["schema_version"] == SOBOL_REPORT_SCHEMA_VERSION
        assert payload["guidance"]["adequacy"] in ("adequate", "marginal", "inadequate")
        assert len(payload["estimates"]) == 2

    def test_run_sobol_sensitivity_with_variation_plan(self) -> None:
        from shared.python.swing_sim.variation import (
            CATEGORY_LAUNCH,
            NoiseSpec,
            VariationPlan,
        )
        from shared.python.swing_sim.variation.sobol_sensitivity import (
            run_sobol_sensitivity,
        )

        plan = VariationPlan(
            mode="launch",
            noise=(
                NoiseSpec(f"{CATEGORY_LAUNCH}.ball_speed_mph", scale=1.0),
                NoiseSpec(f"{CATEGORY_LAUNCH}.launch_angle_deg", scale=1.0),
            ),
            n_runs=8,
            seed=42,
        )
        report = run_sobol_sensitivity(plan, base_samples=32, n_workers=2)
        assert len(report.estimates) > 0
        assert report.method == "saltelli-sobol"

"""
Tests for perturbation_analysis module — Monte Carlo consistency analysis.

Covers:
- Noise generation (white, pink, brown)
- Torque perturbation application
- Batch simulation runner
- Statistical summary computation
- Variability metrics
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.perturbation_analysis import (
    PerturbationConfig,
    batch_perturb_and_simulate,
    generate_noise,
    perturb_torque_coeffs,
    variability_summary,
)

# ---------------------------------------------------------------------------
# Noise generation
# ---------------------------------------------------------------------------


class TestGenerateNoise:
    """Test noise generation functions."""

    def test_white_noise_shape(self):
        noise = generate_noise("white", n_samples=100, amplitude=1.0, seed=42)
        assert noise.shape == (100,)

    def test_white_noise_amplitude(self):
        noise = generate_noise("white", n_samples=10000, amplitude=0.5, seed=42)
        # RMS should be approximately amplitude for uniform noise
        assert np.std(noise) == pytest.approx(0.5, abs=0.1)

    def test_pink_noise_shape(self):
        noise = generate_noise("pink", n_samples=100, amplitude=1.0, seed=42)
        assert noise.shape == (100,)

    def test_brown_noise_shape(self):
        noise = generate_noise("brown", n_samples=100, amplitude=1.0, seed=42)
        assert noise.shape == (100,)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown noise type"):
            generate_noise("unknown", n_samples=100, amplitude=1.0)

    def test_seed_reproducibility(self):
        n1 = generate_noise("white", n_samples=50, amplitude=1.0, seed=123)
        n2 = generate_noise("white", n_samples=50, amplitude=1.0, seed=123)
        np.testing.assert_array_equal(n1, n2)

    def test_different_seeds_differ(self):
        n1 = generate_noise("white", n_samples=50, amplitude=1.0, seed=1)
        n2 = generate_noise("white", n_samples=50, amplitude=1.0, seed=2)
        assert not np.array_equal(n1, n2)


# ---------------------------------------------------------------------------
# Torque coefficient perturbation
# ---------------------------------------------------------------------------


class TestPerturbTorqueCoeffs:
    """Test perturbation of polynomial torque coefficients."""

    def test_preserves_shape(self):
        coeffs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        perturbed = perturb_torque_coeffs(
            coeffs, noise_amplitude=0.1, noise_type="white", seed=42
        )
        assert len(perturbed) == 2
        assert len(perturbed[0]) == 3
        assert len(perturbed[1]) == 3

    def test_zero_amplitude_no_change(self):
        coeffs = [[1.0, 2.0], [3.0, 4.0]]
        perturbed = perturb_torque_coeffs(
            coeffs, noise_amplitude=0.0, noise_type="white", seed=42
        )
        np.testing.assert_allclose(perturbed, coeffs)

    def test_perturbation_changes_values(self):
        coeffs = [[1.0, 2.0, 3.0]]
        perturbed = perturb_torque_coeffs(
            coeffs, noise_amplitude=0.5, noise_type="white", seed=42
        )
        assert not np.allclose(perturbed, coeffs)

    def test_seed_reproducibility(self):
        coeffs = [[1.0, 2.0], [3.0, 4.0]]
        p1 = perturb_torque_coeffs(coeffs, 0.1, "white", seed=99)
        p2 = perturb_torque_coeffs(coeffs, 0.1, "white", seed=99)
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# PerturbationConfig
# ---------------------------------------------------------------------------


class TestPerturbationConfig:
    """Test configuration dataclass."""

    def test_defaults(self):
        cfg = PerturbationConfig()
        assert cfg.n_trials == 100
        assert cfg.noise_type == "white"
        assert cfg.noise_amplitude == 0.1
        assert cfg.seed is None

    def test_custom(self):
        cfg = PerturbationConfig(n_trials=50, noise_type="pink", noise_amplitude=0.2, seed=42)
        assert cfg.n_trials == 50
        assert cfg.noise_type == "pink"

    def test_invalid_n_trials(self):
        with pytest.raises(AssertionError):
            PerturbationConfig(n_trials=0)

    def test_invalid_amplitude(self):
        with pytest.raises(AssertionError):
            PerturbationConfig(noise_amplitude=-0.1)


# ---------------------------------------------------------------------------
# Variability summary
# ---------------------------------------------------------------------------


class TestVariabilitySummary:
    """Test statistical summary of batch results."""

    def test_summary_keys(self):
        # Fake batch results: list of dicts with 'tip_speed_final', 'tip_position_final'
        results = [
            {
                "tip_speed_final": 30.0 + i * 0.1,
                "tip_position_final": np.array([1.0 + i * 0.01, -0.5]),
            }
            for i in range(10)
        ]
        summary = variability_summary(results)
        assert "tip_speed_mean" in summary
        assert "tip_speed_std" in summary
        assert "tip_speed_cv" in summary
        assert "tip_position_mean" in summary
        assert "tip_position_std" in summary

    def test_zero_variance(self):
        """Identical results should give zero std."""
        results = [
            {"tip_speed_final": 30.0, "tip_position_final": np.array([1.0, -0.5])}
            for _ in range(10)
        ]
        summary = variability_summary(results)
        assert summary["tip_speed_std"] == pytest.approx(0.0)

    def test_cv_computation(self):
        """Coefficient of variation = std / mean."""
        results = [
            {"tip_speed_final": v, "tip_position_final": np.array([0.0, 0.0])}
            for v in [10.0, 10.0, 10.0, 10.0, 20.0]
        ]
        summary = variability_summary(results)
        speeds = [10.0, 10.0, 10.0, 10.0, 20.0]
        expected_cv = np.std(speeds) / np.mean(speeds)
        assert summary["tip_speed_cv"] == pytest.approx(expected_cv, rel=1e-6)


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------


class TestBatchPerturbAndSimulate:
    """Test the batch perturbation runner."""

    def test_runs_n_trials(self):
        """Should return one result per trial."""
        base_coeffs = [[1.0, 2.0], [3.0, 4.0]]
        config = PerturbationConfig(n_trials=5, noise_amplitude=0.1, seed=42)

        def simulate_fn(coeffs):
            return {"coeffs": coeffs}

        def extract_fn(result):
            return {
                "tip_speed_final": 30.0,
                "tip_position_final": np.array([1.0, -0.5]),
            }

        results = batch_perturb_and_simulate(base_coeffs, config, simulate_fn, extract_fn)
        assert len(results) == 5

    def test_handles_failures_gracefully(self):
        """Failed trials should be skipped."""
        base_coeffs = [[1.0]]
        config = PerturbationConfig(n_trials=3, noise_amplitude=0.1, seed=42)

        call_count = 0

        def simulate_fn(coeffs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Sim failed")
            return {"ok": True}

        def extract_fn(result):
            return {
                "tip_speed_final": 25.0,
                "tip_position_final": np.array([0.0, 0.0]),
            }

        results = batch_perturb_and_simulate(base_coeffs, config, simulate_fn, extract_fn)
        assert len(results) == 2  # 3 trials, 1 failed

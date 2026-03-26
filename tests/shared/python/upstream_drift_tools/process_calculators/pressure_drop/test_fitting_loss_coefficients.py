"""Comprehensive tests for fitting_loss_coefficients module.

Tests cover get_fitting_k_factor, get_multiple_fittings_k,
k_to_equivalent_length, equivalent_length_to_k,
calculate_two_k_factor, calculate_fitting_pressure_drop,
list_available_fittings, and database integrity.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.pressure_drop_calculator.utils.fitting_loss_coefficients import (
    FITTING_K_FACTORS,
    TWO_K_COEFFICIENTS,
    calculate_fitting_pressure_drop,
    calculate_two_k_factor,
    equivalent_length_to_k,
    get_fitting_k_factor,
    get_multiple_fittings_k,
    k_to_equivalent_length,
    list_available_fittings,
)

# ─── get_fitting_k_factor ────────────────────────────────────


class TestGetFittingKFactor:
    def test_known_fitting(self) -> None:
        k = get_fitting_k_factor("90_elbow_std")
        assert k == 0.75

    def test_gate_valve(self) -> None:
        k = get_fitting_k_factor("gate_valve_open")
        assert k == 0.15

    def test_exit_always_one(self) -> None:
        assert get_fitting_k_factor("exit_sharp") == 1.0
        assert get_fitting_k_factor("exit_rounded") == 1.0
        assert get_fitting_k_factor("exit_submerged") == 1.0

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_fitting_k_factor("nonexistent_fitting")

    def test_error_lists_available(self) -> None:
        with pytest.raises(ValueError, match="Available types"):
            get_fitting_k_factor("fake")


# ─── get_multiple_fittings_k ─────────────────────────────────


class TestGetMultipleFittingsK:
    def test_single_fitting(self) -> None:
        total = get_multiple_fittings_k({"90_elbow_std": 1})
        assert total == 0.75

    def test_multiple_same(self) -> None:
        total = get_multiple_fittings_k({"90_elbow_std": 4})
        assert abs(total - 4 * 0.75) < 1e-10

    def test_mixed_fittings(self) -> None:
        fittings = {
            "90_elbow_std": 2,
            "gate_valve_open": 1,
        }
        expected = 2 * 0.75 + 1 * 0.15
        total = get_multiple_fittings_k(fittings)
        assert abs(total - expected) < 1e-10

    def test_empty_returns_zero(self) -> None:
        total = get_multiple_fittings_k({})
        assert total == 0.0


# ─── k_to_equivalent_length ──────────────────────────────────


class TestKToEquivalentLength:
    def test_basic(self) -> None:
        ld = k_to_equivalent_length(0.6, 0.02)
        assert abs(ld - 30.0) < 1e-10

    def test_zero_k(self) -> None:
        ld = k_to_equivalent_length(0.0, 0.02)
        assert ld == 0.0

    def test_zero_friction_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            k_to_equivalent_length(0.6, 0.0)

    def test_negative_friction_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            k_to_equivalent_length(0.6, -0.01)


# ─── equivalent_length_to_k ──────────────────────────────────


class TestEquivalentLengthToK:
    def test_basic(self) -> None:
        k = equivalent_length_to_k(30.0, 0.02)
        assert abs(k - 0.6) < 1e-10

    def test_round_trip(self) -> None:
        k_original = 0.75
        f = 0.018
        ld = k_to_equivalent_length(k_original, f)
        k_back = equivalent_length_to_k(ld, f)
        assert abs(k_back - k_original) < 1e-10

    def test_zero_friction_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            equivalent_length_to_k(30.0, 0.0)


# ─── calculate_two_k_factor ──────────────────────────────────


class TestCalculateTwoKFactor:
    def test_known_fitting(self) -> None:
        k = calculate_two_k_factor("90_elbow_std_2k", 50000, 4.0)
        assert k > 0.0

    def test_higher_re_lower_k(self) -> None:
        k_low_re = calculate_two_k_factor("90_elbow_std_2k", 1000, 4.0)
        k_high_re = calculate_two_k_factor("90_elbow_std_2k", 100000, 4.0)
        assert k_low_re > k_high_re, "Higher Re → lower total K"

    def test_smaller_pipe_higher_k(self) -> None:
        k_small = calculate_two_k_factor("90_elbow_std_2k", 50000, 1.0)
        k_large = calculate_two_k_factor("90_elbow_std_2k", 50000, 12.0)
        assert k_small > k_large, "Smaller pipe → higher K"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not in Two-K"):
            calculate_two_k_factor("nonexistent_2k", 50000, 4.0)

    def test_exit_always_one(self) -> None:
        # Exit has K1=0 and K∞=1.0 with Kd=0, so K = 0/Re + 1.0*(1+0) = 1.0
        k = calculate_two_k_factor("exit_2k", 50000, 4.0)
        assert abs(k - 1.0) < 0.01


# ─── calculate_fitting_pressure_drop ─────────────────────────


class TestCalculateFittingPressureDrop:
    def test_basic_formula(self) -> None:
        # ΔP = K × (ρV²/2)
        dp = calculate_fitting_pressure_drop(k_factor=1.0, density=1.2, velocity=10.0)
        expected = 1.0 * 0.5 * 1.2 * 100.0  # 60.0
        assert abs(dp - expected) < 1e-10

    def test_zero_velocity_zero_drop(self) -> None:
        dp = calculate_fitting_pressure_drop(
            k_factor=0.75, density=1000.0, velocity=0.0
        )
        assert dp == 0.0

    def test_proportional_to_k(self) -> None:
        dp1 = calculate_fitting_pressure_drop(1.0, 1.2, 10.0)
        dp2 = calculate_fitting_pressure_drop(2.0, 1.2, 10.0)
        assert abs(dp2 / dp1 - 2.0) < 1e-10

    def test_proportional_to_velocity_squared(self) -> None:
        dp1 = calculate_fitting_pressure_drop(0.75, 1.2, 5.0)
        dp2 = calculate_fitting_pressure_drop(0.75, 1.2, 10.0)
        assert abs(dp2 / dp1 - 4.0) < 1e-10


# ─── list_available_fittings ─────────────────────────────────


class TestListAvailableFittings:
    def test_returns_dict(self) -> None:
        fittings = list_available_fittings()
        assert isinstance(fittings, dict)

    def test_returns_copy(self) -> None:
        f1 = list_available_fittings()
        f1["fake"] = 999.0
        f2 = list_available_fittings()
        assert "fake" not in f2

    def test_contains_common_fittings(self) -> None:
        fittings = list_available_fittings()
        assert "90_elbow_std" in fittings
        assert "gate_valve_open" in fittings
        assert "exit_sharp" in fittings


# ─── Database Integrity ──────────────────────────────────────


class TestDatabaseIntegrity:
    def test_all_k_factors_non_negative(self) -> None:
        for name, k in FITTING_K_FACTORS.items():
            assert k >= 0.0, f"K-factor for {name} must be non-negative"

    def test_database_has_elbows_tees_valves(self) -> None:
        names = list(FITTING_K_FACTORS.keys())
        assert any("elbow" in n for n in names)
        assert any("tee" in n for n in names)
        assert any("valve" in n for n in names)

    def test_two_k_coefficients_non_negative(self) -> None:
        for name, (k1, kinf, kd) in TWO_K_COEFFICIENTS.items():
            assert k1 >= 0.0, f"K1 for {name} must be non-negative"
            assert kinf >= 0.0, f"K_inf for {name} must be non-negative"
            assert kd >= 0.0, f"Kd for {name} must be non-negative"

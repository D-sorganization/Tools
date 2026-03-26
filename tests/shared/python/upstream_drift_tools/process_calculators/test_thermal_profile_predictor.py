"""Comprehensive tests for thermal_profile_predictor module.

Tests cover the ODE-based temperature prediction and curve fitting.
"""

from __future__ import annotations

import numpy as np
from upstream_drift_tools.process_calculators.thermal_profile_predictor import (
    fit_heating_parameters,
    predict_temperature_profile,
)

# ─── predict_temperature_profile ─────────────────────────────


class TestPredictTemperatureProfile:
    def test_returns_arrays(self) -> None:
        t, T = predict_temperature_profile(
            t_span=(0, 100),
            t_eval=np.linspace(0, 100, 50),
            initial_temp=25.0,
            thermal_mass=1000.0,
            heat_loss_coeff=10.0,
            ambient_temp=25.0,
            power_func=lambda t: 5000.0,
        )
        assert isinstance(t, np.ndarray)
        assert isinstance(T, np.ndarray)

    def test_initial_temp_matches(self) -> None:
        _, T = predict_temperature_profile(
            t_span=(0, 100),
            t_eval=np.linspace(0, 100, 50),
            initial_temp=25.0,
            thermal_mass=1000.0,
            heat_loss_coeff=10.0,
            ambient_temp=25.0,
            power_func=lambda t: 5000.0,
        )
        assert abs(T[0] - 25.0) < 0.1

    def test_heating_increases_temp(self) -> None:
        _, T = predict_temperature_profile(
            t_span=(0, 1000),
            t_eval=np.linspace(0, 1000, 100),
            initial_temp=25.0,
            thermal_mass=1000.0,
            heat_loss_coeff=5.0,
            ambient_temp=25.0,
            power_func=lambda t: 5000.0,
        )
        assert T[-1] > T[0], "Positive power should increase temperature"

    def test_zero_power_no_change(self) -> None:
        _, T = predict_temperature_profile(
            t_span=(0, 100),
            t_eval=np.linspace(0, 100, 50),
            initial_temp=25.0,
            thermal_mass=1000.0,
            heat_loss_coeff=10.0,
            ambient_temp=25.0,
            power_func=lambda t: 0.0,
        )
        # At ambient temp with zero power, should stay at ambient
        assert abs(T[-1] - 25.0) < 0.1

    def test_approaches_steady_state(self) -> None:
        # Steady state: Q_in = h*(T_ss - T_amb) => T_ss = T_amb + Q_in/h
        q_in = 5000.0
        h = 10.0
        t_amb = 25.0
        expected_ss = t_amb + q_in / h  # 525 °C

        _, T = predict_temperature_profile(
            t_span=(0, 100000),
            t_eval=np.linspace(0, 100000, 500),
            initial_temp=25.0,
            thermal_mass=1000.0,
            heat_loss_coeff=h,
            ambient_temp=t_amb,
            power_func=lambda t: q_in,
        )
        assert abs(T[-1] - expected_ss) < 1.0, f"Expected ~{expected_ss}, got {T[-1]}"

    def test_higher_power_higher_temp(self) -> None:
        def run(power: float) -> float:
            _, T = predict_temperature_profile(
                t_span=(0, 5000),
                t_eval=np.linspace(0, 5000, 100),
                initial_temp=25.0,
                thermal_mass=1000.0,
                heat_loss_coeff=10.0,
                ambient_temp=25.0,
                power_func=lambda t: power,
            )
            return T[-1]

        t_low = run(1000.0)
        t_high = run(10000.0)
        assert t_high > t_low

    def test_higher_heat_loss_lower_temp(self) -> None:
        def run(h: float) -> float:
            _, T = predict_temperature_profile(
                t_span=(0, 5000),
                t_eval=np.linspace(0, 5000, 100),
                initial_temp=25.0,
                thermal_mass=1000.0,
                heat_loss_coeff=h,
                ambient_temp=25.0,
                power_func=lambda t: 5000.0,
            )
            return T[-1]

        t_low_h = run(5.0)
        t_high_h = run(50.0)
        assert t_low_h > t_high_h


# ─── fit_heating_parameters ──────────────────────────────────


class TestFitHeatingParameters:
    def test_recovers_known_parameters(self) -> None:
        # Generate synthetic data with known parameters
        true_tm = 1000.0
        true_h = 10.0
        t_amb = 25.0
        power = 5000.0

        t_eval = np.linspace(0, 5000, 50)
        _, observed = predict_temperature_profile(
            t_span=(0, 5000),
            t_eval=t_eval,
            initial_temp=25.0,
            thermal_mass=true_tm,
            heat_loss_coeff=true_h,
            ambient_temp=t_amb,
            power_func=lambda t: power,
        )

        fitted_tm, fitted_h = fit_heating_parameters(
            times=t_eval,
            observed_temps=observed,
            initial_temp=25.0,
            thermal_mass_guess=800.0,
            heat_loss_guess=8.0,
            ambient_temp=t_amb,
            power_func=lambda t: power,
        )

        assert abs(fitted_tm - true_tm) / true_tm < 0.1, f"Expected ~{true_tm}, got {fitted_tm}"
        assert abs(fitted_h - true_h) / true_h < 0.1, f"Expected ~{true_h}, got {fitted_h}"

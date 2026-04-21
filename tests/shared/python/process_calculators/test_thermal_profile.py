"""Tests for upstream_drift_tools.process_calculators.thermal_profile_predictor.

Covers:
- predict_temperature_profile: basic heating, steady state convergence
- fit_heating_parameters: parameter recovery from generated data
"""

from __future__ import annotations

import numpy as np
import pytest
from upstream_drift_tools.process_calculators.thermal_profile_predictor import (
    fit_heating_parameters,
    predict_temperature_profile,
)


class TestPredictTemperatureProfile:
    def test_constant_power_heats_up(self) -> None:
        """With constant power, T should rise from initial."""
        t_eval = np.linspace(0, 100, 200)
        _, temps = predict_temperature_profile(
            t_span=(0, 100),
            t_eval=t_eval,
            initial_temp=300.0,
            thermal_mass=100.0,
            heat_loss_coeff=1.0,
            ambient_temp=300.0,
            power_func=lambda t: 500.0,  # constant 500 W
        )
        # Temperature should be above initial
        assert temps[-1] > 300.0

    def test_no_power_approaches_ambient(self) -> None:
        """With no power and initial above ambient, T should cool down."""
        t_eval = np.linspace(0, 200, 300)
        _, temps = predict_temperature_profile(
            t_span=(0, 200),
            t_eval=t_eval,
            initial_temp=500.0,
            thermal_mass=50.0,
            heat_loss_coeff=2.0,
            ambient_temp=300.0,
            power_func=lambda t: 0.0,
        )
        # Should cool toward ambient
        assert temps[-1] < 500.0
        assert temps[-1] == pytest.approx(300.0, abs=5.0)

    def test_steady_state(self) -> None:
        """At steady state, dT/dt = 0 => Q_in = h*(T-T_amb).

        For Q=100, h=2, T_amb=300: T_ss = 300 + 100/2 = 350.
        """
        t_eval = np.linspace(0, 500, 500)
        _, temps = predict_temperature_profile(
            t_span=(0, 500),
            t_eval=t_eval,
            initial_temp=300.0,
            thermal_mass=50.0,
            heat_loss_coeff=2.0,
            ambient_temp=300.0,
            power_func=lambda t: 100.0,
        )
        assert temps[-1] == pytest.approx(350.0, abs=1.0)

    def test_output_shapes(self) -> None:
        t_eval = np.linspace(0, 10, 50)
        t_out, temps = predict_temperature_profile(
            t_span=(0, 10),
            t_eval=t_eval,
            initial_temp=300.0,
            thermal_mass=100.0,
            heat_loss_coeff=1.0,
            ambient_temp=300.0,
            power_func=lambda t: 0.0,
        )
        assert len(t_out) == 50
        assert len(temps) == 50


class TestFitHeatingParameters:
    def test_recover_known_parameters(self) -> None:
        """Generate data with known params, then fit and recover them."""
        true_thermal_mass = 80.0
        true_heat_loss = 1.5
        ambient = 300.0
        power = 200.0

        t_eval = np.linspace(0, 300, 100)
        _, observed = predict_temperature_profile(
            t_span=(0, 300),
            t_eval=t_eval,
            initial_temp=300.0,
            thermal_mass=true_thermal_mass,
            heat_loss_coeff=true_heat_loss,
            ambient_temp=ambient,
            power_func=lambda t: power,
        )

        fitted_tm, fitted_hl = fit_heating_parameters(
            times=t_eval,
            observed_temps=observed,
            initial_temp=300.0,
            thermal_mass_guess=50.0,
            heat_loss_guess=1.0,
            ambient_temp=ambient,
            power_func=lambda t: power,
        )

        assert fitted_tm == pytest.approx(true_thermal_mass, rel=0.1)
        assert fitted_hl == pytest.approx(true_heat_loss, rel=0.1)

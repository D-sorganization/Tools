"""Tests for upstream_drift_tools.process_calculators.thermal_profile_predictor.

Covers:
- predict_temperature_profile: basic heating, steady state convergence
- fit_heating_parameters: parameter recovery from generated data
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
import numpy as np
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


def _step_cutoff_closed_form(
    t: np.ndarray,
    *,
    initial_temp: float,
    ambient_temp: float,
    thermal_mass: float,
    heat_loss_coeff: float,
    power: float,
    cutoff_time: float,
) -> np.ndarray:
    """Exact lumped-capacitance response to power switched off at ``cutoff_time``.

    Heating (t < t_c): T = T_amb + (T0 - T_amb) e^{-t/tau} + (P/h)(1 - e^{-t/tau})
    Cooling (t >= t_c): T = T_amb + (T(t_c) - T_amb) e^{-(t - t_c)/tau}
    with tau = m c / h.
    """
    tau = thermal_mass / heat_loss_coeff
    rise = power / heat_loss_coeff

    def heating(time: np.ndarray | float) -> np.ndarray:
        decay = np.exp(-np.asarray(time, dtype=float) / tau)
        return ambient_temp + (initial_temp - ambient_temp) * decay + rise * (1 - decay)

    temp_at_cutoff = heating(cutoff_time)
    cooling = ambient_temp + (temp_at_cutoff - ambient_temp) * np.exp(
        -(t - cutoff_time) / tau
    )
    return np.where(t < cutoff_time, heating(t), cooling)


class TestStepPowerCutoff:
    """Issue #5315: a power discontinuity must not leak integration error."""

    TOLERANCE_DEGC = 0.1

    @pytest.mark.parametrize(
        ("cutoff_time", "num_points"),
        [
            (500.0, 21),  # cutoff on a sample time (parity fixture step_power_cutoff)
            (537.3, 21),  # cutoff between sample times
            (550.0, 201),  # dense sampling
            (1999.0, 2),  # end points only, cutoff just before t_end
        ],
    )
    def test_matches_closed_form_across_cutoff(
        self, cutoff_time: float, num_points: int
    ) -> None:
        params = {
            "initial_temp": 20.0,
            "ambient_temp": 20.0,
            "thermal_mass": 10_000.0,
            "heat_loss_coeff": 10.0,
        }
        power = 1000.0
        t_eval = np.linspace(0.0, 2000.0, num_points)

        times, temps = predict_temperature_profile(
            t_span=(0.0, 2000.0),
            t_eval=t_eval,
            power_func=lambda t: power if t < cutoff_time else 0.0,
            **params,
        )

        expected = _step_cutoff_closed_form(
            t_eval, power=power, cutoff_time=cutoff_time, **params
        )
        np.testing.assert_allclose(times, t_eval)
        worst = float(np.max(np.abs(np.asarray(temps) - expected)))
        assert worst <= self.TOLERANCE_DEGC, (
            f"|model - closed form| = {worst:.4f} degC "
            f"(cutoff {cutoff_time} s, {num_points} samples)"
        )

    @pytest.mark.parametrize("thermal_mass", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_positive_thermal_mass(self, thermal_mass: float) -> None:
        with pytest.raises(ValueError, match="thermal_mass"):
            predict_temperature_profile(
                t_span=(0.0, 10.0),
                t_eval=[0.0, 10.0],
                initial_temp=20.0,
                thermal_mass=thermal_mass,
                heat_loss_coeff=1.0,
                ambient_temp=20.0,
                power_func=lambda t: 0.0,
            )

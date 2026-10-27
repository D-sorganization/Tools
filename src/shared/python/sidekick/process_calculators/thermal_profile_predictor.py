"""thermal_profile_predictor.py module."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

__all__ = [
    "fit_heating_parameters",
    "predict_temperature_profile",
]

# Integrator tolerances (issue #5315). ``power_func`` is an opaque callable, so
# power steps cannot be passed to the solver as breakpoints; with SciPy's
# default ``rtol=1e-3`` RK45 accepts a step straddling a power cutoff and the
# error (0.3-0.8 degC on a 100 degC rise) persists downstream. At these
# tolerances the step controller rejects steps across the discontinuity until
# the local error is below ~0.01 degC.
_RTOL = 1e-6
_ATOL = 1e-8


def _heating_ode(
    t: float,
    y: Sequence[float],
    thermal_mass: float,
    heat_loss_coeff: float,
    ambient_temp: float,
    power_func: Callable[[float], float],
) -> Sequence[float]:
    """ODE for simple vessel heating."""
    if t is None:
        raise ValueError("t must be provided")
    q_in = power_func(t)
    dTdt = (q_in - heat_loss_coeff * (y[0] - ambient_temp)) / thermal_mass
    return [dTdt]


def predict_temperature_profile(
    t_span: tuple[float, float],
    t_eval: Sequence[float],
    initial_temp: float,
    thermal_mass: float,
    heat_loss_coeff: float,
    ambient_temp: float,
    power_func: Callable[[float], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict temperature profile for a heated vessel."""

    if t_span is None:
        raise ValueError("t_span must be provided")
    if not np.isfinite(thermal_mass) or thermal_mass <= 0:
        raise ValueError(
            f"thermal_mass must be positive and finite, got {thermal_mass}"
        )

    def rhs(t: float, y: Any) -> Any:
        return _heating_ode(
            t, y, thermal_mass, heat_loss_coeff, ambient_temp, power_func
        )

    from scipy.integrate import solve_ivp  # lazy import

    sol = solve_ivp(
        rhs,
        t_span,
        [initial_temp],
        t_eval=t_eval,
        vectorized=False,
        rtol=_RTOL,
        atol=_ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"temperature integration failed: {sol.message}")
    return sol.t, sol.y[0]


def fit_heating_parameters(
    times: Sequence[float],
    observed_temps: Sequence[float],
    initial_temp: float,
    thermal_mass_guess: float,
    heat_loss_guess: float,
    ambient_temp: float,
    power_func: Callable[[float], float],
) -> tuple[float, float]:
    """Fit thermal_mass and heat_loss_coeff to observed data.

    Args:
        times: Sequence of observation times in seconds.
        observed_temps: Sequence of observed temperatures in Celsius.
        initial_temp: Initial vessel temperature in Celsius.
        thermal_mass_guess: Initial guess for thermal mass in J/K (must be > 0).
        heat_loss_guess: Initial guess for heat loss in W/K (must be >= 0).
        ambient_temp: Ambient temperature in Celsius.
        power_func: Time-dependent heating power function in Watts.

    Returns:
        Fitted (thermal_mass, heat_loss_coeff) tuple.
    """
    if times is None:
        raise ValueError("times must be provided")
    if observed_temps is None:
        raise ValueError("observed_temps must be provided")
    if len(times) != len(observed_temps):
        raise ValueError(
            f"times and observed_temps must have equal length, "
            f"got {len(times)} and {len(observed_temps)}"
        )
    if not np.isfinite(thermal_mass_guess) or thermal_mass_guess <= 0:
        raise ValueError(
            f"thermal_mass_guess must be positive and finite, got {thermal_mass_guess}"
        )
    if not np.isfinite(heat_loss_guess) or heat_loss_guess < 0:
        raise ValueError(
            f"heat_loss_guess must be non-negative and finite, got {heat_loss_guess}"
        )

    def model(t: Any, thermal_mass: float, heat_loss_coeff: float) -> np.ndarray:
        """Model evaluation for curve_fit."""
        if thermal_mass is None:
            raise ValueError("thermal_mass must be provided")
        _, temps = predict_temperature_profile(
            (t[0], t[-1]),
            t,
            initial_temp,
            thermal_mass,
            heat_loss_coeff,
            ambient_temp,
            power_func,
        )
        return temps

    from scipy.optimize import curve_fit  # lazy import

    popt, _ = curve_fit(
        model,
        np.asarray(times),
        np.asarray(observed_temps),
        p0=[thermal_mass_guess, heat_loss_guess],
        bounds=([1e-6, 0.0], [np.inf, np.inf]),
    )
    return float(popt[0]), float(popt[1])

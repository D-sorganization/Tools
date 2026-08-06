"""Auditable timestep-refinement studies for reduced turf contact."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ._validation import require_finite_float
from .turf_contact import TurfContactProfile, TurfContactStatus
from .turf_dynamics import ReducedTurfContactResult, simulate_reduced_turf_contact


@dataclass(frozen=True)
class TurfConvergenceStudy:
    """Coarse-to-fine results and finest-pair relative changes."""

    time_steps_s: tuple[float, ...]
    results: tuple[ReducedTurfContactResult, ...]
    impulse_relative_change: float
    peak_penetration_relative_change: float
    dissipated_energy_relative_change: float
    tolerance: float
    converged: bool


def _relative_change(coarse: float, fine: float) -> float:
    scale = max(abs(fine), 1e-15)
    return abs(coarse - fine) / scale


def run_turf_convergence_study(
    profile: TurfContactProfile,
    *,
    initial_contact_velocity_mps: object,
    surface_normal_unit: object,
    effective_mass_kg: float,
    time_steps_s: tuple[float, ...] = (2e-5, 1e-5, 5e-6),
    tolerance: float = 0.02,
    max_time_s: float = 0.1,
    cancel_check: Callable[[], bool] | None = None,
) -> TurfConvergenceStudy:
    """Run a declared coarse-to-fine refinement sequence and report convergence."""
    if not isinstance(profile, TurfContactProfile):
        raise TypeError("profile must be TurfContactProfile")
    if len(time_steps_s) < 2:
        raise ValueError("time_steps_s must contain at least two values")
    steps = tuple(
        require_finite_float(value, "time_steps_s", positive=True)
        for value in time_steps_s
    )
    if any(not steps[index] > steps[index + 1] for index in range(len(steps) - 1)):
        raise ValueError("time_steps_s must be strictly decreasing from coarse to fine")
    relative_tolerance = require_finite_float(tolerance, "tolerance", positive=True)
    if relative_tolerance > 1.0:
        raise ValueError("tolerance must be <= 1")
    maximum_time = require_finite_float(max_time_s, "max_time_s", positive=True)
    results = tuple(
        simulate_reduced_turf_contact(
            profile,
            initial_contact_velocity_mps=initial_contact_velocity_mps,
            surface_normal_unit=surface_normal_unit,
            effective_mass_kg=effective_mass_kg,
            time_step_s=step,
            max_time_s=maximum_time,
            cancel_check=cancel_check,
        )
        for step in steps
    )
    coarse, fine = results[-2:]
    impulse_change = _relative_change(
        coarse.normal_impulse_n_s, fine.normal_impulse_n_s
    )
    penetration_change = _relative_change(
        coarse.peak_penetration_m, fine.peak_penetration_m
    )
    energy_change = _relative_change(
        coarse.dissipated_energy_j, fine.dissipated_energy_j
    )
    complete = all(result.status is TurfContactStatus.SEPARATED for result in results)
    converged = complete and all(
        math.isfinite(value) and value <= relative_tolerance
        for value in (impulse_change, penetration_change, energy_change)
    )
    return TurfConvergenceStudy(
        time_steps_s=steps,
        results=results,
        impulse_relative_change=impulse_change,
        peak_penetration_relative_change=penetration_change,
        dissipated_energy_relative_change=energy_change,
        tolerance=relative_tolerance,
        converged=converged,
    )


__all__ = ["TurfConvergenceStudy", "run_turf_convergence_study"]

"""Temperature sweep and surface sweep strategies.

SRP: Iterates over parameter spaces, delegates each point to the engine.
     No feed processing, no optimization, no plotting.
"""

from typing import Any

import numpy as np


def temperature_sweep(
    engine: Any, t_start: float, t_end: float, n_points: int, **solve_kwargs: Any
) -> list[Any]:
    """Run equilibrium across a temperature range with warm-starting.

    Args:
        engine: object with .solve(temperature=, ...) method
        t_start, t_end: Temperature range [K]
        n_points: Number of points
        **solve_kwargs: Passed through to engine.solve()

    Returns:
        list of EquilibriumResult

    Precondition: t_start > 0, t_end > t_start, n_points >= 2
    """
    assert t_start > 0 and t_end > t_start, "Invalid temperature range"
    assert n_points >= 2, "Need at least 2 points"

    temperatures = np.linspace(t_start, t_end, n_points)
    results: list[Any] = []
    warm = None

    for t in temperatures:
        r = engine.solve(temperature=float(t), warm_start=warm, **solve_kwargs)
        results.append(r)
        if r.converged:
            warm = r.moles.copy()

    return results


def surface_sweep(
    engine: Any,
    t_range: tuple[float, float],
    param_name: str,
    param_range: tuple[float, float],
    n_t: int = 30,
    n_param: int = 25,
    **solve_kwargs: Any,
) -> dict[str, Any]:
    """Run 2D parameter sweep for surface plots.

    Args:
        engine: object with .solve() method
        t_range: (t_start, t_end) in Kelvin
        param_name: keyword argument name to vary
        param_range: (start, end) for that parameter
        n_t, n_param: grid resolution
        **solve_kwargs: base kwargs for engine.solve()

    Returns:
        SurfaceData dict

    Precondition: param_name is a valid kwarg for engine.solve()
    """
    temperatures = np.linspace(t_range[0], t_range[1], n_t)
    param_values = np.linspace(param_range[0], param_range[1], n_param)

    species_keys = engine.species_keys
    n_species = len(species_keys)

    compositions = np.zeros((n_t, n_param, n_species))
    h2_co = np.zeros((n_t, n_param))
    carbon_conv = np.zeros((n_t, n_param))
    cge = np.zeros((n_t, n_param))

    for j, pval in enumerate(param_values):
        warm = None
        for i, t in enumerate(temperatures):
            kwargs = dict(solve_kwargs)
            kwargs["temperature"] = float(t)
            kwargs[param_name] = pval
            kwargs["warm_start"] = warm

            r = engine.solve(**kwargs)

            compositions[i, j, :] = r.mole_fractions
            h2_co[i, j] = r.h2_co_ratio
            carbon_conv[i, j] = r.carbon_conversion
            cge[i, j] = r.cold_gas_efficiency

            if r.converged:
                warm = r.moles.copy()

    return {
        "temperatures": temperatures,
        "param_values": param_values,
        "param_name": param_name,
        "species": species_keys,
        "compositions": compositions,
        "h2_co_ratio": h2_co,
        "carbon_conversion": carbon_conv,
        "cge": cge,
    }

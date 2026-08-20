"""Pure calculation-basis formatting for wind-strategy results."""

from __future__ import annotations

import math

from shared.python.swing_sim.flight import (
    StrategyAnalysisRequest,
    WindStrategyAnalysis,
)


def format_wind_strategy_basis(
    request: StrategyAnalysisRequest,
    result: WindStrategyAnalysis,
) -> str:
    """Describe every captured numerical and policy factor behind a result."""
    analysis = request.analysis
    uncertainty = request.uncertainty
    target = request.target
    policies = ", ".join(
        f"{strategy.label} "
        f"{math.degrees(strategy.crosswind_aim_gain_rad_per_mps):+.3f} deg/(m/s)"
        for strategy in request.strategies
    )
    return (
        f"Calculation basis ({result.schema_version}; {result.provenance}): "
        f"model {analysis.model_name}; {uncertainty.trials} paired trials; "
        f"seed {uncertainty.seed}; target {target.forward_m:+.3f} m forward, "
        f"{target.right_m:+.3f} m right; hold radius "
        f"{analysis.target_radius_m:.3f} m; maximum time "
        f"{analysis.max_time_s:g} s; time step {analysis.time_step_s:g} s; "
        f"failure cost {analysis.failure_cost:g}; CVaR alpha "
        f"{analysis.miss_distance_cvar_alpha:g}; "
        f"estimated-crosswind aim policy: {policies}."
    )


__all__ = ["format_wind_strategy_basis"]

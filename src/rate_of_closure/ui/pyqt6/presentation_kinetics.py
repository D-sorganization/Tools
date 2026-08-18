"""Kinetics presentation seam for completed hits and misses."""

from __future__ import annotations

from rate_of_closure.simulation import (
    KineticsSeries,
    SimulationRun,
    compute_kinetics,
    kinetics_for_run,
)

_cached_run: SimulationRun | None = None
_cached_series: KineticsSeries | None = None


def kinetics_for_presentation(run: SimulationRun) -> KineticsSeries | None:
    """Return complete-swing kinetics with an outcome-aware timing marker.

    The persisted run stays honest: a miss still has no impact time and the
    canonical ``kinetics_for_run`` result remains ``None``. This adapter gives
    the desktop UI an explicit analysis cutoff without fabricating contact.
    """
    global _cached_run, _cached_series  # noqa: PLW0603 - intentional one-slot cache
    if _cached_run is run:
        return _cached_series
    if run.impact_time_s is not None:
        series = kinetics_for_run(run)
    else:
        series = compute_kinetics(
            run,
            analysis_time_s=run.impact_outcome.candidate_time_s,
        )
    _cached_run = run
    _cached_series = series
    return series


__all__ = ["kinetics_for_presentation"]

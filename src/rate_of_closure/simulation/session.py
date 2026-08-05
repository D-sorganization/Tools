"""Public orchestration seam for swing, contact, impact, and flight.

The default ``delivery_inspection`` mode preserves legacy behavior by
translating the swing so its selected clubhead reference point meets the
fixed ball.  Opt-in ``fixed_ball_contact`` retains the source trajectory and
returns a typed hit or miss before running contact-dependent phases.
"""

from __future__ import annotations

from rate_of_closure.simulation.contact import ContactMode, ImpactOutcome
from rate_of_closure.simulation.delivery import delivery_at
from rate_of_closure.simulation.records import (
    BALL_POSITION_M,
    SimulationConfig,
    SimulationRun,
)

__all__ = [
    "BALL_POSITION_M",
    "ContactMode",
    "ImpactOutcome",
    "SimulationConfig",
    "SimulationRun",
    "delivery_at",
    "run_simulation",
]


def run_simulation(config: SimulationConfig) -> SimulationRun:
    """Run a complete swing and any contact-dependent downstream phases."""
    from rate_of_closure.simulation.pipeline import execute_simulation

    return execute_simulation(config)

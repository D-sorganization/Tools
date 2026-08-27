"""Mechanism-vs-outcome swing optimization for the double pendulum golf model.

The simulator can already optimize a swing for clubhead speed. This subpackage
adds the ability to optimize instead for the *mechanisms* golf coaching invokes —
centrifugal release, Coriolis kinetic-chain transfer, grip-force energy transfer,
grip-force impulse — under an identical torque budget, and to compare the
resulting swings against the speed-optimal one.

It is built entirely on the existing :mod:`double_pendulum_golf.physics` kernel
and the :mod:`double_pendulum_golf.transfer_strategy` contract; no equations of
motion are re-derived here.

See ``docs/specs/SWING_OBJECTIVE_COMPARISON.md``. Epic #4766.
"""

from double_pendulum_golf.swing_objectives.velocity_terms import (
    VelocityTerms,
    centrifugal_vector,
    coriolis_only_vector,
    coupling_constant,
    decompose_velocity_terms,
)

__all__ = [
    "VelocityTerms",
    "centrifugal_vector",
    "coriolis_only_vector",
    "coupling_constant",
    "decompose_velocity_terms",
]

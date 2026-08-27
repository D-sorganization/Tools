"""The five competing objectives a golf downswing can be optimized for.

A golfer, a coach and a biomechanist each name a different thing as the point of
the downswing: hold the lag and let centrifugal force release the club; sequence
proximal-to-distal so the arms decelerate into the club; route the body's power
through the grip; pull hard on the club for as long as possible; or simply swing
the head as fast as possible. Four of those are statements about *mechanism*, one
is a statement about *outcome*.

Each is a well-defined functional of the trajectory, and all five are defined here
so the same golfer can be optimized against each in turn under an identical torque
budget. **Every objective is to be maximized.**

Centrifugal versus Coriolis
---------------------------
These two are not independent as *work*. The centrifugal power delivered to the
wrist is ``-mu*sin(phi)*dtheta1**2*dphi`` and the Coriolis power at the hub is
``+2*mu*sin(phi)*dtheta1**2*dphi``, so

.. code-block:: text

    P_coriolis_hub = -2 * P_centrifugal_wrist

identically, for every trajectory — they are one energy flow read at its two ends.
Defining both objectives as work would make them the same optimization problem
with a rescaled cost, returning identical swings.

:data:`CENTRIFUGAL` is therefore an **angular impulse**,
``integral of mu*sin(phi)*dtheta1**2 dt``. Dropping the ``dphi`` factor changes
what it rewards — sustaining a large cock angle at high arm speed, rather than
uncocking quickly — and makes it a genuinely different problem. The identity and
the resulting independence are both pinned in ``tests/test_swing_objectives.py``.

Closes #4768.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from double_pendulum_golf.swing_objectives.signals import SwingSignals

__all__ = [
    "SwingObjective",
    "SWING_OBJECTIVES",
    "get_objective",
    "evaluate_all",
    "CLUBHEAD_SPEED",
    "CENTRIFUGAL",
    "CORIOLIS",
    "ENERGY_TRANSFER",
    "IMPULSE_TRANSFER",
]

EvaluateFn = Callable[[SwingSignals], float]


@dataclass(frozen=True, slots=True)
class SwingObjective:
    """A scalar functional of a downswing trajectory, to be maximized.

    Attributes:
        key: Stable identifier used for lookup, plot labels and report keys.
        name: Human-readable objective name.
        units: Units of the raw value.
        description: What the objective rewards, in golf terms.
        scale: Order of magnitude for a tour-level swing. The optimizer divides
            by this so every objective presents a similarly conditioned problem
            to the NLP solver; it never changes the reported value.
        evaluate: Callable mapping :class:`SwingSignals` to the value to maximize.
    """

    key: str
    name: str
    units: str
    description: str
    scale: float
    evaluate: EvaluateFn


def _clubhead_speed_at_impact(signals: SwingSignals) -> float:
    """Clubhead speed at the final sample, which is impact, in m/s."""
    return float(signals.clubhead_speed[-1])


def _centrifugal_release_impulse(signals: SwingSignals) -> float:
    """Angular impulse of the centrifugal release moment at the wrist, in N·m·s."""
    return signals.integrate(signals.centrifugal_wrist_moment)


def _coriolis_chain_transfer(signals: SwingSignals) -> float:
    """Energy the Coriolis coupling drains out of the arms, in J.

    The hub power is negative while the kinetic chain is working, so the value is
    negated to make "more transfer" mean "larger objective".
    """
    return -signals.integrate(signals.coriolis_hub_power)


def _grip_force_energy(signals: SwingSignals) -> float:
    """Work the linear grip force delivers into the club, in J."""
    return signals.integrate(signals.grip_force_power)


def _grip_force_impulse(signals: SwingSignals) -> float:
    """Time integral of grip-force magnitude, in N·s."""
    return signals.integrate(signals.grip_force_magnitude)


CLUBHEAD_SPEED = SwingObjective(
    key="clubhead_speed",
    name="Clubhead speed",
    units="m/s",
    description=(
        "Maximize clubhead speed at impact. This is the outcome every other "
        "objective implicitly claims to be a route to, and the baseline the "
        "mechanism objectives are compared against."
    ),
    scale=50.0,
    evaluate=_clubhead_speed_at_impact,
)

CENTRIFUGAL = SwingObjective(
    key="centrifugal",
    name="Centrifugal release impulse",
    units="N*m*s",
    description=(
        "Maximize the angular impulse of the centrifugal moment at the wrist: "
        "hold the lag deep into the downswing so arm rotation, not wrist effort, "
        "is what flings the clubhead out."
    ),
    scale=12.0,
    evaluate=_centrifugal_release_impulse,
)

CORIOLIS = SwingObjective(
    key="coriolis",
    name="Coriolis kinetic-chain transfer",
    units="J",
    description=(
        "Maximize the energy Coriolis coupling drains out of the arms: the "
        "uncocking club should slow the arms down, because that deceleration is "
        "momentum arriving at the club."
    ),
    scale=120.0,
    evaluate=_coriolis_chain_transfer,
)

ENERGY_TRANSFER = SwingObjective(
    key="energy_transfer",
    name="Grip-force energy transfer",
    units="J",
    description=(
        "Maximize the work delivered into the club through the linear force at "
        "the hands, rewarding power routed through the grip rather than spent as "
        "wrist torque."
    ),
    scale=200.0,
    evaluate=_grip_force_energy,
)

IMPULSE_TRANSFER = SwingObjective(
    key="impulse_transfer",
    name="Grip-force impulse",
    units="N*s",
    description=(
        "Maximize the time integral of grip-force magnitude: sustained pull on "
        "the club throughout the downswing rather than a late spike."
    ),
    scale=120.0,
    evaluate=_grip_force_impulse,
)

#: Every objective, keyed by identifier. Iteration order is the order used for
#: comparison tables and plots, with the outcome baseline first.
SWING_OBJECTIVES: dict[str, SwingObjective] = {
    objective.key: objective
    for objective in (
        CLUBHEAD_SPEED,
        CENTRIFUGAL,
        CORIOLIS,
        ENERGY_TRANSFER,
        IMPULSE_TRANSFER,
    )
}


def get_objective(objective: str | SwingObjective) -> SwingObjective:
    """Resolve an objective from its key, passing instances through unchanged.

    Args:
        objective: Objective key such as ``"coriolis"``, or an objective instance.

    Returns:
        The resolved objective.

    Raises:
        KeyError: If the key is not one of the five defined objectives.

    Pre: none.
    Post: the returned objective is a member of :data:`SWING_OBJECTIVES`.
    """
    if isinstance(objective, SwingObjective):
        return objective
    if objective not in SWING_OBJECTIVES:
        raise KeyError(
            f"Unknown swing objective {objective!r}; available: {sorted(SWING_OBJECTIVES)}"
        )
    return SWING_OBJECTIVES[objective]


def evaluate_all(signals: SwingSignals) -> dict[str, float]:
    """Score a trajectory against every objective.

    Args:
        signals: Per-sample downswing signals.

    Returns:
        Mapping from objective key to its value in that objective's own units.

    Post: every value is finite.
    """
    scores = {key: obj.evaluate(signals) for key, obj in SWING_OBJECTIVES.items()}
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError(f"Objective evaluation produced non-finite values: {scores}")
    return scores

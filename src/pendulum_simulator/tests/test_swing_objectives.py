"""Contract tests for the five competing swing objectives.

The registry exists to answer one question: does optimizing a downswing for a
*mechanism* produce the same swing as optimizing it for the *outcome*? That only
means anything if the mechanism objectives are genuinely different functionals,
so the degeneracy pin below is the most important test in this file.

Closes #4768.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics import PendulumParams
from double_pendulum_golf.swing_objectives.objectives import (
    SWING_OBJECTIVES,
    SwingObjective,
    get_objective,
)
from double_pendulum_golf.swing_objectives.signals import build_swing_signals

_PARAMS = PendulumParams(m1=5.0, m2=0.30, L1=0.65, L2=1.10, mClub=0.20)
_SAMPLE_COUNT = 48

_EXPECTED_KEYS = {
    "clubhead_speed",
    "centrifugal",
    "coriolis",
    "energy_transfer",
    "impulse_transfer",
}


def _signals(seed: int = 4768, uncock_scale: float = 1.0):
    """Build signals for a downswing-like trajectory."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 0.28, _SAMPLE_COUNT)
    states = np.column_stack(
        [
            np.linspace(2.6, 0.0, _SAMPLE_COUNT),
            np.linspace(1.7, 0.0, _SAMPLE_COUNT),
            np.linspace(0.0, -18.0, _SAMPLE_COUNT),
            uncock_scale * np.linspace(0.0, -30.0, _SAMPLE_COUNT),
        ]
    )
    torques = rng.uniform(-120.0, 120.0, (_SAMPLE_COUNT, 2))
    return build_swing_signals(time, states, torques, _PARAMS)


def test_registry_exposes_exactly_the_five_named_objectives() -> None:
    """The comparison is defined over these five and no others."""
    assert set(SWING_OBJECTIVES) == _EXPECTED_KEYS
    for key, objective in SWING_OBJECTIVES.items():
        assert isinstance(objective, SwingObjective)
        assert objective.key == key
        assert objective.name
        assert objective.units
        assert objective.description
        assert objective.scale > 0.0


def test_get_objective_resolves_keys_and_rejects_unknown_ones() -> None:
    """Lookup fails closed rather than silently substituting a default."""
    assert get_objective("coriolis") is SWING_OBJECTIVES["coriolis"]
    assert get_objective(SWING_OBJECTIVES["centrifugal"]) is SWING_OBJECTIVES["centrifugal"]
    with pytest.raises(KeyError, match="Unknown swing objective"):
        get_objective("maximum_style_points")


def test_every_objective_returns_a_finite_value() -> None:
    """No objective may produce NaN or infinity on a well-formed trajectory."""
    signals = _signals()
    for objective in SWING_OBJECTIVES.values():
        value = objective.evaluate(signals)
        assert np.isfinite(value), f"{objective.key} produced {value}"


def test_clubhead_speed_objective_reads_the_impact_sample() -> None:
    """Clubhead speed is scored at impact, which is the final sample."""
    signals = _signals()
    value = SWING_OBJECTIVES["clubhead_speed"].evaluate(signals)
    assert value == pytest.approx(float(signals.clubhead_speed[-1]))


def test_coriolis_and_centrifugal_power_are_not_independent() -> None:
    """Pin the exact -2 identity that forces the centrifugal objective to be an impulse.

    Both powers reduce to ``mu*sin(phi)*dtheta1**2*dphi``. Defining the
    centrifugal objective as *work* would therefore make it the Coriolis
    objective rescaled by -2, and the two optimizations would return identical
    swings. This test is why the shipped definition is an angular impulse.
    """
    signals = _signals()
    centrifugal_power = signals.centrifugal_wrist_moment * -signals.states[:, 3]
    assert np.allclose(signals.coriolis_hub_power, -2.0 * centrifugal_power)


def test_centrifugal_objective_is_blind_to_the_uncock_rate() -> None:
    """The release impulse must not collapse back onto the Coriolis objective.

    Tripling and reversing the uncocking rate leaves the centrifugal impulse
    untouched while changing the Coriolis transfer, proving they are different
    functionals of the same trajectory.
    """
    baseline = _signals()
    perturbed = _signals(uncock_scale=-3.0)

    centrifugal = SWING_OBJECTIVES["centrifugal"]
    coriolis = SWING_OBJECTIVES["coriolis"]

    assert centrifugal.evaluate(perturbed) == pytest.approx(centrifugal.evaluate(baseline))
    assert coriolis.evaluate(perturbed) != pytest.approx(coriolis.evaluate(baseline))


def test_coriolis_objective_is_positive_when_the_chain_drains_the_arms() -> None:
    """A downswing that uncocks while the arms turn transfers energy outward.

    Sign convention: the objective is the energy Coriolis coupling removes from
    the arms, so a well-sequenced swing scores positive.
    """
    signals = _signals()
    assert signals.coriolis_hub_power.sum() < 0.0, "fixture is not draining the arms"
    assert SWING_OBJECTIVES["coriolis"].evaluate(signals) > 0.0


def test_transfer_objectives_use_grip_force_signals() -> None:
    """Energy and impulse transfer integrate the grip-force channel."""
    signals = _signals()

    expected_energy = signals.integrate(signals.grip_force_power)
    assert SWING_OBJECTIVES["energy_transfer"].evaluate(signals) == pytest.approx(
        expected_energy
    )

    expected_impulse = signals.integrate(signals.grip_force_magnitude)
    assert SWING_OBJECTIVES["impulse_transfer"].evaluate(signals) == pytest.approx(
        expected_impulse
    )
    assert expected_impulse > 0.0, "grip-force impulse magnitude cannot be negative"


def test_objectives_are_immutable_registry_entries() -> None:
    """Reversibility: an objective definition cannot be mutated in place."""
    objective = SWING_OBJECTIVES["clubhead_speed"]
    with pytest.raises((AttributeError, TypeError)):
        objective.scale = 1.0  # type: ignore[misc]


def test_objective_ordering_puts_the_outcome_baseline_first() -> None:
    """Comparison tables read against clubhead speed, so it leads the registry."""
    assert next(iter(SWING_OBJECTIVES)) == "clubhead_speed"

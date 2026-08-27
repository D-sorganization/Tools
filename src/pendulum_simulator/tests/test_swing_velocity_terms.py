"""Contract tests for the centrifugal/Coriolis velocity-term decomposition.

The decomposition exists so a swing can be optimized for one mechanism without
the other. Its defining obligation is that it must remain an exact partition of
``physics.coriolis_vector`` — if the two ever disagree, every objective built on
top of the split is measuring something the simulator does not simulate.

Closes #4767.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics import PendulumParams, coriolis_vector
from double_pendulum_golf.swing_objectives.velocity_terms import (
    VelocityTerms,
    centrifugal_vector,
    coriolis_only_vector,
    coupling_constant,
    decompose_velocity_terms,
)

# Tour-representative driver swing: arms ~5 kg over 0.65 m, 0.30 kg shaft plus a
# 0.20 kg head at the tip of a 1.10 m club.
_PARAMS = PendulumParams(m1=5.0, m2=0.30, L1=0.65, L2=1.10, mClub=0.20)

_SAMPLE_COUNT = 400
_CLOSURE_ATOL = 1e-12


def _random_states(seed: int = 4767) -> np.ndarray:
    """Draw finite (phi, dtheta1, dphi) triples spanning a downswing envelope."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(-np.pi, np.pi, _SAMPLE_COUNT),
            rng.uniform(-40.0, 40.0, _SAMPLE_COUNT),
            rng.uniform(-60.0, 60.0, _SAMPLE_COUNT),
        ]
    )


def test_coupling_constant_matches_effective_segment_mass() -> None:
    """The coupling constant is (m2 + mClub) * L1 * L2, the head being at the tip."""
    expected = (_PARAMS.m2 + _PARAMS.mClub) * _PARAMS.L1 * _PARAMS.L2
    assert coupling_constant(_PARAMS) == pytest.approx(expected)
    assert coupling_constant(_PARAMS) > 0.0


def test_split_closes_exactly_against_physics_coriolis_vector() -> None:
    """Centrifugal plus Coriolis must reproduce the shipped combined vector.

    This is the anti-drift contract between the new decomposition and the
    existing (optionally Rust-backed) physics kernel.
    """
    for phi, dtheta1, dphi in _random_states():
        combined = coriolis_vector(phi, dtheta1, dphi, _PARAMS)
        centrifugal = centrifugal_vector(phi, dtheta1, dphi, _PARAMS)
        coriolis = coriolis_only_vector(phi, dtheta1, dphi, _PARAMS)
        assert np.allclose(centrifugal + coriolis, combined, atol=_CLOSURE_ATOL)


def test_decompose_returns_consistent_container() -> None:
    """The container's total must equal its parts and the shipped vector."""
    for phi, dtheta1, dphi in _random_states(seed=11):
        terms = decompose_velocity_terms(phi, dtheta1, dphi, _PARAMS)
        assert isinstance(terms, VelocityTerms)
        assert np.allclose(
            terms.total, coriolis_vector(phi, dtheta1, dphi, _PARAMS), atol=_CLOSURE_ATOL
        )
        assert np.allclose(terms.total, terms.centrifugal + terms.coriolis)


def test_coriolis_term_acts_only_on_the_hub() -> None:
    """There is no omega1*dphi cross term in the wrist row.

    This asymmetry is the whole reason the two mechanisms are separable: the
    wrist is driven centrifugally, the hub is drained by Coriolis coupling.
    """
    for phi, dtheta1, dphi in _random_states(seed=22):
        assert coriolis_only_vector(phi, dtheta1, dphi, _PARAMS)[1] == 0.0


def test_centrifugal_wrist_term_is_independent_of_uncock_rate() -> None:
    """The wrist centrifugal drive depends on arm speed squared, never on dphi."""
    phi, dtheta1 = 1.2, -18.0
    baseline = centrifugal_vector(phi, dtheta1, 0.0, _PARAMS)[1]
    for dphi in (-30.0, -5.0, 5.0, 30.0):
        assert centrifugal_vector(phi, dtheta1, dphi, _PARAMS)[1] == pytest.approx(baseline)


def test_coriolis_term_is_odd_in_each_velocity() -> None:
    """Reversing either rate alone flips the Coriolis sign; reversing both restores it."""
    phi, dtheta1, dphi = 0.9, -15.0, -22.0
    reference = coriolis_only_vector(phi, dtheta1, dphi, _PARAMS)[0]
    assert coriolis_only_vector(phi, -dtheta1, dphi, _PARAMS)[0] == pytest.approx(-reference)
    assert coriolis_only_vector(phi, dtheta1, -dphi, _PARAMS)[0] == pytest.approx(-reference)
    assert coriolis_only_vector(phi, -dtheta1, -dphi, _PARAMS)[0] == pytest.approx(reference)


@pytest.mark.parametrize("phi", [0.0, np.pi, -np.pi])
def test_both_terms_vanish_when_the_club_is_aligned_with_the_arms(phi: float) -> None:
    """sin(phi) gates the coupling, so an aligned or folded club has no coupling.

    ``sin(pi)`` is only zero to within one ulp in binary floating point, so the
    residual is bounded relative to the term's own natural scale rather than
    against an absolute constant.
    """
    arm_rate, uncock_rate = -20.0, -30.0
    natural_scale = coupling_constant(_PARAMS) * max(arm_rate**2, uncock_rate**2)
    tolerance = 10.0 * natural_scale * float(np.finfo(np.float64).eps)

    terms = decompose_velocity_terms(phi, arm_rate, uncock_rate, _PARAMS)
    assert np.allclose(terms.centrifugal, 0.0, atol=tolerance)
    assert np.allclose(terms.coriolis, 0.0, atol=tolerance)


def test_centrifugal_wrist_term_releases_a_cocked_club() -> None:
    """With the wrists cocked and the arms turning, the wrist drive must uncock.

    Sign convention: phi > 0 is a trailing club, so a negative generalized force
    on the wrist row of the equations of motion drives phi toward zero.
    """
    wrist_row_on_lhs = centrifugal_vector(0.9, -20.0, 0.0, _PARAMS)[1]
    # equations_of_motion subtracts this term, so a positive LHS entry is a release.
    assert wrist_row_on_lhs > 0.0


def test_rejects_non_finite_inputs() -> None:
    """Contract: every velocity-term entry point refuses non-finite arguments."""
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            centrifugal_vector(bad, 1.0, 1.0, _PARAMS)
        with pytest.raises(ValueError, match="finite"):
            coriolis_only_vector(0.5, bad, 1.0, _PARAMS)
        with pytest.raises(ValueError, match="finite"):
            decompose_velocity_terms(0.5, 1.0, bad, _PARAMS)


def test_velocity_terms_container_is_immutable() -> None:
    """Reversibility: callers must not be able to mutate a returned decomposition."""
    terms = decompose_velocity_terms(0.8, -12.0, -9.0, _PARAMS)
    with pytest.raises((AttributeError, TypeError)):
        terms.centrifugal = np.zeros(2)  # type: ignore[misc]

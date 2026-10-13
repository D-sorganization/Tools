"""Contract tests for the impact-optimality coefficient.

These pin the diagnosis behind epic #4775: the optimizer stops the hands at
impact because, in a point-mass-clubhead model, that *is* the optimum.

Tools #5431 changed the shipped model from that lumped club to the canonical
distributed one (uniform shaft ``m2`` plus a head point mass).  The diagnosis
therefore moved exactly as the contract below warned: the coefficient is zero
only in the head-only (lumped) limit, ``_LUMPED``, and is strictly negative for
the shipped model, so the speed-optimal swing now wants the hands moving
backward rather than stopped.  Follow-up #5523 re-ran the conclusions that
rested on the old zero (see the pinned values below).

Closes #4776.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics import PendulumParams, mass_matrix
from double_pendulum_golf.swing_objectives.impact_optimality import (
    energy_optimal_rates,
    impact_hand_speed_coefficient,
    optimal_hand_speed_sign,
)

_PARAMS = PendulumParams(m1=5.0, m2=0.30, L1=0.65, L2=1.10, mClub=0.20)
# Head-only limit of the canonical club: every gram at the tip (shaft mass -> 0).
_LUMPED = PendulumParams(m1=5.0, m2=1e-15, L1=0.65, L2=1.10, mClub=0.50)
_TIGHT = 1e-12


def test_head_only_limit_has_a_zero_coefficient() -> None:
    """The old headline result survives only in the lumped limit."""
    assert impact_hand_speed_coefficient(_LUMPED) == pytest.approx(0.0, abs=_TIGHT)


def test_shipped_distributed_model_has_a_negative_coefficient() -> None:
    """Tools #5431: the shipped club is distributed, so the zero is gone.

    Hand-derived for m1=5, m2=0.30, mClub=0.20, L1=0.65, L2=1.10:
    Mc=0.5, s=(0.15+0.20)*1.1=0.385, Jw=(0.10+0.20)*1.21=0.363.  The direction
    ``M^-1 [L1+L2, L2]`` has arm component
    ``(Jw*(L1+L2) - (Jw + s*L1)*L2) / det`` with
    ``Jw*L1 - s*L1*L2 = 0.65*(0.363 - 0.4235) < 0``.
    """
    assert impact_hand_speed_coefficient(_PARAMS) < -_TIGHT


@pytest.mark.parametrize("arm_mass", [3.0, 5.0, 9.0])
@pytest.mark.parametrize("arm_length", [0.55, 0.65, 0.80])
@pytest.mark.parametrize("head_mass", [0.15, 0.20, 0.30])
def test_coefficient_is_zero_for_every_parameter_choice(
    arm_mass: float, arm_length: float, head_mass: float
) -> None:
    """The zero is structural, not a coincidence of one parameter set.

    With all of the club's mass at the tip, its kinetic energy *is*
    ``0.5 * Mc * v_head**2``, so any arm motion is energy that did not reach the
    clubhead — whatever the masses and lengths.
    """
    params = PendulumParams(m1=arm_mass, m2=1e-15, L1=arm_length, L2=1.10, mClub=head_mass)
    assert impact_hand_speed_coefficient(params) == pytest.approx(0.0, abs=_TIGHT)


def test_coefficient_matches_the_mass_matrix_derivation() -> None:
    """The closed form must agree with solving ``M qdot = c`` numerically.

    Both quantities are zero here, so they are compared against the scale of the
    rates involved rather than by sign: the numerical solve returns a value at
    the level of floating-point noise (~1e-16) whose sign is meaningless.
    """
    for arm_length in (0.55, 0.65, 0.80):
        params = PendulumParams(m1=5.0, m2=1e-15, L1=arm_length, L2=1.10, mClub=0.50)
        direction = np.linalg.solve(
            mass_matrix(0.0, params),
            np.array([params.L1 + params.L2, params.L2]),
        )
        # Arm component is negligible against the uncock component it sits beside.
        assert abs(direction[0]) < 1e-12 * max(abs(direction[1]), 1.0)
        assert impact_hand_speed_coefficient(params) == pytest.approx(0.0, abs=_TIGHT)


def test_shipped_coefficient_sign_matches_the_numerical_solve() -> None:
    """For the shipped distributed club the closed form and ``M qdot = c`` agree."""
    for arm_length in (0.55, 0.65, 0.80):
        params = PendulumParams(m1=5.0, m2=0.30, L1=arm_length, L2=1.10, mClub=0.20)
        direction = np.linalg.solve(
            mass_matrix(0.0, params),
            np.array([params.L1 + params.L2, params.L2]),
        )
        coefficient = impact_hand_speed_coefficient(params)
        assert coefficient < 0.0
        assert np.sign(direction[0]) == np.sign(coefficient)


def test_coefficient_sign_tracks_the_numerical_solve_when_nonzero() -> None:
    """Where the coefficient is genuinely nonzero, its sign must be trustworthy.

    Built from a distributed-club mass matrix, since the shipped point-mass
    ``mass_matrix`` cannot express a club whose COM is short of the tip.
    """
    club_mass, club_com, club_inertia = 0.31, 0.89, 0.043
    for arm_length in (0.55, 0.65, 0.80):
        params = PendulumParams(m1=5.0, m2=0.30, L1=arm_length, L2=1.143, mClub=0.01)
        wrist_inertia = club_inertia + club_mass * club_com**2
        coupling = club_mass * arm_length * club_com
        hub_inertia = 5.0 * arm_length**2 + club_mass * arm_length**2
        mass = np.array(
            [
                [hub_inertia + wrist_inertia + 2.0 * coupling, wrist_inertia + coupling],
                [wrist_inertia + coupling, wrist_inertia],
            ]
        )
        direction = np.linalg.solve(mass, np.array([arm_length + params.L2, params.L2]))
        coefficient = impact_hand_speed_coefficient(
            params,
            club_mass_kg=club_mass,
            club_com_m=club_com,
            club_inertia_kgm2=club_inertia,
        )
        assert np.sign(direction[0]) == np.sign(coefficient)


def test_distributed_club_inertia_does_not_rescue_the_hands() -> None:
    """A real driver makes it worse, not better.

    Measured driver values put the coefficient *negative*, meaning the
    speed-optimal swing wants the hands moving backward through impact. This is
    why epic #4775 does not pursue distributed club inertia as the fix.
    """
    driver = impact_hand_speed_coefficient(
        _PARAMS, club_com_m=0.89, club_inertia_kgm2=0.043, club_mass_kg=0.31
    )
    assert driver < 0.0


def test_only_an_unphysical_club_gives_a_forward_optimum() -> None:
    """Moving the coefficient positive needs a club nobody could swing."""
    # A real driver's COM sits near 78% of its length; 88% is not a real club.
    unphysical = impact_hand_speed_coefficient(
        _PARAMS, club_com_m=1.02, club_inertia_kgm2=0.043, club_mass_kg=0.31
    )
    assert unphysical > 0.0


def test_optimal_hand_speed_sign_reports_the_three_regimes() -> None:
    """The helper names the regime rather than making callers read a float."""
    assert optimal_hand_speed_sign(_LUMPED) == "stopped"
    assert optimal_hand_speed_sign(_PARAMS) == "backward"
    assert (
        optimal_hand_speed_sign(
            _PARAMS, club_com_m=0.89, club_inertia_kgm2=0.043, club_mass_kg=0.31
        )
        == "backward"
    )
    assert (
        optimal_hand_speed_sign(
            _PARAMS, club_com_m=1.02, club_inertia_kgm2=0.043, club_mass_kg=0.31
        )
        == "forward"
    )


def test_energy_optimal_rates_put_everything_in_the_club() -> None:
    """For the head-only limit the optimal split is all club, no arm."""
    rates = energy_optimal_rates(_LUMPED, kinetic_energy_j=400.0)
    assert rates.arm_rate_rad_s == pytest.approx(0.0, abs=1e-9)
    assert rates.uncock_rate_rad_s > 0.0
    assert rates.hand_speed_m_s == pytest.approx(0.0, abs=1e-9)
    assert rates.clubhead_speed_m_s > 0.0


def test_energy_optimal_rates_honour_the_energy_budget() -> None:
    """The returned rates must actually carry the kinetic energy requested."""
    budget = 400.0
    rates = energy_optimal_rates(_PARAMS, kinetic_energy_j=budget)
    qdot = np.array([rates.arm_rate_rad_s, rates.uncock_rate_rad_s])
    energy = 0.5 * qdot @ mass_matrix(0.0, _PARAMS) @ qdot
    assert energy == pytest.approx(budget, rel=1e-9)


def test_energy_optimal_clubhead_speed_beats_any_other_split() -> None:
    """Sanity: no other rate split at the same energy goes faster."""
    budget = 400.0
    best = energy_optimal_rates(_PARAMS, kinetic_energy_j=budget)
    mass = mass_matrix(0.0, _PARAMS)
    rng = np.random.default_rng(4775)
    for _ in range(400):
        trial = rng.normal(size=2)
        scale = np.sqrt(2.0 * budget / (trial @ mass @ trial))
        arm, uncock = trial * scale
        speed = abs((_PARAMS.L1 + _PARAMS.L2) * arm + _PARAMS.L2 * uncock)
        assert speed <= best.clubhead_speed_m_s + 1e-9


def test_rejects_non_physical_inputs() -> None:
    """Contract: energies must be positive and overrides self-consistent."""
    with pytest.raises(ValueError, match="kinetic_energy_j"):
        energy_optimal_rates(_PARAMS, kinetic_energy_j=0.0)
    with pytest.raises(ValueError, match="club_com_m"):
        impact_hand_speed_coefficient(_PARAMS, club_com_m=-0.1)
    with pytest.raises(ValueError, match="club_inertia_kgm2"):
        impact_hand_speed_coefficient(_PARAMS, club_inertia_kgm2=-1.0)


def test_real_driver_bracket_with_true_com_and_inertia() -> None:
    """Tools #5523: omega1* bracket for the measured driver, derived by hand.

    ``I2 - m2*r2*(L2 - r2)`` with ``DRIVER_SPEC`` (m=0.310, r=0.867, I=0.0551) on
    its physical ``L2 = 1.143`` is ``0.0551 - 0.310*0.867*0.276 = -0.01908``; times
    ``L1 = 0.65`` is ``-0.01240``.  The legacy table row (r2=0.89, I2=0.043) gave
    ``-0.0268`` and is reproduced for comparison.
    """
    from double_pendulum_golf.swing_objectives.club_equivalence import DRIVER_SPEC

    params = PendulumParams(m1=5.0, m2=0.30, L1=0.65, L2=DRIVER_SPEC.length_m, mClub=0.01)
    kwargs = {
        "club_mass_kg": DRIVER_SPEC.mass_kg,
        "club_com_m": DRIVER_SPEC.com_m,
        "club_inertia_kgm2": DRIVER_SPEC.inertia_about_com_kgm2,
    }
    coefficient = impact_hand_speed_coefficient(params, **kwargs)
    assert coefficient / params.L1 == pytest.approx(-0.01908, abs=1e-5)
    assert coefficient == pytest.approx(-0.01240, abs=1e-5)
    assert optimal_hand_speed_sign(params, **kwargs) == "backward"

    legacy = impact_hand_speed_coefficient(
        params, club_mass_kg=0.31, club_com_m=0.89, club_inertia_kgm2=0.043
    )
    assert legacy / params.L1 == pytest.approx(-0.0268, abs=1e-4)


def test_shipped_preset_bracket_is_backward_but_nearly_stopped() -> None:
    """Tools #5523: on the shipped distributed driver the bracket is -0.0217.

    The speed-optimal hands move backward, but only by 0.36% of the clubhead
    speed (0.33 m/s of 91.7 m/s at 1000 J): qualitatively "stopped" survives, the
    exact zero does not.
    """
    from double_pendulum_golf.swing_objectives.presets import DEFAULT_PRESET

    params = DEFAULT_PRESET.to_params()
    assert impact_hand_speed_coefficient(params) / params.L1 == pytest.approx(
        -0.02174, abs=1e-5
    )
    assert optimal_hand_speed_sign(params) == "backward"
    rates = energy_optimal_rates(params, kinetic_energy_j=1000.0)
    assert rates.hand_speed_m_s / rates.clubhead_speed_m_s == pytest.approx(-0.00357, abs=1e-5)
    assert abs(rates.hand_speed_m_s / rates.clubhead_speed_m_s) < 0.01

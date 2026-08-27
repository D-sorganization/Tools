"""Contract tests for the impact-optimality coefficient.

These pin the diagnosis behind epic #4775: the optimizer stops the hands at
impact because, in a point-mass-clubhead model, that *is* the optimum. If the
coefficient below ever stops being identically zero for the shipped model, the
diagnosis has changed and the surrounding conclusions need revisiting.

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
_TIGHT = 1e-12


def test_shipped_point_mass_model_has_a_zero_coefficient() -> None:
    """The headline result: a tip-concentrated club wants the hands stopped."""
    assert impact_hand_speed_coefficient(_PARAMS) == pytest.approx(0.0, abs=_TIGHT)


@pytest.mark.parametrize("arm_mass", [3.0, 5.0, 9.0])
@pytest.mark.parametrize("arm_length", [0.55, 0.65, 0.80])
@pytest.mark.parametrize("head_mass", [0.15, 0.20, 0.30])
def test_coefficient_is_zero_for_every_parameter_choice(
    arm_mass: float, arm_length: float, head_mass: float
) -> None:
    """The zero is structural, not a coincidence of one parameter set.

    With all of segment 2's mass at the tip, the club's kinetic energy *is*
    ``0.5 * me * v_head**2``, so any arm motion is energy that did not reach the
    clubhead — whatever the masses and lengths.
    """
    params = PendulumParams(m1=arm_mass, m2=0.30, L1=arm_length, L2=1.10, mClub=head_mass)
    assert impact_hand_speed_coefficient(params) == pytest.approx(0.0, abs=_TIGHT)


def test_coefficient_matches_the_mass_matrix_derivation() -> None:
    """The closed form must agree with solving ``M qdot = c`` numerically.

    Both quantities are zero here, so they are compared against the scale of the
    rates involved rather than by sign: the numerical solve returns a value at
    the level of floating-point noise (~1e-16) whose sign is meaningless.
    """
    for arm_length in (0.55, 0.65, 0.80):
        params = PendulumParams(m1=5.0, m2=0.30, L1=arm_length, L2=1.10, mClub=0.20)
        direction = np.linalg.solve(
            mass_matrix(0.0, params),
            np.array([params.L1 + params.L2, params.L2]),
        )
        # Arm component is negligible against the uncock component it sits beside.
        assert abs(direction[0]) < 1e-12 * max(abs(direction[1]), 1.0)
        assert impact_hand_speed_coefficient(params) == pytest.approx(0.0, abs=_TIGHT)


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
    assert optimal_hand_speed_sign(_PARAMS) == "stopped"
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
    """For the shipped model the optimal split is all club, no arm."""
    rates = energy_optimal_rates(_PARAMS, kinetic_energy_j=400.0)
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

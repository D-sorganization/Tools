"""Contract tests for the inertia-matched club equivalence.

These exist because getting this wrong produced a published, incorrect
conclusion (#4785). The shipped preset lumped a real club's *mass* at the tip,
which doubles its inertia about the wrist and doubles the arm/club coupling that
fights the release. The optimizer then had to reverse the hub torque hard enough
to stop the hands, and that artifact was reported as a structural limit of the
model.

Closes #4785.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics import PendulumParams
from double_pendulum_golf.swing_objectives.club_equivalence import (
    DRIVER_SPEC,
    SEVEN_IRON_SPEC,
    RealClubSpec,
    coupling_error,
    distributed_club_split,
    distributed_coupling_error,
    equivalent_tip_mass,
    wrist_inertia,
)
from double_pendulum_golf.swing_objectives.impact_optimality import (
    impact_hand_speed_coefficient,
)


def test_driver_spec_matches_published_club_measurements() -> None:
    """A driver is ~0.31 kg with its COM about three quarters down the shaft."""
    assert DRIVER_SPEC.mass_kg == pytest.approx(0.31, abs=0.02)
    assert 0.74 <= DRIVER_SPEC.com_m / DRIVER_SPEC.length_m <= 0.82
    # Inertia about the butt is what clubfitters measure; ~0.29 kg.m^2 for a driver.
    butt = DRIVER_SPEC.inertia_about_wrist_kgm2
    assert butt == pytest.approx(0.288, abs=0.02)


def test_wrist_inertia_uses_the_parallel_axis_theorem() -> None:
    """Inertia about the wrist is I_com + m*r^2, not I_com alone."""
    spec = RealClubSpec(
        name="test", mass_kg=0.30, length_m=1.10, com_m=0.80, inertia_about_com_kgm2=0.05
    )
    assert wrist_inertia(spec) == pytest.approx(0.05 + 0.30 * 0.80**2)


def test_equivalent_tip_mass_reproduces_the_wrist_inertia() -> None:
    """The whole point: the equivalent club must swing like the real one.

    A point mass ``me`` at the tip has wrist inertia ``me * L2**2``; matching
    that to the real club's wrist inertia is what preserves the release
    dynamics.
    """
    for spec in (DRIVER_SPEC, SEVEN_IRON_SPEC):
        me = equivalent_tip_mass(spec, shaft_length_m=spec.length_m)
        assert me * spec.length_m**2 == pytest.approx(wrist_inertia(spec))


def test_a_driver_equivalent_is_much_lighter_than_the_real_club() -> None:
    """The headline correction: the tip mass is not the club's mass."""
    me = equivalent_tip_mass(DRIVER_SPEC, shaft_length_m=1.10)
    assert me < DRIVER_SPEC.mass_kg
    assert me == pytest.approx(0.238, abs=0.01)


def test_the_shipped_preset_was_wrong_on_two_separate_counts() -> None:
    """Quantify both errors separately, because they are different sizes.

    The preset lumped 0.50 kg at the tip. That is wrong twice over: a real
    driver is only 0.310 kg, and even that mass does not belong at the tip. The
    two effects compound to the 2.1x coupling error behind #4785.
    """
    shaft = 1.10
    real = wrist_inertia(DRIVER_SPEC)

    # Error 1: putting the club's true mass at the tip rather than at its COM.
    mass_at_tip = DRIVER_SPEC.mass_kg * shaft**2
    assert mass_at_tip / real == pytest.approx(1.30, abs=0.05)

    # Error 2: the preset's lumped mass was itself 61% above a real driver.
    assert 0.50 / DRIVER_SPEC.mass_kg == pytest.approx(1.61, abs=0.05)

    # Compounded, which is what the optimizer actually saw.
    preset_at_tip = 0.50 * shaft**2
    assert preset_at_tip / real == pytest.approx(2.10, abs=0.10)


def test_equivalent_mass_scales_with_the_modelled_shaft_length() -> None:
    """A shorter modelled shaft needs more tip mass to carry the same inertia."""
    short = equivalent_tip_mass(DRIVER_SPEC, shaft_length_m=0.90)
    long = equivalent_tip_mass(DRIVER_SPEC, shaft_length_m=1.30)
    assert short > long


def test_coupling_scales_with_the_equivalent_mass() -> None:
    """The arm/club coupling that fights the release follows me*L1*L2.

    This is the quantity the correction is really about: it was 2.08x too large.
    """
    arm_length, shaft = 0.65, 1.10
    me = equivalent_tip_mass(DRIVER_SPEC, shaft_length_m=shaft)
    corrected = me * arm_length * shaft
    naive = 0.50 * arm_length * shaft  # the shipped preset's lumped mass
    assert naive / corrected == pytest.approx(2.1, abs=0.15)


def test_seven_iron_is_heavier_and_shorter_than_a_driver() -> None:
    """Sanity on the second preset, so a wrong copy-paste is caught."""
    assert SEVEN_IRON_SPEC.mass_kg > DRIVER_SPEC.mass_kg
    assert SEVEN_IRON_SPEC.length_m < DRIVER_SPEC.length_m


def test_rejects_unphysical_clubs() -> None:
    """Contract: every field has a physically meaningful range."""
    good = dict(name="c", mass_kg=0.31, length_m=1.10, com_m=0.87, inertia_about_com_kgm2=0.05)
    for field, bad in (
        ("mass_kg", 0.0),
        ("length_m", 0.0),
        ("com_m", -0.1),
        ("inertia_about_com_kgm2", -1.0),
    ):
        with pytest.raises(ValueError, match=field):
            RealClubSpec(**{**good, field: bad})


def test_rejects_a_centre_of_mass_off_the_club() -> None:
    """A COM beyond the head is not a club anyone could build."""
    with pytest.raises(ValueError, match="com_m"):
        RealClubSpec(
            name="c", mass_kg=0.31, length_m=1.10, com_m=1.5, inertia_about_com_kgm2=0.05
        )


def test_equivalent_tip_mass_rejects_a_non_positive_shaft() -> None:
    """Contract: the modelled shaft length must be usable."""
    with pytest.raises(ValueError, match="shaft_length_m"):
        equivalent_tip_mass(DRIVER_SPEC, shaft_length_m=0.0)


def test_specs_are_immutable() -> None:
    """Reversibility: a club spec handed to a preset cannot be edited in place."""
    with pytest.raises((AttributeError, TypeError)):
        DRIVER_SPEC.mass_kg = 1.0  # type: ignore[misc]


def test_equivalence_is_finite_and_positive_across_realistic_clubs() -> None:
    """No preset may produce a degenerate equivalent."""
    for spec in (DRIVER_SPEC, SEVEN_IRON_SPEC):
        for shaft in np.linspace(0.85, 1.25, 9):
            me = equivalent_tip_mass(spec, shaft_length_m=float(shaft))
            assert np.isfinite(me) and me > 0.0


def test_inertia_arithmetic_distinguishes_masses() -> None:
    """0.310 kg at 1.10 m has 0.3751 kg*m^2 inertia; 0.605 belongs to 0.500 kg.

    0.310 * 1.10^2 = 0.3751 kg*m^2 (1.30x the real 0.288 kg*m^2).
    0.500 * 1.10^2 = 0.6050 kg*m^2 (2.10x the real 0.288 kg*m^2).
    """
    length = 1.10
    inertia_310 = 0.310 * length**2
    assert inertia_310 == pytest.approx(0.3751, abs=1e-4)

    inertia_500 = 0.500 * length**2
    assert inertia_500 == pytest.approx(0.6050, abs=1e-4)


def test_coupling_error_formula_and_contract() -> None:
    """Coupling error after matching delta is (L1/L2) * [I_com - m*r*(L2 - r)]."""
    l1, l2 = 0.65, 1.10
    for spec in (DRIVER_SPEC, SEVEN_IRON_SPEC):
        err = coupling_error(spec, arm_length_m=l1, shaft_length_m=l2)
        expected_bracket = spec.inertia_about_com_kgm2 - spec.mass_kg * spec.com_m * (
            l2 - spec.com_m
        )
        assert err == pytest.approx((l1 / l2) * expected_bracket)

        # Also verify directly from definitions: mu_model - mu_real
        me = equivalent_tip_mass(spec, shaft_length_m=l2)
        mu_model = me * l1 * l2
        mu_real = spec.mass_kg * l1 * spec.com_m
        assert err == pytest.approx(mu_model - mu_real)


def test_coupling_error_rejects_non_positive_inputs() -> None:
    """Contract: arm_length_m and shaft_length_m must be positive and finite."""
    with pytest.raises(ValueError, match="arm_length_m"):
        coupling_error(DRIVER_SPEC, arm_length_m=0.0, shaft_length_m=1.10)
    with pytest.raises(ValueError, match="arm_length_m"):
        coupling_error(DRIVER_SPEC, arm_length_m=float("nan"), shaft_length_m=1.10)
    with pytest.raises(ValueError, match="shaft_length_m"):
        coupling_error(DRIVER_SPEC, arm_length_m=0.65, shaft_length_m=-1.0)
    with pytest.raises(ValueError, match="shaft_length_m"):
        coupling_error(DRIVER_SPEC, arm_length_m=0.65, shaft_length_m=float("inf"))


def test_matching_wrist_inertia_does_not_match_full_mass_matrix() -> None:
    """Matching delta reproduces M22, but leaves residual coupling and diagonal errors."""
    l1, l2 = 0.65, 1.10
    spec = DRIVER_SPEC
    me = equivalent_tip_mass(spec, shaft_length_m=l2)
    delta = wrist_inertia(spec)

    # Point-mass M22 matches delta
    m22_model = me * l2**2
    assert m22_model == pytest.approx(delta)

    # But coupling differs
    err = coupling_error(spec, arm_length_m=l1, shaft_length_m=l2)
    assert abs(err) > 1e-4

    # And base diagonal arm inertia (at phi = pi/2 where cos(phi) = 0) differs by (me - m)*L1^2
    m1 = 5.0
    diag_model_unloaded = (m1 + me) * l1**2 + delta
    diag_real_unloaded = (m1 + spec.mass_kg) * l1**2 + delta
    assert diag_model_unloaded != pytest.approx(diag_real_unloaded)
    assert diag_model_unloaded - diag_real_unloaded == pytest.approx(
        (me - spec.mass_kg) * l1**2
    )


def test_coupling_error_matches_impact_hand_speed_coefficient_bracket() -> None:
    """The coupling error is identically impact_hand_speed_coefficient / L2."""
    l1, l2 = 0.65, 1.10
    params = PendulumParams(m1=5.0, m2=DRIVER_SPEC.mass_kg, L1=l1, L2=l2)
    coeff = impact_hand_speed_coefficient(
        params,
        club_mass_kg=DRIVER_SPEC.mass_kg,
        club_com_m=DRIVER_SPEC.com_m,
        club_inertia_kgm2=DRIVER_SPEC.inertia_about_com_kgm2,
    )
    err = coupling_error(DRIVER_SPEC, arm_length_m=l1, shaft_length_m=l2)
    assert err == pytest.approx(coeff / l2)


# --- Canonical distributed split (Tools #5431) -------------------------------


def test_distributed_split_reproduces_mass_and_wrist_inertia() -> None:
    """Real masses on the canonical club reproduce the real mass and M22 exactly."""
    from double_pendulum_golf.club_inertia import distal_inertia

    for spec in (DRIVER_SPEC, SEVEN_IRON_SPEC):
        shaft_mass, head_mass = distributed_club_split(spec, shaft_length_m=1.10)
        club = distal_inertia(shaft_mass, head_mass, 1.10)
        assert club.mass_kg == pytest.approx(spec.mass_kg)
        assert club.inertia_about_pivot_kg_m2 == pytest.approx(wrist_inertia(spec))


def test_driver_split_values_and_coupling() -> None:
    """Pins the derivation in the module docstring: 0.108 kg shaft, 0.202 kg head."""
    shaft_mass, head_mass = distributed_club_split(DRIVER_SPEC, shaft_length_m=1.10)
    assert shaft_mass == pytest.approx(0.1078, abs=5e-4)
    assert head_mass == pytest.approx(0.2022, abs=5e-4)
    # Coupling within ~5% of the real club (the point-mass equivalent was -2.6%).
    err = distributed_coupling_error(DRIVER_SPEC, 0.65, 1.10)
    assert abs(err) / (0.65 * DRIVER_SPEC.mass_kg * DRIVER_SPEC.com_m) < 0.06


def test_default_preset_pins_the_real_driver_wrist_row() -> None:
    from double_pendulum_golf.physics import mass_matrix
    from double_pendulum_golf.swing_objectives.presets import DEFAULT_PRESET

    params = DEFAULT_PRESET.to_params()
    assert params.m2 + params.mClub == pytest.approx(DRIVER_SPEC.mass_kg)
    assert mass_matrix(0.3, params)[1, 1] == pytest.approx(0.2881, abs=1e-4)


def test_distributed_split_rejects_unrepresentable_club() -> None:
    with pytest.raises(ValueError, match="shaft_length_m"):
        distributed_club_split(DRIVER_SPEC, shaft_length_m=0.0)
    with pytest.raises(ValueError, match="not representable"):
        distributed_club_split(DRIVER_SPEC, shaft_length_m=0.4)


def test_republished_inertia_and_coupling_table_values() -> None:
    """The #5523 table: wrist inertia and coupling for each club on a 1.10 m shaft.

    Derived by ``scripts/rerun_distributed_club.py club`` (Tools #5523).  Real
    driver ``mu = m * L1 * r = 0.310 * 0.65 * 0.867``; a point mass at the tip has
    ``mu = m * L1 * L2``; the distributed preset has ``mu = (m2/2 + mh) * L2 * L1``.
    """
    from double_pendulum_golf.swing_objectives.presets import DEFAULT_PRESET

    arm, shaft = 0.65, 1.10
    real_mu = DRIVER_SPEC.mass_kg * arm * DRIVER_SPEC.com_m
    assert wrist_inertia(DRIVER_SPEC) == pytest.approx(0.2881, abs=1e-4)
    assert real_mu == pytest.approx(0.1747, abs=1e-4)
    assert 0.310 * shaft**2 == pytest.approx(0.3751, abs=1e-4)
    assert 0.310 * arm * shaft == pytest.approx(0.2217, abs=1e-4)
    assert 0.500 * shaft**2 == pytest.approx(0.6050, abs=1e-4)
    assert 0.500 * arm * shaft == pytest.approx(0.3575, abs=1e-4)
    assert 0.500 * arm * shaft / real_mu == pytest.approx(2.05, abs=0.01)

    params = DEFAULT_PRESET.to_params()
    preset_mu = (params.m2 / 2.0 + params.mClub) * shaft * arm
    assert preset_mu == pytest.approx(0.1831, abs=1e-4)
    assert preset_mu / real_mu == pytest.approx(1.048, abs=0.002)
    assert distributed_coupling_error(DRIVER_SPEC, arm, shaft) == pytest.approx(
        0.0084, abs=1e-4
    )

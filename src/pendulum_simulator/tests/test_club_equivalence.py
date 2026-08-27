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

from double_pendulum_golf.swing_objectives.club_equivalence import (
    DRIVER_SPEC,
    SEVEN_IRON_SPEC,
    RealClubSpec,
    equivalent_tip_mass,
    wrist_inertia,
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

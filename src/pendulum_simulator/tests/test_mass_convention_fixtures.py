"""Shared-fixture parity, numeric stability and preset pinning (Tools #5431)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from mass_convention_support import FIXTURE

from double_pendulum_golf.club_inertia import DistalSegmentInertia, distal_inertia
from double_pendulum_golf.physics import (
    PendulumParams,
    coriolis_vector,
    gravity_vector,
    mass_matrix,
    potential_energy,
)


def test_distal_inertia_matches_shared_fixture() -> None:
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    assert cases
    for case in cases:
        got = distal_inertia(case["shaft_mass_kg"], case["head_mass_kg"], case["length_m"])
        assert isinstance(got, DistalSegmentInertia)
        assert got.mass_kg == pytest.approx(case["mass_kg"], rel=1e-14)
        assert got.first_moment_kg_m == pytest.approx(case["first_moment_kg_m"], rel=1e-14)
        assert got.inertia_about_pivot_kg_m2 == pytest.approx(
            case["inertia_about_pivot_kg_m2"], rel=1e-14
        )
        assert got.centroidal_inertia_kg_m2 == pytest.approx(
            case["centroidal_inertia_kg_m2"], rel=1e-14, abs=1e-300
        )


# ----------------------------------------------------- numeric stability (Icm)


def test_centroidal_inertia_never_cancels_below_zero() -> None:
    """Icm is a closed form, not ``Jw - s*com`` (catastrophic cancellation).

    Swept over L2 in [0.5, 1.5] and m2 log-uniform in [1e-15, 1] with a head
    mass from 0 to 1 kg; the parallel-axis sum is the reference.
    """
    rng = np.random.default_rng(5431)
    for _ in range(4000):
        length = rng.uniform(0.5, 1.5)
        shaft = 10.0 ** rng.uniform(-15.0, 0.0)
        head = rng.choice([0.0, rng.uniform(0.0, 1.0)])
        if shaft + head <= 0.0:
            continue
        club = distal_inertia(shaft, head, length)
        com = club.com_m
        reference = (
            shaft * length**2 / 12.0
            + shaft * (length / 2.0 - com) ** 2
            + head * (length - com) ** 2
        )
        assert club.icm_kg_m2 >= 0.0
        assert club.icm_kg_m2 == pytest.approx(reference, rel=1e-9, abs=1e-30)


def test_impact_optimality_accepts_the_vanishing_shaft_limit() -> None:
    from double_pendulum_golf.swing_objectives.impact_optimality import (
        impact_hand_speed_coefficient,
    )

    rng = np.random.default_rng(1)
    for length in rng.uniform(0.5, 1.5, 200):
        params = PendulumParams(m1=5.0, m2=1e-15, L1=0.65, L2=float(length), mClub=0.5)
        assert abs(impact_hand_speed_coefficient(params)) < 1e-12


# ------------------------------------------------- shared dynamics golden file


def test_dynamics_match_shared_golden_cases() -> None:
    path = FIXTURE.with_name("distal_dynamics_cases.json")
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 5
    for c in cases:
        params = PendulumParams(
            m1=c["m1"],
            m2=c["shaft_mass_kg"],
            mClub=c["head_mass_kg"],
            L1=c["l1"],
            L2=c["l2"],
            g=c["g"],
        )
        mass = mass_matrix(c["phi"], params)
        assert mass[0, 0] == pytest.approx(c["m11"], rel=1e-12)
        assert mass[0, 1] == pytest.approx(c["m12"], rel=1e-12)
        assert mass[1, 1] == pytest.approx(c["m22"], rel=1e-12)
        coriolis = coriolis_vector(c["phi"], c["dtheta1"], c["dphi"], params)
        assert coriolis[0] == pytest.approx(c["c1"], rel=1e-12)
        assert coriolis[1] == pytest.approx(c["c2"], rel=1e-12)
        gravity = gravity_vector(c["theta1"], c["phi"], params)
        assert gravity[0] == pytest.approx(c["g1"], rel=1e-12)
        assert gravity[1] == pytest.approx(c["g2"], rel=1e-12)
        state = np.array([c["theta1"], c["phi"], c["dtheta1"], c["dphi"]])
        assert potential_energy(state, params) == pytest.approx(
            c["potential_energy_j"], rel=1e-12
        )


def test_driver_preset_fixture_matches_the_derived_split() -> None:
    """The web presets are pinned to this file; it must equal the derivation."""
    from double_pendulum_golf.swing_objectives.club_equivalence import (
        DRIVER_SPEC,
        distributed_club_split,
    )

    fixture = json.loads(FIXTURE.with_name("driver_preset.json").read_text(encoding="utf-8"))
    shaft, head = distributed_club_split(DRIVER_SPEC, fixture["length_m"])
    assert fixture["shaft_mass_kg"] == pytest.approx(shaft, rel=1e-14)
    assert fixture["head_mass_kg"] == pytest.approx(head, rel=1e-14)


def test_gui_golf_presets_use_the_derived_split() -> None:
    pytest.importorskip("PyQt6")
    from double_pendulum_golf.gui.controls_widget import ControlsWidget
    from double_pendulum_golf.swing_objectives.club_equivalence import (
        DRIVER_SPEC,
        distributed_club_split,
    )

    shaft, head = distributed_club_split(DRIVER_SPEC, 1.10)
    for name in ("Golf Swing (passive wrist)", "Golf Swing (active wrist)"):
        preset = ControlsWidget.PRESETS[name]
        assert preset[8] == pytest.approx(shaft, rel=1e-14)
        assert preset[9] == pytest.approx(head, rel=1e-14)

# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for golfer anthropometry scaling and delivery generation."""

import numpy as np
import pytest

from rate_of_closure.simulation.anthropometry import (
    DEFAULT_GOLFER_HEIGHT_M,
    DEFAULT_GOLFER_MASS_KG,
    STANDARD_ARM_LENGTH_FRACTION,
    STANDARD_ARM_MASS_FRACTION,
    GolferAnthropometry,
)
from shared.python.swing_sim.types import PlaneOrientation


def test_default_anthropometry_values() -> None:
    anthro = GolferAnthropometry()
    assert anthro.height_m == DEFAULT_GOLFER_HEIGHT_M
    assert anthro.mass_kg == DEFAULT_GOLFER_MASS_KG
    assert anthro.effective_lead_arm_length_m == pytest.approx(
        DEFAULT_GOLFER_HEIGHT_M * STANDARD_ARM_LENGTH_FRACTION
    )
    assert anthro.effective_lead_arm_mass_kg == pytest.approx(
        DEFAULT_GOLFER_MASS_KG * STANDARD_ARM_MASS_FRACTION
    )


def test_custom_anthropometry_scaling() -> None:
    anthro = GolferAnthropometry(height_m=1.90, mass_kg=90.0)
    assert anthro.effective_lead_arm_length_m == pytest.approx(1.90 * 0.42)
    assert anthro.effective_lead_arm_mass_kg == pytest.approx(90.0 * 0.10)

    # Overrides
    custom = GolferAnthropometry(
        height_m=1.80, mass_kg=80.0, lead_arm_length_m=0.70, lead_arm_mass_kg=7.0
    )
    assert custom.effective_lead_arm_length_m == 0.70
    assert custom.effective_lead_arm_mass_kg == 7.0


def test_invalid_anthropometry_raises() -> None:
    with pytest.raises(ValueError):
        GolferAnthropometry(height_m=-1.0)
    with pytest.raises(ValueError):
        GolferAnthropometry(mass_kg=0.0)
    with pytest.raises(ValueError):
        GolferAnthropometry(lead_arm_length_m=float("nan"))


def test_to_pendulum_parameters_consistency() -> None:
    anthro = GolferAnthropometry()
    params = anthro.to_pendulum_parameters()
    assert params.l1 == pytest.approx(anthro.effective_lead_arm_length_m)
    assert params.m1 == pytest.approx(anthro.effective_lead_arm_mass_kg)
    assert params.l2 == 1.0
    assert params.lc1 < params.l1
    assert params.lc2 < params.l2
    assert params.i1 > 0.0
    assert params.i2 > 0.0


def test_generate_delivery_reproducible() -> None:
    anthro = GolferAnthropometry()
    res1 = anthro.generate_delivery(duration_s=0.5, dt=1e-3)
    res2 = anthro.generate_delivery(duration_s=0.5, dt=1e-3)

    assert np.allclose(res1.time_s, res2.time_s)
    assert np.allclose(res1.clubhead_poses, res2.clubhead_poses)
    assert np.allclose(res1.clubhead_twists, res2.clubhead_twists)
    assert res1.joint_ids == ("hub", "wrist", "clubhead")


def test_generate_delivery_with_plane() -> None:
    anthro = GolferAnthropometry()
    plane = PlaneOrientation(side_tilt_deg=-45.0)
    res = anthro.generate_delivery(duration_s=0.5, dt=1e-3, plane=plane)

    assert res.success
    assert len(res.time_s) == len(res.clubhead_poses)
    # The normal to the plane is rotated, so y component of position/velocity
    # is non-zero
    assert np.any(np.abs(res.clubhead_poses[:, 1, 3]) > 1e-4)

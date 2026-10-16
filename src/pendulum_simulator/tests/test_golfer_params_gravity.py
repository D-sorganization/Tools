"""``GolferParams.g`` default and injection (#5635, part 2).

UpstreamDrift's ``GolferParams`` defaults ``g`` to standard gravity
(9.80665 m/s^2); Tools defaults to ``GRAVITY_MSS`` (9.81 m/s^2).  Changing the
Tools default shifts numeric outputs, so it is pending an owner decision.
Until then these tests pin the current default (so any change is deliberate)
and the injection path that UpstreamDrift's adapters rely on: an explicit
``g`` must flow straight into the gravity vector.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from double_pendulum_golf.constants import GRAVITY_MSS, GRAVITY_STANDARD
from double_pendulum_golf.golfer_dynamics import analytical_gravity_vector
from double_pendulum_golf.physics_golfer import N_DOF, GolferParams

_GEOMETRY = {
    "m_hub": 2.0,
    "m_r_upper": 3.0,
    "m_r_fore": 2.0,
    "m_l_upper": 3.0,
    "m_l_fore": 2.0,
    "m_club": 0.5,
    "L_hub": 0.15,
    "L_r_upper": 0.35,
    "L_r_fore": 0.30,
    "L_l_upper": 0.35,
    "L_l_fore": 0.30,
    "L_club": 1.1,
    "d_rs": 0.20,
    "d_ls": 0.20,
    "grip_right": 0.05,
    "grip_left": 0.25,
}


def test_default_g_is_unchanged_pending_owner_decision() -> None:
    assert GolferParams(**_GEOMETRY).g == GRAVITY_MSS == 9.81


def test_standard_gravity_is_available_for_upstreamdrift_parity() -> None:
    assert GRAVITY_STANDARD == 9.80665


def test_explicit_g_is_used_and_gravity_vector_is_linear_in_g() -> None:
    q = np.linspace(-0.6, 0.6, N_DOF)
    standard = GolferParams(**_GEOMETRY, g=GRAVITY_STANDARD)
    unit = GolferParams(**_GEOMETRY, g=1.0)
    assert standard.g == GRAVITY_STANDARD
    np.testing.assert_allclose(
        analytical_gravity_vector(q, standard),
        GRAVITY_STANDARD * analytical_gravity_vector(q, unit),
        rtol=1e-14,
        atol=0.0,
    )


@pytest.mark.parametrize("bad_g", [-1.0, math.nan, math.inf])
def test_rejects_negative_or_non_finite_g(bad_g: float) -> None:
    with pytest.raises(ValueError, match="g must be"):
        GolferParams(**_GEOMETRY, g=bad_g)

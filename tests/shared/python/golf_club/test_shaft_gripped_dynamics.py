"""Physical coordinate-energy controls for frozen grip-supported operators."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shared.python.golf_club._shaft_equilibrium import moving_residual_jacobian
from shared.python.golf_club._shaft_gripped_chain import GrippedSectionChain
from shared.python.golf_club._shaft_gripped_dynamics import linearized_gripped_dynamics
from shared.python.golf_club._shaft_loaded_dynamics import linearized_chain_dynamics
from shared.python.golf_club._shaft_se3 import exp_twist

from .test_shaft_equilibrium import _fixture
from .test_shaft_gripped_chain import _attachment
from .test_shaft_loaded_dynamics import _chain


def test_grip_mass_and_damping_match_independent_coordinate_rate_energy() -> None:
    shaft = _chain()
    _, poses = _fixture([0, 0, 0], [0, 0, 0])
    poses = exp_twist([0.1, -0.2, 0.3, 0.5, -0.3, 0.6]) @ poses
    attachment = _attachment()
    factor = np.diag(np.arange(1.0, 7))
    factor[0, 4] = 0.8
    attachment = replace(
        attachment,
        grip=replace(
            attachment.grip, inertance_factor=factor, damping_factor=factor / 2
        ),
    )
    attachment = replace(
        attachment, anchor=replace(attachment.anchor, observer_id=shaft.frame.frame_id)
    )
    model = GrippedSectionChain(shaft, (attachment,))
    base = linearized_chain_dynamics(shaft, poses)
    result = linearized_gripped_dynamics(model, poses)
    velocity = np.linspace(-0.2, 0.4, 18)
    step = 1e-6

    def coordinates(time: float) -> np.ndarray:
        current = poses[0] @ exp_twist(time * velocity[:6])
        return np.r_[current[:3, 3], Rotation.from_matrix(current[:3, :3]).as_rotvec()]

    rate = (coordinates(step) - coordinates(-step)) / (2 * step)
    assert velocity @ (result.mass - base.mass) @ velocity == pytest.approx(
        np.linalg.norm(factor @ rate) ** 2, rel=2e-8
    )
    assert velocity @ result.damping @ velocity == pytest.approx(
        np.linalg.norm(factor @ rate / 2) ** 2, rel=2e-8
    )
    np.testing.assert_array_equal(result.gyroscopic, base.gyroscopic)
    np.testing.assert_allclose(result.damping, result.damping.T, atol=1e-13)
    np.testing.assert_allclose(
        result.stiffness, moving_residual_jacobian(model.linearize(poses)), atol=2e-12
    )
    np.testing.assert_allclose(
        result.residual, model.linearize(poses).residual, atol=2e-12
    )
    assert result.frame == shaft.frame
    assert result.stability_status == "unqualified"


def test_missing_distributed_inertia_is_not_replaced_with_zero_mass() -> None:
    shaft, poses = _fixture([0, 0, 0], [0, 0, 0])
    with pytest.raises(TypeError, match="RotatingSectionChain"):
        linearized_gripped_dynamics(GrippedSectionChain(shaft, (_attachment(),)), poses)

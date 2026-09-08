"""Independent compliance and preload controls for grip-supported shafts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shared.python.golf_club._grip_finite_response import FinitePoseGrip
from shared.python.golf_club._grip_moving_kinematics import MaterialPointMotion
from shared.python.golf_club._rotating_body_contracts import RotatingFrameState
from shared.python.golf_club._shaft_equilibrium import moving_residual_jacobian
from shared.python.golf_club._shaft_gripped_chain import (
    GripAttachment,
    GrippedSectionChain,
)
from shared.python.golf_club._shaft_gripped_equilibrium import solve_gripped_chain
from shared.python.golf_club._shaft_se3 import exp_twist

from .test_shaft_equilibrium import _controls, _fixture
from .test_shaft_rotating_chain import _radial_rod


def _attachment(stiffness: float = 100.0, node: int = 0) -> GripAttachment:
    law = FinitePoseGrip(
        np.zeros((6, 6)),
        np.zeros((6, 6)),
        np.diag(np.sqrt([300, 300, stiffness, 50, 50, 10])),
        "synthetic",
    )
    anchor = MaterialPointMotion(np.eye(4), np.zeros(6), np.zeros(6), "observer")
    return GripAttachment(node, law, anchor)


@pytest.mark.parametrize("stiffness", [100.0, 1000.0, 10000.0])
def test_root_and_shaft_compliances_add_without_an_implicit_clamp(
    stiffness: float,
) -> None:
    shaft, seed = _fixture([0, 0, 3], [0, 0, 0])
    original = seed.copy()
    model = GrippedSectionChain(shaft, (_attachment(stiffness),))
    result = solve_gripped_chain(model, seed, _controls())
    root_shift = 3 / stiffness
    np.testing.assert_allclose(
        result.poses[:, 2, 3], np.array([0, 0.404, 1.01]) + root_shift, atol=2e-10
    )
    assert result.elastic_energy_j == pytest.approx(
        9 / (2 * 300) + 9 / (2 * stiffness), rel=1e-8
    )
    np.testing.assert_allclose(
        result.grip_responses[0].root_wrench, [0, 0, -3, 0, 0, 0], atol=2e-8
    )
    assert result.force_residual_n <= 1e-8
    assert result.moment_residual_nm <= 1e-9
    assert result.stability_status == "unqualified"
    np.testing.assert_array_equal(seed, original)


def test_torsional_grip_and_shaft_deflections_add() -> None:
    shaft, seed = _fixture([0, 0, 0], [0, 0, 0.2])
    result = solve_gripped_chain(
        GrippedSectionChain(shaft, (_attachment(),)), seed, _controls()
    )
    angles = [
        Rotation.from_matrix(pose[:3, :3]).as_rotvec()[2] for pose in result.poses
    ]
    np.testing.assert_allclose(
        angles, 0.2 / 10 + 0.2 / 2 * np.array([0, 0.4, 1]), atol=2e-9
    )
    np.testing.assert_allclose(
        result.grip_responses[0].anchor_wrench, [0, 0, 0, 0, 0, 0.2], atol=1e-8
    )


def test_nonplanar_supported_root_closes_world_force_and_moment() -> None:
    force, couple = np.array([0.04, -0.02, 0.03]), np.array([0.002, 0.003, -0.004])
    shaft, seed = _fixture(force, couple)
    model = GrippedSectionChain(shaft, (_attachment(),))
    result = solve_gripped_chain(model, seed, _controls())
    root, tip = result.poses[0], result.poses[-1]
    wrench = np.asarray(result.grip_responses[0].root_wrench)
    grip_force = root[:3, :3] @ wrench[:3]
    grip_moment = root[:3, :3] @ wrench[3:]
    np.testing.assert_allclose(grip_force + force, 0, atol=2e-8)
    np.testing.assert_allclose(
        grip_moment
        + np.cross(root[:3, 3], grip_force)
        + couple
        + np.cross(tip[:3, 3], force),
        0,
        atol=2e-8,
    )


def test_preloaded_finite_grip_tangent_differentiates_material_reactions() -> None:
    shaft, seed = _fixture([0.02, -0.01, 0.03], [0, 0, 0])
    attachment = _attachment()
    offset = exp_twist([0.03, -0.02, 0.01, 0.7, -0.4, 0.6])
    attachment = replace(attachment, anchor=replace(attachment.anchor, pose=offset))
    model = GrippedSectionChain(shaft, (attachment,))
    result = model.linearize(seed)
    direction = np.linspace(-0.2, 0.3, 18)
    step = 2e-6
    plus = np.array(
        [
            h @ exp_twist(step * d)
            for h, d in zip(seed, direction.reshape(-1, 6), strict=True)
        ]
    )
    minus = np.array(
        [
            h @ exp_twist(-step * d)
            for h, d in zip(seed, direction.reshape(-1, 6), strict=True)
        ]
    )
    numerical = (model.linearize(plus).residual - model.linearize(minus).residual) / (
        2 * step
    )
    np.testing.assert_allclose(
        moving_residual_jacobian(result) @ direction, numerical, rtol=2e-7, atol=2e-7
    )
    np.testing.assert_allclose(result.tangent, result.tangent.T, atol=2e-11)
    assert np.linalg.norm(moving_residual_jacobian(result) - result.tangent) > 1


def test_multiple_grips_add_and_can_support_an_interior_node() -> None:
    shaft, seed = _fixture([0, 0, 3], [0, 0, 0])
    first = _attachment(100, 1)
    first = replace(first, anchor=replace(first.anchor, pose=seed[1]))
    second = replace(
        first,
        grip=replace(
            first.grip,
            stiffness_factor=np.asarray(first.grip.stiffness_factor) * np.sqrt(2),
        ),
    )
    model = GrippedSectionChain(shaft, (first, second))
    result = solve_gripped_chain(model, seed, _controls())
    np.testing.assert_allclose(result.poses[:, 2, 3], [0.01, 0.41, 1.016], atol=2e-9)
    np.testing.assert_allclose(
        [item.root_wrench[2] for item in result.grip_responses], [-1, -2], atol=2e-8
    )


def test_rotating_loaded_rod_retains_mass_dependent_root_deflection() -> None:
    omega, stiffness = 10.0, 400.0
    frame = RotatingFrameState("observer", (omega, 0, 0), (0, 0, 0), (0, 0, 0))
    shaft, seed = _radial_rod(1, frame)
    model = GrippedSectionChain(shaft, (_attachment(stiffness),))
    result = solve_gripped_chain(model, seed, _controls())
    mass = 0.2 / 6 * np.array([[2, 1], [1, 2]])
    elastic = 1000 * np.array([[1, -1], [-1, 1]])
    expected = np.linalg.solve(
        elastic - omega**2 * mass + np.diag([stiffness, 0]), [-1000, 1000]
    )
    np.testing.assert_allclose(result.poses[:, 2, 3], expected, atol=2e-9)


def test_invalid_anchors_and_unbalanced_free_support_are_refused() -> None:
    shaft, seed = _fixture([0, 0, 3], [0, 0, 0])
    attachment = _attachment()
    with pytest.raises((TypeError, ValueError)):
        replace(attachment, node=True)
    with pytest.raises(ValueError, match="node"):
        GrippedSectionChain(shaft, (replace(attachment, node=99),))
    moving = replace(attachment.anchor, twist=[1, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="stationary"):
        GrippedSectionChain(shaft, (replace(attachment, anchor=moving),))
    free = replace(attachment.grip, stiffness_factor=np.zeros((6, 6)))
    model = GrippedSectionChain(shaft, (replace(attachment, grip=free),))
    with pytest.raises(RuntimeError):
        solve_gripped_chain(model, seed, _controls())


def test_observer_mismatch_and_accelerating_anchor_are_refused() -> None:
    frame = RotatingFrameState("other", (0, 0, 0), (0, 0, 0), (0, 0, 0))
    shaft, _ = _radial_rod(1, frame)
    with pytest.raises(ValueError, match="observer"):
        GrippedSectionChain(shaft, (_attachment(),))
    item = _attachment()
    with pytest.raises(ValueError, match="stationary"):
        replace(item, anchor=replace(item.anchor, twist_rate=[1, 0, 0, 0, 0, 0]))

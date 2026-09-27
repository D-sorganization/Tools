"""Tests for rigid frames, wrench power invariance, and modal projection (#5353).

Validates:
- Plücker wrench and twist transforms preserve power ``F·v + M·ω`` across frames
  (to 1e-12).
- Origin shift of wrench and twist reference point preserves power.
- M-orthogonal projection reproduces basis quadratic forms with zero residual in-span.
- Out-of-span projection reports non-zero residuals and preserves full energy ledger.
- Interoperability with Tools ``RigidTransform``, ``GripPortState``, and
  ``TrajectorySample``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.golf_club.grip_impedance import GripPortState, transform_grip_state
from shared.python.golf_club.modal_state import (
    ModalBasis,
    project_onto_basis,
)
from shared.python.golf_club.pre_impact_bundle import (
    Pose,
    PreImpactBundleError,
    grip_pose_from_delivery_sample,
    shift_twist_reference,
    shift_wrench_origin,
    twist_to_parent,
    wrench_to_parent,
)
from shared.python.golf_club.types import RigidTransform
from shared.python.swing_sim.delivery_interchange import TrajectorySample
from tests.shared.python.golf_club.test_pre_impact_bundle import _bundle

pytestmark = [pytest.mark.unit]


def _power(
    force: np.ndarray, moment: np.ndarray, linear: np.ndarray, angular: np.ndarray
) -> float:
    return float(np.dot(force, linear) + np.dot(moment, angular))


def test_wrench_power_is_invariant_across_world_head_and_grip() -> None:
    bundle = _bundle()
    rng = np.random.default_rng(9703)
    force, moment = rng.normal(size=3) * 100, rng.normal(size=3) * 5
    linear, angular = rng.normal(size=3) * 20, rng.normal(size=3) * 30
    reference = _power(force, moment, linear, angular)
    for source in ("world", "head", "grip"):
        for target in ("world", "head", "grip"):
            pose = bundle.pose_between(target, source)
            f2, m2 = wrench_to_parent(pose, force, moment)
            v2, w2 = twist_to_parent(pose, linear, angular)
            assert math.isclose(
                _power(f2, m2, v2, w2), reference, rel_tol=1e-12, abs_tol=0
            )


def test_moment_about_shifted_origin_preserves_power() -> None:
    rng = np.random.default_rng(1)
    force, moment = rng.normal(size=3) * 80, rng.normal(size=3) * 3
    linear, angular = rng.normal(size=3) * 15, rng.normal(size=3) * 25
    old, new = rng.normal(size=3), rng.normal(size=3)
    f2, m2 = shift_wrench_origin(force, moment, old, new)
    v2, w2 = shift_twist_reference(linear, angular, old, new)
    np.testing.assert_allclose(m2, moment + np.cross(old - new, force), rtol=1e-14)
    assert math.isclose(
        _power(f2, m2, v2, w2),
        _power(force, moment, linear, angular),
        rel_tol=1e-12,
        abs_tol=0,
    )


def test_hand_wrench_transform_matches_manual_composition() -> None:
    bundle = _bundle()
    hand = bundle.hand("lead")
    grip = bundle.pose_between("world", "grip")
    force, moment = bundle.hand_wrench_in("lead", "world")
    rotation = np.asarray(grip.rotation)
    at_grip_origin = hand.moment_n_m.value + np.cross(hand.origin_m, hand.force_n.value)
    expected_force = rotation @ hand.force_n.value
    expected_moment = rotation @ at_grip_origin + np.cross(
        grip.translation_m, expected_force
    )
    np.testing.assert_allclose(force, expected_force, rtol=1e-13)
    np.testing.assert_allclose(moment, expected_moment, rtol=1e-13)


def test_pose_inverse_and_composition_round_trip() -> None:
    bundle = _bundle()
    head_from_grip = bundle.pose_between("head", "grip")
    identity = head_from_grip.compose(head_from_grip.inverse())
    np.testing.assert_allclose(identity.rotation, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(identity.translation_m, 0.0, atol=1e-14)
    with pytest.raises(PreImpactBundleError):
        bundle.pose_between("world", "ball")


def _full_system() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = np.array([[2.0, 0.3, 0.0], [0.3, 1.5, 0.2], [0.0, 0.2, 1.0]])
    stiffness = np.array(
        [[4000.0, -1500.0, 0.0], [-1500.0, 3000.0, -800.0], [0.0, -800.0, 900.0]]
    )
    inverse_factor = np.linalg.inv(np.linalg.cholesky(mass))
    _, vectors = np.linalg.eigh(inverse_factor @ stiffness @ inverse_factor.T)
    shapes = inverse_factor.T @ vectors[:, :2]
    return mass, stiffness, shapes


def _basis(shapes: np.ndarray, mass: np.ndarray, stiffness: np.ndarray) -> ModalBasis:
    return ModalBasis(
        basis_id="synthetic-cantilever-3dof",
        version="1",
        normalization="mass",
        dimension=2,
        generalized_mass=shapes.T @ mass @ shapes,
        generalized_stiffness=shapes.T @ stiffness @ shapes,
    )


def test_in_span_projection_preserves_represented_energy() -> None:
    mass, stiffness, shapes = _full_system()
    basis = _basis(shapes, mass, stiffness)
    amplitudes, velocities = np.array([0.02, -0.004]), np.array([0.5, 0.1])
    result = project_onto_basis(
        basis, shapes, mass, stiffness, shapes @ amplitudes, shapes @ velocities
    )
    np.testing.assert_allclose(result.amplitudes, amplitudes, rtol=1e-12)
    np.testing.assert_allclose(result.velocities, velocities, rtol=1e-12)
    assert result.displacement_residual < 1e-12
    assert result.velocity_residual < 1e-12
    full = result.full_kinetic_energy_j + result.full_potential_energy_j
    represented = (
        result.represented_kinetic_energy_j + result.represented_potential_energy_j
    )
    assert math.isclose(represented, full, rel_tol=1e-12)
    expected = 0.5 * velocities @ velocities + 0.5 * amplitudes @ (
        basis.generalized_stiffness @ amplitudes
    )
    assert math.isclose(represented, expected, rel_tol=1e-12)


def test_out_of_span_projection_reports_residual() -> None:
    mass, stiffness, shapes = _full_system()
    basis = _basis(shapes, mass, stiffness)
    state = np.array([0.01, 0.0, 0.03])
    result = project_onto_basis(basis, shapes, mass, stiffness, state, state)
    assert result.displacement_residual > 1e-3
    assert result.represented_kinetic_energy_j <= result.full_kinetic_energy_j


def test_projection_refuses_inconsistent_basis() -> None:
    mass, stiffness, shapes = _full_system()
    basis = _basis(shapes, mass, stiffness)
    with pytest.raises(PreImpactBundleError):
        project_onto_basis(
            basis, shapes * 2.0, mass, stiffness, shapes[:, 0], shapes[:, 0]
        )


def test_bundle_modal_energy_uses_declared_quadratic_form() -> None:
    bundle = _bundle()
    q, qd = np.array([0.01, -0.002]), np.array([0.3, 0.05])
    expected = 0.5 * qd @ qd + 0.5 * (900.0 * q[0] ** 2 + 25000.0 * q[1] ** 2)
    assert math.isclose(bundle.shaft.modal_energy_j(), expected, rel_tol=1e-14)


def test_twist_convention_matches_tools_grip_state_transform() -> None:
    bundle = _bundle()
    grip = bundle.pose_between("world", "grip")
    linear, angular = np.array([1.0, -2.0, 0.5]), np.array([3.0, 0.2, -1.0])
    transform = RigidTransform(
        from_frame_id="grip",
        to_frame_id="world",
        rotation=np.asarray(grip.rotation),
        translation_m=np.asarray(grip.translation_m),
    )
    state = GripPortState(
        "grip", np.zeros(6), np.concatenate([linear, angular]), np.zeros(6)
    )
    mapped = np.asarray(transform_grip_state(state, transform).velocity)
    v2, w2 = twist_to_parent(grip, linear, angular)
    np.testing.assert_allclose(mapped, np.concatenate([v2, w2]), rtol=1e-13, atol=1e-15)


def test_grip_pose_adapts_tools_delivery_sample() -> None:
    half = math.sqrt(0.5)
    sample = TrajectorySample(
        time_s=0.001,
        position_m=(0.2, 0.9, -0.1),
        quaternion_wxyz=(half, 0.0, half, 0.0),
        linear_velocity_mps=(30.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, 40.0),
    )
    pose = grip_pose_from_delivery_sample(sample, world_frame_id="world")
    np.testing.assert_allclose(pose.rotation, sample.rotation_matrix(), atol=1e-15)
    np.testing.assert_allclose(pose.translation_m, sample.position_m)
    assert isinstance(pose, Pose)
    assert pose.frame_id == "grip"

"""Integration tests for math-primitives PyO3 bindings.

Tests validate that the PyO3 bridge correctly exposes rotation conversions,
quaternion operations, Pose6DOF transforms, and geometric primitives to Python.

Principles:
- TDD: Tests define the expected contract for the Python API.
- DbC: Roundtrip invariants verify mathematical correctness.
- DRY: Tests are parametrized where possible.

Requires: ``maturin develop --features python`` in ``rust_core/tools-core/``.
"""

# mypy: disable-error-code="union-attr"

from __future__ import annotations

import math
from typing import Any

import pytest

# Guard: skip entire module if the compiled wheel is not installed
tools_core = pytest.importorskip(
    "tools_core",
    reason="tools_core wheel not installed (run: maturin develop --features python)",
)

# math_primitives is registered as a submodule of tools_core
mp: Any = getattr(tools_core, "math_primitives", None)
if mp is None:
    pytest.skip(
        "math_primitives submodule not available in this tools_core build",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Rotation conversions
# ---------------------------------------------------------------------------


class TestEulerToRotationMatrix:
    """Test euler_to_rotation_matrix binding."""

    def test_identity_euler(self) -> None:
        """Zero Euler angles produce the 3x3 identity matrix."""
        r = mp.euler_to_rotation_matrix([0.0, 0.0, 0.0])
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(r[i][j] - expected) < 1e-12

    def test_output_shape(self) -> None:
        """Result must be a 3x3 nested list."""
        r = mp.euler_to_rotation_matrix([0.1, 0.2, 0.3])
        assert len(r) == 3
        assert all(len(row) == 3 for row in r)

    def test_orthonormality(self) -> None:
        """Rotation matrix R must satisfy R^T R = I."""
        r = mp.euler_to_rotation_matrix([0.5, -0.3, 1.2])
        for i in range(3):
            for j in range(3):
                dot = sum(r[k][i] * r[k][j] for k in range(3))
                expected = 1.0 if i == j else 0.0
                assert (
                    abs(dot - expected) < 1e-10
                ), f"Orthogonality violated at ({i},{j}): {dot}"


class TestRotationMatrixToEuler:
    """Test rotation_matrix_to_euler binding."""

    @pytest.mark.parametrize(
        "euler",
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            [-0.5, 0.0, 1.0],
            [0.0, 0.0, math.pi / 4],
        ],
    )
    def test_roundtrip(self, euler: list[float]) -> None:
        """euler → rotmat → euler must recover the original angles."""
        r = mp.euler_to_rotation_matrix(euler)
        recovered = mp.rotation_matrix_to_euler(r)
        for i in range(3):
            assert (
                abs(recovered[i] - euler[i]) < 1e-10
            ), f"Roundtrip failed at index {i}: {recovered[i]} != {euler[i]}"


# ---------------------------------------------------------------------------
# Quaternion conversions
# ---------------------------------------------------------------------------


class TestEulerToQuaternion:
    """Test euler_to_quaternion binding."""

    def test_identity(self) -> None:
        """Zero Euler angles produce the identity quaternion [1, 0, 0, 0]."""
        q = mp.euler_to_quaternion([0.0, 0.0, 0.0])
        assert abs(q[0] - 1.0) < 1e-12
        assert abs(q[1]) < 1e-12
        assert abs(q[2]) < 1e-12
        assert abs(q[3]) < 1e-12

    def test_unit_norm(self) -> None:
        """All quaternion results must have unit norm."""
        q = mp.euler_to_quaternion([0.5, -0.3, 1.2])
        norm = math.sqrt(sum(c * c for c in q))
        assert abs(norm - 1.0) < 1e-10


class TestQuaternionToEuler:
    """Test quaternion_to_euler binding."""

    @pytest.mark.parametrize(
        "euler",
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
            [-0.4, 0.0, 0.8],
        ],
    )
    def test_roundtrip_via_quaternion(self, euler: list[float]) -> None:
        """euler → quaternion → euler must recover original angles."""
        q = mp.euler_to_quaternion(euler)
        recovered = mp.quaternion_to_euler(q)
        for i in range(3):
            assert abs(recovered[i] - euler[i]) < 1e-10


class TestQuaternionToRotationMatrix:
    """Test quaternion_to_rotation_matrix binding."""

    def test_identity_quaternion(self) -> None:
        """Identity quaternion [1,0,0,0] must produce the identity matrix."""
        r = mp.quaternion_to_rotation_matrix([1.0, 0.0, 0.0, 0.0])
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(r[i][j] - expected) < 1e-12


# ---------------------------------------------------------------------------
# Quaternion operations
# ---------------------------------------------------------------------------


class TestQuaternionMultiply:
    """Test quaternion_multiply binding."""

    def test_identity_multiply(self) -> None:
        """Multiplying by identity must return the original quaternion."""
        identity = [1.0, 0.0, 0.0, 0.0]
        q = [0.707, 0.707, 0.0, 0.0]
        result = mp.quaternion_multiply(identity, q)
        for i in range(4):
            assert abs(result[i] - q[i]) < 1e-6

    def test_inverse_multiply(self) -> None:
        """q * q_inv must produce the identity quaternion."""
        q = [0.5, 0.5, 0.5, 0.5]
        q_inv = mp.quaternion_inverse(q)
        result = mp.quaternion_multiply(q, q_inv)
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1]) < 1e-10
        assert abs(result[2]) < 1e-10
        assert abs(result[3]) < 1e-10


class TestQuaternionInverse:
    """Test quaternion_inverse binding."""

    def test_conjugate_of_identity(self) -> None:
        """Inverse of identity is identity."""
        result = mp.quaternion_inverse([1.0, 0.0, 0.0, 0.0])
        assert abs(result[0] - 1.0) < 1e-12

    def test_sign_flip(self) -> None:
        """Inverse of unit quaternion negates the vector part."""
        q = [0.5, 0.5, 0.5, 0.5]
        q_inv = mp.quaternion_inverse(q)
        assert abs(q_inv[0] - q[0]) < 1e-12
        assert abs(q_inv[1] + q[1]) < 1e-12
        assert abs(q_inv[2] + q[2]) < 1e-12
        assert abs(q_inv[3] + q[3]) < 1e-12


class TestSlerp:
    """Test slerp binding."""

    def test_slerp_at_zero(self) -> None:
        """slerp(q1, q2, 0) must return q1."""
        q1 = [1.0, 0.0, 0.0, 0.0]
        q2 = [0.707, 0.707, 0.0, 0.0]
        result = mp.slerp(q1, q2, 0.0)
        for i in range(4):
            assert abs(result[i] - q1[i]) < 1e-10

    def test_slerp_at_one(self) -> None:
        """slerp(q1, q2, 1) must return q2."""
        q1 = [1.0, 0.0, 0.0, 0.0]
        q2 = [0.707, 0.707, 0.0, 0.0]
        result = mp.slerp(q1, q2, 1.0)
        for i in range(4):
            assert abs(result[i] - q2[i]) < 1e-6

    def test_slerp_midpoint_unit_norm(self) -> None:
        """Midpoint interpolation must produce a unit quaternion."""
        q1 = [1.0, 0.0, 0.0, 0.0]
        q2 = [0.0, 1.0, 0.0, 0.0]
        result = mp.slerp(q1, q2, 0.5)
        norm = math.sqrt(sum(c * c for c in result))
        assert abs(norm - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Axis-angle conversions
# ---------------------------------------------------------------------------


class TestAxisAngleToRotationMatrix:
    """Test axis_angle_to_rotation_matrix binding."""

    def test_zero_angle(self) -> None:
        """Zero rotation angle must produce identity matrix."""
        r = mp.axis_angle_to_rotation_matrix([0.0, 0.0, 1.0], 0.0)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(r[i][j] - expected) < 1e-12

    def test_90_deg_about_z(self) -> None:
        """90° about z-axis: x→y, y→-x, z→z."""
        r = mp.axis_angle_to_rotation_matrix([0.0, 0.0, 1.0], math.pi / 2)
        # R * [1,0,0] should give [0,1,0]
        x_out = [r[0][0], r[1][0], r[2][0]]
        assert abs(x_out[0]) < 1e-10
        assert abs(x_out[1] - 1.0) < 1e-10
        assert abs(x_out[2]) < 1e-10


# ---------------------------------------------------------------------------
# Pose6DOF
# ---------------------------------------------------------------------------


class TestPose6DOF:
    """Test Pose6DOF class binding."""

    def test_construction(self) -> None:
        """Pose6DOF must be constructable and expose position/euler_angles."""
        pose = mp.Pose6DOF([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        assert abs(pose.x - 1.0) < 1e-12
        assert abs(pose.y - 2.0) < 1e-12
        assert abs(pose.z - 3.0) < 1e-12

    def test_euler_angles_stored(self) -> None:
        """Euler angles must be stored correctly."""
        euler = [0.1, 0.2, 0.3]
        pose = mp.Pose6DOF([0.0, 0.0, 0.0], euler)
        assert abs(pose.roll - euler[0]) < 1e-12
        assert abs(pose.pitch - euler[1]) < 1e-12
        assert abs(pose.yaw - euler[2]) < 1e-12

    def test_translate(self) -> None:
        """Translate must add offset to position."""
        pose = mp.Pose6DOF([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        moved = pose.translate([10.0, 0.0, 0.0])
        assert abs(moved.x - 11.0) < 1e-12
        assert abs(moved.y - 2.0) < 1e-12
        assert abs(moved.z - 3.0) < 1e-12

    def test_inverse_compose_identity(self) -> None:
        """pose.compose(pose.inverse()) must approximate the identity pose."""
        pose = mp.Pose6DOF([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        inv = pose.inverse()
        identity_ish = pose.compose(inv)
        assert abs(identity_ish.x) < 1e-8
        assert abs(identity_ish.y) < 1e-8
        assert abs(identity_ish.z) < 1e-8

    def test_transform_point_identity(self) -> None:
        """Identity pose must not change the point."""
        pose = mp.Pose6DOF([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        pt = pose.transform_point([5.0, 6.0, 7.0])
        assert abs(pt[0] - 5.0) < 1e-12
        assert abs(pt[1] - 6.0) < 1e-12
        assert abs(pt[2] - 7.0) < 1e-12

    def test_to_quaternion_identity(self) -> None:
        """Identity pose must produce identity quaternion."""
        pose = mp.Pose6DOF([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        q = pose.to_quaternion()
        assert abs(q[0] - 1.0) < 1e-12

    def test_repr(self) -> None:
        """repr must contain 'Pose6DOF'."""
        pose = mp.Pose6DOF([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        assert "Pose6DOF" in repr(pose)


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------


class TestSphereDistance:
    """Test sphere_sphere_distance binding."""

    def test_non_overlapping(self) -> None:
        """Two separated spheres must have positive distance."""
        dist, pa, pb = mp.sphere_sphere_distance(
            [0.0, 0.0, 0.0], 1.0, [5.0, 0.0, 0.0], 1.0
        )
        assert dist == pytest.approx(3.0, abs=1e-10)

    def test_touching(self) -> None:
        """Two touching spheres must have distance 0."""
        dist, pa, pb = mp.sphere_sphere_distance(
            [0.0, 0.0, 0.0], 1.0, [2.0, 0.0, 0.0], 1.0
        )
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_overlapping(self) -> None:
        """Two overlapping spheres must have negative distance."""
        dist, pa, pb = mp.sphere_sphere_distance(
            [0.0, 0.0, 0.0], 1.0, [1.0, 0.0, 0.0], 1.0
        )
        assert dist < 0.0

    def test_closest_points_type(self) -> None:
        """Closest points must be length-3 lists."""
        _, pa, pb = mp.sphere_sphere_distance(
            [0.0, 0.0, 0.0], 1.0, [5.0, 0.0, 0.0], 1.0
        )
        assert len(pa) == 3
        assert len(pb) == 3


class TestCheckCollisionSpheres:
    """Test check_collision_spheres binding."""

    def test_no_collision(self) -> None:
        """Distant spheres must not collide."""
        assert not mp.check_collision_spheres(
            [0.0, 0.0, 0.0], 1.0, [10.0, 0.0, 0.0], 1.0, 0.0
        )

    def test_collision(self) -> None:
        """Overlapping spheres must collide."""
        assert mp.check_collision_spheres(
            [0.0, 0.0, 0.0], 1.0, [1.0, 0.0, 0.0], 1.0, 0.0
        )

    def test_margin_triggers_near_collision(self) -> None:
        """Marginally separated spheres must collide with sufficient margin."""
        # Centers 2.5 apart, radii 1.0 each → gap = 0.5
        assert not mp.check_collision_spheres(
            [0.0, 0.0, 0.0], 1.0, [2.5, 0.0, 0.0], 1.0, 0.0
        )
        assert mp.check_collision_spheres(
            [0.0, 0.0, 0.0], 1.0, [2.5, 0.0, 0.0], 1.0, 1.0
        )

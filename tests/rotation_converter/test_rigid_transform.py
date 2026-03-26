"""TDD tests for frame-aware rigid body transformations.

Tests cover:
- RigidTransform: frame-aware SE(3) wrapper with source/target labels
- Frame compatibility checking: composition, application errors
- Comprehensive conversions: SE(3) to/from every representation
- Body/space frame twist conversions via adjoint
- Point and vector transformation with frame validation
- FrameError exception for incompatible operations

Written BEFORE implementation (TDD).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rotation_converter._contracts import PreconditionError
from rotation_converter.converter import Rotation
from rotation_converter.rigid_transform import (
    FrameError,
    RigidTransform,
)

ATOL = 1e-6


# ===========================================================================
# FrameError exception
# ===========================================================================


class TestFrameError:
    """FrameError is raised on incompatible frame operations."""

    def test_is_exception(self) -> None:
        assert issubclass(FrameError, Exception)

    def test_message(self) -> None:
        err = FrameError("body", "world", "compose", "camera")
        assert "body" in str(err)
        assert "world" in str(err)


# ===========================================================================
# RigidTransform construction
# ===========================================================================


class TestRigidTransformConstruction:
    """Factory methods produce valid frame-aware transforms."""

    def test_identity(self) -> None:
        T = RigidTransform.identity("world")
        np.testing.assert_allclose(T.as_matrix(), np.eye(4), atol=ATOL)
        assert T.source_frame == "world"
        assert T.target_frame == "world"

    def test_from_matrix(self) -> None:
        M = np.eye(4)
        M[:3, 3] = [1, 2, 3]
        T = RigidTransform.from_matrix(M, source="body", target="world")
        assert T.source_frame == "body"
        assert T.target_frame == "world"
        np.testing.assert_allclose(T.as_matrix(), M, atol=ATOL)

    def test_from_matrix_rejects_non_SE3(self) -> None:
        M = np.eye(4)
        M[:3, :3] *= 2  # Not SO(3)
        with pytest.raises(PreconditionError):
            RigidTransform.from_matrix(M, source="a", target="b")

    def test_from_rotation_translation(self) -> None:
        R = np.eye(3)
        p = np.array([1.0, 2.0, 3.0])
        T = RigidTransform.from_rotation_translation(
            R, p, source="body", target="world"
        )
        np.testing.assert_allclose(T.translation, p, atol=ATOL)
        np.testing.assert_allclose(T.rotation_matrix, R, atol=ATOL)

    def test_from_rotation_object(self) -> None:
        rot = Rotation.from_axis_angle([0, 0, 1], math.pi / 2)
        p = np.array([1.0, 0.0, 0.0])
        T = RigidTransform.from_rotation(rot, p, source="tool", target="base")
        np.testing.assert_allclose(
            T.rotation_matrix, rot.as_rotation_matrix(), atol=ATOL
        )
        assert T.source_frame == "tool"
        assert T.target_frame == "base"

    def test_from_quaternion_translation(self) -> None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        p = np.array([5.0, 6.0, 7.0])
        T = RigidTransform.from_quaternion_translation(
            q, p, source="sensor", target="base"
        )
        np.testing.assert_allclose(T.translation, p, atol=ATOL)
        q_out = T.as_quaternion_translation()
        np.testing.assert_allclose(q_out[0], q, atol=ATOL)
        np.testing.assert_allclose(q_out[1], p, atol=ATOL)

    def test_from_euler_translation(self) -> None:
        p = np.array([1.0, 0.0, 0.0])
        T = RigidTransform.from_euler_translation(
            0.1, 0.2, 0.3, p, convention="xyz", source="a", target="b"
        )
        assert T.source_frame == "a"
        assert T.target_frame == "b"
        euler, p_out = T.as_euler_translation("xyz")
        assert len(euler) == 3
        np.testing.assert_allclose(p_out, p, atol=ATOL)

    def test_from_axis_angle_translation(self) -> None:
        axis = np.array([0.0, 0.0, 1.0])
        angle = math.pi / 4
        p = np.array([2.0, 3.0, 0.0])
        T = RigidTransform.from_axis_angle_translation(
            axis, angle, p, source="a", target="b"
        )
        ax_out, ang_out, p_out = T.as_axis_angle_translation()
        np.testing.assert_allclose(ax_out, axis, atol=ATOL)
        assert abs(ang_out - angle) < ATOL
        np.testing.assert_allclose(p_out, p, atol=ATOL)

    def test_from_rodrigues_translation(self) -> None:
        r = np.array([0.0, 0.0, 0.5])
        p = np.array([1.0, 2.0, 3.0])
        T = RigidTransform.from_rodrigues_translation(r, p, source="a", target="b")
        r_out, p_out = T.as_rodrigues_translation()
        np.testing.assert_allclose(r_out, r, atol=ATOL)
        np.testing.assert_allclose(p_out, p, atol=ATOL)

    def test_from_twist(self) -> None:
        """Create from twist vector + angle."""
        # Pure rotation about z by pi/2
        twist = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        theta = math.pi / 2
        T = RigidTransform.from_twist(twist, theta, source="a", target="b")
        R = T.rotation_matrix
        expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected_R, atol=ATOL)

    def test_from_screw(self) -> None:
        """Create from screw axis parameters."""
        screw = {
            "axis": np.array([0.0, 0.0, 1.0]),
            "point": np.array([0.0, 0.0, 0.0]),
            "pitch": 0.0,
        }
        theta = math.pi / 4
        T = RigidTransform.from_screw(screw, theta, source="a", target="b")
        assert T.source_frame == "a"
        assert T.target_frame == "b"

    def test_pure_translation(self) -> None:
        p = np.array([1.0, 2.0, 3.0])
        T = RigidTransform.pure_translation(p, source="a", target="b")
        np.testing.assert_allclose(T.rotation_matrix, np.eye(3), atol=ATOL)
        np.testing.assert_allclose(T.translation, p, atol=ATOL)

    def test_pure_rotation(self) -> None:
        rot = Rotation.from_axis_angle([0, 0, 1], math.pi / 2)
        T = RigidTransform.pure_rotation(rot, source="a", target="b")
        np.testing.assert_allclose(T.translation, [0, 0, 0], atol=ATOL)
        np.testing.assert_allclose(
            T.rotation_matrix, rot.as_rotation_matrix(), atol=ATOL
        )


# ===========================================================================
# Output conversions — "convert everything imaginable"
# ===========================================================================


class TestRigidTransformOutputConversions:
    """Every output representation should be available."""

    @pytest.fixture()
    def sample_transform(self) -> RigidTransform:
        """A non-trivial rigid transform for conversion testing."""
        R = Rotation.from_euler(0.3, 0.5, 0.7, "xyz").as_rotation_matrix()
        p = np.array([1.0, 2.0, 3.0])
        return RigidTransform.from_rotation_translation(
            R, p, source="body", target="world"
        )

    def test_as_matrix(self, sample_transform: RigidTransform) -> None:
        M = sample_transform.as_matrix()
        assert M.shape == (4, 4)
        np.testing.assert_allclose(M[3, :], [0, 0, 0, 1], atol=ATOL)

    def test_as_rotation_translation(self, sample_transform: RigidTransform) -> None:
        R, p = sample_transform.as_rotation_translation()
        assert R.shape == (3, 3)
        assert p.shape == (3,)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)

    def test_as_rotation(self, sample_transform: RigidTransform) -> None:
        rot = sample_transform.as_rotation()
        assert isinstance(rot, Rotation)
        np.testing.assert_allclose(
            rot.as_rotation_matrix(),
            sample_transform.rotation_matrix,
            atol=ATOL,
        )

    def test_as_quaternion_translation(self, sample_transform: RigidTransform) -> None:
        q, p = sample_transform.as_quaternion_translation()
        assert q.shape == (4,)
        assert p.shape == (3,)
        assert abs(np.linalg.norm(q) - 1.0) < ATOL

    def test_as_euler_translation(self, sample_transform: RigidTransform) -> None:
        euler, p = sample_transform.as_euler_translation("xyz")
        assert len(euler) == 3
        assert p.shape == (3,)

    def test_as_axis_angle_translation(self, sample_transform: RigidTransform) -> None:
        axis, angle, p = sample_transform.as_axis_angle_translation()
        assert axis.shape == (3,)
        assert abs(np.linalg.norm(axis) - 1.0) < ATOL
        assert angle >= 0
        assert p.shape == (3,)

    def test_as_rodrigues_translation(self, sample_transform: RigidTransform) -> None:
        r, p = sample_transform.as_rodrigues_translation()
        assert r.shape == (3,)
        assert p.shape == (3,)

    def test_as_twist(self, sample_transform: RigidTransform) -> None:
        twist, theta = sample_transform.as_twist()
        assert twist.shape == (6,)
        assert theta >= 0

    def test_as_screw(self, sample_transform: RigidTransform) -> None:
        screw = sample_transform.as_screw()
        assert "axis" in screw
        assert "point" in screw
        assert "pitch" in screw
        assert "theta" in screw

    def test_identity_as_twist(self) -> None:
        T = RigidTransform.identity("a")
        twist, theta = T.as_twist()
        assert abs(theta) < ATOL


# ===========================================================================
# Roundtrip conversions — every path should survive a roundtrip
# ===========================================================================


class TestRoundtripConversions:
    """Constructing from a representation and exporting back should match."""

    def test_matrix_roundtrip(self) -> None:
        M = np.eye(4)
        angle = math.pi / 6
        c, s = math.cos(angle), math.sin(angle)
        M[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        M[:3, 3] = [1, 2, 3]
        T = RigidTransform.from_matrix(M, source="a", target="b")
        np.testing.assert_allclose(T.as_matrix(), M, atol=ATOL)

    def test_quaternion_translation_roundtrip(self) -> None:
        q = np.array([0.7071068, 0.0, 0.7071068, 0.0])
        q = q / np.linalg.norm(q)
        p = np.array([10.0, -5.0, 3.0])
        T = RigidTransform.from_quaternion_translation(q, p, source="a", target="b")
        q_out, p_out = T.as_quaternion_translation()
        # Handle double-cover
        if np.dot(q, q_out) < 0:
            q_out = -q_out
        np.testing.assert_allclose(q_out, q, atol=ATOL)
        np.testing.assert_allclose(p_out, p, atol=ATOL)

    def test_euler_translation_roundtrip(self) -> None:
        a, b, c_angle = 0.3, 0.5, 0.7
        p = np.array([1.0, 2.0, 3.0])
        T = RigidTransform.from_euler_translation(
            a, b, c_angle, p, convention="xyz", source="a", target="b"
        )
        euler, p_out = T.as_euler_translation("xyz")
        np.testing.assert_allclose(euler, [a, b, c_angle], atol=ATOL)
        np.testing.assert_allclose(p_out, p, atol=ATOL)

    def test_axis_angle_translation_roundtrip(self) -> None:
        axis = np.array([1.0, 0.0, 0.0])
        angle = math.pi / 3
        p = np.array([4.0, 5.0, 6.0])
        T = RigidTransform.from_axis_angle_translation(
            axis, angle, p, source="a", target="b"
        )
        ax_out, ang_out, p_out = T.as_axis_angle_translation()
        np.testing.assert_allclose(ax_out, axis, atol=ATOL)
        assert abs(ang_out - angle) < ATOL
        np.testing.assert_allclose(p_out, p, atol=ATOL)

    def test_rodrigues_translation_roundtrip(self) -> None:
        r = np.array([0.1, 0.2, 0.3])
        p = np.array([7.0, 8.0, 9.0])
        T = RigidTransform.from_rodrigues_translation(r, p, source="a", target="b")
        r_out, p_out = T.as_rodrigues_translation()
        np.testing.assert_allclose(r_out, r, atol=ATOL)
        np.testing.assert_allclose(p_out, p, atol=ATOL)

    def test_twist_roundtrip(self) -> None:
        """from_twist -> as_twist -> from_twist should reconstruct same SE(3)."""
        twist = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        theta = math.pi / 4
        T1 = RigidTransform.from_twist(twist, theta, source="a", target="b")
        twist_out, theta_out = T1.as_twist()
        T2 = RigidTransform.from_twist(twist_out, theta_out, source="a", target="b")
        np.testing.assert_allclose(T1.as_matrix(), T2.as_matrix(), atol=ATOL)

    @pytest.mark.parametrize("trial", range(10))
    def test_random_matrix_roundtrip(self, trial: int) -> None:
        rng = np.random.default_rng(seed=42 + trial)
        # Random rotation via QR decomposition
        A = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        p = rng.standard_normal(3) * 5
        M = np.eye(4)
        M[:3, :3] = Q
        M[:3, 3] = p
        T = RigidTransform.from_matrix(M, source="a", target="b")
        np.testing.assert_allclose(T.as_matrix(), M, atol=ATOL)


# ===========================================================================
# Frame-checked composition
# ===========================================================================


class TestFrameCheckedComposition:
    """Composing transforms must obey frame chain rules."""

    def test_compose_compatible_frames(self) -> None:
        """T_world_body @ T_body_tool = T_world_tool."""
        T_wb = RigidTransform.pure_translation([1, 0, 0], source="body", target="world")
        T_bt = RigidTransform.pure_translation([0, 1, 0], source="tool", target="body")
        T_wt = T_wb.compose(T_bt)
        assert T_wt.source_frame == "tool"
        assert T_wt.target_frame == "world"
        np.testing.assert_allclose(T_wt.translation, [1, 1, 0], atol=ATOL)

    def test_compose_incompatible_raises_frame_error(self) -> None:
        """T_A_B @ T_C_D should raise FrameError because B != C."""
        T1 = RigidTransform.identity("a")  # a -> a
        T2 = RigidTransform.pure_translation([1, 0, 0], source="c", target="d")
        # T1 is a->a, T2 is c->d.  T1.source_frame is 'a', T2.target_frame is 'd'
        # For T1.compose(T2): T1 @ T2 means T1.source must == T2.target for chain
        # T1 maps a->a, T2 maps c->d
        # T1 @ T2 requires: T1.source_frame == T2.target_frame? No.
        # Convention: T_{target}^{source}: T_wb maps body->world
        # Composition: T_wb @ T_bt = T_wt. So self.source == other.target
        with pytest.raises(FrameError):
            T1.compose(T2)

    def test_matmul_operator(self) -> None:
        """The @ operator should delegate to compose."""
        T1 = RigidTransform.pure_translation([1, 0, 0], source="body", target="world")
        T2 = RigidTransform.pure_translation([0, 1, 0], source="tool", target="body")
        T3 = T1 @ T2
        assert isinstance(T3, RigidTransform)
        assert T3.source_frame == "tool"
        assert T3.target_frame == "world"

    def test_matmul_incompatible_raises(self) -> None:
        T1 = RigidTransform.pure_translation([1, 0, 0], source="a", target="b")
        T2 = RigidTransform.pure_translation([0, 1, 0], source="c", target="d")
        with pytest.raises(FrameError):
            T1 @ T2

    def test_compose_chain_of_three(self) -> None:
        """T_w_b @ T_b_t @ T_t_e = T_w_e."""
        T_wb = RigidTransform.pure_translation([1, 0, 0], source="body", target="world")
        T_bt = RigidTransform.pure_translation([0, 1, 0], source="tool", target="body")
        T_te = RigidTransform.pure_translation([0, 0, 1], source="end", target="tool")
        T_we = T_wb @ T_bt @ T_te
        assert T_we.source_frame == "end"
        assert T_we.target_frame == "world"
        np.testing.assert_allclose(T_we.translation, [1, 1, 1], atol=ATOL)


# ===========================================================================
# Inverse with frame swap
# ===========================================================================


class TestInverse:
    """Inverse flips source and target frames."""

    def test_inverse_swaps_frames(self) -> None:
        T = RigidTransform.pure_translation([1, 2, 3], source="body", target="world")
        T_inv = T.inverse()
        assert T_inv.source_frame == "world"
        assert T_inv.target_frame == "body"

    def test_inverse_compose_gives_identity(self) -> None:
        R = Rotation.from_euler(0.3, 0.5, 0.7, "xyz").as_rotation_matrix()
        p = np.array([1.0, 2.0, 3.0])
        T = RigidTransform.from_rotation_translation(
            R, p, source="body", target="world"
        )
        T_inv = T.inverse()
        result = T @ T_inv
        np.testing.assert_allclose(result.as_matrix(), np.eye(4), atol=ATOL)
        assert result.source_frame == "world"
        assert result.target_frame == "world"

    def test_inverse_of_inverse(self) -> None:
        T = RigidTransform.from_rotation_translation(
            np.eye(3), [1, 2, 3], source="a", target="b"
        )
        T2 = T.inverse().inverse()
        np.testing.assert_allclose(T2.as_matrix(), T.as_matrix(), atol=ATOL)
        assert T2.source_frame == "a"
        assert T2.target_frame == "b"


# ===========================================================================
# Point and vector transformations
# ===========================================================================


class TestPointVectorTransform:
    """Apply transforms to points and vectors."""

    def test_apply_point_pure_translation(self) -> None:
        T = RigidTransform.pure_translation([1, 2, 3], source="body", target="world")
        p_body = np.array([0.0, 0.0, 0.0])
        p_world = T.apply_point(p_body)
        np.testing.assert_allclose(p_world, [1, 2, 3], atol=ATOL)

    def test_apply_point_rotation_and_translation(self) -> None:
        """90 degree rotation about z then translate by [1,0,0]."""
        angle = math.pi / 2
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        p = np.array([1.0, 0.0, 0.0])
        T = RigidTransform.from_rotation_translation(
            R, p, source="body", target="world"
        )
        # Point at [1,0,0] in body -> R @ [1,0,0] + [1,0,0] = [0,1,0] + [1,0,0] = [1,1,0]
        result = T.apply_point(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [1.0, 1.0, 0.0], atol=ATOL)

    def test_apply_vector_ignores_translation(self) -> None:
        T = RigidTransform.pure_translation(
            [100, 200, 300], source="body", target="world"
        )
        v_body = np.array([1.0, 0.0, 0.0])
        v_world = T.apply_vector(v_body)
        np.testing.assert_allclose(v_world, [1, 0, 0], atol=ATOL)

    def test_apply_vector_rotates(self) -> None:
        angle = math.pi / 2
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T = RigidTransform.from_rotation_translation(
            R, [0, 0, 0], source="body", target="world"
        )
        v = T.apply_vector(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=ATOL)

    def test_apply_points_batch(self) -> None:
        """Transform multiple points at once."""
        T = RigidTransform.pure_translation([1, 0, 0], source="body", target="world")
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        result = T.apply_points(points)
        expected = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=float)
        np.testing.assert_allclose(result, expected, atol=ATOL)


# ===========================================================================
# Body / Space frame twist conversions
# ===========================================================================


class TestBodySpaceTwistConversions:
    """Convert twists between body and space frames via adjoint."""

    def test_body_twist_of_pure_rotation(self) -> None:
        """Pure rotation about z: body twist should have omega_z component."""
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T = RigidTransform.from_rotation_translation(
            R, [0, 0, 0], source="body", target="space"
        )
        Vb = T.body_twist()
        assert Vb.shape == (6,)
        # For rotation about z by pi/4, body twist omega should be [0,0,pi/4]
        # (the MatrixLog gives the twist*theta)

    def test_space_twist_of_pure_rotation(self) -> None:
        """For rotation about origin, body and space twists have same omega."""
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T = RigidTransform.from_rotation_translation(
            R, [0, 0, 0], source="body", target="space"
        )
        Vs = T.space_twist()
        Vb = T.body_twist()
        # For rotation about origin with no translation, body and space
        # twists have same angular component
        np.testing.assert_allclose(Vs[:3], Vb[:3], atol=ATOL)

    def test_body_to_space_twist(self) -> None:
        """Convert a body-frame twist to space-frame twist via adjoint."""
        R = Rotation.from_axis_angle([0, 0, 1], math.pi / 4).as_rotation_matrix()
        p = np.array([1.0, 0.0, 0.0])
        T = RigidTransform.from_rotation_translation(
            R, p, source="body", target="space"
        )
        # Some arbitrary body twist
        Vb = np.array([0.0, 0.0, 1.0, 0.5, 0.0, 0.0])
        Vs = T.body_to_space_twist(Vb)
        assert Vs.shape == (6,)
        # Vs = Ad_T @ Vb
        # Verify by converting back
        Vb_back = T.space_to_body_twist(Vs)
        np.testing.assert_allclose(Vb_back, Vb, atol=ATOL)

    def test_space_to_body_twist(self) -> None:
        """Convert a space-frame twist to body-frame twist."""
        R = Rotation.from_euler(0.3, 0.5, 0.7, "xyz").as_rotation_matrix()
        p = np.array([1.0, 2.0, 3.0])
        T = RigidTransform.from_rotation_translation(
            R, p, source="body", target="space"
        )
        Vs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        Vb = T.space_to_body_twist(Vs)
        Vs_back = T.body_to_space_twist(Vb)
        np.testing.assert_allclose(Vs_back, Vs, atol=ATOL)

    def test_body_space_twist_relationship(self) -> None:
        """Vs = Ad_T * Vb for any transform."""
        from rotation_converter.twist_screw import adjoint_representation

        R = Rotation.from_euler(0.1, 0.2, 0.3, "zyx").as_rotation_matrix()
        p = np.array([3.0, 4.0, 5.0])
        T = RigidTransform.from_rotation_translation(
            R, p, source="body", target="space"
        )
        Vb = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        Vs = T.body_to_space_twist(Vb)
        # Manually compute
        Ad = adjoint_representation(T.as_matrix())
        Vs_expected = Ad @ Vb
        np.testing.assert_allclose(Vs, Vs_expected, atol=ATOL)


# ===========================================================================
# Wrench transformations (co-adjoint)
# ===========================================================================


class TestWrenchTransformations:
    """Transform wrenches between body and space frames."""

    def test_body_to_space_wrench(self) -> None:
        R = Rotation.from_axis_angle([0, 0, 1], math.pi / 4).as_rotation_matrix()
        p = np.array([1.0, 0.0, 0.0])
        T = RigidTransform.from_rotation_translation(
            R, p, source="body", target="space"
        )
        Fb = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        Fs = T.body_to_space_wrench(Fb)
        assert Fs.shape == (6,)
        # Round-trip
        Fb_back = T.space_to_body_wrench(Fs)
        np.testing.assert_allclose(Fb_back, Fb, atol=ATOL)

    def test_wrench_round_trip(self) -> None:
        R = Rotation.from_euler(0.3, 0.5, 0.7, "xyz").as_rotation_matrix()
        p = np.array([1.0, 2.0, 3.0])
        T = RigidTransform.from_rotation_translation(
            R, p, source="body", target="space"
        )
        Fs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        Fb = T.space_to_body_wrench(Fs)
        Fs_back = T.body_to_space_wrench(Fb)
        np.testing.assert_allclose(Fs_back, Fs, atol=ATOL)


# ===========================================================================
# Properties and introspection
# ===========================================================================


class TestProperties:
    """Basic property accessors."""

    def test_source_frame(self) -> None:
        T = RigidTransform.identity("world")
        assert T.source_frame == "world"

    def test_target_frame(self) -> None:
        T = RigidTransform.pure_translation([1, 0, 0], source="body", target="world")
        assert T.target_frame == "world"

    def test_translation_property(self) -> None:
        T = RigidTransform.pure_translation([1, 2, 3], source="a", target="b")
        np.testing.assert_allclose(T.translation, [1, 2, 3], atol=ATOL)

    def test_rotation_matrix_property(self) -> None:
        T = RigidTransform.identity("a")
        np.testing.assert_allclose(T.rotation_matrix, np.eye(3), atol=ATOL)

    def test_is_identity(self) -> None:
        T = RigidTransform.identity("a")
        assert T.is_identity()

    def test_is_not_identity(self) -> None:
        T = RigidTransform.pure_translation([1, 0, 0], source="a", target="b")
        assert not T.is_identity()

    def test_is_pure_translation(self) -> None:
        T = RigidTransform.pure_translation([1, 2, 3], source="a", target="b")
        assert T.is_pure_translation()

    def test_is_pure_rotation(self) -> None:
        rot = Rotation.from_axis_angle([0, 0, 1], math.pi / 4)
        T = RigidTransform.pure_rotation(rot, source="a", target="b")
        assert T.is_pure_rotation()

    def test_repr(self) -> None:
        T = RigidTransform.identity("world")
        r = repr(T)
        assert "world" in r
        assert "RigidTransform" in r


# ===========================================================================
# Immutability
# ===========================================================================


class TestImmutability:
    """RigidTransform should be immutable — output copies only."""

    def test_translation_returns_copy(self) -> None:
        T = RigidTransform.pure_translation([1, 2, 3], source="a", target="b")
        p = T.translation
        p[0] = 999
        np.testing.assert_allclose(T.translation, [1, 2, 3], atol=ATOL)

    def test_rotation_matrix_returns_copy(self) -> None:
        T = RigidTransform.identity("a")
        R = T.rotation_matrix
        R[0, 0] = 999
        np.testing.assert_allclose(T.rotation_matrix, np.eye(3), atol=ATOL)

    def test_as_matrix_returns_copy(self) -> None:
        T = RigidTransform.identity("a")
        M = T.as_matrix()
        M[0, 0] = 999
        np.testing.assert_allclose(T.as_matrix(), np.eye(4), atol=ATOL)


# ===========================================================================
# DbC — invalid inputs
# ===========================================================================


class TestContractViolations:
    """DbC precondition enforcement for bad inputs."""

    def test_from_matrix_wrong_shape(self) -> None:
        with pytest.raises(PreconditionError):
            RigidTransform.from_matrix(np.eye(3), source="a", target="b")

    def test_from_matrix_bad_bottom_row(self) -> None:
        M = np.eye(4)
        M[3, 0] = 1  # Invalid SE(3)
        with pytest.raises(PreconditionError):
            RigidTransform.from_matrix(M, source="a", target="b")

    def test_from_rotation_translation_bad_R(self) -> None:
        R = np.eye(3) * 2  # Not SO(3)
        with pytest.raises(PreconditionError):
            RigidTransform.from_rotation_translation(
                R, [0, 0, 0], source="a", target="b"
            )

    def test_from_quaternion_wrong_length(self) -> None:
        with pytest.raises(PreconditionError):
            RigidTransform.from_quaternion_translation(
                [1, 0, 0], [0, 0, 0], source="a", target="b"
            )

    def test_from_axis_angle_non_unit_axis(self) -> None:
        with pytest.raises(PreconditionError):
            RigidTransform.from_axis_angle_translation(
                [2, 0, 0], 0.5, [0, 0, 0], source="a", target="b"
            )

    def test_apply_point_wrong_shape(self) -> None:
        T = RigidTransform.identity("a")
        with pytest.raises(PreconditionError):
            T.apply_point([1, 2])

    def test_apply_vector_wrong_shape(self) -> None:
        T = RigidTransform.identity("a")
        with pytest.raises(PreconditionError):
            T.apply_vector([1, 2, 3, 4])

    def test_from_twist_wrong_length(self) -> None:
        with pytest.raises(PreconditionError):
            RigidTransform.from_twist([1, 2, 3], 1.0, source="a", target="b")

    def test_body_to_space_twist_wrong_shape(self) -> None:
        T = RigidTransform.identity("a")
        with pytest.raises(PreconditionError):
            T.body_to_space_twist(np.array([1, 2, 3]))


# ===========================================================================
# Cross-representation consistency (all outputs describe same transform)
# ===========================================================================


class TestCrossRepresentationConsistency:
    """All output methods should describe the exact same rigid-body transform."""

    @pytest.mark.parametrize("trial", range(10))
    def test_all_outputs_consistent(self, trial: int) -> None:
        """Build from random SE(3), verify all outputs reconstruct same matrix."""
        rng = np.random.default_rng(seed=100 + trial)
        A = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        p = rng.standard_normal(3) * 5
        M = np.eye(4)
        M[:3, :3] = Q
        M[:3, 3] = p

        T = RigidTransform.from_matrix(M, source="a", target="b")
        M_orig = T.as_matrix()

        # Via quaternion + translation
        q, p_out = T.as_quaternion_translation()
        T2 = RigidTransform.from_quaternion_translation(
            q, p_out, source="a", target="b"
        )
        np.testing.assert_allclose(T2.as_matrix(), M_orig, atol=ATOL)

        # Via rotation + translation
        R, p_out = T.as_rotation_translation()
        T3 = RigidTransform.from_rotation_translation(R, p_out, source="a", target="b")
        np.testing.assert_allclose(T3.as_matrix(), M_orig, atol=ATOL)

        # Via Rotation object
        rot = T.as_rotation()
        T4 = RigidTransform.from_rotation(rot, T.translation, source="a", target="b")
        np.testing.assert_allclose(T4.as_matrix(), M_orig, atol=ATOL)

        # Via Euler + translation
        euler, p_out = T.as_euler_translation("zyx")
        T5 = RigidTransform.from_euler_translation(
            euler[0],
            euler[1],
            euler[2],
            p_out,
            convention="zyx",
            source="a",
            target="b",
        )
        np.testing.assert_allclose(T5.as_matrix(), M_orig, atol=1e-5)

        # Via axis-angle + translation
        ax, ang, p_out = T.as_axis_angle_translation()
        T6 = RigidTransform.from_axis_angle_translation(
            ax, ang, p_out, source="a", target="b"
        )
        np.testing.assert_allclose(T6.as_matrix(), M_orig, atol=ATOL)

        # Via rodrigues + translation
        rv, p_out = T.as_rodrigues_translation()
        T7 = RigidTransform.from_rodrigues_translation(
            rv, p_out, source="a", target="b"
        )
        np.testing.assert_allclose(T7.as_matrix(), M_orig, atol=ATOL)

    @pytest.mark.parametrize("trial", range(10))
    def test_twist_roundtrip_reconstructs_matrix(self, trial: int) -> None:
        """from_matrix -> as_twist -> from_twist should reconstruct same SE(3)."""
        rng = np.random.default_rng(seed=200 + trial)
        # Random rotation with bounded angle (avoid pi singularity in log)
        axis = rng.standard_normal(3)
        axis = axis / np.linalg.norm(axis)
        angle = rng.uniform(0.1, 2.5)
        c, s = math.cos(angle), math.sin(angle)
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        p = rng.standard_normal(3) * 3

        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = p

        T1 = RigidTransform.from_matrix(M, source="a", target="b")
        twist, theta = T1.as_twist()
        T2 = RigidTransform.from_twist(twist, theta, source="a", target="b")
        np.testing.assert_allclose(T1.as_matrix(), T2.as_matrix(), atol=ATOL)


# ===========================================================================
# Batch vector transformations
# ===========================================================================


class TestApplyVectors:
    """Batch direction-vector transformation (Nx3, rotation only)."""

    def test_apply_vectors_ignores_translation(self) -> None:
        T = RigidTransform.pure_translation(
            [100, 200, 300], source="body", target="world"
        )
        vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        result = T.apply_vectors(vecs)
        np.testing.assert_allclose(result, vecs, atol=ATOL)

    def test_apply_vectors_rotates(self) -> None:
        angle = math.pi / 2
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T = RigidTransform.from_rotation_translation(
            R, [5, 10, 15], source="body", target="world"
        )
        vecs = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        result = T.apply_vectors(vecs)
        expected = np.array([[0, 1, 0], [-1, 0, 0]], dtype=float)
        np.testing.assert_allclose(result, expected, atol=ATOL)

    def test_apply_vectors_shape(self) -> None:
        T = RigidTransform.identity("a")
        vecs = np.zeros((5, 3))
        result = T.apply_vectors(vecs)
        assert result.shape == (5, 3)

    def test_apply_vectors_wrong_shape_raises(self) -> None:
        T = RigidTransform.identity("a")
        with pytest.raises(PreconditionError):
            T.apply_vectors(np.zeros((3, 2)))

    def test_apply_vectors_consistent_with_apply_vector(self) -> None:
        """Batch should match one-at-a-time."""
        R = Rotation.from_euler(0.3, 0.5, 0.7, "xyz").as_rotation_matrix()
        p = np.array([1.0, 2.0, 3.0])
        T = RigidTransform.from_rotation_translation(R, p, source="a", target="b")
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((10, 3))
        batch_result = T.apply_vectors(vecs)
        for i in range(10):
            single_result = T.apply_vector(vecs[i])
            np.testing.assert_allclose(batch_result[i], single_result, atol=ATOL)


# ===========================================================================
# Homogeneous coordinate transformations
# ===========================================================================


class TestHomogeneousCoordinates:
    """Transform 4-vectors: [x,y,z,1] for points, [x,y,z,0] for vectors."""

    def test_homogeneous_point_w1(self) -> None:
        """w=1 should behave like apply_point."""
        T = RigidTransform.from_rotation_translation(
            np.eye(3), [1, 2, 3], source="a", target="b"
        )
        ph = np.array([10.0, 20.0, 30.0, 1.0])
        result = T.apply_homogeneous(ph)
        np.testing.assert_allclose(result, [11, 22, 33, 1], atol=ATOL)

    def test_homogeneous_vector_w0(self) -> None:
        """w=0 should behave like apply_vector (no translation)."""
        T = RigidTransform.from_rotation_translation(
            np.eye(3), [100, 200, 300], source="a", target="b"
        )
        vh = np.array([1.0, 0.0, 0.0, 0.0])
        result = T.apply_homogeneous(vh)
        np.testing.assert_allclose(result, [1, 0, 0, 0], atol=ATOL)

    def test_homogeneous_rotation_point(self) -> None:
        angle = math.pi / 2
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T = RigidTransform.from_rotation_translation(
            R, [1, 0, 0], source="a", target="b"
        )
        # Point [1,0,0,1] -> R@[1,0,0]+[1,0,0] = [0,1,0]+[1,0,0] = [1,1,0,1]
        result = T.apply_homogeneous(np.array([1.0, 0.0, 0.0, 1.0]))
        np.testing.assert_allclose(result, [1, 1, 0, 1], atol=ATOL)

    def test_homogeneous_rotation_vector(self) -> None:
        angle = math.pi / 2
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T = RigidTransform.from_rotation_translation(
            R, [1, 0, 0], source="a", target="b"
        )
        # Vector [1,0,0,0] -> R@[1,0,0] = [0,1,0,0] (no translation!)
        result = T.apply_homogeneous(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [0, 1, 0, 0], atol=ATOL)

    def test_homogeneous_wrong_shape(self) -> None:
        T = RigidTransform.identity("a")
        with pytest.raises(PreconditionError):
            T.apply_homogeneous(np.array([1, 2, 3]))

    def test_homogeneous_batch(self) -> None:
        """Batch Nx4 homogeneous transform."""
        T = RigidTransform.from_rotation_translation(
            np.eye(3), [1, 2, 3], source="a", target="b"
        )
        phs = np.array(
            [
                [1, 0, 0, 1],  # point
                [0, 1, 0, 1],  # point
                [1, 0, 0, 0],  # vector
                [0, 1, 0, 0],  # vector
            ],
            dtype=float,
        )
        result = T.apply_homogeneous_batch(phs)
        expected = np.array(
            [
                [2, 2, 3, 1],
                [1, 3, 3, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(result, expected, atol=ATOL)

    def test_homogeneous_batch_wrong_shape(self) -> None:
        T = RigidTransform.identity("a")
        with pytest.raises(PreconditionError):
            T.apply_homogeneous_batch(np.zeros((3, 3)))

    def test_homogeneous_batch_consistent_with_single(self) -> None:
        R = Rotation.from_euler(0.3, 0.5, 0.7, "xyz").as_rotation_matrix()
        T = RigidTransform.from_rotation_translation(
            R, [1, 2, 3], source="a", target="b"
        )
        rng = np.random.default_rng(77)
        phs = rng.standard_normal((8, 3))
        # Add w column: alternate points and vectors
        w = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        phs_h = np.column_stack([phs, w])
        batch_result = T.apply_homogeneous_batch(phs_h)
        for i in range(8):
            single = T.apply_homogeneous(phs_h[i])
            np.testing.assert_allclose(batch_result[i], single, atol=ATOL)


# ===========================================================================
# Batch twist/wrench conversions (motion data vectors)
# ===========================================================================


class TestBatchTwistWrenchConversions:
    """Convert Nx6 arrays of twists and wrenches between frames."""

    @pytest.fixture()
    def transform(self) -> RigidTransform:
        R = Rotation.from_euler(0.3, 0.5, 0.7, "xyz").as_rotation_matrix()
        p = np.array([1.0, 2.0, 3.0])
        return RigidTransform.from_rotation_translation(
            R, p, source="body", target="space"
        )

    def test_body_to_space_twists_batch(self, transform: RigidTransform) -> None:
        rng = np.random.default_rng(42)
        Vb_batch = rng.standard_normal((5, 6))
        Vs_batch = transform.body_to_space_twists(Vb_batch)
        assert Vs_batch.shape == (5, 6)
        # Verify matches single conversion
        for i in range(5):
            Vs_single = transform.body_to_space_twist(Vb_batch[i])
            np.testing.assert_allclose(Vs_batch[i], Vs_single, atol=ATOL)

    def test_space_to_body_twists_batch(self, transform: RigidTransform) -> None:
        rng = np.random.default_rng(43)
        Vs_batch = rng.standard_normal((5, 6))
        Vb_batch = transform.space_to_body_twists(Vs_batch)
        assert Vb_batch.shape == (5, 6)
        for i in range(5):
            Vb_single = transform.space_to_body_twist(Vs_batch[i])
            np.testing.assert_allclose(Vb_batch[i], Vb_single, atol=ATOL)

    def test_twist_batch_roundtrip(self, transform: RigidTransform) -> None:
        rng = np.random.default_rng(44)
        Vb_orig = rng.standard_normal((10, 6))
        Vs = transform.body_to_space_twists(Vb_orig)
        Vb_back = transform.space_to_body_twists(Vs)
        np.testing.assert_allclose(Vb_back, Vb_orig, atol=ATOL)

    def test_body_to_space_wrenches_batch(self, transform: RigidTransform) -> None:
        rng = np.random.default_rng(45)
        Fb_batch = rng.standard_normal((5, 6))
        Fs_batch = transform.body_to_space_wrenches(Fb_batch)
        assert Fs_batch.shape == (5, 6)
        for i in range(5):
            Fs_single = transform.body_to_space_wrench(Fb_batch[i])
            np.testing.assert_allclose(Fs_batch[i], Fs_single, atol=ATOL)

    def test_space_to_body_wrenches_batch(self, transform: RigidTransform) -> None:
        rng = np.random.default_rng(46)
        Fs_batch = rng.standard_normal((5, 6))
        Fb_batch = transform.space_to_body_wrenches(Fs_batch)
        assert Fb_batch.shape == (5, 6)
        for i in range(5):
            Fb_single = transform.space_to_body_wrench(Fs_batch[i])
            np.testing.assert_allclose(Fb_batch[i], Fb_single, atol=ATOL)

    def test_wrench_batch_roundtrip(self, transform: RigidTransform) -> None:
        rng = np.random.default_rng(47)
        Fb_orig = rng.standard_normal((10, 6))
        Fs = transform.body_to_space_wrenches(Fb_orig)
        Fb_back = transform.space_to_body_wrenches(Fs)
        np.testing.assert_allclose(Fb_back, Fb_orig, atol=ATOL)

    def test_batch_wrong_shape_raises(self, transform: RigidTransform) -> None:
        with pytest.raises(PreconditionError):
            transform.body_to_space_twists(np.zeros((3, 4)))
        with pytest.raises(PreconditionError):
            transform.space_to_body_twists(np.zeros((3,)))


# ===========================================================================
# Finiteness checks on apply_point / apply_vector
# ===========================================================================


class TestFiniteChecks:
    """apply_point and apply_vector reject NaN/Inf inputs."""

    def test_apply_point_nan(self) -> None:
        T = RigidTransform.identity("a")
        with pytest.raises(PreconditionError):
            T.apply_point(np.array([1.0, float("nan"), 0.0]))

    def test_apply_point_inf(self) -> None:
        T = RigidTransform.identity("a")
        with pytest.raises(PreconditionError):
            T.apply_point(np.array([1.0, float("inf"), 0.0]))

    def test_apply_vector_nan(self) -> None:
        T = RigidTransform.identity("a")
        with pytest.raises(PreconditionError):
            T.apply_vector(np.array([float("nan"), 0.0, 0.0]))

    def test_apply_vector_inf(self) -> None:
        T = RigidTransform.identity("a")
        with pytest.raises(PreconditionError):
            T.apply_vector(np.array([0.0, 0.0, float("inf")]))

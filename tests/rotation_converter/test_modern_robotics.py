"""TDD tests for Modern Robotics (Lynch & Park) module.

Tests cover the core algorithms from "Modern Robotics: Mechanics,
Planning, and Control" by Lynch & Park, including:

- SO(3) / se(3) / SE(3) helpers
- Forward kinematics (space and body forms, product of exponentials)
- Inverse kinematics (body form, iterative Newton-Raphson)
- Jacobians (space and body)
- Velocity kinematics
- Trajectory generation

Written BEFORE implementation (TDD).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rotation_converter._contracts import PreconditionError
from rotation_converter.modern_robotics import (
    FKinBody,
    FKinSpace,
    IKinBody,
    JacobianBody,
    JacobianSpace,
    MatrixExp3,
    MatrixExp6,
    MatrixLog3,
    MatrixLog6,
    RpToTrans,
    ScrewTrajectory,
    TransInv,
    TransToRp,
    VecTose3,
    VecToso3,
    se3ToVec,
    so3ToVec,
)

ATOL = 1e-6


# ===========================================================================
# SO(3) helpers: VecToso3, so3ToVec, MatrixExp3, MatrixLog3
# ===========================================================================


class TestSO3Helpers:
    """so(3) and SO(3) conversion functions."""

    def test_VecToso3_basic(self) -> None:
        omega = np.array([1.0, 2.0, 3.0])
        result = VecToso3(omega)
        expected = np.array(
            [
                [0, -3, 2],
                [3, 0, -1],
                [-2, 1, 0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(result, expected, atol=ATOL)

    def test_so3ToVec_basic(self) -> None:
        M = np.array(
            [
                [0, -3, 2],
                [3, 0, -1],
                [-2, 1, 0],
            ],
            dtype=float,
        )
        result = so3ToVec(M)
        np.testing.assert_allclose(result, [1, 2, 3], atol=ATOL)

    def test_VecToso3_so3ToVec_roundtrip(self) -> None:
        omega = np.array([0.5, -1.2, 0.7])
        result = so3ToVec(VecToso3(omega))
        np.testing.assert_allclose(result, omega, atol=ATOL)

    def test_MatrixExp3_zero_gives_identity(self) -> None:
        result = MatrixExp3(np.zeros((3, 3)))
        np.testing.assert_allclose(result, np.eye(3), atol=ATOL)

    def test_MatrixExp3_90deg_z(self) -> None:
        omega_hat = VecToso3(np.array([0, 0, 1]))
        theta = math.pi / 2
        R = MatrixExp3(omega_hat * theta)
        expected = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(R, expected, atol=ATOL)

    def test_MatrixExp3_result_is_SO3(self) -> None:
        omega = np.array([1, 2, 3], dtype=float)
        omega = omega / np.linalg.norm(omega)
        so3 = VecToso3(omega) * 1.5
        R = MatrixExp3(so3)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)
        assert abs(np.linalg.det(R) - 1.0) < ATOL

    def test_MatrixLog3_identity_gives_zero(self) -> None:
        result = MatrixLog3(np.eye(3))
        np.testing.assert_allclose(result, np.zeros((3, 3)), atol=ATOL)

    def test_MatrixExp3_MatrixLog3_roundtrip(self) -> None:
        omega = np.array([0.3, -0.5, 0.7])
        omega = omega / np.linalg.norm(omega)
        so3_mat = VecToso3(omega) * 1.2
        R = MatrixExp3(so3_mat)
        so3_back = MatrixLog3(R)
        R_back = MatrixExp3(so3_back)
        np.testing.assert_allclose(R_back, R, atol=ATOL)

    def test_MatrixLog3_180deg(self) -> None:
        """180-degree rotation about z-axis."""
        R = np.array(
            [
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        log_R = MatrixLog3(R)
        R_back = MatrixExp3(log_R)
        np.testing.assert_allclose(R_back, R, atol=ATOL)


# ===========================================================================
# SE(3) helpers: VecTose3, se3ToVec, MatrixExp6, MatrixLog6
# ===========================================================================


class TestSE3Helpers:
    """se(3) and SE(3) conversion functions."""

    def test_VecTose3_basic(self) -> None:
        V = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        result = VecTose3(V)
        assert result.shape == (4, 4)
        np.testing.assert_allclose(result[3, :], [0, 0, 0, 0], atol=ATOL)
        np.testing.assert_allclose(result[:3, 3], [4, 5, 6], atol=ATOL)

    def test_se3ToVec_basic(self) -> None:
        M = np.zeros((4, 4))
        M[:3, :3] = VecToso3(np.array([1.0, 2.0, 3.0]))
        M[:3, 3] = [4, 5, 6]
        result = se3ToVec(M)
        np.testing.assert_allclose(result, [1, 2, 3, 4, 5, 6], atol=ATOL)

    def test_VecTose3_se3ToVec_roundtrip(self) -> None:
        V = np.array([0.1, -0.2, 0.3, 1.0, 2.0, 3.0])
        result = se3ToVec(VecTose3(V))
        np.testing.assert_allclose(result, V, atol=ATOL)

    def test_MatrixExp6_zero_gives_identity(self) -> None:
        result = MatrixExp6(np.zeros((4, 4)))
        np.testing.assert_allclose(result, np.eye(4), atol=ATOL)

    def test_MatrixExp6_pure_rotation(self) -> None:
        V = np.array([0, 0, 1, 0, 0, 0], dtype=float)
        se3 = VecTose3(V) * (math.pi / 2)
        T = MatrixExp6(se3)
        expected_R = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(T[:3, :3], expected_R, atol=ATOL)
        np.testing.assert_allclose(T[:3, 3], [0, 0, 0], atol=ATOL)

    def test_MatrixExp6_pure_translation(self) -> None:
        V = np.array([0, 0, 0, 1, 0, 0], dtype=float)
        se3 = VecTose3(V) * 5.0
        T = MatrixExp6(se3)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=ATOL)
        np.testing.assert_allclose(T[:3, 3], [5, 0, 0], atol=ATOL)

    def test_MatrixExp6_result_is_SE3(self) -> None:
        V = np.array([0.1, 0.2, 0.3, 1, 2, 3], dtype=float)
        se3 = VecTose3(V)
        T = MatrixExp6(se3)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)
        assert abs(np.linalg.det(R) - 1.0) < ATOL
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=ATOL)

    def test_MatrixExp6_MatrixLog6_roundtrip(self) -> None:
        V = np.array([0.1, -0.2, 0.3, 1, 2, 3], dtype=float)
        se3 = VecTose3(V)
        T = MatrixExp6(se3)
        se3_back = MatrixLog6(T)
        T_back = MatrixExp6(se3_back)
        np.testing.assert_allclose(T_back, T, atol=ATOL)


# ===========================================================================
# TransToRp, RpToTrans, TransInv
# ===========================================================================


class TestTransformHelpers:
    """SE(3) decomposition and inversion helpers."""

    def test_TransToRp_identity(self) -> None:
        R, p = TransToRp(np.eye(4))
        np.testing.assert_allclose(R, np.eye(3), atol=ATOL)
        np.testing.assert_allclose(p, [0, 0, 0], atol=ATOL)

    def test_RpToTrans_identity(self) -> None:
        T = RpToTrans(np.eye(3), np.zeros(3))
        np.testing.assert_allclose(T, np.eye(4), atol=ATOL)

    def test_TransToRp_RpToTrans_roundtrip(self) -> None:
        T = np.eye(4)
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        T[:3, 3] = [1, 2, 3]
        R, p = TransToRp(T)
        T2 = RpToTrans(R, p)
        np.testing.assert_allclose(T2, T, atol=ATOL)

    def test_TransInv_identity(self) -> None:
        result = TransInv(np.eye(4))
        np.testing.assert_allclose(result, np.eye(4), atol=ATOL)

    def test_TransInv_is_inverse(self) -> None:
        T = np.eye(4)
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        T[:3, 3] = [1, 2, 3]
        T_inv = TransInv(T)
        product = T @ T_inv
        np.testing.assert_allclose(product, np.eye(4), atol=ATOL)


# ===========================================================================
# Forward Kinematics — Product of Exponentials
# ===========================================================================


class TestForwardKinematics:
    """Forward kinematics via product of exponentials (space and body)."""

    @pytest.fixture
    def simple_2r_robot(self) -> dict:
        """Simple 2R planar robot for testing.

        Two revolute joints about z-axis, link lengths L1=1, L2=1.
        """
        L1, L2 = 1.0, 1.0
        # Home configuration: arm fully extended along x
        M = np.array(
            [
                [1, 0, 0, L1 + L2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        # Space-form screw axes (at home position)
        Slist = np.array(
            [
                [0, 0, 1, 0, 0, 0],  # Joint 1: rotate about z at origin
                [0, 0, 1, 0, -L1, 0],  # Joint 2: rotate about z at (L1, 0, 0)
            ]
        ).T  # 6 x n

        # Body-form screw axes
        Blist = np.array(
            [
                [0, 0, 1, 0, L1 + L2, 0],  # Joint 1 in body frame
                [0, 0, 1, 0, L2, 0],  # Joint 2 in body frame
            ]
        ).T  # 6 x n

        return {"M": M, "Slist": Slist, "Blist": Blist, "L1": L1, "L2": L2}

    def test_FKinSpace_zero_angles(self, simple_2r_robot: dict) -> None:
        """At zero joint angles, should return home configuration M."""
        thetalist = np.array([0.0, 0.0])
        T = FKinSpace(simple_2r_robot["M"], simple_2r_robot["Slist"], thetalist)
        np.testing.assert_allclose(T, simple_2r_robot["M"], atol=ATOL)

    def test_FKinBody_zero_angles(self, simple_2r_robot: dict) -> None:
        """At zero joint angles, should return home configuration M."""
        thetalist = np.array([0.0, 0.0])
        T = FKinBody(simple_2r_robot["M"], simple_2r_robot["Blist"], thetalist)
        np.testing.assert_allclose(T, simple_2r_robot["M"], atol=ATOL)

    def test_FKinSpace_90deg_joint1(self, simple_2r_robot: dict) -> None:
        """Rotate joint 1 by 90 degrees -> end-effector at (0, 2, 0)."""
        thetalist = np.array([math.pi / 2, 0.0])
        T = FKinSpace(simple_2r_robot["M"], simple_2r_robot["Slist"], thetalist)
        np.testing.assert_allclose(T[:3, 3], [0, 2, 0], atol=ATOL)

    def test_FKinBody_90deg_joint1(self, simple_2r_robot: dict) -> None:
        """Same test via body form."""
        thetalist = np.array([math.pi / 2, 0.0])
        T = FKinBody(simple_2r_robot["M"], simple_2r_robot["Blist"], thetalist)
        np.testing.assert_allclose(T[:3, 3], [0, 2, 0], atol=ATOL)

    def test_FKinSpace_FKinBody_agree(self, simple_2r_robot: dict) -> None:
        """Space and body FK should give the same SE(3) result."""
        thetalist = np.array([0.3, -0.7])
        T_space = FKinSpace(simple_2r_robot["M"], simple_2r_robot["Slist"], thetalist)
        T_body = FKinBody(simple_2r_robot["M"], simple_2r_robot["Blist"], thetalist)
        np.testing.assert_allclose(T_space, T_body, atol=ATOL)

    def test_FKinSpace_both_joints_90deg(self, simple_2r_robot: dict) -> None:
        """Both joints at 90deg -> end effector at (-1, 1, 0)."""
        thetalist = np.array([math.pi / 2, math.pi / 2])
        T = FKinSpace(simple_2r_robot["M"], simple_2r_robot["Slist"], thetalist)
        np.testing.assert_allclose(T[:3, 3], [-1, 1, 0], atol=ATOL)

    def test_FKinSpace_result_is_SE3(self, simple_2r_robot: dict) -> None:
        thetalist = np.array([0.5, 1.2])
        T = FKinSpace(simple_2r_robot["M"], simple_2r_robot["Slist"], thetalist)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)
        assert abs(np.linalg.det(R) - 1.0) < ATOL
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=ATOL)


# ===========================================================================
# Jacobians
# ===========================================================================


class TestJacobians:
    """Space and body Jacobian tests."""

    @pytest.fixture
    def simple_2r_robot(self) -> dict:
        L1, L2 = 1.0, 1.0
        M = np.array(
            [
                [1, 0, 0, L1 + L2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        Slist = np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, -L1, 0],
            ]
        ).T
        Blist = np.array(
            [
                [0, 0, 1, 0, L1 + L2, 0],
                [0, 0, 1, 0, L2, 0],
            ]
        ).T
        return {"M": M, "Slist": Slist, "Blist": Blist}

    def test_JacobianSpace_at_zero(self, simple_2r_robot: dict) -> None:
        thetalist = np.array([0.0, 0.0])
        Js = JacobianSpace(simple_2r_robot["Slist"], thetalist)
        assert Js.shape == (6, 2)
        # At zero config, columns should equal the screw axes
        np.testing.assert_allclose(Js, simple_2r_robot["Slist"], atol=ATOL)

    def test_JacobianBody_at_zero(self, simple_2r_robot: dict) -> None:
        thetalist = np.array([0.0, 0.0])
        Jb = JacobianBody(simple_2r_robot["Blist"], thetalist)
        assert Jb.shape == (6, 2)
        # At zero config, columns should equal the body screw axes
        np.testing.assert_allclose(Jb, simple_2r_robot["Blist"], atol=ATOL)

    def test_JacobianSpace_shape(self, simple_2r_robot: dict) -> None:
        thetalist = np.array([0.5, 1.2])
        Js = JacobianSpace(simple_2r_robot["Slist"], thetalist)
        assert Js.shape == (6, 2)

    def test_JacobianBody_shape(self, simple_2r_robot: dict) -> None:
        thetalist = np.array([0.5, 1.2])
        Jb = JacobianBody(simple_2r_robot["Blist"], thetalist)
        assert Jb.shape == (6, 2)


# ===========================================================================
# Inverse Kinematics
# ===========================================================================


class TestInverseKinematics:
    """Iterative inverse kinematics via Newton-Raphson."""

    @pytest.fixture
    def simple_2r_robot(self) -> dict:
        L1, L2 = 1.0, 1.0
        M = np.array(
            [
                [1, 0, 0, L1 + L2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        Blist = np.array(
            [
                [0, 0, 1, 0, L1 + L2, 0],
                [0, 0, 1, 0, L2, 0],
            ]
        ).T
        Slist = np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, -L1, 0],
            ]
        ).T
        return {"M": M, "Blist": Blist, "Slist": Slist}

    def test_IKinBody_reaches_home(self, simple_2r_robot: dict) -> None:
        """IK for the home configuration should return near-zero angles."""
        T_desired = simple_2r_robot["M"].copy()
        thetalist0 = np.array([0.1, 0.1])
        result, success = IKinBody(
            simple_2r_robot["Blist"],
            simple_2r_robot["M"],
            T_desired,
            thetalist0,
            eomg=1e-6,
            ev=1e-6,
        )
        assert success
        # Verify by forward kinematics
        T_achieved = FKinBody(simple_2r_robot["M"], simple_2r_robot["Blist"], result)
        np.testing.assert_allclose(T_achieved[:3, 3], T_desired[:3, 3], atol=1e-4)

    def test_IKinBody_reaches_known_config(self, simple_2r_robot: dict) -> None:
        """IK for end-effector at (0, 2, 0) -> joint1=pi/2, joint2=0."""
        T_desired = np.array(
            [
                [0, -1, 0, 0],
                [1, 0, 0, 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        thetalist0 = np.array([1.0, 0.5])
        result, success = IKinBody(
            simple_2r_robot["Blist"],
            simple_2r_robot["M"],
            T_desired,
            thetalist0,
            eomg=1e-6,
            ev=1e-6,
        )
        assert success
        T_achieved = FKinBody(simple_2r_robot["M"], simple_2r_robot["Blist"], result)
        np.testing.assert_allclose(T_achieved[:3, 3], [0, 2, 0], atol=1e-4)

    def test_IKinBody_FK_roundtrip(self, simple_2r_robot: dict) -> None:
        """FK -> IK -> FK should recover original end-effector pose."""
        theta_orig = np.array([0.3, -0.5])
        T_desired = FKinBody(simple_2r_robot["M"], simple_2r_robot["Blist"], theta_orig)
        thetalist0 = np.array([0.0, 0.0])
        result, success = IKinBody(
            simple_2r_robot["Blist"],
            simple_2r_robot["M"],
            T_desired,
            thetalist0,
            eomg=1e-6,
            ev=1e-6,
        )
        assert success
        T_achieved = FKinBody(simple_2r_robot["M"], simple_2r_robot["Blist"], result)
        np.testing.assert_allclose(T_achieved, T_desired, atol=1e-4)


# ===========================================================================
# Trajectory Generation
# ===========================================================================


class TestTrajectoryGeneration:
    """Screw trajectory generation."""

    def test_ScrewTrajectory_start_and_end(self) -> None:
        Xstart = np.eye(4)
        Xend = np.eye(4)
        Xend[:3, 3] = [1, 0, 0]
        N = 5
        traj = ScrewTrajectory(Xstart, Xend, Tf=1.0, N=N, method=3)
        assert len(traj) == N
        np.testing.assert_allclose(traj[0], Xstart, atol=ATOL)
        np.testing.assert_allclose(traj[-1], Xend, atol=ATOL)

    def test_ScrewTrajectory_all_SE3(self) -> None:
        Xstart = np.eye(4)
        Xend = np.eye(4)
        Xend[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        Xend[:3, 3] = [1, 2, 3]
        traj = ScrewTrajectory(Xstart, Xend, Tf=2.0, N=10, method=5)
        for T in traj:
            R = T[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)
            assert abs(np.linalg.det(R) - 1.0) < ATOL
            np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=ATOL)

    def test_ScrewTrajectory_length(self) -> None:
        traj = ScrewTrajectory(np.eye(4), np.eye(4), Tf=1.0, N=20, method=3)
        assert len(traj) == 20

    def test_ScrewTrajectory_monotonic_progress(self) -> None:
        """Trajectory should smoothly interpolate (no jumping back)."""
        Xstart = np.eye(4)
        Xend = np.eye(4)
        Xend[:3, 3] = [3, 0, 0]
        traj = ScrewTrajectory(Xstart, Xend, Tf=1.0, N=10, method=3)
        x_positions = [T[0, 3] for T in traj]
        # x should be monotonically non-decreasing
        for i in range(1, len(x_positions)):
            assert x_positions[i] >= x_positions[i - 1] - ATOL


# ===========================================================================
# Random round-trip stress tests
# ===========================================================================


class TestRandomMRRoundTrips:
    """Randomised tests for Modern Robotics functions."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=99)

    @pytest.mark.parametrize("trial", range(20))
    def test_MatrixExp3_MatrixLog3_random(
        self, rng: np.random.Generator, trial: int
    ) -> None:
        omega = rng.normal(size=3)
        omega = omega / np.linalg.norm(omega)
        theta = rng.uniform(0.01, math.pi - 0.01)
        so3 = VecToso3(omega) * theta
        R = MatrixExp3(so3)
        so3_back = MatrixLog3(R)
        R_back = MatrixExp3(so3_back)
        np.testing.assert_allclose(R_back, R, atol=1e-9)

    @pytest.mark.parametrize("trial", range(20))
    def test_MatrixExp6_MatrixLog6_random(
        self, rng: np.random.Generator, trial: int
    ) -> None:
        omega = rng.normal(size=3)
        omega = omega / np.linalg.norm(omega)
        v = rng.normal(size=3)
        V = np.concatenate([omega, v])
        theta = rng.uniform(0.01, 2.0)
        se3 = VecTose3(V) * theta
        T = MatrixExp6(se3)
        se3_back = MatrixLog6(T)
        T_back = MatrixExp6(se3_back)
        np.testing.assert_allclose(T_back, T, atol=1e-9)

    @pytest.mark.parametrize("trial", range(10))
    def test_TransInv_random(self, rng: np.random.Generator, trial: int) -> None:
        omega = rng.normal(size=3)
        omega = omega / np.linalg.norm(omega)
        theta = rng.uniform(0.1, 2.0)
        so3 = VecToso3(omega) * theta
        R = MatrixExp3(so3)
        p = rng.normal(size=3)
        T = RpToTrans(R, p)
        T_inv = TransInv(T)
        product = T @ T_inv
        np.testing.assert_allclose(product, np.eye(4), atol=1e-9)


# ===========================================================================
# Edge cases: MatrixLog3 pi-rotation branches, IK failure, NaN/Inf, contracts
# ===========================================================================


class TestMatrixLog3PiBranches:
    """MatrixLog3 pi-rotation column selection (all 3 axes)."""

    def test_pi_rotation_about_x(self) -> None:
        R = np.diag([-1.0, -1.0, 1.0])  # Rz(pi) actually, let's do Rx(pi)
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        so3 = MatrixLog3(R)
        R_back = MatrixExp3(so3)
        np.testing.assert_allclose(R_back, R, atol=1e-9)

    def test_pi_rotation_about_y(self) -> None:
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
        so3 = MatrixLog3(R)
        R_back = MatrixExp3(so3)
        np.testing.assert_allclose(R_back, R, atol=1e-9)

    def test_pi_rotation_about_z(self) -> None:
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        so3 = MatrixLog3(R)
        R_back = MatrixExp3(so3)
        np.testing.assert_allclose(R_back, R, atol=1e-9)

    def test_MatrixLog6_identity(self) -> None:
        """MatrixLog6 of identity SE(3) should be zero."""
        T = np.eye(4)
        se3 = MatrixLog6(T)
        np.testing.assert_allclose(se3, np.zeros((4, 4)), atol=1e-12)


class TestIKFailure:
    """Test that IKinBody returns success=False for unreachable targets."""

    def test_unreachable_target(self) -> None:
        """Target far outside workspace should fail to converge."""
        # Simple 1-DOF revolute around z
        Blist = np.array([[0, 0, 1, 0, 0, 0]], dtype=float).T
        M = np.eye(4)
        M[0, 3] = 1.0  # end-effector at (1,0,0)

        # Unreachable target at (100, 100, 100)
        T_desired = np.eye(4)
        T_desired[:3, 3] = [100, 100, 100]

        thetalist, success = IKinBody(
            Blist, M, T_desired, np.array([0.0]), max_iter=5
        )
        assert success is False

    def test_convergence_with_good_guess(self) -> None:
        """IK should converge when target is reachable with good initial guess."""
        # 2-DOF planar arm
        Blist = np.array([
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=float).T
        M = np.eye(4)
        M[0, 3] = 2.0  # end-effector at (2,0,0) in home config

        # Use FK to get a known reachable target
        theta_target = np.array([0.3, -0.2])
        T_desired = FKinBody(M, Blist, theta_target)

        thetalist, success = IKinBody(
            Blist, M, T_desired, np.array([0.0, 0.0])
        )
        assert success is True


class TestModernRoboticsContracts:
    """NaN/Inf and shape contract tests for modern_robotics functions."""

    def test_nan_so3_raises(self) -> None:
        so3 = np.array([[0, float("nan"), 0], [0, 0, 0], [0, 0, 0]])
        with pytest.raises(PreconditionError):
            MatrixExp3(so3)

    def test_inf_rotation_matrix_log3_raises(self) -> None:
        R = np.eye(3)
        R[0, 0] = float("inf")
        with pytest.raises(PreconditionError):
            MatrixLog3(R)

    def test_nan_se3_exp6_raises(self) -> None:
        se3 = np.zeros((4, 4))
        se3[0, 3] = float("nan")
        with pytest.raises(PreconditionError):
            MatrixExp6(se3)

    def test_nan_se3_log6_raises(self) -> None:
        T = np.eye(4)
        T[0, 3] = float("inf")
        with pytest.raises(PreconditionError):
            MatrixLog6(T)

    def test_nan_slist_fkin_raises(self) -> None:
        M = np.eye(4)
        Slist = np.array([[0, 0, 1, 0, 0, float("nan")]]).T
        with pytest.raises(PreconditionError):
            FKinSpace(M, Slist, np.array([0.0]))

    def test_jacobian_shape_validation(self) -> None:
        """JacobianSpace should reject wrong-shaped Slist."""
        # 5 rows instead of 6
        Slist = np.ones((5, 2))
        with pytest.raises(PreconditionError):
            JacobianSpace(Slist, np.array([0.0, 0.0]))

    def test_ikin_positive_tolerance(self) -> None:
        """IKinBody should reject non-positive tolerances."""
        Blist = np.array([[0, 0, 1, 0, 0, 0]], dtype=float).T
        M = np.eye(4)
        T = np.eye(4)
        with pytest.raises(PreconditionError):
            IKinBody(Blist, M, T, np.array([0.0]), eomg=-1.0)

"""Coverage for the previously-untested public surface of modern_robotics.

Covers IK / trajectory / projection / SO(3)-SE(3) helper functions that had
zero test coverage (issue #3689) plus the boundary-validation contracts added
to the legacy IK/trajectory functions (issue #3688) and a guard that the
module contains no ``-O``-stripped ``assert`` statements (issue #3687).

Each function gets a nominal success path (validated against the canonical
Lynch & Park "Modern Robotics" reference outputs) and, where boundary
validation was added, an exercise of that validation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

from rotation_converter._contracts import PreconditionError
from rotation_converter.modern_robotics import (
    Adjoint,
    AxisAng3,
    AxisAng6,
    CartesianTrajectory,
    CubicTimeScaling,
    DistanceToSE3,
    DistanceToSO3,
    IKinSpace,
    JointTrajectory,
    Normalize,
    ProjectToSE3,
    ProjectToSO3,
    QuinticTimeScaling,
    RotInv,
    ScrewToAxis,
    TestIfSE3,
    TestIfSO3,
)

ATOL = 1e-4


# ===========================================================================
# #3687 guard: no -O-stripped asserts in the module source
# ===========================================================================


def test_module_has_no_stripped_asserts() -> None:
    """Input validation must use require()/raise, never bare ``assert``.

    ``assert`` statements are stripped under ``python -O`` and would silently
    disable boundary validation in optimized deployments.
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "rotation_converter"
        / "modern_robotics.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    asserts = [
        f"line {node.lineno}" for node in ast.walk(tree) if isinstance(node, ast.Assert)
    ]
    assert not asserts, "modern_robotics.py must not use assert: " + ", ".join(asserts)


# ===========================================================================
# Small SO(3) / SE(3) helpers (projection, distance, membership, inverse)
# ===========================================================================


class TestProjectionAndDistance:
    def test_RotInv_transposes_orthonormal_matrix(self) -> None:
        R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        out = RotInv(R)
        # For a rotation matrix R^-1 == R^T and R @ R^-1 == I.
        np.testing.assert_allclose(out, R.T, atol=ATOL)
        np.testing.assert_allclose(np.dot(R, out), np.eye(3), atol=ATOL)

    def test_Adjoint_matches_reference(self) -> None:
        T = np.array(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 3], [0, 0, 0, 1]], dtype=float
        )
        expected = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 3, 1, 0, 0],
                [3, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(Adjoint(T), expected, atol=ATOL)

    def test_ScrewToAxis_matches_reference(self) -> None:
        q = np.array([3, 0, 0])
        s = np.array([0, 0, 1])
        h = 2
        np.testing.assert_allclose(
            ScrewToAxis(q, s, h), np.array([0, 0, 1, 0, -3, 2]), atol=ATOL
        )

    def test_ProjectToSO3_returns_valid_rotation(self) -> None:
        mat = np.array(
            [
                [0.675, 0.150, 0.720],
                [0.370, 0.771, -0.511],
                [-0.630, 0.619, 0.472],
            ]
        )
        R = ProjectToSO3(mat)
        # Result must be a proper rotation: orthonormal with det +1.
        np.testing.assert_allclose(np.dot(R.T, R), np.eye(3), atol=ATOL)
        assert abs(np.linalg.det(R) - 1.0) < ATOL

    def test_ProjectToSE3_returns_valid_transform(self) -> None:
        mat = np.array(
            [
                [0.675, 0.150, 0.720, 1.2],
                [0.370, 0.771, -0.511, 5.4],
                [-0.630, 0.619, 0.472, 3.6],
                [0.003, 0.002, 0.010, 0.9],
            ]
        )
        T = ProjectToSE3(mat)
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=ATOL)
        np.testing.assert_allclose(np.dot(T[:3, :3].T, T[:3, :3]), np.eye(3), atol=ATOL)
        # Translation column is preserved by the projection.
        np.testing.assert_allclose(T[:3, 3], [1.2, 5.4, 3.6], atol=ATOL)

    def test_DistanceToSO3_matches_reference(self) -> None:
        mat = np.array([[1.0, 0.0, 0.0], [0.0, 0.1, -0.95], [0.0, 1.0, 0.1]])
        assert DistanceToSO3(mat) == pytest.approx(0.08835, abs=1e-4)

    def test_DistanceToSO3_negative_determinant_is_large(self) -> None:
        # Reflection has det == -1 -> sentinel large distance.
        mat = np.diag([1.0, 1.0, -1.0])
        assert DistanceToSO3(mat) >= 1e9

    def test_DistanceToSE3_matches_reference(self) -> None:
        mat = np.array(
            [
                [1.0, 0.0, 0.0, 1.2],
                [0.0, 0.1, -0.95, 1.5],
                [0.0, 1.0, 0.1, -0.9],
                [0.0, 0.0, 0.1, 0.98],
            ]
        )
        assert DistanceToSE3(mat) == pytest.approx(0.134931, abs=1e-4)

    def test_TestIfSO3_true_for_identity_false_for_near(self) -> None:
        assert bool(TestIfSO3(np.eye(3))) is True
        mat = np.array([[1.0, 0.0, 0.0], [0.0, 0.1, -0.95], [0.0, 1.0, 0.1]])
        assert bool(TestIfSO3(mat)) is False

    def test_TestIfSE3_true_for_identity_false_for_near(self) -> None:
        assert bool(TestIfSE3(np.eye(4))) is True
        mat = np.array(
            [
                [1.0, 0.0, 0.0, 1.2],
                [0.0, 0.1, -0.95, 1.5],
                [0.0, 1.0, 0.1, -0.9],
                [0.0, 0.0, 0.1, 0.98],
            ]
        )
        assert bool(TestIfSE3(mat)) is False

    def test_Normalize_and_AxisAng_consistency(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        unit, theta = AxisAng3(v)
        np.testing.assert_allclose(unit, Normalize(v), atol=ATOL)
        assert theta == pytest.approx(np.linalg.norm(v), abs=ATOL)

    def test_AxisAng6_zero_vector(self) -> None:
        screw, theta = AxisAng6(np.zeros(6))
        assert theta == pytest.approx(0.0, abs=ATOL)
        np.testing.assert_allclose(screw, np.zeros(6), atol=ATOL)


# ===========================================================================
# Time scaling
# ===========================================================================


class TestTimeScaling:
    def test_cubic_matches_reference(self) -> None:
        assert CubicTimeScaling(2, 0.6) == pytest.approx(0.216, abs=ATOL)

    def test_cubic_endpoints(self) -> None:
        assert CubicTimeScaling(2, 0.0) == pytest.approx(0.0, abs=ATOL)
        assert CubicTimeScaling(2, 2.0) == pytest.approx(1.0, abs=ATOL)

    def test_quintic_matches_reference(self) -> None:
        assert QuinticTimeScaling(2, 0.6) == pytest.approx(0.16308, abs=ATOL)

    def test_quintic_endpoints(self) -> None:
        assert QuinticTimeScaling(2, 0.0) == pytest.approx(0.0, abs=ATOL)
        assert QuinticTimeScaling(2, 2.0) == pytest.approx(1.0, abs=ATOL)


# ===========================================================================
# JointTrajectory — nominal + boundary validation (#3688)
# ===========================================================================


class TestJointTrajectory:
    def test_nominal_endpoints_and_shape(self) -> None:
        thetastart = np.array([1, 0, 0, 1, 1, 0.2, 0, 1], dtype=float)
        thetaend = np.array([1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1], dtype=float)
        traj = JointTrajectory(thetastart, thetaend, Tf=4, N=6, method=3)
        traj = np.asarray(traj)
        assert traj.shape == (6, 8)
        np.testing.assert_allclose(traj[0], thetastart, atol=ATOL)
        np.testing.assert_allclose(traj[-1], thetaend, atol=ATOL)

    def test_quintic_method(self) -> None:
        thetastart = np.zeros(3)
        thetaend = np.ones(3)
        traj = np.asarray(JointTrajectory(thetastart, thetaend, Tf=5, N=4, method=5))
        np.testing.assert_allclose(traj[0], thetastart, atol=ATOL)
        np.testing.assert_allclose(traj[-1], thetaend, atol=ATOL)

    def test_rejects_N_less_than_2(self) -> None:
        with pytest.raises(PreconditionError):
            JointTrajectory(np.zeros(3), np.ones(3), Tf=4, N=1, method=3)

    def test_rejects_nonpositive_Tf(self) -> None:
        with pytest.raises(PreconditionError):
            JointTrajectory(np.zeros(3), np.ones(3), Tf=0, N=4, method=3)

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(PreconditionError):
            JointTrajectory(np.zeros(3), np.ones(4), Tf=4, N=4, method=3)

    def test_rejects_nonfinite(self) -> None:
        bad = np.array([0.0, np.nan, 0.0])
        with pytest.raises(PreconditionError):
            JointTrajectory(bad, np.ones(3), Tf=4, N=4, method=3)


# ===========================================================================
# CartesianTrajectory — nominal + boundary validation (#3688)
# ===========================================================================


class TestCartesianTrajectory:
    Xstart = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float
    )
    Xend = np.array(
        [[0, 0, 1, 0.1], [1, 0, 0, 0], [0, 1, 0, 4.1], [0, 0, 0, 1]], dtype=float
    )

    def test_nominal_endpoints_and_length(self) -> None:
        traj = CartesianTrajectory(self.Xstart, self.Xend, Tf=5, N=4, method=5)
        assert len(traj) == 4
        np.testing.assert_allclose(np.asarray(traj[0]), self.Xstart, atol=ATOL)
        np.testing.assert_allclose(np.asarray(traj[-1]), self.Xend, atol=ATOL)
        for X in traj:
            assert np.asarray(X).shape == (4, 4)

    def test_rejects_N_less_than_2(self) -> None:
        with pytest.raises(PreconditionError):
            CartesianTrajectory(self.Xstart, self.Xend, Tf=5, N=1, method=5)

    def test_rejects_nonpositive_Tf(self) -> None:
        with pytest.raises(PreconditionError):
            CartesianTrajectory(self.Xstart, self.Xend, Tf=-1, N=4, method=5)

    def test_rejects_non_4x4(self) -> None:
        with pytest.raises(PreconditionError):
            CartesianTrajectory(np.eye(3), self.Xend, Tf=5, N=4, method=5)

    def test_rejects_nonfinite(self) -> None:
        bad = self.Xstart.copy()
        bad[0, 3] = np.inf
        with pytest.raises(PreconditionError):
            CartesianTrajectory(bad, self.Xend, Tf=5, N=4, method=5)


# ===========================================================================
# IKinSpace — nominal + boundary validation (#3688)
# ===========================================================================


class TestIKinSpace:
    Slist = np.array(
        [[0, 0, 1, 4, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, -1, -6, 0, -0.1]]
    ).T
    M = np.array(
        [[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]], dtype=float
    )
    T = np.array(
        [[0, 1, 0, -5], [1, 0, 0, 4], [0, 0, -1, 1.6858], [0, 0, 0, 1]], dtype=float
    )

    def test_nominal_solution_converges(self) -> None:
        thetalist0 = np.array([1.5, 2.5, 3])
        result, success = IKinSpace(
            self.Slist, self.M, self.T, thetalist0, eomg=0.01, ev=0.001
        )
        assert success is True
        np.testing.assert_allclose(
            result, np.array([1.57073783, 2.99966384, 3.1415342]), atol=1e-3
        )

    def test_rejects_non_4x4_M(self) -> None:
        with pytest.raises(PreconditionError):
            IKinSpace(self.Slist, np.eye(3), self.T, np.zeros(3), 0.01, 0.001)

    def test_rejects_non_4x4_T(self) -> None:
        with pytest.raises(PreconditionError):
            IKinSpace(self.Slist, self.M, np.eye(3), np.zeros(3), 0.01, 0.001)

    def test_rejects_bad_Slist_shape(self) -> None:
        with pytest.raises(PreconditionError):
            IKinSpace(np.zeros((3, 3)), self.M, self.T, np.zeros(3), 0.01, 0.001)

    def test_rejects_nonpositive_tolerances(self) -> None:
        with pytest.raises(PreconditionError):
            IKinSpace(self.Slist, self.M, self.T, np.array([1.5, 2.5, 3]), 0.0, 0.001)
        with pytest.raises(PreconditionError):
            IKinSpace(self.Slist, self.M, self.T, np.array([1.5, 2.5, 3]), 0.01, -1.0)

    def test_rejects_nonfinite_inputs(self) -> None:
        bad_M = self.M.copy()
        bad_M[0, 0] = np.nan
        with pytest.raises(PreconditionError):
            IKinSpace(self.Slist, bad_M, self.T, np.zeros(3), 0.01, 0.001)

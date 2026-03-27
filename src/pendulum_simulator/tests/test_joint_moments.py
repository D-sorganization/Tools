"""Tests for joint_moments module — torque and moment vector calculations.

TDD: Tests define expected moment computation behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.joint_moments import (
    cross_2d,
    double_pendulum_moments,
    moment_of_force,
    torque_arrow_direction,
    total_moment_at_joint,
    triple_pendulum_moments,
)


class TestCross2D:
    """Tests for 2-D cross product."""

    def test_unit_vectors(self):
        """x × y = +1 (CCW)."""
        assert cross_2d(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)

    def test_antiparallel(self):
        """y × x = -1 (CW)."""
        assert cross_2d(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(-1.0)

    def test_parallel(self):
        """Parallel vectors → zero cross product."""
        assert cross_2d(np.array([3.0, 0.0]), np.array([5.0, 0.0])) == pytest.approx(0.0)

    def test_wrong_shape_raises(self):
        with pytest.raises((ValueError, TypeError), match="r must be shape"):
            cross_2d(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


class TestMomentOfForce:
    """Tests for moment of net force about distal COM."""

    def test_perpendicular_force(self):
        """Force perpendicular to lever arm → full moment."""
        joint = np.array([0.0, 0.0])
        com = np.array([1.0, 0.0])
        force = np.array([0.0, 1.0])
        # r = [1,0], F = [0,1] → moment = 1*1 - 0*0 = 1
        assert moment_of_force(joint, com, force) == pytest.approx(1.0)

    def test_parallel_force(self):
        """Force parallel to lever arm → zero moment."""
        joint = np.array([0.0, 0.0])
        com = np.array([1.0, 0.0])
        force = np.array([1.0, 0.0])
        assert moment_of_force(joint, com, force) == pytest.approx(0.0)

    def test_negative_moment(self):
        """Force in -y with +x lever → negative (CW) moment."""
        joint = np.array([0.0, 0.0])
        com = np.array([1.0, 0.0])
        force = np.array([0.0, -1.0])
        assert moment_of_force(joint, com, force) == pytest.approx(-1.0)


class TestTotalMomentAtJoint:
    """Tests for combined applied torque + moment of force."""

    def test_sums_correctly(self):
        joint = np.array([0.0, 0.0])
        com = np.array([1.0, 0.0])
        force = np.array([0.0, 1.0])
        # moment_of_force = 1.0, applied = 5.0 → total = 6.0
        assert total_moment_at_joint(5.0, joint, com, force) == pytest.approx(6.0)

    def test_cancellation(self):
        joint = np.array([0.0, 0.0])
        com = np.array([1.0, 0.0])
        force = np.array([0.0, 1.0])
        # moment = 1.0, applied = -1.0 → total = 0.0
        assert total_moment_at_joint(-1.0, joint, com, force) == pytest.approx(0.0)

    def test_nan_torque_raises(self):
        with pytest.raises((ValueError, TypeError), match="torque must be finite"):
            total_moment_at_joint(
                float("nan"),
                np.array([0.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
            )


class TestDoublePendulumMoments:
    """Integration tests for double pendulum moments."""

    def test_returns_all_keys(self):
        positions = {
            "shoulder": (0.0, 0.0),
            "wrist": (0.5, -0.5),
            "tip": (1.0, -1.0),
        }
        forces = {
            "shoulder": (10.0, -5.0),
            "wrist": (3.0, -2.0),
        }
        torques = (100.0, 50.0)
        result = double_pendulum_moments(positions, forces, torques, None)
        expected_keys = {
            "shoulder_applied_torque",
            "shoulder_moment_of_force",
            "shoulder_total_moment",
            "wrist_applied_torque",
            "wrist_moment_of_force",
            "wrist_total_moment",
        }
        assert set(result.keys()) == expected_keys

    def test_all_finite(self):
        positions = {
            "shoulder": (0.0, 0.0),
            "wrist": (0.65, 0.0),
            "tip": (1.75, 0.0),
        }
        forces = {"shoulder": (1.0, -9.81), "wrist": (0.5, -3.0)}
        torques = (10.0, 5.0)
        result = double_pendulum_moments(positions, forces, torques, None)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is non-finite: {val}"


class TestTriplePendulumMoments:
    """Integration tests for triple pendulum moments."""

    def test_returns_all_keys(self):
        positions = {
            "shoulder": (0.0, 0.0),
            "elbow": (0.15, -0.1),
            "wrist": (0.65, -0.5),
            "tip": (1.5, -1.0),
        }
        forces = {
            "shoulder": (5.0, -10.0),
            "elbow": (3.0, -6.0),
            "wrist": (1.0, -2.0),
        }
        torques = (50.0, 30.0, 10.0)
        result = triple_pendulum_moments(positions, forces, torques, None)
        expected_keys = {
            "shoulder_applied_torque",
            "shoulder_moment_of_force",
            "shoulder_total_moment",
            "elbow_applied_torque",
            "elbow_moment_of_force",
            "elbow_total_moment",
            "wrist_applied_torque",
            "wrist_moment_of_force",
            "wrist_total_moment",
        }
        assert set(result.keys()) == expected_keys


class TestTorqueArrowDirection:
    """Tests for torque arrow rendering helper."""

    def test_zero_torque_returns_same_point(self):
        pos = np.array([1.0, 2.0])
        start, end = torque_arrow_direction(pos, 0.0, 0.0)
        np.testing.assert_allclose(start, pos)
        np.testing.assert_allclose(end, pos)

    def test_positive_torque_distinct_points(self):
        pos = np.array([0.0, 0.0])
        start, end = torque_arrow_direction(pos, 0.0, 10.0)
        assert not np.allclose(start, end)

    def test_negative_torque_distinct_points(self):
        pos = np.array([0.0, 0.0])
        start, end = torque_arrow_direction(pos, 0.0, -10.0)
        assert not np.allclose(start, end)

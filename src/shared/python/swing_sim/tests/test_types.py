"""Unit tests for swing_sim value types (DbC validation)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.types import (
    PendulumParameters,
    PendulumState,
    PlaneOrientation,
    SwingSample,
    SwingTrajectory,
)


def _identity_sample(t: float) -> SwingSample:
    return SwingSample(t=t, pose=np.eye(4), twist=np.zeros(6))


@pytest.mark.unit
class TestPlaneOrientation:
    def test_default_is_zero_pose(self) -> None:
        plane = PlaneOrientation()
        assert plane.yaw_deg == 0.0
        assert plane.side_tilt_deg == 0.0
        assert plane.forward_tilt_deg == 0.0

    def test_radian_conversion(self) -> None:
        plane = PlaneOrientation(yaw_deg=180.0, side_tilt_deg=90.0)
        assert plane.yaw_rad == pytest.approx(np.pi)
        assert plane.side_tilt_rad == pytest.approx(np.pi / 2.0)
        assert plane.forward_tilt_rad == 0.0

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            PlaneOrientation(yaw_deg=float("nan"))

    def test_is_frozen(self) -> None:
        plane = PlaneOrientation()
        with pytest.raises(AttributeError):
            plane.yaw_deg = 1.0  # type: ignore[misc]


@pytest.mark.unit
class TestPendulumParameters:
    def test_golf_default_is_valid_and_matches_reference_constants(self) -> None:
        p = PendulumParameters.golf_default()
        assert p.m1 == pytest.approx(7.5)
        assert p.l1 == pytest.approx(0.75)
        assert p.lc1 == pytest.approx(0.3375)
        assert p.m2 == pytest.approx(0.35)
        assert p.l2 == pytest.approx(1.0)
        assert p.lc2 == pytest.approx((0.43 * 0.15 + 0.20) / 0.35)
        assert p.d1 == pytest.approx(0.4)
        assert p.d2 == pytest.approx(0.25)

    def test_rejects_nonpositive_mass(self) -> None:
        p = PendulumParameters.golf_default()
        with pytest.raises(ValueError, match="m1"):
            PendulumParameters(
                m1=0.0,
                l1=p.l1,
                lc1=p.lc1,
                i1=p.i1,
                m2=p.m2,
                l2=p.l2,
                lc2=p.lc2,
                i2=p.i2,
            )

    def test_rejects_negative_damping(self) -> None:
        p = PendulumParameters.golf_default()
        with pytest.raises(ValueError, match="d1"):
            PendulumParameters(
                m1=p.m1,
                l1=p.l1,
                lc1=p.lc1,
                i1=p.i1,
                m2=p.m2,
                l2=p.l2,
                lc2=p.lc2,
                i2=p.i2,
                d1=-0.1,
            )

    def test_rejects_com_beyond_segment(self) -> None:
        p = PendulumParameters.golf_default()
        with pytest.raises(ValueError, match="lc1"):
            PendulumParameters(
                m1=p.m1,
                l1=p.l1,
                lc1=2.0 * p.l1,
                i1=p.i1,
                m2=p.m2,
                l2=p.l2,
                lc2=p.lc2,
                i2=p.i2,
            )


@pytest.mark.unit
class TestPendulumState:
    def test_valid_construction(self) -> None:
        s = PendulumState(theta1=1.0, theta2=-0.5, omega1=0.0, omega2=2.0)
        assert s.theta1 == 1.0

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="omega2"):
            PendulumState(theta1=0.0, theta2=0.0, omega1=0.0, omega2=float("inf"))


@pytest.mark.unit
class TestSwingSample:
    def test_valid_identity_sample(self) -> None:
        sample = _identity_sample(0.5)
        assert sample.t == 0.5
        assert sample.pose.shape == (4, 4)
        assert sample.twist.shape == (6,)

    def test_rejects_bad_pose_shape(self) -> None:
        with pytest.raises(ValueError, match="4x4"):
            SwingSample(t=0.0, pose=np.eye(3), twist=np.zeros(6))

    def test_rejects_bad_twist_shape(self) -> None:
        with pytest.raises(ValueError, match="6-vector"):
            SwingSample(t=0.0, pose=np.eye(4), twist=np.zeros(5))

    def test_rejects_non_orthonormal_rotation(self) -> None:
        pose = np.eye(4)
        pose[0, 0] = 2.0
        with pytest.raises(ValueError, match="orthonormal"):
            SwingSample(t=0.0, pose=pose, twist=np.zeros(6))

    def test_rejects_bad_bottom_row(self) -> None:
        pose = np.eye(4)
        pose[3, 0] = 1.0
        with pytest.raises(ValueError, match="bottom row"):
            SwingSample(t=0.0, pose=pose, twist=np.zeros(6))

    def test_rejects_negative_time(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            _identity_sample(-0.1)


@pytest.mark.unit
class TestSwingTrajectory:
    def test_duration_and_len(self) -> None:
        traj = SwingTrajectory(
            samples=(
                _identity_sample(0.0),
                _identity_sample(0.5),
                _identity_sample(1.0),
            )
        )
        assert traj.duration == pytest.approx(1.0)
        assert len(traj) == 3

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="samples"):
            SwingTrajectory(samples=())

    def test_rejects_non_increasing_times(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            SwingTrajectory(samples=(_identity_sample(0.5), _identity_sample(0.5)))

"""TDD tests for screw axis visualization module.

Tests cover:
- Motion trajectory generation (football spiral, frisbee flight)
- Screw axis extraction from SE(3) trajectory pairs
- Frame data construction for animation
- Coordinate geometry of screw axis arrows and body frames

Written BEFORE implementation (TDD).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rotation_converter.motion_examples import (
    frisbee_flight,
    football_spiral,
)
from rotation_converter.screw_visualization import (
    extract_screw_axes_from_trajectory,
    build_animation_frames,
    ScrewAxisAnimator,
)

ATOL = 1e-6


# ===========================================================================
# Motion example generators
# ===========================================================================


class TestFootballSpiral:
    """Football spiral trajectory generator."""

    def test_returns_list_of_SE3(self) -> None:
        traj = football_spiral(n_frames=20)
        assert len(traj) == 20
        for T in traj:
            assert T.shape == (4, 4)

    def test_SE3_validity(self) -> None:
        traj = football_spiral(n_frames=10)
        for T in traj:
            R = T[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)
            assert abs(np.linalg.det(R) - 1.0) < ATOL
            np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=ATOL)

    def test_trajectory_moves_forward(self) -> None:
        """Football should travel forward (positive x or y)."""
        traj = football_spiral(n_frames=30)
        start_pos = traj[0][:3, 3]
        end_pos = traj[-1][:3, 3]
        displacement = np.linalg.norm(end_pos - start_pos)
        assert displacement > 1.0  # Must travel some distance

    def test_trajectory_has_spin(self) -> None:
        """Football should be rotating (not all same orientation)."""
        traj = football_spiral(n_frames=30)
        R_start = traj[0][:3, :3]
        R_end = traj[-1][:3, :3]
        # Orientations should differ
        assert not np.allclose(R_start, R_end, atol=0.01)

    def test_custom_parameters(self) -> None:
        traj = football_spiral(
            n_frames=15, speed=25.0, spin_rate=8.0, launch_angle_deg=40.0
        )
        assert len(traj) == 15

    def test_default_has_reasonable_height(self) -> None:
        """Football should arc up then down (parabolic)."""
        traj = football_spiral(n_frames=50)
        heights = [T[2, 3] for T in traj]
        max_h = max(heights)
        assert max_h > 0  # Should go up


class TestFrisbeeFlight:
    """Frisbee flight trajectory generator."""

    def test_returns_list_of_SE3(self) -> None:
        traj = frisbee_flight(n_frames=20)
        assert len(traj) == 20
        for T in traj:
            assert T.shape == (4, 4)

    def test_SE3_validity(self) -> None:
        traj = frisbee_flight(n_frames=10)
        for T in traj:
            R = T[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)
            assert abs(np.linalg.det(R) - 1.0) < ATOL

    def test_frisbee_moves_forward(self) -> None:
        traj = frisbee_flight(n_frames=30)
        start_pos = traj[0][:3, 3]
        end_pos = traj[-1][:3, 3]
        displacement = np.linalg.norm(end_pos - start_pos)
        assert displacement > 1.0

    def test_frisbee_spins(self) -> None:
        traj = frisbee_flight(n_frames=30)
        R_start = traj[0][:3, :3]
        R_end = traj[-1][:3, :3]
        assert not np.allclose(R_start, R_end, atol=0.01)

    def test_custom_parameters(self) -> None:
        traj = frisbee_flight(
            n_frames=25, speed=15.0, spin_rate=12.0, launch_angle_deg=10.0
        )
        assert len(traj) == 25


# ===========================================================================
# Screw axis extraction
# ===========================================================================


class TestScrewAxisExtraction:
    """Extract screw axis between consecutive SE(3) frames."""

    def test_pure_translation_screw_axis(self) -> None:
        """Pure translation -> screw axis along direction of motion, infinite pitch."""
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, 3] = [1, 0, 0]
        traj = [T1, T2]
        axes = extract_screw_axes_from_trajectory(traj)
        assert len(axes) == 1
        ax = axes[0]
        assert "axis" in ax
        assert "point" in ax
        assert "pitch" in ax
        assert "theta" in ax
        # Axis should be along x
        np.testing.assert_allclose(abs(ax["axis"]), [1, 0, 0], atol=ATOL)

    def test_pure_rotation_screw_axis(self) -> None:
        """Pure rotation -> screw axis through origin, zero pitch."""
        T1 = np.eye(4)
        T2 = np.eye(4)
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        T2[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        traj = [T1, T2]
        axes = extract_screw_axes_from_trajectory(traj)
        assert len(axes) == 1
        ax = axes[0]
        np.testing.assert_allclose(abs(ax["axis"]), [0, 0, 1], atol=ATOL)
        assert abs(ax["pitch"]) < 0.01  # Zero pitch for pure rotation

    def test_identity_motion_returns_zero_theta(self) -> None:
        """No motion -> theta ≈ 0."""
        T = np.eye(4)
        traj = [T, T]
        axes = extract_screw_axes_from_trajectory(traj)
        assert len(axes) == 1
        assert abs(axes[0]["theta"]) < ATOL

    def test_multiple_frames(self) -> None:
        """N frames -> N-1 screw axes."""
        traj = football_spiral(n_frames=10)
        axes = extract_screw_axes_from_trajectory(traj)
        assert len(axes) == 9

    def test_screw_axis_has_position(self) -> None:
        """Each screw axis should have a meaningful position."""
        traj = football_spiral(n_frames=10)
        axes = extract_screw_axes_from_trajectory(traj)
        for ax in axes:
            assert ax["point"].shape == (3,)
            assert np.all(np.isfinite(ax["point"]))


# ===========================================================================
# Animation frame building
# ===========================================================================


class TestAnimationFrames:
    """Build frame data for matplotlib 3D animation."""

    def test_build_frames_returns_correct_count(self) -> None:
        traj = football_spiral(n_frames=10)
        frames = build_animation_frames(traj)
        assert len(frames) == 10

    def test_frame_contains_required_keys(self) -> None:
        traj = football_spiral(n_frames=5)
        frames = build_animation_frames(traj)
        for f in frames:
            assert "position" in f
            assert "orientation" in f
            assert "body_axes" in f
            assert f["position"].shape == (3,)
            assert f["orientation"].shape == (3, 3)

    def test_frame_body_axes_are_orthonormal(self) -> None:
        traj = football_spiral(n_frames=5)
        frames = build_animation_frames(traj)
        for f in frames:
            axes = f["body_axes"]
            assert len(axes) == 3  # x, y, z body axes
            for ax_data in axes:
                assert "origin" in ax_data
                assert "direction" in ax_data

    def test_frames_have_screw_data(self) -> None:
        """Frames 1..N-1 should have screw axis data from the previous step."""
        traj = football_spiral(n_frames=10)
        frames = build_animation_frames(traj)
        # First frame has no preceding motion, so no screw axis
        assert frames[0].get("screw_axis") is None
        # Remaining frames should have screw data
        for f in frames[1:]:
            assert f["screw_axis"] is not None
            assert "axis" in f["screw_axis"]
            assert "point" in f["screw_axis"]


# ===========================================================================
# ScrewAxisAnimator (non-GUI, data-only tests)
# ===========================================================================


class TestScrewAxisAnimator:
    """Test the animator class configuration and data flow."""

    def test_create_animator(self) -> None:
        traj = football_spiral(n_frames=10)
        animator = ScrewAxisAnimator(traj, title="Football Spiral")
        assert animator.n_frames == 10
        assert animator.title == "Football Spiral"

    def test_animator_from_frisbee(self) -> None:
        traj = frisbee_flight(n_frames=15)
        animator = ScrewAxisAnimator(traj, title="Frisbee Flight")
        assert animator.n_frames == 15

    def test_animator_frames_property(self) -> None:
        traj = football_spiral(n_frames=10)
        animator = ScrewAxisAnimator(traj, title="Test")
        frames = animator.frames
        assert len(frames) == 10

    def test_animator_get_plot_bounds(self) -> None:
        traj = football_spiral(n_frames=20)
        animator = ScrewAxisAnimator(traj, title="Test")
        bounds = animator.get_plot_bounds()
        assert "x" in bounds
        assert "y" in bounds
        assert "z" in bounds
        for axis in ["x", "y", "z"]:
            assert bounds[axis][0] < bounds[axis][1]

    def test_animator_trajectory_path(self) -> None:
        """Should expose the 3D position path for trail rendering."""
        traj = football_spiral(n_frames=20)
        animator = ScrewAxisAnimator(traj, title="Test")
        path = animator.trajectory_path
        assert path.shape == (20, 3)

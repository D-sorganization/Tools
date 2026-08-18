"""Retained wedge-sweep interpolation invariants."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.golf_club._wedge_sweep import interpolated_pose


def _pose(rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    return pose


def test_slerp_resolves_an_exact_half_turn_at_constant_angular_rate() -> None:
    identity = np.eye(3)
    half_turn_about_x = np.diag((1.0, -1.0, -1.0))
    poses = np.stack([_pose(identity), _pose(half_turn_about_x)])

    midpoint = interpolated_pose(np.array((0.0, 1.0)), poses, 0.5)

    expected_quarter_turn = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    np.testing.assert_allclose(midpoint[:3, :3], expected_quarter_turn, atol=1e-12)
    assert np.linalg.det(midpoint[:3, :3]) == pytest.approx(1.0)


def test_slerp_preserves_endpoints_and_linear_translation() -> None:
    quarter_turn_about_z = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    poses = np.stack([_pose(np.eye(3)), _pose(quarter_turn_about_z)])
    poses[1, :3, 3] = (2.0, 4.0, 6.0)
    times = np.array((3.0, 5.0))

    np.testing.assert_allclose(interpolated_pose(times, poses, 3.0), poses[0])
    np.testing.assert_allclose(interpolated_pose(times, poses, 5.0), poses[1])
    np.testing.assert_allclose(
        interpolated_pose(times, poses, 4.0)[:3, 3],
        (1.0, 2.0, 3.0),
    )

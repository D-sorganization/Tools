# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for OptimizedSwingSource and OptimizedSwingResult."""

import numpy as np
import pytest

from rate_of_closure.simulation.anthropometry import GolferAnthropometry
from rate_of_closure.simulation.optimized_swing import (
    OptimizedSwingResult,
    OptimizedSwingSource,
)


def test_optimized_swing_source_sample_and_slerp() -> None:
    anthro = GolferAnthropometry()
    result = anthro.generate_delivery(duration_s=0.5, dt=1e-3)
    source = OptimizedSwingSource(result)

    assert source.duration == pytest.approx(0.5, rel=1e-3)
    assert source.frame_convention == "swing_frame"

    s0 = source.sample(0.0)
    assert s0.t == 0.0
    assert s0.pose.shape == (4, 4)
    assert s0.twist.shape == (6,)

    smid = source.sample(0.25)
    assert smid.t == 0.25
    assert np.all(np.isfinite(smid.pose))
    assert np.all(np.isfinite(smid.twist))


def test_optimized_swing_source_duck_typing() -> None:
    class MockOptimizerResult:
        def __init__(self) -> None:
            self.t = np.array([0.0, 0.1, 0.2])
            n = 3
            self.clubhead_poses = np.zeros((n, 4, 4), dtype=float)
            for i in range(n):
                self.clubhead_poses[i] = np.eye(4)
                self.clubhead_poses[i, 0, 3] = float(i) * 0.1
            self.clubhead_twists = np.ones((n, 6), dtype=float)
            self.success = True

    mock = MockOptimizerResult()
    source = OptimizedSwingSource(mock)
    assert source.duration == pytest.approx(0.2)
    s = source.sample(0.05)
    assert s.pose[0, 3] == pytest.approx(0.05)


def test_invalid_result_raises() -> None:
    with pytest.raises(ValueError):
        # Non-monotonic time
        OptimizedSwingResult(
            time_s=np.array([0.0, 0.0]),
            clubhead_poses=np.zeros((2, 4, 4)),
            clubhead_twists=np.zeros((2, 6)),
        )

    with pytest.raises(ValueError):
        # Non-orthonormal pose
        poses = np.zeros((2, 4, 4))
        OptimizedSwingResult(
            time_s=np.array([0.0, 0.1]),
            clubhead_poses=poses,
            clubhead_twists=np.zeros((2, 6)),
        )

# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Reject non-finite solver iterates before CubicSpline (#6187)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from movement_optimizer.trajectory.optimizer_packaging import evaluate_solution


def _reject_trajectory(_splines: object) -> tuple[object, object, object, object]:
    raise AssertionError("trajectory eval must not run for a non-finite iterate")


def test_evaluate_solution_rejects_nonfinite_iterate_before_spline() -> None:
    def build_splines(_x: np.ndarray) -> object:
        raise AssertionError("CubicSpline must not see a non-finite iterate")

    res = SimpleNamespace(x=np.array([0.1, np.nan, 0.2]), fun=1.0)
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_solution(
            res,
            None,  # type: ignore[arg-type]
            "squat",
            60.0,
            build_splines,
            _reject_trajectory,
        )

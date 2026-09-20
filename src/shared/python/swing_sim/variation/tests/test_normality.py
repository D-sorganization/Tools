"""Tests for bivariate normality diagnostics and convex hull fallback (#4253 item c)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.variation.normality import (
    convex_hull_2d,
    mardia_bivariate_normality,
)

pytestmark = pytest.mark.physics


class TestMardiaNormality:
    def test_bivariate_gaussian_passes_normality(self) -> None:
        rng = np.random.default_rng(42)
        # 400 points from a true bivariate normal distribution
        mean = [220.0, 0.0]
        cov = [[36.0, 2.0], [2.0, 9.0]]
        points = rng.multivariate_normal(mean, cov, size=400)

        diag = mardia_bivariate_normality(points)
        assert diag.is_normal is True
        assert diag.p_value >= 0.05
        assert diag.skewness_p_value >= 0.05
        assert diag.kurtosis_p_value >= 0.05
        assert diag.n == 400

    def test_uniform_distribution_fails_normality_due_to_kurtosis(self) -> None:
        rng = np.random.default_rng(42)
        # Uniform distribution on [-1, 1]^2 has platykurtic kurtosis (b2,2 << 8)
        points = rng.uniform(-10.0, 10.0, size=(500, 2))

        diag = mardia_bivariate_normality(points)
        assert diag.is_normal is False
        assert diag.kurtosis_p_value < 0.01

    def test_skewed_distribution_fails_normality_due_to_skewness(self) -> None:
        rng = np.random.default_rng(42)
        # Exponential distribution is heavily right-skewed
        points = rng.exponential(scale=5.0, size=(400, 2))

        diag = mardia_bivariate_normality(points)
        assert diag.is_normal is False
        assert diag.skewness_p_value < 0.01

    def test_small_sample_returns_safe_default(self) -> None:
        points = np.zeros((4, 2))
        diag = mardia_bivariate_normality(points)
        assert diag.is_normal is True
        assert diag.n == 4


class TestConvexHull:
    def test_convex_hull_encloses_all_points(self) -> None:
        rng = np.random.default_rng(42)
        points = rng.normal(size=(50, 2))
        hull = convex_hull_2d(points)

        assert hull.shape[1] == 2
        assert hull.shape[0] >= 3
        # Hull points must be a subset of original points
        for vertex in hull:
            dists = np.linalg.norm(points - vertex, axis=1)
            assert np.min(dists) == pytest.approx(0.0)

    def test_convex_hull_collinear_or_few_points_handled_safely(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        hull = convex_hull_2d(points)
        assert len(hull) == 2

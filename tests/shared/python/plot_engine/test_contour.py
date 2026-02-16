"""Tests for plot_engine.contour module.

Covers:
- scatter_to_grid interpolation
- correlation_matrix computation
- NaN handling
- Edge cases
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from plot_engine.contour import correlation_matrix, scatter_to_grid

# ── scatter_to_grid ──────────────────────────────────────────────────────


class TestScatterToGrid:
    """Test scatter_to_grid interpolation."""

    def test_basic_interpolation(self) -> None:
        """Simple scatter data should produce a valid grid."""
        x = np.array([0, 1, 0, 1, 0.5])
        y = np.array([0, 0, 1, 1, 0.5])
        z = np.array([0, 1, 1, 2, 1])
        x_grid, y_grid, z_grid = scatter_to_grid(x, y, z, resolution=10)
        assert len(x_grid) == 10
        assert len(y_grid) == 10
        assert z_grid.shape == (10, 10)

    def test_output_ranges(self) -> None:
        """Grid x/y should span from min to max of input."""
        x = np.array([1.0, 5.0, 3.0, 1.0, 5.0])
        y = np.array([2.0, 2.0, 5.0, 8.0, 8.0])
        z = np.array([10, 20, 15, 12, 18])
        x_grid, y_grid, _ = scatter_to_grid(x, y, z, resolution=20)
        assert_allclose(x_grid[0], 1.0)
        assert_allclose(x_grid[-1], 5.0)
        assert_allclose(y_grid[0], 2.0)
        assert_allclose(y_grid[-1], 8.0)

    def test_resolution_parameter(self) -> None:
        x = np.array([0, 1, 0, 1])
        y = np.array([0, 0, 1, 1])
        z = np.array([0, 1, 1, 2])
        x_grid, y_grid, z_grid = scatter_to_grid(x, y, z, resolution=50)
        assert len(x_grid) == 50
        assert z_grid.shape == (50, 50)

    def test_nan_removal(self) -> None:
        """NaN values should be stripped before interpolation."""
        x = np.array([0, 1, np.nan, 0, 1])
        y = np.array([0, 0, np.nan, 1, 1])
        z = np.array([0, 1, np.nan, 1, 2])
        x_grid, y_grid, z_grid = scatter_to_grid(x, y, z, resolution=10)
        assert z_grid.shape == (10, 10)

    def test_insufficient_points(self) -> None:
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        z = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="3 valid"):
            scatter_to_grid(x, y, z)

    def test_method_nearest(self) -> None:
        """Nearest interpolation should not produce NaN in-hull."""
        x = np.array([0, 1, 0, 1, 0.5])
        y = np.array([0, 0, 1, 1, 0.5])
        z = np.array([0, 1, 1, 2, 1])
        _, _, z_grid = scatter_to_grid(x, y, z, resolution=5, method="nearest")
        # Nearest should not produce NaN for points within convex hull
        assert not np.all(np.isnan(z_grid))


# ── correlation_matrix ───────────────────────────────────────────────────


class TestCorrelationMatrix:
    """Test correlation_matrix computation."""

    def test_perfect_correlation(self) -> None:
        """Identical columns should have correlation 1."""
        data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
        corr, labels = correlation_matrix(data)
        assert_allclose(corr[0, 1], 1.0, atol=1e-10)

    def test_perfect_anticorrelation(self) -> None:
        """Opposite columns should have correlation -1."""
        data = np.array([[1, 4], [2, 3], [3, 2], [4, 1]], dtype=float)
        corr, labels = correlation_matrix(data)
        assert_allclose(corr[0, 1], -1.0, atol=1e-10)

    def test_diagonal_is_unity(self) -> None:
        """Diagonal should always be 1."""
        rng = np.random.default_rng(42)
        data = rng.random((50, 3))
        corr, _ = correlation_matrix(data)
        assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_auto_labels(self) -> None:
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        _, labels = correlation_matrix(data)
        assert labels == ["Var 0", "Var 1", "Var 2"]

    def test_custom_labels(self) -> None:
        data = np.array([[1, 2], [3, 4]], dtype=float)
        _, labels = correlation_matrix(data, labels=["X", "Y"])
        assert labels == ["X", "Y"]

    def test_1d_data_rejected(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            correlation_matrix(np.array([1, 2, 3]))

    def test_symmetry(self) -> None:
        """Correlation matrix should be symmetric."""
        rng = np.random.default_rng(42)
        data = rng.random((20, 4))
        corr, _ = correlation_matrix(data)
        assert_allclose(corr, corr.T, atol=1e-10)

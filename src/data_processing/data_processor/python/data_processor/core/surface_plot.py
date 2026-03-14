"""3D Surface Plot Module with axis selection and smoothing.

Provides flexible 3D surface plotting capabilities with:
- Selectable X, Y, Z axes from data columns
- Multiple smoothing and interpolation options
- Filtering options for cleaner surfaces
- Ability to overlay regression fits
- Export to various formats

Designed for analyzing relationships in multivariate data
from noisy sources like gasification and robotics data.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter

logger = logging.getLogger(__name__)


class InterpolationMethod(Enum):
    """Available interpolation methods for surface creation."""

    LINEAR = "linear"
    CUBIC = "cubic"
    NEAREST = "nearest"
    RBF_THIN_PLATE = "rbf_thin_plate"
    RBF_MULTIQUADRIC = "rbf_multiquadric"
    RBF_GAUSSIAN = "rbf_gaussian"


class SmoothingMethod(Enum):
    """Available smoothing methods for surface data."""

    NONE = "none"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    UNIFORM = "uniform"
    SAVITZKY_GOLAY = "savitzky_golay"


@dataclass
class SurfacePlotConfig:
    """Configuration for surface plot generation."""

    # Axis columns
    x_column: str
    y_column: str
    z_column: str

    # Grid settings
    grid_resolution: int = 50
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None

    # Interpolation
    interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR

    # Smoothing
    smoothing_method: SmoothingMethod = SmoothingMethod.NONE
    smoothing_sigma: float = 1.0
    smoothing_kernel_size: int = 3

    # Filtering (pre-interpolation)
    remove_outliers: bool = False
    outlier_threshold: float = 3.0

    # Appearance
    colormap: str = "viridis"
    alpha: float = 0.8
    show_wireframe: bool = False
    show_scatter: bool = True
    scatter_alpha: float = 0.3

    # Labels
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    z_label: str = ""


@dataclass
class SurfacePlotResult:
    """Result of surface plot computation."""

    x_grid: np.ndarray
    y_grid: np.ndarray
    z_grid: np.ndarray
    x_data: np.ndarray
    y_data: np.ndarray
    z_data: np.ndarray
    config: SurfacePlotConfig
    statistics: dict[str, Any] = field(default_factory=dict)


class SurfacePlotEngine:
    """Engine for creating 3D surface plots from data.

    Supports various interpolation and smoothing methods to
    create clean surfaces from noisy data.
    """

    def __init__(self) -> None:
        """Initialize the surface plot engine."""
        self._interpolators: dict[
            InterpolationMethod,
            Callable[
                [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
            ],
        ] = {
            InterpolationMethod.LINEAR: self._interpolate_linear,
            InterpolationMethod.CUBIC: self._interpolate_cubic,
            InterpolationMethod.NEAREST: self._interpolate_nearest,
            InterpolationMethod.RBF_THIN_PLATE: self._interpolate_rbf_thin_plate,
            InterpolationMethod.RBF_MULTIQUADRIC: self._interpolate_rbf_multiquadric,
            InterpolationMethod.RBF_GAUSSIAN: self._interpolate_rbf_gaussian,
        }

        self._smoothers: dict[
            SmoothingMethod, Callable[[np.ndarray, SurfacePlotConfig], np.ndarray]
        ] = {
            SmoothingMethod.NONE: lambda z, _: z,
            SmoothingMethod.GAUSSIAN: self._smooth_gaussian,
            SmoothingMethod.MEDIAN: self._smooth_median,
            SmoothingMethod.UNIFORM: self._smooth_uniform,
            SmoothingMethod.SAVITZKY_GOLAY: self._smooth_savgol,
        }

    def create_surface(
        self,
        df: pd.DataFrame,
        config: SurfacePlotConfig,
    ) -> SurfacePlotResult:
        """Create a surface plot from data.

        Args:
            df: DataFrame containing the data
            config: Surface plot configuration

        Returns:
            SurfacePlotResult with grid and data arrays
        """
        # Validate columns exist
        assert df is not None, "df must be provided"
        self._validate_columns(df, config)

        # Extract and clean data
        x_data, y_data, z_data = self._extract_data(df, config)

        # Remove outliers if requested
        if config.remove_outliers:
            x_data, y_data, z_data = self._remove_outliers(
                x_data, y_data, z_data, config.outlier_threshold
            )

        # Create grid
        x_grid, y_grid = self._create_grid(x_data, y_data, config)

        # Interpolate to grid
        z_grid = self._interpolate(x_data, y_data, z_data, x_grid, y_grid, config)

        # Apply smoothing
        z_grid = self._apply_smoothing(z_grid, config)

        # Compute statistics
        statistics = self._compute_statistics(x_data, y_data, z_data, z_grid)

        return SurfacePlotResult(
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=z_grid,
            x_data=x_data,
            y_data=y_data,
            z_data=z_data,
            config=config,
            statistics=statistics,
        )

    def create_regression_surface(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        predict_func: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Create a surface from a regression prediction function.

        Args:
            x_grid: X grid points
            y_grid: Y grid points
            predict_func: Function that takes (n, 2) array and returns predictions

        Returns:
            Z values for the regression surface
        """
        # Flatten grids for prediction
        assert x_grid is not None, "x_grid must be provided"
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        xy_points = np.column_stack([x_flat, y_flat])

        # Get predictions
        z_pred = predict_func(xy_points)

        # Reshape to grid
        return z_pred.reshape(x_grid.shape)

    def _validate_columns(self, df: pd.DataFrame, config: SurfacePlotConfig) -> None:
        """Validate that required columns exist in DataFrame."""
        required = [config.x_column, config.y_column, config.z_column]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _extract_data(
        self,
        df: pd.DataFrame,
        config: SurfacePlotConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract and clean data arrays from DataFrame."""
        assert df is not None, "df must be provided"
        x = pd.to_numeric(df[config.x_column], errors="coerce").values
        y = pd.to_numeric(df[config.y_column], errors="coerce").values
        z = pd.to_numeric(df[config.z_column], errors="coerce").values

        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        return x[valid_mask], y[valid_mask], z[valid_mask]

    def _remove_outliers(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove outliers using z-score method."""
        assert x is not None, "x must be provided"
        z_scores = np.abs((z - np.mean(z)) / np.std(z))
        mask = z_scores < threshold
        return x[mask], y[mask], z[mask]

    def _create_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: SurfacePlotConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create interpolation grid."""
        assert x is not None, "x must be provided"
        x_range = config.x_range or (np.min(x), np.max(x))
        y_range = config.y_range or (np.min(y), np.max(y))

        x_lin = np.linspace(x_range[0], x_range[1], config.grid_resolution)
        y_lin = np.linspace(y_range[0], y_range[1], config.grid_resolution)

        return np.meshgrid(x_lin, y_lin)

    def _interpolate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        config: SurfacePlotConfig,
    ) -> np.ndarray:
        """Interpolate data to grid."""
        interpolator = self._interpolators.get(config.interpolation_method)
        if not interpolator:
            raise ValueError(
                f"Unknown interpolation method: {config.interpolation_method}"
            )
        return interpolator(x, y, z, x_grid, y_grid)

    def _apply_smoothing(
        self,
        z: np.ndarray,
        config: SurfacePlotConfig,
    ) -> np.ndarray:
        """Apply smoothing to surface."""
        smoother = self._smoothers.get(config.smoothing_method)
        if not smoother:
            raise ValueError(f"Unknown smoothing method: {config.smoothing_method}")
        return smoother(z, config)

    def _compute_statistics(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        z_grid: np.ndarray,
    ) -> dict[str, Any]:
        """Compute statistics about the surface."""
        assert x is not None, "x must be provided"
        valid_grid = z_grid[~np.isnan(z_grid)]
        return {
            "n_points": len(z),
            "x_range": (float(np.min(x)), float(np.max(x))),
            "y_range": (float(np.min(y)), float(np.max(y))),
            "z_range": (float(np.min(z)), float(np.max(z))),
            "z_mean": float(np.mean(z)),
            "z_std": float(np.std(z)),
            "grid_z_range": (
                float(np.min(valid_grid)) if len(valid_grid) > 0 else np.nan,
                float(np.max(valid_grid)) if len(valid_grid) > 0 else np.nan,
            ),
            "grid_coverage": float(np.sum(~np.isnan(z_grid))) / z_grid.size,
        }

    # ==========================================================================
    # INTERPOLATION METHODS
    # ==========================================================================

    def _interpolate_linear(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
    ) -> np.ndarray:
        """Linear interpolation using griddata."""
        assert x is not None, "x must be provided"
        points = np.column_stack([x, y])
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        z_interp = interpolate.griddata(points, z, grid_points, method="linear")
        return z_interp.reshape(x_grid.shape)

    def _interpolate_cubic(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
    ) -> np.ndarray:
        """Cubic interpolation using griddata."""
        assert x is not None, "x must be provided"
        points = np.column_stack([x, y])
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        z_interp = interpolate.griddata(points, z, grid_points, method="cubic")
        return z_interp.reshape(x_grid.shape)

    def _interpolate_nearest(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
    ) -> np.ndarray:
        """Nearest neighbor interpolation."""
        assert x is not None, "x must be provided"
        points = np.column_stack([x, y])
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        z_interp = interpolate.griddata(points, z, grid_points, method="nearest")
        return z_interp.reshape(x_grid.shape)

    def _interpolate_rbf_thin_plate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
    ) -> np.ndarray:
        """Radial basis function interpolation with thin plate splines."""
        return self._interpolate_rbf(x, y, z, x_grid, y_grid, "thin_plate_spline")

    def _interpolate_rbf_multiquadric(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
    ) -> np.ndarray:
        """Radial basis function interpolation with multiquadric."""
        return self._interpolate_rbf(x, y, z, x_grid, y_grid, "multiquadric")

    def _interpolate_rbf_gaussian(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
    ) -> np.ndarray:
        """Radial basis function interpolation with Gaussian."""
        return self._interpolate_rbf(x, y, z, x_grid, y_grid, "gaussian")

    def _interpolate_rbf(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        kernel: str,
    ) -> np.ndarray:
        """Generic RBF interpolation."""
        try:
            # Use scipy's RBFInterpolator
            points = np.column_stack([x, y])
            rbf = interpolate.RBFInterpolator(points, z, kernel=kernel)
            grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
            z_interp = rbf(grid_points)
            return z_interp.reshape(x_grid.shape)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.warning(f"RBF interpolation failed: {e}, falling back to linear")
            return self._interpolate_linear(x, y, z, x_grid, y_grid)

    # ==========================================================================
    # SMOOTHING METHODS
    # ==========================================================================

    def _smooth_gaussian(
        self,
        z: np.ndarray,
        config: SurfacePlotConfig,
    ) -> np.ndarray:
        """Apply Gaussian smoothing."""
        # Handle NaN values
        assert z is not None, "z must be provided"
        mask = np.isnan(z)
        z_filled = np.where(mask, 0, z)
        weights = np.where(mask, 0, 1).astype(float)

        # Apply filter to both data and weights
        z_smooth = gaussian_filter(z_filled, sigma=config.smoothing_sigma)
        w_smooth = gaussian_filter(weights, sigma=config.smoothing_sigma)

        # Normalize and restore NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            result = z_smooth / w_smooth
        result[mask] = np.nan

        return result

    def _smooth_median(
        self,
        z: np.ndarray,
        config: SurfacePlotConfig,
    ) -> np.ndarray:
        """Apply median smoothing."""
        # Handle NaN values
        assert z is not None, "z must be provided"
        mask = np.isnan(z)
        z_filled = np.nan_to_num(z, nan=np.nanmedian(z))

        size = config.smoothing_kernel_size
        if size % 2 == 0:
            size += 1

        result = median_filter(z_filled, size=size)
        result[mask] = np.nan

        return result

    def _smooth_uniform(
        self,
        z: np.ndarray,
        config: SurfacePlotConfig,
    ) -> np.ndarray:
        """Apply uniform (box) smoothing."""
        assert z is not None, "z must be provided"
        mask = np.isnan(z)
        z_filled = np.where(mask, 0, z)
        weights = np.where(mask, 0, 1).astype(float)

        z_smooth = uniform_filter(z_filled, size=config.smoothing_kernel_size)
        w_smooth = uniform_filter(weights, size=config.smoothing_kernel_size)

        with np.errstate(divide="ignore", invalid="ignore"):
            result = z_smooth / w_smooth
        result[mask] = np.nan

        return result

    def _smooth_savgol(
        self,
        z: np.ndarray,
        config: SurfacePlotConfig,
    ) -> np.ndarray:
        """Apply 2D Savitzky-Golay smoothing."""
        try:
            from scipy.signal import savgol_filter

            # Apply along each axis
            window = config.smoothing_kernel_size
            if window % 2 == 0:
                window += 1

            polyorder = min(3, window - 1)

            # Handle NaN by interpolating first
            mask = np.isnan(z)
            z_filled = z.copy()
            if np.any(mask):
                # Simple interpolation for NaN values
                z_filled = self._fill_nan_2d(z_filled)

            # Apply Savgol filter along both axes
            result = savgol_filter(z_filled, window, polyorder, axis=0)
            result = savgol_filter(result, window, polyorder, axis=1)

            # Restore NaN positions
            result[mask] = np.nan

            return result
        except ImportError as e:
            logger.warning(f"Savgol smoothing failed: {e}, using Gaussian")
            return self._smooth_gaussian(z, config)

    def _fill_nan_2d(self, arr: np.ndarray) -> np.ndarray:
        """Fill NaN values in 2D array using nearest neighbor."""
        assert arr is not None, "arr must be provided"
        mask = np.isnan(arr)
        if not np.any(mask):
            return arr

        filled = arr.copy()

        # Get indices of valid and invalid points
        valid_y, valid_x = np.where(~mask)
        invalid_y, invalid_x = np.where(mask)

        if len(valid_y) == 0:
            return np.zeros_like(arr)

        # Find nearest valid point for each invalid point
        valid_points = np.column_stack([valid_x, valid_y])
        invalid_points = np.column_stack([invalid_x, invalid_y])

        tree = interpolate.NearestNDInterpolator(valid_points, arr[~mask])
        filled[mask] = tree(invalid_points)

        return filled


def plot_surface_matplotlib(
    result: SurfacePlotResult,
    ax: Any = None,
    show_regression: bool = False,
    regression_surface: np.ndarray | None = None,
) -> Any:
    """Create a matplotlib 3D surface plot.

    Args:
        result: SurfacePlotResult from create_surface
        ax: Optional matplotlib 3D axes
        show_regression: Whether to show regression surface overlay
        regression_surface: Z values for regression surface

    Returns:
        matplotlib figure
    """
    assert result is not None, "result must be provided"
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = None
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

    config = result.config

    # Plot the data surface
    surf = ax.plot_surface(
        result.x_grid,
        result.y_grid,
        result.z_grid,
        cmap=config.colormap,
        alpha=config.alpha,
        linewidth=0 if not config.show_wireframe else 0.5,
        antialiased=True,
    )

    # Add scatter of original data points
    if config.show_scatter:
        ax.scatter(
            result.x_data,
            result.y_data,
            result.z_data,
            c="black",
            alpha=config.scatter_alpha,
            s=5,
            label="Data points",
        )

    # Plot regression surface if provided
    if show_regression and regression_surface is not None:
        ax.plot_surface(
            result.x_grid,
            result.y_grid,
            regression_surface,
            color="red",
            alpha=0.5,
            linewidth=0,
            label="Regression fit",
        )

    # Labels
    ax.set_xlabel(config.x_label or config.x_column)
    ax.set_ylabel(config.y_label or config.y_column)
    ax.set_zlabel(config.z_label or config.z_column)
    ax.set_title(
        config.title or f"{config.z_column} vs {config.x_column}, {config.y_column}"
    )

    # Colorbar
    if fig is not None:
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    return fig or ax.get_figure()


__all__ = [
    "InterpolationMethod",
    "SmoothingMethod",
    "SurfacePlotConfig",
    "SurfacePlotResult",
    "SurfacePlotEngine",
    "plot_surface_matplotlib",
]

"""Principal Component Analysis (PCA) Module.

Provides comprehensive PCA functionality for identifying:
- Components that explain the most variance
- Correlations between variables
- Dimensionality reduction
- Feature importance ranking

Designed for analyzing multivariate data from noisy sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PCAConfig:
    """Configuration for PCA analysis."""

    # Number of components (None = all)
    n_components: int | float | None = None

    # Scaling options
    standardize: bool = True
    center: bool = True

    # Advanced options
    svd_solver: str = "auto"  # "auto", "full", "arpack", "randomized"
    random_state: int | None = None

    # Variance threshold for component selection
    variance_threshold: float = 0.95


@dataclass
class PCAComponent:
    """Information about a single PCA component."""

    index: int
    explained_variance: float
    explained_variance_ratio: float
    cumulative_variance_ratio: float
    loadings: dict[str, float]
    singular_value: float


@dataclass
class PCAResult:
    """Complete PCA analysis results."""

    # Component information
    components: list[PCAComponent]
    n_components: int
    n_features: int
    n_samples: int

    # Transformed data
    transformed_data: pd.DataFrame

    # Feature analysis
    feature_importance: dict[str, float]
    feature_contributions: pd.DataFrame

    # Correlation analysis
    loading_matrix: pd.DataFrame
    correlation_matrix: pd.DataFrame

    # Statistics
    total_variance_explained: float
    kaiser_criterion_components: int
    elbow_point_components: int

    # Original column names
    feature_names: list[str]

    # Scree plot data
    scree_data: dict[str, np.ndarray] = field(default_factory=dict)


class PCAAnalyzer:
    """Comprehensive PCA analysis for multivariate data.

    Provides:
    - Standard PCA decomposition
    - Variance explained analysis
    - Feature importance ranking
    - Component interpretation
    - Automatic component selection
    """

    def __init__(self, config: PCAConfig | None = None) -> None:
        """Initialize the PCA analyzer.

        Args:
            config: PCA configuration options
        """
        self.config = config or PCAConfig()

    def _build_component_list(
        self,
        components: np.ndarray,
        explained_var: np.ndarray,
        explained_var_ratio: np.ndarray,
        singular_values: np.ndarray,
        feature_names: list[str],
    ) -> tuple[list[PCAComponent], np.ndarray]:
        """Build the list of PCAComponent objects and cumulative variance.

        Args:
            components: Principal components matrix
            explained_var: Explained variance per component
            explained_var_ratio: Explained variance ratio per component
            singular_values: Singular values
            feature_names: Names of input features

        Returns:
            Tuple of (component_list, cumulative_variance_ratio)
        """
        cumulative_var = np.cumsum(explained_var_ratio)
        component_list = []
        for i in range(len(explained_var)):
            loadings = {name: components[i, j] for j, name in enumerate(feature_names)}
            component_list.append(
                PCAComponent(
                    index=i + 1,
                    explained_variance=float(explained_var[i]),
                    explained_variance_ratio=float(explained_var_ratio[i]),
                    cumulative_variance_ratio=float(cumulative_var[i]),
                    loadings=loadings,
                    singular_value=float(singular_values[i]),
                )
            )
        return component_list, cumulative_var

    def analyze(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> PCAResult:
        """Perform complete PCA analysis.

        Args:
            df: DataFrame with numeric data
            columns: Columns to include (None = all numeric)

        Returns:
            Complete PCA analysis results
        """
        feature_names = self._select_columns(df, columns)
        data = df[feature_names].copy().dropna()

        if len(data) < 2:
            raise ValueError("Not enough data points for PCA (need at least 2)")

        X = data.values.astype(float)
        n_samples, n_features = X.shape

        X_processed, mean, std = self._preprocess(X)
        components, explained_var, explained_var_ratio, singular_values = self._fit_pca(
            X_processed
        )

        pc_labels = [f"PC{i+1}" for i in range(components.shape[0])]
        transformed_df = pd.DataFrame(
            X_processed @ components.T, columns=pc_labels, index=data.index,
        )
        loading_matrix = pd.DataFrame(
            components.T, index=feature_names, columns=pc_labels,
        )

        component_list, cumulative_var = self._build_component_list(
            components, explained_var, explained_var_ratio,
            singular_values, feature_names,
        )

        return PCAResult(
            components=component_list,
            n_components=len(component_list),
            n_features=n_features,
            n_samples=n_samples,
            transformed_data=transformed_df,
            feature_importance=self._calculate_feature_importance(
                components, explained_var_ratio, feature_names
            ),
            feature_contributions=self._calculate_feature_contributions(
                loading_matrix, explained_var_ratio
            ),
            loading_matrix=loading_matrix,
            correlation_matrix=pd.DataFrame(data).corr(),
            total_variance_explained=(
                float(cumulative_var[-1]) if len(cumulative_var) > 0 else 0.0
            ),
            kaiser_criterion_components=self._kaiser_criterion(explained_var),
            elbow_point_components=self._find_elbow(explained_var_ratio),
            feature_names=feature_names,
            scree_data={
                "eigenvalues": explained_var,
                "variance_ratio": explained_var_ratio,
                "cumulative_variance": cumulative_var,
            },
        )

    def select_components_by_variance(
        self,
        result: PCAResult,
        variance_threshold: float = 0.95,
    ) -> list[int]:
        """Select components that explain a target amount of variance.

        Args:
            result: PCA result
            variance_threshold: Cumulative variance to explain (0-1)

        Returns:
            List of component indices (1-based)
        """
        for comp in result.components:
            if comp.cumulative_variance_ratio >= variance_threshold:
                return list(range(1, comp.index + 1))
        return list(range(1, result.n_components + 1))

    def get_top_features_for_component(
        self,
        result: PCAResult,
        component_index: int,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Get features with highest loadings for a component.

        Args:
            result: PCA result
            component_index: Component index (1-based)
            top_n: Number of top features to return

        Returns:
            List of (feature_name, loading) tuples
        """
        if component_index < 1 or component_index > result.n_components:
            raise ValueError(f"Invalid component index: {component_index}")

        component = result.components[component_index - 1]
        sorted_loadings = sorted(
            component.loadings.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_loadings[:top_n]

    def interpret_components(
        self,
        result: PCAResult,
        loading_threshold: float = 0.3,
    ) -> dict[str, list[str]]:
        """Interpret what each component represents.

        Args:
            result: PCA result
            loading_threshold: Minimum absolute loading to include

        Returns:
            Dict mapping component names to lists of influential features
        """
        interpretations = {}

        for comp in result.components:
            influential = [
                f"{name} ({loading:+.3f})"
                for name, loading in comp.loadings.items()
                if abs(loading) >= loading_threshold
            ]
            # Sort by absolute loading
            influential.sort(
                key=lambda x: abs(float(x.split("(")[1].rstrip(")"))), reverse=True
            )
            interpretations[f"PC{comp.index}"] = influential

        return interpretations

    def _select_columns(
        self,
        df: pd.DataFrame,
        columns: list[str] | None,
    ) -> list[str]:
        """Select numeric columns for analysis."""
        if columns:
            valid = [c for c in columns if c in df.columns]
            if not valid:
                raise ValueError("No valid columns specified")
            return valid

        # Get all numeric columns
        numeric: list[str] = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            raise ValueError("No numeric columns found in DataFrame")
        return numeric

    def _preprocess(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Center and/or standardize data."""
        mean = np.zeros(X.shape[1])
        std = np.ones(X.shape[1])

        if self.config.center:
            mean = np.mean(X, axis=0)
            X = X - mean

        if self.config.standardize:
            std = np.std(X, axis=0)
            std[std == 0] = 1  # Prevent division by zero
            X = X / std

        return X, mean, std

    def _fit_pca(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform PCA using SVD.

        Returns:
            components, explained_variance, explained_variance_ratio, singular_values
        """
        n_samples, n_features = X.shape

        # Determine number of components
        if self.config.n_components is None:
            n_components = min(n_samples, n_features)
        elif isinstance(self.config.n_components, float):
            # Interpreted as variance threshold - compute later
            n_components = min(n_samples, n_features)
        else:
            n_components = min(self.config.n_components, n_samples, n_features)

        # SVD decomposition
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Components (principal axes)
        components = Vt[:n_components]

        # Explained variance
        explained_variance = (S**2) / (n_samples - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance

        # If n_components was a float (variance threshold), select components
        if isinstance(self.config.n_components, float):
            cumsum = np.cumsum(explained_variance_ratio)
            n_components = int(np.searchsorted(cumsum, self.config.n_components)) + 1
            n_components = min(n_components, len(explained_variance))

        return (
            components[:n_components],
            explained_variance[:n_components],
            explained_variance_ratio[:n_components],
            S[:n_components],
        )

    def _calculate_feature_importance(
        self,
        components: np.ndarray,
        explained_var_ratio: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Calculate overall feature importance from PCA.

        Importance = sum of (loading^2 * variance_explained) for each component
        """
        importance = np.zeros(len(feature_names))

        for i, var_ratio in enumerate(explained_var_ratio):
            importance += (components[i] ** 2) * var_ratio

        # Normalize to sum to 1
        importance = importance / np.sum(importance)

        return {
            name: float(imp)
            for name, imp in zip(feature_names, importance, strict=False)
        }

    def _calculate_feature_contributions(
        self,
        loading_matrix: pd.DataFrame,
        explained_var_ratio: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate feature contributions to each component.

        Contribution = loading^2 / sum(loading^2) * 100
        """
        contributions = loading_matrix.copy() ** 2

        # Normalize per component (column)
        for col in contributions.columns:
            col_sum = contributions[col].sum()
            if col_sum > 0:
                contributions[col] = (contributions[col] / col_sum) * 100

        # Add weighted importance column
        weights = explained_var_ratio[: len(contributions.columns)]
        contributions["Weighted_Importance"] = (
            sum(
                contributions[col] * w
                for col, w in zip(contributions.columns[:-1], weights, strict=False)
            )
            / sum(weights)
            if sum(weights) > 0
            else 0
        )

        return contributions

    def _kaiser_criterion(self, eigenvalues: np.ndarray) -> int:
        """Apply Kaiser criterion (eigenvalues > 1 for standardized data)."""
        return int(np.sum(eigenvalues > 1.0))

    def _find_elbow(self, variance_ratio: np.ndarray) -> int:
        """Find elbow point in scree plot using second derivative."""
        if len(variance_ratio) < 3:
            return len(variance_ratio)

        # Second derivative
        second_deriv = np.diff(variance_ratio, n=2)

        # Find maximum curvature (elbow)
        if len(second_deriv) > 0:
            elbow = int(np.argmax(np.abs(second_deriv))) + 2  # +2 for diff offset
            return min(elbow, len(variance_ratio))

        return 1


def create_scree_plot(result: PCAResult, ax: Any = None) -> Any:
    """Create a scree plot showing variance explained.

    Args:
        result: PCA result
        ax: Optional matplotlib axes

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    components = range(1, result.n_components + 1)
    result.scree_data["eigenvalues"]
    cumulative = result.scree_data["cumulative_variance"] * 100

    # Bar plot of individual variance
    ax.bar(
        components,
        [c.explained_variance_ratio * 100 for c in result.components],
        alpha=0.7,
        label="Individual variance",
    )

    # Line plot of cumulative variance
    ax2 = ax.twinx()
    ax2.plot(
        components,
        cumulative,
        "r-o",
        linewidth=2,
        markersize=6,
        label="Cumulative variance",
    )

    # Mark Kaiser criterion threshold
    ax.axhline(
        y=100 / result.n_features,
        color="g",
        linestyle="--",
        alpha=0.5,
        label=f"Kaiser threshold ({100/result.n_features:.1f}%)",
    )

    # Mark elbow point
    ax.axvline(
        x=result.elbow_point_components,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label=f"Elbow point (PC{result.elbow_point_components})",
    )

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax2.set_ylabel("Cumulative Variance (%)", color="r")

    ax.set_title("PCA Scree Plot")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax.set_xticks(list(components))

    return fig or ax.get_figure()


def create_loading_plot(
    result: PCAResult,
    pc_x: int = 1,
    pc_y: int = 2,
    ax: Any = None,
) -> Any:
    """Create a loading plot (biplot) for two components.

    Args:
        result: PCA result
        pc_x: Component for x-axis (1-based)
        pc_y: Component for y-axis (1-based)
        ax: Optional matplotlib axes

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    loadings = result.loading_matrix

    x = loadings[f"PC{pc_x}"]
    y = loadings[f"PC{pc_y}"]

    # Draw arrows
    for i, feature in enumerate(result.feature_names):
        ax.arrow(
            0,
            0,
            x.iloc[i],
            y.iloc[i],
            head_width=0.03,
            head_length=0.02,
            fc="blue",
            ec="blue",
            alpha=0.7,
        )
        ax.text(
            x.iloc[i] * 1.1,
            y.iloc[i] * 1.1,
            feature,
            fontsize=9,
            ha="center",
        )

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
    ax.add_patch(circle)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(
        f"PC{pc_x} ({result.components[pc_x-1].explained_variance_ratio*100:.1f}%)"
    )
    ax.set_ylabel(
        f"PC{pc_y} ({result.components[pc_y-1].explained_variance_ratio*100:.1f}%)"
    )
    ax.set_title("PCA Loading Plot")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="k", linestyle="-", linewidth=0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return fig or ax.get_figure()


__all__ = [
    "PCAConfig",
    "PCAComponent",
    "PCAResult",
    "PCAAnalyzer",
    "create_scree_plot",
    "create_loading_plot",
]

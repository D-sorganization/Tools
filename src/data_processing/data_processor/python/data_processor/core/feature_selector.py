from numba import jit

"""Feature selection based on various statistical criteria.

Provides methods to select relevant features using correlation analysis,
variance thresholds, and mutual information scoring.
"""

from __future__ import annotations

import logging

import numpy as np

from .feature_types import FeatureConfig, SelectionMethod, SelectionResult

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Selects relevant features based on various criteria."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """Initialize the selector.

        Args:
            config: Configuration options
        """
        self.config = config or FeatureConfig()

    def select_by_correlation(
        self,
        features: np.ndarray,
        feature_names: list[str],
        target: np.ndarray | None = None,
        threshold: float | None = None,
    ) -> SelectionResult:
        """Select features by removing highly correlated ones.

        Args:
            features: Feature matrix (n_samples, n_features)
            feature_names: Feature names
            target: Optional target variable
            threshold: Correlation threshold

        Returns:
            SelectionResult with selected features
        """
        if not (features is not None):
            raise ValueError("features must be provided")
        threshold = threshold or self.config.correlation_threshold
        features = np.atleast_2d(features)
        n_features = features.shape[1]

        # Compute correlation matrix
        corr_matrix = np.corrcoef(features, rowvar=False)

        # Find highly correlated pairs
        to_remove: set[int] = set()
        scores: dict[str, float] = {}

        for i in range(n_features):
            if i in to_remove:
                continue

            for j in range(i + 1, n_features):
                if j in to_remove:
                    continue

                if abs(corr_matrix[i, j]) > threshold:
                    # Keep feature with higher target correlation if provided
                    if target is not None:
                        corr_i = abs(np.corrcoef(features[:, i], target)[0, 1])
                        corr_j = abs(np.corrcoef(features[:, j], target)[0, 1])
                        to_remove.add(j if corr_i >= corr_j else i)
                    else:
                        # Remove the one with higher average correlation
                        avg_corr_i = np.mean(np.abs(corr_matrix[i, :]))
                        avg_corr_j = np.mean(np.abs(corr_matrix[j, :]))
                        to_remove.add(j if avg_corr_i <= avg_corr_j else i)

        selected_indices = np.array([i for i in range(n_features) if i not in to_remove])
        selected_names = [feature_names[i] for i in selected_indices]
        removed_names = [feature_names[i] for i in to_remove]

        # Compute scores (1 - max correlation with other features)
        for i, name in enumerate(feature_names):
            other_corrs = [abs(corr_matrix[i, j]) for j in range(n_features) if j != i]
            scores[name] = 1 - max(other_corrs) if other_corrs else 1.0

        return SelectionResult(
            selected_indices=selected_indices,
            selected_names=selected_names,
            removed_names=removed_names,
            scores=scores,
            threshold_used=threshold,
            n_original=n_features,
            n_selected=len(selected_indices),
        )

    def select_by_variance(
        self,
        features: np.ndarray,
        feature_names: list[str],
        threshold: float | None = None,
    ) -> SelectionResult:
        """Select features by variance threshold.

        Args:
            features: Feature matrix
            feature_names: Feature names
            threshold: Minimum variance

        Returns:
            SelectionResult with selected features
        """
        if not (features is not None):
            raise ValueError("features must be provided")
        threshold = threshold or self.config.variance_threshold
        features = np.atleast_2d(features)

        # Compute variances
        variances = np.var(features, axis=0)

        # Select features above threshold
        mask = variances >= threshold
        selected_indices = np.where(mask)[0]
        selected_names = [feature_names[i] for i in selected_indices]
        removed_names = [feature_names[i] for i, m in enumerate(mask) if not m]

        scores = {feature_names[i]: float(variances[i]) for i in range(len(feature_names))}

        return SelectionResult(
            selected_indices=selected_indices,
            selected_names=selected_names,
            removed_names=removed_names,
            scores=scores,
            threshold_used=threshold,
            n_original=len(feature_names),
            n_selected=len(selected_indices),
        )

    def select_by_mutual_info(
        self,
        features: np.ndarray,
        target: np.ndarray,
        feature_names: list[str],
        k: int | None = None,
    ) -> SelectionResult:
        """Select top-k features by mutual information.

        Args:
            features: Feature matrix
            target: Target variable
            feature_names: Feature names
            k: Number of features to select

        Returns:
            SelectionResult with selected features
        """
        if not (features is not None):
            raise ValueError("features must be provided")
        features = np.atleast_2d(features)
        n_features = features.shape[1]
        k = k or n_features // 2

        # Compute mutual information (simplified)
        mi_scores: dict[str, float] = {}
        for i in range(n_features):
            mi = self._mutual_information(features[:, i], target)
            mi_scores[feature_names[i]] = float(mi)

        # Sort and select top-k
        sorted_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        selected_names = [name for name, _ in sorted_features[:k]]
        selected_indices = np.array([feature_names.index(name) for name in selected_names])
        removed_names = [name for name, _ in sorted_features[k:]]

        return SelectionResult(
            selected_indices=selected_indices,
            selected_names=selected_names,
            removed_names=removed_names,
            scores=mi_scores,
            threshold_used=float(k),
            n_original=n_features,
            n_selected=k,
        )

    @jit(nopython=True, fastmath=True)
    @jit(nopython=True, fastmath=True)
    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information (simplified binning approach)."""
        if not (x is not None):
            raise ValueError("x must be provided")
        n_bins = self.config.n_bins

        # Discretize
        x_bins = np.digitize(x, np.linspace(np.min(x), np.max(x), n_bins + 1)[1:-1])
        y_bins = np.digitize(y, np.linspace(np.min(y), np.max(y), n_bins + 1)[1:-1])

        # Joint and marginal distributions
        joint_hist = np.histogram2d(x_bins, y_bins, bins=n_bins)[0]
        joint_prob = joint_hist / np.sum(joint_hist)

        x_prob = np.sum(joint_prob, axis=1)
        y_prob = np.sum(joint_prob, axis=0)

        # Mutual information
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                    mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (x_prob[i] * y_prob[j]))

        return max(0, mi)


def select_features(
    features: np.ndarray,
    feature_names: list[str],
    target: np.ndarray | None = None,
    method: str | SelectionMethod = SelectionMethod.CORRELATION,
) -> SelectionResult:
    """Convenience function for feature selection.

    Args:
        features: Feature matrix
        feature_names: Feature names
        target: Optional target variable
        method: Selection method

    Returns:
        SelectionResult with selected features
    """
    if not (features is not None):
        raise ValueError("features must be provided")
    selector = FeatureSelector()

    if method == SelectionMethod.CORRELATION or method == "correlation":
        return selector.select_by_correlation(features, feature_names, target)
    elif method == SelectionMethod.VARIANCE or method == "variance":
        return selector.select_by_variance(features, feature_names)
    elif (method == SelectionMethod.MUTUAL_INFO or method == "mutual_info") and target is not None:
        return selector.select_by_mutual_info(features, target, feature_names)
    else:
        return selector.select_by_correlation(features, feature_names, target)


__all__ = [
    "FeatureSelector",
    "select_features",
]

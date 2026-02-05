"""Outlier Detection Ensemble Module.

Provides multiple outlier detection methods for robust identification.
Excellent for:
- Sensor fault detection
- Data quality assessment
- Anomaly identification
- Pre-processing noisy data

Includes:
- Isolation Forest
- Local Outlier Factor (LOF)
- DBSCAN-based outlier detection
- Statistical methods (Z-score, IQR, Grubbs)
- Ensemble voting for robust detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class OutlierMethod(Enum):
    """Available outlier detection methods."""

    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    IQR = "iqr"
    GRUBBS = "grubbs"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    DBSCAN = "dbscan"
    MAHALANOBIS = "mahalanobis"
    ENSEMBLE = "ensemble"


@dataclass
class OutlierConfig:
    """Configuration for outlier detection."""

    # Method selection
    method: OutlierMethod = OutlierMethod.ENSEMBLE

    # Statistical thresholds
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    grubbs_alpha: float = 0.05

    # Isolation Forest parameters
    if_n_estimators: int = 100
    if_contamination: float = 0.1
    if_max_samples: str | int = "auto"

    # LOF parameters
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.1

    # DBSCAN parameters
    dbscan_eps: float | None = None  # Auto-estimate if None
    dbscan_min_samples: int = 5

    # Mahalanobis parameters
    mahalanobis_threshold: float = 3.0

    # Ensemble parameters
    ensemble_methods: list[OutlierMethod] = field(
        default_factory=lambda: [
            OutlierMethod.ZSCORE,
            OutlierMethod.IQR,
            OutlierMethod.LOF,
        ]
    )
    ensemble_voting: str = "majority"  # "majority", "any", "all"


@dataclass
class OutlierResult:
    """Results from outlier detection."""

    # Boolean mask of outliers
    outlier_mask: np.ndarray

    # Outlier scores (higher = more outlier-like)
    scores: np.ndarray

    # Number of outliers
    n_outliers: int
    outlier_fraction: float

    # Outlier indices
    outlier_indices: np.ndarray

    # Method used
    method: str

    # Per-method results (for ensemble)
    method_results: dict[str, np.ndarray] = field(default_factory=dict)

    # Statistics
    statistics: dict[str, Any] = field(default_factory=dict)


class OutlierDetector:
    """Ensemble outlier detection for robust anomaly identification."""

    def __init__(self, config: OutlierConfig | None = None) -> None:
        """Initialize the detector.

        Args:
            config: Outlier detection configuration
        """
        self.config = config or OutlierConfig()

    def detect(
        self,
        data: np.ndarray | pd.DataFrame,
        columns: list[str] | None = None,
    ) -> OutlierResult:
        """Detect outliers in data.

        Args:
            data: Input data (1D array, 2D array, or DataFrame)
            columns: Columns to use (for DataFrame)

        Returns:
            OutlierResult with outlier mask and diagnostics
        """
        # Convert to numpy array
        X = self._prepare_data(data, columns)

        # Select method
        if self.config.method == OutlierMethod.ENSEMBLE:
            return self._detect_ensemble(X)
        else:
            mask, scores = self._detect_single(X, self.config.method)
            return self._build_result(mask, scores, self.config.method.value)

    def _prepare_data(
        self,
        data: np.ndarray | pd.DataFrame,
        columns: list[str] | None,
    ) -> np.ndarray:
        """Prepare data for outlier detection."""
        if isinstance(data, pd.DataFrame):
            if columns:
                data = data[columns]
            data = data.select_dtypes(include=[np.number])
            X = data.values
        else:
            X = np.asarray(data)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X

    def _detect_single(
        self,
        X: np.ndarray,
        method: OutlierMethod,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect outliers using a single method."""
        if method == OutlierMethod.ZSCORE:
            return self._detect_zscore(X)
        elif method == OutlierMethod.MODIFIED_ZSCORE:
            return self._detect_modified_zscore(X)
        elif method == OutlierMethod.IQR:
            return self._detect_iqr(X)
        elif method == OutlierMethod.GRUBBS:
            return self._detect_grubbs(X)
        elif method == OutlierMethod.ISOLATION_FOREST:
            return self._detect_isolation_forest(X)
        elif method == OutlierMethod.LOF:
            return self._detect_lof(X)
        elif method == OutlierMethod.DBSCAN:
            return self._detect_dbscan(X)
        elif method == OutlierMethod.MAHALANOBIS:
            return self._detect_mahalanobis(X)
        else:
            return self._detect_zscore(X)

    def _detect_ensemble(self, X: np.ndarray) -> OutlierResult:
        """Detect outliers using ensemble of methods."""
        method_results = {}
        all_masks = []

        # Run each method
        for method in self.config.ensemble_methods:
            try:
                mask, scores = self._detect_single(X, method)
                method_results[method.value] = mask
                all_masks.append(mask)
            except Exception as e:
                logger.warning(f"Method {method.value} failed: {e}")

        if not all_masks:
            # Fallback to Z-score
            mask, scores = self._detect_zscore(X)
            return self._build_result(mask, scores, "zscore_fallback")

        # Combine results
        all_masks = np.array(all_masks)

        if self.config.ensemble_voting == "majority":
            # Outlier if majority of methods agree
            votes = np.sum(all_masks, axis=0)
            threshold = len(all_masks) / 2
            combined_mask = votes > threshold
        elif self.config.ensemble_voting == "any":
            # Outlier if any method flags it
            combined_mask = np.any(all_masks, axis=0)
        else:  # "all"
            # Outlier only if all methods agree
            combined_mask = np.all(all_masks, axis=0)

        # Score based on vote count
        scores = np.sum(all_masks, axis=0) / len(all_masks)

        result = self._build_result(combined_mask, scores, "ensemble")
        result.method_results = method_results
        return result

    def _detect_zscore(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Z-score based outlier detection."""
        # Handle multivariate by computing per-column and combining
        n, p = X.shape
        z_scores = np.zeros((n, p))

        for j in range(p):
            col = X[:, j]
            valid = ~np.isnan(col)
            mean = np.nanmean(col)
            std = np.nanstd(col)
            if std > 0:
                z_scores[valid, j] = np.abs(col[valid] - mean) / std

        # Max Z-score across columns
        max_z = np.nanmax(z_scores, axis=1)
        mask = max_z > self.config.zscore_threshold

        return mask, max_z

    def _detect_modified_zscore(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Modified Z-score (MAD-based) outlier detection."""
        n, p = X.shape
        modified_z = np.zeros((n, p))

        for j in range(p):
            col = X[:, j]
            valid = ~np.isnan(col)
            median = np.nanmedian(col)
            mad = np.nanmedian(np.abs(col[valid] - median))

            if mad > 0:
                modified_z[valid, j] = 0.6745 * np.abs(col[valid] - median) / mad

        max_z = np.nanmax(modified_z, axis=1)
        mask = max_z > self.config.zscore_threshold

        return mask, max_z

    def _detect_iqr(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """IQR-based outlier detection."""
        n, p = X.shape
        outlier_flags = np.zeros((n, p), dtype=bool)
        distances = np.zeros((n, p))

        for j in range(p):
            col = X[:, j]
            q1 = np.nanpercentile(col, 25)
            q3 = np.nanpercentile(col, 75)
            iqr = q3 - q1

            lower = q1 - self.config.iqr_multiplier * iqr
            upper = q3 + self.config.iqr_multiplier * iqr

            outlier_flags[:, j] = (col < lower) | (col > upper)

            # Distance from bounds
            distances[:, j] = np.maximum(lower - col, col - upper) / (iqr + 1e-10)
            distances[:, j] = np.maximum(distances[:, j], 0)

        mask = np.any(outlier_flags, axis=1)
        scores = np.nanmax(distances, axis=1)

        return mask, scores

    def _detect_grubbs(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grubbs test for outliers (iterative)."""
        n, p = X.shape
        mask = np.zeros(n, dtype=bool)
        scores = np.zeros(n)

        for j in range(p):
            col = X[:, j].copy()
            valid = ~np.isnan(col)
            indices = np.where(valid)[0]

            while len(indices) > 2:
                data = col[indices]
                mean = np.mean(data)
                std = np.std(data)

                if std == 0:
                    break

                # Find most extreme point
                g_scores = np.abs(data - mean) / std
                max_idx = np.argmax(g_scores)
                g_stat = g_scores[max_idx]

                # Critical value
                n_curr = len(data)
                t_crit = stats.t.ppf(
                    1 - self.config.grubbs_alpha / (2 * n_curr), n_curr - 2
                )
                g_crit = ((n_curr - 1) / np.sqrt(n_curr)) * np.sqrt(
                    t_crit**2 / (n_curr - 2 + t_crit**2)
                )

                if g_stat > g_crit:
                    outlier_idx = indices[max_idx]
                    mask[outlier_idx] = True
                    scores[outlier_idx] = max(scores[outlier_idx], g_stat)
                    indices = np.delete(indices, max_idx)
                else:
                    break

        return mask, scores

    def _detect_isolation_forest(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Isolation Forest outlier detection."""
        n, p = X.shape

        # Handle NaN by imputing with median
        X_clean = X.copy()
        for j in range(p):
            nan_mask = np.isnan(X_clean[:, j])
            if np.any(nan_mask):
                X_clean[nan_mask, j] = np.nanmedian(X_clean[:, j])

        # Simple Isolation Forest implementation
        scores = self._isolation_forest_scores(X_clean)

        # Threshold based on contamination
        threshold = np.percentile(scores, 100 * (1 - self.config.if_contamination))
        mask = scores > threshold

        return mask, scores

    def _isolation_forest_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute Isolation Forest anomaly scores."""
        n, p = X.shape
        n_trees = self.config.if_n_estimators
        max_depth = int(np.ceil(np.log2(n)))

        # Sample size
        if self.config.if_max_samples == "auto":
            sample_size = min(256, n)
        else:
            sample_size = min(self.config.if_max_samples, n)

        # Build trees and compute path lengths
        path_lengths = np.zeros((n, n_trees))

        for t in range(n_trees):
            # Bootstrap sample
            indices = np.random.choice(n, size=sample_size, replace=False)
            X_sample = X[indices]

            # Build tree and compute path lengths for all points
            tree = self._build_isolation_tree(X_sample, max_depth)
            for i in range(n):
                path_lengths[i, t] = self._path_length(X[i], tree, 0)

        # Average path length
        avg_path = np.mean(path_lengths, axis=1)

        # Expected path length
        c_n = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n if n > 1 else 1

        # Anomaly score
        scores = 2 ** (-avg_path / c_n)

        return scores

    def _build_isolation_tree(
        self,
        X: np.ndarray,
        max_depth: int,
        depth: int = 0,
    ) -> dict[str, Any]:
        """Build a single isolation tree."""
        n, p = X.shape

        if depth >= max_depth or n <= 1:
            return {"type": "leaf", "size": n}

        # Random feature and split
        feature = np.random.randint(p)
        min_val = np.min(X[:, feature])
        max_val = np.max(X[:, feature])

        if min_val == max_val:
            return {"type": "leaf", "size": n}

        split = np.random.uniform(min_val, max_val)

        left_mask = X[:, feature] < split
        right_mask = ~left_mask

        return {
            "type": "split",
            "feature": feature,
            "split": split,
            "left": self._build_isolation_tree(X[left_mask], max_depth, depth + 1),
            "right": self._build_isolation_tree(X[right_mask], max_depth, depth + 1),
        }

    def _path_length(
        self,
        x: np.ndarray,
        tree: dict[str, Any],
        depth: int,
    ) -> float:
        """Compute path length for a single point."""
        if tree["type"] == "leaf":
            n = tree["size"]
            if n <= 1:
                return depth
            c_n = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
            return depth + c_n

        if x[tree["feature"]] < tree["split"]:
            return self._path_length(x, tree["left"], depth + 1)
        else:
            return self._path_length(x, tree["right"], depth + 1)

    def _detect_lof(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Local Outlier Factor detection."""
        n, p = X.shape
        k = min(self.config.lof_n_neighbors, n - 1)

        # Handle NaN
        X_clean = X.copy()
        for j in range(p):
            nan_mask = np.isnan(X_clean[:, j])
            if np.any(nan_mask):
                X_clean[nan_mask, j] = np.nanmedian(X_clean[:, j])

        # Compute pairwise distances
        distances = self._pairwise_distances(X_clean)

        # k-distance and k-neighbors
        k_distances = np.zeros(n)
        k_neighbors = []

        for i in range(n):
            sorted_idx = np.argsort(distances[i])
            k_neighbors.append(sorted_idx[1 : k + 1])  # Exclude self
            k_distances[i] = distances[i, sorted_idx[k]]

        # Reachability distance
        reach_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                reach_dist[i, j] = max(k_distances[j], distances[i, j])

        # Local reachability density
        lrd = np.zeros(n)
        for i in range(n):
            neighbors = k_neighbors[i]
            lrd[i] = k / np.sum(reach_dist[i, neighbors])

        # Local outlier factor
        lof = np.zeros(n)
        for i in range(n):
            neighbors = k_neighbors[i]
            lof[i] = np.mean(lrd[neighbors]) / (lrd[i] + 1e-10)

        # Threshold based on contamination
        threshold = np.percentile(lof, 100 * (1 - self.config.lof_contamination))
        mask = lof > threshold

        return mask, lof

    def _detect_dbscan(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """DBSCAN-based outlier detection (noise points are outliers)."""
        n, p = X.shape

        # Handle NaN
        X_clean = X.copy()
        for j in range(p):
            nan_mask = np.isnan(X_clean[:, j])
            if np.any(nan_mask):
                X_clean[nan_mask, j] = np.nanmedian(X_clean[:, j])

        # Estimate eps if not provided
        eps = self.config.dbscan_eps
        if eps is None:
            # Use k-distance plot heuristic
            distances = self._pairwise_distances(X_clean)
            k = self.config.dbscan_min_samples
            k_dists = np.sort(distances, axis=1)[:, k]
            eps = np.percentile(k_dists, 90)

        # Simple DBSCAN implementation
        labels = self._dbscan(X_clean, eps, self.config.dbscan_min_samples)

        # Noise points (label = -1) are outliers
        mask = labels == -1

        # Score based on distance to nearest core point
        scores = np.zeros(n)
        core_mask = labels >= 0
        if np.any(core_mask):
            core_points = X_clean[core_mask]
            for i in range(n):
                if mask[i]:
                    dists = np.linalg.norm(core_points - X_clean[i], axis=1)
                    scores[i] = np.min(dists)

        return mask, scores

    def _dbscan(
        self,
        X: np.ndarray,
        eps: float,
        min_samples: int,
    ) -> np.ndarray:
        """Simple DBSCAN implementation."""
        n = len(X)
        labels = np.full(n, -1)
        distances = self._pairwise_distances(X)
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1:
                continue

            neighbors = np.where(distances[i] <= eps)[0]

            if len(neighbors) < min_samples:
                continue  # Noise point (for now)

            # Start new cluster
            labels[i] = cluster_id
            seed_set = list(neighbors)

            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                if labels[q] == -1:
                    labels[q] = cluster_id

                if labels[q] != -1 and labels[q] != cluster_id:
                    j += 1
                    continue

                labels[q] = cluster_id
                q_neighbors = np.where(distances[q] <= eps)[0]

                if len(q_neighbors) >= min_samples:
                    for neighbor in q_neighbors:
                        if neighbor not in seed_set:
                            seed_set.append(neighbor)

                j += 1

            cluster_id += 1

        return labels

    def _detect_mahalanobis(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mahalanobis distance based outlier detection."""
        n, p = X.shape

        # Handle NaN
        X_clean = X.copy()
        for j in range(p):
            nan_mask = np.isnan(X_clean[:, j])
            if np.any(nan_mask):
                X_clean[nan_mask, j] = np.nanmedian(X_clean[:, j])

        # Compute mean and covariance
        mean = np.mean(X_clean, axis=0)
        cov = np.cov(X_clean.T)

        # Handle singular covariance
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        # Mahalanobis distances
        diff = X_clean - mean
        distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        # Use configured threshold
        mask = distances > self.config.mahalanobis_threshold

        return mask, distances

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        n = len(X)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(X[i] - X[j])
                distances[i, j] = d
                distances[j, i] = d

        return distances

    def _build_result(
        self,
        mask: np.ndarray,
        scores: np.ndarray,
        method: str,
    ) -> OutlierResult:
        """Build OutlierResult from detection results."""
        n = len(mask)
        n_outliers = int(np.sum(mask))

        return OutlierResult(
            outlier_mask=mask,
            scores=scores,
            n_outliers=n_outliers,
            outlier_fraction=n_outliers / n if n > 0 else 0,
            outlier_indices=np.where(mask)[0],
            method=method,
        )


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "ensemble",
    threshold: float = 3.0,
) -> OutlierResult:
    """Detect outliers in a DataFrame.

    Convenience function for outlier detection.

    Args:
        df: Input DataFrame
        columns: Columns to analyze (None = all numeric)
        method: Detection method
        threshold: Threshold for statistical methods

    Returns:
        OutlierResult
    """
    config = OutlierConfig(
        method=OutlierMethod(method),
        zscore_threshold=threshold,
    )

    detector = OutlierDetector(config)
    return detector.detect(df, columns)


def remove_outliers(
    df: pd.DataFrame,
    result: OutlierResult,
    method: str = "remove",
) -> pd.DataFrame:
    """Remove or flag outliers in a DataFrame.

    Args:
        df: Input DataFrame
        result: OutlierResult from detection
        method: "remove" (drop rows) or "nan" (replace with NaN)

    Returns:
        Cleaned DataFrame
    """
    if method == "remove":
        return df.iloc[~result.outlier_mask].reset_index(drop=True)
    else:
        output = df.copy()
        numeric_cols = output.select_dtypes(include=[np.number]).columns
        output.loc[result.outlier_mask, numeric_cols] = np.nan
        return output


__all__ = [
    "OutlierMethod",
    "OutlierConfig",
    "OutlierResult",
    "OutlierDetector",
    "detect_outliers",
    "remove_outliers",
]

"""Tests for data_processor.core.feature_selector.

Covers the target-correlation precompute optimization (issue #3745): hoisting
``np.corrcoef(feature, target)`` out of the O(F^2) pair loop must not change
which features are removed.
"""

from __future__ import annotations

import numpy as np
import pytest
from data_processor.core.feature_selector import FeatureSelector

pytestmark = pytest.mark.unit


def _correlated_features(seed: int = 0) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build features where f0~f1 (redundant) and the target favors f0."""
    rng = np.random.RandomState(seed)
    base = rng.randn(200)
    f0 = base + rng.randn(200) * 0.01
    f1 = base + rng.randn(200) * 0.01  # ~perfectly correlated with f0
    f2 = rng.randn(200)  # independent
    features = np.column_stack([f0, f1, f2])
    names = ["f0", "f1", "f2"]
    target = f0 + rng.randn(200) * 0.05  # closest to f0
    return features, names, target


def test_select_by_correlation_target_branch_matches_reference() -> None:
    """Precomputed target-correlation vector reproduces the per-pair recompute."""
    features, names, target = _correlated_features()
    selector = FeatureSelector()

    result = selector.select_by_correlation(features, names, target=target)

    # One of the redundant pair (f0, f1) must be dropped; the one more
    # correlated with the target (f0) is kept.
    assert "f0" in result.selected_names
    assert "f1" in result.removed_names
    assert "f2" in result.selected_names


def test_target_precompute_is_numerically_identical_to_inline() -> None:
    """The hoisted |corr(feature, target)| equals the original inline value."""
    features, _names, target = _correlated_features(seed=3)
    n_features = features.shape[1]

    precomputed = np.array(
        [abs(np.corrcoef(features[:, k], target)[0, 1]) for k in range(n_features)]
    )
    for i in range(n_features):
        inline = abs(np.corrcoef(features[:, i], target)[0, 1])
        assert precomputed[i] == pytest.approx(inline, abs=0.0, rel=0.0)


def test_no_target_uses_average_correlation() -> None:
    """Without a target the average-correlation tie-break path still works."""
    features, names, _target = _correlated_features(seed=1)
    selector = FeatureSelector()

    result = selector.select_by_correlation(features, names)

    assert result.n_selected < result.n_original

"""BCa bootstrap limits follow Efron's formula, including when ``a == 0``.

``alpha_k = Phi(z0 + (z0 + z_k) / (1 - a (z0 + z_k)))``. With ``a == 0`` this is
``Phi(2 z0 + z_k)``; a shortcut returning ``Phi(z0 + z_k)`` applies only half the
bias correction. The reference reuses the quantifier's own normal ppf/cdf so
only the BCa formula is under test.
"""

from __future__ import annotations

import numpy as np
import pytest
from data_processor.core.uncertainty_quantification import (
    BootstrapMethod,
    UncertaintyConfig,
    UncertaintyQuantifier,
)

pytestmark = pytest.mark.unit

CONFIDENCE = 0.95


def _quantifier() -> UncertaintyQuantifier:
    config = UncertaintyConfig(
        n_bootstrap=2000, random_seed=7, confidence_level=CONFIDENCE
    )
    return UncertaintyQuantifier(config)


def _bca(data: np.ndarray, statistic):
    return _quantifier().bootstrap_ci(data, statistic, method=BootstrapMethod.BCA)


def _efron_limits(result) -> tuple[float, float]:
    normal = _quantifier()
    z0, a = result.bias, result.acceleration
    alpha = 1 - CONFIDENCE
    limits = []
    for z in (normal._normal_ppf(alpha / 2), normal._normal_ppf(1 - alpha / 2)):
        level = normal._normal_cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        level = float(np.clip(level, 0.001, 0.999))
        limits.append(float(np.percentile(result.bootstrap_statistics, 100 * level)))
    return limits[0], limits[1]


def test_zero_acceleration_applies_the_full_bias_correction() -> None:
    # Symmetric integers: every leave-one-out median is exactly +-0.5 or 0, so
    # the jackknife acceleration is exactly 0, while ties at the sample median
    # make the bootstrap bias correction z0 clearly non-zero.
    data = np.arange(-7.0, 8.0)
    result = _bca(data, np.median)
    assert result.acceleration == 0.0
    assert abs(result.bias) > 0.05
    lower, upper = _efron_limits(result)
    assert result.ci_lower == pytest.approx(lower, abs=1e-12)
    assert result.ci_upper == pytest.approx(upper, abs=1e-12)


def test_nonzero_acceleration_matches_efron() -> None:
    data = np.random.default_rng(3).exponential(size=30)
    result = _bca(data, np.mean)
    assert result.acceleration != 0.0
    lower, upper = _efron_limits(result)
    assert result.ci_lower == pytest.approx(lower, abs=1e-12)
    assert result.ci_upper == pytest.approx(upper, abs=1e-12)

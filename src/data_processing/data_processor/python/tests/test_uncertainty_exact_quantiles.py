"""Interval critical values come from exact t / normal quantiles.

Hand-rolled approximations understated the Student-t quantile badly at small
residual degrees of freedom (t_{0.975, 2} = 4.303, approximated as 3.853), so
OLS prediction intervals were about 10% too narrow exactly where samples are
smallest. The reference intervals here use ``scipy.stats`` directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from data_processor.core.uncertainty_quantification import (
    UncertaintyConfig,
    UncertaintyQuantifier,
)
from scipy import stats

pytestmark = pytest.mark.unit

CONFIDENCE = 0.95


def _quantifier() -> UncertaintyQuantifier:
    return UncertaintyQuantifier(UncertaintyConfig(confidence_level=CONFIDENCE))


def _reference_prediction_half_width(
    x: np.ndarray, y: np.ndarray, x_new: float
) -> float:
    design = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    dof = len(x) - 2
    mse = float(np.sum((y - design @ beta) ** 2) / dof)
    row = np.array([1.0, x_new])
    leverage = float(row @ np.linalg.inv(design.T @ design) @ row)
    t_crit = stats.t.ppf(1 - (1 - CONFIDENCE) / 2, dof)
    return float(t_crit * np.sqrt(mse * (1 + leverage)))


@pytest.mark.parametrize("n", [4, 5, 8, 40])
def test_prediction_interval_uses_exact_t_quantile(n: int) -> None:
    rng = np.random.default_rng(n)
    x = np.linspace(0.0, 1.0, n)
    y = 2.0 + 3.0 * x + rng.normal(0.0, 0.5, n)
    result = _quantifier().prediction_intervals(x, y, np.array([0.5]))
    half_width = float(result.upper[0] - result.predicted[0])
    expected = _reference_prediction_half_width(x, y, 0.5)
    assert half_width == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("p", [0.001, 0.025, 0.3, 0.5, 0.8, 0.975, 0.999])
def test_normal_quantile_and_cdf_are_exact(p: float) -> None:
    quantifier = _quantifier()
    z = quantifier._normal_ppf(p)
    assert z == pytest.approx(stats.norm.ppf(p), abs=1e-12)
    assert quantifier._normal_cdf(z) == pytest.approx(p, abs=1e-12)


@pytest.mark.parametrize("p", [0.0, 1.0, float("nan"), float("inf")])
def test_normal_quantile_rejects_out_of_range(p: float) -> None:
    with pytest.raises(ValueError):
        _quantifier()._normal_ppf(p)

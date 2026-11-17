"""Cook's distance matches its leave-one-out definition.

``D_i = sum_j (yhat_j - yhat_j(i))^2 / ((p + 1) * s^2)``, equivalently
``r_i^2 / (p + 1) * h_i / (1 - h_i)`` with the *internally studentized* residual
``r_i = e_i / (s * sqrt(1 - h_i))``. Using ``e_i / s`` instead understates every
distance by ``1 - h_i``, hiding exactly the high-leverage points it should flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _loo_cooks_distance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    fitted = A @ beta
    s2 = np.sum((y - fitted) ** 2) / (n - p - 1)
    out = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        beta_i = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        out[i] = np.sum((fitted - A @ beta_i) ** 2) / ((p + 1) * s2)
    return out


def _fit(df: pd.DataFrame):
    from data_processor.core.regression import MultivariateRegressor, RegressionConfig

    regressor = MultivariateRegressor(RegressionConfig(compute_diagnostics=True))
    return regressor.fit(df, target="y", predictors=["x1", "x2"])


@pytest.fixture
def data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=40)
    x2 = rng.normal(size=40)
    x1[0], x2[0] = 6.0, -5.0  # one high-leverage point
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(scale=0.3, size=40)
    y[0] += 4.0  # and make it influential
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_cooks_distance_matches_leave_one_out(data: pd.DataFrame) -> None:
    result = _fit(data)
    expected = _loo_cooks_distance(data[["x1", "x2"]].to_numpy(), data["y"].to_numpy())
    np.testing.assert_allclose(result.diagnostics.cooks_distance, expected, rtol=1e-8)


def test_high_leverage_influential_point_is_flagged(data: pd.DataFrame) -> None:
    diagnostics = _fit(data).diagnostics
    assert 0 in diagnostics.high_leverage_points
    assert 0 in diagnostics.influential_points
    assert int(np.argmax(diagnostics.cooks_distance)) == 0


def test_studentized_residuals_use_leverage(data: pd.DataFrame) -> None:
    diagnostics = _fit(data).diagnostics
    n, p = len(data), 2
    mse = np.sum(diagnostics.residuals**2) / (n - p - 1)
    expected = diagnostics.residuals / np.sqrt(mse * (1 - diagnostics.leverage))
    np.testing.assert_allclose(diagnostics.studentized_residuals, expected, rtol=1e-10)

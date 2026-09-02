"""Reproducible regression and optional shallow-neural-network models.

Ported from UpstreamDrift ``src/shared/python/launch_monitor/modeling.py``
(226 lines) under ADR-0046 Stage 1 — step **P8** of the ADR-0046 G1 port plan
(UpstreamDrift ``docs/adr/0048-launch-monitor-port-plan.md``). The
implementation is UpstreamDrift's, carried over unchanged rather than
reimplemented; its authors retain authorship. No behaviour is added, removed,
or limited by the move.

The port plan records **no ``rate_of_closure`` counterpart at all** — "no
scikit-learn anywhere in ``rate_of_closure``" — so nothing here collides by
name and no ADR-0046 G0 divergence applies. The plan's *TypeScript-Twin
Obligation* risk names this module specifically as the one that is not
portable to the web runtime under a 1:1 twin posture, because of the optional
``MLPRegressor`` branch.

Four of the five estimators are hand-written NumPy — least squares, ridge by
normal equations, and lasso/elastic-net by coordinate descent — so the default
path has no scikit-learn dependency at all. Only ``model="mlp"`` imports it,
lazily, and raises a named ``ImportError`` when it is absent.

Three guarantees hold across every model:

* **Reproducibility.** The split is drawn from a seeded ``default_rng`` and
  standardisation statistics come from the *training* rows only, so the same
  frame, seed, and recipe give bit-identical metrics and predictions.
* **No identity leakage.** ``_validate_no_leakage`` refuses a target that is
  also a feature, a feature whose ``derived_from`` record in
  :mod:`shared.python.launch_monitor.schema` names the target, and a feature
  that appears in the target's own ``derived_from``. Predicting
  ``ball_speed`` from ``smash_factor`` is reading the answer off the sheet, and
  it is refused rather than warned about.
* **Grouped splitting.** ``group_column`` puts whole groups on one side of the
  split, so shots from one session cannot appear in both train and test.

.. note::

   The ``ImportError`` raised by the ``mlp`` branch names UpstreamDrift's
   ``upstream-drift[analysis]`` extra. That string is UpstreamDrift's and is
   carried over verbatim, because this step is a pure port and the value of a
   pure port is that its diff is mechanically checkable. Repointing the hint at
   this repo's own dependency set is a one-line follow-up, tracked with the
   ADR-0046 G1 D15/D17 follow-up rather than mixed into the port; a test pins
   the current text so that change cannot pass unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared.python.launch_monitor.schema import METRICS

__all__ = ["PredictiveModelResult", "fit_predictive_model"]


@dataclass(frozen=True)
class PredictiveModelResult:
    """Model recipe, held-out metrics, coefficients, and predictions."""

    model: str
    target: str
    features: tuple[str, ...]
    metrics: dict[str, float]
    coefficients: dict[str, float] | None
    predictions: pd.DataFrame
    random_seed: int
    train_count: int
    test_count: int


def _validate_no_leakage(target: str, features: tuple[str, ...]) -> None:
    if target in features:
        raise ValueError("Target leakage: target cannot also be a feature")
    leaking: list[str] = []
    for feature in features:
        definition = METRICS.get(feature)
        if definition and target in definition.derived_from:
            leaking.append(feature)
    target_definition = METRICS.get(target)
    if target_definition:
        leaking.extend(
            feature for feature in features if feature in target_definition.derived_from
        )
    if leaking:
        raise ValueError(
            "Identity-derived target leakage detected in features: "
            + ", ".join(sorted(set(leaking)))
        )


def _split_indices(
    frame: pd.DataFrame,
    random_seed: int,
    test_fraction: float,
    group_column: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    if group_column:
        if group_column not in frame:
            raise ValueError(f"Group column not present: {group_column}")
        groups = frame[group_column].astype(str).unique()
        if len(groups) < 2:
            raise ValueError("Grouped split requires at least two groups")
        shuffled = rng.permutation(groups)
        n_test_groups = max(1, min(len(groups) - 1, round(len(groups) * test_fraction)))
        test_groups = set(shuffled[:n_test_groups])
        test_mask = frame[group_column].astype(str).isin(test_groups).to_numpy()
        return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
    indices = rng.permutation(len(frame))
    n_test = max(1, min(len(frame) - 1, round(len(frame) * test_fraction)))
    return indices[n_test:], indices[:n_test]


def _coordinate_descent(
    x: np.ndarray,
    y: np.ndarray,
    l1: float,
    l2: float,
    iterations: int = 4000,
    tolerance: float = 1e-9,
) -> np.ndarray:
    coefficients = np.zeros(x.shape[1], dtype=float)
    # ⚡ Bolt: np.einsum avoids intermediate arrays and is ~3x faster than
    # np.sum(x * x, axis=0)
    squared = np.einsum("ij,ij->j", x, x)
    for _ in range(iterations):
        previous = coefficients.copy()
        for column in range(x.shape[1]):
            residual = y - x @ coefficients + x[:, column] * coefficients[column]
            rho = float(x[:, column] @ residual)
            threshold = l1 * len(x)
            if rho < -threshold:
                numerator = rho + threshold
            elif rho > threshold:
                numerator = rho - threshold
            else:
                numerator = 0.0
            coefficients[column] = numerator / (squared[column] + l2 * len(x))
        if np.max(np.abs(coefficients - previous)) < tolerance:
            break
    return coefficients


def _fit_numpy_model(
    model: str, x_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, float]:
    intercept = float(np.mean(y_train))
    centered = y_train - intercept
    if model == "linear":
        coefficients = np.linalg.lstsq(x_train, centered, rcond=None)[0]
    elif model == "ridge":
        alpha = 1.0
        gram = x_train.T @ x_train + alpha * np.eye(x_train.shape[1])
        coefficients = np.linalg.solve(gram, x_train.T @ centered)
    elif model == "lasso":
        coefficients = _coordinate_descent(x_train, centered, l1=0.01, l2=0.0)
    elif model == "elastic_net":
        coefficients = _coordinate_descent(x_train, centered, l1=0.005, l2=0.01)
    else:
        raise ValueError("model must be linear, ridge, lasso, elastic_net, or mlp")
    return coefficients, intercept


def fit_predictive_model(
    frame: pd.DataFrame,
    *,
    target: str,
    features: tuple[str, ...] | list[str],
    model: str = "ridge",
    random_seed: int = 42,
    test_fraction: float = 0.25,
    group_column: str | None = None,
) -> PredictiveModelResult:
    """Fit a standardized model and score a held-out deterministic split."""
    selected = tuple(features)
    if not selected:
        raise ValueError("At least one feature is required")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    _validate_no_leakage(target, selected)
    required = [target, *selected]
    if group_column:
        required.append(group_column)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Columns not present: {sorted(missing)}")
    clean = frame[required].copy()
    clean[target] = pd.to_numeric(clean[target], errors="coerce")
    for feature in selected:
        clean[feature] = pd.to_numeric(clean[feature], errors="coerce")
    clean = clean.dropna(subset=[target, *selected]).reset_index(drop=False)
    if len(clean) < max(12, len(selected) * 4):
        raise ValueError("Insufficient complete rows for predictive modeling")
    train_idx, test_idx = _split_indices(
        clean, random_seed, test_fraction, group_column
    )
    x = clean[list(selected)].to_numpy(float)
    y = clean[target].to_numpy(float)
    mean = x[train_idx].mean(axis=0)
    scale = x[train_idx].std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (x - mean) / scale

    coefficient_map: dict[str, float] | None
    if model == "mlp":
        try:
            from sklearn.neural_network import MLPRegressor
        except ImportError as exc:
            raise ImportError(
                "The shallow MLP requires the analysis extra: "
                "pip install upstream-drift[analysis]"
            ) from exc
        estimator = MLPRegressor(
            hidden_layer_sizes=(max(4, len(selected) * 2),),
            activation="relu",
            solver="lbfgs",
            alpha=0.01,
            max_iter=2000,
            random_state=random_seed,
        )
        target_mean = float(y[train_idx].mean())
        target_scale = float(y[train_idx].std()) or 1.0
        estimator.fit(
            standardized[train_idx], (y[train_idx] - target_mean) / target_scale
        )
        predicted = (
            estimator.predict(standardized[test_idx]) * target_scale + target_mean
        )
        coefficient_map = None
    else:
        coefficients, intercept = _fit_numpy_model(
            model, standardized[train_idx], y[train_idx]
        )
        predicted = intercept + standardized[test_idx] @ coefficients
        coefficient_map = {
            feature: float(value)
            for feature, value in zip(selected, coefficients, strict=True)
        }
    actual = y[test_idx]
    residual = actual - predicted
    # ⚡ Bolt: np.vdot is ~1.7x faster than np.sum(x**2) and avoids temporary
    # array allocations
    ss_res = float(np.vdot(residual, residual))
    centered = actual - actual.mean()
    # ⚡ Bolt: np.vdot is ~1.7x faster than np.sum(x**2) and avoids temporary
    # array allocations
    ss_total = float(np.vdot(centered, centered))
    metrics = {
        "r2": 1.0 - ss_res / ss_total if ss_total > 0 else float("nan"),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }
    predictions = pd.DataFrame(
        {
            "row_index": clean.iloc[test_idx]["index"].to_numpy(),
            "actual": actual,
            "predicted": predicted,
            "residual": residual,
        }
    ).sort_values("row_index", ignore_index=True)
    return PredictiveModelResult(
        model,
        target,
        selected,
        metrics,
        coefficient_map,
        predictions,
        random_seed,
        len(train_idx),
        len(test_idx),
    )

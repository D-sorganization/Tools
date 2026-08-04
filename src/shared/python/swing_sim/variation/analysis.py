"""Dispersion and sensitivity analysis over variation datasets (#4120 V3).

Answers the user's core question — *which output is most affected by which
input's variation* — plus the dispersion summaries that feed the Variation
tab and the plotting suite.

Prior art (surveyed, credited)
------------------------------
- UpstreamDrift ``perturbation/statistics.py`` ``MetricStatistics`` /
  ``compute_metric_statistics``: the mean/std/percentile summary shape,
  reimplemented here per output column (that package aggregates one
  metric dict per trial; we aggregate matrix columns).
- The surveyed UpstreamDrift perturbation packages contain **no**
  sensitivity analysis (their "sensitivity" is dispersion CV only); the
  one-at-a-time matrix and the Spearman rank-correlation check here are
  new, enabled by the subset-stable per-variable RNG streams in
  :mod:`.engine`.

Methods
-------
- :func:`summary_stats` — per-output mean/std/percentiles over runs.
- :func:`one_at_a_time_sensitivity` — re-runs the plan once per noise
  spec with only that spec active (identical draws for it, thanks to the
  per-variable streams) and reports the induced std of every output;
  the ``normalized`` matrix scales each output column by its column max
  so 1.0 marks the dominant input for that output.
- :func:`spearman_matrix` — rank correlation input → output over the
  full dataset: a cheap global-sensitivity cross-check that needs no
  extra simulation runs.
- :func:`dispersion_ellipse` — 2-sigma landing ellipse (carry vs
  lateral) from the sample covariance eigen-decomposition.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from dataclasses import dataclass

import numpy as np

from shared.python.contracts import require

from ..solver.objective import EvaluationConfig
from .engine import VariationDataset, run_variation
from .spec import VariationPlan

_MIN_RUNS_FOR_STATS = 2


@dataclass(frozen=True)
class OutputStats:
    """Dispersion summary for one output column (successful runs only)."""

    name: str
    mean: float
    std: float
    p5: float
    p50: float
    p95: float
    n: int


def summary_stats(dataset: VariationDataset) -> tuple[OutputStats, ...]:
    """Per-output dispersion statistics over the successful runs.

    Returns one :class:`OutputStats` per output column, in dataset order.
    Columns with fewer than two successful samples report ``NaN`` spread.
    """
    stats: list[OutputStats] = []
    for name in dataset.output_names:
        values = dataset.output_column(name)
        n = int(values.size)
        if n == 0:
            stats.append(
                OutputStats(name, math.nan, math.nan, math.nan, math.nan, math.nan, 0)
            )
            continue
        p5, p50, p95 = np.percentile(values, [5.0, 50.0, 95.0])
        std = float(np.std(values, ddof=1)) if n >= _MIN_RUNS_FOR_STATS else math.nan
        stats.append(
            OutputStats(
                name=name,
                mean=float(np.mean(values)),
                std=std,
                p5=float(p5),
                p50=float(p50),
                p95=float(p95),
                n=n,
            )
        )
    return tuple(stats)


@dataclass(frozen=True)
class SensitivityResult:
    """One-at-a-time sensitivity matrix (inputs x outputs).

    ``matrix[i, j]`` is the standard deviation induced in output ``j``
    when only input ``i``'s noise spec is active (same seed, same draws
    for that input as in the full study). ``normalized`` divides each
    output column by its maximum, so 1.0 marks the input that dominates
    that output; all-zero columns stay zero.
    """

    input_keys: tuple[str, ...]
    output_names: tuple[str, ...]
    matrix: np.ndarray
    normalized: np.ndarray

    def dominant_input(self, output_name: str) -> str:
        """The input key whose variation most affects ``output_name``."""
        require(output_name in self.output_names, "unknown output column", output_name)
        column = self.matrix[:, self.output_names.index(output_name)]
        return self.input_keys[int(np.argmax(column))]


def one_at_a_time_sensitivity(
    plan: VariationPlan,
    config: EvaluationConfig | None = None,
    n_workers: int = 4,
    cancel_event: threading.Event | None = None,
) -> SensitivityResult:
    """Vary one noise spec at a time and measure each output's spread.

    Runs ``len(plan.noise)`` sub-studies of ``plan.n_runs`` runs each.
    Because :mod:`.engine` derives one RNG stream per variable key, every
    sub-study draws exactly the values the full study drew for that
    variable — the comparison is paired, not just statistical.
    """
    outputs = None
    rows: list[np.ndarray] = []
    for spec in plan.noise:
        sub_plan = dataclasses.replace(plan, noise=(spec,))
        dataset = run_variation(
            sub_plan, config=config, n_workers=n_workers, cancel_event=cancel_event
        )
        if outputs is None:
            outputs = dataset.output_names
        ok = dataset.outputs[dataset.success]
        if ok.shape[0] >= _MIN_RUNS_FOR_STATS:
            rows.append(np.std(ok, axis=0, ddof=1))
        else:
            rows.append(np.full(len(dataset.output_names), np.nan))
    assert outputs is not None  # plan.noise is non-empty (DbC)
    matrix = np.vstack(rows)
    with np.errstate(invalid="ignore"):
        col_max = np.nanmax(np.abs(matrix), axis=0)
        safe = np.where(col_max > 0.0, col_max, 1.0)
        normalized = np.abs(matrix) / safe
    return SensitivityResult(
        input_keys=tuple(spec.variable_key for spec in plan.noise),
        output_names=outputs,
        matrix=matrix,
        normalized=normalized,
    )


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties averaged), matching Spearman's convention."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    # Average the ranks of exactly-tied values.
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = float(np.mean(ranks[order[i : j + 1]]))
        i = j + 1
    return ranks


def spearman_matrix(dataset: VariationDataset) -> np.ndarray:
    """Spearman rank correlation, inputs (rows) x outputs (columns).

    Computed over the successful runs of the *full* dataset (all noise
    active at once) — a cheap global-sensitivity check that corroborates
    the one-at-a-time matrix without extra simulation. Entries are in
    ``[-1, 1]``; ``NaN`` where a column is constant or there are fewer
    than three successful runs.
    """
    ok = dataset.success
    n = int(np.count_nonzero(ok))
    shape = (len(dataset.input_names), len(dataset.output_names))
    if n < 3:
        return np.full(shape, np.nan)
    inputs = dataset.inputs[ok]
    outputs = dataset.outputs[ok]
    matrix = np.full(shape, np.nan)
    in_ranks = [_ranks(inputs[:, i]) for i in range(shape[0])]
    out_ranks = [_ranks(outputs[:, j]) for j in range(shape[1])]
    for i, ri in enumerate(in_ranks):
        si = float(np.std(ri))
        for j, rj in enumerate(out_ranks):
            sj = float(np.std(rj))
            if si > 0.0 and sj > 0.0:
                cov = float(np.mean((ri - np.mean(ri)) * (rj - np.mean(rj))))
                matrix[i, j] = cov / (si * sj)
    return matrix


@dataclass(frozen=True)
class DispersionEllipse:
    """A 2-sigma landing-dispersion ellipse in the carry/lateral plane.

    Attributes:
        center_carry_m: Mean carry [m].
        center_lateral_m: Mean lateral landing [m, + = right].
        semi_major_m: 2-sigma semi-axis along the principal direction.
        semi_minor_m: 2-sigma semi-axis along the orthogonal direction.
        angle_deg: Principal-axis angle from the carry axis [deg, CCW
            toward + lateral].
        n: Number of samples used.
    """

    center_carry_m: float
    center_lateral_m: float
    semi_major_m: float
    semi_minor_m: float
    angle_deg: float
    n: int


def dispersion_ellipse(
    dataset: VariationDataset, n_sigma: float = 2.0
) -> DispersionEllipse:
    """Fit the n-sigma landing ellipse from ``carry_m`` / ``lateral_m``.

    Eigen-decomposition of the 2x2 sample covariance: semi-axes are
    ``n_sigma * sqrt(eigenvalue)``. Requires at least two successful runs.

    Raises:
        ContractViolationError: If fewer than two successful runs exist
            or the dataset lacks flight outputs.
    """
    require(
        math.isfinite(n_sigma) and n_sigma > 0.0,
        "n_sigma must be finite and > 0",
        n_sigma,
    )
    carry = dataset.output_column("carry_m")
    lateral = dataset.output_column("lateral_m")
    n = int(carry.size)
    require(
        n >= _MIN_RUNS_FOR_STATS,
        "dispersion ellipse needs >= 2 successful runs",
        n,
    )
    points = np.column_stack([carry, lateral])
    cov = np.cov(points, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh sorts ascending; the last eigenpair is the principal axis.
    major = n_sigma * math.sqrt(max(float(eigenvalues[1]), 0.0))
    minor = n_sigma * math.sqrt(max(float(eigenvalues[0]), 0.0))
    principal = eigenvectors[:, 1]
    angle = math.degrees(math.atan2(float(principal[1]), float(principal[0])))
    return DispersionEllipse(
        center_carry_m=float(np.mean(carry)),
        center_lateral_m=float(np.mean(lateral)),
        semi_major_m=major,
        semi_minor_m=minor,
        angle_deg=angle,
        n=n,
    )


__all__ = [
    "DispersionEllipse",
    "OutputStats",
    "SensitivityResult",
    "dispersion_ellipse",
    "one_at_a_time_sensitivity",
    "spearman_matrix",
    "summary_stats",
]

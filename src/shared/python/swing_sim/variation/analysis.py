"""Dispersion and sensitivity analysis over variation datasets (#4120 V3, #4253).

Provides per-output dispersion summaries, one-at-a-time sensitivity matrices,
Spearman rank correlation with permutation significance, bivariate normality
diagnostics for landing ellipses, and truncation mean-shift analysis.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from dataclasses import dataclass
from functools import partial

import numpy as np

from shared.python.contracts import require

from ..solver.objective import EvaluationConfig
from ..solver.solve import ProgressCallback, ProgressReport
from .engine import VariationDataset, run_variation
from .normality import NormalityDiagnostic, convex_hull_2d, mardia_bivariate_normality
from .spec import VariationPlan
from .truncation_analysis import TruncationShiftNote, detect_truncation_mean_shifts

_MIN_RUNS_FOR_STATS = 2


def _emit_offset_progress(
    callback: ProgressCallback,
    iteration_offset: int,
    failure_offset: int,
    started: float,
    report: ProgressReport,
) -> None:
    """Translate one OAT sub-study report onto the complete analysis axis."""
    callback(
        dataclasses.replace(
            report,
            iteration=iteration_offset + report.iteration,
            cost=failure_offset + report.cost,
            elapsed_s=time.monotonic() - started,
        )
    )


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
        percentiles: np.ndarray = np.asarray(
            np.percentile(values, [5.0, 50.0, 95.0]), dtype=float
        )
        std = float(np.std(values, ddof=1)) if n >= _MIN_RUNS_FOR_STATS else math.nan
        stats.append(
            OutputStats(
                name=name,
                mean=float(np.mean(values)),
                std=std,
                p5=float(percentiles[0]),
                p50=float(percentiles[1]),
                p95=float(percentiles[2]),
                n=n,
            )
        )
    return tuple(stats)


def finite_sample_standard_deviation(values: np.ndarray) -> float:
    """Return a stable sample spread for a one-dimensional finite cohort."""
    cohort = np.asarray(values, dtype=float)
    require(cohort.ndim == 1, "standard-deviation cohort must be one-dimensional")
    require(
        bool(np.all(np.isfinite(cohort))), "standard-deviation cohort must be finite"
    )
    if cohort.size < _MIN_RUNS_FOR_STATS:
        return math.nan
    count = 0
    mean = 0.0
    centered_sum = 0.0
    for value in cohort:
        count += 1
        delta = float(value) - mean
        mean += delta / count
        centered_sum += delta * (float(value) - mean)
    return math.sqrt(max(0.0, centered_sum / (count - 1)))


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
        finite = np.isfinite(column)
        require(
            bool(np.any(finite)),
            "output has no available sensitivity values",
            output_name,
        )
        masked = np.where(finite, np.abs(column), -np.inf)
        return self.input_keys[int(np.argmax(masked))]


def one_at_a_time_sensitivity(
    plan: VariationPlan,
    config: EvaluationConfig | None = None,
    n_workers: int = 4,
    cancel_event: threading.Event | None = None,
    progress_cb: ProgressCallback | None = None,
) -> SensitivityResult:
    """Vary one noise spec at a time and measure each output's spread.

    Runs ``len(plan.noise)`` sub-studies of ``plan.n_runs`` runs each.
    Because :mod:`.engine` derives one RNG stream per variable key, every
    sub-study draws exactly the values the full study drew for that
    variable — the comparison is paired, not just statistical.
    """
    outputs = None
    rows: list[np.ndarray] = []
    completed_before = 0
    failed_before = 0
    started = time.monotonic()
    for spec in plan.noise:
        # OAT is an intervention on one marginal at a time. Retaining a
        # multivariate group would reference removed specs and change the
        # method's meaning, so grouped dependence is deliberately absent.
        sub_plan = dataclasses.replace(plan, noise=(spec,), groups=())
        offset = completed_before
        prior_failures = failed_before

        dataset = run_variation(
            sub_plan,
            config=config,
            n_workers=n_workers,
            progress_cb=(
                partial(
                    _emit_offset_progress,
                    progress_cb,
                    offset,
                    prior_failures,
                    started,
                )
                if progress_cb is not None
                else None
            ),
            cancel_event=cancel_event,
        )
        completed_before += plan.n_runs
        failed_before += plan.n_runs - dataset.n_success
        if outputs is None:
            outputs = dataset.output_names
        rows.append(
            np.asarray(
                [
                    (
                        finite_sample_standard_deviation(values)
                        if (values := dataset.output_column(name)).size
                        >= _MIN_RUNS_FOR_STATS
                        else math.nan
                    )
                    for name in dataset.output_names
                ],
                dtype=float,
            )
        )
    assert outputs is not None  # plan.noise is non-empty (DbC)
    return sensitivity_from_standard_deviations(
        tuple(spec.variable_key for spec in plan.noise), outputs, np.vstack(rows)
    )


def sensitivity_from_standard_deviations(
    input_keys: tuple[str, ...],
    output_names: tuple[str, ...],
    matrix: np.ndarray,
) -> SensitivityResult:
    """Build one canonical OAT result from availability-aware sample spreads."""
    inputs = tuple(input_keys)
    outputs = tuple(output_names)
    values = np.array(matrix, dtype=float, copy=True)
    require(bool(inputs) and bool(outputs), "sensitivity axes must be nonempty")
    require(len(set(inputs)) == len(inputs), "sensitivity inputs must be unique")
    require(len(set(outputs)) == len(outputs), "sensitivity outputs must be unique")
    require(
        values.shape == (len(inputs), len(outputs)),
        "sensitivity matrix shape does not match its axes",
        values.shape,
    )
    require(not np.any(np.isinf(values)), "sensitivity matrix cannot contain infinity")
    values.setflags(write=False)
    normalized = _normalize_sensitivity_matrix(values)
    normalized.setflags(write=False)
    return SensitivityResult(inputs, outputs, values, normalized)


def _normalize_sensitivity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize finite cells while preserving per-cell unavailability."""
    normalized: np.ndarray = np.full(matrix.shape, np.nan, dtype=float)
    for output_index in range(matrix.shape[1]):
        column = matrix[:, output_index]
        finite = np.isfinite(column)
        if not np.any(finite):
            continue
        maximum = float(np.max(np.abs(column[finite])))
        normalized[finite, output_index] = (
            np.abs(column[finite]) / maximum if maximum > 0.0 else 0.0
        )
    return normalized


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties averaged), matching Spearman's convention."""
    order = np.argsort(values, kind="mergesort")
    ranks: np.ndarray = np.empty(values.size, dtype=float)
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


@dataclass(frozen=True)
class SpearmanResult:
    """Spearman correlation matrix with permutation p-values and bootstrap CIs."""

    input_keys: tuple[str, ...]
    output_names: tuple[str, ...]
    matrix: np.ndarray
    p_values: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    significant: np.ndarray
    alpha: float = 0.05


def _spearman_rho(r_x: np.ndarray, r_y: np.ndarray) -> float:
    sx = float(np.std(r_x))
    sy = float(np.std(r_y))
    if sx <= 0.0 or sy <= 0.0:
        return math.nan
    cov = float(np.mean((r_x - np.mean(r_x)) * (r_y - np.mean(r_y))))
    return cov / (sx * sy)


def spearman_analysis(
    dataset: VariationDataset,
    n_permutations: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> SpearmanResult:
    """Spearman rank correlation matrix with permutation p-values and CIs."""
    shape = (len(dataset.input_names), len(dataset.output_names))
    matrix = np.full(shape, np.nan, dtype=float)
    p_values = np.full(shape, np.nan, dtype=float)
    ci_lower = np.full(shape, np.nan, dtype=float)
    ci_upper = np.full(shape, np.nan, dtype=float)
    significant = np.zeros(shape, dtype=bool)

    for i in range(shape[0]):
        for j in range(shape[1]):
            available = (
                dataset.success
                & np.isfinite(dataset.inputs[:, i])
                & np.isfinite(dataset.outputs[:, j])
            )
            if np.count_nonzero(available) < 3:
                continue
            rx = _ranks(dataset.inputs[available, i])
            ry = _ranks(dataset.outputs[available, j])
            rho = _spearman_rho(rx, ry)
            if math.isnan(rho):
                continue
            matrix[i, j] = rho

            rng = np.random.default_rng(seed + i * 1000 + j)
            abs_rho = abs(rho)
            exceed = 0
            for _ in range(n_permutations):
                shuffled_ry = rng.permutation(ry)
                if abs(_spearman_rho(rx, shuffled_ry)) >= abs_rho - 1e-12:
                    exceed += 1
            pval = (exceed + 1) / (n_permutations + 1)
            p_values[i, j] = pval

            n_boot = 200
            n_pts = len(rx)
            boot_rhos = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                idx = rng.integers(0, n_pts, size=n_pts)
                boot_rhos[b] = _spearman_rho(rx[idx], ry[idx])
            finite_boots = boot_rhos[np.isfinite(boot_rhos)]
            if finite_boots.size > 0:
                ci_lower[i, j] = float(
                    np.percentile(finite_boots, 100.0 * (alpha / 2.0))
                )
                ci_upper[i, j] = float(
                    np.percentile(finite_boots, 100.0 * (1.0 - alpha / 2.0))
                )

            if pval <= alpha:
                significant[i, j] = True

    return SpearmanResult(
        input_keys=dataset.input_names,
        output_names=dataset.output_names,
        matrix=matrix,
        p_values=p_values,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=significant,
        alpha=alpha,
    )


def spearman_matrix(dataset: VariationDataset) -> np.ndarray:
    """Spearman rank correlation, inputs (rows) x outputs (columns).

    Computed over the successful runs of the *full* dataset (all noise
    active at once) — a cheap global-sensitivity check that corroborates
    the one-at-a-time matrix without extra simulation. Entries are in
    ``[-1, 1]``; ``NaN`` where a paired column is constant or there are
    fewer than three successful, paired finite observations.
    """
    shape = (len(dataset.input_names), len(dataset.output_names))
    matrix = np.full(shape, np.nan)
    for input_index in range(shape[0]):
        for output_index in range(shape[1]):
            available = (
                dataset.success
                & np.isfinite(dataset.inputs[:, input_index])
                & np.isfinite(dataset.outputs[:, output_index])
            )
            if np.count_nonzero(available) < 3:
                continue
            input_ranks = _ranks(dataset.inputs[available, input_index])
            output_ranks = _ranks(dataset.outputs[available, output_index])
            matrix[input_index, output_index] = _spearman_rho(input_ranks, output_ranks)
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
    diagnostic: NormalityDiagnostic | None = None
    convex_hull: np.ndarray | None = None


def dispersion_ellipse(
    dataset: VariationDataset, n_sigma: float = 2.0
) -> DispersionEllipse:
    """Fit n-sigma landing ellipse from carry/lateral with normality check."""
    require(
        math.isfinite(n_sigma) and n_sigma > 0.0,
        "n_sigma must be finite and > 0",
        n_sigma,
    )
    points = dataset.finite_output_rows("carry_m", "lateral_m")
    n = int(points.shape[0])
    require(
        n >= _MIN_RUNS_FOR_STATS,
        "dispersion ellipse needs >= 2 paired finite landing rows",
        n,
    )
    carry = points[:, 0]
    lateral = points[:, 1]
    cov = np.cov(points, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major = n_sigma * math.sqrt(max(float(eigenvalues[1]), 0.0))
    minor = n_sigma * math.sqrt(max(float(eigenvalues[0]), 0.0))
    principal = eigenvectors[:, 1]
    angle = math.degrees(math.atan2(float(principal[1]), float(principal[0])))
    # Bivariate normality test (carry vs lateral) and convex hull fallback
    landing_2d = np.column_stack([lateral, carry])
    diagnostic = mardia_bivariate_normality(landing_2d)
    convex_hull = convex_hull_2d(landing_2d) if not diagnostic.is_normal else None

    return DispersionEllipse(
        center_carry_m=float(np.mean(carry)),
        center_lateral_m=float(np.mean(lateral)),
        semi_major_m=major,
        semi_minor_m=minor,
        angle_deg=angle,
        n=n,
        diagnostic=diagnostic,
        convex_hull=convex_hull,
    )


__all__ = [
    "DispersionEllipse",
    "NormalityDiagnostic",
    "OutputStats",
    "SensitivityResult",
    "SpearmanResult",
    "TruncationShiftNote",
    "convex_hull_2d",
    "detect_truncation_mean_shifts",
    "dispersion_ellipse",
    "finite_sample_standard_deviation",
    "mardia_bivariate_normality",
    "one_at_a_time_sensitivity",
    "sensitivity_from_standard_deviations",
    "spearman_analysis",
    "spearman_matrix",
    "summary_stats",
]

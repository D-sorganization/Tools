"""Seeded, parallel N-run variation executor (epic #4120, V3).

Runs a :class:`~shared.python.swing_sim.variation.spec.VariationPlan`:
samples every noisy variable per run, pushes each sampled variable set
through its pipeline slice (:mod:`.pipeline`), and collects a
:class:`VariationDataset` (inputs matrix, outputs matrix, success flags).

Prior art (surveyed, credited)
------------------------------
- UpstreamDrift ``physics/ball_enhanced_simulator.monte_carlo_simulation``:
  the N-run seeded-loop shape. Its serial loop and ``base_seed + i``
  seeding are replaced with per-variable ``numpy`` seed sequences (subset-
  stable streams, required by the one-at-a-time sensitivity analysis) and
  a ``concurrent.futures`` chunked thread pool.
- UpstreamDrift ``movement_optimizer/trajectory/optimizer_parallel.py`` /
  ``result.py`` — the dispatch/drain/cancel structure and the
  ``ProgressReport`` / ``CancelledError`` shapes, reused *directly* from
  their port in :mod:`shared.python.swing_sim.solver.solve` so GUI
  progress/cancel plumbing is identical to the solver's.
- UpstreamDrift ``perturbation/analyzer_base.py`` — per-trial failure
  capture: a failed run is recorded (success flag ``False``, ``NaN``
  outputs) instead of aborting the batch.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from shared.python.contracts import ContractViolationError, require

from ..solver.objective import EvaluationConfig
from ..solver.solve import CancelledError, ProgressCallback, ProgressReport
from .pipeline import (
    DELIVERY_OUTPUTS,
    FLIGHT_OUTPUTS,
    IMPACT_INTERVAL_OUTPUTS,
    LAUNCH_OUTPUTS,
    evaluate_run,
    outputs_for_mode,
)
from .spec import NoiseSpec, VariationPlan

logger = logging.getLogger(__name__)

_PROGRESS_EVERY = 8
_RUN_ERRORS = (
    ValueError,
    ContractViolationError,
    RuntimeError,
    FloatingPointError,
    OverflowError,
)


@dataclass(frozen=True)
class VariationDataset:
    """The collected result of one variation study.

    Attributes:
        plan: The executed plan (embeds seed, base values, noise specs).
        input_names: Registry keys of the varied variables, one per
            column of ``inputs`` (order matches ``plan.noise``).
        inputs: ``(n_runs, n_inputs)`` sampled variable values.
        output_names: Output-column names (:func:`outputs_for_mode`).
        outputs: ``(n_runs, n_outputs)`` pipeline outputs; failed runs
            hold ``NaN``.
        success: ``(n_runs,)`` boolean per-run success flags.
        elapsed_s: Wall-clock seconds spent in :func:`run_variation`.
    """

    plan: VariationPlan
    input_names: tuple[str, ...]
    inputs: np.ndarray = field(repr=False)
    output_names: tuple[str, ...]
    outputs: np.ndarray = field(repr=False)
    success: np.ndarray = field(repr=False)
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        inputs = np.asarray(self.inputs, dtype=float)
        outputs = np.asarray(self.outputs, dtype=float)
        success = np.asarray(self.success, dtype=bool)
        n = self.plan.n_runs
        require(
            inputs.shape == (n, len(self.input_names)),
            "inputs must be (n_runs, n_inputs)",
            inputs.shape,
        )
        require(
            outputs.shape == (n, len(self.output_names)),
            "outputs must be (n_runs, n_outputs)",
            outputs.shape,
        )
        require(success.shape == (n,), "success must be (n_runs,)", success.shape)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "success", success)

    @property
    def n_success(self) -> int:
        """Number of successful runs."""
        return int(np.count_nonzero(self.success))

    def output_column(self, name: str) -> np.ndarray:
        """Successful-run values of one output column."""
        require(name in self.output_names, "unknown output column", name)
        return cast(
            np.ndarray,
            np.asarray(self.outputs[self.success, self.output_names.index(name)]),
        )


def _stream_for(seed: int, spec: NoiseSpec) -> np.random.Generator:
    """Independent, subset-stable RNG stream for one noise spec.

    Keyed by ``[seed, crc32(variable_key)]`` so removing *other* specs
    from a plan (one-at-a-time sensitivity) leaves this spec's draws
    unchanged — unlike the ``base_seed + i`` idiom in the surveyed
    UpstreamDrift Monte-Carlo code, which correlates streams.
    """
    return np.random.default_rng([seed, zlib.crc32(spec.variable_key.encode())])


def sample_inputs(plan: VariationPlan) -> np.ndarray:
    """Sample the ``(n_runs, n_specs)`` inputs matrix for a plan.

    Vectorized per spec; truncation clips into ``[lower, upper]`` (see
    :class:`NoiseSpec`). Deterministic for a given ``(plan, seed)``.
    """
    base = plan.resolved_base()
    columns: list[np.ndarray] = []
    for spec in plan.noise:
        rng = _stream_for(plan.seed, spec)
        center = base[spec.variable_key]
        if spec.distribution == "normal":
            values = rng.normal(center, spec.scale, plan.n_runs)
        elif spec.distribution == "uniform":
            values = rng.uniform(center - spec.scale, center + spec.scale, plan.n_runs)
        else:  # triangular (validated by NoiseSpec)
            values = rng.triangular(
                center - spec.scale, center, center + spec.scale, plan.n_runs
            )
        lo = -np.inf if spec.lower is None else spec.lower
        hi = np.inf if spec.upper is None else spec.upper
        columns.append(np.clip(values, lo, hi))
    return cast(np.ndarray, np.column_stack(columns))


class _Progress:
    """Thread-safe run counter emitting solver-shaped ProgressReports.

    ``iteration`` counts completed runs and ``cost`` carries the failed-run
    count (the only "cost" a Monte-Carlo batch has); the remaining fields
    keep the ``movement_optimizer`` / ``solver.solve`` shape so existing
    progress UIs plug in unchanged.
    """

    def __init__(self, cb: ProgressCallback | None, total: int) -> None:
        self._cb = cb
        self._total = total
        self._lock = threading.Lock()
        self._done = 0
        self._failed = 0
        self._start = time.monotonic()

    def record(self, ok: bool) -> None:
        with self._lock:
            self._done += 1
            if not ok:
                self._failed += 1
            emit = self._cb is not None and (
                self._done % _PROGRESS_EVERY == 0 or self._done == self._total
            )
            if emit:
                assert self._cb is not None
                self._cb(
                    ProgressReport(
                        iteration=self._done,
                        cost=float(self._failed),
                        best_cost=0.0,
                        improvement_pct=0.0,
                        elapsed_s=time.monotonic() - self._start,
                    )
                )


def _run_chunk(
    indices: range,
    plan: VariationPlan,
    inputs: np.ndarray,
    outputs: np.ndarray,
    success: np.ndarray,
    config: EvaluationConfig,
    names: tuple[str, ...],
    progress: _Progress,
    cancel_event: threading.Event,
) -> None:
    """Worker body: evaluate a contiguous block of runs in place."""
    base = plan.resolved_base()
    input_keys = tuple(spec.variable_key for spec in plan.noise)
    for i in indices:
        if cancel_event.is_set():
            return
        variables = dict(base)
        variables.update(zip(input_keys, inputs[i].tolist(), strict=True))
        try:
            result = evaluate_run(variables, plan.mode, config)
            outputs[i] = [result[name] for name in names]
            success[i] = True
        except _RUN_ERRORS as exc:
            logger.debug("variation run %d failed: %s", i, exc)
            success[i] = False
        progress.record(bool(success[i]))


def run_variation(
    plan: VariationPlan,
    config: EvaluationConfig | None = None,
    n_workers: int = 4,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> VariationDataset:
    """Execute a variation plan and collect the dataset.

    Args:
        plan: The study to run (seeded — same plan, same dataset).
        config: Optional evaluation knobs; ``flight_model`` is always
            taken from the plan so serialized studies replay identically.
        n_workers: Thread-pool size (>= 1); results are identical for
            any worker count (sampling is precomputed, rows are disjoint).
        progress_cb: Optional solver-shaped progress callback.
        cancel_event: Optional cooperative cancellation event; when set,
            the executor stops and :class:`CancelledError` is raised.

    Returns:
        :class:`VariationDataset` with one row per run.

    Raises:
        CancelledError: If ``cancel_event`` is (or becomes) set.
    """
    require(n_workers >= 1, "n_workers must be >= 1", n_workers)
    event = cancel_event or threading.Event()
    if event.is_set():
        raise CancelledError("Variation run cancelled before start")

    t0 = time.monotonic()
    base_cfg = config or EvaluationConfig()
    cfg = EvaluationConfig(
        flight_model=plan.flight_model,
        flight_max_time_s=base_cfg.flight_max_time_s,
        flight_dt_s=base_cfg.flight_dt_s,
        swing_duration_s=base_cfg.swing_duration_s,
        swing_dt_s=base_cfg.swing_dt_s,
    )
    names = outputs_for_mode(plan.mode)
    inputs = sample_inputs(plan)
    outputs = np.full((plan.n_runs, len(names)), np.nan)
    success: np.ndarray = np.zeros(plan.n_runs, dtype=bool)
    progress = _Progress(progress_cb, plan.n_runs)

    workers = min(n_workers, plan.n_runs)
    chunk = math.ceil(plan.n_runs / workers)
    ranges = [
        range(start, min(start + chunk, plan.n_runs))
        for start in range(0, plan.n_runs, chunk)
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _run_chunk,
                block,
                plan,
                inputs,
                outputs,
                success,
                cfg,
                names,
                progress,
                event,
            )
            for block in ranges
        ]
        for future in futures:
            future.result()

    if event.is_set():
        raise CancelledError("Variation run cancelled")
    return VariationDataset(
        plan=plan,
        input_names=tuple(spec.variable_key for spec in plan.noise),
        inputs=inputs,
        output_names=names,
        outputs=outputs,
        success=success,
        elapsed_s=time.monotonic() - t0,
    )


__all__ = [
    "DELIVERY_OUTPUTS",
    "FLIGHT_OUTPUTS",
    "LAUNCH_OUTPUTS",
    "IMPACT_INTERVAL_OUTPUTS",
    "CancelledError",
    "ProgressReport",
    "VariationDataset",
    "evaluate_run",
    "outputs_for_mode",
    "run_variation",
    "sample_inputs",
]

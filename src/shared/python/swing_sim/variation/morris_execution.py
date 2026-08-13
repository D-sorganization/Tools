"""Deterministic UI-neutral execution adapter for Morris designs."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np

from shared.python.contracts import require

from ..solver.solve import CancelledError, ProgressCallback, ProgressReport
from ._morris_vocabulary import (
    EVALUATED_NO_IMPACT_VALUE,
    NUMERICAL_FAILURE_VALUE,
    OUTCOMES,
)
from .morris_design import MorrisDesign, MorrisFactor, MorrisObservations, MorrisOutput

MAX_MORRIS_SAMPLES, MAX_MORRIS_OBSERVATION_CELLS = 100_000, 1_000_000
MAX_MORRIS_WORKERS = 32
MORRIS_PROGRESS_INTERVAL = 8


@dataclass(frozen=True)
class MorrisSample:
    """Canonically identified physical point supplied to an evaluator."""

    ordinal: int
    trajectory_index: int
    point_index: int
    factors: tuple[MorrisFactor, ...]
    physical_values: Mapping[str, float]

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        _require_nonnegative_integer(self.ordinal, "ordinal")
        _require_nonnegative_integer(self.trajectory_index, "trajectory_index")
        _require_nonnegative_integer(self.point_index, "point_index")
        require(bool(factors), "factors must not be empty")
        require(
            all(isinstance(factor, MorrisFactor) for factor in factors),
            "factors must contain only MorrisFactor values",
        )
        expected = tuple(factor.spec_id for factor in factors)
        values = _normalize_finite_mapping(self.physical_values, "physical_values")
        require(
            set(values) == set(expected) and len(values) == len(expected),
            "physical_values must contain the exact factor spec_id set",
            tuple(values),
        )
        ordered = {spec_id: values[spec_id] for spec_id in expected}
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "physical_values", MappingProxyType(ordered))


@dataclass(frozen=True)
class MorrisEvaluation:
    """Canonical status and output availability returned by an evaluator."""

    status: str
    values: Mapping[str, float | None]
    failure_type: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        status = str(getattr(self.status, "value", self.status))
        require(status in OUTCOMES, f"status must be one of {OUTCOMES}", status)
        require(isinstance(self.values, Mapping), "values must be a mapping")
        normalized = {
            _require_output_name(name): _normalize_optional_value(value)
            for name, value in self.values.items()
        }
        require(
            len(normalized) == len(self.values),
            "values must contain unique output names",
        )
        if status == NUMERICAL_FAILURE_VALUE:
            require(
                all(value is None for value in normalized.values()),
                "numerical failure outputs must all be unavailable",
            )
        diagnostics = (self.failure_type, self.failure_message)
        require(
            status == NUMERICAL_FAILURE_VALUE or diagnostics == (None, None),
            "failure diagnostics are permitted only for numerical failures",
            diagnostics,
        )
        require(
            (self.failure_type is None) == (self.failure_message is None),
            "failure type and message must be provided together",
            diagnostics,
        )
        for value, name in zip(
            diagnostics, ("failure_type", "failure_message"), strict=True
        ):
            require(
                value is None
                or (
                    isinstance(value, str)
                    and value == value.strip()
                    and bool(value)
                    and len(value) <= 1_024
                    and all(
                        ord(character) >= 32 and not 127 <= ord(character) <= 159
                        for character in value
                    )
                ),
                f"{name} must be a bounded nonempty trimmed string or None",
                value,
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "values", MappingProxyType(normalized))


@dataclass(frozen=True)
class MorrisExecutionOptions:
    """Bounded worker, progress, and cancellation controls for one execution."""

    n_workers: int = 1
    progress_cb: ProgressCallback | None = None
    cancel_event: threading.Event | None = None

    def __post_init__(self) -> None:
        workers = _require_worker_count(self.n_workers)
        require(
            self.progress_cb is None or callable(self.progress_cb),
            "progress_cb must be callable when provided",
            self.progress_cb,
        )
        valid_event = self.cancel_event is None or isinstance(
            self.cancel_event, threading.Event
        )
        require(
            valid_event, "cancel_event must be a threading.Event", self.cancel_event
        )
        object.__setattr__(self, "n_workers", workers)


class MorrisEvaluator(Protocol):
    """Callable contract injected into the Morris executor."""

    def __call__(self, sample: MorrisSample) -> MorrisEvaluation:
        """Evaluate one immutable physical Morris sample."""
        ...


@dataclass(frozen=True)
class _ExecutionContext:
    design: MorrisDesign
    physical_points: np.ndarray
    outputs: tuple[MorrisOutput, ...]
    evaluator: MorrisEvaluator
    values: np.ndarray
    outcomes: np.ndarray
    failure_types: np.ndarray
    failure_messages: np.ndarray
    cancel_event: threading.Event


@dataclass
class _ProgressState:
    callback: ProgressCallback | None
    started_at: float
    failures: int = 0
    history: list[float] = field(default_factory=list)

    def record(self, ordinal: int, status: str, sample_count: int) -> None:
        self.failures += int(status == NUMERICAL_FAILURE_VALUE)
        self.history.append(float(self.failures))
        completed = ordinal + 1
        if self.callback is None or (
            completed % MORRIS_PROGRESS_INTERVAL != 0 and completed != sample_count
        ):
            return
        self.callback(
            ProgressReport(
                iteration=completed,
                cost=float(self.failures),
                best_cost=0.0,
                improvement_pct=0.0,
                elapsed_s=time.monotonic() - self.started_at,
                cost_history=list(self.history[-200:]),
            )
        )


def _require_nonnegative_integer(value: object, name: str) -> int:
    require(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer)),
        f"{name} must be a nonnegative integer",
        value,
    )
    result = int(cast(int | np.integer, value))
    require(result >= 0, f"{name} must be a nonnegative integer", result)
    return result


def _require_worker_count(value: object) -> int:
    result = _require_nonnegative_integer(value, "n_workers")
    require(
        1 <= result <= MAX_MORRIS_WORKERS,
        f"n_workers must be within [1, {MAX_MORRIS_WORKERS}]",
        result,
    )
    return result


def _require_output_name(value: object) -> str:
    require(
        isinstance(value, str) and bool(value) and value == value.strip(),
        "output names must be non-empty trimmed strings",
        value,
    )
    return cast(str, value)


def _normalize_optional_value(value: object) -> float | None:
    if value is None:
        return None
    require(
        not isinstance(value, (bool, np.bool_)) and isinstance(value, Real),
        "evaluation values must be finite or None",
        value,
    )
    normalized = float(cast(Real, value))
    require(
        math.isfinite(normalized),
        "evaluation values must be finite or None",
        normalized,
    )
    return normalized


def _normalize_finite_mapping(
    value: Mapping[str, float], name: str
) -> dict[str, float]:
    require(isinstance(value, Mapping), f"{name} must be a mapping", value)
    normalized: dict[str, float] = {}
    for key, item in value.items():
        stable_key = _require_output_name(key)
        finite = _normalize_optional_value(item)
        require(finite is not None, f"{name} values must be finite", item)
        normalized[stable_key] = cast(float, finite)
    return normalized


def _validate_inputs(
    design: object, outputs: object, evaluator: object, options: object
) -> tuple[
    MorrisDesign, tuple[MorrisOutput, ...], MorrisEvaluator, MorrisExecutionOptions
]:
    require(isinstance(design, MorrisDesign), "design must be a MorrisDesign")
    require(isinstance(outputs, tuple) and bool(outputs), "outputs must be a tuple")
    typed_design = cast(MorrisDesign, design)
    tuple_outputs = cast(tuple[object, ...], outputs)
    require(
        all(isinstance(output, MorrisOutput) for output in tuple_outputs),
        "outputs must contain only MorrisOutput values",
    )
    typed_outputs = cast(tuple[MorrisOutput, ...], tuple_outputs)
    names = tuple(output.name for output in typed_outputs)
    require(len(set(names)) == len(names), "output names must be unique", names)
    require(callable(evaluator), "evaluator must be callable", evaluator)
    require(
        isinstance(options, MorrisExecutionOptions),
        "options must be a MorrisExecutionOptions",
        options,
    )
    typed_options = cast(MorrisExecutionOptions, options)
    sample_count = typed_design.trajectories * (len(typed_design.factors) + 1)
    sample_message = (
        f"sample count must not exceed MAX_MORRIS_SAMPLES={MAX_MORRIS_SAMPLES}"
    )
    require(sample_count <= MAX_MORRIS_SAMPLES, sample_message, sample_count)
    cell_count = sample_count * len(typed_outputs)
    cell_message = (
        "allocation must not exceed "
        f"MAX_MORRIS_OBSERVATION_CELLS={MAX_MORRIS_OBSERVATION_CELLS}"
    )
    require(cell_count <= MAX_MORRIS_OBSERVATION_CELLS, cell_message, cell_count)
    return typed_design, typed_outputs, cast(MorrisEvaluator, evaluator), typed_options


def _make_sample(context: _ExecutionContext, ordinal: int) -> MorrisSample:
    points_per_trajectory = len(context.design.factors) + 1
    trajectory_index, point_index = divmod(ordinal, points_per_trajectory)
    point = context.physical_points[trajectory_index, point_index]
    physical = {
        factor.spec_id: float(point[index])
        for index, factor in enumerate(context.design.factors)
    }
    return MorrisSample(
        ordinal, trajectory_index, point_index, context.design.factors, physical
    )


def _validate_evaluation(
    evaluation: object, outputs: tuple[MorrisOutput, ...]
) -> MorrisEvaluation:
    require(
        isinstance(evaluation, MorrisEvaluation),
        "evaluator must return a MorrisEvaluation",
        type(evaluation),
    )
    typed_evaluation = cast(MorrisEvaluation, evaluation)
    expected = tuple(output.name for output in outputs)
    exact_names = set(typed_evaluation.values) == set(expected) and len(
        typed_evaluation.values
    ) == len(expected)
    require(
        exact_names,
        "evaluation values must contain the exact output-name set",
        tuple(typed_evaluation.values),
    )
    if typed_evaluation.status == EVALUATED_NO_IMPACT_VALUE:
        downstream = tuple(
            output.name
            for output in outputs
            if output.target_kind in ("impact", "shot-outcome")
        )
        require(
            all(typed_evaluation.values[name] is None for name in downstream),
            "no-impact evaluation cannot provide impact or shot outputs",
            downstream,
        )
    return typed_evaluation


def _evaluate_one(context: _ExecutionContext, ordinal: int) -> str:
    if context.cancel_event.is_set():
        raise CancelledError("Morris execution cancelled")
    sample = _make_sample(context, ordinal)
    evaluation = context.evaluator(sample)
    validated = _validate_evaluation(evaluation, context.outputs)
    point = (sample.trajectory_index, sample.point_index)
    context.outcomes[point] = validated.status
    context.failure_types[point] = validated.failure_type
    context.failure_messages[point] = validated.failure_message
    context.values[point] = [
        (
            np.nan
            if validated.values[output.name] is None
            else validated.values[output.name]
        )
        for output in context.outputs
    ]
    return validated.status


def _execute_bounded(
    context: _ExecutionContext,
    sample_count: int,
    workers: int,
    callback: ProgressCallback | None,
) -> None:
    progress = _ProgressState(callback, time.monotonic())
    with ThreadPoolExecutor(max_workers=min(workers, sample_count)) as pool:
        futures: list[Future[str]] = [
            pool.submit(_evaluate_one, context, ordinal)
            for ordinal in range(min(workers, sample_count))
        ]
        for ordinal in range(sample_count):
            if context.cancel_event.is_set():
                raise CancelledError("Morris execution cancelled")
            status = futures[ordinal].result()
            progress.record(ordinal, status, sample_count)
            if context.cancel_event.is_set():
                raise CancelledError("Morris execution cancelled")
            next_ordinal = ordinal + workers
            if next_ordinal < sample_count:
                futures.append(pool.submit(_evaluate_one, context, next_ordinal))


def evaluate_morris_design(
    design: MorrisDesign,
    outputs: tuple[MorrisOutput, ...],
    evaluator: MorrisEvaluator,
    options: MorrisExecutionOptions | None = None,
) -> MorrisObservations:
    """Evaluate a bounded design in canonical order across any worker count.

    Evaluators normalize expected domain failures into typed evaluations;
    cancellation and every evaluator/contract exception abort the call.
    """
    execution_options = MorrisExecutionOptions() if options is None else options
    typed_design, typed_outputs, typed_evaluator, execution_options = _validate_inputs(
        design, outputs, evaluator, execution_options
    )
    event = execution_options.cancel_event or threading.Event()
    if event.is_set():
        raise CancelledError("Morris execution cancelled before start")
    shape = (typed_design.trajectories, len(typed_design.factors) + 1)
    values = np.full(shape + (len(typed_outputs),), np.nan, dtype=float)
    outcomes: np.ndarray = np.empty(shape, dtype=object)
    failure_types: np.ndarray = np.full(shape, None, dtype=object)
    failure_messages: np.ndarray = np.full(shape, None, dtype=object)
    physical_points = typed_design.physical_points
    physical_points.setflags(write=False)
    context = _ExecutionContext(
        typed_design,
        physical_points,
        typed_outputs,
        typed_evaluator,
        values,
        outcomes,
        failure_types,
        failure_messages,
        event,
    )
    sample_count = shape[0] * shape[1]
    _execute_bounded(
        context,
        sample_count,
        execution_options.n_workers,
        execution_options.progress_cb,
    )
    return MorrisObservations(
        typed_design,
        typed_outputs,
        values,
        outcomes,
        failure_types,
        failure_messages,
    )


__all__ = [
    "MAX_MORRIS_OBSERVATION_CELLS",
    "MAX_MORRIS_SAMPLES",
    "MAX_MORRIS_WORKERS",
    "MORRIS_PROGRESS_INTERVAL",
    "MorrisEvaluation",
    "MorrisEvaluator",
    "MorrisExecutionOptions",
    "MorrisSample",
    "evaluate_morris_design",
]

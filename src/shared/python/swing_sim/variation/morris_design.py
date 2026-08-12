"""Validated deterministic design and observation contracts for Morris screening."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from shared.python.contracts import require

from ._morris_vocabulary import (
    EVALUATED_NO_IMPACT_VALUE,
    NUMERICAL_FAILURE_VALUE,
    OUTCOMES,
    OUTPUT_KINDS,
    normalize_outcomes,
)
from .ensemble_types import immutable_array
from .spec import NoiseSpec, variable_registry

CONSTANT_TOLERANCE = 1e-14


def _require_stable_id(value: str, name: str) -> None:
    """Require a non-empty, whitespace-stable identifier."""
    require(
        isinstance(value, str) and bool(value) and value == value.strip(),
        f"{name} must be a non-empty, trimmed stable ID",
        value,
    )


def _require_integer(value: object, name: str, minimum: int) -> int:
    """Require a true integral value, excluding booleans and floats."""
    require(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer)),
        f"{name} must be an integer >= {minimum}",
        value,
    )
    integer = int(cast(int | np.integer[Any], value))
    require(
        integer >= minimum,
        f"{name} must be an integer >= {minimum}",
        integer,
    )
    return integer


@dataclass(frozen=True)
class MorrisFactor:
    """One bounded source on the normalized Morris grid with optional locus."""

    spec_id: str
    variable_key: str
    lower: float
    upper: float
    unit: str
    source_time_window_s: tuple[float, float] | None = None
    source_point_ids: tuple[str, ...] = ()

    @classmethod
    def from_noise_spec(
        cls, spec: NoiseSpec, lower: float, upper: float
    ) -> MorrisFactor:
        """Create a factor without duplicating a plan source's locus metadata."""
        require(isinstance(spec, NoiseSpec), "spec must be a NoiseSpec", spec)
        assert spec.spec_id is not None
        return cls(
            spec_id=spec.spec_id,
            variable_key=spec.variable_key,
            lower=lower,
            upper=upper,
            unit=variable_registry()[spec.variable_key].unit,
            source_time_window_s=spec.time_window_s,
            source_point_ids=spec.point_ids,
        )

    def __post_init__(self) -> None:
        _require_stable_id(self.spec_id, "spec_id")
        _require_stable_id(self.variable_key, "variable_key")
        _require_stable_id(self.unit, "factor unit")
        registry = variable_registry()
        require(
            self.variable_key in registry,
            "factor variable_key must be registered",
            self.variable_key,
        )
        require(
            self.unit == registry[self.variable_key].unit,
            "factor unit must match the registered variable unit",
            (self.unit, registry[self.variable_key].unit),
        )
        lower = float(self.lower)
        upper = float(self.upper)
        require(
            math.isfinite(lower) and math.isfinite(upper) and lower < upper,
            "factor bounds must be finite and satisfy lower < upper",
            (lower, upper),
        )
        window = self.source_time_window_s
        if window is not None:
            normalized_window = tuple(float(value) for value in window)
            require(
                len(normalized_window) == 2
                and all(math.isfinite(value) for value in normalized_window)
                and normalized_window[0] < normalized_window[1],
                "source_time_window_s must contain finite start < end",
                normalized_window,
            )
            object.__setattr__(self, "source_time_window_s", normalized_window)
        points = tuple(self.source_point_ids)
        for point_id in points:
            _require_stable_id(point_id, "source_point_id")
        require(len(set(points)) == len(points), "source_point_ids must be unique")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "source_point_ids", points)


@dataclass(frozen=True)
class MorrisOutput:
    """One downstream scalar metric with optional point/time attribution."""

    name: str
    unit: str
    target_kind: str = "scalar"
    target_time_s: float | None = None
    target_point_id: str | None = None
    coordinate_frame: str | None = None

    def __post_init__(self) -> None:
        _require_stable_id(self.name, "output name")
        _require_stable_id(self.unit, "output unit")
        require(
            self.target_kind in OUTPUT_KINDS,
            f"target_kind must be one of {OUTPUT_KINDS}",
            self.target_kind,
        )
        require(
            self.target_time_s is None or math.isfinite(self.target_time_s),
            "target_time_s must be finite when provided",
            self.target_time_s,
        )
        if self.target_point_id is not None:
            _require_stable_id(self.target_point_id, "target_point_id")
        if self.coordinate_frame is not None:
            _require_stable_id(self.coordinate_frame, "coordinate_frame")
        if self.target_kind == "state-point":
            require(
                self.target_point_id is not None and self.coordinate_frame is not None,
                "state-point output requires target_point_id and coordinate_frame",
            )


@dataclass(frozen=True)
class MorrisDesign:
    """Immutable normalized Morris trajectories and step provenance."""

    factors: tuple[MorrisFactor, ...]
    trajectories: int
    levels: int
    seed: int
    normalized_points: np.ndarray = field(repr=False)
    changed_factor_indices: np.ndarray = field(repr=False)
    signed_steps: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        require(bool(factors), "Morris design needs at least one factor")
        require(
            len({factor.spec_id for factor in factors}) == len(factors),
            "Morris factor spec_id values must be unique",
        )
        trajectories = _require_integer(self.trajectories, "trajectories", 1)
        levels = _require_integer(self.levels, "levels", 4)
        require(levels % 2 == 0, "levels must be even and >= 4", levels)
        seed = _require_integer(self.seed, "seed", 0)
        arrays = _design_arrays(self)
        _validate_design_shapes(arrays, trajectories, len(factors))
        _validate_design_paths(arrays, trajectories, len(factors))
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "trajectories", trajectories)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(
            self, "normalized_points", immutable_array(arrays.points, float)
        )
        object.__setattr__(
            self, "changed_factor_indices", immutable_array(arrays.changed, int)
        )
        object.__setattr__(self, "signed_steps", immutable_array(arrays.steps, float))

    @property
    def physical_points(self) -> np.ndarray:
        """Return the design mapped into each factor's physical bounds."""
        lower = np.asarray([factor.lower for factor in self.factors], dtype=float)
        span = np.asarray(
            [factor.upper - factor.lower for factor in self.factors], dtype=float
        )
        result: np.ndarray = lower + self.normalized_points * span
        return result


@dataclass(frozen=True)
class _DesignArrays:
    """Normalized constructor arrays used by Morris design checks."""

    points: np.ndarray
    changed: np.ndarray
    steps: np.ndarray


def _design_arrays(design: MorrisDesign) -> _DesignArrays:
    """Normalize a design's array-like constructor values."""
    changed_raw = np.asarray(design.changed_factor_indices)
    require(
        np.issubdtype(changed_raw.dtype, np.integer)
        and not np.issubdtype(changed_raw.dtype, np.bool_),
        "changed_factor_indices must contain integers",
        changed_raw.dtype,
    )
    return _DesignArrays(
        points=np.asarray(design.normalized_points, dtype=float),
        changed=np.asarray(changed_raw, dtype=int),
        steps=np.asarray(design.signed_steps, dtype=float),
    )


def _validate_design_shapes(
    arrays: _DesignArrays, trajectories: int, factor_count: int
) -> None:
    """Require the canonical trajectory and step array shapes."""
    require(
        arrays.points.shape == (trajectories, factor_count + 1, factor_count),
        "normalized_points has invalid Morris trajectory shape",
        arrays.points.shape,
    )
    require(
        arrays.changed.shape == arrays.steps.shape == (trajectories, factor_count),
        "changed-factor and signed-step arrays have invalid shape",
        (arrays.changed.shape, arrays.steps.shape),
    )


def _validate_design_paths(
    arrays: _DesignArrays, trajectories: int, factor_count: int
) -> None:
    """Require bounded one-factor-at-a-time trajectories and exact steps."""
    require(
        np.all(np.isfinite(arrays.points))
        and np.all((arrays.points >= 0.0) & (arrays.points <= 1.0)),
        "normalized Morris points must be finite and within [0, 1]",
    )
    require(
        np.all(np.sort(arrays.changed, axis=1) == np.arange(factor_count)),
        "each trajectory must change every factor exactly once",
    )
    differences = np.diff(arrays.points, axis=1)
    nonzero = np.count_nonzero(np.abs(differences) > CONSTANT_TOLERANCE, axis=2)
    require(np.all(nonzero == 1), "each Morris step must change one factor")
    rows = np.arange(trajectories)[:, np.newaxis]
    step_numbers = np.arange(factor_count)[np.newaxis, :]
    actual = differences[rows, step_numbers, arrays.changed]
    require(
        np.allclose(actual, arrays.steps, rtol=0.0, atol=CONSTANT_TOLERANCE),
        "signed_steps must match the changed normalized coordinates",
    )


def generate_morris_design(
    factors: tuple[MorrisFactor, ...],
    trajectories: int,
    levels: int = 4,
    seed: int = 0,
) -> MorrisDesign:
    """Generate deterministic randomized Morris trajectories.

    Each trajectory uses the classical step, changes every factor once, and
    remains inside the closed normalized domain in randomized order.
    """
    levels = _require_integer(levels, "levels", 4)
    require(levels % 2 == 0, "levels must be even and >= 4", levels)
    trajectories = _require_integer(trajectories, "trajectories", 1)
    seed = _require_integer(seed, "seed", 0)
    factor_count = len(factors)
    require(factor_count >= 1, "Morris design needs at least one factor")
    rng = np.random.default_rng(seed)
    step_units = levels // 2
    delta = step_units / (levels - 1)
    points: npt.NDArray[np.float64] = np.empty(
        (trajectories, factor_count + 1, factor_count), dtype=float
    )
    changed: npt.NDArray[np.int_] = np.empty((trajectories, factor_count), dtype=int)
    signed_steps: npt.NDArray[np.float64] = np.empty(
        (trajectories, factor_count), dtype=float
    )
    for trajectory in range(trajectories):
        signs = rng.choice(np.array((-1, 1), dtype=int), size=factor_count)
        low_starts = rng.integers(0, levels - step_units, size=factor_count)
        high_starts = rng.integers(step_units, levels, size=factor_count)
        current = np.where(signs > 0, low_starts, high_starts) / (levels - 1)
        order = rng.permutation(factor_count)
        points[trajectory, 0] = current
        changed[trajectory] = order
        for step, factor_index in enumerate(order):
            signed_step = float(signs[factor_index]) * delta
            current = current.copy()
            current[factor_index] += signed_step
            points[trajectory, step + 1] = current
            signed_steps[trajectory, step] = signed_step
    return MorrisDesign(
        factors=tuple(factors),
        trajectories=trajectories,
        levels=levels,
        seed=seed,
        normalized_points=points,
        changed_factor_indices=changed,
        signed_steps=signed_steps,
    )


@dataclass(frozen=True)
class MorrisObservations:
    """Evaluated design with one typed outcome retained for every sample."""

    design: MorrisDesign
    outputs: tuple[MorrisOutput, ...]
    values: np.ndarray = field(repr=False)
    outcomes: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        outputs = tuple(self.outputs)
        require(bool(outputs), "Morris observations need at least one output")
        require(
            len({output.name for output in outputs}) == len(outputs),
            "Morris output names must be unique",
        )
        values = np.asarray(self.values, dtype=float)
        outcomes = normalize_outcomes(self.outcomes)
        expected_prefix = (self.design.trajectories, len(self.design.factors) + 1)
        require(
            values.shape == expected_prefix + (len(outputs),),
            "values has invalid Morris observation shape",
            values.shape,
        )
        require(
            outcomes.shape == expected_prefix,
            "outcomes has invalid Morris observation shape",
            outcomes.shape,
        )
        require(
            np.all(np.isin(outcomes, OUTCOMES)),
            f"outcomes must be one of {OUTCOMES}",
            np.unique(outcomes),
        )
        failure = outcomes == NUMERICAL_FAILURE_VALUE
        require(
            np.all(np.isnan(values[failure])),
            "numerical-failure samples must not contain outputs",
        )
        _validate_noimpact_availability(outputs, values, outcomes)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "values", immutable_array(values, float))
        object.__setattr__(self, "outcomes", immutable_array(outcomes, str))


def _validate_noimpact_availability(
    outputs: tuple[MorrisOutput, ...], values: np.ndarray, outcomes: np.ndarray
) -> None:
    """Keep miss-state metrics while forbidding fabricated impact/shot values."""
    no_impact = outcomes == EVALUATED_NO_IMPACT_VALUE
    downstream = np.asarray(
        [output.target_kind in ("impact", "shot-outcome") for output in outputs]
    )
    if np.any(no_impact) and np.any(downstream):
        require(
            np.all(np.isnan(values[no_impact][:, downstream])),
            "no-impact samples must not contain impact or shot outputs",
        )


__all__ = [
    "CONSTANT_TOLERANCE",
    "MorrisDesign",
    "MorrisFactor",
    "MorrisObservations",
    "MorrisOutput",
    "generate_morris_design",
]

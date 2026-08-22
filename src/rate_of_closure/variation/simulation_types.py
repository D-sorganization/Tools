"""Immutable Rate simulation-ensemble requests and scalar outcomes."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import cast

import numpy as np

from rate_of_closure.simulation import SimulationConfig
from shared.python.contracts import require
from shared.python.swing_sim.variation.engine import VariationDataset
from shared.python.swing_sim.variation.ensemble_types import EnsemblePositionTraces
from shared.python.swing_sim.variation.spec import VariationPlan

from ._ensemble_limits import require_ensemble_shape_limits
from .ensemble_source import EnsembleWorkChunk

APP_FRAME_ID = "app_frame:x_target,y_up,z_right"

CONTACT_OUTPUT_NAMES: tuple[str, ...] = (
    "candidate_time_s",
    "closest_approach_m",
    "contact_margin_m",
)
IMPACT_OUTPUT_NAMES: tuple[str, ...] = (
    "impact_time_s",
    "clubhead_speed_mps",
    "spin_loft_deg",
    "face_to_path_deg",
    "spin_axis_tilt_deg",
)
SHOT_OUTPUT_NAMES: tuple[str, ...] = (
    "ball_speed_mph",
    "launch_angle_deg",
    "launch_azimuth_deg",
    "spin_rpm",
    "carry_m",
    "lateral_m",
    "max_height_m",
    "flight_time_s",
    "landing_angle_deg",
)
ALL_OUTPUT_NAMES = CONTACT_OUTPUT_NAMES + IMPACT_OUTPUT_NAMES + SHOT_OUTPUT_NAMES


class TrialEvaluationStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Mutually exclusive outcome of attempting one configured trial."""

    EVALUATED_HIT = "evaluated_hit"
    EVALUATED_NO_IMPACT = "evaluated_no_impact"
    NUMERICAL_FAILURE = "numerical_failure"


EVALUATED_HIT = TrialEvaluationStatus.EVALUATED_HIT
EVALUATED_NO_IMPACT = TrialEvaluationStatus.EVALUATED_NO_IMPACT
NUMERICAL_FAILURE = TrialEvaluationStatus.NUMERICAL_FAILURE


@dataclass(frozen=True)
class SimulationTrialOutcome:
    """Scalar result and explicit status for one ensemble trial.

    Missing impact/shot quantities are ``None`` rather than fabricated zeroes.
    Contact quantities remain available for an evaluated no-impact trial.
    """

    trial_index: int
    status: TrialEvaluationStatus
    values: Mapping[str, float | None]
    failure_type: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        require(self.trial_index >= 0, "trial_index must be >= 0", self.trial_index)
        require(
            isinstance(self.status, TrialEvaluationStatus),
            "status must be a TrialEvaluationStatus",
            self.status,
        )
        require(
            set(self.values) == set(ALL_OUTPUT_NAMES),
            "values must contain the canonical scalar output set",
            tuple(self.values),
        )
        normalized = _normalize_scalar_values(self.values)
        _validate_outcome_availability(self.status, normalized)
        _validate_failure_metadata(self)
        object.__setattr__(self, "values", MappingProxyType(normalized))

    def value(self, name: str) -> float | None:
        """Return one canonical scalar value by stable output name."""
        require(name in self.values, "unknown scalar output", name)
        return self.values[name]


@dataclass(frozen=True)
class SimulationEnsembleRequest:
    """Complete per-trial simulation configs aligned to sampled inputs."""

    plan: VariationPlan
    sampled_inputs: np.ndarray = field(repr=False)
    configs: tuple[SimulationConfig, ...]

    def __post_init__(self) -> None:
        require(isinstance(self.plan, VariationPlan), "plan must be a VariationPlan")
        configs = tuple(self.configs)
        require(
            len(configs) == self.plan.n_runs,
            "configs must contain one SimulationConfig per plan run",
            len(configs),
        )
        require(
            all(isinstance(config, SimulationConfig) for config in configs),
            "configs must contain only SimulationConfig values",
        )
        _require_common_trace_contract(configs)
        samples = np.array(self.sampled_inputs, dtype=float, copy=True)
        expected = (self.plan.n_runs, len(self.plan.noise))
        require(
            samples.shape == expected,
            "sampled_inputs has invalid shape",
            samples.shape,
        )
        require(bool(np.all(np.isfinite(samples))), "sampled_inputs must be finite")
        samples.setflags(write=False)
        object.__setattr__(self, "configs", configs)
        object.__setattr__(self, "sampled_inputs", samples)

    def reference_config(self) -> SimulationConfig:
        """Return the first complete configuration."""
        return self.configs[0]

    def work_chunks(
        self, *, chunk_size: int, start_index: int = 0
    ) -> Iterator[EnsembleWorkChunk]:
        """Expose the materialized compatibility request as bounded chunks."""
        require(
            type(chunk_size) is int and chunk_size > 0,
            "chunk_size must be a positive integer",
        )
        require(
            type(start_index) is int and 0 <= start_index <= self.plan.n_runs,
            "start_index must lie within the plan",
        )
        return self._iter_work_chunks(chunk_size, start_index)

    def _iter_work_chunks(
        self, chunk_size: int, start_index: int
    ) -> Iterator[EnsembleWorkChunk]:
        for start in range(start_index, self.plan.n_runs, chunk_size):
            stop = min(start + chunk_size, self.plan.n_runs)
            yield EnsembleWorkChunk(
                start, self.sampled_inputs[start:stop], self.configs[start:stop]
            )


@dataclass(frozen=True)
class SimulationEnsembleResult:
    """Scalar trial outcomes and common-grid traces from one request."""

    outcomes: tuple[SimulationTrialOutcome, ...]
    variation: VariationDataset
    traces: EnsemblePositionTraces

    def __post_init__(self) -> None:
        outcomes = tuple(self.outcomes)
        trial_count = len(self.variation.success)
        require(len(outcomes) == trial_count, "outcomes must align to trials")
        require(
            tuple(outcome.trial_index for outcome in outcomes)
            == tuple(range(trial_count)),
            "outcomes must be in canonical trial order",
        )
        require(
            self.variation.output_names == ALL_OUTPUT_NAMES,
            "variation output_names must be canonical",
        )
        expected_inputs = tuple(spec.variable_key for spec in self.variation.plan.noise)
        require(
            self.variation.input_names == expected_inputs,
            "input_names must match plan provenance",
        )
        require(
            bool(np.all(np.isfinite(self.variation.inputs))),
            "ensemble sampled inputs must be finite",
        )
        require(
            bool(
                np.all(
                    np.isfinite(self.variation.outputs)
                    | np.isnan(self.variation.outputs)
                )
            ),
            "ensemble outputs must be finite or unavailable NaN",
        )
        elapsed_s = float(self.variation.elapsed_s)
        require(
            math.isfinite(elapsed_s) and elapsed_s >= 0.0,
            "elapsed_s must be finite and non-negative",
            self.variation.elapsed_s,
        )
        require(
            all(
                bool(self.variation.success[index])
                == (outcome.status is not NUMERICAL_FAILURE)
                for index, outcome in enumerate(outcomes)
            ),
            "outcome statuses must agree with variation success",
        )
        require(
            self.traces.variation is self.variation,
            "traces and result must share one VariationDataset",
        )
        require_ensemble_shape_limits(
            trial_count,
            int(self.traces.sample_times_s.size),
            len(self.traces.point_ids),
        )
        _require_outcome_scalar_binding(outcomes, self.variation)
        _require_trace_status_binding(outcomes, self.traces)
        object.__setattr__(self, "outcomes", outcomes)

    @property
    def impact_output_names(self) -> tuple[str, ...]:
        """Canonical impact-only scalar names."""
        return IMPACT_OUTPUT_NAMES

    @property
    def shot_output_names(self) -> tuple[str, ...]:
        """Canonical downstream launch/flight scalar names."""
        return SHOT_OUTPUT_NAMES


def _require_common_trace_contract(configs: tuple[SimulationConfig, ...]) -> None:
    """Require configs whose simulations naturally share one trace grid."""
    if not configs:
        return
    first = configs[0]
    require(
        all(config.source_kind == first.source_kind for config in configs),
        "all configs must use the same source_kind",
    )
    require(
        all(config.swing_duration_s == first.swing_duration_s for config in configs),
        "all configs must use the same swing_duration_s",
    )


def _normalize_scalar_values(values: Mapping[str, object]) -> dict[str, float | None]:
    """Return finite built-in floats so typed outcomes are always wire-safe."""
    normalized: dict[str, float | None] = {}
    for name in ALL_OUTPUT_NAMES:
        value = values[name]
        if value is None:
            normalized[name] = None
            continue
        require(
            isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
            "available scalar outputs must be real numbers excluding booleans",
            (name, value),
        )
        numeric = float(cast(float, value))
        require(
            math.isfinite(numeric),
            "available scalar outputs must be finite",
            (name, numeric),
        )
        normalized[name] = numeric
    return normalized


def _validate_outcome_availability(
    status: TrialEvaluationStatus, values: Mapping[str, float | None]
) -> None:
    """Enforce hit/miss/failure scalar-availability invariants."""
    if status is NUMERICAL_FAILURE:
        require(
            all(value is None for value in values.values()),
            "failure values must be null",
        )
        return
    require(
        all(values[name] is not None for name in CONTACT_OUTPUT_NAMES),
        "evaluated trials require contact outputs",
    )
    downstream = IMPACT_OUTPUT_NAMES + SHOT_OUTPUT_NAMES
    expected_available = status is EVALUATED_HIT
    require(
        all((values[name] is not None) == expected_available for name in downstream),
        "impact and shot outputs must agree with impact status",
    )


def _validate_failure_metadata(outcome: SimulationTrialOutcome) -> None:
    """Keep failure diagnostics present only for numerical failures."""
    if outcome.status is NUMERICAL_FAILURE:
        require(bool(outcome.failure_type), "numerical failure requires failure_type")
        require(outcome.failure_message is not None, "failure_message must be present")
        return
    require(
        outcome.failure_type is None and outcome.failure_message is None,
        "evaluated trials cannot carry failure metadata",
    )


def _require_outcome_scalar_binding(
    outcomes: tuple[SimulationTrialOutcome, ...], variation: VariationDataset
) -> None:
    """Bind every typed outcome to the canonical scalar matrix row."""
    for outcome in outcomes:
        expected = np.array(
            [
                math.nan if outcome.value(name) is None else outcome.value(name)
                for name in ALL_OUTPUT_NAMES
            ],
            dtype=float,
        )
        require(
            bool(
                np.array_equal(
                    variation.outputs[outcome.trial_index], expected, equal_nan=True
                )
            ),
            "outcome values must match variation outputs",
        )


def _require_trace_status_binding(
    outcomes: tuple[SimulationTrialOutcome, ...], traces: EnsemblePositionTraces
) -> None:
    """Bind typed trial status to trace availability and impact provenance."""
    for outcome in outcomes:
        index = outcome.trial_index
        if outcome.status is NUMERICAL_FAILURE:
            require(
                not np.any(traces.sample_valid[index]),
                "numerical failure trace must be unavailable",
            )
            require(
                traces.impact_sample_indices[index] == -1,
                "numerical failure impact marker must be -1",
            )
            continue
        require(
            np.any(traces.sample_valid[index]),
            "evaluated trial trace must retain an available sample",
        )
        expected_impact = outcome.status is EVALUATED_HIT
        impact_index = int(traces.impact_sample_indices[index])
        require(
            (impact_index >= 0) == expected_impact,
            "impact marker must match typed trial status",
        )
        if expected_impact:
            impact_time = outcome.value("impact_time_s")
            assert impact_time is not None
            nearest = int(np.argmin(np.abs(traces.sample_times_s - impact_time)))
            require(
                impact_index == nearest,
                "impact marker must match impact-time provenance",
            )


__all__ = [
    "ALL_OUTPUT_NAMES",
    "APP_FRAME_ID",
    "CONTACT_OUTPUT_NAMES",
    "EVALUATED_HIT",
    "EVALUATED_NO_IMPACT",
    "IMPACT_OUTPUT_NAMES",
    "NUMERICAL_FAILURE",
    "SHOT_OUTPUT_NAMES",
    "SimulationEnsembleRequest",
    "SimulationEnsembleResult",
    "SimulationTrialOutcome",
    "TrialEvaluationStatus",
]

"""Immutable Rate simulation-ensemble requests and scalar outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import numpy as np

from rate_of_closure.simulation import SimulationConfig
from shared.python.contracts import require
from shared.python.swing_sim.variation.engine import VariationDataset
from shared.python.swing_sim.variation.ensemble_types import EnsemblePositionTraces
from shared.python.swing_sim.variation.spec import VariationPlan

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


class TrialEvaluationStatus(str, Enum):
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
        normalized = {name: self.values.get(name) for name in ALL_OUTPUT_NAMES}
        require(
            set(self.values) == set(ALL_OUTPUT_NAMES),
            "values must contain the canonical scalar output set",
            tuple(self.values),
        )
        _validate_scalar_values(normalized)
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


@dataclass(frozen=True)
class SimulationEnsembleResult:
    """Scalar trial outcomes and common-grid traces from one request."""

    outcomes: tuple[SimulationTrialOutcome, ...]
    variation: VariationDataset
    traces: EnsemblePositionTraces

    def __post_init__(self) -> None:
        trial_count = len(self.variation.success)
        require(len(self.outcomes) == trial_count, "outcomes must align to trials")
        require(
            self.traces.variation is self.variation,
            "traces and result must share one VariationDataset",
        )

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


def _validate_scalar_values(values: Mapping[str, float | None]) -> None:
    """Require each available scalar to be finite."""
    require(
        all(value is None or math.isfinite(value) for value in values.values()),
        "available scalar outputs must be finite",
    )


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

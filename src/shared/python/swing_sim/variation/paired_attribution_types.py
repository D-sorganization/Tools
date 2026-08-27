"""Immutable input contracts for paired model-intervention attribution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from shared.python.contracts import require

from .ensemble_types import immutable_array, require_coordinate_frame_id
from .identity_contracts import stable_id
from .noise_response_types import require_sha256

PAIRED_ATTRIBUTION_SCHEMA_ID = "swing-sim/paired-localized-attribution"
PAIRED_ATTRIBUTION_SCHEMA_VERSION = 1
PAIRED_INTERVENTION_METHOD_ID = "paired-model-intervention-difference/v1"
INTERPRETATION_BOUNDARY = (
    "Model-scenario paired intervention response only: not rank association, "
    "not a global main effect, not anatomical or human causal attribution, "
    "and not coaching authority."
)

TRIAL_EVALUATED_HIT = "evaluated_hit"
TRIAL_EVALUATED_NO_IMPACT = "evaluated_no_impact"
TRIAL_NUMERICAL_FAILURE = "numerical_failure"
TRIAL_STATUSES = (
    TRIAL_EVALUATED_HIT,
    TRIAL_EVALUATED_NO_IMPACT,
    TRIAL_NUMERICAL_FAILURE,
)

AVAILABILITY_AVAILABLE = "available"
AVAILABILITY_NO_IMPACT = "no-impact-unavailable"
AVAILABILITY_NUMERICAL_FAILURE = "numerical-failure"
AVAILABILITY_MISSING = "missing-unavailable"
AVAILABILITY_NONFINITE = "nonfinite-unavailable"
AVAILABILITY_UNSUPPORTED = "unsupported"
VALUE_STATES = (
    AVAILABILITY_AVAILABLE,
    AVAILABILITY_MISSING,
    AVAILABILITY_NONFINITE,
    AVAILABILITY_UNSUPPORTED,
)
PAIR_AVAILABILITY_STATES = (
    AVAILABILITY_AVAILABLE,
    AVAILABILITY_NO_IMPACT,
    AVAILABILITY_NUMERICAL_FAILURE,
    AVAILABILITY_MISSING,
    AVAILABILITY_NONFINITE,
    AVAILABILITY_UNSUPPORTED,
)

TARGET_KINDS = ("state", "impact", "shot")
MAX_PAIRS = 4096
MAX_TARGETS = 256
MAX_OBSERVATIONS = 131_072
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class AttributionRunContext:
    """Identity shared by the two sides of every governed pair."""

    model_id: str
    adapter_id: str
    coordinate_frame: str
    trace_grid_sha256: str
    plan_sha256: str
    registry_sha256: str
    execution_sha256: str
    source_adapter_id: str | None = None

    def __post_init__(self) -> None:
        stable_id(self.model_id, "model_id")
        stable_id(self.adapter_id, "adapter_id")
        if self.source_adapter_id is not None:
            stable_id(self.source_adapter_id, "source_adapter_id")
        require_coordinate_frame_id(self.coordinate_frame)
        for name in (
            "trace_grid_sha256",
            "plan_sha256",
            "registry_sha256",
            "execution_sha256",
        ):
            require_sha256(getattr(self, name), name)


@dataclass(frozen=True)
class AttributionSource:
    """One source parameter and its optional spatial/temporal locus."""

    source_id: str
    variable_key: str
    unit: str
    point_id: str | None = None
    time_window_s: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        stable_id(self.source_id, "source_id")
        stable_id(self.variable_key, "variable_key")
        stable_id(self.unit, "source unit")
        if self.point_id is not None:
            stable_id(self.point_id, "source point_id")
        if self.time_window_s is not None:
            require(len(self.time_window_s) == 2, "time window must contain two values")
            start, end = (float(value) for value in self.time_window_s)
            require(
                math.isfinite(start) and math.isfinite(end) and 0.0 <= start < end,
                "time window must satisfy finite 0 <= start < end",
            )
            object.__setattr__(self, "time_window_s", (start, end))


@dataclass(frozen=True)
class AttributionTarget:
    """One scalar state, impact, or shot response coordinate."""

    target_id: str
    kind: str
    unit: str
    metric_id: str | None = None
    coordinate_frame: str | None = None
    point_id: str | None = None
    coordinate_value: float | None = None
    coordinate_unit: str | None = None

    def __post_init__(self) -> None:
        stable_id(self.target_id, "target_id")
        require(self.kind in TARGET_KINDS, "unsupported target kind", self.kind)
        stable_id(self.unit, "target unit")
        metric_id = self.target_id if self.metric_id is None else self.metric_id
        object.__setattr__(self, "metric_id", stable_id(metric_id, "metric_id"))
        if self.kind == "state":
            require(self.coordinate_frame is not None, "state target requires frame")
            require(self.point_id is not None, "state target requires point_id")
            require(
                self.coordinate_value is not None, "state target requires coordinate"
            )
            require(
                self.coordinate_unit is not None,
                "state target requires coordinate unit",
            )
            require_coordinate_frame_id(cast(str, self.coordinate_frame))
            stable_id(cast(str, self.point_id), "target point_id")
            stable_id(cast(str, self.coordinate_unit), "target coordinate unit")
            value = float(cast(float, self.coordinate_value))
            require(math.isfinite(value), "target coordinate must be finite")
            object.__setattr__(self, "coordinate_value", value)
            return
        require(
            self.coordinate_frame is None
            and self.point_id is None
            and self.coordinate_value is None
            and self.coordinate_unit is None,
            "impact and shot targets cannot carry a state locus",
        )


def _validate_values(
    values: np.ndarray, states: tuple[str, ...], label: str
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    require(array.ndim == 1 and array.size == len(states), f"{label} shape mismatch")
    require(all(state in VALUE_STATES for state in states), f"invalid {label} state")
    available = np.asarray([state == AVAILABILITY_AVAILABLE for state in states])
    require(
        bool(np.all(np.isfinite(array[available]))), f"available {label} must be finite"
    )
    require(
        bool(np.all(np.isnan(array[~available]))), f"unavailable {label} must be NaN"
    )
    return immutable_array(array, float)


@dataclass(frozen=True)
class AttributionPair:
    """One exact baseline/perturbed trial pair and its scalar observations."""

    pair_id: str
    baseline_trial_id: str
    perturbed_trial_id: str
    baseline_status: str
    perturbed_status: str
    baseline_source_value: float
    perturbed_source_value: float
    baseline_values: np.ndarray = field(repr=False)
    perturbed_values: np.ndarray = field(repr=False)
    baseline_value_states: tuple[str, ...]
    perturbed_value_states: tuple[str, ...]

    def __post_init__(self) -> None:
        stable_id(self.pair_id, "pair_id")
        baseline_id = stable_id(self.baseline_trial_id, "baseline_trial_id")
        perturbed_id = stable_id(self.perturbed_trial_id, "perturbed_trial_id")
        require(baseline_id != perturbed_id, "pair trial IDs must differ")
        require(self.baseline_status in TRIAL_STATUSES, "invalid baseline status")
        require(self.perturbed_status in TRIAL_STATUSES, "invalid perturbed status")
        baseline_source = float(self.baseline_source_value)
        perturbed_source = float(self.perturbed_source_value)
        require(
            math.isfinite(baseline_source) and math.isfinite(perturbed_source),
            "source values must be finite",
        )
        require(baseline_source != perturbed_source, "source delta must be nonzero")
        baseline_states = tuple(self.baseline_value_states)
        perturbed_states = tuple(self.perturbed_value_states)
        require(
            len(baseline_states) == len(perturbed_states), "value-state shape mismatch"
        )
        object.__setattr__(self, "baseline_source_value", baseline_source)
        object.__setattr__(self, "perturbed_source_value", perturbed_source)
        object.__setattr__(self, "baseline_value_states", baseline_states)
        object.__setattr__(self, "perturbed_value_states", perturbed_states)
        object.__setattr__(
            self,
            "baseline_values",
            _validate_values(self.baseline_values, baseline_states, "baseline values"),
        )
        object.__setattr__(
            self,
            "perturbed_values",
            _validate_values(
                self.perturbed_values, perturbed_states, "perturbed values"
            ),
        )


@dataclass(frozen=True)
class PairedAttributionContract:
    """Pair-independent attribution contract used for bounded accumulation."""

    source: AttributionSource
    targets: tuple[AttributionTarget, ...]
    context: AttributionRunContext
    source_sha256: str

    def __post_init__(self) -> None:
        require(isinstance(self.source, AttributionSource), "invalid source")
        require(isinstance(self.context, AttributionRunContext), "invalid context")
        targets = tuple(self.targets)
        require(0 < len(targets) <= MAX_TARGETS, "target count exceeds resource cap")
        require(
            all(isinstance(target, AttributionTarget) for target in targets),
            "invalid target",
        )
        require(
            len({target.target_id for target in targets}) == len(targets),
            "target IDs must be unique",
        )
        require(
            all(
                target.coordinate_frame in (None, self.context.coordinate_frame)
                for target in targets
            ),
            "state target frame mismatch",
        )
        require_sha256(self.source_sha256, "source_sha256")
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True)
class PairedAttributionInput:
    """Validated paired observations under one exact run identity."""

    source: AttributionSource
    targets: tuple[AttributionTarget, ...]
    pairs: tuple[AttributionPair, ...]
    baseline_context: AttributionRunContext
    perturbed_context: AttributionRunContext
    source_sha256: str

    def __post_init__(self) -> None:
        require(
            self.baseline_context == self.perturbed_context,
            "baseline and perturbed context mismatch",
        )
        contract = self.contract_without_pairs()
        pairs = tuple(self.pairs)
        require(len(pairs) <= MAX_PAIRS, "pair count exceeds resource cap")
        require(
            len(pairs) * len(contract.targets) <= MAX_OBSERVATIONS,
            "pair-target matrix exceeds resource cap",
        )
        require(
            all(isinstance(pair, AttributionPair) for pair in pairs), "invalid pair"
        )
        require(
            len({pair.pair_id for pair in pairs}) == len(pairs),
            "pair IDs must be unique",
        )
        require(
            all(pair.baseline_values.size == len(contract.targets) for pair in pairs),
            "pair target count mismatch",
        )
        object.__setattr__(self, "targets", contract.targets)
        object.__setattr__(self, "pairs", pairs)

    def contract_without_pairs(self) -> PairedAttributionContract:
        """Return the pair-independent identity for streaming accumulation."""
        return PairedAttributionContract(
            self.source,
            tuple(self.targets),
            self.baseline_context,
            self.source_sha256,
        )

    def with_pairs(self, pairs: tuple[AttributionPair, ...]) -> PairedAttributionInput:
        """Return a bounded chunk under the identical contract."""
        return PairedAttributionInput(
            self.source,
            tuple(self.targets),
            tuple(pairs),
            self.baseline_context,
            self.perturbed_context,
            self.source_sha256,
        )


__all__ = [
    name
    for name in globals()
    if name.startswith(("AVAILABILITY_", "TRIAL_", "PAIRED_", "INTERPRETATION_"))
] + [
    "AttributionPair",
    "AttributionRunContext",
    "AttributionSource",
    "AttributionTarget",
    "MAX_ARCHIVE_BYTES",
    "MAX_OBSERVATIONS",
    "MAX_PAIRS",
    "MAX_TARGETS",
    "PairedAttributionContract",
    "PairedAttributionInput",
    "PAIR_AVAILABILITY_STATES",
    "TARGET_KINDS",
    "VALUE_STATES",
]

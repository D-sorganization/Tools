"""Input and resumable-state contracts for geometric response fields."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from shared.python.contracts import require

from .ensemble_types import immutable_array
from .execution_metadata import (
    VariationExecutionMetadata,
    validate_execution_metadata,
)
from .identity_contracts import stable_id
from .registry import variable_registry
from .spec import NoiseSpec
from .trace_resampling import TRACE_RESAMPLING_POLICY_ID, TraceResamplingResult

POSITION_NOISE_RESPONSE_FIELD_SCHEMA_ID = "swing-sim/position-noise-response-field"
POSITION_NOISE_RESPONSE_FIELD_SCHEMA_VERSION = 1
PAIRED_OAT_RESPONSE_METHOD = "paired-oat-linear-through-origin/v1"
DECLARED_SCALE_NORMALIZATION = "declared-distribution-standard-deviation/v1"

ADEQUACY_ESTIMABLE = "estimable"
ADEQUACY_INSUFFICIENT_PAIRS = "insufficient-pairs"
ADEQUACY_ZERO_PERTURBATION = "zero-perturbation"
ADEQUACY_UNSUPPORTED_BOUNDED = "unsupported-bounded-input"
ADEQUACY_UNSUPPORTED_CORRELATED = "unsupported-correlated-input"
ADEQUACY_UNSUPPORTED_DISCRETE = "unsupported-discrete-input"
ADEQUACY_STATES = (
    ADEQUACY_ESTIMABLE,
    ADEQUACY_INSUFFICIENT_PAIRS,
    ADEQUACY_ZERO_PERTURBATION,
    ADEQUACY_UNSUPPORTED_BOUNDED,
    ADEQUACY_UNSUPPORTED_CORRELATED,
    ADEQUACY_UNSUPPORTED_DISCRETE,
)

METRIC_IDS = (
    "signed-cartesian-response",
    "response-magnitude",
    "matched-absolute-rms-scatter",
    "all-eligible-absolute-rms-scatter",
)
METRIC_UNITS = ("m/(declared-standard-deviation)",) * 2 + ("m", "m")
SCIENTIFIC_BOUNDARY = (
    "Model-scenario geometry only: normalized response is not causal anatomy, "
    "human validation, joint work, energy transfer, or coaching authority."
)
_INPUT_KINDS = ("continuous", "discrete")
_SHA256_HEX = frozenset("0123456789abcdef")
_DISTRIBUTION_SD_DIVISOR = {
    "normal": 1.0,
    "uniform": math.sqrt(3.0),
    "triangular": math.sqrt(6.0),
}


def require_sha256(value: str, name: str) -> str:
    """Return one lowercase SHA-256 digest after fail-closed validation."""
    require(
        isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX,
        f"{name} must be a lowercase SHA-256 digest",
        value,
    )
    return value


def _stable_tuple(value: tuple[str, ...], name: str) -> tuple[str, ...]:
    require(type(value) is tuple and bool(value), f"{name} must be a nonempty tuple")
    result = tuple(stable_id(item, name) for item in value)
    require(len(set(result)) == len(result), f"{name} must be unique")
    return result


def _spec_for_input(field_input: ResponseFieldInput) -> NoiseSpec:
    specs = field_input.baseline.traces.variation.plan.noise
    matches = tuple(spec for spec in specs if spec.spec_id == field_input.spec_id)
    require(len(matches) == 1, "spec_id must identify one plan noise spec")
    return matches[0]


def _validate_trace_layout(field_input: ResponseFieldInput) -> None:
    baseline = field_input.baseline.traces
    perturbed = field_input.perturbed.traces
    require(
        field_input.baseline.policy_id == TRACE_RESAMPLING_POLICY_ID, "policy drift"
    )
    require(
        field_input.perturbed.policy_id == TRACE_RESAMPLING_POLICY_ID, "policy drift"
    )
    require(baseline.coordinate_frame == perturbed.coordinate_frame, "frame mismatch")
    require(baseline.point_ids == perturbed.point_ids, "point layout mismatch")
    require(
        np.array_equal(baseline.sample_times_s, perturbed.sample_times_s),
        "time-grid mismatch",
    )
    require(baseline.n_trials == perturbed.n_trials, "paired trial count mismatch")


def _validate_trial_ids(field_input: ResponseFieldInput) -> None:
    trial_ids = _stable_tuple(field_input.trial_ids, "trial_ids")
    baseline_ids = _stable_tuple(field_input.baseline_trial_ids, "baseline_trial_ids")
    perturbed_ids = _stable_tuple(
        field_input.perturbed_trial_ids, "perturbed_trial_ids"
    )
    require(trial_ids == baseline_ids, "baseline trial order mismatch")
    require(trial_ids == perturbed_ids, "perturbed trial order mismatch")
    require(
        len(trial_ids) == field_input.baseline.traces.n_trials,
        "trial_ids must match trace rows",
    )
    object.__setattr__(field_input, "trial_ids", trial_ids)
    object.__setattr__(field_input, "baseline_trial_ids", baseline_ids)
    object.__setattr__(field_input, "perturbed_trial_ids", perturbed_ids)


@dataclass(frozen=True)
class ResponseFieldInput:
    """One governed paired intervention for a declared plan input."""

    spec_id: str
    adapter_id: str
    source_layout_id: str
    trial_ids: tuple[str, ...]
    baseline_trial_ids: tuple[str, ...]
    perturbed_trial_ids: tuple[str, ...]
    baseline: TraceResamplingResult
    perturbed: TraceResamplingResult
    execution_metadata: VariationExecutionMetadata
    source_sha256: str
    input_kind: str = "continuous"

    def __post_init__(self) -> None:
        stable_id(self.spec_id, "spec_id")
        stable_id(self.adapter_id, "adapter_id")
        stable_id(self.source_layout_id, "source_layout_id")
        require(self.input_kind in _INPUT_KINDS, "unsupported input_kind")
        require(isinstance(self.baseline, TraceResamplingResult), "invalid baseline")
        require(isinstance(self.perturbed, TraceResamplingResult), "invalid perturbed")
        require_sha256(self.source_sha256, "source_sha256")
        _validate_trace_layout(self)
        _validate_trial_ids(self)
        baseline_plan = self.baseline.traces.variation.plan
        perturbed_plan = self.perturbed.traces.variation.plan
        validate_execution_metadata(baseline_plan, self.execution_metadata)
        validate_execution_metadata(perturbed_plan, self.execution_metadata)
        spec = _spec_for_input(self)
        require(
            self.baseline.traces.variation.input_names
            == self.perturbed.traces.variation.input_names,
            "input registry order mismatch",
        )
        require(
            spec.variable_key in self.input_names, "spec variable missing from traces"
        )
        if self.support_status == ADEQUACY_ESTIMABLE:
            self._require_oat_pairing()

    @property
    def spec(self) -> NoiseSpec:
        """Return the exact plan-bound perturbation specification."""
        return _spec_for_input(self)

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the exact ordered variation-input registry keys."""
        return self.baseline.traces.variation.input_names

    @property
    def input_delta(self) -> np.ndarray:
        """Return the paired target-input difference in its registry unit."""
        column = self.input_names.index(self.spec.variable_key)
        baseline = self.baseline.traces.variation.inputs[:, column]
        perturbed = self.perturbed.traces.variation.inputs[:, column]
        return np.asarray(perturbed - baseline, dtype=float)

    @property
    def normalization_scale(self) -> float:
        """Return the declared distribution standard deviation in input units."""
        return self.spec.scale / _DISTRIBUTION_SD_DIVISOR[self.spec.distribution]

    @property
    def input_unit(self) -> str:
        """Return the registered physical unit for this input."""
        return variable_registry()[self.spec.variable_key].unit

    @property
    def support_status(self) -> str:
        """Return estimable or one explicit unsupported design status."""
        if self.input_kind == "discrete":
            return ADEQUACY_UNSUPPORTED_DISCRETE
        if self.spec.lower is not None or self.spec.upper is not None:
            return ADEQUACY_UNSUPPORTED_BOUNDED
        groups = self.baseline.traces.variation.plan.groups
        if any(self.spec_id in group.spec_ids for group in groups):
            return ADEQUACY_UNSUPPORTED_CORRELATED
        return ADEQUACY_ESTIMABLE

    def _require_oat_pairing(self) -> None:
        baseline = self.baseline.traces.variation.inputs
        perturbed = self.perturbed.traces.variation.inputs
        target = self.input_names.index(self.spec.variable_key)
        other = np.arange(baseline.shape[1]) != target
        require(
            np.array_equal(baseline[:, other], perturbed[:, other]),
            "paired OAT rows may differ only in the declared input",
        )


@dataclass(frozen=True)
class ResponseAccumulatorSnapshot:
    """Immutable resumable sufficient statistics bound to one input contract."""

    contract_sha256: str
    accepted_trials: int
    arrays: tuple[np.ndarray, ...] = field(repr=False)

    def __post_init__(self) -> None:
        require_sha256(self.contract_sha256, "contract_sha256")
        require(
            type(self.accepted_trials) is int and self.accepted_trials >= 0,
            "accepted_trials must be a non-negative integer",
        )
        immutable = tuple(
            immutable_array(np.asarray(value), value.dtype) for value in self.arrays
        )
        object.__setattr__(self, "arrays", immutable)


__all__ = [
    "ADEQUACY_ESTIMABLE",
    "ADEQUACY_INSUFFICIENT_PAIRS",
    "ADEQUACY_STATES",
    "ADEQUACY_UNSUPPORTED_BOUNDED",
    "ADEQUACY_UNSUPPORTED_CORRELATED",
    "ADEQUACY_UNSUPPORTED_DISCRETE",
    "ADEQUACY_ZERO_PERTURBATION",
    "DECLARED_SCALE_NORMALIZATION",
    "METRIC_IDS",
    "METRIC_UNITS",
    "PAIRED_OAT_RESPONSE_METHOD",
    "POSITION_NOISE_RESPONSE_FIELD_SCHEMA_ID",
    "POSITION_NOISE_RESPONSE_FIELD_SCHEMA_VERSION",
    "ResponseAccumulatorSnapshot",
    "ResponseFieldInput",
    "SCIENTIFIC_BOUNDARY",
    "require_sha256",
]

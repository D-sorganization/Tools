"""Streaming declared-scale response and denominator-matched scatter analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.python.contracts import require

from .noise_response_fingerprint import (
    input_contract_fingerprint,
    response_field_fingerprint,
)
from .noise_response_record import PositionNoiseResponseField
from .noise_response_types import (
    ADEQUACY_ESTIMABLE,
    ADEQUACY_INSUFFICIENT_PAIRS,
    ADEQUACY_UNSUPPORTED_BOUNDED,
    ADEQUACY_UNSUPPORTED_CORRELATED,
    ADEQUACY_UNSUPPORTED_DISCRETE,
    ADEQUACY_ZERO_PERTURBATION,
    POSITION_NOISE_RESPONSE_FIELD_SCHEMA_ID,
    POSITION_NOISE_RESPONSE_FIELD_SCHEMA_VERSION,
    ResponseAccumulatorSnapshot,
    ResponseFieldInput,
)

MAX_RESPONSE_ACCUMULATOR_BYTES = 256_000_000
_DENOMINATOR_EPSILON = 64.0 * np.finfo(float).eps
_SNAPSHOT_ARRAY_COUNT = 8


@dataclass(frozen=True)
class _PositionMomentBatch:
    positions: np.ndarray
    paired_valid: np.ndarray
    all_valid: np.ndarray


@dataclass(frozen=True)
class _FrozenMetrics:
    adequacy: np.ndarray
    signed: np.ndarray
    magnitude: np.ndarray
    matched: np.ndarray
    all_scatter: np.ndarray


def _validate_field_inputs(
    values: tuple[ResponseFieldInput, ...],
) -> tuple[ResponseFieldInput, ...]:
    require(bool(values), "at least one response input is required")
    require(
        all(isinstance(item, ResponseFieldInput) for item in values), "invalid input"
    )
    first = values[0]
    plan = first.baseline.traces.variation.plan
    expected_ids = tuple(str(spec.spec_id) for spec in plan.noise)
    require(
        tuple(item.spec_id for item in values) == expected_ids,
        "incomplete input design",
    )
    for item in values[1:]:
        require(item.trial_ids == first.trial_ids, "trial identity drift")
        require(
            item.baseline.traces.point_ids == first.baseline.traces.point_ids,
            "point drift",
        )
        require(
            item.baseline.traces.coordinate_frame
            == first.baseline.traces.coordinate_frame,
            "frame drift",
        )
        require(
            np.array_equal(
                item.baseline.traces.sample_times_s,
                first.baseline.traces.sample_times_s,
            ),
            "time-grid drift",
        )
        require(
            item.execution_metadata.plan_sha256 == first.execution_metadata.plan_sha256,
            "plan drift",
        )
        require(
            item.execution_metadata.registry_sha256
            == first.execution_metadata.registry_sha256,
            "registry drift",
        )
    return values


def _estimated_accumulator_bytes(input_count: int, samples: int, points: int) -> int:
    scalar_cells = input_count * samples
    point_cells = scalar_cells * points
    return 8 * (4 * scalar_cells + 4 * point_cells + 2 * point_cells * 3)


class ResponseFieldAccumulator:
    """Bounded streaming sufficient statistics for one complete paired design."""

    def __init__(self, inputs: tuple[ResponseFieldInput, ...]) -> None:
        self._inputs = _validate_field_inputs(tuple(inputs))
        traces = self._inputs[0].baseline.traces
        shape = (len(self._inputs), traces.sample_times_s.size, len(traces.point_ids))
        require(
            _estimated_accumulator_bytes(*shape) <= MAX_RESPONSE_ACCUMULATOR_BYTES,
            "response accumulator memory budget exceeded",
        )
        scalar_shape = shape[:2]
        vector_shape = shape + (3,)
        self._contract_sha256 = input_contract_fingerprint(self._inputs)
        self._accepted_trials = 0
        self._paired_count = np.zeros(scalar_shape, dtype=np.int64)
        self._all_count = np.zeros(scalar_shape, dtype=np.int64)
        self._normalized_input_square_sum = np.zeros(scalar_shape)
        self._input_displacement_cross_sum = np.zeros(vector_shape)
        self._paired_position_sum = np.zeros(vector_shape)
        self._paired_position_square_sum = np.zeros(shape)
        self._all_position_sum = np.zeros(vector_shape)
        self._all_position_square_sum = np.zeros(shape)

    @property
    def accepted_trials(self) -> int:
        """Return the contiguous trial prefix incorporated so far."""
        return self._accepted_trials

    def accept_trial_slice(self, start: int, stop: int) -> None:
        """Accept one nonempty contiguous trial slice exactly once."""
        total = len(self._inputs[0].trial_ids)
        require(
            type(start) is int and type(stop) is int, "slice bounds must be integers"
        )
        require(start == self._accepted_trials, "trial slices must be contiguous")
        require(start < stop <= total, "trial slice is empty or out of range")
        for input_index, field_input in enumerate(self._inputs):
            self._accept_input_slice(input_index, field_input, slice(start, stop))
        self._accepted_trials = stop

    def _accept_input_slice(
        self, input_index: int, field_input: ResponseFieldInput, trial_slice: slice
    ) -> None:
        baseline = field_input.baseline.traces
        perturbed = field_input.perturbed.traces
        baseline_positions = baseline.positions_m[trial_slice]
        perturbed_positions = perturbed.positions_m[trial_slice]
        paired_valid = (
            baseline.sample_valid[trial_slice] & perturbed.sample_valid[trial_slice]
        )
        all_valid = perturbed.sample_valid[trial_slice]
        normalized_input = (
            field_input.input_delta[trial_slice] / field_input.normalization_scale
        )
        self._paired_count[input_index] += np.count_nonzero(paired_valid, axis=0)
        self._all_count[input_index] += np.count_nonzero(all_valid, axis=0)
        self._normalized_input_square_sum[input_index] += np.einsum(
            "t,ts->s", np.square(normalized_input), paired_valid
        )
        displacement = np.where(
            paired_valid[:, :, None, None],
            perturbed_positions - baseline_positions,
            0.0,
        )
        self._input_displacement_cross_sum[input_index] += np.einsum(
            "t,tspc->spc", normalized_input, displacement
        )
        self._accept_position_moments(
            input_index,
            _PositionMomentBatch(perturbed_positions, paired_valid, all_valid),
        )

    def _accept_position_moments(
        self,
        input_index: int,
        batch: _PositionMomentBatch,
    ) -> None:
        paired = np.where(batch.paired_valid[:, :, None, None], batch.positions, 0.0)
        eligible = np.where(batch.all_valid[:, :, None, None], batch.positions, 0.0)
        self._paired_position_sum[input_index] += np.sum(paired, axis=0)
        self._all_position_sum[input_index] += np.sum(eligible, axis=0)
        self._paired_position_square_sum[input_index] += np.sum(
            np.einsum("tspc,tspc->tsp", paired, paired), axis=0
        )
        self._all_position_square_sum[input_index] += np.sum(
            np.einsum("tspc,tspc->tsp", eligible, eligible), axis=0
        )

    def snapshot(self) -> ResponseAccumulatorSnapshot:
        """Return immutable resumable sufficient statistics for this prefix."""
        return ResponseAccumulatorSnapshot(
            contract_sha256=self._contract_sha256,
            accepted_trials=self._accepted_trials,
            arrays=self._snapshot_arrays(),
        )

    def _snapshot_arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self._paired_count,
            self._all_count,
            self._normalized_input_square_sum,
            self._input_displacement_cross_sum,
            self._paired_position_sum,
            self._paired_position_square_sum,
            self._all_position_sum,
            self._all_position_square_sum,
        )

    @classmethod
    def from_snapshot(
        cls,
        inputs: tuple[ResponseFieldInput, ...],
        snapshot: ResponseAccumulatorSnapshot,
    ) -> ResponseFieldAccumulator:
        """Restore a prefix only when its full source contract still matches."""
        require(isinstance(snapshot, ResponseAccumulatorSnapshot), "invalid snapshot")
        result = cls(inputs)
        require(
            snapshot.contract_sha256 == result._contract_sha256,
            "snapshot contract drift",
        )
        require(len(snapshot.arrays) == _SNAPSHOT_ARRAY_COUNT, "snapshot array drift")
        for target, source in zip(
            result._snapshot_arrays(), snapshot.arrays, strict=True
        ):
            require(target.shape == source.shape, "snapshot shape drift")
            target[...] = source
        result._accepted_trials = snapshot.accepted_trials
        require(
            result._accepted_trials <= len(result._inputs[0].trial_ids),
            "snapshot overflow",
        )
        return result

    def freeze(self) -> PositionNoiseResponseField:
        """Return the immutable field after every declared trial is accepted."""
        require(
            self._accepted_trials == len(self._inputs[0].trial_ids),
            "cannot freeze an incomplete response field",
        )
        adequacy = self._adequacy()
        signed = self._signed_response(adequacy)
        magnitude = np.linalg.norm(signed, axis=-1)
        magnitude[adequacy != ADEQUACY_ESTIMABLE] = np.nan
        matched = _rms_scatter(
            self._paired_count,
            self._paired_position_sum,
            self._paired_position_square_sum,
        )
        all_scatter = _rms_scatter(
            self._all_count,
            self._all_position_sum,
            self._all_position_square_sum,
        )
        return self._build_field(
            _FrozenMetrics(adequacy, signed, magnitude, matched, all_scatter)
        )

    def _adequacy(self) -> np.ndarray:
        shape = self._paired_position_square_sum.shape
        result = np.full(shape, ADEQUACY_INSUFFICIENT_PAIRS, dtype="<U32")
        for input_index, field_input in enumerate(self._inputs):
            if field_input.support_status != ADEQUACY_ESTIMABLE:
                result[input_index] = field_input.support_status
                continue
            cell_shape = result[input_index].shape
            enough = np.broadcast_to(
                self._paired_count[input_index, :, None] >= 2, cell_shape
            )
            nonzero = np.broadcast_to(
                self._normalized_input_square_sum[input_index, :, None]
                > _DENOMINATOR_EPSILON,
                cell_shape,
            )
            result[input_index][enough] = ADEQUACY_ZERO_PERTURBATION
            result[input_index][enough & nonzero] = ADEQUACY_ESTIMABLE
        return result

    def _signed_response(self, adequacy: np.ndarray) -> np.ndarray:
        result = np.full(self._input_displacement_cross_sum.shape, np.nan)
        denominators = self._normalized_input_square_sum[:, :, None, None]
        estimable = adequacy == ADEQUACY_ESTIMABLE
        np.divide(
            self._input_displacement_cross_sum,
            denominators,
            out=result,
            where=estimable[:, :, :, None],
        )
        return result

    def _build_field(self, metrics: _FrozenMetrics) -> PositionNoiseResponseField:
        inputs = self._inputs
        first = inputs[0]
        metadata = tuple(item.execution_metadata for item in inputs)
        shape = self._paired_position_square_sum.shape
        return PositionNoiseResponseField(
            sample_times_s=first.baseline.traces.sample_times_s,
            coordinate_frame=first.baseline.traces.coordinate_frame,
            point_ids=first.baseline.traces.point_ids,
            trial_ids=first.trial_ids,
            input_ids=tuple(item.spec_id for item in inputs),
            input_units=tuple(item.input_unit for item in inputs),
            input_declared_scales=np.array([item.spec.scale for item in inputs]),
            input_normalization_scales=np.array(
                [item.normalization_scale for item in inputs]
            ),
            source_layout_ids=tuple(item.source_layout_id for item in inputs),
            adapter_ids=tuple(item.adapter_id for item in inputs),
            source_sha256=tuple(item.source_sha256 for item in inputs),
            plan_sha256=tuple(item.plan_sha256 for item in metadata),
            registry_sha256=tuple(item.registry_sha256 for item in metadata),
            execution_provenance_sha256=tuple(
                item.provenance_sha256 for item in metadata
            ),
            availability_count=np.broadcast_to(self._paired_count[:, :, None], shape),
            all_eligible_count=np.broadcast_to(self._all_count[:, :, None], shape),
            adequacy=metrics.adequacy,
            signed_response_m_per_declared_scale=metrics.signed,
            response_magnitude_m_per_declared_scale=metrics.magnitude,
            matched_absolute_rms_scatter_m=metrics.matched,
            all_eligible_absolute_rms_scatter_m=metrics.all_scatter,
        )


def _rms_scatter(
    counts: np.ndarray, position_sum: np.ndarray, position_square_sum: np.ndarray
) -> np.ndarray:
    divisor = counts[:, :, None]
    centered = position_square_sum.copy()
    correction = np.sum(np.square(position_sum), axis=-1)
    np.divide(correction, divisor, out=correction, where=divisor > 0)
    centered -= correction
    result = np.full(position_square_sum.shape, np.nan)
    np.divide(np.maximum(centered, 0.0), divisor, out=result, where=divisor > 0)
    np.sqrt(result, out=result, where=divisor > 0)
    return result


def compute_position_noise_response_field(
    inputs: tuple[ResponseFieldInput, ...], chunk_size: int | None = None
) -> PositionNoiseResponseField:
    """Compute one complete immutable field with bounded streaming moments."""
    values = tuple(inputs)
    accumulator = ResponseFieldAccumulator(values)
    trial_count = len(values[0].trial_ids)
    size = trial_count if chunk_size is None else chunk_size
    require(type(size) is int and size >= 1, "chunk_size must be a positive integer")
    for start in range(0, trial_count, size):
        accumulator.accept_trial_slice(start, min(start + size, trial_count))
    return accumulator.freeze()


__all__ = [
    "ADEQUACY_ESTIMABLE",
    "ADEQUACY_INSUFFICIENT_PAIRS",
    "ADEQUACY_UNSUPPORTED_BOUNDED",
    "ADEQUACY_UNSUPPORTED_CORRELATED",
    "ADEQUACY_UNSUPPORTED_DISCRETE",
    "ADEQUACY_ZERO_PERTURBATION",
    "MAX_RESPONSE_ACCUMULATOR_BYTES",
    "POSITION_NOISE_RESPONSE_FIELD_SCHEMA_ID",
    "POSITION_NOISE_RESPONSE_FIELD_SCHEMA_VERSION",
    "PositionNoiseResponseField",
    "ResponseAccumulatorSnapshot",
    "ResponseFieldAccumulator",
    "ResponseFieldInput",
    "compute_position_noise_response_field",
    "response_field_fingerprint",
]

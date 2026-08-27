"""Immutable result record for paired localized attribution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.python.contracts import require

from .ensemble_types import immutable_array
from .paired_attribution_types import (
    AVAILABILITY_AVAILABLE,
    AVAILABILITY_MISSING,
    AVAILABILITY_NO_IMPACT,
    AVAILABILITY_NONFINITE,
    AVAILABILITY_NUMERICAL_FAILURE,
    AVAILABILITY_UNSUPPORTED,
    INTERPRETATION_BOUNDARY,
    PAIR_AVAILABILITY_STATES,
    PAIRED_ATTRIBUTION_SCHEMA_ID,
    PAIRED_ATTRIBUTION_SCHEMA_VERSION,
    PAIRED_INTERVENTION_METHOD_ID,
    AttributionRunContext,
    AttributionSource,
    AttributionTarget,
)


@dataclass(frozen=True)
class PairedAttributionRecord:
    """Complete plot/reviewer-ready paired response matrix."""

    source: AttributionSource
    targets: tuple[AttributionTarget, ...]
    context: AttributionRunContext
    source_sha256: str
    pair_ids: tuple[str, ...]
    baseline_trial_ids: tuple[str, ...]
    perturbed_trial_ids: tuple[str, ...]
    baseline_statuses: tuple[str, ...]
    perturbed_statuses: tuple[str, ...]
    baseline_source_values: np.ndarray = field(repr=False)
    perturbed_source_values: np.ndarray = field(repr=False)
    baseline_values: np.ndarray = field(repr=False)
    perturbed_values: np.ndarray = field(repr=False)
    availability: np.ndarray = field(repr=False)
    signed_response: np.ndarray = field(repr=False)
    response_magnitude: np.ndarray = field(repr=False)
    local_response_per_source_unit: np.ndarray = field(repr=False)
    available_count: np.ndarray = field(repr=False)
    no_impact_count: np.ndarray = field(repr=False)
    numerical_failure_count: np.ndarray = field(repr=False)
    missing_count: np.ndarray = field(repr=False)
    nonfinite_count: np.ndarray = field(repr=False)
    unsupported_count: np.ndarray = field(repr=False)
    schema_id: str = PAIRED_ATTRIBUTION_SCHEMA_ID
    schema_version: int = PAIRED_ATTRIBUTION_SCHEMA_VERSION
    method_id: str = PAIRED_INTERVENTION_METHOD_ID
    interpretation_boundary: str = INTERPRETATION_BOUNDARY

    def __post_init__(self) -> None:
        require(self.schema_id == PAIRED_ATTRIBUTION_SCHEMA_ID, "schema drift")
        require(
            self.schema_version == PAIRED_ATTRIBUTION_SCHEMA_VERSION,
            "schema-version drift",
        )
        require(self.method_id == PAIRED_INTERVENTION_METHOD_ID, "method drift")
        require(
            self.interpretation_boundary == INTERPRETATION_BOUNDARY,
            "interpretation drift",
        )
        pairs = len(self.pair_ids)
        targets = len(self.targets)
        require(pairs > 0 and targets > 0, "record matrix must be nonempty")
        for values in (
            self.baseline_trial_ids,
            self.perturbed_trial_ids,
            self.baseline_statuses,
            self.perturbed_statuses,
        ):
            require(len(values) == pairs, "pair metadata length mismatch")
        require(len(set(self.pair_ids)) == pairs, "pair IDs must be unique")
        matrix_shape = (pairs, targets)
        self._validate_matrix_shapes(matrix_shape)
        self._validate_values(matrix_shape)
        self._freeze_arrays()

    def _validate_matrix_shapes(self, matrix_shape: tuple[int, int]) -> None:
        require(
            np.asarray(self.baseline_source_values).shape == (matrix_shape[0],),
            "baseline source shape mismatch",
        )
        require(
            np.asarray(self.perturbed_source_values).shape == (matrix_shape[0],),
            "perturbed source shape mismatch",
        )
        for name in (
            "baseline_values",
            "perturbed_values",
            "availability",
            "signed_response",
            "response_magnitude",
            "local_response_per_source_unit",
        ):
            require(
                np.asarray(getattr(self, name)).shape == matrix_shape,
                f"{name} shape mismatch",
            )
        for name in self._count_names():
            require(
                np.asarray(getattr(self, name)).shape == (matrix_shape[1],),
                f"{name} shape mismatch",
            )

    def _validate_values(self, matrix_shape: tuple[int, int]) -> None:
        availability = np.asarray(self.availability)
        require(
            bool(np.all(np.isin(availability, PAIR_AVAILABILITY_STATES))),
            "invalid availability",
        )
        available = availability == AVAILABILITY_AVAILABLE
        for name in (
            "signed_response",
            "response_magnitude",
            "local_response_per_source_unit",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            require(
                bool(np.all(np.isfinite(values[available]))),
                f"available {name} must be finite",
            )
            require(
                bool(np.all(np.isnan(values[~available]))),
                f"unavailable {name} must be NaN",
            )
        magnitude = np.asarray(self.response_magnitude)
        signed = np.asarray(self.signed_response)
        require(
            np.array_equal(magnitude[available], np.abs(signed[available])),
            "response magnitude mismatch",
        )
        delta = np.asarray(self.perturbed_source_values) - np.asarray(
            self.baseline_source_values
        )
        require(
            bool(np.all(np.isfinite(delta))) and bool(np.all(delta != 0.0)),
            "invalid source deltas",
        )
        expected = signed / delta[:, np.newaxis]
        local = np.asarray(self.local_response_per_source_unit)
        require(
            np.allclose(local[available], expected[available]),
            "local response mismatch",
        )
        require(matrix_shape == availability.shape, "matrix shape drift")
        self._validate_counts(availability)

    def _validate_counts(self, availability: np.ndarray) -> None:
        states = (
            AVAILABILITY_AVAILABLE,
            AVAILABILITY_NO_IMPACT,
            AVAILABILITY_NUMERICAL_FAILURE,
            AVAILABILITY_MISSING,
            AVAILABILITY_NONFINITE,
            AVAILABILITY_UNSUPPORTED,
        )
        for name, state in zip(self._count_names(), states, strict=True):
            counts = np.asarray(getattr(self, name))
            require(np.issubdtype(counts.dtype, np.integer), f"{name} must be integer")
            require(
                np.array_equal(counts, np.sum(availability == state, axis=0)),
                f"{name} mismatch",
            )

    @staticmethod
    def _count_names() -> tuple[str, ...]:
        return (
            "available_count",
            "no_impact_count",
            "numerical_failure_count",
            "missing_count",
            "nonfinite_count",
            "unsupported_count",
        )

    def _freeze_arrays(self) -> None:
        fields = (
            ("baseline_source_values", float),
            ("perturbed_source_values", float),
            ("baseline_values", float),
            ("perturbed_values", float),
            ("availability", str),
            ("signed_response", float),
            ("response_magnitude", float),
            ("local_response_per_source_unit", float),
            ("available_count", int),
            ("no_impact_count", int),
            ("numerical_failure_count", int),
            ("missing_count", int),
            ("nonfinite_count", int),
            ("unsupported_count", int),
        )
        for name, dtype in fields:
            object.__setattr__(
                self, name, immutable_array(np.asarray(getattr(self, name)), dtype)
            )


__all__ = ["PairedAttributionRecord"]

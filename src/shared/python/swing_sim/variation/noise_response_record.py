"""Immutable plot-ready record for geometric noise-response evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.python.contracts import require

from .ensemble_types import (
    CARTESIAN_DIMENSIONS,
    immutable_array,
    require_coordinate_frame_id,
    require_point_ids,
    validated_sample_times,
)
from .identity_contracts import stable_id
from .noise_response_types import (
    ADEQUACY_ESTIMABLE,
    ADEQUACY_STATES,
    DECLARED_SCALE_NORMALIZATION,
    METRIC_IDS,
    METRIC_UNITS,
    PAIRED_OAT_RESPONSE_METHOD,
    POSITION_NOISE_RESPONSE_FIELD_SCHEMA_ID,
    POSITION_NOISE_RESPONSE_FIELD_SCHEMA_VERSION,
    SCIENTIFIC_BOUNDARY,
    require_sha256,
)
from .trace_resampling import TRACE_RESAMPLING_POLICY_ID


def _stable_tuple(value: tuple[str, ...], name: str) -> tuple[str, ...]:
    require(type(value) is tuple and bool(value), f"{name} must be a nonempty tuple")
    result = tuple(stable_id(item, name) for item in value)
    require(len(set(result)) == len(result), f"{name} must be unique")
    return result


def _require_metric_shapes(field_value: PositionNoiseResponseField) -> tuple[int, ...]:
    shape = (
        len(field_value.input_ids),
        field_value.sample_times_s.size,
        len(field_value.point_ids),
    )
    require(field_value.availability_count.shape == shape, "invalid count shape")
    require(field_value.all_eligible_count.shape == shape, "invalid all-count shape")
    require(field_value.adequacy.shape == shape, "invalid adequacy shape")
    require(
        field_value.signed_response_m_per_declared_scale.shape
        == shape + (CARTESIAN_DIMENSIONS,),
        "invalid signed-response shape",
    )
    for name in (
        "response_magnitude_m_per_declared_scale",
        "matched_absolute_rms_scatter_m",
        "all_eligible_absolute_rms_scatter_m",
    ):
        require(
            np.asarray(getattr(field_value, name)).shape == shape, f"invalid {name}"
        )
    return shape


@dataclass(frozen=True)
class PositionNoiseResponseField:
    """Plot-ready absolute scatter and declared-scale response for all inputs."""

    sample_times_s: np.ndarray = field(repr=False)
    coordinate_frame: str
    point_ids: tuple[str, ...]
    trial_ids: tuple[str, ...]
    input_ids: tuple[str, ...]
    input_units: tuple[str, ...]
    input_declared_scales: np.ndarray = field(repr=False)
    input_normalization_scales: np.ndarray = field(repr=False)
    source_layout_ids: tuple[str, ...]
    adapter_ids: tuple[str, ...]
    source_sha256: tuple[str, ...]
    plan_sha256: tuple[str, ...]
    registry_sha256: tuple[str, ...]
    execution_provenance_sha256: tuple[str, ...]
    availability_count: np.ndarray = field(repr=False)
    all_eligible_count: np.ndarray = field(repr=False)
    adequacy: np.ndarray = field(repr=False)
    signed_response_m_per_declared_scale: np.ndarray = field(repr=False)
    response_magnitude_m_per_declared_scale: np.ndarray = field(repr=False)
    matched_absolute_rms_scatter_m: np.ndarray = field(repr=False)
    all_eligible_absolute_rms_scatter_m: np.ndarray = field(repr=False)
    schema_id: str = POSITION_NOISE_RESPONSE_FIELD_SCHEMA_ID
    schema_version: int = POSITION_NOISE_RESPONSE_FIELD_SCHEMA_VERSION
    method_id: str = PAIRED_OAT_RESPONSE_METHOD
    normalization_id: str = DECLARED_SCALE_NORMALIZATION
    resampling_policy_id: str = TRACE_RESAMPLING_POLICY_ID
    coordinate_kind: str = "time"
    coordinate_unit: str = "s"
    position_unit: str = "m"
    metric_ids: tuple[str, ...] = METRIC_IDS
    metric_units: tuple[str, ...] = METRIC_UNITS
    scientific_boundary: str = SCIENTIFIC_BOUNDARY

    def __post_init__(self) -> None:
        self._validate_metadata()
        shape = _require_metric_shapes(self)
        self._validate_values(shape)
        self._freeze_arrays()

    def _validate_metadata(self) -> None:
        require(
            self.schema_id == POSITION_NOISE_RESPONSE_FIELD_SCHEMA_ID, "schema drift"
        )
        require(
            self.schema_version == POSITION_NOISE_RESPONSE_FIELD_SCHEMA_VERSION,
            "schema-version drift",
        )
        require(self.method_id == PAIRED_OAT_RESPONSE_METHOD, "method drift")
        require(
            self.normalization_id == DECLARED_SCALE_NORMALIZATION, "normalization drift"
        )
        require(self.resampling_policy_id == TRACE_RESAMPLING_POLICY_ID, "policy drift")
        validated_sample_times(self.sample_times_s)
        require_coordinate_frame_id(self.coordinate_frame)
        require_point_ids(tuple(self.point_ids))
        _stable_tuple(self.trial_ids, "trial_ids")
        _stable_tuple(self.input_ids, "input_ids")
        require(
            self.metric_ids == METRIC_IDS and self.metric_units == METRIC_UNITS,
            "metric drift",
        )
        count = len(self.input_ids)
        for values in (
            self.input_units,
            self.source_layout_ids,
            self.adapter_ids,
            self.source_sha256,
            self.plan_sha256,
            self.registry_sha256,
            self.execution_provenance_sha256,
        ):
            require(len(values) == count, "per-input metadata length mismatch")
        self._validate_digests()

    def _validate_digests(self) -> None:
        for digests in (
            self.source_sha256,
            self.plan_sha256,
            self.registry_sha256,
            self.execution_provenance_sha256,
        ):
            for digest in digests:
                require_sha256(digest, "field provenance digest")

    def _validate_values(self, shape: tuple[int, ...]) -> None:
        counts = np.asarray(self.availability_count)
        all_counts = np.asarray(self.all_eligible_count)
        require(np.issubdtype(counts.dtype, np.integer), "counts must be integers")
        require(
            np.issubdtype(all_counts.dtype, np.integer), "all counts must be integers"
        )
        require(
            bool(np.all(counts >= 0)) and bool(np.all(all_counts >= counts)),
            "invalid counts",
        )
        adequacy = np.asarray(self.adequacy)
        require(bool(np.all(np.isin(adequacy, ADEQUACY_STATES))), "invalid adequacy")
        self._validate_response_values(adequacy)
        self._validate_scatter_values(shape, counts, all_counts)

    def _validate_response_values(self, adequacy: np.ndarray) -> None:
        signed = np.asarray(self.signed_response_m_per_declared_scale, dtype=float)
        magnitude = np.asarray(
            self.response_magnitude_m_per_declared_scale, dtype=float
        )
        estimable = adequacy == ADEQUACY_ESTIMABLE
        require(
            bool(np.all(np.isfinite(signed[estimable]))),
            "estimable response must be finite",
        )
        require(
            bool(np.all(np.isnan(signed[~estimable]))),
            "unavailable response must be NaN",
        )
        require(
            bool(np.all(np.isfinite(magnitude[estimable]))),
            "estimable magnitude must be finite",
        )
        require(
            bool(np.all(np.isnan(magnitude[~estimable]))),
            "unavailable magnitude must be NaN",
        )
        require(
            np.allclose(
                magnitude[estimable], np.linalg.norm(signed[estimable], axis=-1)
            ),
            "response magnitude must match signed components",
        )

    def _validate_scatter_values(
        self, shape: tuple[int, ...], counts: np.ndarray, all_counts: np.ndarray
    ) -> None:
        pairs = (
            (self.matched_absolute_rms_scatter_m, counts > 0),
            (self.all_eligible_absolute_rms_scatter_m, all_counts > 0),
        )
        for values, available in pairs:
            array = np.asarray(values, dtype=float)
            require(array.shape == shape, "scatter shape drift")
            require(
                bool(np.all(np.isfinite(array[available]))),
                "available scatter must be finite",
            )
            require(
                bool(np.all(array[available] >= 0.0)),
                "scatter must be non-negative",
            )
            require(
                bool(np.all(np.isnan(array[~available]))),
                "unavailable scatter must be NaN",
            )

    def _freeze_arrays(self) -> None:
        fields = (
            ("sample_times_s", float),
            ("input_declared_scales", float),
            ("input_normalization_scales", float),
            ("availability_count", int),
            ("all_eligible_count", int),
            ("adequacy", str),
            ("signed_response_m_per_declared_scale", float),
            ("response_magnitude_m_per_declared_scale", float),
            ("matched_absolute_rms_scatter_m", float),
            ("all_eligible_absolute_rms_scatter_m", float),
        )
        for name, dtype in fields:
            value = np.asarray(getattr(self, name))
            object.__setattr__(self, name, immutable_array(value, dtype))


__all__ = ["PositionNoiseResponseField"]

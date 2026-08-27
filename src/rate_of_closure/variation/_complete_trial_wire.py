"""Strict pickle-free wire projection for complete trial records."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from shared.python.contracts import require

from .complete_trial_record import (
    COMPLETE_TRIAL_SCHEMA,
    CompleteTrialRecord,
)
from .simulation_types import TrialEvaluationStatus

WIRE_SCHEMA = "rate-complete-trial-wire/v1"
METADATA_ARRAY_KEY = "complete_records_json"
VALUES_ARRAY_KEY = "complete_record_values"
ARRAY_FIELDS = (
    "sampled_inputs",
    "swing_times_s",
    "swing_positions_m",
    "swing_poses",
    "swing_twists",
    "swing_joint_positions_m",
    "swing_applied_torques_nm",
    "flight_times_s",
    "flight_positions_m",
    "flight_velocities_mps",
)
_RECORD_KEYS = frozenset(
    {
        "trial_index",
        "status",
        "plan_sha256",
        "execution_sha256",
        "stream_configuration_sha256",
        "configuration_sha256",
        "sampled_input_sha256",
        "registry_sha256",
        "adapter_ids",
        "source_repository",
        "source_revision",
        "source_revision_status",
        "source_revision_reason",
        "source_kind",
        "coordinate_frame",
        "spatial_point_ids",
        "torque_joint_ids",
        "units",
        "candidate_time_s",
        "impact_time_s",
        "event_sample_index",
        "event_interpolation_status",
        "pre_impact_sample_count",
        "failure_type",
        "failure_message",
        "impact_outcome",
        "delivery_state",
        "post_impact_state",
        "launch_state",
        "arrays",
    }
)


def pack_complete_records(
    records: Sequence[CompleteTrialRecord],
) -> dict[str, np.ndarray]:
    """Flatten heterogeneous trial arrays into one bounded numeric payload."""
    metadata: list[dict[str, object]] = []
    values: list[np.ndarray] = []
    offset = 0
    for record in records:
        require(isinstance(record, CompleteTrialRecord), "invalid complete record")
        descriptors: dict[str, object] = {}
        for name in ARRAY_FIELDS:
            array = np.asarray(getattr(record, name), dtype="<f8")
            flat = np.ascontiguousarray(array).reshape(-1)
            descriptors[name] = {
                "offset": offset,
                "length": int(flat.size),
                "shape": list(array.shape),
            }
            values.append(flat)
            offset += int(flat.size)
        metadata.append(_record_metadata(record, descriptors))
    document = {
        "schema_version": WIRE_SCHEMA,
        "record_schema": COMPLETE_TRIAL_SCHEMA,
        "records": metadata,
    }
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    combined = np.concatenate(values) if values else np.empty(0, dtype="<f8")
    return {
        METADATA_ARRAY_KEY: np.frombuffer(payload, dtype=np.uint8).copy(),
        VALUES_ARRAY_KEY: np.asarray(combined, dtype="<f8"),
    }


def unpack_complete_records(
    metadata_array: np.ndarray,
    values_array: np.ndarray,
    expected_count: int,
) -> tuple[CompleteTrialRecord, ...]:
    """Decode and validate one exact complete-record payload."""
    require(
        metadata_array.ndim == 1 and metadata_array.dtype == np.dtype(np.uint8),
        "complete-record metadata must be uint8 bytes",
    )
    require(
        values_array.ndim == 1
        and values_array.dtype == np.dtype("<f8")
        and bool(np.all(np.isfinite(values_array))),
        "complete-record values must be finite little-endian float64",
    )
    try:
        document = json.loads(
            metadata_array.tobytes().decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        require(False, "complete-record metadata must be strict JSON", str(exc))
        raise AssertionError from exc
    require(type(document) is dict, "complete-record document must be an object")
    root = cast(dict[str, object], document)
    require(
        set(root) == {"schema_version", "record_schema", "records"},
        "complete-record document fields are invalid",
    )
    require(root["schema_version"] == WIRE_SCHEMA, "unsupported trial wire schema")
    require(
        root["record_schema"] == COMPLETE_TRIAL_SCHEMA,
        "unsupported complete trial schema",
    )
    raw_records = root["records"]
    require(type(raw_records) is list, "complete records must be a list")
    entries = cast(list[object], raw_records)
    require(len(entries) == expected_count, "complete record count is invalid")
    cursor = 0
    result: list[CompleteTrialRecord] = []
    for raw in entries:
        require(type(raw) is dict, "complete record metadata must be an object")
        item = cast(dict[str, object], raw)
        require(
            set(item) == _RECORD_KEYS, "complete record metadata fields are invalid"
        )
        arrays, cursor = _decode_arrays(item["arrays"], values_array, cursor)
        result.append(_decode_record(item, arrays))
    require(cursor == values_array.size, "complete-record values contain trailing data")
    return tuple(result)


def _record_metadata(
    record: CompleteTrialRecord, descriptors: Mapping[str, object]
) -> dict[str, object]:
    return {
        "trial_index": record.trial_index,
        "status": record.status.value,
        "plan_sha256": record.plan_sha256,
        "execution_sha256": record.execution_sha256,
        "stream_configuration_sha256": record.stream_configuration_sha256,
        "configuration_sha256": record.configuration_sha256,
        "sampled_input_sha256": record.sampled_input_sha256,
        "registry_sha256": record.registry_sha256,
        "adapter_ids": list(record.adapter_ids),
        "source_repository": record.source_repository,
        "source_revision": record.source_revision,
        "source_revision_status": record.source_revision_status,
        "source_revision_reason": record.source_revision_reason,
        "source_kind": record.source_kind,
        "coordinate_frame": record.coordinate_frame,
        "spatial_point_ids": list(record.spatial_point_ids),
        "torque_joint_ids": list(record.torque_joint_ids),
        "units": dict(record.units),
        "candidate_time_s": record.candidate_time_s,
        "impact_time_s": record.impact_time_s,
        "event_sample_index": record.event_sample_index,
        "event_interpolation_status": record.event_interpolation_status,
        "pre_impact_sample_count": record.pre_impact_sample_count,
        "failure_type": record.failure_type,
        "failure_message": record.failure_message,
        "impact_outcome": _json_ready(record.impact_outcome),
        "delivery_state": _json_ready(record.delivery_state),
        "post_impact_state": _json_ready(record.post_impact_state),
        "launch_state": _json_ready(record.launch_state),
        "arrays": descriptors,
    }


def _decode_arrays(
    raw: object, values: np.ndarray, cursor: int
) -> tuple[dict[str, np.ndarray], int]:
    require(type(raw) is dict, "complete record arrays must be an object")
    descriptors = cast(dict[str, object], raw)
    require(set(descriptors) == set(ARRAY_FIELDS), "complete record arrays are invalid")
    result: dict[str, np.ndarray] = {}
    for name in ARRAY_FIELDS:
        descriptor = descriptors[name]
        require(type(descriptor) is dict, "array descriptor must be an object")
        value = cast(dict[str, object], descriptor)
        require(
            set(value) == {"offset", "length", "shape"},
            "array descriptor fields are invalid",
        )
        offset = value["offset"]
        length = value["length"]
        shape = value["shape"]
        require(
            type(offset) is int and offset == cursor, "array offsets are not contiguous"
        )
        require(type(length) is int and length >= 0, "array length is invalid")
        length = cast(int, length)
        require(
            type(shape) is list
            and all(type(dimension) is int and dimension >= 0 for dimension in shape),
            "array shape is invalid",
        )
        dimensions = tuple(cast(list[int], shape))
        require(
            int(np.prod(dimensions, dtype=np.int64)) == length,
            "array shape does not match length",
        )
        stop = cursor + length
        require(stop <= values.size, "array slice exceeds complete-record values")
        result[name] = np.array(values[cursor:stop], copy=True).reshape(dimensions)
        cursor = stop
    return result, cursor


def _decode_record(
    item: dict[str, object], arrays: dict[str, np.ndarray]
) -> CompleteTrialRecord:
    try:
        status = TrialEvaluationStatus(str(item["status"]))
    except ValueError as exc:
        require(False, "complete record status is invalid", str(exc))
        raise AssertionError from exc
    return CompleteTrialRecord(
        trial_index=_integer(item, "trial_index"),
        status=status,
        sampled_inputs=arrays["sampled_inputs"],
        plan_sha256=_string(item, "plan_sha256"),
        execution_sha256=_string(item, "execution_sha256"),
        stream_configuration_sha256=_string(item, "stream_configuration_sha256"),
        configuration_sha256=_string(item, "configuration_sha256"),
        sampled_input_sha256=_string(item, "sampled_input_sha256"),
        registry_sha256=_string(item, "registry_sha256"),
        adapter_ids=_string_tuple(item, "adapter_ids"),
        source_repository=_string(item, "source_repository"),
        source_revision=_optional_string(item, "source_revision"),
        source_revision_status=_string(item, "source_revision_status"),
        source_revision_reason=_optional_string(item, "source_revision_reason"),
        source_kind=_string(item, "source_kind"),
        coordinate_frame=_string(item, "coordinate_frame"),
        spatial_point_ids=_string_tuple(item, "spatial_point_ids"),
        torque_joint_ids=_string_tuple(item, "torque_joint_ids"),
        units=_string_mapping(item, "units"),
        candidate_time_s=_optional_float(item, "candidate_time_s"),
        impact_time_s=_optional_float(item, "impact_time_s"),
        event_sample_index=_optional_integer(item, "event_sample_index"),
        event_interpolation_status=_string(item, "event_interpolation_status"),
        pre_impact_sample_count=_integer(item, "pre_impact_sample_count"),
        failure_type=_optional_string(item, "failure_type"),
        failure_message=_optional_string(item, "failure_message"),
        swing_times_s=arrays["swing_times_s"],
        swing_positions_m=arrays["swing_positions_m"],
        swing_poses=arrays["swing_poses"],
        swing_twists=arrays["swing_twists"],
        swing_joint_positions_m=arrays["swing_joint_positions_m"],
        swing_applied_torques_nm=arrays["swing_applied_torques_nm"],
        impact_outcome=_optional_mapping(item, "impact_outcome"),
        delivery_state=_optional_mapping(item, "delivery_state"),
        post_impact_state=_optional_mapping(item, "post_impact_state"),
        launch_state=_optional_mapping(item, "launch_state"),
        flight_times_s=arrays["flight_times_s"],
        flight_positions_m=arrays["flight_positions_m"],
        flight_velocities_mps=arrays["flight_velocities_mps"],
    )


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _integer(item: Mapping[str, object], name: str) -> int:
    value = item[name]
    require(type(value) is int, f"{name} must be an integer")
    return cast(int, value)


def _optional_integer(item: Mapping[str, object], name: str) -> int | None:
    value = item[name]
    require(value is None or type(value) is int, f"{name} must be an optional integer")
    return cast(int | None, value)


def _optional_float(item: Mapping[str, object], name: str) -> float | None:
    value = item[name]
    require(value is None or type(value) is float, f"{name} must be an optional float")
    return cast(float | None, value)


def _string(item: Mapping[str, object], name: str) -> str:
    value = item[name]
    require(type(value) is str, f"{name} must be a string")
    return cast(str, value)


def _optional_string(item: Mapping[str, object], name: str) -> str | None:
    value = item[name]
    require(value is None or type(value) is str, f"{name} must be an optional string")
    return cast(str | None, value)


def _string_tuple(item: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = item[name]
    require(
        type(value) is list and all(type(entry) is str for entry in value),
        f"{name} must contain strings",
    )
    return tuple(cast(list[str], value))


def _optional_mapping(
    item: Mapping[str, object], name: str
) -> Mapping[str, object] | None:
    value = item[name]
    require(value is None or type(value) is dict, f"{name} must be an optional object")
    return cast(Mapping[str, object] | None, value)


def _string_mapping(item: Mapping[str, object], name: str) -> Mapping[str, str]:
    value = item[name]
    require(
        type(value) is dict
        and all(
            type(key) is str and type(entry) is str for key, entry in value.items()
        ),
        f"{name} must map strings to strings",
    )
    return cast(Mapping[str, str], value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate complete-record field", key)
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    require(False, "complete-record numbers must be finite", value)


__all__ = [
    "METADATA_ARRAY_KEY",
    "VALUES_ARRAY_KEY",
    "WIRE_SCHEMA",
    "pack_complete_records",
    "unpack_complete_records",
]

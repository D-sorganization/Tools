"""Strict bounded parser for Morris scalar-observation archives."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np

from shared.python.contracts import require

from ._morris_observation_validation import (
    exact_mapping,
    finite,
    nonnegative_integer,
    nullable_finite,
    optional_text,
    provenance_mapping,
    sha256_hex,
    stable_text,
)
from .morris_design import MorrisDesign, MorrisFactor, MorrisObservations, MorrisOutput
from .morris_observation_io import (
    _DESIGN_FIELDS,
    _FACTOR_FIELDS,
    _MAX_ARCHIVE_NODES,
    _MAX_ARCHIVE_TEXT_BYTES,
    _OUTPUT_FIELDS,
    _PHYSICAL_FIELDS,
    _RECORD_FIELDS,
    _ROOT_FIELDS,
    _VALUE_FIELDS,
    MORRIS_OBSERVATION_SCHEMA_ID,
    MORRIS_OBSERVATION_SCHEMA_VERSION,
    MorrisObservationArchive,
    _canonical_bytes,
    _design_sha256,
    _require_observation_counts,
    _require_observation_shape,
    _sample_id,
    morris_observations_to_json_dict,
)


def _parse_factor(value: object) -> MorrisFactor:
    item = exact_mapping(value, _FACTOR_FIELDS, "factor")
    raw_window = item["source_time_window_s"]
    window = None
    if raw_window is not None:
        require(
            isinstance(raw_window, list) and len(raw_window) == 2,
            "source time window must contain two values",
        )
        window = (
            finite(raw_window[0], "window start"),
            finite(raw_window[1], "window end"),
        )
    raw_points = item["source_point_ids"]
    require(isinstance(raw_points, list), "source_point_ids must be an array")
    return MorrisFactor(
        stable_text(item["spec_id"], "spec_id"),
        stable_text(item["variable_key"], "variable_key"),
        finite(item["lower"], "factor lower"),
        finite(item["upper"], "factor upper"),
        stable_text(item["unit"], "factor unit"),
        window,
        tuple(stable_text(point, "source point ID") for point in raw_points),
    )


def _parse_output(value: object) -> MorrisOutput:
    item = exact_mapping(value, _OUTPUT_FIELDS, "output")
    return MorrisOutput(
        stable_text(item["name"], "output name"),
        stable_text(item["unit"], "output unit"),
        stable_text(item["target_kind"], "target kind"),
        nullable_finite(item["target_time_s"], "target time"),
        optional_text(item["target_point_id"], "target point ID"),
        optional_text(item["coordinate_frame"], "coordinate frame"),
    )


def _parse_design(value: object) -> MorrisDesign:
    item = exact_mapping(value, _DESIGN_FIELDS, "design")
    raw_factors = item["factors"]
    require(
        isinstance(raw_factors, list) and bool(raw_factors), "factors must be nonempty"
    )
    trajectories = nonnegative_integer(item["trajectories"], "trajectories")
    _require_observation_counts(trajectories, len(raw_factors), 0)
    design = MorrisDesign(
        tuple(_parse_factor(factor) for factor in raw_factors),
        trajectories,
        nonnegative_integer(item["levels"], "levels"),
        nonnegative_integer(item["seed"], "seed"),
        np.asarray(item["normalized_points"], dtype=float),
        np.asarray(item["changed_factor_indices"]),
        np.asarray(item["signed_steps"], dtype=float),
    )
    return design


def _bounded(value: object) -> None:
    pending = [value]
    nodes = text_bytes = 0
    while pending:
        current = pending.pop()
        nodes += 1
        require(nodes <= _MAX_ARCHIVE_NODES, "archive exceeds node limit")
        if isinstance(current, str):
            text_bytes += len(current.encode("utf-8"))
            require(
                text_bytes <= _MAX_ARCHIVE_TEXT_BYTES, "archive exceeds text-byte limit"
            )
        elif isinstance(current, Mapping):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)


def parse_morris_observations(value: object) -> MorrisObservationArchive:
    """Parse an exact archive and reconstruct immutable observation arrays."""
    _bounded(value)
    item = exact_mapping(value, _ROOT_FIELDS, "Morris observation archive")
    require(item["schema_id"] == MORRIS_OBSERVATION_SCHEMA_ID, "unsupported schema ID")
    require(
        type(item["schema_version"]) is int
        and item["schema_version"] == MORRIS_OBSERVATION_SCHEMA_VERSION,
        "unsupported schema version",
    )
    archive_digest = sha256_hex(item["archive_sha256"], "archive_sha256")
    unsigned = {key: item[key] for key in item if key != "archive_sha256"}
    require(
        archive_digest == hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
        "archive digest mismatch",
    )
    design = _parse_design(item["design"])
    design_digest = sha256_hex(item["design_sha256"], "design_sha256")
    require(design_digest == _design_sha256(design), "design digest mismatch")
    raw_outputs = item["outputs"]
    require(
        isinstance(raw_outputs, list) and bool(raw_outputs), "outputs must be nonempty"
    )
    sample_count = design.trajectories * (len(design.factors) + 1)
    _require_observation_shape(design, len(raw_outputs))
    outputs = tuple(_parse_output(output) for output in raw_outputs)
    records = item["records"]
    require(
        isinstance(records, list) and len(records) == sample_count,
        "records must cover design",
    )
    shape = (design.trajectories, len(design.factors) + 1)
    values = np.full(shape + (len(outputs),), np.nan, dtype=float)
    outcomes: np.ndarray = np.empty(shape, dtype=object)
    types: np.ndarray = np.full(shape, None, dtype=object)
    messages: np.ndarray = np.full(shape, None, dtype=object)
    physical = design.physical_points
    for ordinal, raw_record in enumerate(records):
        record = exact_mapping(raw_record, _RECORD_FIELDS, "record")
        trajectory, point = divmod(ordinal, shape[1])
        identity = (
            nonnegative_integer(record["ordinal"], "ordinal"),
            nonnegative_integer(record["trajectory_index"], "trajectory_index"),
            nonnegative_integer(record["point_index"], "point_index"),
        )
        require(
            identity == (ordinal, trajectory, point), "record identity is not canonical"
        )
        require(
            record["sample_id"] == _sample_id(design_digest, ordinal),
            "sample ID mismatch",
        )
        raw_physical = record["physical_values"]
        require(
            isinstance(raw_physical, list) and len(raw_physical) == len(design.factors),
            "physical values mismatch",
        )
        for index, (raw_entry, factor) in enumerate(
            zip(raw_physical, design.factors, strict=True)
        ):
            entry = exact_mapping(raw_entry, _PHYSICAL_FIELDS, "physical value")
            require(
                (entry["spec_id"], entry["variable_key"], entry["unit"])
                == (factor.spec_id, factor.variable_key, factor.unit),
                "factor identity mismatch",
            )
            require(
                finite(entry["value"], "physical value")
                == float(physical[trajectory, point, index]),
                "physical value mismatch",
            )
        raw_values = record["outputs"]
        require(
            isinstance(raw_values, list) and len(raw_values) == len(outputs),
            "output values mismatch",
        )
        for index, (raw_entry, output) in enumerate(
            zip(raw_values, outputs, strict=True)
        ):
            entry = exact_mapping(raw_entry, _VALUE_FIELDS, "record output")
            require(
                (entry["name"], entry["unit"]) == (output.name, output.unit),
                "output identity mismatch",
            )
            parsed = nullable_finite(entry["value"], "output value")
            values[trajectory, point, index] = np.nan if parsed is None else parsed
        outcomes[trajectory, point] = stable_text(record["status"], "status")
        types[trajectory, point] = optional_text(record["failure_type"], "failure type")
        messages[trajectory, point] = optional_text(
            record["failure_message"], "failure message"
        )
    observations = MorrisObservations(
        design, outputs, values, outcomes, types, messages
    )
    archive = MorrisObservationArchive(
        stable_text(item["study_id"], "study_id"),
        design_digest,
        provenance_mapping(item["provenance"]),
        observations,
    )
    require(
        morris_observations_to_json_dict(
            observations, study_id=archive.study_id, provenance=archive.provenance
        )
        == dict(item),
        "archive is not canonical",
    )
    return archive


__all__ = ["parse_morris_observations"]

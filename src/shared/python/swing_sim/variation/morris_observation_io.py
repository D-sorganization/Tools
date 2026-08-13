"""Strict lossless archive for raw Morris design-point observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from shared.python.contracts import require

from ._morris_observation_validation import (
    provenance_mapping as _provenance,
)
from ._morris_observation_validation import (
    sha256_hex as _sha256,
)
from ._morris_observation_validation import (
    stable_text as _stable_text,
)
from .morris_design import MorrisDesign, MorrisFactor, MorrisObservations, MorrisOutput
from .morris_execution import MAX_MORRIS_OBSERVATION_CELLS, MAX_MORRIS_SAMPLES

MORRIS_OBSERVATION_SCHEMA_ID = "swing-sim/morris-observation-archive"
MORRIS_OBSERVATION_SCHEMA_VERSION = 1

_ROOT_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "study_id",
        "design_sha256",
        "archive_sha256",
        "provenance",
        "design",
        "outputs",
        "records",
    }
)
_DESIGN_FIELDS = frozenset(
    {
        "factors",
        "trajectories",
        "levels",
        "seed",
        "normalized_points",
        "changed_factor_indices",
        "signed_steps",
    }
)
_FACTOR_FIELDS = frozenset(
    {
        "spec_id",
        "variable_key",
        "lower",
        "upper",
        "unit",
        "source_time_window_s",
        "source_point_ids",
    }
)
_OUTPUT_FIELDS = frozenset(
    {
        "name",
        "unit",
        "target_kind",
        "target_time_s",
        "target_point_id",
        "coordinate_frame",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "sample_id",
        "ordinal",
        "trajectory_index",
        "point_index",
        "status",
        "physical_values",
        "outputs",
        "failure_type",
        "failure_message",
    }
)
_PHYSICAL_FIELDS = frozenset({"spec_id", "variable_key", "unit", "value"})
_VALUE_FIELDS = frozenset({"name", "unit", "value"})
_MAX_ARCHIVE_NODES = 10_000_000
_MAX_ARCHIVE_TEXT_BYTES = 64_000_000


@dataclass(frozen=True)
class MorrisObservationArchive:
    """Parsed raw archive with immutable observations and provenance."""

    study_id: str
    design_sha256: str
    provenance: Mapping[str, str]
    observations: MorrisObservations

    def __post_init__(self) -> None:
        _stable_text(self.study_id, "study_id")
        _sha256(self.design_sha256, "design_sha256")
        provenance = _provenance(self.provenance)
        require(
            isinstance(self.observations, MorrisObservations),
            "observations must be MorrisObservations",
        )
        _require_observation_shape(
            self.observations.design, len(self.observations.outputs)
        )
        _require_hit_availability(self.observations)
        failure_types = cast(np.ndarray, self.observations.failure_types)
        failure_messages = cast(np.ndarray, self.observations.failure_messages)
        failures = self.observations.outcomes == "numerical_failure"
        require(
            np.all(failure_types[failures] != None)  # noqa: E711
            and np.all(failure_messages[failures] != None),  # noqa: E711
            "raw numerical failures require type and message diagnostics",
        )
        for value in failure_types[failures]:
            _stable_text(value, "failure type")
        for value in failure_messages[failures]:
            _stable_text(value, "failure message")
        expected = _design_sha256(self.observations.design)
        require(
            self.design_sha256 == expected,
            "design_sha256 must bind the exact design",
            (self.design_sha256, expected),
        )
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    @property
    def observation_cells(self) -> int:
        """Return weighted 8-byte retention units for arrays and bounded text."""
        design = self.observations.design
        samples = design.trajectories * (len(design.factors) + 1)
        numeric = samples * (len(design.factors) + len(self.observations.outputs))
        diagnostic_arrays = (
            cast(np.ndarray, self.observations.failure_types),
            cast(np.ndarray, self.observations.failure_messages),
        )
        diagnostic_bytes = 0
        for array in diagnostic_arrays:
            for raw_value in cast(list[object], array.ravel().tolist()):
                if raw_value is not None:
                    diagnostic_bytes += len(cast(str, raw_value).encode("utf-8"))
        text_bytes = diagnostic_bytes + sum(
            len(key.encode("utf-8")) + len(value.encode("utf-8"))
            for key, value in self.provenance.items()
        )
        return int(numeric + (text_bytes + 7) // 8)


def _require_hit_availability(observations: MorrisObservations) -> None:
    """Require every declared downstream result for each evaluated hit."""
    hits = observations.outcomes == "evaluated_hit"
    downstream = np.asarray(
        [
            output.target_kind in ("impact", "shot-outcome")
            for output in observations.outputs
        ],
        dtype=bool,
    )
    if np.any(hits) and np.any(downstream):
        require(
            np.all(np.isfinite(observations.values[hits][:, downstream])),
            "raw hit observations require every impact and shot output",
        )


def _require_observation_shape(design: MorrisDesign, output_count: int) -> None:
    """Enforce shared Morris allocation limits before archive materialization."""
    _require_observation_counts(design.trajectories, len(design.factors), output_count)


def _require_observation_counts(
    trajectories: int, factor_count: int, output_count: int
) -> None:
    """Validate allocation products from scalar document dimensions."""
    sample_count = trajectories * (factor_count + 1)
    require(
        sample_count <= MAX_MORRIS_SAMPLES,
        f"sample count must not exceed MAX_MORRIS_SAMPLES={MAX_MORRIS_SAMPLES}",
        sample_count,
    )
    observation_cells = sample_count * output_count
    require(
        observation_cells <= MAX_MORRIS_OBSERVATION_CELLS,
        "observation count must not exceed "
        f"MAX_MORRIS_OBSERVATION_CELLS={MAX_MORRIS_OBSERVATION_CELLS}",
        observation_cells,
    )


def _factor_document(factor: MorrisFactor) -> dict[str, Any]:
    _stable_text(factor.spec_id, "factor spec_id")
    _stable_text(factor.variable_key, "factor variable_key")
    _stable_text(factor.unit, "factor unit")
    for point_id in factor.source_point_ids:
        _stable_text(point_id, "source point ID")
    return {
        "spec_id": factor.spec_id,
        "variable_key": factor.variable_key,
        "lower": factor.lower,
        "upper": factor.upper,
        "unit": factor.unit,
        "source_time_window_s": None
        if factor.source_time_window_s is None
        else list(factor.source_time_window_s),
        "source_point_ids": list(factor.source_point_ids),
    }


def _output_document(output: MorrisOutput) -> dict[str, Any]:
    _stable_text(output.name, "output name")
    _stable_text(output.unit, "output unit")
    _stable_text(output.target_kind, "target kind")
    if output.target_point_id is not None:
        _stable_text(output.target_point_id, "target point ID")
    if output.coordinate_frame is not None:
        _stable_text(output.coordinate_frame, "coordinate frame")
    return {
        "name": output.name,
        "unit": output.unit,
        "target_kind": output.target_kind,
        "target_time_s": output.target_time_s,
        "target_point_id": output.target_point_id,
        "coordinate_frame": output.coordinate_frame,
    }


def _design_document(design: MorrisDesign) -> dict[str, Any]:
    return {
        "factors": [_factor_document(factor) for factor in design.factors],
        "trajectories": design.trajectories,
        "levels": design.levels,
        "seed": design.seed,
        "normalized_points": design.normalized_points.tolist(),
        "changed_factor_indices": design.changed_factor_indices.tolist(),
        "signed_steps": design.signed_steps.tolist(),
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _design_sha256(design: MorrisDesign) -> str:
    return hashlib.sha256(_canonical_bytes(_design_document(design))).hexdigest()


def morris_design_sha256(design: MorrisDesign) -> str:
    """Return the canonical digest for one validated Morris design."""
    require(isinstance(design, MorrisDesign), "design must be MorrisDesign")
    return _design_sha256(design)


def _sample_id(design_sha256: str, ordinal: int) -> str:
    return hashlib.sha256(f"{design_sha256}:{ordinal}".encode()).hexdigest()


def make_morris_observation_archive(
    observations: MorrisObservations,
    *,
    study_id: str,
    provenance: Mapping[str, str],
) -> MorrisObservationArchive:
    """Bind immutable observations to stable study and design provenance."""
    require(
        isinstance(observations, MorrisObservations),
        "observations must be MorrisObservations",
    )
    _require_observation_shape(observations.design, len(observations.outputs))
    return MorrisObservationArchive(
        _stable_text(study_id, "study_id"),
        _design_sha256(observations.design),
        _provenance(provenance),
        observations,
    )


def morris_observations_to_json_dict(
    observations: MorrisObservations,
    *,
    study_id: str,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    """Serialize every physical point, output, status, and diagnostic exactly."""
    require(
        isinstance(observations, MorrisObservations),
        "observations must be MorrisObservations",
    )
    _require_observation_shape(observations.design, len(observations.outputs))
    _require_hit_availability(observations)
    stable_study_id = _stable_text(study_id, "study_id")
    stable_provenance = _provenance(provenance)
    design = observations.design
    design_sha256 = _design_sha256(design)
    physical = design.physical_points
    failure_types = cast(np.ndarray, observations.failure_types)
    failure_messages = cast(np.ndarray, observations.failure_messages)
    records: list[dict[str, Any]] = []
    points_per_trajectory = len(design.factors) + 1
    for ordinal in range(design.trajectories * points_per_trajectory):
        trajectory, point = divmod(ordinal, points_per_trajectory)
        status = str(observations.outcomes[trajectory, point])
        if status == "numerical_failure":
            require(
                failure_types[trajectory, point] is not None
                and failure_messages[trajectory, point] is not None,
                "raw numerical failures require type and message diagnostics",
                ordinal,
            )
        records.append(
            {
                "sample_id": _sample_id(design_sha256, ordinal),
                "ordinal": ordinal,
                "trajectory_index": trajectory,
                "point_index": point,
                "status": status,
                "physical_values": [
                    {
                        "spec_id": factor.spec_id,
                        "variable_key": factor.variable_key,
                        "unit": factor.unit,
                        "value": float(physical[trajectory, point, index]),
                    }
                    for index, factor in enumerate(design.factors)
                ],
                "outputs": [
                    {
                        "name": output.name,
                        "unit": output.unit,
                        "value": None
                        if np.isnan(observations.values[trajectory, point, index])
                        else float(observations.values[trajectory, point, index]),
                    }
                    for index, output in enumerate(observations.outputs)
                ],
                "failure_type": failure_types[trajectory, point],
                "failure_message": failure_messages[trajectory, point],
            }
        )
    document = {
        "schema_id": MORRIS_OBSERVATION_SCHEMA_ID,
        "schema_version": MORRIS_OBSERVATION_SCHEMA_VERSION,
        "study_id": stable_study_id,
        "design_sha256": design_sha256,
        "provenance": stable_provenance,
        "design": _design_document(design),
        "outputs": [_output_document(output) for output in observations.outputs],
        "records": records,
    }
    return {
        **document,
        "archive_sha256": hashlib.sha256(_canonical_bytes(document)).hexdigest(),
    }


def morris_observations_from_json_dict(value: object) -> MorrisObservationArchive:
    """Parse an exact archive and reject crossed identities or fabricated values."""
    from ._morris_observation_parser import parse_morris_observations

    return parse_morris_observations(value)


__all__ = [
    "MORRIS_OBSERVATION_SCHEMA_ID",
    "MORRIS_OBSERVATION_SCHEMA_VERSION",
    "MorrisObservationArchive",
    "make_morris_observation_archive",
    "morris_design_sha256",
    "morris_observations_from_json_dict",
    "morris_observations_to_json_dict",
]

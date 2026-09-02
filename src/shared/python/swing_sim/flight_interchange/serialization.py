"""JSON codec for ``swing_sim.ball_flight_trajectory/1`` (ADR-0047 H1).

The wire's reader and writer, split from the record types in
:mod:`.trajectory` only for file size — the contract they implement is
documented there and is the single authority. Posture is the package
idiom: sorted keys, compact separators, ``allow_nan=False``, unknown
fields refused, missing fields refused, and every value re-validated
through the same constructors a locally built record passes, so a
parsed record can never be weaker than one built in process.
"""

from __future__ import annotations

import json
from typing import Any, cast

from shared.python.contracts import require

from .trajectory import (
    BALL_FLIGHT_TRAJECTORY_FORMAT,
    OPTIONAL_CHANNELS,
    BallFlightSample,
    BallFlightTrajectory,
    TrajectoryProvenance,
    _finite_triplet,
    _identifier,
)

_SAMPLE_REQUIRED = ("position_m", "time_s")
_TRAJECTORY_FIELDS = frozenset(
    {"channels", "format", "frame_id", "provenance", "samples", "source_id"}
)
_PROVENANCE_FIELDS = frozenset({"model_family", "model_name", "parameter_digest"})

__all__ = [
    "ball_flight_trajectory_from_json",
    "ball_flight_trajectory_to_json",
]


def _sample_payload(sample: BallFlightSample) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "time_s": sample.time_s,
        "position_m": list(sample.position_m),
    }
    for name in sample.channels:
        payload[name] = list(getattr(sample, name))
    return payload


def ball_flight_trajectory_to_json(trajectory: BallFlightTrajectory) -> str:
    """Serialize deterministically; identical records are byte-identical.

    Sorted keys, compact separators, and ``allow_nan=False`` — the
    package idiom. Float formatting is runtime-local (Python ``repr``
    and JavaScript's shortest round-trip disagree on integral floats),
    so cross-runtime interchange is by JSON *value*, not by bytes.
    """
    require(
        isinstance(trajectory, BallFlightTrajectory),
        "trajectory must be BallFlightTrajectory",
    )
    payload: dict[str, Any] = {
        "format": BALL_FLIGHT_TRAJECTORY_FORMAT,
        "source_id": trajectory.source_id,
        "frame_id": trajectory.frame_id,
        "channels": list(trajectory.channels),
        "provenance": {
            "model_family": trajectory.provenance.model_family,
            "model_name": trajectory.provenance.model_name,
            "parameter_digest": trajectory.provenance.parameter_digest,
        },
        "samples": [_sample_payload(sample) for sample in trajectory.samples],
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _provenance_from_payload(data: object) -> TrajectoryProvenance:
    require(isinstance(data, dict), "provenance must be an object and is mandatory")
    section: dict[str, Any] = data  # type: ignore[assignment]
    require(
        set(section) == _PROVENANCE_FIELDS,
        f"provenance fields must be exactly {sorted(_PROVENANCE_FIELDS)}",
        sorted(section),
    )
    return TrajectoryProvenance(
        model_family=_identifier(section["model_family"], "model_family"),
        model_name=_identifier(section["model_name"], "model_name"),
        parameter_digest=section["parameter_digest"],
    )


def _declared_channels(data: object) -> tuple[str, ...]:
    require(isinstance(data, list), "channels must be a list")
    names: tuple[Any, ...] = tuple(cast("list[Any]", data))
    for name in names:
        require(name in OPTIONAL_CHANNELS, f"unknown channel: {name!r}")
    require(
        list(names) == sorted(set(names)),
        "channels must be sorted and free of duplicates",
        list(names),
    )
    return tuple(str(name) for name in names)


def ball_flight_trajectory_from_json(text: str) -> BallFlightTrajectory:
    """Parse and validate; unknown fields and wrong formats are refused.

    Raises:
        ContractViolationError: On any wire violation — an unknown
            top-level or per-sample field, a missing or malformed
            provenance, an undeclared frame, non-monotone times, a
            sample whose keys disagree with ``channels``, or a
            non-finite value.
    """
    require(isinstance(text, str), "text must be str")
    data = json.loads(text)
    require(isinstance(data, dict), "ball flight trajectory must be an object")
    unknown = set(data) - _TRAJECTORY_FIELDS
    require(not unknown, f"unknown trajectory fields: {sorted(unknown)}")
    missing = _TRAJECTORY_FIELDS - set(data)
    require(not missing, f"missing trajectory fields: {sorted(missing)}")
    require(
        data["format"] == BALL_FLIGHT_TRAJECTORY_FORMAT,
        f"format must be {BALL_FLIGHT_TRAJECTORY_FORMAT!r}",
        data["format"],
    )
    channels = _declared_channels(data["channels"])
    expected_keys = set(_SAMPLE_REQUIRED) | set(channels)
    raw_samples = data["samples"]
    require(isinstance(raw_samples, list), "samples must be a list")
    samples = []
    for raw in raw_samples:
        require(isinstance(raw, dict), "each sample must be an object")
        require(
            set(raw) == expected_keys,
            f"sample fields must be exactly {sorted(expected_keys)}",
            sorted(raw),
        )
        samples.append(
            BallFlightSample(
                time_s=raw["time_s"],
                position_m=_finite_triplet(raw["position_m"], "position_m"),
                velocity_mps=(
                    _finite_triplet(raw["velocity_mps"], "velocity_mps")
                    if "velocity_mps" in channels
                    else None
                ),
                spin_rad_s=(
                    _finite_triplet(raw["spin_rad_s"], "spin_rad_s")
                    if "spin_rad_s" in channels
                    else None
                ),
            )
        )
    return BallFlightTrajectory(
        source_id=_identifier(data["source_id"], "source_id"),
        frame_id=data["frame_id"],
        provenance=_provenance_from_payload(data["provenance"]),
        samples=tuple(samples),
    )

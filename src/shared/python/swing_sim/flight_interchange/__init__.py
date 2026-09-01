"""Neutral ball-flight trajectory interchange (ADR-0047 H1, UD #9350).

The seam between the fleet's two independent flight-model families —
UpstreamDrift's named published models and this repo's
:mod:`shared.python.swing_sim.flight` — and every viewer that draws a
trajectory. Both families keep their own physics and their own name;
what they gain is one versioned, fail-closed, byte-deterministic export
format, so a curve from either can be replayed or overlaid beside the
other *because it is labelled*.

See :mod:`.trajectory` for the wire itself (frames, units, mandatory
provenance, and the :func:`~.trajectory.from_samples` producer seam),
:mod:`.serialization` for its JSON codec, and :mod:`.adapters` for the
Tools-side exporter.
"""

from .adapters import (
    TOOLS_FLIGHT_FAMILY,
    flight_model_parameters,
    trajectory_from_flight_result,
)
from .serialization import (
    ball_flight_trajectory_from_json,
    ball_flight_trajectory_to_json,
)
from .trajectory import (
    APP_FRAME_ID,
    BALL_FLIGHT_TRAJECTORY_FORMAT,
    FLIGHT_FRAME_ID,
    FRAME_IDS,
    OPTIONAL_CHANNELS,
    BallFlightSample,
    BallFlightTrajectory,
    TrajectoryProvenance,
    from_samples,
    parameter_digest,
)

__all__ = [
    "APP_FRAME_ID",
    "BALL_FLIGHT_TRAJECTORY_FORMAT",
    "FLIGHT_FRAME_ID",
    "FRAME_IDS",
    "OPTIONAL_CHANNELS",
    "TOOLS_FLIGHT_FAMILY",
    "BallFlightSample",
    "BallFlightTrajectory",
    "TrajectoryProvenance",
    "ball_flight_trajectory_from_json",
    "ball_flight_trajectory_to_json",
    "flight_model_parameters",
    "from_samples",
    "parameter_digest",
    "trajectory_from_flight_result",
]

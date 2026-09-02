"""Runtime-neutral lift of an imported trajectory record onto the shared
playback timeline (ADR-0047 H4, UD #9353).

The Impact Explorer's 3D playback already derives its frames from
retained samples (#4800 P8) regardless of who produced them; this
module is the missing seam that lets it replay a
``swing_sim.ball_flight_trajectory/1`` record (ADR-0047 H1,
:mod:`shared.python.swing_sim.flight_interchange`) — whether the record
came from this repo's ``swing_sim.flight`` or from UpstreamDrift's
named published models — without re-simulating or resampling a single
value.

It lives beside :mod:`~rate_of_closure.simulation.flight_playback` and
:mod:`~rate_of_closure.simulation.putt_playback`, its two siblings in
the same seam: read a validated result, lift its retained samples onto
the shared app-frame timeline, change nothing else. The one thing this
loader owns that the others do not is frame conversion — an imported
record may declare either of the two frames the wire allows, so the
conversion is explicit and total over that closed set
(:data:`~shared.python.swing_sim.flight_interchange.FRAME_IDS`); a
frame this loader does not recognize is refused by
:class:`UnsupportedTrajectoryFrameError` rather than silently drawn
un-converted, which would mirror or rotate the imported shot.

The TypeScript twin of the frame-conversion mapping (the part P8 pins
cross-runtime) is
``web/src/model/flightRecordPlayback.ts``; both are pinned by the
``imported_trajectory`` block of the shared golden fixture
``web/src/model/__fixtures__/playback_transport_golden_v1.json``. The
wire codec itself stays Python-only per
:mod:`shared.python.swing_sim.flight_interchange.trajectory` — this
module never re-implements it, only consumes the validated record.

Transport and camera state live elsewhere by design — the timeline
math is :mod:`~rate_of_closure.simulation.playback_transport` and
camera state belongs to #4571.
"""

from __future__ import annotations

import numpy as np

from rate_of_closure.simulation.flight_playback import TimedTrajectory
from shared.python.swing_sim.flight.frames import from_flight_frame
from shared.python.swing_sim.flight_interchange import (
    APP_FRAME_ID,
    FLIGHT_FRAME_ID,
    BallFlightTrajectory,
)

__all__ = [
    "UnsupportedTrajectoryFrameError",
    "timed_trajectory_from_ball_flight_record",
]


class UnsupportedTrajectoryFrameError(ValueError):
    """Raised when a record declares a frame this loader cannot place.

    The wire's ``frame_id`` is a closed enum
    (:data:`~shared.python.swing_sim.flight_interchange.FRAME_IDS`), but
    this loader converts explicitly rather than defaulting, so a future
    frame added to the wire is refused here — loudly — until this
    module is taught the new conversion, instead of silently drawing an
    unconverted (and likely mirrored or rotated) trajectory.
    """

    def __init__(self, frame_id: str) -> None:
        super().__init__(
            f"unsupported ball_flight_trajectory frame_id: {frame_id!r}; "
            "this playback loader converts only "
            f"{FLIGHT_FRAME_ID!r} and {APP_FRAME_ID!r}"
        )
        self.frame_id = frame_id


def timed_trajectory_from_ball_flight_record(
    record: BallFlightTrajectory,
) -> TimedTrajectory:
    """Lift one imported ``ball_flight_trajectory/1`` record to playback.

    Args:
        record: A validated record from either flight-model family
            (:mod:`shared.python.swing_sim.flight_interchange`). Its
            samples are replayed exactly — never re-simulated or
            resampled — the same posture P8 already holds for
            solver-produced flights and putts.

    Returns:
        A :class:`TimedTrajectory` of ``(x, y, z)`` app-frame ball
        positions [m] at the record's retained sample times: ``x``
        target (downrange), ``y`` up, ``z`` right.

    Raises:
        TypeError: If ``record`` is not a :class:`BallFlightTrajectory`.
        UnsupportedTrajectoryFrameError: If ``record.frame_id`` is not
            one of the two frames this loader converts.
    """
    if not isinstance(record, BallFlightTrajectory):
        raise TypeError("record must be a BallFlightTrajectory")
    positions = np.array([sample.position_m for sample in record.samples], dtype=float)
    if record.frame_id == APP_FRAME_ID:
        positions_m = positions
    elif record.frame_id == FLIGHT_FRAME_ID:
        positions_m = from_flight_frame(positions)
    else:
        raise UnsupportedTrajectoryFrameError(record.frame_id)
    times_s = np.array([sample.time_s for sample in record.samples], dtype=float)
    return TimedTrajectory(times_s=times_s, positions_m=positions_m)

"""Runtime-neutral lift of a recorded putt onto the shared playback timeline.

The putting half of #4800 P8. Playback frames must come from the
retained integrator samples, never from re-simulation, so this module
does exactly one thing: it reads the ``PuttResult`` the tab already
accepted and the ``GreenSurface`` that result was integrated on, and
returns the ``TimedTrajectory`` every playback surface already knows how
to interpolate. Nothing is re-integrated and nothing is resampled.

It lives beside :mod:`~rate_of_closure.simulation.flight_playback`
rather than inside a Qt widget because the sample→frame mapping is where
P8 pins twin parity: the TypeScript twin is
``web/src/model/puttPlayback.ts`` and both are pinned by the ``putt``
block of the shared golden fixture
``web/src/model/__fixtures__/playback_transport_golden_v1.json``.

Transport and camera state live elsewhere by design — the timeline math
is :mod:`~rate_of_closure.simulation.playback_transport` and camera
state belongs to #4571.
"""

from __future__ import annotations

import numpy as np

from rate_of_closure.simulation.flight_playback import TimedTrajectory
from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M
from shared.python.swing_sim.putting import GreenSurface, PuttResult

__all__ = ["putt_playback_trajectory"]


def putt_playback_trajectory(
    result: PuttResult, surface: GreenSurface
) -> TimedTrajectory:
    """Lift one integrated putt to the shared playback trajectory.

    Args:
        result: The integrated putt whose retained samples are replayed.
        surface: The exact green the putt was integrated on; elevations
            are read from it so the ball rides the drawn surface.

    Returns:
        A :class:`TimedTrajectory` of ``(x, y, z)`` ball-centre
        positions [m] at the recorded sample times, where ``x`` is the
        target line, ``y`` is lateral (left positive) and ``z`` is
        elevation.

    Raises:
        TypeError: If ``result`` is not a :class:`PuttResult`.
        ValueError: If the retained samples are not a valid timeline.
    """
    if not isinstance(result, PuttResult):
        raise TypeError("result must be a PuttResult")
    heights = [
        surface.height_m(x_m, y_m) + GOLF_BALL_RADIUS_M
        for x_m, y_m in zip(result.path_x_m, result.path_y_m, strict=True)
    ]
    positions = np.column_stack(
        (
            np.asarray(result.path_x_m, dtype=float),
            np.asarray(result.path_y_m, dtype=float),
            np.asarray(heights, dtype=float),
        )
    )
    return TimedTrajectory(
        times_s=np.asarray(result.times_s, dtype=float), positions_m=positions
    )

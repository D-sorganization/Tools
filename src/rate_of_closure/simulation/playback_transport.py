"""Runtime-neutral playback transport math for 3D shot playback (#4800 P8).

One implementation of the timeline transport semantics — time
normalization, scrub-index quantization, and wall-clock advance under a
speed multiplier — consumed by every playback surface. The TypeScript
twin is ``web/src/model/playbackTransport.ts``; both are pinned by the
shared golden fixture
``web/src/model/__fixtures__/playback_transport_golden_v1.json``.

Trajectory-source independence (the putting seam): nothing in this
module knows about ball flight. It operates on physical seconds only, so
the putting vertical (#4800 P6/P7) drives the same functions with
putt-result timelines unchanged.

Camera seam (#4571): camera state belongs to
``rate_of_closure.application.camera_commands`` and its viewport mixins;
this module deliberately owns only timeline math and must never grow
camera responsibilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Canonical speed multipliers offered by every playback surface.
PLAYBACK_SPEEDS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)

#: Real-time playback rate; both surfaces default their selector to this.
DEFAULT_SPEED: float = 1.0

#: Scrub-slider quantization shared by the Qt and React timelines.
SCRUB_STEPS: int = 10_000


@dataclass(frozen=True)
class PlaybackAdvance:
    """Result of advancing playback by one wall-clock interval."""

    time_s: float
    finished: bool


def _require_duration(duration_s: float) -> float:
    if not math.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("duration_s must be finite and >= 0")
    return float(duration_s)


def _require_steps(steps: int) -> int:
    if steps <= 0:
        raise ValueError("steps must be a positive integer")
    return int(steps)


def clamp_time(time_s: float, duration_s: float) -> float:
    """Normalize a finite requested time onto the ``[0, duration]`` timeline."""
    if not math.isfinite(time_s):
        raise ValueError("time_s must be finite")
    duration = _require_duration(duration_s)
    return min(max(float(time_s), 0.0), duration)


def scrub_value(time_s: float, duration_s: float, steps: int = SCRUB_STEPS) -> int:
    """Quantize a physical time to its integer scrub-slider position.

    Half-up rounding (``floor(x + 0.5)``) so both runtime twins agree at
    exact half-steps; an empty timeline always maps to position zero.
    """
    quantum = _require_steps(steps)
    duration = _require_duration(duration_s)
    time = clamp_time(time_s, duration)
    if duration <= 0.0:
        return 0
    return int(math.floor(quantum * (time / duration) + 0.5))


def time_at_scrub(value: int, duration_s: float, steps: int = SCRUB_STEPS) -> float:
    """Physical time for an integer scrub-slider position in ``[0, steps]``."""
    quantum = _require_steps(steps)
    duration = _require_duration(duration_s)
    if not 0 <= int(value) <= quantum:
        raise ValueError("value must lie within [0, steps]")
    return duration * (int(value) / quantum)


def advance_playback(
    time_s: float, elapsed_s: float, speed: float, duration_s: float
) -> PlaybackAdvance:
    """Advance playback by an elapsed wall-clock interval at a speed multiplier.

    Physical timestamps are never altered — the multiplier scales only
    the wall-clock interval. The result clamps at the timeline end and
    reports ``finished`` so callers stop their timers identically.
    """
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ValueError("elapsed_s must be finite and >= 0")
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("speed must be finite and > 0")
    duration = _require_duration(duration_s)
    time = clamp_time(time_s, duration)
    advanced = min(duration, time + elapsed_s * speed)
    return PlaybackAdvance(time_s=advanced, finished=advanced >= duration)


__all__ = [
    "DEFAULT_SPEED",
    "PLAYBACK_SPEEDS",
    "SCRUB_STEPS",
    "PlaybackAdvance",
    "advance_playback",
    "clamp_time",
    "scrub_value",
    "time_at_scrub",
]

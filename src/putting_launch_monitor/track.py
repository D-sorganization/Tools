"""From a stream of ball observations to a putt.

:class:`PuttTracker` is a small state machine fed one observation per frame
(``None`` when the ball was not seen):

    WAITING ─(ball still for ``settle_frames``)─▶ ARMED
    ARMED   ─(ball moves past ``start_mm``)──────▶ ROLLING
    ROLLING ─(travelled ``window_mm``, left the view, or stopped)─▶ putt out,
            back to WAITING

The ball must be seen resting first, so a hand placing it, a club head or
a second ball rolling through never counts as a putt. Positions are
converted to the ground plane before any decision, so every threshold is
in millimetres and means the same thing anywhere in the frame. The launch
is fitted by :func:`geometry.fit_launch` over the opening window and
reported with its quality, and the caller decides what is good enough.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from shared.python.contracts import require

from .detect import BallObservation
from .geometry import GroundPlane, Launch, fit_launch, hla_degrees

NS_PER_S = 1_000_000_000


class Phase(StrEnum):
    WAITING = "waiting"
    ARMED = "armed"
    ROLLING = "rolling"


@dataclass(frozen=True)
class TrackerSettings:
    """Thresholds, all in ground-plane millimetres or frames.

    Invariants: positive distances; ``settle_frames >= 2``; the launch
    window is longer than the start threshold.
    """

    settle_frames: int = 12  # ~0.2 s at 60 fps of a resting ball
    still_mm: float = 4.0  # jitter allowed while "resting"
    start_mm: float = 12.0  # movement that counts as the putt starting
    window_mm: float = 300.0  # launch fit over this much travel
    max_missing: int = 6  # consecutive unseen frames before giving up
    max_jump_mm: float = 250.0  # farthest our ball can move between frames
    min_points: int = 4
    min_r2: float = 0.98
    min_speed_mph: float = 0.3
    max_speed_mph: float = 25.0
    max_hla_deg: float = 45.0
    target: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        require(self.settle_frames >= 2, "settle_frames", self.settle_frames)
        require(0 < self.still_mm < self.start_mm, "still < start", self)
        require(self.window_mm > self.start_mm, "window > start", self)
        require(self.max_missing >= 1, "max_missing", self.max_missing)
        require(self.max_jump_mm > self.start_mm, "max_jump > start", self)
        require(self.min_points >= 3, "min_points", self.min_points)
        require(0.0 <= self.min_r2 <= 1.0, "min_r2", self.min_r2)
        require(0 < self.min_speed_mph < self.max_speed_mph, "speed bounds", self)
        require(0 < self.max_hla_deg <= 90, "max_hla_deg", self.max_hla_deg)


@dataclass(frozen=True)
class Putt:
    """A detected putt with its launch fit and the reason it was, or was
    not, accepted. ``accepted`` is the tracker's verdict against the
    settings; ``reason`` explains a rejection.
    """

    launch: Launch
    hla_deg: float
    start_mm: tuple[float, float]
    accepted: bool
    reason: str = ""

    @property
    def speed_mph(self) -> float:
        return self.launch.speed_mph


@dataclass
class PuttTracker:
    """Feed :meth:`update` once per frame; a :class:`Putt` comes back when done."""

    plane: GroundPlane
    settings: TrackerSettings = field(default_factory=TrackerSettings)
    phase: Phase = Phase.WAITING
    _rest: list[tuple[float, float]] = field(default_factory=list, repr=False)
    _origin: tuple[float, float] | None = field(default=None, repr=False)
    _times: list[float] = field(default_factory=list, repr=False)
    _points: list[tuple[float, float]] = field(default_factory=list, repr=False)
    _missing: int = field(default=0, repr=False)

    def reset(self) -> None:
        self.phase = Phase.WAITING
        self._rest, self._origin = [], None
        self._times, self._points, self._missing = [], [], 0

    def update(
        self, timestamp_ns: int, candidates: Sequence[BallObservation]
    ) -> Putt | None:
        """Advance one frame with every ball the detector saw (best first).

        Precondition: ``timestamp_ns >= 0``. With several balls on the mat the
        tracker follows *its* ball: while armed, the one at its resting spot;
        while rolling, the one nearest where it last was. A stray ball sitting
        elsewhere can neither arm nor capture the track.
        Postcondition: returns a :class:`Putt` exactly when a roll ends; the
        tracker is back in ``WAITING`` afterwards.
        """
        require(timestamp_ns >= 0, "timestamp", timestamp_ns)
        world = self._choose(candidates)
        if self.phase is Phase.ROLLING:
            return self._roll(timestamp_ns, world)
        if world is None:
            self._rest.clear()
            self.phase = Phase.WAITING
            return None
        self._settle_or_start(timestamp_ns, world)
        return None

    # -- phases ---------------------------------------------------------------------
    def _settle_or_start(self, t_ns: int, world: tuple[float, float]) -> None:
        s = self.settings
        if self.phase is Phase.ARMED and self._origin is not None:
            if _dist(world, self._origin) >= s.start_mm:
                self.phase = Phase.ROLLING
                self._times = [t_ns / NS_PER_S]
                self._points = [world]
                self._missing = 0
            return  # armed: still resting at the origin, or now rolling
        self._rest.append(world)
        if len(self._rest) > s.settle_frames:
            self._rest.pop(0)
        if len(self._rest) == s.settle_frames and _spread(self._rest) <= s.still_mm:
            self._origin = _mean(self._rest)
            self.phase = Phase.ARMED

    def _roll(self, t_ns: int, world: tuple[float, float] | None) -> Putt | None:
        s = self.settings
        if world is None:
            self._missing += 1
            if self._missing >= s.max_missing:
                return self._finish("ball left the view")
            return None
        self._missing = 0
        self._times.append(t_ns / NS_PER_S)
        self._points.append(world)
        if _dist(world, self._points[0]) >= s.window_mm:
            return self._finish("")
        if (
            len(self._points) >= s.settle_frames
            and _spread(self._points[-s.settle_frames :]) <= s.still_mm
        ):
            return self._finish("stopped inside the launch window")
        return None

    def _finish(self, note: str) -> Putt | None:
        times, points = self._times, self._points
        origin = self._origin or points[0]
        self.reset()
        if len(points) < self.settings.min_points:
            return None  # a twitch, not a putt
        launch = fit_launch(
            np.asarray(times),
            np.asarray(points),
            window_mm=self.settings.window_mm,
            min_points=self.settings.min_points,
        )
        hla = hla_degrees(launch.direction, self.settings.target)
        accepted, reason = self._judge(launch, hla, note)
        return Putt(
            launch=launch,
            hla_deg=hla,
            start_mm=origin,
            accepted=accepted,
            reason=reason,
        )

    def _judge(self, launch: Launch, hla: float, note: str) -> tuple[bool, str]:
        s = self.settings
        if note == "stopped inside the launch window":
            return False, note
        if launch.r2 < s.min_r2:
            return False, f"launch fit too noisy (r2={launch.r2:.3f})"
        if not (s.min_speed_mph <= launch.speed_mph <= s.max_speed_mph):
            return False, f"speed {launch.speed_mph:.2f} mph out of range"
        if abs(hla) > s.max_hla_deg:
            return False, f"HLA {hla:+.1f} deg out of range"
        return True, note

    # -- helpers ----------------------------------------------------------------------
    def _choose(
        self, candidates: Sequence[BallObservation]
    ) -> tuple[float, float] | None:
        """The ground position of the candidate that is *our* ball, if any."""
        worlds = [self._to_world(c) for c in candidates]
        if not worlds:
            return None
        anchor: tuple[float, float] | None = None
        if self.phase is Phase.ROLLING and self._points:
            anchor = self._points[-1]
        elif self.phase is Phase.ARMED and self._origin is not None:
            anchor = self._origin
        if anchor is None:
            return worlds[0]  # waiting: the most ball-like blob
        nearest = min(worlds, key=lambda w: _dist(w, anchor))
        limit = (
            self.settings.max_jump_mm
            if self.phase is Phase.ROLLING
            else self.settings.window_mm
        )
        return nearest if _dist(nearest, anchor) <= limit else None

    def _to_world(self, seen: BallObservation) -> tuple[float, float]:
        x, y = self.plane.to_world(np.array([seen.cx, seen.cy]))
        return (float(x), float(y))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _mean(points: list[tuple[float, float]]) -> tuple[float, float]:
    arr = np.asarray(points, dtype=np.float64)
    m = arr.mean(axis=0)
    return (float(m[0]), float(m[1]))


def _spread(points: list[tuple[float, float]]) -> float:
    """Largest distance of any point from the group's mean."""
    arr = np.asarray(points, dtype=np.float64)
    return float(np.max(np.linalg.norm(arr - arr.mean(axis=0), axis=1)))

"""Putting-stroke interchange: recorded strokes drive the impact solve (#4800 P4).

The seam between motion sources (Drake, MuJoCo, OpenSim, a putting lab,
or an OEM capture rig) and the putting pipeline, built as the sibling of
:mod:`shared.python.swing_sim.delivery_interchange`: one versioned
fail-closed wire, runtime-free engine adapters, and derivations feeding
:func:`~.impact.strike` — no invented conventions, no second copy of the
delivery machinery.

Frames (declared, not assumed)
------------------------------
The world frame is the AffineDrift frame the impact package documents:
``x`` = target line, ``y`` = up, ``z`` = right (right-handed). The
recorded **putter body frame** is the same convention
``delivery_interchange`` documents for the grip: ``+z`` along the shaft
toward the head, ``+x`` the face normal of a square face; right-handedness
then makes ``+y`` the toe direction and ``-z`` "up the face". A pose is a
unit quaternion ``(w, x, y, z)`` mapping body coordinates into world. The
recorded point is the exporter's body origin (a grip marker, a hosel
body, a MuJoCo site); ``body_to_face_m`` is the rigid distance from it
down ``+z`` to the face center — exactly the ``grip_to_head_m`` extension
``delivery_interchange`` applies, and every derivation routes through it.

Wire ``swing_sim.putting_stroke/1``
-----------------------------------
Pose-only (position + face orientation) — capture rigs report pose, not
velocity — plus the address geometry the impact solve needs: the ball
center in the declared frame and the aim line. Same posture as the
delivery wire: at least three samples (central differences need an
interior), strictly increasing timestamps, unit quaternions, finite SI
values, deterministic sorted-keys JSON, unknown fields refused.
:meth:`PuttingStroke.to_delivery_trajectory` lifts a stroke into the
neutral :class:`~..delivery_interchange.DeliveryTrajectory` by central
differences — the same deliberate v1 choice the ``.sto`` adapter makes
for position-only tables — with angular velocity from the exact rigid
identity ``omega = 2 * vec(qdot (x) q*)`` on sign-aligned quaternions.

Derivation (conventions reused, never invented)
-----------------------------------------------
Impact is the sample nearest face-ball contact: the signed face-normal
separation ``s = (p_ball - p_face) . n`` must start beyond the ball and
cross the ball radius, and the sample nearest ``s = R_ball`` is taken
(the discrete analogue of ``index_at_time``). There,
:func:`~..delivery_interchange.delivery_view_at` supplies head speed,
attack angle, club path, and face angle **verbatim** — same ``atan2``
expressions, same AffineDrift signs the full swing uses. This module only

* subtracts the declared ``aim_deg`` so face and path are measured off
  the aim line, which is how :func:`~.impact.strike` defines them (aim
  re-enters the start-line sum, so the absolute start azimuth is
  independent of it, as P1 documents);
* resolves the ball center in the face frame for the strike location:
  toe (+) along ``R.(0, 1, 0)``, high (+) along ``R.(0, 0, -1)``;
* reports the face pitch ``atan2(n_y, |n_xz|)`` as the delivered dynamic
  loft, so ``shaft_lean_deg`` is recovered, not guessed.

Runtime-free engine adapters live in :mod:`.stroke_adapters`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.python.contracts import ensure, require, require_finite
from shared.python.swing_sim.delivery_interchange import (
    DeliveryTrajectory,
    TrajectorySample,
    delivery_view_at,
    head_state_at,
)
from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M

from .green import CaptureModel, PuttResult, simulate_putt_on_surface
from .impact import PutterSpec, PuttLaunch, strike
from .roll import DEFAULT_SLIDING_MU
from .surface import GreenSurface

PUTTING_STROKE_FORMAT = "swing_sim.putting_stroke/1"

_SAMPLE_FIELDS = frozenset({"time_s", "position_m", "quaternion_wxyz"})
_STROKE_FIELDS = frozenset(
    {"format", "source_id", "frame_id", "ball_position_m", "aim_deg", "samples"}
)
_M_TO_MM = 1e3

__all__ = [
    "PUTTING_STROKE_FORMAT",
    "PuttingStroke",
    "StrokePutt",
    "StrokeSample",
    "StrokeStrike",
    "impact_sample_index",
    "putt_from_stroke",
    "putting_stroke_from_json",
    "putting_stroke_to_json",
    "strike_from_stroke",
    "strike_parameters",
]


def _triplet(row: Any) -> tuple[float, float, float]:
    """A float triple from any 3-element sequence."""
    return (float(row[0]), float(row[1]), float(row[2]))


@dataclass(frozen=True)
class StrokeSample:
    """One time-stamped putter-body pose in the declared world frame.

    Validated by the delivery wire itself: a zero-velocity
    ``TrajectorySample`` checks the triples and the unit quaternion.
    """

    time_s: float
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        checked = TrajectorySample(
            time_s=self.time_s,
            position_m=self.position_m,
            quaternion_wxyz=self.quaternion_wxyz,
            linear_velocity_mps=(0.0, 0.0, 0.0),
            angular_velocity_rad_s=(0.0, 0.0, 0.0),
        )
        object.__setattr__(self, "time_s", checked.time_s)
        object.__setattr__(self, "position_m", checked.position_m)
        object.__setattr__(self, "quaternion_wxyz", checked.quaternion_wxyz)


def _canonical_quaternions(samples: tuple[StrokeSample, ...]) -> np.ndarray:
    """Sign-align consecutive quaternions (``q`` and ``-q`` are one rotation)."""
    quats = np.asarray([item.quaternion_wxyz for item in samples], dtype=np.float64)
    for index in range(1, len(quats)):
        if float(quats[index] @ quats[index - 1]) < 0.0:
            quats[index] = -quats[index]
    return quats


def _angular_velocity(quats: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """World angular velocity ``2 * vec(qdot (x) q*)`` for unit ``q``."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    dw, dx, dy, dz = rates[:, 0], rates[:, 1], rates[:, 2], rates[:, 3]
    stacked: np.ndarray = np.stack(
        (
            -dw * x + dx * w - dy * z + dz * y,
            -dw * y + dx * z + dy * w - dz * x,
            -dw * z - dx * y + dy * x + dz * w,
        ),
        axis=1,
    )
    return 2.0 * stacked


@dataclass(frozen=True)
class PuttingStroke:
    """A validated putter-body stroke plus the address geometry.

    Attributes:
        source_id: Provenance tag (``"drake:putter_head"``, a rig name).
        frame_id: The declared world frame the samples live in.
        ball_position_m: Ball center in that frame [m].
        samples: At least three strictly time-ordered poses.
        aim_deg: Aim line relative to the frame's ``x`` axis [deg],
            + = right; face and path are reported off this line.
    """

    source_id: str
    frame_id: str
    ball_position_m: tuple[float, float, float]
    samples: tuple[StrokeSample, ...]
    aim_deg: float = 0.0

    def __post_init__(self) -> None:
        for name, text in (("source_id", self.source_id), ("frame_id", self.frame_id)):
            require(
                isinstance(text, str) and text.strip() == text and text != "",
                f"{name} must be a trimmed nonempty string",
            )
        object.__setattr__(
            self, "ball_position_m", _validated_triplet(self.ball_position_m)
        )
        require_finite(self.aim_deg, "aim_deg")
        require(abs(self.aim_deg) <= 45.0, "aim must be within +/-45 deg", self.aim_deg)
        object.__setattr__(self, "aim_deg", float(self.aim_deg))
        require(
            isinstance(self.samples, tuple)
            and len(self.samples) >= 3
            and all(isinstance(item, StrokeSample) for item in self.samples),
            "samples must be a tuple of at least three StrokeSample records",
        )
        times = [item.time_s for item in self.samples]
        require(
            all(b > a for a, b in zip(times, times[1:], strict=False)),
            "sample times must be strictly increasing",
        )

    def to_delivery_trajectory(self) -> DeliveryTrajectory:
        """Lift to the neutral delivery wire (module docstring derivation)."""
        times = np.asarray([item.time_s for item in self.samples], dtype=np.float64)
        positions = np.asarray(
            [item.position_m for item in self.samples], dtype=np.float64
        )
        quats = _canonical_quaternions(self.samples)
        linear = np.gradient(positions, times, axis=0)
        angular = _angular_velocity(quats, np.gradient(quats, times, axis=0))
        return DeliveryTrajectory(
            source_id=self.source_id,
            frame_id=self.frame_id,
            samples=tuple(
                TrajectorySample(
                    time_s=item.time_s,
                    position_m=item.position_m,
                    quaternion_wxyz=item.quaternion_wxyz,
                    linear_velocity_mps=_triplet(linear[index]),
                    angular_velocity_rad_s=_triplet(angular[index]),
                )
                for index, item in enumerate(self.samples)
            ),
        )


def _validated_triplet(value: Any) -> tuple[float, float, float]:
    """A finite 3-vector, reusing the delivery wire's own triple check."""
    return TrajectorySample(
        time_s=0.0,
        position_m=value,
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity_mps=(0.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, 0.0),
    ).position_m


def putting_stroke_to_json(stroke: PuttingStroke) -> str:
    """Serialize with deterministic key ordering and no non-finite values."""
    require(isinstance(stroke, PuttingStroke), "stroke must be PuttingStroke")
    payload: dict[str, Any] = {
        "format": PUTTING_STROKE_FORMAT,
        "source_id": stroke.source_id,
        "frame_id": stroke.frame_id,
        "ball_position_m": list(stroke.ball_position_m),
        "aim_deg": stroke.aim_deg,
        "samples": [
            {
                "time_s": item.time_s,
                "position_m": list(item.position_m),
                "quaternion_wxyz": list(item.quaternion_wxyz),
            }
            for item in stroke.samples
        ],
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def putting_stroke_from_json(text: str) -> PuttingStroke:
    """Parse and validate; unknown fields and wrong formats are refused."""
    require(isinstance(text, str), "text must be str")
    data = json.loads(text)
    require(isinstance(data, dict), "putting stroke must be an object")
    unknown = set(data) - _STROKE_FIELDS
    require(not unknown, f"unknown putting-stroke fields: {sorted(unknown)}")
    require(
        data.get("format") == PUTTING_STROKE_FORMAT,
        f"format must be {PUTTING_STROKE_FORMAT!r}",
    )
    raw_samples = data.get("samples")
    require(isinstance(raw_samples, list), "samples must be a list")
    return PuttingStroke(
        source_id=data.get("source_id"),
        frame_id=data.get("frame_id"),
        ball_position_m=tuple(data.get("ball_position_m", ())),
        aim_deg=data.get("aim_deg", 0.0),
        samples=_samples_from_records(raw_samples),
    )


def _samples_from_records(records: list[Any]) -> tuple[StrokeSample, ...]:
    """Pose records (wire or engine export) to validated stroke samples."""
    samples = []
    for record in records:
        require(isinstance(record, dict), "each sample must be an object")
        unknown = set(record) - _SAMPLE_FIELDS
        require(not unknown, f"unknown sample fields: {sorted(unknown)}")
        samples.append(
            StrokeSample(
                time_s=record.get("time_s"),
                position_m=tuple(record.get("position_m", ())),
                quaternion_wxyz=tuple(record.get("quaternion_wxyz", ())),
            )
        )
    return tuple(samples)


@dataclass(frozen=True)
class StrokeStrike:
    """The :func:`~.impact.strike` parameters recovered from a stroke.

    Angles use the ``swing_sim.impact`` conventions verbatim; face and
    path are relative to the declared aim line.
    """

    impact_index: int
    impact_time_s: float
    head_speed_mps: float
    aim_deg: float
    face_angle_deg: float
    path_angle_deg: float
    attack_angle_deg: float
    face_pitch_deg: float
    strike_offset_toe_mm: float
    strike_offset_high_mm: float


def _impact_index(
    trajectory: DeliveryTrajectory,
    ball_position_m: tuple[float, float, float],
    body_to_face_m: float,
) -> int:
    """Sample nearest face-ball contact; refuses a stroke that misses."""
    ball = np.asarray(ball_position_m, dtype=np.float64)
    separations = []
    for index in range(len(trajectory.samples)):
        center, _ = head_state_at(trajectory, index, grip_to_head_m=body_to_face_m)
        normal = trajectory.samples[index].rotation_matrix()[:, 0]
        separations.append(float((ball - center) @ normal))
    require(
        separations[0] > GOLF_BALL_RADIUS_M,
        "stroke must start with the face behind the ball",
        separations[0],
    )
    require(
        min(separations) <= GOLF_BALL_RADIUS_M,
        "stroke never reaches the ball",
        min(separations),
    )
    deltas = [abs(item - GOLF_BALL_RADIUS_M) for item in separations]
    return deltas.index(min(deltas))


def impact_sample_index(stroke: PuttingStroke, *, body_to_face_m: float) -> int:
    """Index of the sample nearest face-ball contact (module docstring).

    Raises:
        ValueError: If the stroke does not start behind the ball or
            never reaches it.
    """
    require(isinstance(stroke, PuttingStroke), "stroke must be PuttingStroke")
    return _impact_index(
        stroke.to_delivery_trajectory(), stroke.ball_position_m, body_to_face_m
    )


def strike_parameters(stroke: PuttingStroke, *, body_to_face_m: float) -> StrokeStrike:
    """Recover the P1 strike parameters at ball contact.

    Args:
        stroke: The recorded stroke.
        body_to_face_m: Rigid distance from the recorded body origin
            down ``+z`` to the face center [m]; positive.

    Returns:
        The :class:`StrokeStrike` feeding :func:`~.impact.strike`.
    """
    require(isinstance(stroke, PuttingStroke), "stroke must be PuttingStroke")
    trajectory = stroke.to_delivery_trajectory()
    index = _impact_index(trajectory, stroke.ball_position_m, body_to_face_m)
    view = delivery_view_at(trajectory, index, grip_to_head_m=body_to_face_m)
    center, _ = head_state_at(trajectory, index, grip_to_head_m=body_to_face_m)
    rotation = trajectory.samples[index].rotation_matrix()
    offset = np.asarray(stroke.ball_position_m, dtype=np.float64) - center
    normal = rotation[:, 0]
    pitch_deg = math.degrees(
        math.atan2(float(normal[1]), math.hypot(float(normal[0]), float(normal[2])))
    )
    result = StrokeStrike(
        impact_index=index,
        impact_time_s=stroke.samples[index].time_s,
        head_speed_mps=view.clubhead_speed_mps,
        aim_deg=stroke.aim_deg,
        face_angle_deg=view.face_angle_deg - stroke.aim_deg,
        path_angle_deg=view.club_path_deg - stroke.aim_deg,
        attack_angle_deg=view.attack_angle_deg,
        face_pitch_deg=pitch_deg,
        strike_offset_toe_mm=_M_TO_MM * float(offset @ rotation[:, 1]),
        strike_offset_high_mm=-_M_TO_MM * float(offset @ rotation[:, 2]),
    )
    ensure(result.head_speed_mps > 0.0, "head must be moving at impact")
    return result


def strike_from_stroke(
    stroke: PuttingStroke,
    putter: PutterSpec,
    *,
    body_to_face_m: float,
    shaft_lean_deg: float | None = None,
    head_moi_kg_m2: float | None = None,
) -> PuttLaunch:
    """Drive :func:`~.impact.strike` from a recorded stroke.

    Args:
        stroke: The recorded stroke.
        putter: Putter head description (P1 v1 spec, or
            ``golf_club.putter_head.putter_spec`` of a P3 document).
        body_to_face_m: Rigid body-origin-to-face-center distance [m].
        shaft_lean_deg: Forward press [deg]; ``None`` recovers it from
            the recorded face pitch as ``pitch - putter.loft_deg``, so
            the delivered dynamic loft comes from the capture.
        head_moi_kg_m2: P3 head MOI hook, passed through unchanged.

    Returns:
        The post-impact :class:`~.impact.PuttLaunch`.
    """
    parameters = strike_parameters(stroke, body_to_face_m=body_to_face_m)
    lean = (
        parameters.face_pitch_deg - putter.loft_deg
        if shaft_lean_deg is None
        else shaft_lean_deg
    )
    return strike(
        putter,
        parameters.head_speed_mps,
        lean,
        aim_deg=parameters.aim_deg,
        face_angle_deg=parameters.face_angle_deg,
        path_angle_deg=parameters.path_angle_deg,
        attack_angle_deg=parameters.attack_angle_deg,
        strike_offset_toe_mm=parameters.strike_offset_toe_mm,
        strike_offset_high_mm=parameters.strike_offset_high_mm,
        head_moi_kg_m2=head_moi_kg_m2,
    )


@dataclass(frozen=True)
class StrokePutt:
    """A recorded stroke integrated to rest or capture, with provenance.

    ``result`` is in the P2 putt frame, whose ``x`` axis is the start
    line; ``launch.start_azimuth_deg`` is that line's bearing in the
    stroke frame.
    """

    parameters: StrokeStrike
    launch: PuttLaunch
    result: PuttResult
    source_id: str
    frame_id: str
    sample_count: int
    stroke_format: str = PUTTING_STROKE_FORMAT


def putt_from_stroke(
    stroke: PuttingStroke,
    putter: PutterSpec,
    surface: GreenSurface,
    *,
    body_to_face_m: float,
    stimp_ft: float,
    hole_distance_m: float,
    shaft_lean_deg: float | None = None,
    head_moi_kg_m2: float | None = None,
    mu_slide: float = DEFAULT_SLIDING_MU,
    capture_model: CaptureModel = "effective_radius",
) -> StrokePutt:
    """End to end: recorded stroke -> P1 strike -> P2 surface roll.

    Returns:
        The :class:`StrokePutt`, carrying the engine provenance of the
        stroke that produced it.
    """
    parameters = strike_parameters(stroke, body_to_face_m=body_to_face_m)
    launch = strike_from_stroke(
        stroke,
        putter,
        body_to_face_m=body_to_face_m,
        shaft_lean_deg=shaft_lean_deg,
        head_moi_kg_m2=head_moi_kg_m2,
    )
    result = simulate_putt_on_surface(
        launch,
        surface,
        stimp_ft=stimp_ft,
        hole_distance_m=hole_distance_m,
        mu_slide=mu_slide,
        capture_model=capture_model,
    )
    return StrokePutt(
        parameters=parameters,
        launch=launch,
        result=result,
        source_id=stroke.source_id,
        frame_id=stroke.frame_id,
        sample_count=len(stroke.samples),
    )

"""Versioned immutable pre-impact bundle for impact kernels (IA-U2, #9703 / #5353).

This module defines the canonical shared wire and records for pre-impact club,
ball, and shaft state. It composes Tools conventions:
- ``golf_club.types.RigidTransform`` point convention
- ``swing_sim.delivery_interchange`` grip frame and ``(w, x, y, z)`` quaternions
- ``golf_club.impact_mobility.RigidContactBody`` head mass and COM inertia
- ``golf_club.grip_impedance`` linear-then-angular twist order

Postconditions: every constructed bundle holds finite SI values, proper
rotations, strictly increasing samples, an event inside a sampled interpolation
interval, a physically realizable COM inertia and modal arrays matching their
declared basis. Absent fields are explicit and raise :class:`AbsentFieldError`
on numeric use; they are never zero.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._pre_impact_contracts import (
    AbsentFieldError,
    FieldOrigin,
    FloatArray,
    PreImpactBundleError,
    Quantity,
    com_inertia,
    fail,
    identifier,
    nonnegative,
    positive,
    positive_semidefinite,
    sha256_hex,
    strictly_increasing,
    unit_vector,
)
from ._pre_impact_serde import bundle_from_dict, bundle_to_dict
from .modal_state import (
    ModalBasis,
    ModalProjection,
    ShaftState,
    project_onto_basis,
)
from .pre_impact_frames import (
    Pose,
    shift_twist_reference,
    shift_wrench_origin,
    twist_to_parent,
    wrench_to_parent,
)

PRE_IMPACT_BUNDLE_SCHEMA = "upstreamdrift.pre_impact_bundle"
PRE_IMPACT_BUNDLE_VERSION = 1
UNIT_SYSTEM = "SI"
WORLD_FRAME_ID = "world"
HEAD_FRAME_ID = "head"
GRIP_FRAME_ID = "grip"

#: Declared conventions; a payload carrying different ones is refused.
CONVENTIONS: Mapping[str, str] = {
    "pose": "p_parent = rotation @ p_child + translation_m",
    "quaternion": "wxyz, unit norm, never normalized",
    "head_frame": "origin at head COM; inertia, contact offset and normal in head",
    "grip_frame": "origin at butt, +z along shaft to head (Tools delivery wire)",
    "twist": "(linear, angular) of the head COM, expressed in world",
    "wrench": "(force, moment) in frame_id, moment about origin_m",
    "hand_impedance": "6x6 symmetric PSD, translations then small rotations",
    "modal_energy": "0.5 qd^T M_r qd + 0.5 q^T K_r q with basis matrices",
}
_INTERPOLANTS = frozenset({"linear", "cubic_hermite"})
_HANDS = frozenset({"lead", "trail"})


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        fail("type", f"{name} must be a nonempty list of strings")
    return tuple(identifier(item, name) for item in value)


@dataclass(frozen=True, eq=False)
class Provenance:
    """Equipment/ball/calibration identity, source digests and model tier."""

    equipment_id: str
    ball_id: str
    calibration_id: str
    model_tier: str
    source_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in ("equipment_id", "ball_id", "calibration_id", "model_tier"):
            identifier(getattr(self, name), name)
        if not self.source_hashes:
            fail("hash", "at least one source hash is required")
        for key, digest in self.source_hashes:
            identifier(key, "source_hashes key")
            sha256_hex(digest, f"source_hashes[{key}]")


@dataclass(frozen=True, eq=False)
class TimeBase:
    """Sampled source times, event time and bounded interpolation interval.

    The interval must lie inside the sampled span and contain the event, so no
    state is extrapolated through contact.
    """

    sample_times_s: FloatArray
    event_time_s: float
    interval_s: tuple[float, float]
    interpolant: str
    time_uncertainty_s: Quantity
    interpolation_error_m: Quantity

    def __post_init__(self) -> None:
        times = self.sample_times_s
        if times.ndim != 1 or times.size < 2:
            fail("shape", "sample_times_s needs at least two samples")
        strictly_increasing(times, "sample_times_s")
        start, end = self.interval_s
        if not times[0] <= start < end <= times[-1]:
            fail("extrapolation", "interval_s must lie within the sampled span")
        if not start <= self.event_time_s <= end:
            fail("event_outside_interval", "event_time_s must lie in interval_s")
        if self.interpolant not in _INTERPOLANTS:
            fail("interpolant", f"interpolant must be one of {sorted(_INTERPOLANTS)}")
        nonnegative(self.time_uncertainty_s, "time_uncertainty_s")
        nonnegative(self.interpolation_error_m, "interpolation_error_m")


@dataclass(frozen=True, eq=False)
class HeadState:
    """Rigid head COM state; vectors other than velocities are in ``head``.

    ``face_curvature_1_m`` is an optional symmetric 2x2 face curvature tensor
    [1/m] in face-tangent coordinates; no pinned Tools provider supplies it, so
    it is usually ``absent``. Mass, inertia, velocities, contact offset and
    normal are required; direct construction with them absent raises.
    """

    mass_kg: Quantity
    inertia_com_kg_m2: Quantity
    linear_velocity_mps: Quantity
    angular_velocity_rad_s: Quantity
    contact_offset_m: Quantity
    contact_normal: Quantity
    face_curvature_1_m: Quantity

    def __post_init__(self) -> None:
        positive(self.mass_kg, "head.mass_kg")
        com_inertia(self.inertia_com_kg_m2.value, "head.inertia_com_kg_m2")
        unit_vector(self.contact_normal.value, "head.contact_normal")
        if not self.face_curvature_1_m.is_absent:
            from ._pre_impact_contracts import symmetric

            symmetric(self.face_curvature_1_m.value, 2, "head.face_curvature_1_m")

    def tools_contact_body_fields(self) -> dict[str, Any]:
        """Keyword arguments for ``RigidContactBody``."""
        return {
            "mass_kg": float(self.mass_kg),
            "inertia_at_com_kg_m2": tuple(
                tuple(float(item) for item in row)
                for row in self.inertia_com_kg_m2.value
            ),
            "contact_offset_m": tuple(self.contact_offset_m.value),
        }


@dataclass(frozen=True, eq=False)
class BallState:
    """Ball COM position/velocity and spin, all in ``world``."""

    position_m: Quantity
    velocity_mps: Quantity
    spin_rad_s: Quantity


@dataclass(frozen=True, eq=False)
class HandWrench:
    """One hand's applied wrench at ``origin_m`` in ``frame_id``, plus impedance."""

    hand: str
    frame_id: str
    origin_m: FloatArray
    force_n: Quantity
    moment_n_m: Quantity
    stiffness: Quantity
    damping: Quantity
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.hand not in _HANDS:
            fail("hand", f"hand must be one of {sorted(_HANDS)}")
        if self.frame_id not in (WORLD_FRAME_ID, HEAD_FRAME_ID, GRIP_FRAME_ID):
            fail("frame_mismatch", f"unknown wrench frame {self.frame_id!r}")
        for name in ("stiffness", "damping"):
            quantity = getattr(self, name)
            if not quantity.is_absent:
                positive_semidefinite(quantity.value, 6, f"{self.hand}.{name}")


def grip_pose_from_delivery_sample(
    sample: Any, *, world_frame_id: str = WORLD_FRAME_ID
) -> Pose:
    """Adapt a Tools ``swing_sim.delivery_interchange.TrajectorySample``.

    The Tools wire already maps grip into world with a ``(w, x, y, z)``
    quaternion; this only re-validates it against the bundle's stricter
    unit-norm tolerance and never normalizes.
    """
    return Pose.from_quaternion(
        world_frame_id, GRIP_FRAME_ID, sample.quaternion_wxyz, sample.position_m
    )


@dataclass(frozen=True, eq=False)
class PreImpactBundle:
    """Versioned immutable pre-impact state; see module postconditions."""

    provenance: Provenance
    timebase: TimeBase
    head_pose: Pose
    grip_pose: Pose
    head: HeadState
    ball: BallState
    shaft: ShaftState
    hands: tuple[HandWrench, ...]
    constraints: tuple[str, ...]
    schema: str = PRE_IMPACT_BUNDLE_SCHEMA
    version: int = PRE_IMPACT_BUNDLE_VERSION
    units: str = UNIT_SYSTEM

    def __post_init__(self) -> None:
        if self.schema != PRE_IMPACT_BUNDLE_SCHEMA:
            fail("unsupported_schema", f"schema must be {PRE_IMPACT_BUNDLE_SCHEMA!r}")
        if self.version != PRE_IMPACT_BUNDLE_VERSION or isinstance(self.version, bool):
            fail("unsupported_version", f"version {self.version!r} is not supported")
        if self.units != UNIT_SYSTEM:
            fail("units", f"units must be {UNIT_SYSTEM!r}")
        for pose, frame in (
            (self.head_pose, HEAD_FRAME_ID),
            (self.grip_pose, GRIP_FRAME_ID),
        ):
            if (pose.parent_frame_id, pose.frame_id) != (WORLD_FRAME_ID, frame):
                fail("frame_mismatch", f"{frame} pose must be expressed in world")
        names = [hand.hand for hand in self.hands]
        if len(set(names)) != len(names):
            fail("hand", "each hand may appear at most once")

    # --- frames ---------------------------------------------------------------

    def _world_from(self, frame_id: str) -> Pose:
        poses = {HEAD_FRAME_ID: self.head_pose, GRIP_FRAME_ID: self.grip_pose}
        if frame_id == WORLD_FRAME_ID:
            return Pose.identity(WORLD_FRAME_ID)
        if frame_id not in poses:
            fail("frame_mismatch", f"unknown frame {frame_id!r}")
        return poses[frame_id]

    def pose_between(self, target_frame_id: str, source_frame_id: str) -> Pose:
        """Pose of ``source`` expressed in ``target`` (maps source -> target)."""
        return (
            self._world_from(target_frame_id)
            .inverse()
            .compose(self._world_from(source_frame_id))
        )

    def hand(self, name: str) -> HandWrench:
        for hand in self.hands:
            if hand.hand == name:
                return hand
        raise PreImpactBundleError("hand", f"no {name!r} hand wrench declared")

    def hand_wrench_in(self, name: str, frame_id: str) -> tuple[FloatArray, FloatArray]:
        """Hand force and moment about ``frame_id``'s origin, in ``frame_id``.

        Raises :class:`AbsentFieldError` when the hand wrench is absent.
        """
        hand = self.hand(name)
        force, moment = shift_wrench_origin(
            hand.force_n.value, hand.moment_n_m.value, hand.origin_m, np.zeros(3)
        )
        return wrench_to_parent(
            self.pose_between(frame_id, hand.frame_id), force, moment
        )

    def field_origins(self) -> dict[str, FieldOrigin]:
        """Origin of every quantity, keyed by dotted field path."""
        groups: list[tuple[str, object]] = [
            ("timebase", self.timebase),
            ("head", self.head),
            ("ball", self.ball),
            ("shaft", self.shaft),
        ]
        groups += [(f"hands.{hand.hand}", hand) for hand in self.hands]
        return {
            f"{prefix}.{name}": value.origin
            for prefix, record in groups
            for name, value in vars(record).items()
            if isinstance(value, Quantity)
        }

    # --- serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return bundle_to_dict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True
        )

    @classmethod
    def from_dict(cls, payload: object) -> PreImpactBundle:
        return bundle_from_dict(payload)

    @classmethod
    def from_json(cls, text: str) -> PreImpactBundle:
        if not isinstance(text, str):
            fail("type", "text must be str")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise PreImpactBundleError("json", str(error)) from error
        return cls.from_dict(payload)


__all__ = [
    "CONVENTIONS",
    "GRIP_FRAME_ID",
    "HEAD_FRAME_ID",
    "PRE_IMPACT_BUNDLE_SCHEMA",
    "PRE_IMPACT_BUNDLE_VERSION",
    "UNIT_SYSTEM",
    "WORLD_FRAME_ID",
    "AbsentFieldError",
    "BallState",
    "FieldOrigin",
    "HandWrench",
    "HeadState",
    "ModalBasis",
    "ModalProjection",
    "Pose",
    "PreImpactBundle",
    "PreImpactBundleError",
    "Provenance",
    "Quantity",
    "ShaftState",
    "TimeBase",
    "grip_pose_from_delivery_sample",
    "project_onto_basis",
    "shift_twist_reference",
    "shift_wrench_origin",
    "twist_to_parent",
    "wrench_to_parent",
]

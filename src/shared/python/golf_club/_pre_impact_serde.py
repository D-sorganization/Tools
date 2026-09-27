"""Serialization helpers for PreImpactBundle (Tools #5353)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ._pre_impact_contracts import (
    Quantity,
    fail,
    finite_array,
    finite_scalar,
    identifier,
    parse_quantity,
    reject_unknown,
)
from .modal_state import _shaft_from_dict
from .pre_impact_frames import Pose

if TYPE_CHECKING:
    from .pre_impact_bundle import (
        HandWrench,
        PreImpactBundle,
        Provenance,
        TimeBase,
    )

_TOP = frozenset(
    {"schema", "version", "units", "provenance", "timebase", "frames", "head"}
    | {"ball", "shaft", "hands", "constraints"}
)
_HEAD_SHAPES = {
    "mass_kg": (),
    "inertia_com_kg_m2": (3, 3),
    "linear_velocity_mps": (3,),
    "angular_velocity_rad_s": (3,),
    "contact_offset_m": (3,),
    "contact_normal": (3,),
    "face_curvature_1_m": (2, 2),
}
_HEAD_REQUIRED = frozenset(_HEAD_SHAPES) - {"face_curvature_1_m"}
_BALL_SHAPES = {"position_m": (3,), "velocity_mps": (3,), "spin_rad_s": (3,)}
_HAND_FIELDS = frozenset(
    {"hand", "frame_id", "origin_m", "force_n", "moment_n_m", "stiffness"}
    | {"damping", "assumptions"}
)


def _quantities(
    raw: Mapping[str, Any],
    shapes: Mapping[str, tuple[int, ...]],
    prefix: str,
    required: frozenset[str] = frozenset(),
) -> dict[str, Quantity]:
    return {
        name: parse_quantity(
            raw[name], f"{prefix}.{name}", shape, required=name in required
        )
        for name, shape in shapes.items()
    }


def _pose_from_dict(raw: object, name: str) -> Pose:
    keys = frozenset(raw) if isinstance(raw, Mapping) else frozenset()
    rotation_key = "quaternion_wxyz" if "quaternion_wxyz" in keys else "rotation"
    data = reject_unknown(
        raw,
        frozenset({"parent_frame_id", "frame_id", rotation_key, "translation_m"}),
        name,
    )
    if rotation_key == "quaternion_wxyz":
        return Pose.from_quaternion(
            data["parent_frame_id"],
            data["frame_id"],
            data[rotation_key],
            data["translation_m"],
        )
    return Pose(
        data["parent_frame_id"],
        data["frame_id"],
        finite_array(data["rotation"], (3, 3), f"{name}.rotation"),
        finite_array(data["translation_m"], (3,), f"{name}.translation_m"),
    )


def _provenance_from_dict(raw: object) -> Provenance:
    from .pre_impact_bundle import Provenance

    data = reject_unknown(
        raw,
        frozenset(
            {"equipment_id", "ball_id", "calibration_id", "model_tier", "source_hashes"}
        ),
        "provenance",
    )
    hashes = data["source_hashes"]
    if not isinstance(hashes, Mapping):
        fail("type", "source_hashes must be an object")
    return Provenance(
        data["equipment_id"],
        data["ball_id"],
        data["calibration_id"],
        data["model_tier"],
        tuple(sorted((str(key), value) for key, value in hashes.items())),
    )


def _timebase_from_dict(raw: object) -> TimeBase:
    from .pre_impact_bundle import TimeBase

    fields = {"sample_times_s", "event_time_s", "interval_s", "interpolant"}
    data = reject_unknown(
        raw,
        frozenset(fields | {"time_uncertainty_s", "interpolation_error_m"}),
        "timebase",
    )
    times = data["sample_times_s"]
    count = len(times) if isinstance(times, (list, tuple, np.ndarray)) else -1
    interval = finite_array(data["interval_s"], (2,), "interval_s")
    return TimeBase(
        sample_times_s=finite_array(times, (count,), "sample_times_s"),
        event_time_s=finite_scalar(data["event_time_s"], "event_time_s"),
        interval_s=(float(interval[0]), float(interval[1])),
        interpolant=identifier(data["interpolant"], "interpolant"),
        time_uncertainty_s=parse_quantity(
            data["time_uncertainty_s"], "time_uncertainty_s", ()
        ),
        interpolation_error_m=parse_quantity(
            data["interpolation_error_m"], "interpolation_error_m", ()
        ),
    )


def _hand_from_dict(raw: object) -> HandWrench:
    from .pre_impact_bundle import HandWrench, _strings

    data = reject_unknown(raw, _HAND_FIELDS, "hand")
    prefix = f"hands.{data['hand']}"
    shapes = {
        "force_n": (3,),
        "moment_n_m": (3,),
        "stiffness": (6, 6),
        "damping": (6, 6),
    }
    return HandWrench(
        hand=identifier(data["hand"], "hand"),
        frame_id=identifier(data["frame_id"], "frame_id"),
        origin_m=finite_array(data["origin_m"], (3,), f"{prefix}.origin_m"),
        assumptions=_strings(data["assumptions"], f"{prefix}.assumptions"),
        **_quantities(data, shapes, prefix),
    )


def bundle_from_dict(payload: object) -> PreImpactBundle:
    """Construct PreImpactBundle from serialized dictionary."""
    from .pre_impact_bundle import (
        CONVENTIONS,
        PRE_IMPACT_BUNDLE_SCHEMA,
        PRE_IMPACT_BUNDLE_VERSION,
        BallState,
        HeadState,
        PreImpactBundle,
        _strings,
    )

    allowed = (
        _TOP | {"conventions"}
        if isinstance(payload, Mapping) and "conventions" in payload
        else _TOP
    )
    data = reject_unknown(payload, frozenset(allowed), "bundle")
    if data["schema"] != PRE_IMPACT_BUNDLE_SCHEMA:
        fail("unsupported_schema", f"schema {data['schema']!r} is not supported")
    if data["version"] != PRE_IMPACT_BUNDLE_VERSION or isinstance(
        data["version"], bool
    ):
        fail("unsupported_version", f"version {data['version']!r} is not supported")
    if "conventions" in data and dict(data["conventions"]) != dict(CONVENTIONS):
        fail("conventions", "declared conventions differ from v1")
    frames = reject_unknown(data["frames"], frozenset({"head", "grip"}), "frames")
    head = reject_unknown(data["head"], frozenset(_HEAD_SHAPES), "head")
    ball = reject_unknown(data["ball"], frozenset(_BALL_SHAPES), "ball")
    hands = data["hands"]
    if not isinstance(hands, (list, tuple)):
        fail("type", "hands must be a list")
    return PreImpactBundle(
        schema=data["schema"],
        version=data["version"],
        units=data["units"],
        provenance=_provenance_from_dict(data["provenance"]),
        timebase=_timebase_from_dict(data["timebase"]),
        head_pose=_pose_from_dict(frames["head"], "frames.head"),
        grip_pose=_pose_from_dict(frames["grip"], "frames.grip"),
        head=HeadState(**_quantities(head, _HEAD_SHAPES, "head", _HEAD_REQUIRED)),
        ball=BallState(**_quantities(ball, _BALL_SHAPES, "ball")),
        shaft=_shaft_from_dict(data["shaft"]),
        hands=tuple(_hand_from_dict(hand) for hand in hands),
        constraints=_strings(data["constraints"], "constraints"),
    )


def _record_quantities(record: object) -> dict[str, Any]:
    return {
        name: value.to_dict()
        for name, value in vars(record).items()
        if isinstance(value, Quantity)
    }


def bundle_to_dict(bundle: PreImpactBundle) -> dict[str, Any]:
    """Serialize PreImpactBundle to dictionary."""
    from .pre_impact_bundle import CONVENTIONS

    provenance, timebase, shaft = bundle.provenance, bundle.timebase, bundle.shaft
    return {
        "schema": bundle.schema,
        "version": bundle.version,
        "units": bundle.units,
        "conventions": dict(CONVENTIONS),
        "provenance": {
            "equipment_id": provenance.equipment_id,
            "ball_id": provenance.ball_id,
            "calibration_id": provenance.calibration_id,
            "model_tier": provenance.model_tier,
            "source_hashes": dict(provenance.source_hashes),
        },
        "timebase": {
            "sample_times_s": timebase.sample_times_s.tolist(),
            "event_time_s": timebase.event_time_s,
            "interval_s": list(timebase.interval_s),
            "interpolant": timebase.interpolant,
            **_record_quantities(timebase),
        },
        "frames": {
            "head": bundle.head_pose.to_dict(),
            "grip": bundle.grip_pose.to_dict(),
        },
        "head": _record_quantities(bundle.head),
        "ball": _record_quantities(bundle.ball),
        "shaft": {
            "basis": shaft.basis.to_dict(),
            "basis_id": shaft.basis_id,
            "basis_version": shaft.basis_version,
            "axial_stations_m": shaft.axial_stations_m.tolist(),
            **_record_quantities(shaft),
        },
        "hands": [
            {
                "hand": hand.hand,
                "frame_id": hand.frame_id,
                "origin_m": hand.origin_m.tolist(),
                "assumptions": list(hand.assumptions),
                **_record_quantities(hand),
            }
            for hand in bundle.hands
        ],
        "constraints": list(bundle.constraints),
    }


__all__ = [
    "bundle_from_dict",
    "bundle_to_dict",
]

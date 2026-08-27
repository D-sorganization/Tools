"""Engine export adapters for the putting-stroke wire (#4800 P4).

Each adapter parses one engine's *documented export* — the same three
``delivery_interchange`` supports, in the pose-only subset a putter
capture emits — into a validated
:class:`~.stroke_interchange.PuttingStroke`. The engine runtimes are
never imported; these are format adapters, fixture-tested, so the
putting pipeline works wherever the export files can be produced.

**Drake** / **MuJoCo** — the ``drake.body_export/1`` and
``mujoco.site_export/1`` envelopes are parsed by
:func:`~..delivery_interchange.body_export_envelope` (shared with the
delivery adapters, not re-implemented); only the per-record schema
differs, carrying ``time_s``, ``position_m``, and ``quaternion_wxyz``
for the putter body/site.

**OpenSim** — the standard ``BodyKinematics`` ``.sto`` table, read by
delegating to :func:`~..delivery_interchange.trajectory_from_opensim_sto`
and keeping its pose columns.

Engine exports carry no address geometry, so the ball center and the aim
line are adapter arguments declared by the caller; axis remapping into
the AffineDrift world frame stays the exporter's responsibility and is
declared through ``frame_id``.
"""

from __future__ import annotations

from shared.python.swing_sim.delivery_interchange import (
    DRAKE_EXPORT_FORMAT,
    MUJOCO_EXPORT_FORMAT,
    body_export_envelope,
    trajectory_from_opensim_sto,
)

from .stroke_interchange import PuttingStroke, StrokeSample, _samples_from_records

__all__ = [
    "putting_stroke_from_drake_json",
    "putting_stroke_from_mujoco_json",
    "putting_stroke_from_opensim_sto",
]


def _stroke_from_export(
    text: str,
    *,
    expected_format: str,
    name_field: str,
    source_prefix: str,
    ball_position_m: tuple[float, float, float],
    aim_deg: float,
) -> PuttingStroke:
    """Shared engine-export path: the delivery envelope, pose records."""
    name, frame_id, records = body_export_envelope(
        text, expected_format=expected_format, name_field=name_field
    )
    return PuttingStroke(
        source_id=f"{source_prefix}:{name}",
        frame_id=frame_id,
        ball_position_m=ball_position_m,
        aim_deg=aim_deg,
        samples=_samples_from_records(records),
    )


def putting_stroke_from_drake_json(
    text: str,
    *,
    ball_position_m: tuple[float, float, float],
    aim_deg: float = 0.0,
) -> PuttingStroke:
    """Parse a pose-only ``drake.body_export/1`` putter-body document."""
    return _stroke_from_export(
        text,
        expected_format=DRAKE_EXPORT_FORMAT,
        name_field="body_name",
        source_prefix="drake",
        ball_position_m=ball_position_m,
        aim_deg=aim_deg,
    )


def putting_stroke_from_mujoco_json(
    text: str,
    *,
    ball_position_m: tuple[float, float, float],
    aim_deg: float = 0.0,
) -> PuttingStroke:
    """Parse a pose-only ``mujoco.site_export/1`` putter-site document."""
    return _stroke_from_export(
        text,
        expected_format=MUJOCO_EXPORT_FORMAT,
        name_field="site_name",
        source_prefix="mujoco",
        ball_position_m=ball_position_m,
        aim_deg=aim_deg,
    )


def putting_stroke_from_opensim_sto(
    text: str,
    *,
    body_name: str,
    frame_id: str,
    ball_position_m: tuple[float, float, float],
    aim_deg: float = 0.0,
) -> PuttingStroke:
    """Parse an OpenSim ``BodyKinematics`` ``.sto`` table for a putter body.

    Delegates the whole table read to
    :func:`~..delivery_interchange.trajectory_from_opensim_sto` and keeps
    the pose columns; this wire re-derives velocity itself, so the
    delivery adapter's Euler-rate approximation is discarded rather than
    carried.
    """
    trajectory = trajectory_from_opensim_sto(
        text, body_name=body_name, frame_id=frame_id
    )
    return PuttingStroke(
        source_id=trajectory.source_id,
        frame_id=trajectory.frame_id,
        ball_position_m=ball_position_m,
        aim_deg=aim_deg,
        samples=tuple(
            StrokeSample(
                time_s=item.time_s,
                position_m=item.position_m,
                quaternion_wxyz=item.quaternion_wxyz,
            )
            for item in trajectory.samples
        ),
    )

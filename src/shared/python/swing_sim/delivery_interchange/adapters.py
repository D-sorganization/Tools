"""Engine export adapters for the delivery-trajectory wire (C5, #4554).

Each adapter parses one engine's *documented export* into the neutral
:class:`~.trajectory.DeliveryTrajectory`. The engine runtimes are never
imported — these are format adapters, fixture-tested, so the fitting
pipeline works wherever the export files can be produced. Model owners
export as follows:

**Drake** — serialize the grip body's pose and spatial velocity per step::

    # plant: MultibodyPlant, context per step, body = grip body
    X = plant.EvalBodyPoseInWorld(context, body)
    V = plant.EvalBodySpatialVelocityInWorld(context, body)
    record = {"time_s": t,
              "position_m": list(X.translation()),
              "quaternion_wxyz": [X.rotation().ToQuaternion().w(), ...xyz],
              "linear_velocity_mps": list(V.translational()),
              "angular_velocity_rad_s": list(V.rotational())}

wrapped as ``{"format": "drake.body_export/1", "body_name": ...,
"frame_id": "world", "records": [...]}``.

**MuJoCo** — attach a site to the grip body and record per step::

    record = {"time_s": data.time,
              "position_m": list(data.site_xpos[site_id]),
              "quaternion_wxyz": list(quat_from_xmat(data.site_xmat[site_id])),
              "linear_velocity_mps": ...,   # mj_objectVelocity, flipped order
              "angular_velocity_rad_s": ...}

wrapped as ``{"format": "mujoco.site_export/1", "site_name": ...,
"frame_id": "world", "records": [...]}``.

**OpenSim** — the standard ``.sto`` table (``BodyKinematics`` positions
report): tab/whitespace-separated columns after ``endheader``, with the
grip body's ``<body>_X/_Y/_Z`` positions in meters and ``<body>_Ox/_Oy/_Oz``
body-fixed XYZ Euler angles in degrees. Velocities are derived by central
differences — the documented, deliberate v1 behavior for ``.sto`` sources
whose velocity tables are not provided.

Axis remapping into the AffineDrift world frame (x target, y up, z right)
is the exporter's responsibility and is declared through ``frame_id``; the
adapters refuse exports that do not declare it.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from shared.python.contracts import require

from .trajectory import DeliveryTrajectory, TrajectorySample

DRAKE_EXPORT_FORMAT = "drake.body_export/1"
MUJOCO_EXPORT_FORMAT = "mujoco.site_export/1"

_RECORD_FIELDS = frozenset(
    {
        "time_s",
        "position_m",
        "quaternion_wxyz",
        "linear_velocity_mps",
        "angular_velocity_rad_s",
    }
)

__all__ = [
    "DRAKE_EXPORT_FORMAT",
    "MUJOCO_EXPORT_FORMAT",
    "trajectory_from_drake_json",
    "trajectory_from_mujoco_json",
    "trajectory_from_opensim_sto",
]


def _records_to_trajectory(
    records: list[Any], source_id: str, frame_id: str
) -> DeliveryTrajectory:
    samples = []
    for record in records:
        require(isinstance(record, dict), "each record must be an object")
        unknown = set(record) - _RECORD_FIELDS
        require(not unknown, f"unknown record fields: {sorted(unknown)}")
        samples.append(
            TrajectorySample(
                time_s=record.get("time_s"),
                position_m=tuple(record.get("position_m", ())),
                quaternion_wxyz=tuple(record.get("quaternion_wxyz", ())),
                linear_velocity_mps=tuple(record.get("linear_velocity_mps", ())),
                angular_velocity_rad_s=tuple(record.get("angular_velocity_rad_s", ())),
            )
        )
    return DeliveryTrajectory(
        source_id=source_id, frame_id=frame_id, samples=tuple(samples)
    )


def _body_export_from_json(
    text: str, *, expected_format: str, name_field: str, source_prefix: str
) -> DeliveryTrajectory:
    require(isinstance(text, str), "text must be str")
    data = json.loads(text)
    require(isinstance(data, dict), "export must be an object")
    require(
        data.get("format") == expected_format,
        f"format must be {expected_format!r}",
    )
    name = data.get(name_field)
    require(isinstance(name, str) and name != "", f"{name_field} must be nonempty")
    frame_id = data.get("frame_id")
    require(
        isinstance(frame_id, str) and frame_id != "",
        "frame_id must declare the export's world frame",
    )
    records = data.get("records")
    require(isinstance(records, list), "records must be a list")
    return _records_to_trajectory(records, f"{source_prefix}:{name}", frame_id)


def trajectory_from_drake_json(text: str) -> DeliveryTrajectory:
    """Parse a ``drake.body_export/1`` document (see module docstring)."""
    return _body_export_from_json(
        text,
        expected_format=DRAKE_EXPORT_FORMAT,
        name_field="body_name",
        source_prefix="drake",
    )


def trajectory_from_mujoco_json(text: str) -> DeliveryTrajectory:
    """Parse a ``mujoco.site_export/1`` document (see module docstring)."""
    return _body_export_from_json(
        text,
        expected_format=MUJOCO_EXPORT_FORMAT,
        name_field="site_name",
        source_prefix="mujoco",
    )


def _euler_xyz_deg_to_quaternion(
    rx_deg: float, ry_deg: float, rz_deg: float
) -> tuple[float, float, float, float]:
    """Body-fixed XYZ Euler angles (OpenSim BodyKinematics) to wxyz."""
    hx = math.radians(rx_deg) / 2.0
    hy = math.radians(ry_deg) / 2.0
    hz = math.radians(rz_deg) / 2.0
    cx, sx = math.cos(hx), math.sin(hx)
    cy, sy = math.cos(hy), math.sin(hy)
    cz, sz = math.cos(hz), math.sin(hz)
    return (
        cx * cy * cz - sx * sy * sz,
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
    )


def trajectory_from_opensim_sto(
    text: str,
    *,
    body_name: str,
    frame_id: str,
) -> DeliveryTrajectory:
    """Parse an OpenSim ``BodyKinematics`` position ``.sto`` table.

    Requires columns ``time``, ``{body}_X/_Y/_Z`` (meters) and
    ``{body}_Ox/_Oy/_Oz`` (body-fixed XYZ Euler, degrees). Linear and
    angular velocities are central-differenced from the sampled poses —
    a documented v1 choice for ``.sto`` sources without velocity tables.
    """
    require(isinstance(text, str), "text must be str")
    require(
        isinstance(body_name, str) and body_name != "", "body_name must be nonempty"
    )
    require(isinstance(frame_id, str) and frame_id != "", "frame_id must be nonempty")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    require("endheader" in lines, ".sto input must contain an endheader line")
    body_start = lines.index("endheader") + 1
    require(len(lines) > body_start + 2, ".sto input must contain data rows")
    header = lines[body_start].split()
    columns = {name: position for position, name in enumerate(header)}
    needed = ["time"] + [
        f"{body_name}_{suffix}" for suffix in ("X", "Y", "Z", "Ox", "Oy", "Oz")
    ]
    missing = [name for name in needed if name not in columns]
    require(not missing, f"missing .sto columns: {missing}")

    rows = []
    for line in lines[body_start + 1 :]:
        values = [float(item) for item in line.split()]
        require(len(values) == len(header), ".sto row width must match its header")
        rows.append([values[columns[name]] for name in needed])
    table = np.asarray(rows, dtype=np.float64)
    require(bool(np.isfinite(table).all()), ".sto values must be finite")

    times = table[:, 0]
    positions = table[:, 1:4]
    eulers_deg = table[:, 4:7]
    linear = np.gradient(positions, times, axis=0)
    angular = np.gradient(np.radians(eulers_deg), times, axis=0)

    samples = tuple(
        TrajectorySample(
            time_s=float(times[i]),
            position_m=(
                float(positions[i][0]),
                float(positions[i][1]),
                float(positions[i][2]),
            ),
            quaternion_wxyz=_euler_xyz_deg_to_quaternion(*eulers_deg[i]),
            linear_velocity_mps=(
                float(linear[i][0]),
                float(linear[i][1]),
                float(linear[i][2]),
            ),
            angular_velocity_rad_s=(
                float(angular[i][0]),
                float(angular[i][1]),
                float(angular[i][2]),
            ),
        )
        for i in range(len(times))
    )
    return DeliveryTrajectory(
        source_id=f"opensim:{body_name}", frame_id=frame_id, samples=samples
    )

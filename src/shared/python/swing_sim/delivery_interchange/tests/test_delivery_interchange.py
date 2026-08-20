"""Analytic circular-swing and adapter gates for the interchange (C5, #4554)."""

from __future__ import annotations

import json
import math

import pytest

from shared.python.contracts import PreconditionError
from shared.python.swing_sim.delivery_interchange import (
    DELIVERY_TRAJECTORY_FORMAT,
    DRAKE_EXPORT_FORMAT,
    MUJOCO_EXPORT_FORMAT,
    DeliveryTrajectory,
    TrajectorySample,
    delivery_trajectory_from_json,
    delivery_trajectory_to_json,
    delivery_view_at,
    grip_kinematics_at,
    head_state_at,
    trajectory_from_drake_json,
    trajectory_from_mujoco_json,
    trajectory_from_opensim_sto,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_GRIP_RADIUS_M = 1.0
_HEAD_EXTENSION_M = 0.15
_OMEGA_RAD_S = 30.0


def _q_multiply(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _circular_swing(sample_count: int = 21) -> DeliveryTrajectory:
    """A rigid club rotating at constant omega about +z through the low point.

    Grip on a circle of radius 1 m in the vertical x-y plane; the grip
    frame's +z (shaft axis) points radially outward, +x is the face
    normal, square at the low point. Every derived quantity has a closed
    form, so the assertions below are analytic, not regression pins.
    """
    q_x90 = (math.cos(math.pi / 4.0), math.sin(math.pi / 4.0), 0.0, 0.0)
    samples = []
    for i in range(sample_count):
        theta = -0.2 + 0.4 * i / (sample_count - 1)
        time_s = theta / _OMEGA_RAD_S
        q_z = (math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0))
        samples.append(
            TrajectorySample(
                time_s=time_s,
                position_m=(
                    _GRIP_RADIUS_M * math.sin(theta),
                    -_GRIP_RADIUS_M * math.cos(theta),
                    0.0,
                ),
                quaternion_wxyz=_q_multiply(q_z, q_x90),
                linear_velocity_mps=(
                    _OMEGA_RAD_S * _GRIP_RADIUS_M * math.cos(theta),
                    _OMEGA_RAD_S * _GRIP_RADIUS_M * math.sin(theta),
                    0.0,
                ),
                angular_velocity_rad_s=(0.0, 0.0, _OMEGA_RAD_S),
            )
        )
    return DeliveryTrajectory(
        source_id="analytic-circular-swing",
        frame_id="affine_drift.world",
        samples=tuple(samples),
    )


class TestAnalyticCircularSwing:
    def test_head_state_extends_the_rigid_club_exactly(self) -> None:
        trajectory = _circular_swing()
        low_point = trajectory.index_at_time(0.0)
        position, velocity = head_state_at(
            trajectory, low_point, grip_to_head_m=_HEAD_EXTENSION_M
        )
        head_radius = _GRIP_RADIUS_M + _HEAD_EXTENSION_M
        assert position == pytest.approx((0.0, -head_radius, 0.0), abs=1e-12)
        assert velocity == pytest.approx(
            (_OMEGA_RAD_S * head_radius, 0.0, 0.0), abs=1e-9
        )

    def test_grip_kinematics_recover_omega_alpha_and_radius(self) -> None:
        trajectory = _circular_swing()
        low_point = trajectory.index_at_time(0.0)
        kinematics = grip_kinematics_at(
            trajectory, low_point, grip_to_head_m=_HEAD_EXTENSION_M
        )
        assert kinematics["omega_rad_s"] == pytest.approx(_OMEGA_RAD_S, rel=1e-12)
        assert kinematics["alpha_rad_s2"] == pytest.approx(0.0, abs=1e-9)
        assert kinematics["swing_radius_m"] == pytest.approx(
            _GRIP_RADIUS_M + _HEAD_EXTENSION_M, rel=1e-9
        )

    def test_delivery_view_is_square_and_level_at_the_low_point(self) -> None:
        trajectory = _circular_swing()
        low_point = trajectory.index_at_time(0.0)
        view = delivery_view_at(trajectory, low_point, grip_to_head_m=_HEAD_EXTENSION_M)
        assert view.clubhead_speed_mps == pytest.approx(
            _OMEGA_RAD_S * (_GRIP_RADIUS_M + _HEAD_EXTENSION_M), rel=1e-9
        )
        assert view.attack_angle_deg == pytest.approx(0.0, abs=1e-9)
        assert view.club_path_deg == pytest.approx(0.0, abs=1e-9)
        assert view.face_angle_deg == pytest.approx(0.0, abs=1e-9)

    def test_attack_angle_is_negative_before_the_low_point(self) -> None:
        trajectory = _circular_swing()
        before = trajectory.index_at_time(-0.1 / _OMEGA_RAD_S)
        view = delivery_view_at(trajectory, before, grip_to_head_m=_HEAD_EXTENSION_M)
        assert view.attack_angle_deg < -0.1


class TestWire:
    def test_round_trip_is_deterministic_and_lossless(self) -> None:
        trajectory = _circular_swing(sample_count=5)
        first = delivery_trajectory_to_json(trajectory)
        assert delivery_trajectory_to_json(trajectory) == first
        restored = delivery_trajectory_from_json(first)
        assert delivery_trajectory_to_json(restored) == first
        assert json.loads(first)["format"] == DELIVERY_TRAJECTORY_FORMAT

    def test_unknown_fields_and_bad_quaternions_are_refused(self) -> None:
        payload = json.loads(
            delivery_trajectory_to_json(_circular_swing(sample_count=3))
        )
        payload["extra"] = 1
        with pytest.raises(PreconditionError):
            delivery_trajectory_from_json(json.dumps(payload))
        with pytest.raises(PreconditionError, match="unit length"):
            TrajectorySample(
                time_s=0.0,
                position_m=(0.0, 0.0, 0.0),
                quaternion_wxyz=(1.0, 1.0, 0.0, 0.0),
                linear_velocity_mps=(0.0, 0.0, 0.0),
                angular_velocity_rad_s=(0.0, 0.0, 0.0),
            )

    def test_times_must_strictly_increase(self) -> None:
        sample = _circular_swing(sample_count=3).samples[0]
        with pytest.raises(PreconditionError, match="strictly increasing"):
            DeliveryTrajectory(
                source_id="bad",
                frame_id="affine_drift.world",
                samples=(sample, sample),
            )


def _engine_records() -> list[dict[str, object]]:
    return [
        {
            "time_s": sample.time_s,
            "position_m": list(sample.position_m),
            "quaternion_wxyz": list(sample.quaternion_wxyz),
            "linear_velocity_mps": list(sample.linear_velocity_mps),
            "angular_velocity_rad_s": list(sample.angular_velocity_rad_s),
        }
        for sample in _circular_swing(sample_count=5).samples
    ]


class TestEngineAdapters:
    def test_drake_export_parses_to_the_same_trajectory(self) -> None:
        text = json.dumps(
            {
                "format": DRAKE_EXPORT_FORMAT,
                "body_name": "grip_body",
                "frame_id": "affine_drift.world",
                "records": _engine_records(),
            }
        )
        trajectory = trajectory_from_drake_json(text)
        assert trajectory.source_id == "drake:grip_body"
        low = trajectory.index_at_time(0.0)
        view = delivery_view_at(trajectory, low, grip_to_head_m=_HEAD_EXTENSION_M)
        assert view.attack_angle_deg == pytest.approx(0.0, abs=1e-9)

    def test_mujoco_export_parses_and_refuses_missing_frame(self) -> None:
        base = {
            "format": MUJOCO_EXPORT_FORMAT,
            "site_name": "grip_site",
            "frame_id": "affine_drift.world",
            "records": _engine_records(),
        }
        trajectory = trajectory_from_mujoco_json(json.dumps(base))
        assert trajectory.source_id == "mujoco:grip_site"
        missing = dict(base)
        del missing["frame_id"]
        with pytest.raises(PreconditionError, match="frame_id"):
            trajectory_from_mujoco_json(json.dumps(missing))

    def test_wrong_engine_format_is_refused(self) -> None:
        text = json.dumps(
            {
                "format": "drake.body_export/999",
                "body_name": "grip",
                "frame_id": "w",
                "records": [],
            }
        )
        with pytest.raises(PreconditionError, match="format"):
            trajectory_from_drake_json(text)

    def test_opensim_sto_parses_positions_and_differentiates(self) -> None:
        rows = []
        for i in range(9):
            t = 0.01 * i
            # Pure translation at 2 m/s along x, constant orientation.
            rows.append(f"{t:.6f}\t{2.0 * t:.6f}\t0.0\t0.0\t0.0\t0.0\t0.0")
        text = "\n".join(
            [
                "BodyKinematics_pos",
                "version=1",
                "nRows=9",
                "nColumns=7",
                "inDegrees=yes",
                "endheader",
                "time\tgrip_X\tgrip_Y\tgrip_Z\tgrip_Ox\tgrip_Oy\tgrip_Oz",
            ]
            + rows
        )
        trajectory = trajectory_from_opensim_sto(
            text, body_name="grip", frame_id="affine_drift.world"
        )
        assert trajectory.source_id == "opensim:grip"
        middle = trajectory.samples[4]
        assert middle.linear_velocity_mps == pytest.approx((2.0, 0.0, 0.0), abs=1e-9)
        assert middle.quaternion_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))

    def test_opensim_sto_refuses_missing_columns(self) -> None:
        text = "\n".join(
            [
                "endheader",
                "time\tother_X",
                "0.0\t1.0",
                "0.1\t2.0",
            ]
        )
        with pytest.raises(PreconditionError, match="missing .sto columns"):
            trajectory_from_opensim_sto(
                text, body_name="grip", frame_id="affine_drift.world"
            )

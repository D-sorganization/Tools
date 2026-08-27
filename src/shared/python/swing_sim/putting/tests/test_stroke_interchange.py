"""Analytic stroke gates, wire posture, adapters, end-to-end (#4800 P4).

The synthetic stroke below is authored *from* the strike parameters: a
rigid putter head moving at constant velocity with a fixed pose, placed
so the face reaches the ball exactly on the middle sample. Every
kinematic quantity therefore has a closed form and the recovery
assertions are analytic, not regression pins. The end-to-end gate runs
the shipped ``fixtures/drake_putter_stroke.json`` engine export through
the whole chain to a holed putt.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from shared.python.contracts import PreconditionError
from shared.python.swing_sim.impact import GOLF_BALL_MASS_KG, GOLF_BALL_RADIUS_M
from shared.python.swing_sim.putting import (
    MINIMAL_PUTTERS,
    PUTTING_STROKE_FORMAT,
    PlanarGreenSurface,
    PuttingStroke,
    StrokeSample,
    impact_sample_index,
    putt_from_stroke,
    putting_stroke_from_drake_json,
    putting_stroke_from_json,
    putting_stroke_from_mujoco_json,
    putting_stroke_from_opensim_sto,
    putting_stroke_to_json,
    strike,
    strike_from_stroke,
    strike_parameters,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

#: Grip-marker-to-face-center distance for the synthetic strokes [m].
BODY_TO_FACE_M = 0.85
FRAME_ID = "affine_drift.world"
HOLE_DISTANCE_M = 3.0
STIMP_FT = 10.0
#: 2/7 tangential rolling cap, the constant the impact modules derive.
ROLLING_CAP = 2.0 / 7.0
PUTTER = MINIMAL_PUTTERS["Blade Putter"]
FIXTURE = Path(__file__).parent / "fixtures" / "drake_putter_stroke.json"


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


def _axis_quaternion(
    axis: tuple[float, float, float], degrees: float
) -> tuple[float, float, float, float]:
    half = math.radians(degrees) / 2.0
    sin = math.sin(half)
    return (math.cos(half), axis[0] * sin, axis[1] * sin, axis[2] * sin)


def _rotation(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _pose(loft_deg: float, azimuth_deg: float) -> tuple[float, float, float, float]:
    """Putter-body pose: address frame, pitched by loft, yawed to azimuth."""
    address = _axis_quaternion((1.0, 0.0, 0.0), 90.0)
    pitch = _axis_quaternion((0.0, 0.0, 1.0), loft_deg)
    yaw = _axis_quaternion((0.0, 1.0, 0.0), -azimuth_deg)
    return _q_multiply(yaw, _q_multiply(pitch, address))


def _stroke(
    *,
    speed_mps: float = 1.6,
    loft_deg: float = 3.0,
    face_deg: float = 0.0,
    path_deg: float = 0.0,
    attack_deg: float = 0.0,
    toe_mm: float = 0.0,
    high_mm: float = 0.0,
    aim_deg: float = 0.0,
    count: int = 9,
    dt_s: float = 0.002,
    lead_m: float = 0.0,
) -> PuttingStroke:
    """A constant-velocity stroke authored from its strike parameters."""
    quaternion = _pose(loft_deg, aim_deg + face_deg)
    rotation = _rotation(quaternion)
    azimuth = math.radians(aim_deg + path_deg)
    attack = math.radians(attack_deg)
    velocity = speed_mps * np.array(
        [
            math.cos(attack) * math.cos(azimuth),
            math.sin(attack),
            math.cos(attack) * math.sin(azimuth),
        ]
    )
    ball = np.array([1.0, GOLF_BALL_RADIUS_M, 0.0])
    normal, toe, up = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
    contact = (
        ball
        - normal * (GOLF_BALL_RADIUS_M + lead_m)
        - toe * (toe_mm * 1e-3)
        - up * (high_mm * 1e-3)
    )
    middle = count // 2
    arm = rotation @ np.array([0.0, 0.0, BODY_TO_FACE_M])
    samples = tuple(
        StrokeSample(
            time_s=index * dt_s,
            position_m=tuple(contact + velocity * (index * dt_s - middle * dt_s) - arm),
            quaternion_wxyz=quaternion,
        )
        for index in range(count)
    )
    return PuttingStroke(
        source_id="analytic-stroke",
        frame_id=FRAME_ID,
        ball_position_m=tuple(ball),
        aim_deg=aim_deg,
        samples=samples,
    )


def _parameters(stroke: PuttingStroke) -> object:
    return strike_parameters(stroke, body_to_face_m=BODY_TO_FACE_M)


class TestAnalyticSquareStroke:
    def test_square_stroke_recovers_the_authored_delivery_exactly(self) -> None:
        recovered = _parameters(_stroke(speed_mps=1.6, loft_deg=3.0))
        assert recovered.head_speed_mps == pytest.approx(1.6, rel=1e-12)
        assert recovered.face_angle_deg == pytest.approx(0.0, abs=1e-12)
        assert recovered.path_angle_deg == pytest.approx(0.0, abs=1e-12)
        assert recovered.attack_angle_deg == pytest.approx(0.0, abs=1e-9)
        assert recovered.aim_deg == 0.0
        assert recovered.strike_offset_toe_mm == pytest.approx(0.0, abs=1e-9)
        assert recovered.strike_offset_high_mm == pytest.approx(0.0, abs=1e-9)

    def test_face_pitch_reports_the_delivered_dynamic_loft(self) -> None:
        assert _parameters(_stroke(loft_deg=4.5)).face_pitch_deg == pytest.approx(
            4.5, rel=1e-12
        )

    def test_impact_lands_on_the_authored_contact_sample(self) -> None:
        stroke = _stroke(count=11)
        index = impact_sample_index(stroke, body_to_face_m=BODY_TO_FACE_M)
        assert index == 5
        assert stroke.samples[index].time_s == pytest.approx(0.010, rel=1e-12)


class TestAuthoredDelivery:
    @pytest.mark.parametrize(
        ("face_deg", "path_deg", "attack_deg", "aim_deg"),
        [
            (1.5, -0.5, 1.2, 2.0),
            (-2.0, 2.0, -1.5, -3.0),
            (0.0, 3.0, 0.0, 0.0),
        ],
    )
    def test_angles_are_recovered_off_the_declared_aim_line(
        self, face_deg: float, path_deg: float, attack_deg: float, aim_deg: float
    ) -> None:
        recovered = _parameters(
            _stroke(
                face_deg=face_deg,
                path_deg=path_deg,
                attack_deg=attack_deg,
                aim_deg=aim_deg,
            )
        )
        assert recovered.aim_deg == pytest.approx(aim_deg, rel=1e-12)
        assert recovered.face_angle_deg == pytest.approx(face_deg, abs=1e-9)
        assert recovered.path_angle_deg == pytest.approx(path_deg, abs=1e-9)
        assert recovered.attack_angle_deg == pytest.approx(attack_deg, abs=1e-9)

    @pytest.mark.parametrize(("toe_mm", "high_mm"), [(6.0, -2.0), (-9.0, 3.5)])
    def test_strike_location_resolves_in_the_face_frame(
        self, toe_mm: float, high_mm: float
    ) -> None:
        recovered = _parameters(_stroke(toe_mm=toe_mm, high_mm=high_mm, face_deg=1.0))
        assert recovered.strike_offset_toe_mm == pytest.approx(toe_mm, abs=1e-6)
        assert recovered.strike_offset_high_mm == pytest.approx(high_mm, abs=1e-6)


class TestStartLine:
    def test_face_path_split_reproduces_the_P1_closed_form(self) -> None:
        face_deg, path_deg = 1.0, -1.0
        launch = strike_from_stroke(
            _stroke(face_deg=face_deg, path_deg=path_deg),
            PUTTER,
            body_to_face_m=BODY_TO_FACE_M,
        )
        mass_ratio = PUTTER.head_mass_kg / (PUTTER.head_mass_kg + GOLF_BALL_MASS_KG)
        transfer = (1.0 + PUTTER.cor) * mass_ratio
        face_to_path = math.radians(path_deg - face_deg)
        expected = face_deg + math.degrees(
            math.atan2(
                ROLLING_CAP * math.sin(face_to_path),
                transfer * math.cos(face_to_path),
            )
        )
        assert launch.start_azimuth_deg == pytest.approx(expected, rel=1e-9)

    def test_square_stroke_starts_on_the_declared_aim_line(self) -> None:
        launch = strike_from_stroke(
            _stroke(aim_deg=2.0), PUTTER, body_to_face_m=BODY_TO_FACE_M
        )
        assert launch.start_azimuth_deg == pytest.approx(2.0, abs=1e-9)

    def test_strike_from_stroke_matches_a_direct_P1_call(self) -> None:
        stroke = _stroke(face_deg=1.5, path_deg=-0.5, attack_deg=1.0, toe_mm=4.0)
        recovered = _parameters(stroke)
        direct = strike(
            PUTTER,
            recovered.head_speed_mps,
            recovered.face_pitch_deg - PUTTER.loft_deg,
            aim_deg=recovered.aim_deg,
            face_angle_deg=recovered.face_angle_deg,
            path_angle_deg=recovered.path_angle_deg,
            attack_angle_deg=recovered.attack_angle_deg,
            strike_offset_toe_mm=recovered.strike_offset_toe_mm,
            strike_offset_high_mm=recovered.strike_offset_high_mm,
        )
        assert (
            strike_from_stroke(stroke, PUTTER, body_to_face_m=BODY_TO_FACE_M) == direct
        )


class TestDeliveryLift:
    def test_central_differences_recover_the_constant_velocity(self) -> None:
        trajectory = _stroke(speed_mps=1.9, path_deg=2.0).to_delivery_trajectory()
        middle = trajectory.samples[4]
        speed = float(np.linalg.norm(middle.linear_velocity_mps))
        assert speed == pytest.approx(1.9, rel=1e-9)
        assert middle.angular_velocity_rad_s == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)

    def test_rotating_stroke_recovers_its_angular_velocity(self) -> None:
        omega = 2.0
        samples = tuple(
            StrokeSample(
                time_s=0.002 * index,
                position_m=(0.0, 1.0, 0.0),
                quaternion_wxyz=_q_multiply(
                    _axis_quaternion(
                        (0.0, 1.0, 0.0), math.degrees(omega * 0.002 * index)
                    ),
                    _pose(3.0, 0.0),
                ),
            )
            for index in range(9)
        )
        trajectory = PuttingStroke(
            source_id="rotating",
            frame_id=FRAME_ID,
            ball_position_m=(1.0, GOLF_BALL_RADIUS_M, 0.0),
            samples=samples,
        ).to_delivery_trajectory()
        assert trajectory.samples[4].angular_velocity_rad_s == pytest.approx(
            (0.0, omega, 0.0), abs=1e-4
        )


class TestWire:
    def test_round_trip_is_deterministic_and_lossless(self) -> None:
        stroke = _stroke(face_deg=1.0, aim_deg=1.0)
        first = putting_stroke_to_json(stroke)
        assert putting_stroke_to_json(stroke) == first
        assert putting_stroke_to_json(putting_stroke_from_json(first)) == first
        assert json.loads(first)["format"] == PUTTING_STROKE_FORMAT

    def test_unknown_fields_are_refused(self) -> None:
        payload = json.loads(putting_stroke_to_json(_stroke()))
        payload["extra"] = 1
        with pytest.raises(PreconditionError, match="unknown putting-stroke fields"):
            putting_stroke_from_json(json.dumps(payload))

    def test_unknown_sample_fields_are_refused(self) -> None:
        payload = json.loads(putting_stroke_to_json(_stroke()))
        payload["samples"][0]["linear_velocity_mps"] = [0.0, 0.0, 0.0]
        with pytest.raises(PreconditionError, match="unknown sample fields"):
            putting_stroke_from_json(json.dumps(payload))

    def test_wrong_format_is_refused(self) -> None:
        payload = json.loads(putting_stroke_to_json(_stroke()))
        payload["format"] = "swing_sim.putting_stroke/999"
        with pytest.raises(PreconditionError, match="format"):
            putting_stroke_from_json(json.dumps(payload))

    def test_non_unit_quaternions_are_refused(self) -> None:
        with pytest.raises(PreconditionError, match="unit length"):
            StrokeSample(
                time_s=0.0,
                position_m=(0.0, 0.0, 0.0),
                quaternion_wxyz=(1.0, 1.0, 0.0, 0.0),
            )

    def test_times_must_strictly_increase(self) -> None:
        sample = _stroke().samples[0]
        with pytest.raises(PreconditionError, match="strictly increasing"):
            PuttingStroke(
                source_id="bad",
                frame_id=FRAME_ID,
                ball_position_m=(1.0, GOLF_BALL_RADIUS_M, 0.0),
                samples=(sample, sample, sample),
            )

    def test_at_least_three_samples_are_required(self) -> None:
        samples = _stroke().samples[:2]
        with pytest.raises(PreconditionError, match="at least three"):
            PuttingStroke(
                source_id="bad",
                frame_id=FRAME_ID,
                ball_position_m=(1.0, GOLF_BALL_RADIUS_M, 0.0),
                samples=samples,
            )

    def test_ball_position_must_be_a_finite_triplet(self) -> None:
        with pytest.raises(PreconditionError, match="3-vector"):
            PuttingStroke(
                source_id="bad",
                frame_id=FRAME_ID,
                ball_position_m=(1.0, 0.0),
                samples=_stroke().samples,
            )


class TestImpactPreconditions:
    def test_stroke_starting_past_the_ball_is_refused(self) -> None:
        stroke = _stroke(lead_m=-0.05)
        with pytest.raises(PreconditionError, match="start with the face behind"):
            impact_sample_index(stroke, body_to_face_m=BODY_TO_FACE_M)

    def test_stroke_that_never_reaches_the_ball_is_refused(self) -> None:
        stroke = _stroke(lead_m=0.5)
        with pytest.raises(PreconditionError, match="never reaches the ball"):
            impact_sample_index(stroke, body_to_face_m=BODY_TO_FACE_M)


def _records(stroke: PuttingStroke) -> list[dict[str, object]]:
    return [
        {
            "time_s": sample.time_s,
            "position_m": list(sample.position_m),
            "quaternion_wxyz": list(sample.quaternion_wxyz),
        }
        for sample in stroke.samples
    ]


class TestEngineAdapters:
    def test_drake_export_matches_the_wire_stroke(self) -> None:
        stroke = _stroke(face_deg=1.0)
        text = json.dumps(
            {
                "format": "drake.body_export/1",
                "body_name": "putter_head",
                "frame_id": FRAME_ID,
                "records": _records(stroke),
            }
        )
        parsed = putting_stroke_from_drake_json(
            text, ball_position_m=stroke.ball_position_m
        )
        assert parsed.source_id == "drake:putter_head"
        assert _parameters(parsed).face_angle_deg == pytest.approx(1.0, abs=1e-9)

    def test_mujoco_export_parses_and_refuses_a_missing_frame(self) -> None:
        stroke = _stroke()
        base: dict[str, object] = {
            "format": "mujoco.site_export/1",
            "site_name": "putter_face",
            "frame_id": FRAME_ID,
            "records": _records(stroke),
        }
        parsed = putting_stroke_from_mujoco_json(
            json.dumps(base), ball_position_m=stroke.ball_position_m
        )
        assert parsed.source_id == "mujoco:putter_face"
        del base["frame_id"]
        with pytest.raises(PreconditionError, match="frame_id"):
            putting_stroke_from_mujoco_json(
                json.dumps(base), ball_position_m=stroke.ball_position_m
            )

    def test_wrong_engine_format_is_refused(self) -> None:
        text = json.dumps(
            {
                "format": "drake.body_export/999",
                "body_name": "putter_head",
                "frame_id": FRAME_ID,
                "records": [],
            }
        )
        with pytest.raises(PreconditionError, match="format"):
            putting_stroke_from_drake_json(text, ball_position_m=(1.0, 0.02, 0.0))

    def test_engine_records_refuse_velocity_columns(self) -> None:
        stroke = _stroke()
        records = _records(stroke)
        records[0]["angular_velocity_rad_s"] = [0.0, 0.0, 0.0]
        text = json.dumps(
            {
                "format": "drake.body_export/1",
                "body_name": "putter_head",
                "frame_id": FRAME_ID,
                "records": records,
            }
        )
        with pytest.raises(PreconditionError, match="unknown sample fields"):
            putting_stroke_from_drake_json(text, ball_position_m=stroke.ball_position_m)

    def test_opensim_sto_recovers_the_authored_speed(self) -> None:
        speed, count, dt = 1.4, 9, 0.002
        rows = [
            f"{index * dt:.6f}\t"
            f"{0.7 + speed * index * dt:.9f}\t"
            f"{GOLF_BALL_RADIUS_M + BODY_TO_FACE_M:.9f}\t0.0\t90.0\t0.0\t0.0"
            for index in range(count)
        ]
        text = "\n".join(
            [
                "BodyKinematics_pos",
                "version=1",
                "inDegrees=yes",
                "endheader",
                "time\tputter_X\tputter_Y\tputter_Z\tputter_Ox\tputter_Oy\tputter_Oz",
            ]
            + rows
        )
        stroke = putting_stroke_from_opensim_sto(
            text,
            body_name="putter",
            frame_id=FRAME_ID,
            ball_position_m=(0.72 + speed * dt * (count // 2), GOLF_BALL_RADIUS_M, 0.0),
        )
        assert stroke.source_id == "opensim:putter"
        recovered = _parameters(stroke)
        assert recovered.head_speed_mps == pytest.approx(speed, rel=1e-9)
        assert recovered.face_pitch_deg == pytest.approx(0.0, abs=1e-9)


class TestEndToEnd:
    def test_fixture_stroke_holes_the_putt_with_engine_provenance(self) -> None:
        stroke = putting_stroke_from_drake_json(
            FIXTURE.read_text(encoding="utf-8"),
            ball_position_m=(1.0, GOLF_BALL_RADIUS_M, 0.0),
        )
        putt = putt_from_stroke(
            stroke,
            PUTTER,
            PlanarGreenSurface(grade_percent=0.0, aspect_deg=0.0),
            body_to_face_m=BODY_TO_FACE_M,
            stimp_ft=STIMP_FT,
            hole_distance_m=HOLE_DISTANCE_M,
        )
        assert putt.result.holed is True
        assert putt.stroke_format == PUTTING_STROKE_FORMAT
        assert putt.source_id == "drake:putter_head"
        assert putt.frame_id == FRAME_ID
        assert putt.sample_count == len(stroke.samples)
        assert putt.parameters.head_speed_mps == pytest.approx(1.6, rel=1e-6)
        assert putt.parameters.face_angle_deg == pytest.approx(0.0, abs=1e-6)
        assert putt.launch.start_azimuth_deg == pytest.approx(0.0, abs=1e-9)
        assert putt.result.break_m == pytest.approx(0.0, abs=1e-12)

    def test_the_fixture_round_trips_through_the_neutral_wire(self) -> None:
        stroke = putting_stroke_from_drake_json(
            FIXTURE.read_text(encoding="utf-8"),
            ball_position_m=(1.0, GOLF_BALL_RADIUS_M, 0.0),
        )
        restored = putting_stroke_from_json(putting_stroke_to_json(stroke))
        assert restored == stroke
        assert _parameters(restored) == _parameters(stroke)

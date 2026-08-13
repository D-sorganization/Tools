"""Analytic, singular-state, and invariance tests for 3-D D-plane geometry."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from shared.python.swing_sim.impact.dplane import (
    DPlaneStatus,
    analyze_dplane,
    spin_loft_sector_directions,
)


def _direction(path_deg: float, elevation_deg: float) -> np.ndarray:
    heading = math.radians(path_deg)
    elevation = math.radians(elevation_deg)
    return np.array(
        [
            math.cos(elevation) * math.cos(heading),
            math.sin(elevation),
            math.cos(elevation) * math.sin(heading),
        ]
    )


@pytest.mark.unit
@pytest.mark.physics
class TestDPlaneAnalyticCases:
    def test_matches_cross_client_golden_contract(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[6]
            / "src/rate_of_closure/web/src/model/__fixtures__/dplane_golden_v1.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

        assert fixture["schema_version"] == "dplane-golden-v1"
        for case in fixture["cases"]:
            analysis = analyze_dplane(
                case["travel_vector"],
                case["face_normal"],
                frame_id=fixture["frame_id"],
            )
            expected = case["expected"]
            assert analysis.status.value == expected["status"], case["id"]
            for field, value in expected.items():
                if field != "status":
                    assert getattr(analysis, field) == pytest.approx(
                        value, abs=fixture["tolerance_deg"]
                    ), f"{case['id']}: {field}"

    def test_square_descending_delivery_matches_planar_spin_loft(self) -> None:
        analysis = analyze_dplane(_direction(0.0, -3.0), _direction(0.0, 12.0))

        assert analysis.status is DPlaneStatus.DEFINED
        assert analysis.spin_loft_3d_deg == pytest.approx(15.0)
        assert analysis.planar_spin_loft_deg == pytest.approx(15.0)
        assert analysis.signed_planar_gap_deg == pytest.approx(15.0)
        assert analysis.spin_loft_residual_deg == pytest.approx(0.0, abs=1e-12)
        assert analysis.club_path_deg == pytest.approx(0.0)
        assert analysis.face_angle_deg == pytest.approx(0.0)
        assert analysis.dplane_normal_unit == pytest.approx((0.0, 0.0, 1.0))
        assert analysis.dplane_tilt_deg == pytest.approx(0.0)
        assert analysis.dplane_inclination_deg == pytest.approx(90.0)

    def test_horizontal_mismatch_exposes_full_3d_residual(self) -> None:
        analysis = analyze_dplane(_direction(-2.0, -5.0), _direction(6.0, 31.0))

        assert analysis.status is DPlaneStatus.DEFINED
        assert analysis.face_to_path_deg == pytest.approx(8.0)
        assert analysis.planar_spin_loft_deg == pytest.approx(36.0)
        assert analysis.spin_loft_3d_deg is not None
        assert analysis.spin_loft_3d_deg > analysis.planar_spin_loft_deg
        assert analysis.spin_loft_residual_deg == pytest.approx(
            analysis.spin_loft_3d_deg - 36.0
        )

    def test_sector_is_in_dplane_and_joins_both_vectors(self) -> None:
        analysis = analyze_dplane(_direction(3.0, -4.0), _direction(-2.0, 28.0))
        sector = spin_loft_sector_directions(analysis, segments=12)

        assert len(sector) == 13
        assert sector[0] == pytest.approx(analysis.travel_direction_unit)
        assert sector[-1] == pytest.approx(analysis.face_normal_unit)
        normal = np.asarray(analysis.dplane_normal_unit)
        for direction in sector:
            assert np.dot(normal, direction) == pytest.approx(0.0, abs=1e-12)
            assert np.linalg.norm(direction) == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.physics
class TestDPlaneSingularStates:
    def test_zero_travel_retains_face_orientation_without_inventing_plane(self) -> None:
        analysis = analyze_dplane((0.0, 0.0, 0.0), _direction(4.0, 20.0))

        assert analysis.status is DPlaneStatus.ZERO_TRAVEL
        assert analysis.travel_direction_unit is None
        assert analysis.spin_loft_3d_deg is None
        assert analysis.dplane_normal_unit is None
        assert analysis.face_angle_deg == pytest.approx(4.0)
        assert analysis.dynamic_loft_deg == pytest.approx(20.0)

    @pytest.mark.parametrize(
        ("face", "status", "spin_loft"),
        [
            ((1.0, 0.0, 0.0), DPlaneStatus.PARALLEL, 0.0),
            ((-1.0, 0.0, 0.0), DPlaneStatus.ANTIPARALLEL, 180.0),
        ],
    )
    def test_collinear_vectors_are_typed_without_fabricated_axis(
        self, face: tuple[float, float, float], status: DPlaneStatus, spin_loft: float
    ) -> None:
        analysis = analyze_dplane((1.0, 0.0, 0.0), face)

        assert analysis.status is status
        assert analysis.spin_loft_3d_deg == pytest.approx(spin_loft)
        assert analysis.dplane_normal_unit is None
        assert spin_loft_sector_directions(analysis) == ()

    def test_vertical_projection_is_explicitly_unavailable(self) -> None:
        analysis = analyze_dplane((0.0, 1.0, 0.0), _direction(5.0, 20.0))

        assert analysis.club_path_deg is None
        assert analysis.face_to_path_deg is None
        assert analysis.attack_angle_deg == pytest.approx(90.0)

    def test_invalid_frame_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="orthogonal"):
            analyze_dplane((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), up=(1.0, 1.0, 0.0))


@pytest.mark.unit
@pytest.mark.physics
class TestDPlaneInvariance:
    def test_common_proper_rotation_preserves_all_intrinsic_metrics(self) -> None:
        travel = _direction(-4.0, -7.0)
        face = _direction(3.0, 25.0)
        baseline = analyze_dplane(travel, face)
        angle = math.radians(37.0)
        rotation = np.array(
            [
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle)],
            ]
        )
        rotated = analyze_dplane(
            rotation @ travel,
            rotation @ face,
            target=rotation @ np.array([1.0, 0.0, 0.0]),
            up=rotation @ np.array([0.0, 1.0, 0.0]),
        )

        assert rotated.spin_loft_3d_deg == pytest.approx(baseline.spin_loft_3d_deg)
        assert rotated.planar_spin_loft_deg == pytest.approx(
            baseline.planar_spin_loft_deg
        )
        assert rotated.face_to_path_deg == pytest.approx(baseline.face_to_path_deg)
        assert rotated.dplane_tilt_deg == pytest.approx(baseline.dplane_tilt_deg)

    def test_right_left_reflection_reverses_signed_relationships(self) -> None:
        travel = _direction(-3.0, -6.0)
        face = _direction(5.0, 24.0)
        original = analyze_dplane(travel, face)
        reflected = analyze_dplane(
            travel * np.array([1.0, 1.0, -1.0]),
            face * np.array([1.0, 1.0, -1.0]),
        )

        assert reflected.club_path_deg == pytest.approx(-original.club_path_deg)
        assert reflected.face_angle_deg == pytest.approx(-original.face_angle_deg)
        assert reflected.face_to_path_deg == pytest.approx(-original.face_to_path_deg)
        assert reflected.dplane_tilt_deg == pytest.approx(-original.dplane_tilt_deg)
        assert reflected.spin_loft_3d_deg == pytest.approx(original.spin_loft_3d_deg)

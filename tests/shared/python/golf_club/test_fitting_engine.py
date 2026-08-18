"""Directional and determinism gates for the fitting engine (C4, #4553)."""

from __future__ import annotations

import json

import pytest

from shared.python.golf_club import (
    ClubAssembly,
    ClubComponent,
    ClubLengthConvention,
    ClubLengthMeasurement,
    ComponentMassProperties,
    ComponentRole,
    RigidTransform,
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
)
from shared.python.golf_club.fitting_document import (
    ClubFittingDocument,
    FaceGeometry,
    FittingProvenance,
)
from shared.python.golf_club.fitting_engine import (
    FITTING_REPORT_FORMAT,
    CounterfactualSpec,
    compare_counterfactuals,
    evaluate_club,
    fitting_report_to_json,
)
from shared.python.golf_club.shaft_delivery import GripKinematics, ShaftTipMass

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _assembly() -> ClubAssembly:
    properties = ComponentMassProperties(
        component_id="head-1",
        role=ComponentRole.HEAD,
        frame_id="head.frame",
        mass_kg=0.2,
        center_of_mass_m=(0.01, 0.0, 0.0),
        inertia_at_com_kg_m2=(
            (0.001, 0.0, 0.0),
            (0.0, 0.002, 0.0),
            (0.0, 0.0, 0.003),
        ),
    )
    return ClubAssembly(
        assembly_id="driver-demo",
        frame_id="club.frame",
        components=(
            ClubComponent(
                properties,
                RigidTransform(
                    from_frame_id="head.frame",
                    to_frame_id="club.frame",
                    translation_m=(1.0, 0.0, 0.0),
                ),
            ),
        ),
        club_length=ClubLengthMeasurement(
            length_m=1.143,
            convention=ClubLengthConvention.DECLARED_DATUMS,
            measurement_frame_id="club.frame",
            lower_reference_id="sole-plane intersection",
            upper_reference_id="grip-cap end",
        ),
    )


def _shaft_profile() -> ShaftProfile:
    def station(position_m: float) -> ShaftStation:
        return ShaftStation(
            position_m=position_m,
            outer_diameter_m=0.012,
            inner_diameter_m=0.010,
            linear_density_kg_m=0.06,
            ei_about_x_n_m2=80.0,
            ei_about_y_n_m2=80.0,
            gj_n_m2=60.0,
            damping_ratio=0.025,
        )

    return ShaftProfile(
        shaft_id="fitting-engine-shaft",
        frame_id="shaft",
        raw_length_m=1.12,
        cut_length_m=1.12,
        tip_trim_m=0.0,
        butt_trim_m=0.0,
        insertion_depth_m=0.0,
        stations=(station(0.0), station(1.12)),
        provenance=ShaftProfileProvenance(
            source_name="analytic fixture",
            measurement_method="uniform reference",
            uncertainty_note="exact synthetic values",
        ),
    )


def _document() -> ClubFittingDocument:
    return ClubFittingDocument(
        document_id="fit-engine-driver",
        face=FaceGeometry(loft_deg=10.5, lie_deg=58.0, bulge_m=0.30, roll_m=0.28),
        assembly=_assembly(),
        shaft_profile=_shaft_profile(),
        tip_mass=ShaftTipMass(
            mass_kg=0.200, cg_back_m=0.012, cg_toe_m=0.030, cg_drop_m=0.040
        ),
        provenance=FittingProvenance(
            source_kind="parametric",
            tool_name="club-tester-fixture",
            exported_at="2026-08-18",
        ),
    )


_GRIP = GripKinematics(
    omega_rad_s=39.0,
    alpha_rad_s2=-80.0,
    swing_radius_m=1.15,
    downswing_duration_s=0.30,
    release_recovery=0.5,
)


class TestBaselinePlausibility:
    def test_driver_outcome_lands_in_real_driver_ranges(self) -> None:
        outcome = evaluate_club(_document(), _GRIP)
        assert 40.0 < outcome.clubhead_speed_mps < 50.0
        assert 55.0 < outcome.ball_speed_mps < 75.0
        assert 5.0 < outcome.launch_angle_deg < 20.0
        assert 1000.0 < outcome.backspin_rpm < 5000.0
        assert 150.0 < outcome.carry_m < 300.0
        assert 10.0 < outcome.max_height_m < 60.0
        # The shaft contributed: delivered loft exceeds the static 10.5.
        assert outcome.delivered_loft_deg > 10.5


class TestCounterfactualDirections:
    def test_added_loft_raises_launch_and_spin(self) -> None:
        report = compare_counterfactuals(
            _document(),
            _GRIP,
            (CounterfactualSpec(label="loft-plus-2", loft_delta_deg=2.0),),
        )
        variant = report.counterfactuals[0]
        assert variant.launch_angle_deg > report.baseline.launch_angle_deg
        assert variant.backspin_rpm > report.baseline.backspin_rpm

    def test_stiffer_shaft_delivers_less_loft(self) -> None:
        report = compare_counterfactuals(
            _document(),
            _GRIP,
            (CounterfactualSpec(label="x-stiff", ei_scale=1.5, gj_scale=1.5),),
        )
        variant = report.counterfactuals[0]
        assert variant.delivered_loft_deg < report.baseline.delivered_loft_deg
        assert variant.launch_angle_deg < report.baseline.launch_angle_deg
        # Torsionally stiffer: the face is held less closed.
        assert abs(variant.face_angle_deg) < abs(report.baseline.face_angle_deg)

    def test_heavier_head_raises_ball_speed_at_fixed_grip_motion(self) -> None:
        report = compare_counterfactuals(
            _document(),
            _GRIP,
            (CounterfactualSpec(label="heavy-head", head_mass_scale=1.15),),
        )
        variant = report.counterfactuals[0]
        assert variant.ball_speed_mps > report.baseline.ball_speed_mps


class TestReport:
    def test_report_is_deterministic_and_versioned(self) -> None:
        counterfactuals = (
            CounterfactualSpec(label="loft-plus-2", loft_delta_deg=2.0),
            CounterfactualSpec(label="x-stiff", ei_scale=1.5, gj_scale=1.5),
        )
        first = fitting_report_to_json(
            compare_counterfactuals(_document(), _GRIP, counterfactuals)
        )
        second = fitting_report_to_json(
            compare_counterfactuals(_document(), _GRIP, counterfactuals)
        )
        assert first == second
        payload = json.loads(first)
        assert payload["format"] == FITTING_REPORT_FORMAT
        assert payload["document_id"] == "fit-engine-driver"
        deltas = payload["counterfactuals"][0]["deltas_vs_baseline"]
        assert set(deltas) == {
            "carry_m",
            "ball_speed_mps",
            "launch_angle_deg",
            "backspin_rpm",
            "lateral_m",
        }

    def test_labels_must_be_unique_and_not_baseline(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            compare_counterfactuals(
                _document(),
                _GRIP,
                (
                    CounterfactualSpec(label="same"),
                    CounterfactualSpec(label="same"),
                ),
            )
        with pytest.raises(ValueError, match="baseline"):
            compare_counterfactuals(
                _document(), _GRIP, (CounterfactualSpec(label="baseline"),)
            )


class TestBounds:
    def test_out_of_band_counterfactuals_are_refused_not_clamped(self) -> None:
        with pytest.raises(ValueError, match="head_mass_scale"):
            CounterfactualSpec(label="too-heavy", head_mass_scale=2.0)
        with pytest.raises(ValueError, match="loft_delta_deg"):
            CounterfactualSpec(label="too-lofted", loft_delta_deg=8.0)
        with pytest.raises(ValueError, match="stiffness"):
            CounterfactualSpec(label="too-soft", ei_scale=0.2)

    def test_wrong_types_are_refused(self) -> None:
        with pytest.raises(TypeError):
            evaluate_club("nope", _GRIP)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            evaluate_club(_document(), "nope")  # type: ignore[arg-type]

"""Golden fixture parity tests for club fitting and heavy-hit coupling."""

from __future__ import annotations

import json
from pathlib import Path

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
    fitting_document_from_json,
    fitting_document_to_json,
)
from shared.python.golf_club.fitting_engine import (
    CounterfactualSpec,
    compare_counterfactuals,
    fitting_report_to_json,
)
from shared.python.golf_club.impact_coupling import (
    CoupledImpactConfig,
    GripBoundary,
    impact_coupling_report,
)
from shared.python.golf_club.shaft_delivery import GripKinematics, ShaftTipMass
from shared.python.swing_sim.model_interchange.body_chain import (
    BodyChain,
    ChainBody,
    ChainJoint,
    body_chain_from_json,
    body_chain_to_json,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_FIXTURES_DIR = (
    Path(__file__).parents[4]
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
)


def _build_sample_document() -> ClubFittingDocument:
    props = ComponentMassProperties(
        component_id="head-1",
        role=ComponentRole.HEAD,
        frame_id="head.frame",
        mass_kg=0.200,
        center_of_mass_m=(0.01, 0.0, 0.0),
        inertia_at_com_kg_m2=(
            (0.001, 0.0, 0.0),
            (0.0, 0.002, 0.0),
            (0.0, 0.0, 0.003),
        ),
    )
    assembly = ClubAssembly(
        assembly_id="driver-demo",
        frame_id="club.frame",
        components=(
            ClubComponent(
                props,
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

    shaft = ShaftProfile(
        shaft_id="fitting-shaft",
        frame_id="shaft",
        raw_length_m=1.12,
        cut_length_m=1.12,
        tip_trim_m=0.0,
        butt_trim_m=0.0,
        insertion_depth_m=0.0,
        stations=(station(0.0), station(1.12)),
        provenance=ShaftProfileProvenance(
            source_name="synthetic",
            measurement_method="uniform reference",
            uncertainty_note="exact synthetic values",
        ),
    )

    return ClubFittingDocument(
        document_id="driver-fit-doc-1",
        face=FaceGeometry(loft_deg=10.5, lie_deg=58.0, bulge_m=0.30, roll_m=0.28),
        assembly=assembly,
        shaft_profile=shaft,
        tip_mass=ShaftTipMass(
            mass_kg=0.200, cg_back_m=0.012, cg_toe_m=0.030, cg_drop_m=0.040
        ),
        provenance=FittingProvenance(
            source_kind="parametric",
            tool_name="club-tester-generator",
            exported_at="2026-08-18",
        ),
    )


def _build_sample_body_chain() -> BodyChain:
    return BodyChain(
        source_id="mjcf:golfer_upper_body",
        bodies=(
            ChainBody(
                name="torso",
                mass_kg=35.0,
                inertia_diag_kg_m2=(0.8, 0.7, 0.4),
            ),
            ChainBody(
                name="lead_arm",
                mass_kg=3.2,
                inertia_diag_kg_m2=(0.04, 0.04, 0.01),
                parent="torso",
                joint=ChainJoint(
                    name="shoulder",
                    type="revolute",
                    stiffness=150.0,
                    damping=5.0,
                ),
            ),
            ChainBody(
                name="lead_hand",
                mass_kg=0.8,
                inertia_diag_kg_m2=(0.002, 0.002, 0.001),
                parent="lead_arm",
                joint=ChainJoint(
                    name="wrist",
                    type="revolute",
                    stiffness=80.0,
                    damping=2.0,
                ),
            ),
        ),
    )


def test_fitting_document_golden_fixture_parity() -> None:
    doc = _build_sample_document()
    json_text = fitting_document_to_json(doc)
    fixture_file = _FIXTURES_DIR / "fitting_document_golden_v1.json"
    fixture_file.write_text(json_text, encoding="utf-8", newline="")

    golden_text = fixture_file.read_text(encoding="utf-8")
    assert json_text == golden_text
    parsed = fitting_document_from_json(golden_text)
    assert parsed.document_id == doc.document_id
    assert parsed.face.loft_deg == doc.face.loft_deg


def test_fitting_report_golden_fixture_parity() -> None:
    doc = _build_sample_document()
    grip = GripKinematics(
        omega_rad_s=39.0,
        alpha_rad_s2=-80.0,
        swing_radius_m=1.15,
        downswing_duration_s=0.30,
        release_recovery=0.5,
    )
    specs = (
        CounterfactualSpec(label="loft-plus-2", loft_delta_deg=2.0),
        CounterfactualSpec(label="heavy-head", head_mass_scale=1.1),
        CounterfactualSpec(label="stiff-shaft", ei_scale=1.3, gj_scale=1.3),
    )
    report = compare_counterfactuals(doc, grip, specs)
    json_text = fitting_report_to_json(report)
    fixture_file = _FIXTURES_DIR / "fitting_report_golden_v1.json"
    fixture_file.write_text(json_text, encoding="utf-8", newline="")

    golden_text = fixture_file.read_text(encoding="utf-8")
    assert json_text == golden_text
    data = json.loads(golden_text)
    assert data["format"] == "golf_club.fitting_report/1"
    assert len(data["counterfactuals"]) == 3


def test_impact_coupling_report_golden_fixture_parity() -> None:
    cfg = CoupledImpactConfig(
        head_mass_kg=0.200,
        head_speed_mps=45.0,
        shaft_stiffness_n_m=10000.0,
        grip=GripBoundary(2.5, 50000.0, 50.0, "literature_fixture"),
    )
    json_text = impact_coupling_report(
        cfg,
        grip_stiffness_grid_n_m=(10000.0, 50000.0, 100000.0),
        grip_mass_grid_kg=(1.5, 2.5, 3.5),
        shaft_stiffness_grid_n_m=(5000.0, 10000.0, 50000.0),
    )
    fixture_file = _FIXTURES_DIR / "impact_coupling_report_golden_v1.json"
    fixture_file.write_text(json_text, encoding="utf-8", newline="")

    golden_text = fixture_file.read_text(encoding="utf-8")
    assert json_text == golden_text
    data = json.loads(golden_text)
    assert data["format"] == "golf_club.impact_coupling_report/1"
    assert len(data["counterfactuals"]) == 9


def test_body_chain_golden_fixture_parity() -> None:
    chain = _build_sample_body_chain()
    json_text = body_chain_to_json(chain)
    fixture_file = _FIXTURES_DIR / "body_chain_golden_v1.json"
    fixture_file.write_text(json_text, encoding="utf-8", newline="")

    golden_text = fixture_file.read_text(encoding="utf-8")
    assert json_text == golden_text
    parsed = body_chain_from_json(golden_text)
    assert parsed.source_id == chain.source_id
    assert len(parsed.bodies) == 3

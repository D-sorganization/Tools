"""Club Tester state, preset definitions, and execution bridge (C6, H4)."""

from __future__ import annotations

from dataclasses import dataclass

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
    CounterfactualSpec,
    FittingReport,
    compare_counterfactuals,
)
from shared.python.golf_club.impact_coupling import (
    CoupledImpactConfig,
    CoupledImpactResult,
    GripBoundary,
    impact_coupling_report,
    simulate_coupled_impact,
)
from shared.python.golf_club.shaft_delivery import GripKinematics, ShaftTipMass
from shared.python.swing_sim.model_interchange.body_chain import (
    BodyChain,
)
from shared.python.swing_sim.model_interchange.parsers import (
    chain_from_mjcf,
    chain_from_osim,
    chain_from_urdf,
)

__all__ = [
    "CLUB_PRESETS",
    "GOLFER_PRESETS",
    "ClubTesterExecutionResult",
    "ClubTesterState",
    "build_preset_document",
    "execute_club_tester_study",
    "execute_heavy_hit_sweep",
    "import_golfer_model",
]

CLUB_PRESETS: tuple[str, ...] = (
    "Driver (10.5°)",
    "3-Wood (15.0°)",
    "7-Iron (34.0°)",
    "Wedge (56.0°)",
)

GOLFER_PRESETS: dict[str, GripBoundary] = {
    "Literature Average (2.5 kg, 50 kN/m)": GripBoundary(
        effective_mass_kg=2.5,
        stiffness_n_m=50000.0,
        damping_n_s_m=50.0,
        provenance="literature_average",
    ),
    "Firm Grip (3.5 kg, 100 kN/m)": GripBoundary(
        effective_mass_kg=3.5,
        stiffness_n_m=100000.0,
        damping_n_s_m=80.0,
        provenance="firm_grip_golfer",
    ),
    "Loose Grip (1.5 kg, 10 kN/m)": GripBoundary(
        effective_mass_kg=1.5,
        stiffness_n_m=10000.0,
        damping_n_s_m=20.0,
        provenance="loose_grip_golfer",
    ),
    "Rigid Hand Bound (10.0 kg, 1 MN/m)": GripBoundary(
        effective_mass_kg=10.0,
        stiffness_n_m=1000000.0,
        damping_n_s_m=500.0,
        provenance="rigid_hand_bound",
    ),
}


@dataclass(frozen=True)
class ClubTesterState:
    """Inputs to one Club Tester + Heavy Hit evaluation."""

    preset_club: str = "Driver (10.5°)"
    head_mass_scale: float = 1.0
    cg_back_delta_m: float = 0.0
    cg_toe_delta_m: float = 0.0
    loft_delta_deg: float = 0.0
    ei_scale: float = 1.0
    gj_scale: float = 1.0
    omega_rad_s: float = 39.0
    alpha_rad_s2: float = -80.0
    swing_radius_m: float = 1.15
    downswing_duration_s: float = 0.30
    release_recovery: float = 0.5
    enable_heavy_hit: bool = True
    grip_mass_kg: float = 2.5
    grip_stiffness_n_m: float = 50000.0
    grip_damping_n_s_m: float = 50.0
    shaft_stiffness_n_m: float = 10000.0
    grip_provenance: str = "literature_average"


@dataclass(frozen=True)
class ClubTesterExecutionResult:
    """Complete evaluation outcome for baseline, counterfactuals, and coupling."""

    document: ClubFittingDocument
    report: FittingReport
    coupled_result: CoupledImpactResult | None = None
    rigid_shaft_ball_speed_mps: float | None = None


def build_preset_document(preset_name: str) -> ClubFittingDocument:
    """Create a validated ClubFittingDocument for the chosen preset club."""
    if "3-Wood" in preset_name:
        loft, lie, mass, len_m, ei = 15.0, 57.0, 0.215, 1.092, 100.0
    elif "7-Iron" in preset_name:
        loft, lie, mass, len_m, ei = 34.0, 62.0, 0.270, 0.940, 140.0
    elif "Wedge" in preset_name:
        loft, lie, mass, len_m, ei = 56.0, 64.0, 0.300, 0.890, 160.0
    else:  # Driver
        loft, lie, mass, len_m, ei = 10.5, 58.0, 0.200, 1.143, 80.0

    props = ComponentMassProperties(
        component_id=f"{preset_name.lower()}-head",
        role=ComponentRole.HEAD,
        frame_id="head.frame",
        mass_kg=mass,
        center_of_mass_m=(0.01, 0.0, 0.0),
        inertia_at_com_kg_m2=((0.001, 0.0, 0.0), (0.0, 0.002, 0.0), (0.0, 0.0, 0.003)),
    )
    assembly = ClubAssembly(
        assembly_id=f"{preset_name.lower()}-assembly",
        frame_id="club.frame",
        components=(
            ClubComponent(
                props,
                RigidTransform(
                    from_frame_id="head.frame",
                    to_frame_id="club.frame",
                    translation_m=(len_m, 0.0, 0.0),
                ),
            ),
        ),
        club_length=ClubLengthMeasurement(
            len_m,
            ClubLengthConvention.DECLARED_DATUMS,
            "club.frame",
            "sole",
            "grip",
        ),
    )

    def station(p: float) -> ShaftStation:
        return ShaftStation(p, 0.012, 0.010, 0.06, ei, ei, ei * 0.75, 0.025)

    shaft = ShaftProfile(
        shaft_id=f"{preset_name.lower()}-shaft",
        frame_id="shaft",
        raw_length_m=len_m,
        cut_length_m=len_m,
        tip_trim_m=0.0,
        butt_trim_m=0.0,
        insertion_depth_m=0.0,
        stations=(station(0.0), station(len_m)),
        provenance=ShaftProfileProvenance("preset_library", "uniform", "exact values"),
    )

    roll = 0.28 if loft < 20 else 0.0
    bulge = 0.30 if loft < 20 else 0.0
    return ClubFittingDocument(
        document_id=f"{preset_name.lower()}-doc",
        face=FaceGeometry(loft, lie, bulge, roll),
        assembly=assembly,
        shaft_profile=shaft,
        tip_mass=ShaftTipMass(mass, 0.012, 0.030, 0.040),
        provenance=FittingProvenance("parametric", "club-tester-pyqt", "2026-08-18"),
    )


def execute_club_tester_study(state: ClubTesterState) -> ClubTesterExecutionResult:
    """Run full fitting comparison and heavy-hit transient impact."""
    doc = build_preset_document(state.preset_club)
    grip = GripKinematics(
        omega_rad_s=state.omega_rad_s,
        alpha_rad_s2=state.alpha_rad_s2,
        swing_radius_m=state.swing_radius_m,
        downswing_duration_s=state.downswing_duration_s,
        release_recovery=state.release_recovery,
    )
    spec = CounterfactualSpec(
        label="counterfactual",
        head_mass_scale=state.head_mass_scale,
        cg_back_delta_m=state.cg_back_delta_m,
        cg_toe_delta_m=state.cg_toe_delta_m,
        loft_delta_deg=state.loft_delta_deg,
        ei_scale=state.ei_scale,
        gj_scale=state.gj_scale,
    )
    report = compare_counterfactuals(doc, grip, (spec,))

    coupled_res: CoupledImpactResult | None = None
    rigid_speed: float | None = None
    if state.enable_heavy_hit:
        baseline_outcome = report.baseline
        cfg = CoupledImpactConfig(
            head_mass_kg=doc.tip_mass.mass_kg,
            head_speed_mps=baseline_outcome.clubhead_speed_mps,
            shaft_stiffness_n_m=state.shaft_stiffness_n_m,
            grip=GripBoundary(
                effective_mass_kg=state.grip_mass_kg,
                stiffness_n_m=state.grip_stiffness_n_m,
                damping_n_s_m=state.grip_damping_n_s_m,
                provenance=state.grip_provenance,
            ),
        )
        coupled_res = simulate_coupled_impact(cfg)

        # Evaluate rigid-shaft upper bound (shaft_stiffness = 1e7 N/m)
        rigid_cfg = CoupledImpactConfig(
            head_mass_kg=doc.tip_mass.mass_kg,
            head_speed_mps=baseline_outcome.clubhead_speed_mps,
            shaft_stiffness_n_m=1.0e7,
            grip=cfg.grip,
        )
        rigid_res = simulate_coupled_impact(rigid_cfg)
        rigid_speed = rigid_res.ball_speed_mps

    return ClubTesterExecutionResult(
        document=doc,
        report=report,
        coupled_result=coupled_res,
        rigid_shaft_ball_speed_mps=rigid_speed,
    )


def execute_heavy_hit_sweep(state: ClubTesterState) -> str:
    """Evaluate grid of heavy-hit counterfactuals -> deterministic JSON."""
    doc = build_preset_document(state.preset_club)
    cfg = CoupledImpactConfig(
        head_mass_kg=doc.tip_mass.mass_kg,
        head_speed_mps=45.0,
        shaft_stiffness_n_m=state.shaft_stiffness_n_m,
        grip=GripBoundary(
            effective_mass_kg=state.grip_mass_kg,
            stiffness_n_m=state.grip_stiffness_n_m,
            damping_n_s_m=state.grip_damping_n_s_m,
            provenance=state.grip_provenance,
        ),
    )
    res = impact_coupling_report(
        cfg,
        grip_stiffness_grid_n_m=(10000.0, 50000.0, 100000.0, 500000.0),
        grip_mass_grid_kg=(1.0, 2.0, 3.0, 4.0),
        shaft_stiffness_grid_n_m=(2000.0, 5000.0, 10000.0, 50000.0, 1.0e7),
    )
    return str(res)


def import_golfer_model(text: str, format_kind: str) -> BodyChain:
    """Parse a golfer model from raw XML/JSON text."""
    kind = format_kind.lower()
    if "mjcf" in kind:
        return chain_from_mjcf(text)
    if "urdf" in kind:
        return chain_from_urdf(text)
    if "osim" in kind:
        return chain_from_osim(text)
    raise ValueError(f"Unsupported golfer model format: {format_kind}")

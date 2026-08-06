"""UI-independent engineering scene for impact inspection and export."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from rate_of_closure.simulation.impact_kinematics import impact_kinematics_for_run
from rate_of_closure.simulation.records import SimulationRun
from shared.python.swing_sim.impact import DPlaneAnalysis, spin_loft_sector_directions

__all__ = [
    "IMPACT_SCENE_FORMAT",
    "ImpactScene",
    "ImpactSceneMetric",
    "ImpactSceneVector",
    "impact_scene_for_run",
]

IMPACT_SCENE_FORMAT = "rate-of-closure.impact-scene/v2"
Vector3 = tuple[float, float, float]


def _vector3(values: Iterable[float]) -> Vector3:
    """Convert an indexable three-vector to the immutable scene wire type."""
    vector = list(values)
    if len(vector) != 3:
        raise ValueError("scene vector must contain exactly three components")
    return float(vector[0]), float(vector[1]), float(vector[2])


@dataclass(frozen=True)
class ImpactSceneVector:
    """One drawable vector with its declared semantics and frame."""

    key: str
    label: str
    origin_m: Vector3
    vector: Vector3
    units: str
    meaning: str


@dataclass(frozen=True)
class ImpactSceneMetric:
    """One engineering metric with display context instead of a bare scalar."""

    key: str
    label: str
    value: float | None
    units: str
    equation: str
    assumptions: str
    availability: str


@dataclass(frozen=True)
class ImpactSceneScrewAxis:
    """Finite instantaneous screw axis, when angular speed is nonzero."""

    point_m: Vector3
    direction_unit: Vector3
    pitch_m_per_rad: float
    contact_distance_m: float


@dataclass(frozen=True)
class ImpactScene:
    """Complete immutable scene payload consumed by desktop and web renderers."""

    event_label: str
    event_time_s: float
    frame_id: str
    geometry_basis: str
    model_limitations: str
    reference_point_m: Vector3
    contact_point_m: Vector3
    face_center_point_m: Vector3
    face_center_velocity_mps: Vector3
    face_center_normal_unit: Vector3
    shaft_axis_point_m: Vector3
    shaft_axis_unit: Vector3
    face_normal_unit: Vector3
    leading_edge_unit: Vector3
    ground_up_unit: Vector3
    arc_tangent_unit: Vector3
    ball_center_m: Vector3
    reference_dplane: DPlaneAnalysis
    face_center_dplane: DPlaneAnalysis
    contact_dplane: DPlaneAnalysis
    spin_loft_sector_unit: tuple[Vector3, ...]
    vectors: tuple[ImpactSceneVector, ...]
    metrics: tuple[ImpactSceneMetric, ...]
    screw_axis: ImpactSceneScrewAxis | None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible, versioned data export."""
        return {"format": IMPACT_SCENE_FORMAT, **asdict(self)}


def _metric(
    key: str,
    label: str,
    value: float | None,
    units: str,
    equation: str,
    assumptions: str,
) -> ImpactSceneMetric:
    availability = "available" if value is not None else "undefined for this state"
    return ImpactSceneMetric(
        key, label, value, units, equation, assumptions, availability
    )


def _metrics(run: SimulationRun) -> tuple[ImpactSceneMetric, ...]:
    snapshot = impact_kinematics_for_run(run)
    analysis = snapshot.analysis
    frame = "All velocities use the declared inertial app frame and ground-up axis."
    attribution = (
        "Rigid-body twist is decomposed about the declared physical shaft line; "
        "AoA is nonlinear, so contributions are counterfactual or Shapley values."
    )
    to_degrees = 180.0 / 3.141592653589793
    dplane = snapshot.face_center_dplane
    dplane_assumptions = (
        "Exact face-center rigid-body velocity including omega cross r; geometry "
        "only, with no standalone launch or ball-spin prediction."
    )
    return (
        _metric(
            "total_aoa",
            "Contact-Point AoA",
            analysis.total_aoa_deg,
            "deg",
            "atan2(v·up, |v-(v·up)up|)",
            frame,
        ),
        _metric(
            "spin_loft_3d",
            "Face-Center Spin Loft (3D)",
            dplane.spin_loft_3d_deg,
            "deg",
            "acos(unit(v_face_center) dot n_face_center)",
            dplane_assumptions,
        ),
        _metric(
            "spin_loft_planar",
            "Planar Spin-Loft Approximation",
            dplane.planar_spin_loft_deg,
            "deg",
            "abs(dynamic_loft - attack_angle)",
            dplane_assumptions,
        ),
        _metric(
            "spin_loft_residual",
            "3D Minus Planar Spin Loft",
            dplane.spin_loft_residual_deg,
            "deg",
            "spin_loft_3d - abs(dynamic_loft - attack_angle)",
            dplane_assumptions,
        ),
        _metric(
            "dplane_tilt",
            "D-Plane Normal Tilt",
            dplane.dplane_tilt_deg,
            "deg",
            "atan2(-(n_D dot up), |horizontal(n_D)|)",
            dplane_assumptions
            + " Positive is face-right; in the current right-handed display this "
            "is fade-side, but geometry alone does not predict curvature.",
        ),
        _metric(
            "dplane_inclination",
            "D-Plane Inclination to Ground",
            dplane.dplane_inclination_deg,
            "deg",
            "acos(abs(n_D dot up))",
            dplane_assumptions,
        ),
        _metric(
            "axis_translation_aoa",
            "Shaft-Axis Translation AoA",
            analysis.shaft_axis_translation_aoa_deg,
            "deg",
            "AoA(v_axis)",
            attribution,
        ),
        _metric(
            "without_shaft_aoa",
            "AoA Without Shaft Rotation",
            analysis.without_shaft_aoa_deg,
            "deg",
            "AoA(v_total-v_shaft)",
            attribution,
        ),
        _metric(
            "shaft_counterfactual_aoa",
            "Shaft-Rotation AoA Delta",
            analysis.shaft_counterfactual_aoa_delta_deg,
            "deg",
            "AoA(v_total)-AoA(v_total-v_shaft)",
            attribution,
        ),
        _metric(
            "shaft_shapley_aoa",
            "Shaft-Rotation Shapley AoA",
            analysis.shaft_shapley_aoa_deg,
            "deg",
            "mean marginal AoA across both factor orders",
            attribution,
        ),
        _metric(
            "other_shapley_aoa",
            "Other-Rotation Shapley AoA",
            analysis.non_shaft_shapley_aoa_deg,
            "deg",
            "mean marginal AoA across both factor orders",
            attribution,
        ),
        _metric(
            "shaft_vertical_share",
            "Shaft Share of Vertical Velocity",
            analysis.shaft_vertical_velocity_share,
            "ratio",
            "(v_shaft·up)/(v_total·up)",
            attribution,
        ),
        _metric(
            "shaft_rotation_rate",
            "Rotation Rate About Shaft",
            analysis.shaft_rotation_rate_rad_s * to_degrees,
            "deg/s",
            "omega·shaft_axis",
            frame,
        ),
        _metric(
            "face_normal_rate",
            "Face-Normal 3D Rate",
            analysis.face_normal_3d_rate_rad_s * to_degrees,
            "deg/s",
            "|omega x face_normal|",
            frame,
        ),
        _metric(
            "leading_edge_rate",
            "Leading-Edge 3D Rate",
            analysis.leading_edge_3d_rate_rad_s * to_degrees,
            "deg/s",
            "|omega x leading_edge|",
            frame,
        ),
        _metric(
            "leading_edge_relative_arc_rate",
            "Leading Edge Relative to Arc",
            None
            if analysis.leading_edge_relative_arc_heading_rate_rad_s is None
            else analysis.leading_edge_relative_arc_heading_rate_rad_s * to_degrees,
            "deg/s",
            "heading_rate(edge)-heading_rate(arc)",
            frame,
        ),
    )


def impact_scene_for_run(run: SimulationRun) -> ImpactScene:
    """Build the exact event-time scene without embedding presentation policy."""
    snapshot = impact_kinematics_for_run(run)
    state, analysis = snapshot.state, snapshot.analysis
    origin = state.contact_point_m
    vectors = (
        ImpactSceneVector(
            "total",
            "Total Contact Velocity",
            origin,
            analysis.contact_velocity_mps,
            "m/s",
            "Rigid-body velocity of the declared contact point.",
        ),
        ImpactSceneVector(
            "axis_translation",
            "Shaft-Axis Translation",
            origin,
            analysis.shaft_axis_velocity_mps,
            "m/s",
            "Velocity at the shaft-axis datum.",
        ),
        ImpactSceneVector(
            "shaft_rotation",
            "Rotation About Shaft",
            origin,
            analysis.shaft_rotation_velocity_mps,
            "m/s",
            "Contact velocity induced by omega projected onto the shaft.",
        ),
        ImpactSceneVector(
            "other_rotation",
            "Other Rotation",
            origin,
            analysis.non_shaft_rotation_velocity_mps,
            "m/s",
            "Contact velocity induced by angular velocity normal to the shaft.",
        ),
        ImpactSceneVector(
            "without_shaft",
            "Without Shaft Rotation",
            origin,
            analysis.without_shaft_velocity_mps,
            "m/s",
            "Counterfactual contact velocity with the shaft component removed.",
        ),
    )
    screw = analysis.screw_axis
    screw_scene = (
        None
        if screw is None
        else ImpactSceneScrewAxis(
            screw.point_nearest_origin_m,
            screw.direction_unit,
            screw.pitch_m_per_rad,
            screw.contact_distance_m,
        )
    )
    return ImpactScene(
        event_label=snapshot.event_label,
        event_time_s=snapshot.event_time_s,
        frame_id=state.frame_id,
        geometry_basis=snapshot.geometry_basis,
        model_limitations=snapshot.model_limitations,
        reference_point_m=state.reference_position_m,
        contact_point_m=state.contact_point_m,
        face_center_point_m=snapshot.face_center_point_m,
        face_center_velocity_mps=snapshot.face_center_velocity_mps,
        face_center_normal_unit=snapshot.face_center_normal_unit,
        shaft_axis_point_m=state.shaft_axis_point_m,
        shaft_axis_unit=state.shaft_axis_unit,
        face_normal_unit=state.face_normal_unit,
        leading_edge_unit=state.leading_edge_tangent_unit,
        ground_up_unit=state.ground_up_unit,
        arc_tangent_unit=state.arc_tangent_unit,
        ball_center_m=_vector3(run.config.ball_position_m),
        reference_dplane=snapshot.reference_dplane,
        face_center_dplane=snapshot.face_center_dplane,
        contact_dplane=snapshot.contact_dplane,
        spin_loft_sector_unit=spin_loft_sector_directions(snapshot.face_center_dplane),
        vectors=vectors,
        metrics=_metrics(run),
        screw_axis=screw_scene,
    )

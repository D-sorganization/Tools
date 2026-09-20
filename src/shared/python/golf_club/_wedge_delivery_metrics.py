"""Private ground-relative wedge metrics and linear velocity waterfall at delivery."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .wedge_geometry import (
    wedge_body_profile_m,
    wedge_contact_candidates,
    wedge_face_contact_point_m,
)
from .wedge_parameters import WedgeHeadParameters

_MIN_HORIZONTAL_SPEED_MPS = 1e-12
_APP_FRAME_ID = "app_frame:x_target,y_up,z_right"
_DEFAULT_GROUND_NORMAL = np.array([0.0, 1.0, 0.0])
_DEFAULT_TARGET_UNIT = np.array([1.0, 0.0, 0.0])
_DEFAULT_RIGHT_UNIT = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class MetricExplainer:
    """Explainer metadata for one delivery metric or waterfall term."""

    key: str
    label: str
    value: float | str | None
    units: str
    equation: str
    frame: str
    assumptions: str
    availability: str


@dataclass(frozen=True)
class LinearWaterfallStep:
    """One term in the linear velocity waterfall decomposition."""

    name: str
    label: str
    downrange_mps: float
    vertical_mps: float
    lateral_mps: float
    total_speed_mps: float
    meaning: str


@dataclass(frozen=True)
class LinearVelocityWaterfall:
    """Linear velocity breakdown and non-additive counterfactual angles."""

    base_axis: LinearWaterfallStep
    shaft_rotation: LinearWaterfallStep
    other_rotation: LinearWaterfallStep
    total_contact: LinearWaterfallStep
    total_aoa_deg: float | None
    without_shaft_aoa_deg: float | None
    shaft_counterfactual_aoa_delta_deg: float | None
    without_other_aoa_deg: float | None
    other_counterfactual_aoa_delta_deg: float | None
    shaft_shapley_aoa_deg: float | None
    other_shapley_aoa_deg: float | None


_EXPLAINER_TITLES: dict[str, tuple[str, str]] = {
    "total_aoa": ("Contact Attack Angle", "°"),
    "without_shaft_aoa": ("AoA Without Shaft Rotation", "°"),
    "shaft_counterfactual_aoa_delta": ("Shaft Rotation AoA Delta", "°"),
    "shaft_shapley_aoa": ("Shaft-Rotation Shapley AoA", "°"),
    "le_downrange_rate": ("LE Downrange Rate", "m/s"),
    "le_vertical_rate": ("LE Vertical Rate", "m/s"),
    "le_lateral_rate": ("LE Lateral Rate", "m/s"),
    "le_3d_angular_rate": ("Leading Edge 3D Rate", "°/s"),
    "dynamic_loft": ("Delivered Dynamic Loft", "°"),
    "dynamic_lie": ("Delivered Dynamic Lie", "°"),
    "dynamic_face_angle": ("Dynamic Face Angle", "°"),
    "delivered_bounce": ("Delivered Bounce", "°"),
    "low_point": ("Delivery Low Point", "m"),
    "waterfall_axis": ("Shaft-Axis Translation", "m/s"),
    "waterfall_shaft": ("Shaft Rotation Velocity", "m/s"),
    "waterfall_other": ("Other Rotation Velocity", "m/s"),
}

_EXPLAINER_DOCS: dict[str, tuple[str, str]] = {
    "total_aoa": ("atan2(v_c · up, |v_c,h|)", "Contact trajectory in inertial frame."),
    "without_shaft_aoa": ("AoA(v_c − v_s)", "Angle without shaft rotation."),
    "shaft_counterfactual_aoa_delta": ("AoA(v_c) − AoA(v_c − v_s)", "Nonlinear delta."),
    "shaft_shapley_aoa": ("mean marginal AoA across orders", "Two-factor Shapley."),
    "le_downrange_rate": (
        "(v_ref + ω × r_le) · target",
        "Target-line speed of LE center.",
    ),
    "le_vertical_rate": ("(v_ref + ω × r_le) · up", "Vertical linear velocity of LE."),
    "le_lateral_rate": (
        "(v_ref + ω × r_le) · right",
        "Lateral linear speed of LE center.",
    ),
    "le_3d_angular_rate": ("|ω × e_leading|", "3D angular rate of LE tangent."),
    "dynamic_loft": ("atan2(n_f · up, |n_f,h|)", "Delivered face-normal elevation."),
    "dynamic_lie": ("atan2(|s · up|, |s,h|)", "Delivered shaft-axis elevation."),
    "dynamic_face_angle": (
        "atan2(n_f · right, n_f · target)",
        "Heading of face normal.",
    ),
    "delivered_bounce": (
        "atan2(w_sole · up, |w_sole,h|)",
        "Elevation of central sole.",
    ),
    "low_point": ("argmin_{cand} (cand · up)", "Spatial lowest point of wedge."),
    "waterfall_axis": (
        "v_axis = v_ref + ω × (s - r_ref)",
        "Linear translation of shaft.",
    ),
    "waterfall_shaft": ("v_shaft = (ω · ŝ) ŝ × (c - s)", "Contact speed from shaft."),
    "waterfall_other": (
        "v_other = (ω - (ω · ŝ) ŝ) × (c - s)",
        "Contact speed non-shaft.",
    ),
}


@dataclass(frozen=True)
class WedgeDeliveryMetrics:
    """Complete wedge delivery metrics at ball impact or closest approach."""

    frame_id: str
    total_aoa_deg: float | None
    without_shaft_aoa_deg: float | None
    shaft_counterfactual_aoa_delta_deg: float | None
    shaft_shapley_aoa_deg: float | None
    le_downrange_rate_mps: float
    le_vertical_rate_mps: float
    le_lateral_rate_mps: float
    le_total_speed_mps: float
    le_3d_angular_rate_dps: float
    dynamic_loft_deg: float
    dynamic_lie_deg: float
    dynamic_face_angle_deg: float | None
    delivered_bounce_deg: float
    path_projected_effective_bounce_deg: float | None
    bounce_utilization_margin_deg: float | None
    low_point_world_m: tuple[float, float, float] | None
    waterfall: LinearVelocityWaterfall

    def explainers(self) -> tuple[MetricExplainer, ...]:
        """Return comprehensive explainer entries."""
        lp = self.low_point_world_m
        lp_str = f"({lp[0]:.3f}, {lp[1]:.3f}, {lp[2]:.3f}) m" if lp else None
        vals: dict[str, float | str | None] = {
            "total_aoa": self.total_aoa_deg,
            "without_shaft_aoa": self.without_shaft_aoa_deg,
            "shaft_counterfactual_aoa_delta": self.shaft_counterfactual_aoa_delta_deg,
            "shaft_shapley_aoa": self.shaft_shapley_aoa_deg,
            "le_downrange_rate": self.le_downrange_rate_mps,
            "le_vertical_rate": self.le_vertical_rate_mps,
            "le_lateral_rate": self.le_lateral_rate_mps,
            "le_3d_angular_rate": self.le_3d_angular_rate_dps,
            "dynamic_loft": self.dynamic_loft_deg,
            "dynamic_lie": self.dynamic_lie_deg,
            "dynamic_face_angle": self.dynamic_face_angle_deg,
            "delivered_bounce": self.delivered_bounce_deg,
            "low_point": lp_str,
            "waterfall_axis": self.waterfall.base_axis.total_speed_mps,
            "waterfall_shaft": self.waterfall.shaft_rotation.total_speed_mps,
            "waterfall_other": self.waterfall.other_rotation.total_speed_mps,
        }
        return tuple(
            MetricExplainer(
                key=k,
                label=_EXPLAINER_TITLES[k][0],
                value=vals[k],
                units=_EXPLAINER_TITLES[k][1],
                equation=_EXPLAINER_DOCS[k][0],
                frame=self.frame_id,
                assumptions=_EXPLAINER_DOCS[k][1],
                availability="available" if vals[k] is not None else "undefined",
            )
            for k in _EXPLAINER_TITLES
        )


def _world_sole(parameters: WedgeHeadParameters, pose: np.ndarray) -> np.ndarray:
    bounce = math.radians(parameters.bounce_deg)
    local_sole = np.array(
        [
            -parameters.sole_width_m * math.cos(bounce),
            parameters.sole_width_m * math.sin(bounce),
            0.0,
        ]
    )
    result: np.ndarray = pose[:3, :3] @ local_sole
    return result


def delivered_bounce_deg(
    parameters: WedgeHeadParameters,
    pose: np.ndarray,
    ground_normal: np.ndarray,
) -> float:
    """Return central-sole elevation above the ground plane."""
    world_sole = _world_sole(parameters, pose)
    vertical = float(np.dot(world_sole, ground_normal))
    horizontal = world_sole - vertical * ground_normal
    return math.degrees(math.atan2(vertical, float(np.linalg.norm(horizontal))))


def path_projected_metrics(
    parameters: WedgeHeadParameters,
    pose: np.ndarray,
    twist: np.ndarray,
    ground_normal: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Return path-projected bounce, reference AoA, and remaining angle margin."""
    velocity = twist[3:]
    vertical_velocity = float(np.dot(velocity, ground_normal))
    horizontal_velocity = velocity - vertical_velocity * ground_normal
    horizontal_speed = float(np.linalg.norm(horizontal_velocity))
    if horizontal_speed <= _MIN_HORIZONTAL_SPEED_MPS:
        return None, None, None
    path_direction = horizontal_velocity / horizontal_speed
    world_sole = _world_sole(parameters, pose)
    sole_vertical = float(np.dot(world_sole, ground_normal))
    sole_horizontal = world_sole - sole_vertical * ground_normal
    trailing_along_path = float(np.dot(sole_horizontal, -path_direction))
    effective_bounce = math.degrees(math.atan2(sole_vertical, trailing_along_path))
    reference_aoa = math.degrees(math.atan2(vertical_velocity, horizontal_speed))
    return effective_bounce, reference_aoa, effective_bounce + reference_aoa


def _aoa_deg(velocity: np.ndarray, ground_up: np.ndarray) -> float | None:
    vert = float(np.dot(velocity, ground_up))
    horiz = velocity - vert * ground_up
    horiz_speed = float(np.linalg.norm(horiz))
    if horiz_speed <= _MIN_HORIZONTAL_SPEED_MPS:
        return None
    return math.degrees(math.atan2(vert, horiz_speed))


def linear_velocity_waterfall(
    reference_velocity_mps: np.ndarray,
    angular_velocity_rad_s: np.ndarray,
    shaft_axis_point_m: np.ndarray,
    shaft_axis_unit: np.ndarray,
    contact_point_m: np.ndarray,
    ground_up_unit: np.ndarray | None = None,
    target_unit: np.ndarray | None = None,
    right_unit: np.ndarray | None = None,
) -> LinearVelocityWaterfall:
    """Decompose linear velocity: v_contact = v_axis + v_shaft + v_other."""
    up = _DEFAULT_GROUND_NORMAL if ground_up_unit is None else ground_up_unit
    target = _DEFAULT_TARGET_UNIT if target_unit is None else target_unit
    right = _DEFAULT_RIGHT_UNIT if right_unit is None else right_unit

    v_r = np.asarray(reference_velocity_mps, dtype=float)
    omega = np.asarray(angular_velocity_rad_s, dtype=float)
    s = np.asarray(shaft_axis_point_m, dtype=float)
    s_hat = np.asarray(shaft_axis_unit, dtype=float)
    s_hat = s_hat / float(np.linalg.norm(s_hat))
    c = np.asarray(contact_point_m, dtype=float)

    v_axis = v_r + np.cross(omega, s)
    omega_shaft_scalar = float(np.dot(omega, s_hat))
    omega_shaft = omega_shaft_scalar * s_hat
    omega_other = omega - omega_shaft

    lever = c - s
    v_shaft = np.cross(omega_shaft, lever)
    v_other = np.cross(omega_other, lever)
    v_contact = v_axis + v_shaft + v_other

    names = (
        "shaft_axis_translation",
        "shaft_rotation",
        "other_rotation",
        "total_contact",
    )
    labels = (
        "Shaft-Axis Translation",
        "Rotation About Shaft",
        "Other Rotation",
        "Total Contact Velocity",
    )
    vecs = (v_axis, v_shaft, v_other, v_contact)
    means = (
        "Rigid translation.",
        "Shaft rotation.",
        "Non-shaft rotation.",
        "Total contact.",
    )
    st = tuple(
        LinearWaterfallStep(
            name=n,
            label=lbl,
            downrange_mps=float(np.dot(v, target)),
            vertical_mps=float(np.dot(v, up)),
            lateral_mps=float(np.dot(v, right)),
            total_speed_mps=float(np.linalg.norm(v)),
            meaning=m,
        )
        for n, lbl, v, m in zip(names, labels, vecs, means, strict=True)
    )
    base_axis, shaft_rotation, other_rotation, total_contact = st

    aoa_total = _aoa_deg(v_contact, up)
    aoa_no_shaft = _aoa_deg(v_contact - v_shaft, up)
    aoa_no_other = _aoa_deg(v_contact - v_other, up)
    aoa_axis = _aoa_deg(v_axis, up)

    shaft_cf_delta = (
        aoa_total - aoa_no_shaft
        if (aoa_total is not None and aoa_no_shaft is not None)
        else None
    )
    other_cf_delta = (
        aoa_total - aoa_no_other
        if (aoa_total is not None and aoa_no_other is not None)
        else None
    )

    shaft_shapley: float | None = None
    other_shapley: float | None = None
    if (
        aoa_total is not None
        and aoa_no_shaft is not None
        and aoa_no_other is not None
        and aoa_axis is not None
    ):
        shaft_shapley = 0.5 * ((aoa_no_other - aoa_axis) + (aoa_total - aoa_no_shaft))
        other_shapley = 0.5 * ((aoa_no_shaft - aoa_axis) + (aoa_total - aoa_no_other))

    return LinearVelocityWaterfall(
        base_axis=base_axis,
        shaft_rotation=shaft_rotation,
        other_rotation=other_rotation,
        total_contact=total_contact,
        total_aoa_deg=aoa_total,
        without_shaft_aoa_deg=aoa_no_shaft,
        shaft_counterfactual_aoa_delta_deg=shaft_cf_delta,
        without_other_aoa_deg=aoa_no_other,
        other_counterfactual_aoa_delta_deg=other_cf_delta,
        shaft_shapley_aoa_deg=shaft_shapley,
        other_shapley_aoa_deg=other_shapley,
    )


def compute_wedge_delivery_metrics(
    parameters: WedgeHeadParameters,
    pose: np.ndarray,
    twist: np.ndarray,
    ground_normal: np.ndarray | None = None,
    target_unit: np.ndarray | None = None,
    right_unit: np.ndarray | None = None,
    shaft_axis_point_m: np.ndarray | None = None,
    shaft_axis_unit: np.ndarray | None = None,
    contact_point_m: np.ndarray | None = None,
    low_point_world_m: tuple[float, float, float] | None = None,
) -> WedgeDeliveryMetrics:
    """Compute all synchronized delivery metrics, LE rates, and linear waterfall."""
    up = _DEFAULT_GROUND_NORMAL if ground_normal is None else ground_normal
    target = _DEFAULT_TARGET_UNIT if target_unit is None else target_unit
    right = _DEFAULT_RIGHT_UNIT if right_unit is None else right_unit

    rotation = pose[:3, :3]
    reference_pos = pose[:3, 3]
    omega = twist[:3]
    v_ref = twist[3:]

    if shaft_axis_unit is None:
        lie_rad = math.radians(parameters.lie_deg)
        local_shaft = np.array([0.0, math.sin(lie_rad), -math.cos(lie_rad)])
        shaft_axis = rotation @ local_shaft
        shaft_axis /= float(np.linalg.norm(shaft_axis))
    else:
        shaft_axis = shaft_axis_unit / float(np.linalg.norm(shaft_axis_unit))

    shaft_pt = reference_pos if shaft_axis_point_m is None else shaft_axis_point_m

    if contact_point_m is None:
        contact_pt = reference_pos + rotation @ np.asarray(
            wedge_face_contact_point_m(parameters, 0.0, 0.0)
        )
    else:
        contact_pt = contact_point_m

    waterfall = linear_velocity_waterfall(
        reference_velocity_mps=v_ref,
        angular_velocity_rad_s=omega,
        shaft_axis_point_m=shaft_pt,
        shaft_axis_unit=shaft_axis,
        contact_point_m=contact_pt,
        ground_up_unit=up,
        target_unit=target,
        right_unit=right,
    )

    profile = wedge_body_profile_m(parameters)
    le_local_pt = np.array([profile[0][0], profile[0][1], 0.0])
    le_world_pt = reference_pos + rotation @ le_local_pt
    le_velocity = v_ref + np.cross(omega, le_world_pt - reference_pos)

    le_downrange = float(np.dot(le_velocity, target))
    le_vertical = float(np.dot(le_velocity, up))
    le_lateral = float(np.dot(le_velocity, right))
    le_speed = float(np.linalg.norm(le_velocity))

    le_local_tangent = np.array([0.0, 0.0, 1.0])
    le_world_tangent = rotation @ le_local_tangent
    le_tangent_rate = np.cross(omega, le_world_tangent)
    le_3d_angular_rate = math.degrees(float(np.linalg.norm(le_tangent_rate)))

    loft_rad = math.radians(parameters.loft_deg)
    local_face_normal = np.array([math.cos(loft_rad), math.sin(loft_rad), 0.0])
    world_face_normal = rotation @ local_face_normal
    world_face_normal /= float(np.linalg.norm(world_face_normal))

    fn_vert = float(np.dot(world_face_normal, up))
    fn_horiz = world_face_normal - fn_vert * up
    fn_horiz_norm = float(np.linalg.norm(fn_horiz))
    dynamic_loft = math.degrees(math.atan2(fn_vert, fn_horiz_norm))

    dynamic_face = (
        math.degrees(
            math.atan2(float(np.dot(fn_horiz, right)), float(np.dot(fn_horiz, target)))
        )
        if fn_horiz_norm > _MIN_HORIZONTAL_SPEED_MPS
        else None
    )

    shaft_up = float(np.dot(shaft_axis, up))
    shaft_horiz = shaft_axis - shaft_up * up
    shaft_horiz_norm = float(np.linalg.norm(shaft_horiz))
    dynamic_lie = math.degrees(math.atan2(abs(shaft_up), shaft_horiz_norm))

    del_bounce = delivered_bounce_deg(parameters, pose, up)
    eff_bounce, _ref_aoa, margin = path_projected_metrics(parameters, pose, twist, up)

    if low_point_world_m is None:
        cand_pts = [
            reference_pos + rotation @ np.asarray(c.local_point_m)
            for c in wedge_contact_candidates(parameters)
        ]
        lowest = min(cand_pts, key=lambda p: float(np.dot(p, up)))
        low_pt_tuple: tuple[float, float, float] | None = (
            float(lowest[0]),
            float(lowest[1]),
            float(lowest[2]),
        )
    else:
        low_pt_tuple = low_point_world_m

    return WedgeDeliveryMetrics(
        frame_id=_APP_FRAME_ID,
        total_aoa_deg=waterfall.total_aoa_deg,
        without_shaft_aoa_deg=waterfall.without_shaft_aoa_deg,
        shaft_counterfactual_aoa_delta_deg=waterfall.shaft_counterfactual_aoa_delta_deg,
        shaft_shapley_aoa_deg=waterfall.shaft_shapley_aoa_deg,
        le_downrange_rate_mps=le_downrange,
        le_vertical_rate_mps=le_vertical,
        le_lateral_rate_mps=le_lateral,
        le_total_speed_mps=le_speed,
        le_3d_angular_rate_dps=le_3d_angular_rate,
        dynamic_loft_deg=dynamic_loft,
        dynamic_lie_deg=dynamic_lie,
        dynamic_face_angle_deg=dynamic_face,
        delivered_bounce_deg=del_bounce,
        path_projected_effective_bounce_deg=eff_bounce,
        bounce_utilization_margin_deg=margin,
        low_point_world_m=low_pt_tuple,
        waterfall=waterfall,
    )


__all__ = [
    "LinearVelocityWaterfall",
    "LinearWaterfallStep",
    "MetricExplainer",
    "WedgeDeliveryMetrics",
    "compute_wedge_delivery_metrics",
    "delivered_bounce_deg",
    "linear_velocity_waterfall",
    "path_projected_metrics",
]

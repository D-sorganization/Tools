"""Delivery extraction at a selected clubhead sample."""

from __future__ import annotations

import math

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.club import ClubSpec, face_normal_at_offset
from rate_of_closure.model import ImpactScenario
from shared.python.swing_sim.impact import (
    DeliveryDerived,
    DeliveryParameters,
    derive_delivery,
)
from shared.python.swing_sim.swing_source import SwingSource

_MAX_DELIVERY_ANGLE_DEG = 89.0


def delivery_at(
    source: SwingSource,
    tau: float,
    scenario: ImpactScenario,
    club: ClubSpec,
) -> DeliveryDerived:
    """Return delivery vectors and D-plane diagnostics at ``tau``."""
    sample = source.sample(tau)
    face_normal = None
    if bool(getattr(source, "uses_declared_head_pose", False)):
        local_normal = np.asarray(face_normal_at_offset(club, 0.0, 0.0))
        face_normal = sample.pose[:3, :3] @ local_normal
    params = _delivery_parameters(sample.twist[3:], scenario, club, face_normal)
    return derive_delivery(params, clubhead_angular_velocity=sample.twist[:3])


def _delivery_parameters(
    velocity: np.ndarray,
    scenario: ImpactScenario,
    club: ClubSpec,
    face_normal: np.ndarray | None = None,
) -> DeliveryParameters:
    """Build impact-package delivery inputs from an app-frame velocity."""
    speed = float(np.linalg.norm(velocity))
    require(speed > 1e-6, "clubhead speed at impact must be > 0", speed)
    path_deg = math.degrees(math.atan2(float(velocity[2]), float(velocity[0])))
    aoa_deg = math.degrees(
        math.atan2(
            float(velocity[1]), math.hypot(float(velocity[0]), float(velocity[2]))
        )
    )
    face_angle_deg, dynamic_loft_deg = _delivered_face_angles(face_normal, club)
    return DeliveryParameters(
        clubhead_speed_mps=speed,
        club_path_deg=_clamped_angle(path_deg),
        face_angle_deg=face_angle_deg,
        attack_angle_deg=_clamped_angle(aoa_deg),
        dynamic_loft_deg=dynamic_loft_deg,
        lie_deg=0.0,
        impact_offset_toe_mm=scenario.impact_offset_toe_mm,
        impact_offset_high_mm=scenario.impact_offset_high_mm,
    )


def _delivered_face_angles(
    face_normal: np.ndarray | None, club: ClubSpec
) -> tuple[float, float]:
    """Return face angle and dynamic loft from an optional world normal."""
    if face_normal is None:
        return (0.0, club.loft_deg)
    normal = np.asarray(face_normal, dtype=float)
    require(normal.shape == (3,), "face_normal must have shape (3,)")
    magnitude = float(np.linalg.norm(normal))
    require(magnitude > 1e-12, "face_normal must be nonzero")
    unit = normal / magnitude
    face_angle_deg = math.degrees(math.atan2(float(unit[2]), float(unit[0])))
    dynamic_loft_deg = math.degrees(
        math.atan2(float(unit[1]), math.hypot(float(unit[0]), float(unit[2])))
    )
    return (face_angle_deg, dynamic_loft_deg)


def _clamped_angle(value_deg: float) -> float:
    """Clamp an angle into the impact package's delivery domain."""
    return max(-_MAX_DELIVERY_ANGLE_DEG, min(_MAX_DELIVERY_ANGLE_DEG, value_deg))

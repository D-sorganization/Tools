"""Delivery extraction at a selected clubhead sample."""

from __future__ import annotations

import math

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.club import ClubSpec
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
    params = _delivery_parameters(sample.twist[3:], scenario, club)
    return derive_delivery(params, clubhead_angular_velocity=sample.twist[:3])


def _delivery_parameters(
    velocity: np.ndarray, scenario: ImpactScenario, club: ClubSpec
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
    return DeliveryParameters(
        clubhead_speed_mps=speed,
        club_path_deg=_clamped_angle(path_deg),
        face_angle_deg=0.0,
        attack_angle_deg=_clamped_angle(aoa_deg),
        dynamic_loft_deg=_clamped_angle(club.loft_deg),
        lie_deg=0.0,
        impact_offset_toe_mm=scenario.impact_offset_toe_mm,
        impact_offset_high_mm=scenario.impact_offset_high_mm,
    )


def _clamped_angle(value_deg: float) -> float:
    """Clamp an angle into the impact package's delivery domain."""
    return max(-_MAX_DELIVERY_ANGLE_DEG, min(_MAX_DELIVERY_ANGLE_DEG, value_deg))

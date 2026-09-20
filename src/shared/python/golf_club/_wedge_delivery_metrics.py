"""Private ground-relative wedge metrics evaluated at ball contact."""

from __future__ import annotations

import math

import numpy as np

from .wedge_parameters import WedgeHeadParameters

_MIN_HORIZONTAL_SPEED_MPS = 1e-12


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


__all__: list[str] = []

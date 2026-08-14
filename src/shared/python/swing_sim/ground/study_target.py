"""Canonical target helpers for deterministic ground-study records."""

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.solver.spatial_targets import SpatialTarget
from shared.python.swing_sim.solver.target_serialization import (
    spatial_target_from_json,
    spatial_target_to_json_dict,
)


def canonical_ground_target(target: SpatialTarget) -> SpatialTarget:
    """Return the target represented by the shared canonical numeric wire."""
    return spatial_target_from_json(
        canonical_numeric_json(spatial_target_to_json_dict(target))
    )


__all__ = ["canonical_ground_target"]

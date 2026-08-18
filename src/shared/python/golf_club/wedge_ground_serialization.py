"""Versioned JSON-ready payloads for wedge ground-clearance visualization."""

from __future__ import annotations

from typing import Any

from .wedge_ground_contact import (
    WedgeGroundClearanceAnalysis,
    WedgeGroundContactEvent,
)

WEDGE_GROUND_CLEARANCE_FORMAT = "upstreamdrift.wedge-ground-clearance/v1"


def _event_to_dict(event: WedgeGroundContactEvent | None) -> dict[str, Any] | None:
    if event is None:
        return None
    return {
        "feature": event.feature.value,
        "normal_velocity_mps": event.normal_velocity_mps,
        "pose_head_to_ground": event.pose_head_to_ground,
        "tangential_velocity_mps": event.tangential_velocity_mps,
        "time_s": event.time_s,
        "world_point_m": event.world_point_m,
    }


def wedge_ground_clearance_to_json_dict(
    analysis: WedgeGroundClearanceAnalysis,
) -> dict[str, Any]:
    """Return the deterministic canonical payload consumed by UI adapters."""
    if not isinstance(analysis, WedgeGroundClearanceAnalysis):
        raise TypeError("analysis must be WedgeGroundClearanceAnalysis")
    return {
        "format": WEDGE_GROUND_CLEARANCE_FORMAT,
        "frame_id": analysis.frame_id,
        "units": {
            "angle": "deg",
            "angular_velocity": "rad/s",
            "length": "m",
            "time": "s",
            "velocity": "m/s",
        },
        "sequence": analysis.sequence.value,
        "ball_contact_time_s": analysis.ball_contact_time_s,
        "first_ground_contact": _event_to_dict(analysis.first_ground_contact),
        "metrics": {
            "bounce_utilization_margin_deg": analysis.bounce_utilization_margin_deg,
            "delivered_bounce_deg_at_ball": analysis.delivered_bounce_deg_at_ball,
            "ground_after_ball_time_margin_s": (
                analysis.ground_after_ball_time_margin_s
            ),
            "leading_edge_clearance_at_ball_m": (
                analysis.leading_edge_clearance_at_ball_m
            ),
            "minimum_pre_ball_clearance_m": analysis.minimum_pre_ball_clearance_m,
            "path_projected_effective_bounce_deg_at_ball": (
                analysis.path_projected_effective_bounce_deg_at_ball
            ),
            "reference_aoa_deg_at_ball": analysis.reference_aoa_deg_at_ball,
            "sole_entry_margin_m": analysis.sole_entry_margin_m,
        },
        "low_point": {
            "feature": analysis.low_point_feature.value,
            "time_s": analysis.low_point_time_s,
            "world_point_m": analysis.low_point_world_m,
        },
        "envelope": [
            {
                "feature": sample.feature.value,
                "minimum_clearance_m": sample.minimum_clearance_m,
                "time_s": sample.time_s,
                "world_point_m": sample.world_point_m,
            }
            for sample in analysis.envelope
        ],
        "limitations": analysis.limitations,
    }


__all__ = [
    "WEDGE_GROUND_CLEARANCE_FORMAT",
    "wedge_ground_clearance_to_json_dict",
]

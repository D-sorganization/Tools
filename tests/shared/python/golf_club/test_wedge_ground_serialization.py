"""Versioned wedge ground-clearance visualization payload contracts."""

from __future__ import annotations

import json

import numpy as np

from shared.python.golf_club import (
    WEDGE_GROUND_CLEARANCE_FORMAT,
    GroundPlane,
    WedgePreset,
    analyze_wedge_ground_clearance,
    wedge_ground_clearance_to_json_dict,
    wedge_preset,
)


def _analysis(*, ball_contact_time_s: float | None):
    poses = np.stack([np.eye(4), np.eye(4)])
    poses[0, 1, 3] = 0.01
    poses[1, 1, 3] = -0.01
    twists = np.zeros((2, 6))
    twists[:, 3] = 10.0
    twists[:, 4] = -0.02
    return analyze_wedge_ground_clearance(
        wedge_preset(WedgePreset.MID_BOUNCE),
        (0.0, 1.0),
        poses,
        twists,
        GroundPlane(),
        ball_contact_time_s=ball_contact_time_s,
    )


def test_payload_is_versioned_complete_and_json_serializable() -> None:
    payload = wedge_ground_clearance_to_json_dict(_analysis(ball_contact_time_s=0.25))

    assert payload["format"] == WEDGE_GROUND_CLEARANCE_FORMAT
    assert payload["frame_id"] == "ground_frame:x_target,y_up,z_right"
    assert payload["units"] == {
        "angle": "deg",
        "angular_velocity": "rad/s",
        "length": "m",
        "time": "s",
        "velocity": "m/s",
    }
    assert payload["sequence"] == "ball_first"
    assert payload["first_ground_contact"]["feature"] == "leading_edge_center"
    assert len(payload["first_ground_contact"]["pose_head_to_ground"]) == 4
    assert len(payload["envelope"]) == 9
    assert payload["metrics"]["sole_entry_margin_m"] is not None
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_payload_preserves_missing_ball_contact_for_a_miss() -> None:
    payload = wedge_ground_clearance_to_json_dict(_analysis(ball_contact_time_s=None))

    assert payload["sequence"] == "ground_only_miss"
    assert payload["ball_contact_time_s"] is None
    assert payload["metrics"]["leading_edge_clearance_at_ball_m"] is None

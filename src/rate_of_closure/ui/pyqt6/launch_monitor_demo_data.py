"""Deterministic non-vendor data used by the analytics workbench."""

from __future__ import annotations

import numpy as np
import pandas as pd


def demo_frame() -> pd.DataFrame:
    """Return deterministic non-vendor demonstration shots."""

    index = np.arange(120)
    club_speed = 38.0 + index * 0.11
    attack_angle = -4.0 + (index % 17) * 0.4
    club_path = -3.0 + (index % 13) * 0.5
    ball_speed = club_speed * 1.46 + attack_angle * 0.08 + np.sin(index) * 0.25
    return pd.DataFrame(
        {
            "shot_id": [f"demo-{item + 1}" for item in index],
            "session_id": np.where(index < 60, "demo-a", "demo-b"),
            "monitor_vendor": np.where(index % 2, "FlightScope", "TrackMan"),
            "club_speed": club_speed,
            "attack_angle": attack_angle,
            "club_path": club_path,
            "ball_speed": ball_speed,
            "carry_distance": ball_speed * 3.25 + attack_angle * 0.9,
            "lateral_distance": club_path * 2.2,
        }
    )


__all__ = ["demo_frame"]

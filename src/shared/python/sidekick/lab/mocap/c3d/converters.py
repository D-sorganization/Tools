"""Biomechanical dataframe conversions, scaling, and COP (TOOLS-M9 #4716)."""

from __future__ import annotations

import numpy as np


def unit_scale_factor(source_units: str, target_units: str | None) -> float:
    """Calculate scaling factor to convert between metric and imperial length units."""
    if target_units is None:
        return 1.0
    src = source_units.strip().lower()
    tgt = target_units.strip().lower()
    if src == tgt:
        return 1.0

    to_meters = {
        "m": 1.0,
        "mm": 0.001,
        "cm": 0.01,
        "in": 0.0254,
        "ft": 0.3048,
    }
    if src not in to_meters:
        raise ValueError(f"Unsupported source unit: {source_units}")
    if tgt not in to_meters:
        raise ValueError(f"Unsupported target unit: {target_units}")
    return to_meters[src] / to_meters[tgt]


def compute_center_of_pressure(
    fz: np.ndarray,
    mx: np.ndarray,
    my: np.ndarray,
    min_force_threshold_n: float = 10.0,
    ground_height_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Center of Pressure (COP) coordinates from ground force
    and horizontal moments.

    cop_x = -My / Fz
    cop_y =  Mx / Fz
    cop_z =  ground_height_m
    """
    valid_contact = np.abs(fz) > min_force_threshold_n
    safe_fz = np.where(valid_contact, fz, 1.0)
    cop_x = np.where(valid_contact, -my / safe_fz, np.nan)
    cop_y = np.where(valid_contact, mx / safe_fz, np.nan)
    cop_z = np.where(valid_contact, ground_height_m, np.nan)
    return cop_x, cop_y, cop_z

# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Planar ergonomic estimate of L5/S1 spinal loading.

Computes compression and anterior-posterior (AP) shear at the L5/S1
vertebral junction from the sagittal-plane 3-link model.  The L5/S1
joint is located at the base of the torso segment (hip joint in our model).

What this is NOT: these outputs are a **planar ergonomic estimate**
(rigid-body load on the torso segment, compared against the NIOSH limit).
They are **not a musculoskeletal joint reaction**: no muscle forces, passive
elements or out-of-plane (mediolateral) loads are modelled.  Do not compare
them with static-optimization joint-reaction results without that caveat.
``MODEL_METADATA`` carries the same label for return-value consumers.

Sign conventions versus UpstreamDrift ADR 0055 (UD ADR 0055,
``docs/adr/0055-segmental-spine-kinematics-and-loads.md`` in UpstreamDrift;
``SpinalLoadSeries``, force on the superior vertebra by the inferior
vertebra, ISB frame X anterior, Y cranial):

    * Compression: positive, equal to +F_y.  Same sign as UD ADR 0055.
    * AP shear (``spinal_shear``): positive when the superior segment loads
      the inferior vertebra anteriorly (forward trunk lean).  This is the
      *opposite* sign of UD ADR 0055 F_x, which is the reaction of the inferior
      vertebra (posterior for a forward lean).  Use ``to_adr0055_ap_shear``
      to convert; never negate by hand.

The UD ADR 0055 contract itself is not copied into Tools (UD#12774).

Design Principles:
    DBC -- preconditions checked on public functions.
    DRY -- shared mass lookup factored into ``_mass_above_l5``.
    LoD -- callers only use the public functions and ``MODEL_METADATA``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import BodyModel

# NIOSH recommended compression limit for occupational lifting (N)
NIOSH_COMPRESSION_LIMIT: float = 3400.0

# Labels the model for consumers (issue #5630).  Read-only by convention.
MODEL_METADATA: dict[str, object] = {
    "model": "planar_ergonomic_l5s1_estimate",
    "is_joint_reaction": False,
    "plane": "sagittal",
    "compression_sign": "positive",
    "ap_shear_sign": "anterior_load_on_inferior_vertebra",
    "adr0055_ap_shear_conversion": "to_adr0055_ap_shear",
    "niosh_compression_limit_n": NIOSH_COMPRESSION_LIMIT,
}


def to_adr0055_ap_shear(shear: float | NDArray) -> float | NDArray:
    """Convert ``spinal_shear`` output to the UD ADR 0055 F_x convention (N).

    ``spinal_shear`` is positive for an anterior load of the superior segment
    on the inferior vertebra.  UD ADR 0055 F_x is the force on the superior
    vertebra by the inferior vertebra (anterior positive), i.e. its negation.

    Preconditions:
        ``shear`` is finite (scalar or array).
    Postconditions:
        Result has the same shape as ``shear`` and equals ``-shear``.
    """
    arr = np.asarray(shear, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("shear must be finite")
    if arr.ndim == 0:
        return -float(arr)
    return -arr


def _mass_above_l5(body: BodyModel, exercise_type: str) -> float:
    """Return the mass of body segments above L5/S1.

    For squat-type exercises the torso segment already includes arms.
    For deadlift-type exercises the torso segment is trunk+head only.
    """
    if exercise_type in ("squat", "full_squat"):
        return float(body.m_squat[2])
    return float(body.m_deadlift[2])


def _ensure_2d(arr: NDArray) -> NDArray:
    """Ensure the array is 2-D (N, 3).  If 1-D, reshape to (1, 3)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {arr.shape}")
        return arr.reshape(1, 3)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")
    if arr.shape[1] != 3:
        raise ValueError(f"Expected 3 columns (ank, kn, hip), got {arr.shape[1]}")
    return arr


def spinal_compression(
    q: NDArray,
    qd: NDArray,
    qdd: NDArray,
    body: BodyModel,
    bar_mass: float,
    exercise_type: str,
) -> float | NDArray:
    """Compute axial compression force at L5/S1 (N), planar ergonomic estimate.

    Compression is positive (same sign as UD ADR 0055 +F_y).  Not a
    musculoskeletal joint reaction; see the module docstring.

    At the L5/S1 joint the compressive force along the spine axis is:

        F_comp = (m_above + bar_mass) * g * cos(torso_angle)
                 + m_torso * qdd_torso * d_com_torso

    Parameters:
        q:  joint angles, shape (3,) or (N, 3)
        qd: joint velocities, shape (3,) or (N, 3)
        qdd: joint accelerations, shape (3,) or (N, 3)
        body: BodyModel instance
        bar_mass: external load in kg
        exercise_type: "squat", "full_squat", "deadlift", etc.

    Returns:
        Compression force in Newtons.  Scalar if input was (3,),
        array of shape (N,) if input was (N, 3).
    """
    scalar_input = np.asarray(q).ndim == 1
    q2d = _ensure_2d(q)
    qdd2d = _ensure_2d(qdd)

    m_above = _mass_above_l5(body, exercise_type)
    torso_angle = q2d[:, 2]

    # Static gravitational component along spine axis
    gravity_comp = (m_above + bar_mass) * body.g * np.cos(torso_angle)

    # Inertial component from torso angular acceleration
    m_torso = (
        float(body.m_squat[2])
        if exercise_type in ("squat", "full_squat")
        else float(body.m_deadlift[2])
    )
    inertial_comp = m_torso * qdd2d[:, 2] * body.d[2]

    result = gravity_comp + inertial_comp

    if scalar_input:
        return float(result[0])
    return result


def spinal_shear(
    q: NDArray,
    qd: NDArray,
    qdd: NDArray,
    body: BodyModel,
    bar_mass: float,
    exercise_type: str,
) -> float | NDArray:
    """Compute anterior-posterior shear force at L5/S1 (N), planar estimate.

    Sign: positive for anterior load of the superior segment on the inferior
    vertebra (forward lean).  This is the opposite of the UD ADR 0055 F_x
    convention; convert with ``to_adr0055_ap_shear``.  Not a musculoskeletal
    joint reaction; see the module docstring.

    The shear force perpendicular to the spine axis includes both
    gravitational and inertial components:

        F_shear = (m_above + bar_mass) * g * sin(torso_angle)
                  + m_torso * qdd_torso * d_com * cos(torso_angle)
                  + m_torso * qd_torso^2 * d_com * sin(torso_angle)

    The first term is the static gravitational shear.  The second is
    the tangential (angular-acceleration) inertial shear, and the third
    is the centripetal inertial shear.

    Parameters:
        q:  joint angles, shape (3,) or (N, 3)
        qd: joint velocities, shape (3,) or (N, 3)
        qdd: joint accelerations, shape (3,) or (N, 3)
        body: BodyModel instance
        bar_mass: external load in kg
        exercise_type: "squat", "full_squat", "deadlift", etc.

    Returns:
        Shear force in Newtons.  Scalar if input was (3,),
        array of shape (N,) if input was (N, 3).
    """
    scalar_input = np.asarray(q).ndim == 1
    q2d = _ensure_2d(q)
    qd2d = _ensure_2d(qd)
    qdd2d = _ensure_2d(qdd)

    m_above = _mass_above_l5(body, exercise_type)
    torso_angle = q2d[:, 2]

    # Static gravitational shear perpendicular to spine axis
    gravity_shear = (m_above + bar_mass) * body.g * np.sin(torso_angle)

    # Inertial shear from torso angular acceleration and centripetal force
    m_torso = (
        float(body.m_squat[2])
        if exercise_type in ("squat", "full_squat")
        else float(body.m_deadlift[2])
    )
    d_com = body.d[2]

    # Tangential inertial shear: m * qdd * d_com * cos(angle)
    tangential_shear = m_torso * qdd2d[:, 2] * d_com * np.cos(torso_angle)
    # Centripetal inertial shear: m * qd^2 * d_com * sin(angle)
    centripetal_shear = m_torso * qd2d[:, 2] ** 2 * d_com * np.sin(torso_angle)

    result = gravity_shear + tangential_shear + centripetal_shear

    if scalar_input:
        return float(result[0])
    return result

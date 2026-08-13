"""Impact energy-balance validation.

Ported from UpstreamDrift ``src/shared/python/physics/impact_model/utils.py``
(epic #4103 / issue #4106), rewritten self-contained against the vendored
:mod:`.constants`.

The empirical ``compute_gear_effect_spin`` (three tuned constants with a
hard-coded world-up axis) present in the UpstreamDrift source is
intentionally NOT ported; it is replaced by the physics-derived treatment
in :mod:`.gear_effect`.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import GOLF_BALL_MASS_KG, GOLF_BALL_MOMENT_OF_INERTIA_KG_M2
from .types import ImpactParameters, PostImpactState, PreImpactState


def validate_energy_balance(
    pre_state: PreImpactState,
    post_state: PostImpactState,
    params: ImpactParameters,
) -> dict[str, float]:
    """Validate energy balance before and after impact.

    Total mechanical energy should be conserved up to COR losses:
    the expected loss in the relative (COM) frame is
    ``dKE = 1/2 * mu * v_rel_n^2 * (1 - e^2)``.

    Args:
        pre_state: Pre-impact state
        post_state: Post-impact state
        params: Impact parameters

    Returns:
        Dictionary with energy analysis results.
    """
    if pre_state is None:
        raise ValueError("pre_state must be provided")
    m_ball = GOLF_BALL_MASS_KG
    m_club = pre_state.clubhead_mass
    i_ball = GOLF_BALL_MOMENT_OF_INERTIA_KG_M2

    n_raw = np.asarray(pre_state.clubhead_orientation, dtype=float).reshape(-1)
    n_norm = math.sqrt(float(np.dot(n_raw, n_raw))) if n_raw.size > 0 else 0.0
    n_unit = n_raw / n_norm if n_norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    v_rel = np.asarray(pre_state.clubhead_velocity, dtype=float) - np.asarray(
        pre_state.ball_velocity, dtype=float
    )
    mu = (m_ball * m_club) / (m_ball + m_club)
    expected_loss_j = (
        0.5 * mu * float(np.dot(v_rel, n_unit)) ** 2 * (1.0 - params.cor**2)
    )

    ke_ball_pre = (
        0.5 * m_ball * float(np.dot(pre_state.ball_velocity, pre_state.ball_velocity))
    )
    ke_ball_rot_pre = (
        0.5
        * i_ball
        * float(
            np.dot(pre_state.ball_angular_velocity, pre_state.ball_angular_velocity)
        )
    )
    ke_club_pre = (
        0.5
        * m_club
        * float(np.dot(pre_state.clubhead_velocity, pre_state.clubhead_velocity))
    )
    total_ke_pre = ke_ball_pre + ke_ball_rot_pre + ke_club_pre

    ke_ball_post = (
        0.5 * m_ball * float(np.dot(post_state.ball_velocity, post_state.ball_velocity))
    )
    ke_ball_rot_post = (
        0.5
        * i_ball
        * float(
            np.dot(post_state.ball_angular_velocity, post_state.ball_angular_velocity)
        )
    )
    ke_club_post = (
        0.5
        * m_club
        * float(np.dot(post_state.clubhead_velocity, post_state.clubhead_velocity))
    )
    total_ke_post = ke_ball_post + ke_ball_rot_post + ke_club_post

    energy_lost = total_ke_pre - total_ke_post
    expected_loss_factor = 1 - params.cor**2  # COR relates velocities, not energy

    return {
        "total_ke_pre": float(total_ke_pre),
        "total_ke_post": float(total_ke_post),
        "energy_lost": float(energy_lost),
        "energy_loss_ratio": (
            float(energy_lost / total_ke_pre) if total_ke_pre > 0 else 0.0
        ),
        "expected_loss_factor": expected_loss_factor,
        "expected_loss_j": float(expected_loss_j),
        "ball_ke_post": float(ke_ball_post),
        "ball_launch_speed": float(
            math.sqrt(float(np.dot(post_state.ball_velocity, post_state.ball_velocity)))
        ),
    }

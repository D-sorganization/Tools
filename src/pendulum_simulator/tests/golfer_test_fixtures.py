"""Shared nonsingular golfer state for pendulum contract tests."""

from __future__ import annotations

import numpy as np

from double_pendulum_golf.physics_golfer import GolferParams


def default_golfer_params() -> GolferParams:
    """Minimal valid golfer parameters used across pendulum tests."""
    return GolferParams(
        m_hub=10.0,
        L_hub=0.3,
        m_r_upper=2.0,
        L_r_upper=0.3,
        m_r_fore=1.5,
        L_r_fore=0.25,
        m_l_upper=2.0,
        L_l_upper=0.3,
        m_l_fore=1.5,
        L_l_fore=0.25,
        m_club=0.3,
        L_club=1.1,
        d_rs=0.1,
        d_ls=0.1,
        grip_right=0.1,
        grip_left=0.3,
        g=9.81,
    )


def nonsingular_golfer_state() -> np.ndarray:
    """16-DOF state vector known to exercise the golfer dynamics oracle."""
    state = np.zeros(16)
    state[0] = 0.1
    return state

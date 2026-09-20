"""Tests for ball impact dipole acoustic radiation (IA-T5, #5074)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.vibroacoustics._boundary_radiation_records import (
    AcousticMedium,
)
from shared.python.swing_sim.vibroacoustics.ball_radiation import BallAcousticRadiation
from shared.python.swing_sim.vibroacoustics.observer import ObserverLocation


@pytest.mark.unit
def test_ball_dipole_radiation_directivity_and_delay() -> None:
    medium = AcousticMedium()
    ball_rad = BallAcousticRadiation(medium=medium)

    # Half-sine impact force pulse: duration 0.5 ms, peak 10,000 N along x-axis
    sample_rate_hz = 200_000.0  # 200 kHz
    duration = 0.0005
    t = np.arange(0.0, 0.005, 1.0 / sample_rate_hz)
    f_mag = np.where(t < duration, 10000.0 * np.sin(np.pi * t / duration), 0.0)

    # Force vector along x: (F_x, 0, 0)
    force_history = np.zeros((3, len(t)))
    force_history[0, :] = f_mag

    # Impact center of pressure at origin (0, 0, 0)
    origin = np.array([0.0, 0.0, 0.0])

    # On-axis observer (+x, 1m) vs off-axis perpendicular observer (+y, 1m)
    obs_on_axis = ObserverLocation(
        name="mic_x", coordinates_m=np.array([1.0, 0.0, 0.0])
    )
    obs_perpendicular = ObserverLocation(
        name="mic_y", coordinates_m=np.array([0.0, 1.0, 0.0])
    )
    obs_negative_x = ObserverLocation(
        name="mic_neg_x", coordinates_m=np.array([-1.0, 0.0, 0.0])
    )

    p_x = ball_rad.solve_ball_pressure(
        observer=obs_on_axis,
        force_history_n=force_history,
        time=t,
        source_origin_m=origin,
    )
    p_y = ball_rad.solve_ball_pressure(
        observer=obs_perpendicular,
        force_history_n=force_history,
        time=t,
        source_origin_m=origin,
    )
    p_neg_x = ball_rad.solve_ball_pressure(
        observer=obs_negative_x,
        force_history_n=force_history,
        time=t,
        source_origin_m=origin,
    )

    # Dipole directivity property:
    # Perpendicular observer has zero pressure because R . F = 0
    assert np.allclose(p_y, 0.0, atol=1e-10)

    # Negative x observer has opposite polarity (dipole symmetry: cos(pi) = -1)
    assert np.allclose(p_neg_x, -p_x, atol=1e-6)

    # Delay check: arrival time at 1m is 1 / 343.2 s ~ 0.002914 s
    idx_arr = np.where(np.abs(p_x) > 1e-2)[0][0]
    t_arr = t[idx_arr]
    assert t_arr == pytest.approx(1.0 / medium.sound_speed_mps, abs=1e-4)


@pytest.mark.unit
def test_ball_radiation_contracts() -> None:
    ball_rad = BallAcousticRadiation()
    t = np.linspace(0, 0.001, 100)
    forces_bad_dim = np.zeros((2, 100))
    obs = ObserverLocation(name="obs", coordinates_m=np.array([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="3D force"):
        ball_rad.solve_ball_pressure(
            observer=obs,
            force_history_n=forces_bad_dim,
            time=t,
        )

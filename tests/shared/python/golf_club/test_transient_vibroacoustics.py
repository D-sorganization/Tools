"""Comprehensive integration test for transient vibroacoustic radiation
and acoustic field solver (IA-T5, #5074).

Couples standard driver face/hosel modes from swing_sim.impact with the transient
Rayleigh integral boundary radiation solver and ball impact dipole radiation.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.impact._face_hosel_modes import (
    standard_driver_face_hosel_modes,
)
from shared.python.swing_sim.vibroacoustics._boundary_radiation_records import (
    AcousticMedium,
    RadiatingSurfaceMesh,
)
from shared.python.swing_sim.vibroacoustics.ball_radiation import BallAcousticRadiation
from shared.python.swing_sim.vibroacoustics.observer import (
    HeldOutObserverComparison,
    MicrophoneArray,
)
from shared.python.swing_sim.vibroacoustics.psychoacoustics import (
    AcousticReferenceAlgorithm,
    calculate_sharpness,
    calculate_specific_loudness,
    compute_peak_spl,
    compute_sound_exposure_level,
)
from shared.python.swing_sim.vibroacoustics.radiation import (
    ModalRadiationTransfer,
)


@pytest.mark.unit
def test_driver_impact_transient_vibroacoustics_and_held_out_microphones() -> None:
    # 1. Driver face mesh: 0.10 m (heel-toe) x 0.05 m (crown-sole)
    face_mesh = RadiatingSurfaceMesh.planar_rectangular_plate(
        length_x_m=0.10,
        width_y_m=0.05,
        nx=16,
        ny=8,
    )
    medium = AcousticMedium()
    face_mesh.validate_mesh_convergence(
        max_frequency_hz=6000.0, sound_speed_mps=medium.sound_speed_mps
    )

    # 2. Structural modal system from IA-T4
    modal_system = standard_driver_face_hosel_modes()
    assert modal_system.mode_count >= 3

    mode_shapes = tuple(
        lambda x, y, m=m: modal_system.modes[m].shape_at(x, y)
        for m in range(modal_system.mode_count)
    )
    mode_names = tuple(
        modal_system.modes[m].name for m in range(modal_system.mode_count)
    )

    modal_transfer = ModalRadiationTransfer(
        mesh=face_mesh,
        medium=medium,
        mode_shapes=mode_shapes,
        mode_names=mode_names,
    )

    # 3. Simulate impact transient: duration 0.5 ms contact pulse + ringdown
    sample_rate_hz = 100_000.0  # 100 kHz sampling
    t_end = 0.025  # 25 ms
    time = np.arange(0.0, t_end, 1.0 / sample_rate_hz)

    # Contact force pulse: 8000 N peak, 0.45 ms duration
    t_contact = 0.00045
    f_contact = np.where(
        time < t_contact, 8000.0 * np.sin(np.pi * time / t_contact), 0.0
    )
    force_history = np.vstack([f_contact, np.zeros_like(time), np.zeros_like(time)])

    # Modal coordinates response: modal ring-down at each natural frequency
    modal_accel = np.zeros((modal_system.mode_count, len(time)))
    for m_idx, mode in enumerate(modal_system.modes):
        omega_d = 2.0 * np.pi * mode.natural_frequency_hz
        decay = mode.damping_ratio * omega_d
        # Mode response excited by impact impulse
        modal_accel[m_idx, :] = np.where(
            time >= 0.0002,
            1200.0
            * np.sin(omega_d * (time - 0.0002))
            * np.exp(-decay * (time - 0.0002)),
            0.0,
        )

    # 4. Multi-microphone observer array
    mic_array = MicrophoneArray.standard_golfer_and_field_microphones()
    assert len(mic_array.observers) >= 3

    golfer_ear = mic_array.get_observer("golfer_ear")
    front_mic = mic_array.get_observer("front_1m")

    # 5. Radiated sound from face modes
    p_face_ear = modal_transfer.solve_modal_pressure(
        observer=golfer_ear,
        modal_accelerations=modal_accel,
        time=time,
    )
    p_face_front = modal_transfer.solve_modal_pressure(
        observer=front_mic,
        modal_accelerations=modal_accel,
        time=time,
    )

    # 6. Radiated sound from ball dipole
    ball_solver = BallAcousticRadiation(medium=medium)
    p_ball_ear = ball_solver.solve_ball_pressure(
        observer=golfer_ear,
        force_history_n=force_history,
        time=time,
    )
    p_ball_front = ball_solver.solve_ball_pressure(
        observer=front_mic,
        force_history_n=force_history,
        time=time,
    )

    # Total sound pressure
    p_total_ear = p_face_ear + p_ball_ear
    p_total_front = p_face_front + p_ball_front

    # 7. Acoustic metrics
    peak_spl_ear = compute_peak_spl(p_total_ear)
    peak_spl_front = compute_peak_spl(p_total_front)

    assert 80.0 < peak_spl_ear < 185.0
    assert 80.0 < peak_spl_front < 185.0

    exp_ear, sel_ear = compute_sound_exposure_level(
        p_total_ear, sample_rate_hz=sample_rate_hz
    )
    assert exp_ear > 0.0
    assert sel_ear > 50.0

    # 8. Psychoacoustics: DIN 45692 sharpness & ISO 532-1 loudness
    sharpness_ear = calculate_sharpness(
        pressure_samples=p_total_ear,
        sample_rate_hz=sample_rate_hz,
        algorithm=AcousticReferenceAlgorithm.DIN_45692,
    )
    assert 1.0 < sharpness_ear < 4.0  # Realistic golf impact sharpness

    spec_loud, total_loud = calculate_specific_loudness(
        pressure_samples=p_total_ear,
        sample_rate_hz=sample_rate_hz,
        algorithm=AcousticReferenceAlgorithm.ISO_532_1,
    )
    assert total_loud > 0.0
    assert len(spec_loud) == 24

    # 9. Held-out observer comparison
    comparison = HeldOutObserverComparison.compare_waveforms(
        predicted_pressure=p_total_front,
        target_pressure=p_total_front,
        observer=front_mic,
        sample_rate_hz=sample_rate_hz,
    )
    assert comparison.rms_error_pa == pytest.approx(0.0, abs=1e-10)
    assert comparison.peak_spl_error_db == pytest.approx(0.0, abs=1e-10)
    assert comparison.cross_correlation_max == pytest.approx(1.0, abs=1e-5)

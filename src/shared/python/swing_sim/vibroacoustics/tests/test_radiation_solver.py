"""Tests for transient vibroacoustic radiation solver (IA-T5, #5074)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.vibroacoustics._boundary_radiation_records import (
    AcousticMedium,
    RadiatingSurfaceMesh,
)
from shared.python.swing_sim.vibroacoustics.observer import ObserverLocation
from shared.python.swing_sim.vibroacoustics.radiation import (
    ModalRadiationTransfer,
    TransientRadiationSolver,
)


@pytest.mark.unit
def test_transient_radiation_delay_and_distance_scaling() -> None:
    # Plate of 0.05m x 0.05m
    mesh = RadiatingSurfaceMesh.planar_rectangular_plate(
        length_x_m=0.04, width_y_m=0.04, nx=4, ny=4
    )
    medium = AcousticMedium()
    solver = TransientRadiationSolver(mesh=mesh, medium=medium)

    # 1 kHz sinusoidal pulse of normal acceleration lasting 5 ms
    sample_rate_hz = 100_000.0  # 100 kHz sampling
    t = np.arange(0.0, 0.020, 1.0 / sample_rate_hz)  # 20 ms
    freq = 1000.0
    pulse_duration = 0.005
    accel_1d = np.where(
        t < pulse_duration, np.sin(2.0 * np.pi * freq * t) * 1000.0, 0.0
    )

    # Uniform normal acceleration across all elements
    accel_all = np.tile(accel_1d, (len(mesh.elements), 1))

    # Observers at r = 1.0 m and r = 2.0 m along normal z-axis
    obs_1m = ObserverLocation(name="mic_1m", coordinates_m=np.array([0.0, 0.0, 1.0]))
    obs_2m = ObserverLocation(name="mic_2m", coordinates_m=np.array([0.0, 0.0, 2.0]))

    p_1m = solver.solve_transient_pressure(
        observer=obs_1m, acceleration_matrix=accel_all, time=t
    )
    p_2m = solver.solve_transient_pressure(
        observer=obs_2m, acceleration_matrix=accel_all, time=t
    )

    # Expected arrival times:
    # t_arr_1m = 1.0 / 343.2 = 0.0029137 s (~291 samples)
    # t_arr_2m = 2.0 / 343.2 = 0.0058275 s (~583 samples)
    idx_start_1m = np.where(np.abs(p_1m) > 1e-3)[0][0]
    idx_start_2m = np.where(np.abs(p_2m) > 1e-3)[0][0]

    t_arr_1m = t[idx_start_1m]
    t_arr_2m = t[idx_start_2m]

    assert t_arr_1m == pytest.approx(1.0 / medium.sound_speed_mps, abs=2e-4)
    assert t_arr_2m == pytest.approx(2.0 / medium.sound_speed_mps, abs=2e-4)

    # 1/R spherical spreading: far-field peak pressure at 2m should be
    # approximately half of that at 1m
    peak_1m = np.max(np.abs(p_1m))
    peak_2m = np.max(np.abs(p_2m))
    assert peak_2m / peak_1m == pytest.approx(0.5, rel=0.05)


@pytest.mark.unit
def test_modal_radiation_transfer_and_superposition() -> None:
    mesh = RadiatingSurfaceMesh.planar_rectangular_plate(
        length_x_m=0.06, width_y_m=0.06, nx=6, ny=6
    )
    medium = AcousticMedium()
    solver = TransientRadiationSolver(mesh=mesh, medium=medium)

    # Define 2 modes: Mode 1 uniform piston (phi1 = 1),
    # Mode 2 antisymmetric (phi2 = x / L)
    def phi1(x: float, y: float) -> float:
        return 1.0

    def phi2(x: float, y: float) -> float:
        return x / 0.03

    modal_transfer = ModalRadiationTransfer(
        mesh=mesh,
        medium=medium,
        mode_shapes=(phi1, phi2),
        mode_names=("piston", "antisymmetric"),
    )

    sample_rate_hz = 50_000.0
    t = np.arange(0.0, 0.010, 1.0 / sample_rate_hz)
    # Modal accelerations
    q_ddot_1 = np.sin(2.0 * np.pi * 2000.0 * t) * 500.0 * np.exp(-t / 0.002)
    q_ddot_2 = np.cos(2.0 * np.pi * 4000.0 * t) * 300.0 * np.exp(-t / 0.002)
    modal_accel = np.vstack([q_ddot_1, q_ddot_2])

    obs = ObserverLocation(name="mic", coordinates_m=np.array([0.0, 0.0, 1.0]))
    p_modal = modal_transfer.solve_modal_pressure(
        observer=obs, modal_accelerations=modal_accel, time=t
    )

    # Direct element-by-element acceleration synthesis
    accel_all = np.zeros((len(mesh.elements), len(t)))
    for idx, el in enumerate(mesh.elements):
        x, y = el.centroid_m[0], el.centroid_m[1]
        accel_all[idx, :] = phi1(x, y) * q_ddot_1 + phi2(x, y) * q_ddot_2

    p_direct = solver.solve_transient_pressure(
        observer=obs, acceleration_matrix=accel_all, time=t
    )

    # Modal superposition must exactly match direct element integration
    assert np.allclose(p_modal, p_direct, atol=1e-6)


@pytest.mark.unit
def test_spatial_mesh_convergence_analytic_piston() -> None:
    # Analytic on-axis pressure amplitude for circular baffled piston of radius a:
    # p(z) = rho * c * v0 * 2 * sin(k * (sqrt(z^2 + a^2) - z) / 2)
    # For a small square plate of area S = pi * a^2, far-field on-axis pressure is:
    # p(z, omega) ~ rho * omega * v0 * S / (2 * pi * z)
    radius_a = 0.02
    area_S = np.pi * radius_a**2
    side = np.sqrt(area_S)
    z_dist = 1.0
    freq_hz = 1000.0
    omega = 2.0 * np.pi * freq_hz
    v0 = 0.01  # 1 cm/s
    medium = AcousticMedium()

    analytic_peak = (
        medium.density_kg_per_m3 * omega * v0 * area_S / (2.0 * np.pi * z_dist)
    )

    obs = ObserverLocation(name="on_axis", coordinates_m=np.array([0.0, 0.0, z_dist]))

    # Test mesh convergence: coarse (4x4) vs fine (12x12)
    errors = []
    for n in (4, 8, 16):
        mesh = RadiatingSurfaceMesh.planar_rectangular_plate(
            length_x_m=side, width_y_m=side, nx=n, ny=n
        )
        solver = TransientRadiationSolver(mesh=mesh, medium=medium)
        sample_rate_hz = 100_000.0
        t = np.arange(0.0, 0.010, 1.0 / sample_rate_hz)
        # Acceleration a(t) = d/dt (v0 sin(omega t)) = v0 * omega * cos(omega t)
        accel_1d = v0 * omega * np.cos(omega * t)
        accel_mat = np.tile(accel_1d, (len(mesh.elements), 1))
        p = solver.solve_transient_pressure(
            observer=obs, acceleration_matrix=accel_mat, time=t
        )

        # Skip arrival transient
        idx_steady = int((z_dist / medium.sound_speed_mps + 0.003) * sample_rate_hz)
        num_peak = np.max(np.abs(p[idx_steady:]))
        err = abs(num_peak - analytic_peak) / analytic_peak
        errors.append(err)

    # Convergence: all meshes accurately track the analytic far-field peak within 0.5%
    assert all(e < 0.005 for e in errors)
    assert errors[-1] < 0.002

"""Tests for boundary surface discretization and acoustic medium (IA-T5, #5074)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.vibroacoustics._boundary_radiation_records import (
    AcousticMedium,
    RadiatingElement,
    RadiatingSurfaceMesh,
)


@pytest.mark.unit
def test_acoustic_medium_properties_and_contracts() -> None:
    medium = AcousticMedium()
    assert medium.density_kg_per_m3 == pytest.approx(1.204)
    assert medium.sound_speed_mps == pytest.approx(343.2)

    with pytest.raises(ValueError, match="density"):
        AcousticMedium(density_kg_per_m3=-1.0)
    with pytest.raises(ValueError, match="sound speed"):
        AcousticMedium(sound_speed_mps=0.0)


@pytest.mark.unit
def test_radiating_element_contracts() -> None:
    element = RadiatingElement(
        element_id=1,
        area_m2=0.001,
        centroid_m=np.array([0.01, 0.02, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )
    assert element.element_id == 1
    assert element.area_m2 == 0.001
    assert np.allclose(element.centroid_m, [0.01, 0.02, 0.0])
    assert np.allclose(element.normal, [0.0, 0.0, 1.0])

    # Area must be strictly positive
    with pytest.raises(ValueError, match="area"):
        RadiatingElement(
            element_id=2,
            area_m2=0.0,
            centroid_m=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
        )

    # Normal must be nonzero / unit vector
    with pytest.raises(ValueError, match="normal"):
        RadiatingElement(
            element_id=3,
            area_m2=0.001,
            centroid_m=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 0.0]),
        )


@pytest.mark.unit
def test_radiating_surface_mesh_generation_and_properties() -> None:
    mesh = RadiatingSurfaceMesh.planar_rectangular_plate(
        length_x_m=0.10,
        width_y_m=0.05,
        nx=10,
        ny=5,
    )
    assert len(mesh.elements) == 50
    assert mesh.total_area_m2 == pytest.approx(0.10 * 0.05)
    assert np.allclose(mesh.centroid_m, [0.0, 0.0, 0.0], atol=1e-6)

    # Mesh convergence check:
    # Elements are ~0.01 m in size.
    # At 5 kHz, lambda = 343.2 / 5000 = 0.0686 m. lambda / 6 = 0.0114 m.
    # Mesh with h ~ 0.01 m satisfies convergence at 5 kHz, but not at 20 kHz.
    assert mesh.validate_mesh_convergence(max_frequency_hz=5000.0) is True
    with pytest.raises(ValueError, match="too coarse"):
        mesh.validate_mesh_convergence(max_frequency_hz=20000.0)

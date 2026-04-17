"""Tests for glass_bath_fea.core.config module.

Covers:
- GlassComposition validation and total_percent
- MeshConfig defaults
- GlassBathFEAConfig computed properties
"""

from __future__ import annotations

import math

import pytest

from glass_bath_fea.core.config import (
    INCHES_TO_METERS,
    GlassBathFEAConfig,
    GlassComposition,
    MeshConfig,
)


class TestGlassComposition:
    """Tests for GlassComposition dataclass."""

    def test_default_valid(self) -> None:
        gc = GlassComposition()
        assert gc.validate()

    def test_total_percent(self) -> None:
        gc = GlassComposition()
        total = gc.total_percent()
        assert 99.0 <= total <= 101.0

    def test_custom_invalid_composition(self) -> None:
        gc = GlassComposition(
            sio2=50.0, na2o=5.0, cao=5.0, mgo=0.0, al2o3=0.0, fe2o3=0.0
        )
        assert gc.validate() is False

    def test_custom_valid_composition(self) -> None:
        gc = GlassComposition(
            sio2=80.0, na2o=10.0, cao=5.0, mgo=3.0, al2o3=1.5, fe2o3=0.5
        )
        assert gc.validate()


class TestMeshConfig:
    """Tests for MeshConfig dataclass."""

    def test_defaults(self) -> None:
        mc = MeshConfig()
        assert mc.element_size_glass == 0.01
        assert mc.mesh_algorithm == "delaunay"
        assert mc.mesh_order == 1
        assert mc.export_format == "msh22"


class TestGlassBathFEAConfig:
    """Tests for GlassBathFEAConfig dataclass."""

    def test_default_construction(self) -> None:
        cfg = GlassBathFEAConfig()
        assert cfg.bath_diameter == 120.0
        assert cfg.num_electrodes == 3

    def test_total_height(self) -> None:
        cfg = GlassBathFEAConfig(glass_depth=15.0, metal_layer_thickness=2.0)
        assert cfg.total_height == pytest.approx(17.0)

    def test_bath_radius(self) -> None:
        cfg = GlassBathFEAConfig(bath_diameter=120.0)
        assert cfg.bath_radius == pytest.approx(60.0)

    def test_get_dimensions_meters(self) -> None:
        cfg = GlassBathFEAConfig()
        dims = cfg.get_dimensions_meters()
        assert "bath_diameter" in dims
        assert "glass_depth" in dims
        assert dims["bath_diameter"] == pytest.approx(120.0 * INCHES_TO_METERS)

    def test_electrode_angles_three_electrodes(self) -> None:
        cfg = GlassBathFEAConfig(num_electrodes=3, electrode_spacing_degrees=120.0)
        angles = cfg.get_electrode_angles_radians()
        assert len(angles) == 3
        assert angles[0] == pytest.approx(0.0)
        assert angles[1] == pytest.approx(2 * math.pi / 3, rel=1e-6)
        assert angles[2] == pytest.approx(4 * math.pi / 3, rel=1e-6)

    def test_electrode_angles_six_electrodes(self) -> None:
        cfg = GlassBathFEAConfig(num_electrodes=6, electrode_spacing_degrees=60.0)
        angles = cfg.get_electrode_angles_radians()
        assert len(angles) == 6
        assert angles[-1] == pytest.approx(5 * math.pi / 3, rel=1e-6)

    def test_conversion_factor(self) -> None:
        assert INCHES_TO_METERS == pytest.approx(0.0254)

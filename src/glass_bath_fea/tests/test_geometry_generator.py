"""Tests for Glass Bath FEA geometry generator."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add paths for imports (when running tests directly)
TOOLS_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(TOOLS_ROOT / "src"))


class TestGeometryGenerator:
    """Tests for cylindrical vessel geometry generation."""

    def test_create_vessel_geometry(self, default_fea_config) -> None:
        """Test creating basic cylindrical vessel geometry."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        geometry = gen.create_vessel_geometry()

        # Should return a dict with geometry components
        assert geometry is not None
        assert "glass_region" in geometry
        assert "metal_region" in geometry

    def test_vessel_dimensions(self, default_fea_config) -> None:
        """Test that vessel dimensions match configuration."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        dims = gen.get_dimensions()

        # Convert expected values to meters
        inches_to_meters = 0.0254
        expected_radius = (default_fea_config.bath_diameter / 2) * inches_to_meters
        expected_glass_depth = default_fea_config.glass_depth * inches_to_meters
        expected_metal_thickness = (
            default_fea_config.metal_layer_thickness * inches_to_meters
        )

        assert dims["radius"] == pytest.approx(expected_radius, rel=1e-6)
        assert dims["glass_depth"] == pytest.approx(expected_glass_depth, rel=1e-6)
        assert dims["metal_thickness"] == pytest.approx(
            expected_metal_thickness, rel=1e-6
        )

    def test_vessel_volume(self, default_fea_config) -> None:
        """Test that vessel volumes are physically reasonable."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        volumes = gen.calculate_region_volumes()

        # All volumes should be positive
        assert volumes["glass"] > 0
        assert volumes["metal"] > 0
        assert volumes["total"] > 0

        # Total should equal sum of parts
        assert volumes["total"] == pytest.approx(
            volumes["glass"] + volumes["metal"], rel=1e-6
        )

    def test_cylindrical_coordinates(self, default_fea_config) -> None:
        """Test cylindrical coordinate system setup."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)

        # Test point on vessel wall
        radius = gen.get_dimensions()["radius"]
        angle = math.pi / 4

        point = gen.cylindrical_to_cartesian(radius, angle, 0.0)

        assert point[0] == pytest.approx(radius * math.cos(angle), rel=1e-6)
        assert point[1] == pytest.approx(radius * math.sin(angle), rel=1e-6)
        assert point[2] == pytest.approx(0.0, abs=1e-10)


class TestElectrodeGeometry:
    """Tests for electrode geometry generation."""

    def test_electrode_positions(self, default_fea_config) -> None:
        """Test that electrode positions are correctly computed."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        positions = gen.get_electrode_positions()

        # Should have 3 electrodes
        assert len(positions) == 3

        # Each position should have required fields
        for pos in positions:
            assert "tip" in pos
            assert "base" in pos
            assert "angle" in pos

    def test_electrode_angular_spacing(self, default_fea_config) -> None:
        """Test that electrodes are spaced at 120 degrees."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        positions = gen.get_electrode_positions()

        # Calculate angular spacing between electrodes
        angles = [pos["angle"] for pos in positions]
        expected_spacing = 2 * math.pi / 3  # 120 degrees in radians

        for i in range(3):
            j = (i + 1) % 3
            spacing = (angles[j] - angles[i]) % (2 * math.pi)
            assert spacing == pytest.approx(expected_spacing, rel=1e-6)

    def test_electrode_tip_inside_vessel(self, default_fea_config) -> None:
        """Test that electrode tips are inside the vessel boundary."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        positions = gen.get_electrode_positions()
        radius = gen.get_dimensions()["radius"]

        for pos in positions:
            tip = pos["tip"]
            # Distance from center should be less than radius
            r_tip = np.sqrt(tip[0] ** 2 + tip[1] ** 2)
            assert r_tip < radius

    def test_electrode_base_on_vessel_wall(self, default_fea_config) -> None:
        """Test that electrode bases are on the vessel wall."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        positions = gen.get_electrode_positions()
        radius = gen.get_dimensions()["radius"]

        for pos in positions:
            base = pos["base"]
            # Distance from center should equal radius
            r_base = np.sqrt(base[0] ** 2 + base[1] ** 2)
            assert r_base == pytest.approx(radius, rel=1e-6)

    def test_electrode_insertion_depth(self, default_fea_config) -> None:
        """Test that electrode insertion depth matches configuration."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        positions = gen.get_electrode_positions()
        inches_to_meters = 0.0254
        expected_depth = default_fea_config.electrode_insertion_depth * inches_to_meters

        for pos in positions:
            tip = pos["tip"]
            base = pos["base"]

            # Distance from base to tip should equal insertion depth
            distance = np.sqrt(np.sum((np.array(base) - np.array(tip)) ** 2))
            assert distance == pytest.approx(expected_depth, rel=1e-3)


class TestRegionDefinitions:
    """Tests for material region definitions."""

    def test_glass_region_bounds(self, default_fea_config) -> None:
        """Test glass region Z-bounds."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        bounds = gen.get_region_bounds()

        inches_to_meters = 0.0254
        metal_top = default_fea_config.metal_layer_thickness * inches_to_meters
        glass_top = (
            default_fea_config.metal_layer_thickness + default_fea_config.glass_depth
        ) * inches_to_meters

        # Glass region should start above metal and extend to top
        assert bounds["glass"]["z_min"] == pytest.approx(metal_top, rel=1e-6)
        assert bounds["glass"]["z_max"] == pytest.approx(glass_top, rel=1e-6)

    def test_metal_region_bounds(self, default_fea_config) -> None:
        """Test metal region Z-bounds."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        bounds = gen.get_region_bounds()

        inches_to_meters = 0.0254
        metal_top = default_fea_config.metal_layer_thickness * inches_to_meters

        # Metal region should be at bottom
        assert bounds["metal"]["z_min"] == pytest.approx(0.0, abs=1e-10)
        assert bounds["metal"]["z_max"] == pytest.approx(metal_top, rel=1e-6)

    def test_regions_connected(self, default_fea_config) -> None:
        """Test that glass and metal regions share an interface."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        bounds = gen.get_region_bounds()

        # Metal top should equal glass bottom
        assert bounds["metal"]["z_max"] == pytest.approx(
            bounds["glass"]["z_min"], rel=1e-6
        )

    def test_material_id_assignment(self, default_fea_config) -> None:
        """Test that material IDs are correctly assigned."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        material_ids = gen.get_material_ids()

        # Should have distinct IDs for glass and metal
        assert "glass" in material_ids
        assert "metal" in material_ids
        assert material_ids["glass"] != material_ids["metal"]


class TestGeometryExport:
    """Tests for geometry export functionality."""

    def test_export_to_dict(self, default_fea_config) -> None:
        """Test exporting geometry as dictionary for mesh generation."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        data = gen.export_geometry_data()

        # Should contain all necessary data for mesh generation
        assert "dimensions" in data
        assert "electrodes" in data
        assert "regions" in data
        assert "material_ids" in data

    def test_export_electrode_cylinders(self, default_fea_config) -> None:
        """Test that electrode geometry includes cylinder definitions."""
        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(default_fea_config)
        data = gen.export_geometry_data()

        inches_to_meters = 0.0254
        expected_diameter = default_fea_config.electrode_diameter * inches_to_meters

        for electrode in data["electrodes"]:
            # Each electrode should define a cylinder
            assert "radius" in electrode
            assert electrode["radius"] == pytest.approx(expected_diameter / 2, rel=1e-6)

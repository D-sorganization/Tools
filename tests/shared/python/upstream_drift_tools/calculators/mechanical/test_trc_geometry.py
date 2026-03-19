"""Comprehensive tests for TRC Geometry Engine.

Tests cover LayerConfig, VesselDimensions, helper geometry functions,
TRCGeometryEngine.calculate_geometry, and calculate_residence_time.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.calculators.mechanical.trc_geometry import (
    LayerConfig,
    LayerResult,
    TRCGeometryEngine,
    VesselDimensions,
    VesselGeometryResult,
)

# ─── Fixtures ─────────────────────────────────────────────────


def _make_dimensions(
    cyl_h: float = 60.0,
    cyl_d: float = 48.0,
    cone_h: float = 24.0,
    cone_bot_d: float = 12.0,
    cone_hole: float = 6.0,
    top_ref: float = 3.0,
) -> VesselDimensions:
    return VesselDimensions(
        cylinder_height=cyl_h,
        cylinder_diameter=cyl_d,
        cone_height=cone_h,
        cone_bottom_diameter=cone_bot_d,
        cone_interior_hole=cone_hole,
        top_refractory_thickness=top_ref,
    )


def _make_layers() -> list[LayerConfig]:
    return [
        LayerConfig(name="Metal Shell", thickness=0.5, density=490.0, color="#888888"),
        LayerConfig(name="Refractory A", thickness=4.5, density=150.0, color="#cc6633"),
        LayerConfig(name="Refractory B", thickness=3.0, density=120.0, color="#ffcc66"),
    ]


# ─── LayerConfig Tests ───────────────────────────────────────


class TestLayerConfig:
    """Test the LayerConfig dataclass."""

    def test_basic_construction(self) -> None:
        layer = LayerConfig(name="Shell", thickness=0.5, density=490.0, color="#aaa")
        assert layer.name == "Shell"
        assert layer.thickness == 0.5
        assert layer.density == 490.0

    def test_default_visible(self) -> None:
        layer = LayerConfig(name="Shell", thickness=1.0, density=100.0, color="red")
        assert layer.visible is True

    def test_default_transparency(self) -> None:
        layer = LayerConfig(name="Shell", thickness=1.0, density=100.0, color="red")
        assert layer.transparency == 0.3

    def test_auto_top_section_name(self) -> None:
        layer = LayerConfig(name="Brick", thickness=1.0, density=100.0, color="red")
        assert layer.top_section_name == "Brick Top"

    def test_custom_top_section_name(self) -> None:
        layer = LayerConfig(
            name="Brick",
            thickness=1.0,
            density=100.0,
            color="red",
            top_section_name="Custom",
        )
        assert layer.top_section_name == "Custom"

    def test_post_init_converts_types(self) -> None:
        layer = LayerConfig(name="Shell", thickness=1, density=100, color="red")
        assert isinstance(layer.thickness, float)
        assert isinstance(layer.density, float)


# ─── VesselDimensions Tests ──────────────────────────────────


class TestVesselDimensions:
    """Test VesselDimensions dataclass."""

    def test_default_display_flags(self) -> None:
        d = _make_dimensions()
        assert d.display_lid is True
        assert d.display_cylinder is True
        assert d.display_cone is True

    def test_custom_values(self) -> None:
        d = VesselDimensions(
            cylinder_height=100.0,
            cylinder_diameter=80.0,
            cone_height=30.0,
            cone_bottom_diameter=20.0,
            cone_interior_hole=10.0,
            top_refractory_thickness=5.0,
        )
        assert d.cylinder_height == 100.0
        assert d.cylinder_diameter == 80.0


# ─── LayerResult & VesselGeometryResult Tests ────────────────


class TestResults:
    def test_layer_result_defaults(self) -> None:
        lr = LayerResult(name="Test", volume_ft3=1.0, mass_lb=100.0, density=100.0)
        assert lr.outer_surface_area_ft2 == 0.0

    def test_vessel_result_defaults(self) -> None:
        vr = VesselGeometryResult()
        assert vr.layers == []
        assert vr.total_volume_ft3 == 0.0
        assert vr.total_mass_lb == 0.0


# ─── TRCGeometryEngine.calculate_geometry Tests ──────────────


class TestCalculateGeometry:
    """Test the main geometry calculation."""

    def test_returns_result_type(self) -> None:
        engine = TRCGeometryEngine()
        dims = _make_dimensions()
        layers = _make_layers()
        result = engine.calculate_geometry(dims, layers)
        assert isinstance(result, VesselGeometryResult)

    def test_correct_number_of_layers(self) -> None:
        engine = TRCGeometryEngine()
        dims = _make_dimensions()
        layers = _make_layers()
        result = engine.calculate_geometry(dims, layers)
        assert len(result.layers) == 3

    def test_total_volume_positive(self) -> None:
        engine = TRCGeometryEngine()
        dims = _make_dimensions()
        layers = _make_layers()
        result = engine.calculate_geometry(dims, layers)
        assert result.total_volume_ft3 > 0.0

    def test_total_mass_positive(self) -> None:
        engine = TRCGeometryEngine()
        dims = _make_dimensions()
        layers = _make_layers()
        result = engine.calculate_geometry(dims, layers)
        assert result.total_mass_lb > 0.0

    def test_interior_volume_positive(self) -> None:
        engine = TRCGeometryEngine()
        dims = _make_dimensions()
        layers = _make_layers()
        result = engine.calculate_geometry(dims, layers)
        assert result.interior_volume_ft3 > 0.0

    def test_void_diameter_reasonable(self) -> None:
        engine = TRCGeometryEngine()
        dims = _make_dimensions()
        layers = _make_layers()
        result = engine.calculate_geometry(dims, layers)
        # Void diameter should be less than the cylinder diameter
        assert result.void_diameter_inches < dims.cylinder_diameter
        assert result.void_diameter_inches > 0.0

    def test_metal_shell_has_surface_area(self) -> None:
        engine = TRCGeometryEngine()
        dims = _make_dimensions()
        layers = _make_layers()
        result = engine.calculate_geometry(dims, layers)
        shell = [lr for lr in result.layers if lr.name == "Metal Shell"][0]
        assert shell.outer_surface_area_ft2 > 0.0

    def test_non_shell_layers_zero_surface_area(self) -> None:
        engine = TRCGeometryEngine()
        dims = _make_dimensions()
        layers = _make_layers()
        result = engine.calculate_geometry(dims, layers)
        for lr in result.layers:
            if lr.name != "Metal Shell":
                assert lr.outer_surface_area_ft2 == 0.0

    def test_empty_layers_returns_empty_result(self) -> None:
        engine = TRCGeometryEngine()
        dims = _make_dimensions()
        result = engine.calculate_geometry(dims, [])
        assert result.total_volume_ft3 == 0.0
        assert result.total_mass_lb == 0.0
        assert len(result.layers) == 0

    def test_invisible_layer_excluded(self) -> None:
        engine = TRCGeometryEngine()
        dims = _make_dimensions()
        layers = [
            LayerConfig(
                name="Hidden", thickness=2.0, density=100.0, color="red", visible=False
            ),
        ]
        result = engine.calculate_geometry(dims, layers)
        assert len(result.layers) == 0

    def test_zero_thickness_layer_excluded(self) -> None:
        engine = TRCGeometryEngine()
        dims = _make_dimensions()
        layers = [
            LayerConfig(name="Thin", thickness=0.0, density=100.0, color="red"),
        ]
        result = engine.calculate_geometry(dims, layers)
        assert len(result.layers) == 0

    def test_negative_diameter_raises(self) -> None:
        engine = TRCGeometryEngine()
        dims = VesselDimensions(
            cylinder_height=60.0,
            cylinder_diameter=-10.0,
            cone_height=24.0,
            cone_bottom_diameter=12.0,
            cone_interior_hole=6.0,
            top_refractory_thickness=3.0,
        )
        with pytest.raises(AssertionError, match="cylinder_diameter"):
            engine.calculate_geometry(dims, _make_layers())

    def test_negative_height_raises(self) -> None:
        engine = TRCGeometryEngine()
        dims = VesselDimensions(
            cylinder_height=-5.0,
            cylinder_diameter=48.0,
            cone_height=24.0,
            cone_bottom_diameter=12.0,
            cone_interior_hole=6.0,
            top_refractory_thickness=3.0,
        )
        with pytest.raises(AssertionError, match="cylinder_height"):
            engine.calculate_geometry(dims, _make_layers())


class TestCalculateGeometryDisplayFlags:
    """Test display flag effects."""

    def test_no_lid_changes_interior(self) -> None:
        engine = TRCGeometryEngine()
        dims_lid = _make_dimensions()
        dims_lid.display_lid = True
        dims_no_lid = _make_dimensions()
        dims_no_lid.display_lid = False
        layers = _make_layers()

        r_lid = engine.calculate_geometry(dims_lid, layers)
        r_no = engine.calculate_geometry(dims_no_lid, layers)
        # Without lid, the interior height is different
        assert r_lid.total_volume_ft3 != r_no.total_volume_ft3

    def test_no_cylinder_reduces_volume(self) -> None:
        engine = TRCGeometryEngine()
        dims_full = _make_dimensions()
        dims_no_cyl = _make_dimensions()
        dims_no_cyl.display_cylinder = False
        layers = _make_layers()

        r_full = engine.calculate_geometry(dims_full, layers)
        r_no_cyl = engine.calculate_geometry(dims_no_cyl, layers)
        assert r_full.total_volume_ft3 > r_no_cyl.total_volume_ft3

    def test_no_cone_reduces_volume(self) -> None:
        engine = TRCGeometryEngine()
        dims_full = _make_dimensions()
        dims_no_cone = _make_dimensions()
        dims_no_cone.display_cone = False
        layers = _make_layers()

        r_full = engine.calculate_geometry(dims_full, layers)
        r_no_cone = engine.calculate_geometry(dims_no_cone, layers)
        assert r_full.total_volume_ft3 > r_no_cone.total_volume_ft3


# ─── Residence Time Tests ─────────────────────────────────────


class TestResidenceTime:
    """Test calculate_residence_time."""

    def test_basic_calculation(self) -> None:
        engine = TRCGeometryEngine()
        # 100 ft3, 50 acfm = 2 minutes = 120 seconds
        rt = engine.calculate_residence_time(100.0, 50.0)
        assert abs(rt - 120.0) < 0.01

    def test_zero_flow_returns_zero(self) -> None:
        engine = TRCGeometryEngine()
        rt = engine.calculate_residence_time(100.0, 0.0)
        assert rt == 0.0

    def test_negative_flow_returns_zero(self) -> None:
        engine = TRCGeometryEngine()
        rt = engine.calculate_residence_time(100.0, -10.0)
        assert rt == 0.0

    def test_higher_volume_longer_time(self) -> None:
        engine = TRCGeometryEngine()
        rt1 = engine.calculate_residence_time(50.0, 25.0)
        rt2 = engine.calculate_residence_time(100.0, 25.0)
        assert rt2 > rt1

    def test_higher_flow_shorter_time(self) -> None:
        engine = TRCGeometryEngine()
        rt1 = engine.calculate_residence_time(100.0, 25.0)
        rt2 = engine.calculate_residence_time(100.0, 50.0)
        assert rt2 < rt1

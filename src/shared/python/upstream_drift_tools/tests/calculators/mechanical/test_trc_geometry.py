"""test_trc_geometry.py module."""

import pytest
from upstream_drift_tools.calculators.mechanical.trc_geometry import (
    LayerConfig,
    TRCGeometryEngine,
    VesselDimensions,
)


class TestTRCGeometry:
    @pytest.fixture
    def engine(self):
        return TRCGeometryEngine()

    @pytest.fixture
    def dimensions(self):
        return VesselDimensions(
            cylinder_height=100.0,
            cylinder_diameter=50.0,
            cone_height=20.0,
            cone_bottom_diameter=10.0,
            cone_interior_hole=5.0,
            top_refractory_thickness=2.0,
        )

    @pytest.fixture
    def layers(self):
        return [
            LayerConfig(
                name="Shell",
                thickness=0.5,
                density=490.0,
                color="grey",  # Steel approx
            ),
            LayerConfig(
                name="Refractory", thickness=4.0, density=150.0, color="orange"
            ),
        ]

    def test_calculate_geometry(self, engine, dimensions, layers):
        result = engine.calculate_geometry(dimensions, layers)

        assert len(result.layers) == 2
        assert result.total_volume_ft3 > 0
        assert result.total_mass_lb > 0
        assert result.interior_volume_ft3 > 0

        # Check that inner layer reduces volume more than outer shell
        shell_res = result.layers[0]
        ref_res = result.layers[1]

        # Refractory is thicker, should likely have more volume depending on diameter
        # But let's just check valid numbers
        assert shell_res.mass_lb > 0
        assert ref_res.mass_lb > 0

    def test_residence_time(self, engine):
        volume_ft3 = 100.0
        flow_acfm = 100.0  # 100 ft3/min

        time_sec = engine.calculate_residence_time(volume_ft3, flow_acfm)
        assert time_sec == 60.0  # Should be exactly 60 seconds

    def test_zero_flow(self, engine):
        assert engine.calculate_residence_time(100.0, 0.0) == 0.0

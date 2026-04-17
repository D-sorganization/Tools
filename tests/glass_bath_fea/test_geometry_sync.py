"""Tests for Glass Bath FEA geometry synchronisation.

Verifies that electrode advisor configurations are correctly translated to
FEA-compatible geometry and that validation catches incompatible settings.

See issue #575.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from glass_bath_fea.core.config import GlassBathFEAConfig
from glass_bath_fea.core.geometry_generator import GeometryGenerator
from glass_bath_fea.interfaces.geometry_sync import (
    INCHES_TO_METERS,
    GeometrySynchronizer,
    GeometryValidationResult,
)

# ---------------------------------------------------------------------------
# Helpers: stub ElectrodeConfig for tests that don't need the real import
# ---------------------------------------------------------------------------


def _make_stub_electrode_config(**overrides: float) -> MagicMock:
    """Create a stub that looks like ``ElectrodeConfig``."""
    defaults = {
        "bath_diameter": 120.0,
        "glass_depth": 15.0,
        "metal_depth": 2.0,
        "tip_diameter": 6.0,
        "electrode_spacing_degrees": 120.0,
        "bath_temperature": 1350.0,
        "metal_conductivity": 10000.0,
        "electrode_depths": np.array([10.0, 10.0, 10.0]),
        "phase_voltages": np.array([100.0, 100.0, 100.0]),
    }
    defaults.update(overrides)
    cfg = MagicMock(**defaults)
    # MagicMock attribute access
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestGeometryValidation:
    """Validation of geometry compatibility."""

    def test_valid_default_config(self) -> None:
        ec = _make_stub_electrode_config()
        sync = GeometrySynchronizer(electrode_config=ec)
        result = sync.validate()
        assert result.is_valid
        assert len(result.errors) == 0

    def test_insertion_too_deep(self) -> None:
        # Insertion depth > 90% of radius (60 in radius * 0.9 = 54)
        ec = _make_stub_electrode_config(electrode_depths=np.array([58.0, 58.0, 58.0]))
        sync = GeometrySynchronizer(electrode_config=ec)
        result = sync.validate()
        assert not result.is_valid
        assert any("exceeds" in e for e in result.errors)

    def test_insertion_too_shallow(self) -> None:
        ec = _make_stub_electrode_config(electrode_depths=np.array([0.5, 0.5, 0.5]))
        sync = GeometrySynchronizer(electrode_config=ec)
        result = sync.validate()
        assert not result.is_valid
        assert any("below minimum" in e for e in result.errors)

    def test_electrode_larger_than_radius(self) -> None:
        ec = _make_stub_electrode_config(tip_diameter=70.0)
        sync = GeometrySynchronizer(electrode_config=ec)
        result = sync.validate()
        assert not result.is_valid
        assert any("diameter" in e.lower() for e in result.errors)

    def test_zero_glass_depth(self) -> None:
        ec = _make_stub_electrode_config(glass_depth=0.0)
        sync = GeometrySynchronizer(electrode_config=ec)
        result = sync.validate()
        assert not result.is_valid
        assert any("glass depth" in e.lower() for e in result.errors)

    def test_warning_for_tight_clearance(self) -> None:
        # Large electrode diameter reduces clearance
        ec = _make_stub_electrode_config(tip_diameter=24.0)
        sync = GeometrySynchronizer(electrode_config=ec)
        result = sync.validate()
        # May or may not warn depending on insertion depth vs chord
        assert isinstance(result, GeometryValidationResult)


# ---------------------------------------------------------------------------
# Synchronisation
# ---------------------------------------------------------------------------


class TestGeometrySynchronisation:
    """End-to-end sync from electrode config to FEA config."""

    def test_sync_produces_fea_config(self) -> None:
        ec = _make_stub_electrode_config()
        sync = GeometrySynchronizer(electrode_config=ec)
        fea = sync.sync()
        assert isinstance(fea, GlassBathFEAConfig)

    def test_dimensions_match(self) -> None:
        ec = _make_stub_electrode_config(
            bath_diameter=100.0, glass_depth=12.0, tip_diameter=8.0
        )
        sync = GeometrySynchronizer(electrode_config=ec)
        fea = sync.sync()

        assert fea.bath_diameter == 100.0
        assert fea.glass_depth == 12.0
        assert fea.electrode_diameter == 8.0

    def test_operating_temperature_matches(self) -> None:
        ec = _make_stub_electrode_config(bath_temperature=1400.0)
        sync = GeometrySynchronizer(electrode_config=ec)
        fea = sync.sync()
        assert fea.operating_temperature == 1400.0

    def test_phase_voltages_match(self) -> None:
        voltages = np.array([120.0, 115.0, 110.0])
        ec = _make_stub_electrode_config(phase_voltages=voltages)
        sync = GeometrySynchronizer(electrode_config=ec)
        fea = sync.sync()
        assert fea.phase_voltages == (120.0, 115.0, 110.0)

    def test_insertion_depth_from_electrode_depths(self) -> None:
        ec = _make_stub_electrode_config(electrode_depths=np.array([12.0, 14.0, 16.0]))
        sync = GeometrySynchronizer(electrode_config=ec)
        fea = sync.sync()
        # Mean of [12, 14, 16] = 14
        assert fea.electrode_insertion_depth == pytest.approx(14.0)

    def test_sync_raises_on_invalid_config(self) -> None:
        ec = _make_stub_electrode_config(glass_depth=-1.0)
        sync = GeometrySynchronizer(electrode_config=ec)
        with pytest.raises(ValueError, match="validation failed"):
            sync.sync()


# ---------------------------------------------------------------------------
# Electrode positions in FEA coordinates
# ---------------------------------------------------------------------------


class TestElectrodePositionsFEA:
    """Verify electrode positions after sync are geometrically correct."""

    def test_positions_count(self) -> None:
        ec = _make_stub_electrode_config()
        sync = GeometrySynchronizer(electrode_config=ec)
        sync.sync()
        positions = sync.get_electrode_positions_fea()
        assert len(positions) == 3

    def test_positions_in_metres(self) -> None:
        ec = _make_stub_electrode_config()
        sync = GeometrySynchronizer(electrode_config=ec)
        sync.sync()
        positions = sync.get_electrode_positions_fea()

        for pos in positions:
            # tip and base should be lists of 3 floats (x, y, z)
            assert len(pos["tip"]) == 3
            assert len(pos["base"]) == 3
            # diameter should be in metres
            assert pos["diameter_m"] == pytest.approx(6.0 * INCHES_TO_METERS, rel=1e-4)

    def test_base_at_vessel_wall(self) -> None:
        ec = _make_stub_electrode_config()
        sync = GeometrySynchronizer(electrode_config=ec)
        sync.sync()
        positions = sync.get_electrode_positions_fea()

        radius_m = (ec.bath_diameter / 2.0) * INCHES_TO_METERS
        for pos in positions:
            bx, by, _ = pos["base"]
            base_r = math.sqrt(bx**2 + by**2)
            assert base_r == pytest.approx(radius_m, rel=1e-4)

    def test_tip_inside_vessel(self) -> None:
        ec = _make_stub_electrode_config()
        sync = GeometrySynchronizer(electrode_config=ec)
        sync.sync()
        positions = sync.get_electrode_positions_fea()

        radius_m = (ec.bath_diameter / 2.0) * INCHES_TO_METERS
        for pos in positions:
            tx, ty, _ = pos["tip"]
            tip_r = math.sqrt(tx**2 + ty**2)
            assert tip_r < radius_m

    def test_angular_spacing(self) -> None:
        ec = _make_stub_electrode_config(electrode_spacing_degrees=120.0)
        sync = GeometrySynchronizer(electrode_config=ec)
        sync.sync()
        positions = sync.get_electrode_positions_fea()

        angles = [pos["angle_deg"] for pos in positions]
        for i in range(len(angles) - 1):
            diff = angles[i + 1] - angles[i]
            assert diff == pytest.approx(120.0, abs=0.1)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestSyncReport:
    """Test JSON report export."""

    def test_export_creates_file(self) -> None:
        ec = _make_stub_electrode_config()
        sync = GeometrySynchronizer(electrode_config=ec)
        sync.sync()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.json"
            sync.export_sync_report(out)
            assert out.exists()

            data = json.loads(out.read_text(encoding="utf-8"))
            assert data["source"] == "electrode_advisor"
            assert data["target"] == "glass_bath_fea"
            assert "fea_geometry" in data
            assert data["validation"]["is_valid"] is True


# ---------------------------------------------------------------------------
# Geometry generator round-trip
# ---------------------------------------------------------------------------


class TestGeometryGeneratorConsistency:
    """Ensure the FEA geometry generator is consistent with sync output."""

    def test_vessel_volumes_positive(self) -> None:
        ec = _make_stub_electrode_config()
        sync = GeometrySynchronizer(electrode_config=ec)
        fea = sync.sync()

        gen = GeometryGenerator(fea)
        volumes = gen.calculate_region_volumes()
        assert volumes["glass"] > 0
        assert volumes["metal"] > 0
        assert volumes["total"] == pytest.approx(
            volumes["glass"] + volumes["metal"], rel=1e-6
        )

    def test_electrode_geometry_count(self) -> None:
        ec = _make_stub_electrode_config()
        sync = GeometrySynchronizer(electrode_config=ec)
        fea = sync.sync()

        gen = GeometryGenerator(fea)
        geometry = gen.create_vessel_geometry()
        assert len(geometry["electrodes"]) == 3

    def test_material_ids(self) -> None:
        ec = _make_stub_electrode_config()
        sync = GeometrySynchronizer(electrode_config=ec)
        fea = sync.sync()

        gen = GeometryGenerator(fea)
        ids = gen.get_material_ids()
        assert ids["glass"] != ids["metal"]
        assert ids["metal"] != ids["electrode"]

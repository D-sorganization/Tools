"""Tests for Flare Calculator GUI components."""

from __future__ import annotations

import math

import pytest


class TestFlareCalculatorEngine:
    """Test the standalone flare calculator engine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        from flare_calculator.ui.pyqt6.main_window import FlareCalculatorEngine

        return FlareCalculatorEngine()

    def test_calculate_flare_size_basic(self, engine):
        """Test basic flare sizing calculation."""
        composition = {"H2": 35, "CO": 30, "CH4": 5, "CO2": 15, "N2": 5, "H2O": 10}
        design = engine.calculate_flare_size(
            total_flow=1000,
            gas_composition=composition,
            temperature=473,
            pressure=1.5,
        )

        assert design.height > 10  # Minimum height
        assert design.diameter > 0
        assert design.exit_velocity == 170  # Target velocity
        assert design.heat_release > 0

    def test_calculate_flare_size_high_hydrogen(self, engine):
        """Test flare sizing with high hydrogen content."""
        composition = {"H2": 80, "CO": 10, "N2": 10}
        design = engine.calculate_flare_size(
            total_flow=500,
            gas_composition=composition,
            temperature=500,
            pressure=2.0,
        )

        # High hydrogen should give high heat release
        assert design.heat_release > 10000  # kW
        assert design.height > 10

    def test_calculate_radiation_zones(self, engine):
        """Test radiation zone calculation."""
        from flare_calculator.ui.pyqt6.main_window import FlareDesign

        design = FlareDesign(
            height=30,
            diameter=0.1,
            exit_velocity=170,
            heat_release=10000,
            radiation_intensity=1.6,
        )

        zones = engine.calculate_radiation_zones(design)

        # Lethal zone should be closest
        assert zones["lethal"] < zones["damage"]
        assert zones["damage"] < zones["safe"]
        assert zones["safe"] < zones["comfort"]

    def test_calculate_flare_size_zero_flow(self, engine):
        """Test with zero flow rate."""
        composition = {"H2": 50, "CO": 50}
        design = engine.calculate_flare_size(
            total_flow=0,
            gas_composition=composition,
            temperature=473,
            pressure=1.5,
        )

        assert design.heat_release == 0
        assert design.height == 10  # Minimum height

    def test_calculate_flare_size_inert_gas(self, engine):
        """Test with inert gas only."""
        composition = {"N2": 100}
        design = engine.calculate_flare_size(
            total_flow=1000,
            gas_composition=composition,
            temperature=473,
            pressure=1.5,
        )

        # No combustible gas, no heat release
        assert design.heat_release == 0


class TestFlareCalculatorMainWindow:
    """Test the main window initialization."""

    @pytest.fixture
    def app(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_main_window_creation(self, app):
        """Test that main window can be created."""
        from flare_calculator.ui.pyqt6.main_window import FlareCalculatorMainWindow

        window = FlareCalculatorMainWindow()
        assert window is not None
        assert window.windowTitle() == "Flare Calculator"

    def test_main_window_has_engine(self, app):
        """Test that main window has calculation engine."""
        from flare_calculator.ui.pyqt6.main_window import FlareCalculatorMainWindow

        window = FlareCalculatorMainWindow()
        assert window.engine is not None

    def test_main_window_has_gas_inputs(self, app):
        """Test that main window has gas composition inputs."""
        from flare_calculator.ui.pyqt6.main_window import FlareCalculatorMainWindow

        window = FlareCalculatorMainWindow()
        assert "H2" in window.gas_inputs
        assert "CO" in window.gas_inputs
        assert "CH4" in window.gas_inputs


class TestFlarePhysics:
    """Test physical calculations."""

    def test_radiation_distance_formula(self):
        """Test radiation distance calculation."""
        # From point source model: I = (em * Q) / (4 * pi * r^2)
        # r = sqrt((em * Q) / (4 * pi * I))
        emissivity = 0.3
        heat_release = 10000  # kW
        radiation = 1.6  # kW/m²

        distance = math.sqrt(emissivity * heat_release / (4 * math.pi * radiation))
        assert distance > 0
        assert 10 < distance < 50  # Reasonable range for these values

    def test_gas_density_calculation(self):
        """Test ideal gas law density calculation."""
        R = 8.314  # J/(mol·K)
        pressure = 150000  # Pa (1.5 bar)
        temperature = 473  # K
        mw = 20  # g/mol (approx. syngas mixture)
        mw_kg = mw / 1000  # kg/mol

        density = pressure / ((R / mw_kg) * temperature)
        assert 0.5 < density < 2  # Reasonable range for syngas

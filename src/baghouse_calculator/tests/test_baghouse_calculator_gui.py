"""Tests for Baghouse Calculator GUI components."""

from __future__ import annotations

import pytest


class TestBaghouseCalculatorEngine:
    """Test the baghouse calculator engine integration."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        from upstream_drift_tools.process_calculators.baghouse_calculator import (
            BaghouseCalculator,
        )

        return BaghouseCalculator()

    def test_engine_instantiation(self, engine):
        """Test that engine can be instantiated."""
        assert engine is not None

    def test_calculate_with_typical_values(self, engine):
        """Test calculation with typical plant values."""
        result = engine.calculate(
            gas_flow_kg_s=10.0,
            inlet_temp_k=473.0,
            pressure_pa=101325,
            composition={"H2": 0.35, "CO": 0.30, "CO2": 0.15, "N2": 0.10, "H2O": 0.10},
            solid_carbon_in_kg_hr=50.0,
            ash_in_kg_hr=20.0,
            carbon_removal_efficiency=0.99,
            ash_removal_efficiency=0.99,
            heat_loss_w=5000,
            drum_volume_m3=0.5,
            solid_density_kg_m3=500,
            bag_area_ft2=1000,
        )

        assert result.carbon_removed_rate > 0
        assert result.ash_removed_rate > 0
        assert result.flow_acfm > 0
        assert result.air_to_cloth_ratio > 0

    def test_zero_solids_input(self, engine):
        """Test with zero solids input."""
        result = engine.calculate(
            gas_flow_kg_s=10.0,
            inlet_temp_k=473.0,
            pressure_pa=101325,
            composition={"N2": 1.0},
            solid_carbon_in_kg_hr=0.0,
            ash_in_kg_hr=0.0,
            carbon_removal_efficiency=0.99,
            ash_removal_efficiency=0.99,
            heat_loss_w=0,
            drum_volume_m3=0.5,
            solid_density_kg_m3=500,
            bag_area_ft2=1000,
        )

        assert result.carbon_removed_rate == 0
        assert result.ash_removed_rate == 0


class TestBaghouseCalculatorMainWindow:
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
        from baghouse_calculator.ui.pyqt6.main_window import BaghouseCalculatorMainWindow

        window = BaghouseCalculatorMainWindow()
        assert window is not None
        assert window.windowTitle() == "Baghouse Calculator"

    def test_main_window_has_engine(self, app):
        """Test that main window has calculation engine."""
        from baghouse_calculator.ui.pyqt6.main_window import BaghouseCalculatorMainWindow

        window = BaghouseCalculatorMainWindow()
        assert window.engine is not None

    def test_calculate_button_exists(self, app):
        """Test that calculate button exists."""
        from baghouse_calculator.ui.pyqt6.main_window import BaghouseCalculatorMainWindow

        window = BaghouseCalculatorMainWindow()
        assert hasattr(window, "calculate_btn")

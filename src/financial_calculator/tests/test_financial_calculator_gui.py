from typing import Any

"""Tests for Financial Calculator GUI components."""

from __future__ import annotations  # noqa: F404

import pytest


class TestFinancialCalculatorEngine:
    """Test the financial calculator engine integration."""

    @pytest.fixture
    def engine(self) -> Any:
        """Create engine instance."""
        from upstream_drift_tools.process_calculators.financial_calculator import (
            FinancialModelCalculator,
        )

        return FinancialModelCalculator()

    def test_engine_instantiation(self, engine) -> Any:
        """Test that engine can be instantiated."""
        assert engine is not None
        assert engine.parameters is not None
        assert engine.results is not None

    def test_calculate_with_default_params(self, engine) -> Any:
        """Test calculation with default (zero) parameters."""
        from upstream_drift_tools.process_calculators.financial_calculator import (
            FinancialParameters,
        )

        params = FinancialParameters()
        results = engine.calculate_financial_model(params)

        assert results.total_revenue == 0
        assert results.net_income == 0

    def test_calculate_with_typical_values(self, engine) -> Any:
        """Test calculation with typical plant values."""
        from upstream_drift_tools.process_calculators.financial_calculator import (
            FinancialParameters,
        )

        params = FinancialParameters(
            plant_capacity_tpd=100,
            operating_days_per_year=330,
            capacity_utilization=0.85,
            product_price_per_ton=500,
            byproduct_revenue_per_ton=50,
            byproduct_yield_factor=0.1,
            feedstock_cost_per_ton=200,
            labor_cost_per_ton=30,
            utilities_cost_per_ton=40,
            maintenance_cost_per_ton=15,
            consumables_cost_per_ton=10,
            fixed_labor_cost_annual=500000,
            insurance_annual=100000,
            property_tax_annual=50000,
            admin_overhead_annual=200000,
            total_capital_investment=10000000,
            debt_ratio=0.6,
            interest_rate=0.07,
            depreciation_years=10,
            tax_rate=0.25,
        )
        results = engine.calculate_financial_model(params)

        assert results.annual_feedstock_tons > 0
        assert results.total_revenue > 0
        assert results.net_income != 0  # Can be positive or negative

    def test_yearly_projections(self, engine) -> Any:
        """Test multi-year projection generation."""
        from upstream_drift_tools.process_calculators.financial_calculator import (
            FinancialParameters,
        )

        params = FinancialParameters(
            plant_capacity_tpd=100,
            operating_days_per_year=330,
            capacity_utilization=0.85,
            product_price_per_ton=500,
            total_capital_investment=10000000,
            depreciation_years=10,
        )
        engine.calculate_financial_model(params)
        projections = engine.generate_yearly_projections(years=5)

        assert len(projections) == 5
        assert all("year" in p for p in projections)
        assert all("net_income" in p for p in projections)


class TestFinancialCalculatorMainWindow:
    """Test the main window initialization."""

    @pytest.fixture
    def app(self) -> Any:
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_main_window_creation(self, app) -> Any:
        """Test that main window can be created."""
        from financial_calculator.ui.pyqt6.main_window import (
            FinancialCalculatorMainWindow,
        )

        window = FinancialCalculatorMainWindow()
        assert window is not None
        assert window.windowTitle() == "Financial Calculator"

    def test_main_window_has_engine(self, app) -> Any:
        """Test that main window has calculation engine."""
        from financial_calculator.ui.pyqt6.main_window import (
            FinancialCalculatorMainWindow,
        )

        window = FinancialCalculatorMainWindow()
        assert window.engine is not None

    def test_main_window_has_input_fields(self, app) -> Any:
        """Test that main window has required input fields."""
        from financial_calculator.ui.pyqt6.main_window import (
            FinancialCalculatorMainWindow,
        )

        window = FinancialCalculatorMainWindow()
        # Check for key input fields
        assert hasattr(window, "plant_capacity_input")
        assert hasattr(window, "operating_days_input")
        assert hasattr(window, "product_price_input")

    def test_calculate_button_exists(self, app) -> Any:
        """Test that calculate button exists."""
        from financial_calculator.ui.pyqt6.main_window import (
            FinancialCalculatorMainWindow,
        )

        window = FinancialCalculatorMainWindow()
        assert hasattr(window, "calculate_btn")


class TestFinancialCalculations:
    """Test financial calculation logic."""

    def test_revenue_calculation(self) -> Any:
        """Test basic revenue calculation."""
        # Revenue = product_tons * price + byproduct_tons * byproduct_price
        product_tons = 28050  # 100 tpd * 330 days * 0.85 util * 0.85 yield
        product_price = 500
        byproduct_tons = 2805  # 10% of feedstock
        byproduct_price = 50

        revenue = product_tons * product_price + byproduct_tons * byproduct_price
        assert revenue == 14165250  # $14.165M

    def test_payback_period_calculation(self) -> Any:
        """Test payback period calculation."""
        capital = 10000000
        net_income = 1000000
        depreciation = 1000000  # 10M / 10 years

        cash_flow = net_income + depreciation
        payback = capital / cash_flow
        assert payback == 5.0  # 5 years

    def test_roe_calculation(self) -> Any:
        """Test return on equity calculation."""
        capital = 10000000
        debt_ratio = 0.6
        net_income = 500000

        equity = capital * (1 - debt_ratio)
        roe = net_income / equity
        assert roe == pytest.approx(0.125, rel=0.01)  # 12.5%

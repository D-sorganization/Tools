"""Tests for Financial Calculator GUI components."""

from __future__ import annotations

import pytest


class TestFinancialCalculatorEngine:
    """Test the financial calculator engine integration."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        from upstream_drift_tools.process_calculators.financial_calculator import (
            FinancialModelCalculator,
        )

        return FinancialModelCalculator()

    def test_engine_instantiation(self, engine):
        """Test that engine can be instantiated."""
        assert engine is not None
        assert engine.parameters is not None
        assert engine.results is not None

    def test_calculate_with_default_params(self, engine):
        """Test calculation with default (zero) parameters."""
        from upstream_drift_tools.process_calculators.financial_calculator import (
            FinancialParameters,
        )

        params = FinancialParameters()
        results = engine.calculate_financial_model(params)

        assert results.total_revenue == 0
        assert results.net_income == 0

    def test_calculate_with_typical_values(self, engine):
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

    def test_yearly_projections(self, engine):
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
    def app(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_main_window_creation(self, app):
        """Test that main window can be created."""
        from financial_calculator.ui.pyqt6.main_window import (
            FinancialCalculatorMainWindow,
        )

        window = FinancialCalculatorMainWindow()
        assert window is not None
        assert window.windowTitle() == "Financial Calculator"

    def test_main_window_has_engine(self, app):
        """Test that main window has calculation engine."""
        from financial_calculator.ui.pyqt6.main_window import (
            FinancialCalculatorMainWindow,
        )

        window = FinancialCalculatorMainWindow()
        assert window.engine is not None

    def test_main_window_has_input_fields(self, app):
        """Test that main window has required input fields."""
        from financial_calculator.ui.pyqt6.main_window import (
            FinancialCalculatorMainWindow,
        )

        window = FinancialCalculatorMainWindow()
        # Check for key input fields
        assert hasattr(window, "plant_capacity_input")
        assert hasattr(window, "operating_days_input")
        assert hasattr(window, "product_price_input")

    def test_calculate_button_exists(self, app):
        """Test that calculate button exists."""
        from financial_calculator.ui.pyqt6.main_window import (
            FinancialCalculatorMainWindow,
        )

        window = FinancialCalculatorMainWindow()
        assert hasattr(window, "calculate_btn")


class TestFinancialCalculatorEngineDbC:
    """Test DbC preconditions on FinancialCalculatorEngine methods."""

    @pytest.fixture
    def engine_wrapper(self):
        """Create FinancialCalculatorEngine wrapper instance."""
        from financial_calculator.ui.pyqt6.main_window import FinancialCalculatorEngine

        return FinancialCalculatorEngine()

    def test_calculate_rejects_non_numeric_plant_capacity(self, engine_wrapper):
        """calculate() raises TypeError for non-numeric plant_capacity."""
        with pytest.raises(TypeError, match="plant_capacity must be a number"):
            engine_wrapper.calculate(
                plant_capacity="bad",
                operating_days=330,
                utilization=85.0,
                product_price=500.0,
                feedstock_cost=200.0,
                labor_cost=30.0,
                utilities_cost=40.0,
                maintenance_cost=15.0,
                fixed_labor=500000.0,
                insurance=100000.0,
                capital=10000000.0,
                debt_ratio=60.0,
                interest_rate=7.0,
                depreciation_years=10,
                tax_rate=25.0,
            )

    def test_calculate_rejects_negative_plant_capacity(self, engine_wrapper):
        """calculate() raises ValueError for negative plant_capacity."""
        with pytest.raises(ValueError, match="plant_capacity must be non-negative"):
            engine_wrapper.calculate(
                plant_capacity=-1.0,
                operating_days=330,
                utilization=85.0,
                product_price=500.0,
                feedstock_cost=200.0,
                labor_cost=30.0,
                utilities_cost=40.0,
                maintenance_cost=15.0,
                fixed_labor=500000.0,
                insurance=100000.0,
                capital=10000000.0,
                debt_ratio=60.0,
                interest_rate=7.0,
                depreciation_years=10,
                tax_rate=25.0,
            )

    def test_calculate_rejects_non_int_operating_days(self, engine_wrapper):
        """calculate() raises TypeError for non-integer operating_days."""
        with pytest.raises(TypeError, match="operating_days must be an int"):
            engine_wrapper.calculate(
                plant_capacity=100.0,
                operating_days=330.5,
                utilization=85.0,
                product_price=500.0,
                feedstock_cost=200.0,
                labor_cost=30.0,
                utilities_cost=40.0,
                maintenance_cost=15.0,
                fixed_labor=500000.0,
                insurance=100000.0,
                capital=10000000.0,
                debt_ratio=60.0,
                interest_rate=7.0,
                depreciation_years=10,
                tax_rate=25.0,
            )

    def test_calculate_rejects_out_of_range_operating_days(self, engine_wrapper):
        """calculate() raises ValueError for operating_days outside [0, 365]."""
        with pytest.raises(ValueError, match="operating_days must be in"):
            engine_wrapper.calculate(
                plant_capacity=100.0,
                operating_days=400,
                utilization=85.0,
                product_price=500.0,
                feedstock_cost=200.0,
                labor_cost=30.0,
                utilities_cost=40.0,
                maintenance_cost=15.0,
                fixed_labor=500000.0,
                insurance=100000.0,
                capital=10000000.0,
                debt_ratio=60.0,
                interest_rate=7.0,
                depreciation_years=10,
                tax_rate=25.0,
            )

    def test_calculate_rejects_non_positive_depreciation_years(self, engine_wrapper):
        """calculate() raises ValueError for non-positive depreciation_years."""
        with pytest.raises(ValueError, match="depreciation_years must be positive"):
            engine_wrapper.calculate(
                plant_capacity=100.0,
                operating_days=330,
                utilization=85.0,
                product_price=500.0,
                feedstock_cost=200.0,
                labor_cost=30.0,
                utilities_cost=40.0,
                maintenance_cost=15.0,
                fixed_labor=500000.0,
                insurance=100000.0,
                capital=10000000.0,
                debt_ratio=60.0,
                interest_rate=7.0,
                depreciation_years=0,
                tax_rate=25.0,
            )

    def test_generate_projections_rejects_non_int_years(self, engine_wrapper):
        """generate_projections() raises TypeError for non-integer years."""
        with pytest.raises(TypeError, match="years must be an int"):
            engine_wrapper.generate_projections(years=5.5)

    def test_generate_projections_rejects_zero_years(self, engine_wrapper):
        """generate_projections() raises ValueError for non-positive years."""
        with pytest.raises(ValueError, match="years must be positive"):
            engine_wrapper.generate_projections(years=0)


class TestFinancialCalculatorMainWindowDbC:
    """Test DbC preconditions on FinancialCalculatorMainWindow."""

    @pytest.fixture
    def app(self):
        """Create QApplication for tests."""
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_init_rejects_non_widget_parent(self, app):
        """__init__ raises TypeError if parent is not a QWidget or None."""
        from financial_calculator.ui.pyqt6.main_window import (
            FinancialCalculatorMainWindow,
        )

        with pytest.raises(TypeError, match="parent must be a QWidget or None"):
            FinancialCalculatorMainWindow(parent="not_a_widget")  # type: ignore[arg-type]

    def test_update_results_rejects_wrong_type(self, app):
        """_update_results raises TypeError if results is not FinancialDesign."""
        from financial_calculator.ui.pyqt6.main_window import (
            FinancialCalculatorMainWindow,
        )

        window = FinancialCalculatorMainWindow()
        with pytest.raises(TypeError, match="results must be a FinancialDesign"):
            window._update_results("not_a_design")  # type: ignore[arg-type]

    def test_update_projections_rejects_wrong_type(self, app):
        """_update_projections raises TypeError if projections is not a list."""
        from financial_calculator.ui.pyqt6.main_window import (
            FinancialCalculatorMainWindow,
        )

        window = FinancialCalculatorMainWindow()
        with pytest.raises(TypeError, match="projections must be a list"):
            window._update_projections("not_a_list")  # type: ignore[arg-type]


class TestFinancialCalculations:
    """Test financial calculation logic."""

    def test_revenue_calculation(self):
        """Test basic revenue calculation."""
        # Revenue = product_tons * price + byproduct_tons * byproduct_price
        product_tons = 28050  # 100 tpd * 330 days * 0.85 util * 0.85 yield
        product_price = 500
        byproduct_tons = 2805  # 10% of feedstock
        byproduct_price = 50

        revenue = product_tons * product_price + byproduct_tons * byproduct_price
        assert revenue == 14165250  # $14.165M

    def test_payback_period_calculation(self):
        """Test payback period calculation."""
        capital = 10000000
        net_income = 1000000
        depreciation = 1000000  # 10M / 10 years

        cash_flow = net_income + depreciation
        payback = capital / cash_flow
        assert payback == 5.0  # 5 years

    def test_roe_calculation(self):
        """Test return on equity calculation."""
        capital = 10000000
        debt_ratio = 0.6
        net_income = 500000

        equity = capital * (1 - debt_ratio)
        roe = net_income / equity
        assert roe == pytest.approx(0.125, rel=0.01)  # 12.5%

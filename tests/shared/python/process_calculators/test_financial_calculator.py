"""Comprehensive tests for financial_calculator.

Tests cover FinancialParameters, FinancialResults, FinancialModelCalculator,
volume/revenue calculations, operating costs, income statement, unit economics,
return metrics, and multi-year projections.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.financial_calculator import (
    FinancialModelCalculator,
    FinancialParameters,
    FinancialResults,
)

# ─── Helper ──────────────────────────────────────────────────


def _make_typical_params() -> FinancialParameters:
    """Create a realistic set of parameters for testing."""
    return FinancialParameters(
        plant_capacity_tpd=100.0,
        operating_days_per_year=300,
        capacity_utilization=0.90,
        product_price_per_ton=200.0,
        byproduct_revenue_per_ton=50.0,
        byproduct_yield_factor=0.10,
        feedstock_cost_per_ton=30.0,
        labor_cost_per_ton=10.0,
        utilities_cost_per_ton=15.0,
        maintenance_cost_per_ton=5.0,
        consumables_cost_per_ton=3.0,
        fixed_labor_cost_annual=500_000.0,
        insurance_annual=100_000.0,
        property_tax_annual=50_000.0,
        admin_overhead_annual=150_000.0,
        total_capital_investment=10_000_000.0,
        debt_ratio=0.60,
        interest_rate=0.05,
        depreciation_years=20,
        tax_rate=0.25,
    )


# ─── FinancialParameters Tests ───────────────────────────────


class TestFinancialParameters:
    """Test the FinancialParameters dataclass."""

    def test_default_construction(self) -> None:
        p = FinancialParameters()
        assert p.plant_capacity_tpd == 0.0
        assert p.operating_days_per_year == 0
        assert p.capacity_utilization == 0.0

    def test_custom_construction(self) -> None:
        p = FinancialParameters(plant_capacity_tpd=200.0, operating_days_per_year=365)
        assert p.plant_capacity_tpd == 200.0
        assert p.operating_days_per_year == 365

    def test_all_defaults_zero(self) -> None:
        p = FinancialParameters()
        for field_name in [
            "plant_capacity_tpd",
            "product_price_per_ton",
            "feedstock_cost_per_ton",
            "total_capital_investment",
        ]:
            assert getattr(p, field_name) == 0.0


# ─── FinancialResults Tests ──────────────────────────────────


class TestFinancialResults:
    """Test the FinancialResults dataclass."""

    def test_default_construction(self) -> None:
        r = FinancialResults()
        assert r.total_revenue == 0.0
        assert r.net_income == 0.0

    def test_all_defaults_zero(self) -> None:
        r = FinancialResults()
        for field_name in [
            "product_revenue",
            "ebitda",
            "roe",
            "roa",
            "payback_period_years",
        ]:
            assert getattr(r, field_name) == 0.0


# ─── Calculator Construction ─────────────────────────────────


class TestCalculatorConstruction:
    """Test FinancialModelCalculator instantiation."""

    def test_creates_default(self) -> None:
        calc = FinancialModelCalculator()
        assert calc.parameters is not None
        assert calc.results is not None
        assert calc.yearly_projections == []

    def test_initial_results_are_zeroed(self) -> None:
        calc = FinancialModelCalculator()
        assert calc.results.total_revenue == 0.0


# ─── Volume & Revenue Calculations ──────────────────────────


class TestVolumesAndRevenues:
    """Test volume and revenue calculations."""

    def test_annual_feedstock_tons(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        expected = 100.0 * 300 * 0.90  # 27,000 tons
        assert abs(results.annual_feedstock_tons - expected) < 0.01

    def test_annual_product_tons(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        expected = 27_000.0 * 0.85  # 22,950 tons
        assert abs(results.annual_product_tons - expected) < 0.01

    def test_product_revenue(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        expected_product = 27_000.0 * 0.85 * 200.0
        assert abs(results.product_revenue - expected_product) < 0.01

    def test_byproduct_revenue(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        expected_byproduct = 27_000.0 * 0.10 * 50.0
        assert abs(results.byproduct_revenue - expected_byproduct) < 0.01

    def test_total_revenue_is_sum(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        assert (
            abs(
                results.total_revenue
                - (results.product_revenue + results.byproduct_revenue)
            )
            < 0.01
        )

    def test_zero_params_yield_zero_revenue(self) -> None:
        calc = FinancialModelCalculator()
        params = FinancialParameters()
        results = calc.calculate_financial_model(params)
        assert results.total_revenue == 0.0


# ─── Operating Costs ─────────────────────────────────────────


class TestOperatingCosts:
    """Test operating cost calculations."""

    def test_feedstock_costs(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        expected = 27_000.0 * 30.0
        assert abs(results.feedstock_costs - expected) < 0.01

    def test_variable_costs_positive(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        assert results.total_variable_costs > 0.0

    def test_fixed_costs(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        expected_fixed = 500_000 + 100_000 + 50_000 + 150_000
        assert abs(results.total_fixed_costs - expected_fixed) < 0.01

    def test_fixed_costs_sum(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        manual_sum = (
            results.fixed_labor_costs
            + results.insurance_costs
            + results.property_tax_costs
            + results.admin_overhead_costs
        )
        assert abs(results.total_fixed_costs - manual_sum) < 0.01


# ─── Income Statement ────────────────────────────────────────


class TestIncomeStatement:
    """Test income statement calculations."""

    def test_gross_margin(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        expected = results.total_revenue - results.total_variable_costs
        assert abs(results.gross_margin - expected) < 0.01

    def test_ebitda(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        assert (
            abs(results.ebitda - (results.gross_margin - results.total_fixed_costs))
            < 0.01
        )

    def test_depreciation(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        expected = 10_000_000 / 20.0
        assert abs(results.depreciation - expected) < 0.01

    def test_depreciation_zero_years_safe(self) -> None:
        """Zero depreciation years should not divide by zero."""
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        params.depreciation_years = 0
        results = calc.calculate_financial_model(params)
        # max(0, 1) → divides by 1
        assert abs(results.depreciation - 10_000_000.0) < 0.01

    def test_interest_expense(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        expected = 10_000_000 * 0.60 * 0.05
        assert abs(results.interest_expense - expected) < 0.01

    def test_taxes_never_negative(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        params.product_price_per_ton = 0.0  # No revenue → loss
        results = calc.calculate_financial_model(params)
        assert results.taxes >= 0.0


# ─── Unit Economics & Return Metrics ──────────────────────────


class TestUnitAndReturnMetrics:
    """Test per-ton and return metrics."""

    def test_revenue_per_ton(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        assert results.revenue_per_ton > 0.0

    def test_zero_volume_zeroes_unit_metrics(self) -> None:
        calc = FinancialModelCalculator()
        params = FinancialParameters()
        results = calc.calculate_financial_model(params)
        assert results.revenue_per_ton == 0.0
        assert results.variable_cost_per_ton == 0.0
        assert results.margin_per_ton == 0.0

    def test_roe_positive_for_profitable_plant(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        assert results.roe > 0.0

    def test_roa_positive_for_profitable_plant(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        assert results.roa > 0.0

    def test_payback_period_positive(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        results = calc.calculate_financial_model(params)
        assert results.payback_period_years > 0.0

    def test_debt_ratio_one_raises(self) -> None:
        """Debt ratio of 1.0 means zero equity → ValueError."""
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        params.debt_ratio = 1.0
        with pytest.raises(ValueError, match="debt_ratio"):
            calc.calculate_financial_model(params)


# ─── Yearly Projections ──────────────────────────────────────


class TestYearlyProjections:
    """Test multi-year financial projections."""

    def test_default_10_years(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        calc.calculate_financial_model(params)
        projections = calc.generate_yearly_projections()
        assert len(projections) == 10

    def test_custom_years(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        calc.calculate_financial_model(params)
        projections = calc.generate_yearly_projections(years=5)
        assert len(projections) == 5

    def test_year_numbers_sequential(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        calc.calculate_financial_model(params)
        projections = calc.generate_yearly_projections(years=3)
        assert [p["year"] for p in projections] == [1, 2, 3]

    def test_revenue_escalates(self) -> None:
        """Revenue should grow year-over-year due to price escalation."""
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        calc.calculate_financial_model(params)
        projections = calc.generate_yearly_projections(years=3)
        assert projections[1]["total_revenue"] > projections[0]["total_revenue"]

    def test_cumulative_cash_flow(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        calc.calculate_financial_model(params)
        projections = calc.generate_yearly_projections(years=3)
        manual_cum = sum(p["cash_flow"] for p in projections)
        assert abs(projections[-1]["cumulative_cash_flow"] - manual_cum) < 0.01

    def test_stored_on_calculator(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        calc.calculate_financial_model(params)
        calc.generate_yearly_projections(years=5)
        assert len(calc.yearly_projections) == 5

    def test_each_projection_has_required_keys(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        calc.calculate_financial_model(params)
        projections = calc.generate_yearly_projections(years=1)
        required = [
            "year",
            "total_revenue",
            "total_costs",
            "ebitda",
            "net_income",
            "cash_flow",
            "cumulative_cash_flow",
        ]
        for key in required:
            assert key in projections[0], f"Missing key: {key}"


# ─── DbC Preconditions ───────────────────────────────────────


class TestContracts:
    """Test Design by Contract preconditions."""

    def test_negative_capital_raises(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        params.total_capital_investment = -1.0
        with pytest.raises((AssertionError, ValueError), match="Capital investment"):
            calc.calculate_financial_model(params)

    def test_negative_operating_days_raises(self) -> None:
        calc = FinancialModelCalculator()
        params = _make_typical_params()
        params.operating_days_per_year = -10
        with pytest.raises((AssertionError, ValueError), match="Operating days"):
            calc.calculate_financial_model(params)

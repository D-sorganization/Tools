"""Edge case and boundary value tests for FinancialModelCalculator.

Covers:
  - Debt ratio boundaries (0.0, 1.0, values in between)
  - Zero revenue / zero capacity scenarios
  - Negative cost handling
  - Break-even analysis
  - Depreciation edge cases (0 years)
  - Per-ton metric safety (division by zero protection)
  - Yearly projection generation

Design principles:
  - TDD: Tests describe the desired behaviour.
  - DRY: Common setup is shared via fixtures.
  - DbC: Each test documents pre/post-conditions.
  - Orthogonality: Each test class covers one category of edge cases.
"""

from __future__ import annotations

import math

import pytest
from upstream_drift_tools.process_calculators.financial_calculator import (
    FinancialModelCalculator,
    FinancialParameters,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator() -> FinancialModelCalculator:
    """Provide a fresh FinancialModelCalculator instance."""
    return FinancialModelCalculator()


@pytest.fixture
def baseline_params() -> FinancialParameters:
    """Provide reasonable baseline financial parameters for testing."""
    return FinancialParameters(
        plant_capacity_tpd=100.0,
        operating_days_per_year=330,
        capacity_utilization=0.90,
        product_price_per_ton=200.0,
        byproduct_revenue_per_ton=50.0,
        byproduct_yield_factor=0.10,
        feedstock_cost_per_ton=30.0,
        labor_cost_per_ton=10.0,
        utilities_cost_per_ton=15.0,
        maintenance_cost_per_ton=5.0,
        consumables_cost_per_ton=3.0,
        fixed_labor_cost_annual=500000.0,
        insurance_annual=100000.0,
        property_tax_annual=50000.0,
        admin_overhead_annual=200000.0,
        total_capital_investment=10000000.0,
        debt_ratio=0.60,
        interest_rate=0.05,
        depreciation_years=20,
        tax_rate=0.25,
    )


# ---------------------------------------------------------------------------
# Tests: Debt ratio boundaries
# ---------------------------------------------------------------------------


class TestDebtRatioBoundaries:
    """Edge cases for debt_ratio parameter."""

    def test_zero_debt_ratio_no_interest(self, calculator, baseline_params):
        """Zero debt ratio means 100% equity, zero interest expense."""
        baseline_params.debt_ratio = 0.0
        result = calculator.calculate_financial_model(baseline_params)
        assert result.interest_expense == pytest.approx(0.0)
        # ROE should be based on full capital as equity
        if result.net_income != 0:
            expected_roe = result.net_income / baseline_params.total_capital_investment
            assert result.roe == pytest.approx(expected_roe, rel=1e-6)

    def test_full_debt_ratio_zero_equity_raises(self, calculator, baseline_params):
        """Debt ratio = 1.0 with capital invested should raise ValueError.

        Zero equity would cause division by zero in ROE calculation, so the
        calculator validates this up front.
        """
        baseline_params.debt_ratio = 1.0
        with pytest.raises(ValueError, match="debt_ratio"):
            calculator.calculate_financial_model(baseline_params)

    def test_full_debt_ratio_zero_capital_is_safe(self, calculator, baseline_params):
        """Debt ratio = 1.0 with zero capital is fine (no equity needed)."""
        baseline_params.debt_ratio = 1.0
        baseline_params.total_capital_investment = 0.0
        result = calculator.calculate_financial_model(baseline_params)
        assert result.interest_expense == pytest.approx(0.0)
        assert result.roe == pytest.approx(0.0)

    def test_half_debt_ratio(self, calculator, baseline_params):
        """50% debt ratio should split capital evenly."""
        baseline_params.debt_ratio = 0.50
        result = calculator.calculate_financial_model(baseline_params)
        expected_interest = (
            baseline_params.total_capital_investment
            * 0.50
            * baseline_params.interest_rate
        )
        assert result.interest_expense == pytest.approx(expected_interest, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: Zero revenue / zero capacity scenarios
# ---------------------------------------------------------------------------


class TestZeroRevenueScenarios:
    """Edge cases where revenue is zero due to zero capacity or price."""

    def test_zero_capacity_all_zeros(self, calculator):
        """All-default (zero) parameters should produce all-zero results."""
        params = FinancialParameters()
        result = calculator.calculate_financial_model(params)
        assert result.total_revenue == pytest.approx(0.0)
        assert result.annual_feedstock_tons == pytest.approx(0.0)
        assert result.net_income == pytest.approx(0.0)
        assert result.revenue_per_ton == pytest.approx(0.0)
        assert result.total_cost_per_ton == pytest.approx(0.0)

    def test_zero_product_price(self, calculator, baseline_params):
        """Zero product price means product revenue is zero."""
        baseline_params.product_price_per_ton = 0.0
        baseline_params.byproduct_revenue_per_ton = 0.0
        result = calculator.calculate_financial_model(baseline_params)
        assert result.product_revenue == pytest.approx(0.0)
        assert result.total_revenue == pytest.approx(0.0)

    def test_zero_operating_days(self, calculator, baseline_params):
        """Zero operating days means zero production and zero revenue."""
        baseline_params.operating_days_per_year = 0
        result = calculator.calculate_financial_model(baseline_params)
        assert result.annual_feedstock_tons == pytest.approx(0.0)
        assert result.total_revenue == pytest.approx(0.0)
        # Per-ton metrics should be 0, not NaN/inf
        assert result.revenue_per_ton == pytest.approx(0.0)
        assert math.isfinite(result.revenue_per_ton)

    def test_zero_utilization(self, calculator, baseline_params):
        """Zero capacity utilization means zero production."""
        baseline_params.capacity_utilization = 0.0
        result = calculator.calculate_financial_model(baseline_params)
        assert result.annual_feedstock_tons == pytest.approx(0.0)
        assert result.total_revenue == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: Negative cost handling
# ---------------------------------------------------------------------------


class TestNegativeCosts:
    """Tests for negative cost values (credits or incentives)."""

    def test_negative_feedstock_cost_acts_as_credit(self, calculator, baseline_params):
        """Negative feedstock cost (tipping fee / credit) reduces total costs."""
        baseline_params.feedstock_cost_per_ton = -20.0
        result = calculator.calculate_financial_model(baseline_params)
        assert result.feedstock_costs < 0
        # Total variable costs should be lower than without the credit
        baseline_params_no_credit = FinancialParameters()
        baseline_params_no_credit.__dict__.update(baseline_params.__dict__)
        baseline_params_no_credit.feedstock_cost_per_ton = 0.0
        result_no_credit = calculator.calculate_financial_model(
            baseline_params_no_credit
        )
        assert result.total_variable_costs < result_no_credit.total_variable_costs

    def test_negative_cost_improves_net_income(self, calculator, baseline_params):
        """Negative costs (credits) should improve net income."""
        # First, get baseline net income
        result_baseline = calculator.calculate_financial_model(baseline_params)
        # Now add a feedstock credit
        baseline_params.feedstock_cost_per_ton = -50.0
        result_credit = calculator.calculate_financial_model(baseline_params)
        assert result_credit.net_income > result_baseline.net_income


# ---------------------------------------------------------------------------
# Tests: Break-even scenarios
# ---------------------------------------------------------------------------


class TestBreakEvenScenarios:
    """Tests for break-even and near-break-even conditions."""

    def test_net_income_sign_changes_with_price(self, calculator, baseline_params):
        """Increasing product price should eventually turn loss into profit.

        We test a low price (likely a loss) and a high price (likely profit)
        and verify the net income sign changes.
        """
        # Low price -> likely a loss
        baseline_params.product_price_per_ton = 1.0
        result_low = calculator.calculate_financial_model(baseline_params)

        # High price -> likely a profit
        baseline_params.product_price_per_ton = 1000.0
        result_high = calculator.calculate_financial_model(baseline_params)

        assert result_low.net_income < result_high.net_income
        # At $1/ton, should be a loss; at $1000/ton, should be a profit
        assert result_low.net_income < 0
        assert result_high.net_income > 0

    def test_taxes_zero_when_ebt_negative(self, calculator, baseline_params):
        """When EBT is negative (loss), taxes should be zero."""
        # Set very high costs to guarantee a loss
        baseline_params.feedstock_cost_per_ton = 500.0
        baseline_params.product_price_per_ton = 10.0
        result = calculator.calculate_financial_model(baseline_params)
        assert result.ebt < 0
        assert result.taxes == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: Depreciation edge cases
# ---------------------------------------------------------------------------


class TestDepreciationEdgeCases:
    """Edge cases for depreciation calculation."""

    def test_zero_depreciation_years_defaults_to_one(self, calculator, baseline_params):
        """Zero depreciation years uses max(years, 1) to avoid division by zero."""
        baseline_params.depreciation_years = 0
        result = calculator.calculate_financial_model(baseline_params)
        # max(0, 1) = 1, so depreciation = total_capital / 1
        assert result.depreciation == pytest.approx(
            baseline_params.total_capital_investment, rel=1e-6
        )

    def test_one_year_depreciation(self, calculator, baseline_params):
        """One-year depreciation means full capital is expensed in year 1."""
        baseline_params.depreciation_years = 1
        result = calculator.calculate_financial_model(baseline_params)
        assert result.depreciation == pytest.approx(
            baseline_params.total_capital_investment, rel=1e-6
        )

    def test_long_depreciation_period(self, calculator, baseline_params):
        """Long depreciation period (50 years) gives small annual depreciation."""
        baseline_params.depreciation_years = 50
        result = calculator.calculate_financial_model(baseline_params)
        expected = baseline_params.total_capital_investment / 50
        assert result.depreciation == pytest.approx(expected, rel=1e-6)

    def test_zero_capital_zero_depreciation(self, calculator, baseline_params):
        """Zero capital investment should produce zero depreciation."""
        baseline_params.total_capital_investment = 0.0
        result = calculator.calculate_financial_model(baseline_params)
        assert result.depreciation == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: Return metrics edge cases
# ---------------------------------------------------------------------------


class TestReturnMetricsEdgeCases:
    """Edge cases for ROE, ROA, and payback period calculations."""

    def test_zero_capital_zero_roa(self, calculator, baseline_params):
        """Zero capital investment should produce zero ROA (not inf)."""
        baseline_params.total_capital_investment = 0.0
        result = calculator.calculate_financial_model(baseline_params)
        assert result.roa == pytest.approx(0.0)

    def test_negative_net_income_zero_payback(self, calculator, baseline_params):
        """Negative net income should produce zero payback period."""
        baseline_params.product_price_per_ton = 0.0
        baseline_params.byproduct_revenue_per_ton = 0.0
        result = calculator.calculate_financial_model(baseline_params)
        assert result.net_income <= 0
        assert result.payback_period_years == pytest.approx(0.0)

    def test_high_profit_short_payback(self, calculator, baseline_params):
        """Very high profit should produce a short payback period."""
        baseline_params.product_price_per_ton = 5000.0
        result = calculator.calculate_financial_model(baseline_params)
        assert result.payback_period_years > 0
        assert result.payback_period_years < 10  # Should be well under 10 years

    def test_all_return_metrics_finite(self, calculator, baseline_params):
        """All return metrics should be finite for reasonable inputs."""
        result = calculator.calculate_financial_model(baseline_params)
        assert math.isfinite(result.roe)
        assert math.isfinite(result.roa)
        assert math.isfinite(result.payback_period_years)


# ---------------------------------------------------------------------------
# Tests: Per-ton metrics division safety
# ---------------------------------------------------------------------------


class TestPerTonMetricsSafety:
    """Ensure per-ton metrics don't produce NaN or inf values."""

    def test_zero_production_per_ton_metrics_zero(self, calculator):
        """Zero production should give zero per-ton metrics, not inf/NaN."""
        params = FinancialParameters(
            plant_capacity_tpd=0.0,
            operating_days_per_year=330,
            capacity_utilization=0.90,
        )
        result = calculator.calculate_financial_model(params)
        assert result.revenue_per_ton == pytest.approx(0.0)
        assert result.variable_cost_per_ton == pytest.approx(0.0)
        assert result.total_cost_per_ton == pytest.approx(0.0)
        assert result.margin_per_ton == pytest.approx(0.0)
        assert math.isfinite(result.revenue_per_ton)
        assert math.isfinite(result.variable_cost_per_ton)
        assert math.isfinite(result.total_cost_per_ton)
        assert math.isfinite(result.margin_per_ton)


# ---------------------------------------------------------------------------
# Tests: Yearly projections
# ---------------------------------------------------------------------------


class TestYearlyProjections:
    """Edge cases for generate_yearly_projections."""

    def test_zero_years_empty_list(self, calculator, baseline_params):
        """Zero years should produce an empty projection list."""
        calculator.calculate_financial_model(baseline_params)
        projections = calculator.generate_yearly_projections(years=0)
        assert projections == []

    def test_one_year_projection(self, calculator, baseline_params):
        """One year should produce exactly one projection entry."""
        calculator.calculate_financial_model(baseline_params)
        projections = calculator.generate_yearly_projections(years=1)
        assert len(projections) == 1
        assert projections[0]["year"] == 1
        assert "total_revenue" in projections[0]
        assert "net_income" in projections[0]

    def test_cumulative_cash_flow_monotonic(self, calculator, baseline_params):
        """For a profitable plant, cumulative cash flow should be non-decreasing."""
        baseline_params.product_price_per_ton = 500.0
        calculator.calculate_financial_model(baseline_params)
        projections = calculator.generate_yearly_projections(years=10)
        cumulative_values = [p["cumulative_cash_flow"] for p in projections]
        # Each year adds positive cash flow, so cumulative should increase
        for i in range(1, len(cumulative_values)):
            assert cumulative_values[i] >= cumulative_values[i - 1]

    def test_projection_revenue_grows_with_escalation(
        self, calculator, baseline_params
    ):
        """Revenue should grow year over year due to 2% price escalation."""
        calculator.calculate_financial_model(baseline_params)
        projections = calculator.generate_yearly_projections(years=5)
        revenues = [p["total_revenue"] for p in projections]
        for i in range(1, len(revenues)):
            assert revenues[i] > revenues[i - 1]

    def test_ten_year_projection_length(self, calculator, baseline_params):
        """Default 10-year projection should have 10 entries."""
        calculator.calculate_financial_model(baseline_params)
        projections = calculator.generate_yearly_projections()
        assert len(projections) == 10


# ---------------------------------------------------------------------------
# Tests: Financial model consistency
# ---------------------------------------------------------------------------


class TestFinancialModelConsistency:
    """Tests for internal consistency of the financial model."""

    def test_gross_margin_equals_revenue_minus_variable_costs(
        self, calculator, baseline_params
    ):
        """Gross margin must equal total revenue minus total variable costs."""
        result = calculator.calculate_financial_model(baseline_params)
        expected_gm = result.total_revenue - result.total_variable_costs
        assert result.gross_margin == pytest.approx(expected_gm, rel=1e-10)

    def test_ebitda_equals_gross_margin_minus_fixed_costs(
        self, calculator, baseline_params
    ):
        """EBITDA must equal gross margin minus fixed costs."""
        result = calculator.calculate_financial_model(baseline_params)
        expected_ebitda = result.gross_margin - result.total_fixed_costs
        assert result.ebitda == pytest.approx(expected_ebitda, rel=1e-10)

    def test_net_income_equals_ebt_minus_taxes(self, calculator, baseline_params):
        """Net income must equal EBT minus taxes."""
        result = calculator.calculate_financial_model(baseline_params)
        expected_ni = result.ebt - result.taxes
        assert result.net_income == pytest.approx(expected_ni, rel=1e-10)

    def test_total_fixed_costs_sum(self, calculator, baseline_params):
        """Total fixed costs must equal sum of individual fixed cost components."""
        result = calculator.calculate_financial_model(baseline_params)
        expected_fixed = (
            result.fixed_labor_costs
            + result.insurance_costs
            + result.property_tax_costs
            + result.admin_overhead_costs
        )
        assert result.total_fixed_costs == pytest.approx(expected_fixed, rel=1e-10)

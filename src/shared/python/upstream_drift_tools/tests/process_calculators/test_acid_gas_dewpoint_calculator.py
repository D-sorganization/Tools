import pytest
from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
    AcidGasComposition,
    AcidGasDewpointCalculator,
    DewpointResult,
    estimate_condensation_risk,
    quick_dewpoint_calculation,
)


@pytest.fixture
def dewpoint_calc() -> AcidGasDewpointCalculator:
    return AcidGasDewpointCalculator()


def test_calculate_vapor_pressure(dewpoint_calc: AcidGasDewpointCalculator) -> None:
    # Test valid components
    h2o_vp = dewpoint_calc.calculate_vapor_pressure(100.0, "H2O")
    assert h2o_vp > 0.0

    hf_vp = dewpoint_calc.calculate_vapor_pressure(25.0, "HF")
    assert hf_vp > 0.0

    # Extended antoine
    h2o_vp_high = dewpoint_calc.calculate_vapor_pressure(
        150.0, "H2O", "extended_antoine"
    )
    assert h2o_vp_high > 0.0

    # Invalid component
    with pytest.raises(ValueError, match="Unknown component"):
        dewpoint_calc.calculate_vapor_pressure(100.0, "InvalidGas")


def test_calculate_dewpoint(dewpoint_calc: AcidGasDewpointCalculator) -> None:
    # Test reverse calculation
    vp = dewpoint_calc.calculate_vapor_pressure(80.0, "H2O")
    dp = dewpoint_calc.calculate_dewpoint(vp, "H2O")
    assert dp == pytest.approx(80.0, abs=1e-3)

    with pytest.raises(ValueError):
        dewpoint_calc.calculate_dewpoint(0, "H2O")

    with pytest.raises(ValueError, match="unknown component"):
        dewpoint_calc.calculate_dewpoint(1000, "Invalid")


def test_calculate_dewpoint_mixture(dewpoint_calc: AcidGasDewpointCalculator) -> None:
    comp = AcidGasComposition(h2o=0.1, hf=0.01, hcl=0.02, h2s=0.05)
    result = dewpoint_calc.calculate_dewpoint_mixture(
        temperature_c=150.0,
        pressure_bar=10.0,
        composition=comp,
    )

    assert isinstance(result, DewpointResult)
    assert result.overall_dewpoint_c > 0.0
    assert result.limiting_component in ["H2O", "HF", "HCl", "H2S"]
    assert result.dewpoint_margin_c > 0.0
    assert result.condensation_risk != "Unknown"


def test_calculate_dewpoint_mixture_invalid(
    dewpoint_calc: AcidGasDewpointCalculator,
) -> None:
    comp = AcidGasComposition(h2o=0.1)
    with pytest.raises(ValueError, match="pressure_bar must be > 0"):
        dewpoint_calc.calculate_dewpoint_mixture(150.0, -1.0, comp)

    with pytest.raises(ValueError, match="temperature must yield a positive Kelvin"):
        dewpoint_calc.calculate_dewpoint_mixture(-300.0, 10.0, comp)


def test_quick_dewpoint_calculation() -> None:
    res = quick_dewpoint_calculation(
        temperature_c=150.0,
        pressure_bar=10.0,
        h2o_fraction=0.1,
    )
    assert res["overall_dewpoint_c"] > 0
    assert isinstance(res["condensation_risk"], str)


def test_estimate_condensation_risk() -> None:
    comp = AcidGasComposition(h2o=0.1)
    res = estimate_condensation_risk(150.0, 10.0, comp, safety_margin_c=10.0)
    assert "risk_level" in res
    assert "recommendation" in res
    assert res["limiting_component"] == "H2O"

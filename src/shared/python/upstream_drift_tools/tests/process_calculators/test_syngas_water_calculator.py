import numpy as np
import pytest
from upstream_drift_tools.process_calculators.syngas_water_calculator import (
    SyngasComposition,
    SyngasWaterCalculator,
    WaterContentResult,
    estimate_condensation_risk,
    quick_water_content,
)
from upstream_drift_tools.process_calculators.water_vapor_pressure_calculator import (
    WaterVaporPressureCalculator,
)


@pytest.fixture
def calc() -> SyngasWaterCalculator:
    return SyngasWaterCalculator()


def test_syngas_composition_normalization() -> None:
    comp = SyngasComposition(h2=1.0, co=1.0)
    norm = comp.normalize()
    assert norm.h2 == 0.5
    assert norm.co == 0.5
    assert norm.total == 1.0


def test_calculate_vapor_pressure_methods(calc: SyngasWaterCalculator) -> None:
    vp_antoine, m_antoine = calc.calculate_vapor_pressure(100.0, "antoine")
    assert vp_antoine > 0

    vp_buck, m_buck = calc.calculate_vapor_pressure(50.0, "buck")
    assert vp_buck > 0

    vp_magnus, m_magnus = calc.calculate_vapor_pressure(20.0, "magnus")
    assert vp_magnus > 0

    vp_iapws, m_iapws = calc.calculate_vapor_pressure(200.0, "iapws")
    assert vp_iapws > 0

    with pytest.raises(ValueError):
        calc.calculate_vapor_pressure(500.0, "iapws")


def test_water_vapor_pressure_wrapper() -> None:
    wrapper = WaterVaporPressureCalculator()
    vp = wrapper.calculate_vapor_pressure(100.0, "antoine")
    assert vp > 0


def test_vapor_pressure_fast(calc: SyngasWaterCalculator) -> None:
    vp = calc.vapor_pressure_fast(373.15)
    assert not np.isnan(vp)
    assert vp > 0


def test_calculate_dew_point(calc: SyngasWaterCalculator) -> None:
    vp, _ = calc.calculate_vapor_pressure(20.0, "buck")
    dp = calc.calculate_dew_point(vp, 101325.0)
    assert dp == pytest.approx(20.0, abs=0.1)


def test_calculate_water_content(calc: SyngasWaterCalculator) -> None:
    res = calc.calculate_water_content(50.0, 1.0)
    assert isinstance(res, WaterContentResult)
    assert res.mole_fraction_water > 0
    assert res.water_content_ppmv > 0
    assert res.dew_point_margin_c >= 0


def test_generate_water_content_curve(calc: SyngasWaterCalculator) -> None:
    df = calc.generate_water_content_curve(1.0, (20, 50), 3)
    assert len(df) == 3
    assert "water_mole_fraction" in df.columns


def test_quick_functions() -> None:
    res = quick_water_content(50.0, 1.0)
    assert "water_content_ppmv" in res
    assert res["mole_fraction"] > 0

    risk = estimate_condensation_risk(20.0, 1.0)
    assert "condensation_risk" in risk
    assert isinstance(risk["condensation_occurring"], bool)

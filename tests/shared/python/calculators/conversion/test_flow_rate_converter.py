"""Tests for the legacy flow-rate converter.

Restores real coverage for the conversion math behind issue #3101 (the suite
was a 0-byte stub). Covers the ``ton`` semantics (F2), the STP reconciliation
and DRY constants (F4/F5), and the ``Nm3/hr`` spelling (F8).
"""

from __future__ import annotations

import math

import pytest
from sidekick.calculators.conversion import flow_rate_converter as frc
from sidekick.utils import unit_constants as uc


@pytest.mark.unit
def test_ton_is_short_ton_not_metric() -> None:
    """``ton`` resolves to a short ton (907.18 kg), fleet-wide (F2)."""
    expected = uc.SHORT_TON_TO_KILOGRAM / 3600.0
    assert frc.MASS_FLOW_CONVERSIONS["ton/h"] == pytest.approx(expected)
    assert frc.MASS_FLOW_CONVERSIONS["ton/hr"] == pytest.approx(expected)
    # A metric ton must be spelled "tonne".
    assert frc.MASS_FLOW_CONVERSIONS["tonne/h"] == pytest.approx(1000.0 / 3600.0)
    # The two differ by ~9.3% — the original silent-error magnitude.
    ratio = frc.MASS_FLOW_CONVERSIONS["tonne/h"] / frc.MASS_FLOW_CONVERSIONS["ton/h"]
    assert ratio == pytest.approx(1000.0 / uc.SHORT_TON_TO_KILOGRAM)


@pytest.mark.unit
def test_one_ton_hr_to_kg_s_known_value() -> None:
    """1 short ton/hr == 0.2520 kg/s (hand-computed anchor, F2)."""
    result = frc.mass_to_mass(1.0, "ton/hr", "kg/s")
    assert result == pytest.approx(0.25199576, rel=1e-6)


@pytest.mark.unit
def test_constants_sourced_from_dry_layer() -> None:
    """No re-declared gas constant / STP definitions (F5/F4)."""
    assert frc.R_UNIVERSAL == uc.R_UNIVERSAL_KMOL
    t_stp, p_stp, _ = frc.STANDARD_CONDITIONS["STP"]
    assert t_stp == uc.STP_TEMPERATURE_K
    assert p_stp == uc.STP_PRESSURE_PA  # 100000 Pa, not 101325


@pytest.mark.unit
def test_stp_density_consistent_between_paths() -> None:
    """STP density agrees with the ideal-gas value at the DRY STP (F4)."""
    t_stp, p_stp, _ = frc.STANDARD_CONDITIONS["STP"]
    rho_expected = (p_stp * uc.MW_AIR) / (uc.R_UNIVERSAL_KMOL * t_stp)
    rho = frc._standard_density(p_stp, uc.MW_AIR, t_stp)
    assert rho == pytest.approx(rho_expected, rel=1e-12)


@pytest.mark.unit
def test_scfm_air_to_kg_s_known_value() -> None:
    """1000 SCFM air -> 0.5770 kg/s (hand-computed anchor, F1)."""
    result = frc.standard_volumetric_to_mass(1000, "ft3/min", uc.MW_AIR, "SCFM", "kg/s")
    assert result == pytest.approx(0.57701906, rel=1e-5)


@pytest.mark.unit
def test_nm3_hr_spelling_recognized() -> None:
    """The ``Nm3/hr`` / ``Nm³/hr`` spellings convert (F8)."""
    assert frc._is_standard_volumetric_unit("Nm3/hr")
    assert frc._is_standard_volumetric_unit("Nm³/hr")
    result = frc.convert_flow_rate_to_mass(100, "Nm3/hr", uc.MW_AIR)
    assert result > 0
    assert math.isfinite(result)


@pytest.mark.unit
@pytest.mark.parametrize("unit", ["kg/s", "kg/h", "lb/hr", "ton/hr"])
def test_mass_flow_round_trip(unit: str) -> None:
    """X -> kg/s -> X round-trips to the original value."""
    kg_s = frc.mass_to_mass(123.0, unit, "kg/s")
    back = frc.mass_to_mass(kg_s, "kg/s", unit)
    assert back == pytest.approx(123.0, rel=1e-9)


@pytest.mark.unit
def test_unknown_unit_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        frc.mass_to_mass(1.0, "bogus/hr", "kg/s")

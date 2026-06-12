"""Tests for the UnitConversionService (issue #3101 F7 + #3102 F9).

Was a 0-byte stub. Covers the normalize-raises contract on typo'd units and
the public ``get_compatible_units`` façade.
"""

from __future__ import annotations

import pytest
from sidekick.calculators.conversion.service import (
    IncompatibleUnitsError,
    UnitConversionService,
    UnknownUnitError,
)


@pytest.fixture
def service() -> UnitConversionService:
    return UnitConversionService()


@pytest.mark.unit
def test_normalize_unit_raises_on_typo(service: UnitConversionService) -> None:
    """Unresolved units raise instead of silently echoing the input (F7)."""
    with pytest.raises(UnknownUnitError):
        service._normalize_unit("kgg/sss")


@pytest.mark.unit
def test_normalize_unit_resolves_canonical(service: UnitConversionService) -> None:
    assert service._normalize_unit("kg/s") == "kg/s"
    assert service._normalize_unit(" K ") == "K"


@pytest.mark.unit
def test_convert_known_value(service: UnitConversionService) -> None:
    result = service.convert(1000.0, "kg/hr", "kg/s")
    assert result.value == pytest.approx(1000.0 / 3600.0)


@pytest.mark.scientific
@pytest.mark.parametrize(
    ("from_unit", "to_unit", "expected"),
    [
        ("BTU", "J", 1055.05585262),
        ("gal", "L", 3.785411784),
        ("psi", "Pa", 6894.757293168),
        ("ft", "m", 0.3048),
        ("lb", "kg", 0.45359237),
        ("in", "m", 0.0254),
    ],
)
def test_defined_factors_match_nist_sp_811(
    service: UnitConversionService,
    from_unit: str,
    to_unit: str,
    expected: float,
) -> None:
    """Defined unit factors stay pinned to NIST SP 811 reference values (#3391)."""
    result = service.convert(1.0, from_unit, to_unit)
    assert result.value == pytest.approx(expected, rel=1e-9)


@pytest.mark.scientific
def test_factor_table_round_trip_exactness(service: UnitConversionService) -> None:
    """Within-category factor conversions round-trip without hidden drift (#3391)."""
    checked = 0
    for category, factors in service.category_map.items():
        units = list(factors)
        if len(units) < 2:
            continue
        base = units[0]
        for other in units[1:]:
            try:
                forward = service.convert(1.0, base, other).value
                back = service.convert(forward, other, base).value
            except (IncompatibleUnitsError, TypeError, ValueError, UnknownUnitError):
                continue
            assert back == pytest.approx(1.0, rel=1e-9), (
                f"{category}: {base}->{other}->{base} lost precision"
            )
            checked += 1
    assert checked > 0


@pytest.mark.unit
def test_convert_unknown_unit_raises(service: UnitConversionService) -> None:
    with pytest.raises(UnknownUnitError):
        service.convert(1.0, "not_a_unit", "kg/s")


@pytest.mark.unit
def test_get_compatible_units_public_api(service: UnitConversionService) -> None:
    """Public façade returns same-category units (F9, no private reach-through)."""
    compatible = service.get_compatible_units("kg/s")
    assert "kg/s" in compatible
    assert "kg/hr" in compatible
    # A length unit must not appear among mass-flow units.
    assert "m" not in compatible


@pytest.mark.unit
def test_get_compatible_units_unknown_returns_empty(
    service: UnitConversionService,
) -> None:
    assert service.get_compatible_units("totally_bogus") == []
    assert service.get_compatible_units("") == []

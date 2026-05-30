"""Tests for the UnitConversionService (issue #3101 F7 + #3102 F9).

Was a 0-byte stub. Covers the normalize-raises contract on typo'd units and
the public ``get_compatible_units`` façade.
"""

from __future__ import annotations

import pytest
from sidekick.calculators.conversion.service import (
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

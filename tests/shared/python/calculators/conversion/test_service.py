"""Tests for the UnitConversionService (issue #3101 F7 + #3102 F9).

Was a 0-byte stub. Covers the normalize-raises contract on typo'd units and
the public ``get_compatible_units`` façade.
"""

from __future__ import annotations

import pytest
from sidekick.calculators.conversion.service import (
    IncompatibleUnitsError,
    InvalidValueError,
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


@pytest.mark.unit
def test_convert_rejects_invalid_unknown_and_incompatible_inputs(
    service: UnitConversionService,
) -> None:
    with pytest.raises(InvalidValueError):
        service.convert(float("inf"), "m", "cm")
    with pytest.raises(UnknownUnitError, match="not_a_unit"):
        service.convert(1.0, "not_a_unit", "m")
    with pytest.raises(UnknownUnitError, match="not_a_unit"):
        service.convert(1.0, "m", "not_a_unit")
    with pytest.raises(IncompatibleUnitsError):
        service.convert(1.0, "m", "kg")


@pytest.mark.unit
def test_validation_and_warning_helpers(service: UnitConversionService) -> None:
    pressure_result = service.convert(-1.0, "Pa", "bar")
    assert any("Negative pressure" in warning for warning in pressure_result.warnings)
    assert service._validate_value(-1000.0, "temperature", "C")
    assert service._validate_value(-1000.0, "temperature", "not-temperature") == []
    assert (
        UnitConversionService(enable_validation=False)._collect_conversion_warnings(
            -1.0, "pressure", "Pa"
        )
        == []
    )
    with pytest.raises(ValueError, match="value must be provided"):
        service._validate_value(None, "length")
    with pytest.raises(ValueError, match="value must be provided"):
        service._collect_conversion_warnings(None, "length", "m")


@pytest.mark.unit
def test_custom_units_aliases_and_user_warnings(
    service: UnitConversionService,
) -> None:
    service.add_unit("length", "league_test", "m", 4828.032, aliases=["lg_test"])

    assert service.convert(1.0, "lg_test", "m").value == pytest.approx(4828.032)
    assert service._normalize_unit("league_test") == "league_test"
    assert service._normalize_unit("lg_test") == "league_test"
    result = service.convert(1.0, "league_test", "m")
    assert any("user-defined" in warning for warning in result.warnings)

    with pytest.raises(ValueError, match="Unsupported category"):
        service.add_unit("unknown_category", "x", "m", 1.0)
    with pytest.raises(UnknownUnitError, match="Unknown reference unit"):
        service.add_unit("length", "x", "not_a_reference", 1.0)
    with pytest.raises(ValueError, match="already exists"):
        service.add_unit("length", "m", "m", 1.0)
    with pytest.raises(ValueError, match="positive"):
        service.add_unit("length", "negative_factor", "m", 0.0)


@pytest.mark.unit
def test_normalization_cache_and_static_guards(service: UnitConversionService) -> None:
    service._normalized_cache[" cm "] = "cm"
    assert service._normalize_unit(" cm ") == "cm"
    assert service._normalize_unit("meters") == "m"
    assert service._normalize_unit("c") == "C"

    with pytest.raises(ValueError, match="enable_validation"):
        UnitConversionService(enable_validation=None)
    with pytest.raises(ValueError, match="unit must be provided"):
        service._normalize_unit(None)
    with pytest.raises(ValueError, match="unit must be provided"):
        service._get_category(None)
    with pytest.raises(ValueError, match="positive and finite"):
        UnitConversionService._require_positive_finite(float("inf"), "x")
    with pytest.raises(ValueError, match="finite"):
        UnitConversionService._require_finite(float("nan"), "x")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("category", "expected_unit"),
    [
        ("length", "m"),
        ("temperature", "K"),
        ("gas_flow", "SCFM"),
        ("heating_value", "mj/kg"),
        ("tar_concentration", "mg/nm3"),
        ("performance", "%"),
    ],
)
def test_get_supported_units_category_filters(
    service: UnitConversionService, category: str, expected_unit: str
) -> None:
    units = service.get_supported_units(category)

    assert expected_unit in units[category]


@pytest.mark.unit
def test_get_supported_units_all_and_unknown(service: UnitConversionService) -> None:
    units = service.get_supported_units()

    assert "length" in units
    assert "temperature" in units
    assert "gas_flow" in units
    assert service.get_supported_units("missing_category") == {}


@pytest.mark.unit
def test_gas_flow_dispatch_paths(service: UnitConversionService) -> None:
    assert service._get_category("SCFM") == "gas_flow"
    assert service.convert(100.0, "SCFM", "SCFM").value == pytest.approx(100.0)
    assert service._convert_gas_flow(100.0, "SCFM", "Nm3/hr") > 0
    assert service._convert_gas_flow(1000.0, "Nm3/hr", "SCFM") > 0
    assert service._convert_gas_flow(100.0, "SCFM", "kg/s") > 0
    assert service._convert_gas_flow(0.1, "kg/s", "SCFM") > 0
    with pytest.raises(UnknownUnitError, match="Unknown gas flow unit"):
        service._convert_gas_flow(100.0, "strange_flow", "SCFM")
    with pytest.raises(UnknownUnitError, match="Unknown gas flow unit"):
        service._convert_gas_flow(100.0, "SCFM", "strange_flow")


@pytest.mark.unit
def test_syngas_and_performance_helpers(service: UnitConversionService) -> None:
    assert service.syngas_composition(1.0, "ppm", "ppb") == pytest.approx(1000.0)
    assert service.syngas_composition(1.0, "mol%", "vol%") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="not supported"):
        service.syngas_composition(1.0, "ppm", "kg/s")

    assert service.gasifier_performance(50.0, "%", "fraction") == pytest.approx(0.5)
    assert service.gasifier_performance(
        0.5, "fraction", "%", "carbon_conversion"
    ) == pytest.approx(50.0)
    assert service.gasifier_performance(
        1.0, "Nm3/kg", "scf/lb", "specific_production"
    ) == pytest.approx(1.0 / 0.0624)
    assert service.gasifier_performance(
        1.0, "scf/lb", "Nm³/kg", "specific_production"
    ) == pytest.approx(0.0624)
    with pytest.raises(ValueError, match="Unknown conversion"):
        service.gasifier_performance(1.0, "ppm", "fraction")
    with pytest.raises(ValueError, match="Unknown specific production"):
        service.gasifier_performance(1.0, "scf/lb", "unknown", "specific_production")
    with pytest.raises(ValueError, match="Unknown metric type"):
        service.gasifier_performance(1.0, "%", "fraction", "unknown_metric")


@pytest.mark.unit
def test_global_service_helpers() -> None:
    from sidekick.calculators.conversion import service as service_module

    service_module._ServiceHolder.instance = None
    first = service_module.get_service()
    second = service_module.get_service()

    assert first is second
    assert service_module.convert(1.0, "m", "cm") == pytest.approx(100.0)
    with pytest.raises(ValueError, match="value must be provided"):
        service_module.convert(None, "m", "cm")

#!/usr/bin/env python3
"""Tests for conversion service module."""

import pytest
from upstream_drift_tools.calculators.conversion.service import (
    IncompatibleUnitsError,
    InvalidValueError,
    UnitConversionService,
    UnknownUnitError,
    convert,
    get_service,
)
from upstream_drift_tools.calculators.conversion.tables import StandardCondition


@pytest.fixture
def service() -> UnitConversionService:
    return UnitConversionService()


def test_convert_length_returns_conversion_result(
    service: UnitConversionService,
) -> None:
    result = service.convert(1.0, "m", "ft")
    assert result.value == pytest.approx(3.28084, rel=1e-5)
    assert result.from_unit == "m"
    assert result.to_unit == "ft"
    assert result.warnings == []


def test_convert_unknown_unit_raises(service: UnitConversionService) -> None:
    with pytest.raises(UnknownUnitError, match="Unknown unit"):
        service.convert(1.0, "not-a-unit", "m")


def test_convert_non_finite_value_raises(service: UnitConversionService) -> None:
    with pytest.raises(InvalidValueError, match="finite"):
        service.convert(float("nan"), "m", "ft")


def test_convert_incompatible_units_raises(service: UnitConversionService) -> None:
    with pytest.raises(IncompatibleUnitsError, match="Cannot convert"):
        service.convert(1.0, "m", "kg")


def test_temperature_validation_warning_below_absolute_zero(
    service: UnitConversionService,
) -> None:
    result = service.convert(-500.0, "C", "K")
    assert "Temperature below absolute zero" in result.warnings


def test_pressure_validation_warning_negative(service: UnitConversionService) -> None:
    result = service.convert(-1.0, "Pa", "kPa")
    assert "Negative pressure is invalid" in result.warnings


def test_normalize_unit_handles_special_characters(
    service: UnitConversionService,
) -> None:
    assert service._normalize_unit("  °C  ") == "C"
    assert service._normalize_unit(" k_g ") == "kg"


def test_add_unit_supports_aliases_and_conversion(
    service: UnitConversionService,
) -> None:
    service.add_unit("length", "smoot", "m", 1.7018, aliases=["sm", "SMOOT"])

    result_alias = service.convert(1.0, "sm", "m")
    result_canonical = service.convert(2.0, "smoot", "m")

    assert result_alias.value == pytest.approx(1.7018)
    assert result_canonical.value == pytest.approx(3.4036)
    assert any("user-defined" in warning for warning in result_alias.warnings)


def test_add_unit_rejects_invalid_inputs(service: UnitConversionService) -> None:
    with pytest.raises(ValueError, match="Unsupported category"):
        service.add_unit("invalid_category", "x", "m", 1.0)

    with pytest.raises(UnknownUnitError, match="Unknown reference unit"):
        service.add_unit("length", "x", "unknown_ref", 1.0)

    with pytest.raises(ValueError, match="positive"):
        service.add_unit("length", "x", "m", 0.0)


def test_convert_gas_flow_requires_temperature_pressure_for_acfm(
    service: UnitConversionService,
) -> None:
    with pytest.raises(ValueError, match="required for ACFM"):
        service._convert_gas_flow(10.0, "ACFM", "SCFM")


def test_convert_gas_flow_scfm_to_nm3(service: UnitConversionService) -> None:
    result = service._convert_gas_flow(
        100.0,
        "SCFM",
        "Nm3/hr",
        standard_condition=StandardCondition.SCFM_60F,
    )
    assert result > 0


def test_convert_gas_flow_mass_rate_to_acfm(service: UnitConversionService) -> None:
    result = service._convert_gas_flow(
        1.0,
        "kg/s",
        "ACFM",
        temperature=320.0,
        pressure=150000.0,
        gas_type="co2",
    )
    assert result > 0


def test_convert_gas_flow_scfm_acfm_applies_compressibility(
    service: UnitConversionService,
) -> None:
    no_z = service.convert_gas_flow_scfm_acfm(
        1000.0,
        "SCFM",
        "ACFM",
        actual_temp_K=320.0,
        actual_pressure_kPa=101.325,
        compressibility_factor=1.0,
    )
    with_z = service.convert_gas_flow_scfm_acfm(
        1000.0,
        "SCFM",
        "ACFM",
        actual_temp_K=320.0,
        actual_pressure_kPa=101.325,
        compressibility_factor=0.9,
    )
    assert with_z == pytest.approx(no_z * 0.9)


def test_convert_gas_flow_acfm_to_scfm_nonpositive_z_no_division(
    service: UnitConversionService,
) -> None:
    baseline = service.convert_gas_flow_scfm_acfm(
        1000.0,
        "ACFM",
        "SCFM",
        actual_temp_K=320.0,
        actual_pressure_kPa=101.325,
        compressibility_factor=1.0,
    )
    no_divide = service.convert_gas_flow_scfm_acfm(
        1000.0,
        "ACFM",
        "SCFM",
        actual_temp_K=320.0,
        actual_pressure_kPa=101.325,
        compressibility_factor=0.0,
    )
    assert no_divide == pytest.approx(baseline)


def test_heating_value_requires_density_for_volumetric(
    service: UnitConversionService,
) -> None:
    with pytest.raises(ValueError, match="Gas density required"):
        service.heating_value(10.0, "mj/nm3", "mj/kg")


def test_heating_value_volumetric_roundtrip(service: UnitConversionService) -> None:
    mj_per_kg = service.heating_value(20.0, "mj/nm3", "mj/kg", gas_density_stp=0.8)
    mj_per_nm3 = service.heating_value(
        mj_per_kg, "mj/kg", "mj/nm3", gas_density_stp=0.8
    )
    assert mj_per_nm3 == pytest.approx(20.0)


def test_heating_value_nonpositive_density_raises(
    service: UnitConversionService,
) -> None:
    with pytest.raises(ValueError, match="Gas density must be positive"):
        service.heating_value(20.0, "mj/nm3", "mj/kg", gas_density_stp=0.0)


def test_tar_concentration_ppm_requires_molecular_weight(
    service: UnitConversionService,
) -> None:
    with pytest.raises(ValueError, match="Molecular weight required"):
        service.tar_concentration(100.0, "ppm_mass", "mg/nm3")


def test_tar_concentration_nonpositive_molecular_weight_raises(
    service: UnitConversionService,
) -> None:
    with pytest.raises(ValueError, match="Molecular weight must be positive"):
        service.tar_concentration(100.0, "ppm_mass", "mg/nm3", molecular_weight=0.0)


def test_tar_concentration_pressure_temperature_adjustment(
    service: UnitConversionService,
) -> None:
    result = service.tar_concentration(
        100.0,
        "mg/m3",
        "mg/nm3",
        temperature=320.0,
        pressure=95.0,
    )
    assert result > 0


def test_tar_concentration_nonpositive_pressure_raises(
    service: UnitConversionService,
) -> None:
    with pytest.raises(ValueError, match="pressure must be positive"):
        service.tar_concentration(100.0, "mg/m3", "mg/nm3", pressure=0.0)


def test_syngas_composition_conversions(service: UnitConversionService) -> None:
    assert service.syngas_composition(1.0, "ppm", "ppb") == pytest.approx(1000.0)
    assert service.syngas_composition(5.0, "mol%", "vol%") == pytest.approx(5.0)
    with pytest.raises(ValueError, match="not supported"):
        service.syngas_composition(1.0, "foo", "bar")


def test_gasifier_performance_metric_types(service: UnitConversionService) -> None:
    assert service.gasifier_performance(50.0, "%", "fraction") == pytest.approx(0.5)
    assert service.gasifier_performance(
        1.0, "nm3/kg", "scf/lb", metric_type="specific_production"
    ) == pytest.approx(1.0 / 0.0624)
    with pytest.raises(ValueError, match="Unknown metric type"):
        service.gasifier_performance(1.0, "%", "fraction", metric_type="unknown")


def test_compressibility_factor_regime(service: UnitConversionService) -> None:
    z_good = service.compressibility_factor("air", temperature=400.0, pressure=5e5)
    z_fallback = service.compressibility_factor("air", temperature=100.0, pressure=2e7)
    assert z_good > 0.1
    assert z_fallback == pytest.approx(0.1)


def test_compressibility_factor_nonpositive_temperature_raises(
    service: UnitConversionService,
) -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        service.compressibility_factor("air", temperature=0.0, pressure=101325.0)


def test_compressibility_factor_nonpositive_pressure_raises(
    service: UnitConversionService,
) -> None:
    with pytest.raises(ValueError, match="pressure must be positive"):
        service.compressibility_factor("air", temperature=300.0, pressure=0.0)


def test_get_supported_units_category_and_all(service: UnitConversionService) -> None:
    temperature_only = service.get_supported_units("temperature")
    assert temperature_only["temperature"] == ["K", "C", "F", "R"]

    all_units = service.get_supported_units()
    assert "length" in all_units
    assert "gas_flow" in all_units
    assert "performance" in all_units


def test_get_service_and_convert_global_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import upstream_drift_tools.calculators.conversion.service as service_module

    service_module._global_service = None
    first = get_service()
    second = get_service()
    assert first is second

    monkeypatch.setattr(service_module, "_global_service", UnitConversionService())
    value = convert(1.0, "m", "cm")
    assert value == pytest.approx(100.0)

"""Extended tests for the UnitConversionService and flow rate converter.

Covers:
- Edge cases and validation for the conversion service
- Flow rate conversion consistency between modules
- Error paths and exception typing
- DbC precondition enforcement

Addresses #827 (test depth), #826 (DbC coverage), #830 (typed errors).
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.calculators.conversion.flow_rate_converter import (
    MASS_FLOW_CONVERSIONS,
    MOLAR_FLOW_CONVERSIONS,
    VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
    acfm_to_scfm,
    mass_to_mass,
    mass_to_molar,
    molar_to_mass,
    molar_to_molar,
    scfm_to_acfm,
    volumetric_actual_to_mass,
)
from upstream_drift_tools.calculators.conversion.service import (
    ConversionResult,
    IncompatibleUnitsError,
    UnitConversionError,
    UnitConversionService,
    UnknownUnitError,
    convert,
    get_service,
)

# ── Service-level tests ──────────────────────────────────────────


class TestConversionServiceBasic:
    """Basic conversion service tests."""

    @pytest.mark.parametrize(
        "value, from_unit, to_unit, expected, tolerance",
        [
            (1.0, "kg", "lb", 2.20462, {"rel": 1e-4}),
            (1.0, "m3", "L", 1000.0, {"rel": 1e-4}),
            (1000.0, "J", "kJ", 1.0, {"rel": 1e-4}),
            (42.0, "kg", "kg", 42.0, {"rel": 1e-10}),
            (273.15, "K", "C", 0.0, {"abs": 0.01}),
            (491.67, "R", "K", 273.15, {"rel": 1e-4}),
        ],
        ids=["kg-to-lb", "m3-to-L", "J-to-kJ", "identity", "K-to-C", "R-to-K"],
    )
    def test_unit_conversion(self, value, from_unit, to_unit, expected, tolerance) -> None:
        """Test unit conversion accuracy across categories."""
        val = convert(value, from_unit, to_unit)
        assert val == pytest.approx(expected, **tolerance)


class TestConversionServiceErrors:
    """Error path tests for the conversion service."""

    def test_unknown_from_unit_raises(self) -> None:
        with pytest.raises(UnknownUnitError):
            convert(1.0, "nonsense_unit", "kg")

    def test_unknown_to_unit_raises(self) -> None:
        with pytest.raises(UnknownUnitError):
            convert(1.0, "kg", "nonsense_unit")

    def test_incompatible_units_raises(self) -> None:
        with pytest.raises((IncompatibleUnitsError, UnknownUnitError)):
            convert(1.0, "kg", "m")

    def test_exception_hierarchy(self) -> None:
        """All conversion errors must inherit from UnitConversionError."""
        assert issubclass(UnknownUnitError, UnitConversionError)
        assert issubclass(IncompatibleUnitsError, UnitConversionError)


class TestConversionResult:
    """Tests for the ConversionResult dataclass."""

    def test_result_has_value(self) -> None:
        svc = get_service()
        result = svc.convert(100.0, "cm", "m")
        assert isinstance(result, ConversionResult)
        assert result.value == pytest.approx(1.0, rel=1e-4)

    def test_result_preserves_unit_names(self) -> None:
        svc = get_service()
        result = svc.convert(1.0, "kg", "lb")
        assert result.from_unit == "kg"
        assert result.to_unit == "lb"


class TestUserDefinedUnits:
    """Tests for user-defined unit registration."""

    def test_add_custom_unit(self) -> None:
        svc = UnitConversionService()
        svc.add_unit("length", "my_foot", "m", 0.3048)
        result = svc.convert(1.0, "my_foot", "m")
        assert result.value == pytest.approx(0.3048, rel=1e-4)

    def test_add_duplicate_unit_raises(self) -> None:
        svc = UnitConversionService()
        with pytest.raises(ValueError, match="already exists"):
            svc.add_unit("length", "m", "m", 1.0)

    def test_add_unit_invalid_category_raises(self) -> None:
        svc = UnitConversionService()
        with pytest.raises(ValueError, match="Unsupported category"):
            svc.add_unit("antimatter", "qubit", "m", 1.0)

    def test_add_unit_negative_factor_raises(self) -> None:
        svc = UnitConversionService()
        with pytest.raises(ValueError, match="positive"):
            svc.add_unit("length", "neg_unit", "m", -1.0)


# ── Flow rate converter tests ────────────────────────────────────


class TestFlowRateMassConversion:
    """Tests for mass flow rate conversions."""

    def test_kg_h_to_lb_h(self) -> None:
        result = mass_to_mass(1000.0, "kg/h", "lb/hr")
        assert result == pytest.approx(2204.62, rel=1e-3)

    def test_round_trip_mass(self) -> None:
        """Converting kg/s -> lb/h -> kg/s must be lossless."""
        original = 5.0
        intermediate = mass_to_mass(original, "kg/s", "lb/hr")
        back = mass_to_mass(intermediate, "lb/hr", "kg/s")
        assert back == pytest.approx(original, rel=1e-10)

    def test_unknown_from_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            mass_to_mass(1.0, "bananas/s", "kg/s")

    def test_unknown_to_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            mass_to_mass(1.0, "kg/s", "bananas/hr")


class TestFlowRateMolarConversion:
    """Tests for molar flow rate conversions."""

    def test_kmol_h_to_lbmol_h(self) -> None:
        result = molar_to_molar(1.0, "kmol/h", "lbmol/h")
        assert result == pytest.approx(1000.0 / 453.592, rel=1e-3)

    def test_round_trip_molar(self) -> None:
        original = 10.0
        intermediate = molar_to_molar(original, "mol/s", "kmol/h")
        back = molar_to_molar(intermediate, "kmol/h", "mol/s")
        assert back == pytest.approx(original, rel=1e-10)


class TestFlowRateCrossConversion:
    """Tests for mass <-> molar conversions."""

    def test_mass_to_molar_air(self) -> None:
        """100 kg/h of air (MW=29) should be ~3.45 kmol/h."""
        result = mass_to_molar(100.0, "kg/h", 29.0, "kmol/h")
        assert result == pytest.approx(100.0 / 29.0, rel=1e-3)

    def test_molar_to_mass_co2(self) -> None:
        """10 kmol/h of CO2 (MW=44) should be 440 kg/h."""
        result = molar_to_mass(10.0, "kmol/h", 44.0, "kg/h")
        assert result == pytest.approx(440.0, rel=1e-3)

    def test_round_trip_mass_molar(self) -> None:
        mw = 29.0
        molar = mass_to_molar(100.0, "kg/h", mw, "kmol/h")
        back = molar_to_mass(molar, "kmol/h", mw, "kg/h")
        assert back == pytest.approx(100.0, rel=1e-6)


class TestFlowRateVolumetric:
    """Tests for volumetric flow conversions."""

    def test_volumetric_to_mass(self) -> None:
        result = volumetric_actual_to_mass(1000.0, "m3/h", 1.2, "kg/h")
        assert result == pytest.approx(1200.0, rel=1e-3)

    def test_unknown_volumetric_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            volumetric_actual_to_mass(1.0, "bananas/s", 1.0, "kg/s")


class TestSCFMtoACFM:
    """Tests for SCFM/ACFM conversion."""

    def test_identity_at_standard_conditions(self) -> None:
        """At standard conditions, SCFM and ACFM should be equal."""
        # SCFM standard: 60F = 288.71K, 14.696 psia = 101325 Pa
        T_std = 288.71
        P_std = 101325
        result = scfm_to_acfm(1000.0, T_std, P_std, "SCFM")
        assert result == pytest.approx(1000.0, rel=1e-3)

    def test_round_trip_scfm_acfm(self) -> None:
        T, P = 500.0, 3e5
        acfm = scfm_to_acfm(1000.0, T, P, "SCFM")
        back = acfm_to_scfm(acfm, T, P, "SCFM")
        assert back == pytest.approx(1000.0, rel=1e-6)


class TestConversionTableConsistency:
    """Verify that conversion tables are self-consistent."""

    @pytest.mark.parametrize(
        "table, base_unit",
        [
            (MASS_FLOW_CONVERSIONS, "kg/s"),
            (MOLAR_FLOW_CONVERSIONS, "mol/s"),
            (VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S, "m3/s"),
        ],
        ids=["mass", "molar", "volumetric"],
    )
    def test_conversion_factors_positive(self, table, base_unit) -> None:
        """All flow conversion factors must be positive."""
        for unit, factor in table.items():
            assert factor > 0, f"Factor for {unit} is not positive: {factor}"

    @pytest.mark.parametrize(
        "table, base_unit",
        [
            (MASS_FLOW_CONVERSIONS, "kg/s"),
            (MOLAR_FLOW_CONVERSIONS, "mol/s"),
            (VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S, "m3/s"),
        ],
        ids=["mass", "molar", "volumetric"],
    )
    def test_base_unit_is_unity(self, table, base_unit) -> None:
        """The base SI unit in each table must have factor 1.0."""
        assert table[base_unit] == 1.0


# ── DbC precondition tests ─────────────────────────────────────


class TestFlowRatePreconditions:
    """DbC: verify that invalid inputs are rejected at function entry."""

    @pytest.mark.parametrize(
        "bad_value",
        [float("nan"), float("inf")],
        ids=["nan", "inf"],
    )
    def test_mass_to_mass_non_finite_raises(self, bad_value: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            mass_to_mass(bad_value, "kg/s", "lb/hr")

    def test_molar_to_molar_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            molar_to_molar(float("nan"), "mol/s", "kmol/h")

    @pytest.mark.parametrize(
        "bad_mw",
        [0.0, -29.0, float("nan")],
        ids=["zero", "negative", "nan"],
    )
    def test_mass_to_molar_bad_mw_raises(self, bad_mw: float) -> None:
        with pytest.raises(ValueError, match="positive"):
            mass_to_molar(100.0, "kg/h", bad_mw, "kmol/h")

    def test_molar_to_mass_zero_mw_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            molar_to_mass(10.0, "kmol/h", 0.0, "kg/h")

    @pytest.mark.parametrize(
        "bad_density",
        [0.0, -1.2],
        ids=["zero", "negative"],
    )
    def test_volumetric_to_mass_bad_density_raises(self, bad_density: float) -> None:
        with pytest.raises(ValueError, match="positive"):
            volumetric_actual_to_mass(1000.0, "m3/h", bad_density, "kg/h")

    @pytest.mark.parametrize(
        "temperature, pressure, match_str",
        [
            (0.0, 101325.0, "positive"),
            (300.0, 0.0, "positive"),
        ],
        ids=["zero-temp", "zero-pressure"],
    )
    def test_scfm_to_acfm_bad_conditions_raises(
        self, temperature: float, pressure: float, match_str: str
    ) -> None:
        with pytest.raises(ValueError, match=match_str):
            scfm_to_acfm(1000.0, temperature, pressure, "SCFM")

    def test_acfm_to_scfm_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            acfm_to_scfm(1000.0, -100.0, 101325.0, "SCFM")

    def test_acfm_to_scfm_inf_pressure_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            acfm_to_scfm(1000.0, 300.0, float("inf"), "SCFM")

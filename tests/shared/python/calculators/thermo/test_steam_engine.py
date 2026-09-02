#!/usr/bin/env python3
"""Targeted tests for steam engine fallback and selection logic."""

from __future__ import annotations

import math
from dataclasses import asdict

import pytest
import upstream_drift_tools.calculators.thermo.steam_engine as steam_engine
from upstream_drift_tools.calculators.thermo.steam_engine import (
    DEFAULT_DEW_POINT_TEMPERATURE_CELSIUS,
    SteamCalculationEngine,
    SteamProperties,
)


def _sentinel_props(temperature: float, pressure: float, phase: str) -> SteamProperties:
    return SteamProperties(
        temperature=temperature,
        pressure=pressure,
        density=1.0,
        specific_volume=1.0,
        enthalpy=1.0,
        entropy=1.0,
        internal_energy=1.0,
        cp=2.0,
        cv=1.0,
        speed_of_sound=1.0,
        thermal_conductivity=1.0,
        dynamic_viscosity=1.0,
        kinematic_viscosity=1.0,
        quality=0.5,
        phase=phase,
        compressibility_factor=1.0,
        prandtl_number=2.0,
        specific_heat_ratio=2.0,
    )


def test_assert_finite_accepts_healthy_props() -> None:
    steam_engine._assert_finite(_sentinel_props(400.0, 101325.0, "vapor"))


def test_assert_finite_tolerates_nan_quality() -> None:
    """quality=NaN is the documented value when quality isn't applicable."""
    props = _sentinel_props(400.0, 101325.0, "vapor")
    props.quality = math.nan
    steam_engine._assert_finite(props)


@pytest.mark.parametrize(
    "field_name",
    [
        "temperature",
        "pressure",
        "density",
        "specific_volume",
        "enthalpy",
        "entropy",
        "internal_energy",
        "cp",
        "cv",
        "speed_of_sound",
        "thermal_conductivity",
        "dynamic_viscosity",
        "kinematic_viscosity",
    ],
)
def test_assert_finite_rejects_nan_in_each_required_field(field_name: str) -> None:
    """Regression test for issue #3981: only enthalpy was checked before."""
    props = _sentinel_props(400.0, 101325.0, "vapor")
    setattr(props, field_name, math.nan)
    with pytest.raises(ValueError, match=f"{field_name} must be finite"):
        steam_engine._assert_finite(props)


def test_calculate_saturated_from_temperature_rejects_non_finite_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for issue #3981: this method had no postcondition."""
    engine = SteamCalculationEngine()
    monkeypatch.setattr(steam_engine, "COOLPROP_AVAILABLE", False)
    monkeypatch.setattr(steam_engine, "CANTERA_AVAILABLE", False)
    engine.water = None
    bad_props = _sentinel_props(400.0, 101325.0, "simplified")
    bad_props.enthalpy = math.nan
    monkeypatch.setattr(
        engine, "_calculate_saturated_simplified_from_temp", lambda *_: bad_props
    )
    with pytest.raises(ValueError, match="enthalpy must be finite"):
        engine.calculate_saturated_properties_from_temperature(400.0)


def test_calculate_saturated_from_pressure_rejects_non_finite_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for issue #3981: this method had no postcondition."""
    engine = SteamCalculationEngine()
    monkeypatch.setattr(steam_engine, "COOLPROP_AVAILABLE", False)
    monkeypatch.setattr(steam_engine, "CANTERA_AVAILABLE", False)
    engine.water = None
    bad_props = _sentinel_props(400.0, 101325.0, "simplified")
    bad_props.entropy = math.inf
    monkeypatch.setattr(
        engine, "_calculate_saturated_simplified_from_pressure", lambda *_: bad_props
    )
    with pytest.raises(ValueError, match="entropy must be finite"):
        engine.calculate_saturated_properties_from_pressure(101325.0)


def test_steam_properties_to_dict_contains_advanced_fields() -> None:
    props = _sentinel_props(400.0, 101325.0, "vapor")
    data = props.to_dict()
    assert data["Temperature (K)"] == 400.0
    assert data["Temperature (°C)"] == pytest.approx(126.85)
    assert data["Cp/Cv (k)"] == 2.0
    assert data["Compressibility Factor (Z)"] == 1.0


def test_select_best_engine_auto_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SteamCalculationEngine()

    monkeypatch.setattr(steam_engine, "COOLPROP_AVAILABLE", True)
    monkeypatch.setattr(steam_engine, "CANTERA_AVAILABLE", True)
    engine.water = object()
    assert engine._select_best_engine("auto") == "coolprop"

    monkeypatch.setattr(steam_engine, "COOLPROP_AVAILABLE", False)
    assert engine._select_best_engine("auto") == "cantera"

    engine.water = None
    assert engine._select_best_engine("auto") == "simplified"


def test_select_best_engine_requested_unavailable_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SteamCalculationEngine()
    monkeypatch.setattr(steam_engine, "COOLPROP_AVAILABLE", False)
    monkeypatch.setattr(steam_engine, "CANTERA_AVAILABLE", False)
    engine.water = None
    assert engine._select_best_engine("coolprop") == "simplified"
    assert engine._select_best_engine("cantera") == "simplified"
    assert engine._select_best_engine("simplified") == "simplified"


def test_calculate_properties_uses_selected_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SteamCalculationEngine()
    monkeypatch.setattr(steam_engine, "COOLPROP_AVAILABLE", True)
    monkeypatch.setattr(steam_engine, "CANTERA_AVAILABLE", True)
    engine.water = object()
    cp_props = _sentinel_props(500.0, 2e5, "coolprop")
    ct_props = _sentinel_props(500.0, 2e5, "cantera")
    sp_props = _sentinel_props(500.0, 2e5, "simplified")

    monkeypatch.setattr(engine, "_calculate_coolprop_properties", lambda *_: cp_props)
    monkeypatch.setattr(engine, "_calculate_cantera_properties", lambda *_: ct_props)
    monkeypatch.setattr(engine, "_calculate_simplified_properties", lambda *_: sp_props)

    assert engine.calculate_properties(500.0, 2e5, "coolprop").phase == "coolprop"
    assert engine.calculate_properties(500.0, 2e5, "cantera").phase == "cantera"
    assert engine.calculate_properties(500.0, 2e5, "simplified").phase == "simplified"


def test_calculate_properties_falls_back_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SteamCalculationEngine()
    fallback = _sentinel_props(420.0, 101325.0, "fallback")
    monkeypatch.setattr(
        engine,
        "_calculate_coolprop_properties",
        lambda *_: (_ for _ in ()).throw(ValueError("boom")),
    )
    monkeypatch.setattr(engine, "_calculate_simplified_properties", lambda *_: fallback)
    assert engine.calculate_properties(420.0, 101325.0, "coolprop").phase == "fallback"


def test_calculate_properties_tags_engine_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """engine_used reflects the engine that actually ran (#3318)."""
    engine = SteamCalculationEngine()
    monkeypatch.setattr(steam_engine, "COOLPROP_AVAILABLE", True)
    engine.water = object()
    monkeypatch.setattr(
        engine,
        "_calculate_coolprop_properties",
        lambda *_: _sentinel_props(500.0, 2e5, "vapor"),
    )
    monkeypatch.setattr(
        engine,
        "_calculate_simplified_properties",
        lambda *_: _sentinel_props(500.0, 2e5, "vapor"),
    )
    result = engine.calculate_properties(500.0, 2e5, "coolprop")
    assert result.engine_used == "coolprop"
    assert result.to_dict()["Engine Used"] == "coolprop"


def test_calculate_properties_fallback_reports_simplified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent fallback tags engine_used='simplified', not requested (#3318)."""
    engine = SteamCalculationEngine()
    monkeypatch.setattr(steam_engine, "COOLPROP_AVAILABLE", True)
    engine.water = object()
    monkeypatch.setattr(
        engine,
        "_calculate_coolprop_properties",
        lambda *_: (_ for _ in ()).throw(ValueError("backend failed")),
    )
    monkeypatch.setattr(
        engine,
        "_calculate_simplified_properties",
        lambda *_: _sentinel_props(420.0, 101325.0, "vapor"),
    )
    result = engine.calculate_properties(420.0, 101325.0, "coolprop")
    # Requested coolprop, but the numbers came from the simplified engine.
    assert result.engine_used == "simplified"


def test_vapor_pressure_methods_and_default() -> None:
    engine = SteamCalculationEngine()
    buck = engine.calculate_water_vapor_pressure(25.0, method="buck")
    antoine = engine.calculate_water_vapor_pressure(25.0, method="antoine")
    defaulted = engine.calculate_water_vapor_pressure(25.0, method="unknown-method")
    assert buck > 0
    assert antoine > 0
    assert defaulted == pytest.approx(buck)


def test_iapws_falls_back_to_buck_when_coolprop_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SteamCalculationEngine()
    monkeypatch.setattr(steam_engine, "COOLPROP_AVAILABLE", False)
    monkeypatch.setattr(engine, "_buck_equation", lambda _: 12345.0)
    assert engine._iapws_equation(50.0) == pytest.approx(12345.0)


def test_calculate_dew_point_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SteamCalculationEngine()
    monkeypatch.setattr(
        engine,
        "calculate_water_vapor_pressure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("x")),
    )
    result = engine.calculate_dew_point(
        partial_pressure_pa=1000.0, total_pressure_pa=1e5
    )
    assert result == pytest.approx(DEFAULT_DEW_POINT_TEMPERATURE_CELSIUS)


def test_get_saturation_pressure_falls_back_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SteamCalculationEngine()
    monkeypatch.setattr(steam_engine, "CANTERA_AVAILABLE", True)

    class _BadWater:
        @property
        def P(self):
            raise ValueError("bad")

        @property
        def TQ(self):
            return (0.0, 0.0)

        @TQ.setter
        def TQ(self, _value):
            pass

    engine.water = _BadWater()
    result = engine.get_saturation_pressure(300.0)
    assert result > 0


def test_saturation_preconditions_raise_before_backend_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SteamCalculationEngine()
    monkeypatch.setattr(
        engine,
        "_calculate_saturated_simplified_from_temp",
        lambda *_: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )

    with pytest.raises(ValueError, match="saturation bounds"):
        engine.calculate_saturated_properties_from_temperature(200.0)


def test_get_saturation_temperature_rejects_invalid_pressure() -> None:
    engine = SteamCalculationEngine()

    with pytest.raises(ValueError, match="saturation bounds"):
        engine.get_saturation_temperature(-5.0)


def test_coolprop_quality_error_is_unknown_not_saturated_liquid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SteamCalculationEngine()

    def fake_props(prop: str, *_args: object) -> float:
        values = {
            "D": 1.0,
            "H": 2.0,
            "S": 3.0,
            "U": 4.0,
            "Cpmass": 5.0,
            "Cvmass": 4.0,
            "A": 6.0,
            "L": 0.5,
            "VISCOSITY": 0.01,
        }
        if prop == "Q":
            raise ValueError("quality unavailable")
        return values[prop]

    monkeypatch.setattr(steam_engine, "PropsSI", fake_props)
    monkeypatch.setattr(steam_engine, "PhaseSI", lambda *_args: "unknown")

    props = engine._calculate_coolprop_properties(400.0, 101325.0)

    assert math.isnan(props.quality)
    assert props.phase == "unknown"


def test_determine_phase_and_quality_paths() -> None:
    engine = SteamCalculationEngine()
    supercritical = engine._determine_phase_and_quality(700.0, 1e5)
    assert supercritical == ("supercritical", 1.0)


def test_calculate_simplified_properties_liquid_and_vapor() -> None:
    engine = SteamCalculationEngine()
    liquid = engine._calculate_simplified_properties(300.0, 2e5)
    vapor = engine._calculate_simplified_properties(450.0, 101325.0)
    assert liquid.phase == "liquid"
    assert vapor.phase == "vapor"
    assert liquid.cp > vapor.cv


def test_calculate_simplified_properties_catastrophic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SteamCalculationEngine()
    monkeypatch.setattr(
        steam_engine.np,
        "sqrt",
        lambda *_: (_ for _ in ()).throw(TypeError("bad")),
    )
    error_props = engine._calculate_simplified_properties(450.0, 101325.0)
    assert asdict(error_props)["phase"] == "error"

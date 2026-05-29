"""Unit tests for pressure-drop support modules.

Covers the previously low/zero-coverage helpers:
``pressure_drop_units`` (unit converters), ``pressure_drop_reference``
(discovery/listing helpers), ``pressure_drop_results`` (formatting/rendering)
and ``utils.fitting_loss_coefficients`` (K-factor helpers).
"""

from __future__ import annotations

import types

import pytest
from sidekick.process_calculators.pressure_drop_calculator import (
    pressure_drop_reference as ref,
)
from sidekick.process_calculators.pressure_drop_calculator import (
    pressure_drop_results as res,
)
from sidekick.process_calculators.pressure_drop_calculator import (
    pressure_drop_units as units,
)
from sidekick.process_calculators.pressure_drop_calculator.utils import (
    fitting_loss_coefficients as flc,
)

# ---------------------------------------------------------------------------
# pressure_drop_units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "from_u", "to_u", "expected"),
    [
        (0.0, "C", "K", 273.15),
        (100.0, "C", "F", 212.0),
        (32.0, "F", "C", 0.0),
        (300.0, "K", "K", 300.0),
        (273.15, "K", "C", 0.0),
    ],
)
def test_convert_temperature(
    value: float, from_u: str, to_u: str, expected: float
) -> None:
    assert units._convert_temperature(value, from_u, to_u) == pytest.approx(
        expected, abs=1e-6
    )


def test_convert_temperature_lowercase_units_accepted() -> None:
    # Units are upper-cased internally.
    assert units._convert_temperature(0.0, "c", "k") == pytest.approx(273.15)


def test_convert_temperature_none_raises() -> None:
    with pytest.raises(ValueError, match="value must be provided"):
        units._convert_temperature(None, "C", "K")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_unit", ["X", "rankine"])
def test_convert_temperature_unknown_unit_raises(bad_unit: str) -> None:
    with pytest.raises(ValueError, match="Unknown temperature unit"):
        units._convert_temperature(10.0, bad_unit, "K")
    with pytest.raises(ValueError, match="Unknown temperature unit"):
        units._convert_temperature(10.0, "K", bad_unit)


@pytest.mark.parametrize(
    ("value", "from_u", "to_u", "expected"),
    [
        (1.0, "bar", "Pa", 1e5),
        (1.0, "atm", "Pa", 101325.0),
        (1000.0, "Pa", "kPa", 1.0),
        (1.0, "MPa", "bar", 10.0),
    ],
)
def test_convert_pressure(
    value: float, from_u: str, to_u: str, expected: float
) -> None:
    assert units._convert_pressure(value, from_u, to_u) == pytest.approx(expected)


@pytest.mark.parametrize("bad_unit", ["torr", "Hg"])
def test_convert_pressure_unknown_unit_raises(bad_unit: str) -> None:
    with pytest.raises(ValueError, match="Unknown pressure unit"):
        units._convert_pressure(1.0, bad_unit, "Pa")
    with pytest.raises(ValueError, match="Unknown pressure unit"):
        units._convert_pressure(1.0, "Pa", bad_unit)


# ---------------------------------------------------------------------------
# pressure_drop_reference
# ---------------------------------------------------------------------------


def test_list_gas_components_returns_property_dicts() -> None:
    components = ref.list_gas_components()
    assert isinstance(components, dict)
    assert components, "expected at least one gas component"
    sample = next(iter(components.values()))
    assert {"molecular_weight", "critical_temp"} <= set(sample)


def test_list_fittings_all_and_filtered() -> None:
    every = ref.list_fittings()
    assert every and all(isinstance(v, float) for v in every.values())
    valves = ref.list_fittings(category="valve")
    # Filtering must be a subset and every key must look like a valve.
    assert set(valves) <= set(every)
    assert all("valve" in name for name in valves)


def test_list_pipe_sizes_maps_sizes_to_schedules() -> None:
    sizes = ref.list_pipe_sizes()
    assert isinstance(sizes, dict)
    assert all(isinstance(v, list) for v in sizes.values())


def test_list_flow_units_has_expected_groups() -> None:
    flow_units = ref.list_flow_units()
    assert {"mass", "molar", "volumetric", "standard_conditions"} <= set(flow_units)


def test_list_materials_includes_roughness_in_m() -> None:
    materials = ref.list_materials()
    assert materials
    sample = next(iter(materials.values()))
    assert sample["roughness_m"] == pytest.approx(sample["roughness_mm"] / 1000)


def test_compare_friction_methods_returns_four_methods() -> None:
    methods = ref.compare_friction_methods(1.0e5, 0.0002)
    assert set(methods) == {"colebrook", "swamee-jain", "churchill", "haaland"}
    assert all(v > 0 for v in methods.values())


def test_compare_friction_methods_none_raises() -> None:
    with pytest.raises(ValueError, match="reynolds_number must be provided"):
        ref.compare_friction_methods(None)  # type: ignore[arg-type]


def test_show_help_logs(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.INFO):
        ref.show_help()
    assert "PRESSURE DROP CALCULATOR" in caplog.text


# ---------------------------------------------------------------------------
# utils.fitting_loss_coefficients
# ---------------------------------------------------------------------------


def test_get_fitting_k_factor_known() -> None:
    assert flc.get_fitting_k_factor("90_elbow_std") == pytest.approx(0.75)


def test_get_fitting_k_factor_unknown_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        flc.get_fitting_k_factor("not_a_real_fitting")


def test_get_multiple_fittings_k_sums() -> None:
    total = flc.get_multiple_fittings_k({"90_elbow_std": 2, "gate_valve_open": 1})
    assert total == pytest.approx(0.75 * 2 + 0.15)


def test_k_equivalent_length_roundtrip() -> None:
    f = 0.02
    ld = flc.k_to_equivalent_length(0.6, f)
    assert flc.equivalent_length_to_k(ld, f) == pytest.approx(0.6)


@pytest.mark.parametrize(
    "func", [flc.k_to_equivalent_length, flc.equivalent_length_to_k]
)
def test_equivalent_length_nonpositive_friction_raises(func) -> None:
    with pytest.raises(ValueError, match="Friction factor must be positive"):
        func(1.0, 0.0)


def test_calculate_two_k_factor_known() -> None:
    k = flc.calculate_two_k_factor("90_elbow_std_2k", 50000.0, 4.0)
    # K = K1/Re + K_inf*(1 + Kd/ID^0.3) with (800, 0.25, 4.0)
    expected = 800 / 50000.0 + 0.25 * (1.0 + 4.0 / (4.0**0.3))
    assert k == pytest.approx(expected)


def test_calculate_two_k_factor_unknown_raises() -> None:
    with pytest.raises(ValueError, match="not in Two-K database"):
        flc.calculate_two_k_factor("nope_2k", 50000.0, 4.0)


def test_list_available_fittings_is_copy() -> None:
    listed = flc.list_available_fittings()
    listed.clear()
    assert flc.FITTING_K_FACTORS, "mutating the returned dict must not affect source"


def test_apply_k_factor_known() -> None:
    assert flc.apply_k_factor(2.0, 1000.0, 5.0) == pytest.approx(2.0 * 0.5 * 1000 * 25)


def test_apply_k_factor_none_raises() -> None:
    with pytest.raises(ValueError, match="k_factor must be provided"):
        flc.apply_k_factor(None, 1000.0, 5.0)  # type: ignore[arg-type]


def test_print_fitting_database_logs(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.INFO):
        flc.print_fitting_database()
    assert "K-FACTORS" in caplog.text


# ---------------------------------------------------------------------------
# pressure_drop_results
# ---------------------------------------------------------------------------


def test_wrap_text_basic() -> None:
    lines = res._wrap_text("the quick brown fox jumps", width=10)
    assert all(len(line) <= 10 for line in lines)
    assert " ".join(lines) == "the quick brown fox jumps"


def test_wrap_text_empty_returns_single_empty() -> None:
    assert res._wrap_text("", width=10) == [""]


def test_wrap_text_none_raises() -> None:
    with pytest.raises(ValueError, match="text must be provided"):
        res._wrap_text(None, width=10)  # type: ignore[arg-type]


def _results_obj() -> types.SimpleNamespace:
    flow = types.SimpleNamespace(
        reynolds_number=1.5e5,
        velocity=12.0,
        mach_number=0.05,
        density=1.2,
        viscosity=1.8e-5,
        compressibility_factor=0.99,
        molecular_weight=18.0,
    )
    return types.SimpleNamespace(
        total_pressure_drop=2.0e4,
        friction_pressure_drop=1.5e4,
        fitting_pressure_drop=4.0e3,
        elevation_pressure_drop=1.0e3,
        outlet_pressure=4.8e5,
        friction_factor=0.02,
        flow_properties=flow,
        flow_regime="turbulent",
        erosional_velocity=20.0,
        erosion_ratio=0.6,
        pressure_drop_per_100ft=500.0,
        velocity_pressure=86.4,
        warnings=["sample warning"],
    )


def test_format_results_produces_unit_conversions() -> None:
    formatted = res._format_results(_results_obj())
    assert formatted["pressure_drop_bar"] == pytest.approx(2.0e4 / 1e5)
    assert formatted["pressure_drop_psi"] == pytest.approx(2.0e4 / 6894.76)
    assert formatted["flow_regime"] == "turbulent"
    assert formatted["erosion_ratio_percent"] == pytest.approx(60.0)


def test_generate_recommendations_flags_erosion_and_fittings() -> None:
    formatted = res._format_results(_results_obj())
    # Push fitting loss above friction loss and erosion above 0.8 to trip rules.
    formatted["fitting_loss_pa"] = formatted["friction_loss_pa"] + 1.0
    formatted["erosion_ratio"] = 0.85
    recs = res._generate_recommendations(formatted)
    joined = " ".join(recs)
    assert "erosional" in joined.lower()
    assert "fitting" in joined.lower()


def test_print_results_runs_without_error(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    formatted = res._format_results(_results_obj())
    with caplog.at_level(logging.INFO):
        res.print_results(formatted, show_recommendations=True)
    assert "PRESSURE DROP CALCULATION RESULTS" in caplog.text


def test_print_results_none_raises() -> None:
    with pytest.raises(ValueError, match="results must be provided"):
        res.print_results(None)  # type: ignore[arg-type]

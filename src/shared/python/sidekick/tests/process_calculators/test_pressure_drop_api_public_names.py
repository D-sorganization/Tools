"""Issue #3991 regression tests for the pressure drop calculator facade.

``pressure_drop_api.py`` used to launder sibling privates into its public
namespace (``from .pressure_drop_results import _format_results as
format_results``), and ``pressure_drop_validation.py`` imported the private
``_wrap_text`` across modules. The sidekick public-API contract requires that
exported names are public (underscore-free) at their definition site; these
tests pin that invariant for the pressure drop calculator.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

import pytest
from sidekick.process_calculators.pressure_drop_calculator import (
    pressure_drop_api,
    print_results,
)
from sidekick.process_calculators.pressure_drop_calculator.engine import (
    PressureDropCalculationEngine,
)
from sidekick.process_calculators.pressure_drop_calculator.models import (
    GasComposition,
    PressureDropInputs,
)
from sidekick.process_calculators.pressure_drop_calculator.pressure_drop_api import (
    build_fitting_list,
    calculate_pressure_drop,
    calculate_pressure_drop_custom_gas,
    calculate_pressure_drop_syngas,
    convert_pressure,
    convert_temperature,
    format_results,
    resolve_gas_and_flow,
    resolve_pipe_geometry,
)

_PACKAGE_DIR = Path(pressure_drop_api.__file__).resolve().parent
_SIDEKICK_MARKER = "sidekick.process_calculators.pressure_drop_calculator"
_ISSUE_MODULE_FILES = ("pressure_drop_api.py", "pressure_drop_validation.py")


# ---------------------------------------------------------------------------
# Public-API contract: no laundered sibling privates
# ---------------------------------------------------------------------------


def _home_module(obj: Any) -> ModuleType | None:
    """Return the sidekick calculator module that defines *obj*."""
    module_name = getattr(obj, "__module__", "")
    if _SIDEKICK_MARKER not in module_name:
        return None
    return importlib.import_module(module_name)


def _ast_defines_public_name(module: ModuleType, name: str) -> bool:
    """True when *module* defines *name* as a top-level public def/class."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
        for node in tree.body
    )


def test_all_names_defined_public_in_home_module() -> None:
    """Every name in pressure_drop_api.__all__ is public at its definition site."""
    for name in pressure_drop_api.__all__:
        obj = getattr(pressure_drop_api, name)
        home = _home_module(obj)
        assert home is not None, (
            f"{name!r} does not resolve to a pressure drop calculator module "
            f"({getattr(obj, '__module__', '?')!r})"
        )
        assert _ast_defines_public_name(home, name), (
            f"{name!r} is exported by pressure_drop_api but home module "
            f"{home.__name__} does not define a public (underscore-free) "
            f"top-level def/class named {name!r} (issue #3991)"
        )


def test_public_attributes_not_laundered_from_sibling_privates() -> None:
    """Every public function/class attribute of the facade is public at home."""
    violations: list[str] = []
    for name in dir(pressure_drop_api):
        if name.startswith("_"):
            continue
        obj = getattr(pressure_drop_api, name)
        if not isinstance(obj, (FunctionType, type)):
            continue
        home = _home_module(obj)
        if home is None:
            continue
        if not _ast_defines_public_name(home, name):
            violations.append(
                f"{name!r} is a public attribute of pressure_drop_api but "
                f"{home.__name__} does not define it as a public top-level "
                f"def/class (laundered sibling private)"
            )
    assert not violations, (
        "pressure_drop_api launders sibling privates (issue #3991):\n"
        + "\n".join(violations)
    )


def test_issue_modules_do_not_import_sibling_privates() -> None:
    """api/validation import no underscore-prefixed names from siblings."""
    violations: list[str] = []
    for module_file in _ISSUE_MODULE_FILES:
        path = _PACKAGE_DIR / module_file
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    violations.append(
                        f"{module_file}:{node.lineno} imports private "
                        f"{alias.name!r} from {node.module!r}"
                    )
    assert not violations, (
        "Cross-module private imports detected (issue #3991):\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Facade behavior (keeps the renamed surface exercised for coverage)
# ---------------------------------------------------------------------------


def test_calculate_pressure_drop_end_to_end() -> None:
    """Full facade call with explicit units and fittings."""
    result = calculate_pressure_drop(
        pipe_size="4",
        pipe_schedule="40",
        pipe_length=100.0,
        flow_rate=1000.0,
        flow_unit="kg/h",
        pressure=10.0,
        pressure_unit="bar",
        temperature=500.0,
        temperature_unit="K",
        fittings=[{"type": "90_elbow_std", "quantity": 4}],
    )
    assert result["pressure_drop_bar"] > 0
    assert result["fitting_loss_pa"] > 0


def test_calculate_pressure_drop_temperature_unit_conversion() -> None:
    """Celsius input matches the equivalent Kelvin input."""
    kelvin = calculate_pressure_drop(
        pipe_size="4",
        pipe_schedule="40",
        pipe_length=50.0,
        flow_rate=1000.0,
        flow_unit="kg/h",
        pressure=10.0,
        temperature=500.0,
    )
    celsius = calculate_pressure_drop(
        pipe_size="4",
        pipe_schedule="40",
        pipe_length=50.0,
        flow_rate=1000.0,
        flow_unit="kg/h",
        pressure=10.0,
        temperature=226.85,
        temperature_unit="C",
    )
    assert celsius["pressure_drop_pa"] == pytest.approx(
        kelvin["pressure_drop_pa"], rel=1e-9
    )


def test_calculate_pressure_drop_custom_gas_and_syngas() -> None:
    """Simplified custom-gas and syngas wrappers produce formatted results."""
    custom = calculate_pressure_drop_custom_gas(
        pipe_diameter=0.1,
        pipe_length=100.0,
        gas_composition={"H2": 0.7, "CO": 0.2, "CO2": 0.1},
        flow_rate=1000.0,
        flow_unit="kg/h",
        pressure=10.0,
        temperature=500.0,
    )
    assert custom["pressure_drop_pa"] > 0

    syngas = calculate_pressure_drop_syngas(
        pipe_size="4",
        pipe_schedule="40",
        pipe_length=100.0,
        flow_rate=1000.0,
        flow_unit="kg/h",
        pressure=10.0,
        temperature=500.0,
    )
    assert syngas["pressure_drop_pa"] > 0


def test_resolve_pipe_geometry_explicit_and_spec_paths() -> None:
    """Explicit diameter bypasses the spec lookup; spec path fills roughness."""
    diameter, roughness = resolve_pipe_geometry(
        None, None, 0.102, "Commercial Steel", 4.5e-05
    )
    assert diameter == pytest.approx(0.102)
    assert roughness == pytest.approx(4.5e-05)

    spec_diameter, material_roughness = resolve_pipe_geometry(
        "4", "40", None, "Commercial Steel", None
    )
    assert spec_diameter > 0
    assert material_roughness > 0


def test_resolve_gas_and_flow_defaults_to_air() -> None:
    """A missing composition defaults to air and returns a mass flow."""
    composition, mass_flow = resolve_gas_and_flow(
        1000.0, "kg/h", None, 500.0, 1_000_000.0, True, "STP"
    )
    assert composition.components["Air"] == pytest.approx(1.0)
    assert mass_flow > 0


def test_build_fitting_list_from_dicts() -> None:
    """Raw fitting dicts convert to PipeFitting objects; empty input is empty."""
    fittings = build_fitting_list(
        [
            {"type": "elbow", "quantity": 3, "k_factor": 0.3},
            {"type": "gate_valve"},
        ]
    )
    assert [fitting.fitting_type for fitting in fittings] == [
        "elbow",
        "gate_valve",
    ]
    assert fittings[0].quantity == 3
    assert fittings[0].k_factor == pytest.approx(0.3)
    assert build_fitting_list(None) == []
    assert build_fitting_list([]) == []


# ---------------------------------------------------------------------------
# Unit conversion helpers (imported through the facade, renamed in #3991)
# ---------------------------------------------------------------------------


def test_convert_temperature_units() -> None:
    """Temperature conversion supports K/C/F and rejects unknown units."""
    assert convert_temperature(0.0, "C", "K") == pytest.approx(273.15)
    assert convert_temperature(100.0, "C", "F") == pytest.approx(212.0)
    assert convert_temperature(32.0, "F", "C") == pytest.approx(0.0)
    assert convert_temperature(300.0, "K", "K") == pytest.approx(300.0)
    with pytest.raises(ValueError, match="Unknown temperature unit"):
        convert_temperature(1.0, "X", "K")
    with pytest.raises(ValueError, match="Unknown temperature unit"):
        convert_temperature(1.0, "K", "X")


def test_convert_pressure_units() -> None:
    """Pressure conversion round-trips common units and rejects unknown ones."""
    assert convert_pressure(1.0, "bar", "Pa") == pytest.approx(1e5)
    assert convert_pressure(1.0, "MPa", "bar") == pytest.approx(10.0)
    assert convert_pressure(1.0, "psi", "bar") == pytest.approx(0.0689476, rel=1e-6)
    with pytest.raises(ValueError, match="Unknown pressure unit"):
        convert_pressure(1.0, "bogus", "Pa")
    with pytest.raises(ValueError, match="Unknown pressure unit"):
        convert_pressure(1.0, "Pa", "bogus")


# ---------------------------------------------------------------------------
# Result formatting helpers (format_results renamed in #3991)
# ---------------------------------------------------------------------------


def _facade_results() -> dict[str, Any]:
    """Return a formatted result dictionary from the facade."""
    return calculate_pressure_drop(
        pipe_size="4",
        pipe_schedule="40",
        pipe_length=100.0,
        flow_rate=1000.0,
        flow_unit="kg/h",
        pressure=10.0,
        temperature=500.0,
        fittings=[{"type": "elbow", "quantity": 4}],
    )


def test_format_results_exposes_engine_output() -> None:
    """format_results maps raw engine results onto the documented keys."""
    engine = PressureDropCalculationEngine()
    inputs = PressureDropInputs(
        pipe_diameter=0.102,
        pipe_length=100.0,
        pipe_roughness=4.5e-05,
        elevation_change=0.0,
        mass_flow_rate=0.5,
        inlet_pressure=1_000_000.0,
        inlet_temperature=500.0,
        gas_composition=GasComposition(components={"Air": 1.0}),
        fittings=[],
        compressibility_correction=True,
        friction_method="colebrook",
    )
    raw = engine.calculate(inputs)
    formatted = format_results(raw)
    assert formatted["pressure_drop_pa"] == pytest.approx(raw.total_pressure_drop)
    assert formatted["reynolds_number"] == raw.flow_properties.reynolds_number
    assert formatted["warnings"] == raw.warnings


def test_print_results_renders_report_and_recommendations() -> None:
    """print_results renders every section without raising."""
    formatted = _facade_results()
    print_results(formatted)
    print_results(formatted, show_recommendations=False)


def _synthetic_results(**overrides: Any) -> dict[str, Any]:
    """Build a minimal formatted result dictionary for recommendation paths."""
    base: dict[str, Any] = {
        "pressure_drop_pa": 1.0,
        "pressure_drop_bar": 1e-05,
        "pressure_drop_psi": 1.45e-04,
        "pressure_drop_kpa": 1e-03,
        "friction_loss_pa": 1.0,
        "friction_loss_bar": 1e-05,
        "fitting_loss_pa": 0.0,
        "fitting_loss_bar": 0.0,
        "elevation_loss_pa": 0.0,
        "outlet_pressure_pa": 1e6,
        "outlet_pressure_bar": 10.0,
        "outlet_pressure_psi": 145.0,
        "friction_factor": 0.02,
        "reynolds_number": 1e5,
        "flow_velocity_m_s": 1.0,
        "flow_velocity_ft_s": 3.28084,
        "mach_number": 0.1,
        "flow_regime": "turbulent",
        "density_kg_m3": 1.0,
        "viscosity_pa_s": 1e-05,
        "compressibility_factor": 1.0,
        "molecular_weight": 28.97,
        "erosional_velocity_m_s": 10.0,
        "erosion_ratio": 0.1,
        "erosion_ratio_percent": 10.0,
        "pressure_drop_per_100ft_pa": 1.0,
        "velocity_pressure_pa": 1.0,
        "warnings": [],
    }
    base.update(overrides)
    return base


def test_print_results_recommendation_branches() -> None:
    """Each recommendation branch renders through the public print path."""
    hot = _synthetic_results(
        pressure_drop_pa=300_000.0,
        outlet_pressure_pa=700_000.0,
        erosion_ratio=0.9,
        fitting_loss_pa=2000.0,
        friction_loss_pa=1000.0,
        mach_number=0.5,
        warnings=[
            "",
            "Check inlet pressure at the compressor station before the next "
            "scheduled maintenance window",
        ],
    )
    print_results(hot, title="synthetic")

    low_re = _synthetic_results(reynolds_number=2000.0, erosion_ratio=0.6)
    print_results(low_re, title="synthetic-low-re")

    high_re = _synthetic_results(reynolds_number=2.0e7)
    print_results(high_re, title="synthetic-high-re")

    print_results(low_re, show_recommendations=False)

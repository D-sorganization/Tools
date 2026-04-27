"""Tests for pressure drop interface functions."""

from __future__ import annotations

from upstream_drift_tools.process_calculators.pressure_drop_calculator.pressure_drop_interface import (
    calculate_pressure_drop,
    compare_friction_methods,
    list_fittings,
    list_flow_units,
    list_gas_components,
    list_materials,
    list_pipe_sizes,
    validate_inputs,
)


class TestInterfaceInfoFunctions:
    def test_list_gas_components(self) -> None:
        components = list_gas_components()
        assert "Air" in components
        assert "H2" in components
        assert components["Air"]["molecular_weight"] > 28.0

    def test_list_fittings(self) -> None:
        all_fittings = list_fittings()
        assert len(all_fittings) > 0
        assert "90_elbow_std" in all_fittings

        elbows = list_fittings(category="elbow")
        assert "90_elbow_std" in elbows
        assert "gate_valve_open" not in elbows

    def test_list_pipe_sizes(self) -> None:
        sizes = list_pipe_sizes()
        assert "4" in sizes
        assert "40" in sizes["4"]

    def test_list_flow_units(self) -> None:
        units = list_flow_units()
        assert "kg/h" in units["mass"]
        assert "kmol/h" in units["molar"]
        assert "m³/h" in units["volumetric"]

    def test_list_materials(self) -> None:
        materials = list_materials()
        assert "Commercial Steel" in materials
        assert "roughness_mm" in materials["Commercial Steel"]


class TestCompareFrictionMethods:
    def test_compare(self) -> None:
        results = compare_friction_methods(100000, 0.001)
        assert "colebrook" in results
        assert "swamee-jain" in results
        assert "churchill" in results
        assert "haaland" in results
        assert all(v > 0 for v in results.values())


class TestValidateInputs:
    def test_valid_inputs(self) -> None:
        is_valid, errors, warnings = validate_inputs(
            pipe_size="4",
            pipe_schedule="40",
            flow_rate=100,
            flow_unit="kg/h",
            pressure=10,
            temperature=300,
            gas_composition={"Air": 1.0},
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_geometry(self) -> None:
        is_valid, errors, _ = validate_inputs(
            flow_rate=100,
            flow_unit="kg/h",
        )
        assert is_valid is False
        assert any("pipe_size" in e or "diameter" in e for e in errors)

    def test_invalid_flow_rate(self) -> None:
        is_valid, errors, _ = validate_inputs(
            pipe_diameter=0.1,
            flow_rate=-10,
            flow_unit="kg/h",
        )
        assert is_valid is False
        assert any("flow_rate" in e for e in errors)

    def test_unknown_component(self) -> None:
        is_valid, errors, _ = validate_inputs(
            pipe_diameter=0.1,
            flow_rate=100,
            flow_unit="kg/h",
            gas_composition={"Unobtanium": 1.0},
        )
        assert is_valid is False
        assert any("Unobtanium" in e for e in errors)

    def test_composition_warning(self) -> None:
        is_valid, _, warnings = validate_inputs(
            pipe_diameter=0.1,
            flow_rate=100,
            flow_unit="kg/h",
            gas_composition={"Air": 0.5},  # Doesn't sum to 1
        )
        assert is_valid is True
        assert any("sums to" in w for w in warnings)


class TestCalculatePressureDrop:
    def test_basic_calculation(self) -> None:
        result = calculate_pressure_drop(
            pipe_diameter=0.1,
            pipe_length=100,
            flow_rate=5000,
            flow_unit="kg/h",
            pressure=10,
            temperature=300,
            gas_composition={"Air": 1.0},
        )
        assert "pressure_drop_pa" in result
        assert result["pressure_drop_pa"] > 0
        assert result["reynolds_number"] > 4000  # Should be turbulent

    def test_with_standard_pipe(self) -> None:
        result = calculate_pressure_drop(
            pipe_size="4",
            pipe_schedule="40",
            pipe_length=100,
            flow_rate=5000,
            flow_unit="kg/h",
            pressure=10,
            temperature=300,
        )
        assert result["pressure_drop_pa"] > 0

    def test_with_fittings(self) -> None:
        res_no_fit = calculate_pressure_drop(
            pipe_diameter=0.1, pipe_length=100, flow_rate=1000, pressure=10
        )
        res_fit = calculate_pressure_drop(
            pipe_diameter=0.1,
            pipe_length=100,
            flow_rate=1000,
            pressure=10,
            fittings=[{"type": "90_elbow_std", "quantity": 10}],
        )
        assert res_fit["fitting_loss_pa"] > 0
        assert res_fit["pressure_drop_pa"] > res_no_fit["pressure_drop_pa"]

    def test_erosional_velocity_warning(self) -> None:
        # Flow rate > 90 m/s to trigger erosional velocity but < 340 m/s
        result = calculate_pressure_drop(
            pipe_diameter=0.1,  # 10cm pipe
            pipe_length=10,
            flow_rate=2.0,  # kg/s -> velocity ~ 210 m/s
            flow_unit="kg/s",
            pressure=1,  # 1 bar
            temperature=300,
        )
        assert "warnings" in result
        assert any("erosional" in w.lower() for w in result["warnings"])

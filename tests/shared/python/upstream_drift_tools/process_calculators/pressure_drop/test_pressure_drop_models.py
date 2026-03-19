"""Comprehensive tests for pressure_drop_data_models.

Tests cover GasComposition, PipeFitting, PressureDropInputs validation,
FlowProperties, PressureDropResults, PipeSpecification, and FlowRateInput.
"""

from __future__ import annotations

from upstream_drift_tools.process_calculators.pressure_drop_calculator.models.pressure_drop_data_models import (
    FlowProperties,
    FlowRateInput,
    GasComposition,
    PipeFitting,
    PipeSpecification,
    PressureDropInputs,
    PressureDropResults,
)

# ─── GasComposition Tests ────────────────────────────────────


class TestGasComposition:
    def test_empty_composition(self) -> None:
        gc = GasComposition()
        assert gc.components == {}

    def test_valid_composition(self) -> None:
        gc = GasComposition(components={"N2": 0.79, "O2": 0.21})
        assert gc.validate() is True

    def test_invalid_sum_low(self) -> None:
        gc = GasComposition(components={"N2": 0.5, "O2": 0.1})
        assert gc.validate() is False

    def test_invalid_sum_high(self) -> None:
        gc = GasComposition(components={"N2": 0.8, "O2": 0.5})
        assert gc.validate() is False

    def test_negative_fraction(self) -> None:
        gc = GasComposition(components={"N2": -0.1, "O2": 1.1})
        assert gc.validate() is False

    def test_fraction_above_one(self) -> None:
        gc = GasComposition(components={"N2": 1.5})
        assert gc.validate() is False

    def test_normalize(self) -> None:
        gc = GasComposition(components={"N2": 0.6, "O2": 0.4})
        gc.normalize()
        total = sum(gc.components.values())
        assert abs(total - 1.0) < 1e-10

    def test_normalize_unnormalized(self) -> None:
        gc = GasComposition(components={"N2": 60.0, "O2": 40.0})
        gc.normalize()
        assert abs(gc.components["N2"] - 0.6) < 1e-10
        assert abs(gc.components["O2"] - 0.4) < 1e-10

    def test_normalize_zero_total(self) -> None:
        gc = GasComposition(components={"N2": 0.0, "O2": 0.0})
        gc.normalize()  # Should not crash
        assert gc.components["N2"] == 0.0


# ─── PipeFitting Tests ───────────────────────────────────────


class TestPipeFitting:
    def test_basic_construction(self) -> None:
        fitting = PipeFitting(fitting_type="elbow", quantity=2, k_factor=0.75)
        assert fitting.fitting_type == "elbow"
        assert fitting.quantity == 2
        assert fitting.k_factor == 0.75

    def test_defaults(self) -> None:
        fitting = PipeFitting(fitting_type="valve")
        assert fitting.quantity == 1
        assert fitting.k_factor == 0.0
        assert fitting.description == ""


# ─── PressureDropInputs Validation Tests ─────────────────────


class TestPressureDropInputsValidation:
    def _valid_inputs(self) -> PressureDropInputs:
        return PressureDropInputs(
            pipe_diameter=0.1,
            pipe_length=100.0,
            pipe_roughness=0.00004,
            mass_flow_rate=5.0,
            inlet_pressure=500000.0,
            inlet_temperature=400.0,
            gas_composition=GasComposition(components={"N2": 0.79, "O2": 0.21}),
        )

    def test_valid_inputs_pass(self) -> None:
        inputs = self._valid_inputs()
        valid, msg = inputs.validate()
        assert valid is True

    def test_zero_diameter_fails(self) -> None:
        inputs = self._valid_inputs()
        inputs.pipe_diameter = 0.0
        valid, msg = inputs.validate()
        assert valid is False
        assert "diameter" in msg.lower()

    def test_negative_length_fails(self) -> None:
        inputs = self._valid_inputs()
        inputs.pipe_length = -10.0
        valid, msg = inputs.validate()
        assert valid is False
        assert "length" in msg.lower()

    def test_negative_roughness_fails(self) -> None:
        inputs = self._valid_inputs()
        inputs.pipe_roughness = -0.001
        valid, msg = inputs.validate()
        assert valid is False
        assert "roughness" in msg.lower()

    def test_zero_flow_fails(self) -> None:
        inputs = self._valid_inputs()
        inputs.mass_flow_rate = 0.0
        valid, msg = inputs.validate()
        assert valid is False

    def test_zero_pressure_fails(self) -> None:
        inputs = self._valid_inputs()
        inputs.inlet_pressure = 0.0
        valid, msg = inputs.validate()
        assert valid is False

    def test_zero_temperature_fails(self) -> None:
        inputs = self._valid_inputs()
        inputs.inlet_temperature = 0.0
        valid, msg = inputs.validate()
        assert valid is False

    def test_default_fittings_empty(self) -> None:
        inputs = self._valid_inputs()
        assert inputs.fittings == []

    def test_default_friction_method(self) -> None:
        inputs = self._valid_inputs()
        assert inputs.friction_method == "colebrook"


# ─── PipeSpecification Tests ─────────────────────────────────


class TestPipeSpecification:
    def test_get_id_meters(self) -> None:
        pipe = PipeSpecification(
            nominal_size="4",
            schedule="40",
            outer_diameter=114.3,
            wall_thickness=6.02,
            inner_diameter=102.26,
        )
        assert abs(pipe.get_id_meters() - 0.10226) < 1e-6

    def test_get_od_meters(self) -> None:
        pipe = PipeSpecification(
            nominal_size="4",
            schedule="40",
            outer_diameter=114.3,
            wall_thickness=6.02,
            inner_diameter=102.26,
        )
        assert abs(pipe.get_od_meters() - 0.1143) < 1e-6

    def test_default_material(self) -> None:
        pipe = PipeSpecification(
            nominal_size="2",
            schedule="40",
            outer_diameter=60.3,
            wall_thickness=3.91,
            inner_diameter=52.5,
        )
        assert pipe.material == "Carbon Steel"

    def test_max_pressure_default_none(self) -> None:
        pipe = PipeSpecification(
            nominal_size="2",
            schedule="40",
            outer_diameter=60.3,
            wall_thickness=3.91,
            inner_diameter=52.5,
        )
        assert pipe.max_pressure is None


# ─── PressureDropResults Tests ───────────────────────────────


class TestPressureDropResults:
    def _make_flow_props(self) -> FlowProperties:
        return FlowProperties(
            density=1.2,
            viscosity=1.8e-5,
            velocity=20.0,
            reynolds_number=150000.0,
            mach_number=0.06,
            compressibility_factor=1.0,
            molecular_weight=28.0,
            mass_flux=24.0,
            volumetric_flow_rate=0.5,
        )

    def test_to_dict_keys(self) -> None:
        result = PressureDropResults(
            total_pressure_drop=5000.0,
            outlet_pressure=495000.0,
            friction_pressure_drop=4000.0,
            fitting_pressure_drop=800.0,
            elevation_pressure_drop=100.0,
            acceleration_pressure_drop=100.0,
            friction_factor=0.02,
            flow_properties=self._make_flow_props(),
            pressure_drop_per_100ft=1500.0,
            velocity_pressure=240.0,
            erosional_velocity=90.0,
            erosion_ratio=0.22,
            flow_regime="turbulent",
        )
        d = result.to_dict()
        assert "Pressure Drop (Pa)" in d
        assert "Pressure Drop (bar)" in d
        assert "Pressure Drop (psi)" in d
        assert "Flow Regime" in d
        assert "Reynolds Number" in d

    def test_to_dict_unit_conversion(self) -> None:
        result = PressureDropResults(
            total_pressure_drop=100000.0,
            outlet_pressure=400000.0,
            friction_pressure_drop=90000.0,
            fitting_pressure_drop=10000.0,
            elevation_pressure_drop=0.0,
            acceleration_pressure_drop=0.0,
            friction_factor=0.02,
            flow_properties=self._make_flow_props(),
            pressure_drop_per_100ft=3000.0,
            velocity_pressure=240.0,
            erosional_velocity=90.0,
            erosion_ratio=0.22,
            flow_regime="turbulent",
        )
        d = result.to_dict()
        assert abs(d["Pressure Drop (bar)"] - 1.0) < 0.01  # 100000 Pa = 1 bar


# ─── FlowRateInput Tests ─────────────────────────────────────


class TestFlowRateInput:
    def test_construction(self) -> None:
        fri = FlowRateInput(value=10.0, unit="kg/s")
        assert fri.value == 10.0
        assert fri.unit == "kg/s"

    def test_default_reference_conditions(self) -> None:
        fri = FlowRateInput(value=10.0, unit="kg/s")
        assert fri.reference_temperature == 273.15
        assert fri.reference_pressure == 101325.0

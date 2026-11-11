"""Static CalculatorSpec metadata (handlers attached at router import)."""

from __future__ import annotations

from typing import Any

from ..contracts.acid_gas_dewpoint import (
    AcidGasDewpointRequest,
    AcidGasDewpointResponse,
)
from ..contracts.baghouse import BaghouseRequest, BaghouseResponse
from ..contracts.financial import FinancialRequest, FinancialResponse
from ..contracts.flare import FlareRequest, FlareResponse
from ..contracts.flow_rate import FlowRateConvertRequest, FlowRateConvertResponse
from ..contracts.ode_solver import ODESolverRequest, ODESolverResponse
from ..contracts.rotation_converter import (
    ReferenceFrameConversionRequest,
    ReferenceFrameConversionResponse,
    RotationConverterRequest,
    RotationConverterResponse,
)
from ..contracts.scrubber import ScrubberRequest, ScrubberResponse
from ..contracts.symbolic_solver import (
    SymbolicDerivativeRequest,
    SymbolicDerivativeResponse,
    SymbolicSimplifyRequest,
    SymbolicSimplifyResponse,
    SymbolicSolveRequest,
    SymbolicSolveResponse,
)
from ..contracts.syngas_water import SyngasWaterRequest, SyngasWaterResponse
from ..contracts.thermal_profile import ThermalProfileRequest, ThermalProfileResponse
from ..contracts.wgs_reactor import WGSReactorRequest, WGSReactorResponse
from ..models.pressure_drop import PressureDropRequest
from .spec import WorkedExample

CatalogEntry = dict[str, Any]

CATALOG: dict[str, CatalogEntry] = {
    "acid-gas-dewpoint": {
        "id": "acid-gas-dewpoint",
        "title": "Acid gas dew point",
        "category": "process",
        "request_model": AcidGasDewpointRequest,
        "response_model": AcidGasDewpointResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {"temperature_c": "degC", "pressure_bar": "bar"},
        "worked_example": WorkedExample(
            request={"temperature_c": 150.0, "pressure_bar": 1.0, "method": "antoine"},
            response_summary={"condensation_risk": "Low"},
        ),
        "route_subpath": "",
    },
    "baghouse": {
        "id": "baghouse",
        "title": "Baghouse sizing",
        "category": "process",
        "request_model": BaghouseRequest,
        "response_model": BaghouseResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {
            "gas_flow_kg_s": "kg/s",
            "inlet_temp_k": "K",
            "pressure_pa": "Pa",
        },
        "worked_example": WorkedExample(
            request={
                "gas_flow_kg_s": 5.0,
                "inlet_temp_k": 400.0,
                "pressure_pa": 101325.0,
                "particulate_loading_kg_m3": 0.01,
            },
            response_summary={"filtration_area_m2": "> 0"},
        ),
        "route_subpath": "",
    },
    "financial": {
        "id": "financial",
        "title": "Financial analysis",
        "category": "economics",
        "request_model": FinancialRequest,
        "response_model": FinancialResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {"capital_cost": "USD", "annual_revenue": "USD/yr"},
        "worked_example": WorkedExample(
            request={
                "capital_cost": 1_000_000.0,
                "annual_revenue": 250_000.0,
                "annual_operating_cost": 120_000.0,
                "discount_rate": 0.08,
                "project_life_years": 15,
            },
            response_summary={"npv": "finite"},
        ),
        "route_subpath": "",
    },
    "flare": {
        "id": "flare",
        "title": "Flare design",
        "category": "process",
        "request_model": FlareRequest,
        "response_model": FlareResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {
            "total_flow_kg_hr": "kg/hr",
            "temperature_k": "K",
            "pressure_bar": "bar",
        },
        "worked_example": WorkedExample(
            request={
                "total_flow_kg_hr": 5000.0,
                "gas_composition": {"H2": 30, "CO": 40, "CH4": 10, "N2": 20},
                "temperature_k": 400.0,
                "pressure_bar": 1.5,
            },
            response_summary={"design.height_m": "> 0"},
        ),
        "route_subpath": "",
    },
    "flow-rate": {
        "id": "flow-rate",
        "title": "Flow rate unit conversion",
        "category": "conversion",
        "request_model": FlowRateConvertRequest,
        "response_model": FlowRateConvertResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {"value": "varies", "from_unit": "—", "to_unit": "—"},
        "worked_example": WorkedExample(
            request={
                "value": 100.0,
                "from_unit": "kg/h",
                "to_unit": "kg/s",
                "category": "mass",
            },
            response_summary={"result": 0.027777777777777776},
        ),
        "route_subpath": "",
    },
    "ode-solver": {
        "id": "ode-solver",
        "title": "ODE solver",
        "category": "numerics",
        "request_model": ODESolverRequest,
        "response_model": ODESolverResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {"t_start": "s", "t_end": "s"},
        "worked_example": WorkedExample(
            request={
                "derivatives": {"y": "-0.1*y"},
                "initial_conditions": {"y": 100.0},
                "parameters": {},
                "t_start": 0.0,
                "t_end": 10.0,
                "num_points": 50,
            },
            response_summary={"success": False},
        ),
        "route_subpath": "",
    },
    "pressure-drop": {
        "id": "pressure-drop",
        "title": "Pressure drop (Darcy-Weisbach)",
        "category": "process",
        "request_model": PressureDropRequest,
        "response_model": None,
        "maturity": "stable",
        "formula_ids": (),
        "units": {
            "pipe_diameter_m": "m",
            "pipe_length_m": "m",
            "flow_rate_kg_s": "kg/s",
            "temperature_k": "K",
            "pressure_pa": "Pa",
        },
        "worked_example": WorkedExample(
            request={
                "pipe_diameter_m": 0.1,
                "pipe_length_m": 100.0,
                "roughness_m": 4.5e-05,
                "flow_rate_kg_s": 5.0,
                "temperature_k": 300.0,
                "pressure_pa": 101325.0,
                "molecular_weight_kg_mol": 28.97,
            },
            response_summary={"status": "success"},
        ),
        "route_subpath": "",
    },
    "rotation-converter": {
        "id": "rotation-converter",
        "title": "Rotation representation converter",
        "category": "kinematics",
        "request_model": RotationConverterRequest,
        "response_model": RotationConverterResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {"type": "—", "value": "varies"},
        "worked_example": WorkedExample(
            request={
                "type": "quaternion",
                "value": [1.0, 0.0, 0.0, 0.0],
                "output_type": "euler",
                "euler_convention": "xyz",
            },
            response_summary={"operation": "quaternion_to_euler"},
        ),
        "route_subpath": "",
    },
    "rotation-reference-frame": {
        "id": "rotation-reference-frame",
        "title": "Reference frame rotation conversion",
        "category": "kinematics",
        "request_model": ReferenceFrameConversionRequest,
        "response_model": ReferenceFrameConversionResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {"operation": "—"},
        "worked_example": WorkedExample(
            request={
                "operation": "direction_cosine_to_quaternion",
                "value": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
            response_summary={"result_type": "quaternion"},
        ),
        "route_subpath": "reference-frame",
    },
    "scrubber": {
        "id": "scrubber",
        "title": "Scrubber design",
        "category": "process",
        "request_model": ScrubberRequest,
        "response_model": ScrubberResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {
            "gas_flow_kg_hr": "kg/hr",
            "gas_temperature_k": "K",
            "gas_pressure_pa": "Pa",
        },
        "worked_example": WorkedExample(
            request={
                "gas_flow_kg_hr": 5000.0,
                "gas_temperature_k": 350.0,
                "gas_pressure_pa": 101325.0,
                "pollutant_concentration_ppm": 500.0,
                "removal_efficiency": 0.95,
            },
            response_summary={"tower_diameter_m": "> 0"},
        ),
        "route_subpath": "",
    },
    "symbolic-solve": {
        "id": "symbolic-solve",
        "title": "Symbolic equation solve",
        "category": "symbolic",
        "request_model": SymbolicSolveRequest,
        "response_model": SymbolicSolveResponse,
        "maturity": "beta",
        "formula_ids": (),
        "units": {"equation": "—", "variable": "—"},
        "worked_example": WorkedExample(
            request={"equation": "x**2 - 4 = 0", "variable": "x"},
            response_summary={"solutions": ["-2", "2"]},
        ),
        "route_subpath": "solve",
    },
    "symbolic-derivative": {
        "id": "symbolic-derivative",
        "title": "Symbolic derivative",
        "category": "symbolic",
        "request_model": SymbolicDerivativeRequest,
        "response_model": SymbolicDerivativeResponse,
        "maturity": "beta",
        "formula_ids": (),
        "units": {"expression": "—", "variable": "—"},
        "worked_example": WorkedExample(
            request={"expression": "x**2", "variable": "x"},
            response_summary={"derivative": "2*x"},
        ),
        "route_subpath": "derivative",
    },
    "symbolic-simplify": {
        "id": "symbolic-simplify",
        "title": "Symbolic simplify",
        "category": "symbolic",
        "request_model": SymbolicSimplifyRequest,
        "response_model": SymbolicSimplifyResponse,
        "maturity": "beta",
        "formula_ids": (),
        "units": {"expression": "—"},
        "worked_example": WorkedExample(
            request={"expression": "x + x"},
            response_summary={"simplified": "2*x"},
        ),
        "route_subpath": "simplify",
    },
    "syngas-water": {
        "id": "syngas-water",
        "title": "Syngas water content",
        "category": "process",
        "request_model": SyngasWaterRequest,
        "response_model": SyngasWaterResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {"temperature_c": "C", "pressure_bar": "bar"},
        "worked_example": WorkedExample(
            request={
                "temperature_c": 200.0,
                "pressure_bar": 30.0,
                "composition_key": "typical_syngas",
            },
            response_summary={"status": "SUCCEEDED"},
        ),
        "route_subpath": "",
    },
    "thermal-profile": {
        "id": "thermal-profile",
        "title": "Thermal profile prediction",
        "category": "process",
        "request_model": ThermalProfileRequest,
        "response_model": ThermalProfileResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {"thermal_mass_j_per_k": "J/K", "duration_s": "s"},
        "worked_example": WorkedExample(
            request={
                "thermal_mass_j_per_k": 1_000_000.0,
                "initial_temperature_c": 25.0,
                "heat_input_w": 50_000.0,
                "duration_s": 3600.0,
                "ambient_temperature_c": 20.0,
            },
            response_summary={"final_temperature_c": "> initial"},
        ),
        "route_subpath": "",
    },
    "wgs-reactor": {
        "id": "wgs-reactor",
        "title": "WGS reactor equilibrium",
        "category": "process",
        "request_model": WGSReactorRequest,
        "response_model": WGSReactorResponse,
        "maturity": "stable",
        "formula_ids": (),
        "units": {"temperature_k": "K", "pressure_bar": "bar"},
        "worked_example": WorkedExample(
            request={
                "temperature_k": 500.0,
                "pressure_bar": 20.0,
                "inlet_composition": {"CO": 0.3, "H2O": 0.3, "CO2": 0.2, "H2": 0.2},
            },
            response_summary={"equilibrium_constants": "present"},
        ),
        "route_subpath": "",
    },
}

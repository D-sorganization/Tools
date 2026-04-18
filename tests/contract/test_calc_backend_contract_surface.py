"""Contract tests for downstream-critical calc backend API models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from calc_backend.contracts.ode_solver import ODESolverRequest, ODESolverResponse
from calc_backend.contracts.pressure_drop import (
    PressureDropRequest,
    PressureDropResponse,
)
from calc_backend.contracts.rotation_converter import (
    ReferenceFrameConversionRequest,
    ReferenceFrameConversionResponse,
    RotationConverterRequest,
    RotationConverterResponse,
)


@pytest.mark.contract
@pytest.mark.parametrize(
    ("model", "expected_fields"),
    [
        (
            ODESolverRequest,
            {
                "derivatives",
                "parameters",
                "initial_conditions",
                "t_start",
                "t_end",
                "num_points",
            },
        ),
        (
            ODESolverResponse,
            {"times", "solutions", "variable_summaries", "success", "message"},
        ),
        (
            PressureDropRequest,
            {
                "pipe_diameter_m",
                "pipe_length_m",
                "roughness_m",
                "flow_rate_kg_s",
                "temperature_k",
                "pressure_pa",
                "molecular_weight_kg_mol",
            },
        ),
        (
            PressureDropResponse,
            {
                "pressure_drop_pa",
                "reynolds_number",
                "friction_factor",
                "velocity_m_s",
                "flow_regime",
                "density_kg_m3",
                "viscosity_pa_s",
            },
        ),
        (
            RotationConverterRequest,
            {"type", "value", "euler_convention"},
        ),
        (
            RotationConverterResponse,
            {"representations"},
        ),
        (
            ReferenceFrameConversionRequest,
            {
                "operation",
                "transform",
                "twist",
                "rotation_matrix",
                "translation",
                "so3_vector",
                "so3_matrix",
            },
        ),
        (
            ReferenceFrameConversionResponse,
            {
                "operation",
                "results",
                "explanation_markdown",
                "explanation_latex",
            },
        ),
    ],
)
def test_calc_backend_contract_fields_are_stable(model, expected_fields):
    """Public request/response contracts keep downstream-visible field names."""
    assert set(model.model_fields) == expected_fields


@pytest.mark.contract
def test_ode_solver_request_rejects_too_few_output_points():
    with pytest.raises(ValidationError):
        ODESolverRequest(
            derivatives={"y": "-k*y"},
            parameters={"k": 0.1},
            initial_conditions={"y": 1.0},
            num_points=1,
        )


@pytest.mark.contract
def test_pressure_drop_request_rejects_non_positive_pipe_diameter():
    with pytest.raises(ValidationError):
        PressureDropRequest(
            pipe_diameter_m=0.0,
            pipe_length_m=10.0,
            flow_rate_kg_s=1.0,
            temperature_k=300.0,
            pressure_pa=101325.0,
            molecular_weight_kg_mol=0.028,
        )


@pytest.mark.contract
def test_rotation_converter_request_rejects_unknown_representation():
    with pytest.raises(ValidationError):
        RotationConverterRequest(type="matrix", value=[1.0, 0.0, 0.0, 0.0])


@pytest.mark.contract
def test_reference_frame_request_requires_operation_specific_payload():
    with pytest.raises(ValidationError):
        ReferenceFrameConversionRequest(operation="twist_frame_conversion")

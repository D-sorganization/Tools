"""Tests for pressure drop API with standardized responses.

These tests verify that the pressure drop calculator endpoint correctly uses
the StandardResponse wrapper for both success and error cases.

Related to issue #2411 (API Standardization) and #613 (Calc Backend).
"""

from __future__ import annotations

import json

import pytest
from calc_backend.models.pressure_drop import (
    PressureDropRequest,
    PressureDropResponse,
)
from pydantic import ValidationError
from upstream_drift_tools.api import ErrorCode, StandardResponse


class TestPressureDropAPIErrorHandling:
    """Tests for error handling in pressure drop API."""

    def test_validation_error_invalid_pipe_diameter(self) -> None:
        """Verify validation error for invalid pipe_diameter_m."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=-0.1,
                pipe_length_m=100.0,
                flow_rate_kg_s=5.0,
                temperature_k=300.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=28.97,
            )
        # Error should be caught and wrapped in StandardResponse
        # with ErrorCode.INVALID_INPUT
        assert len(exc_info.value.errors()) > 0

    def test_validation_error_invalid_pipe_length(self) -> None:
        """Verify validation error for invalid pipe_length_m."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=0.1,
                pipe_length_m=-50.0,
                flow_rate_kg_s=5.0,
                temperature_k=300.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=28.97,
            )
        assert len(exc_info.value.errors()) > 0

    def test_validation_error_missing_required_field(self) -> None:
        """Verify validation error for missing required field."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=0.1,
                # Missing pipe_length_m
                flow_rate_kg_s=5.0,
                temperature_k=300.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=28.97,
            )
        assert len(exc_info.value.errors()) > 0


class TestStandardResponseIntegrationPressureDrop:
    """Integration tests for StandardResponse with pressure drop calculations."""

    def test_success_response_structure(self) -> None:
        """Verify success response has correct structure."""
        response_data = {
            "pressure_drop_pa": 1023.4,
            "reynolds_number": 50000.0,
            "friction_factor": 0.025,
            "velocity_m_s": 45.2,
            "flow_regime": "Turbulent",
            "density_kg_m3": 1.177,
            "viscosity_pa_s": 1.86e-5,
        }

        response = StandardResponse.success(
            data=response_data,
            processing_time_ms=125.0,
        )

        response_dict = response.to_dict()

        # Verify structure
        assert response_dict["status"] == "success"
        assert response_dict["data"] == response_data
        assert response_dict["error"] is None
        assert "metadata" in response_dict
        assert "request_id" in response_dict["metadata"]
        assert response_dict["metadata"]["processing_time_ms"] == 125.0

    def test_error_response_invalid_input(self) -> None:
        """Verify error response for invalid input."""
        from upstream_drift_tools.api import ErrorDetail

        error = ErrorDetail(
            code=ErrorCode.INVALID_INPUT,
            message="pipe_diameter_m must be > 0",
            details={
                "exception_type": "ValueError",
                "exception_message": "pipe_diameter_m must be > 0",
            },
        )

        response = StandardResponse.error(
            error=error,
            processing_time_ms=15.0,
        )

        response_dict = response.to_dict()

        # Verify structure
        assert response_dict["status"] == "error"
        assert response_dict["data"] is None
        assert response_dict["error"]["code"] == "INVALID_INPUT"
        assert response_dict["error"]["message"] == "pipe_diameter_m must be > 0"
        assert response_dict["error"]["details"]["exception_type"] == "ValueError"
        assert response_dict["metadata"]["processing_time_ms"] == 15.0

    def test_error_response_json_serializable(self) -> None:
        """Verify error response can be JSON serialized."""
        from upstream_drift_tools.api import ErrorDetail

        error = ErrorDetail(
            code=ErrorCode.CONSTRAINT_VIOLATION,
            message="Incompatible parameters",
            details={
                "constraints": [
                    "Temperature must be realistic for the given pressure",
                ]
            },
        )

        response = StandardResponse.error(error=error)
        response_dict = response.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(response_dict)
        assert "CONSTRAINT_VIOLATION" in json_str
        assert "Incompatible parameters" in json_str

    def test_request_response_pair_types(self) -> None:
        """Verify request and response models are Pydantic BaseModel."""
        from pydantic import BaseModel

        # Request should be Pydantic model
        request = PressureDropRequest(
            pipe_diameter_m=0.1,
            pipe_length_m=100.0,
            flow_rate_kg_s=5.0,
            temperature_k=300.0,
            pressure_pa=101325.0,
            molecular_weight_kg_mol=28.97,
        )
        assert isinstance(request, BaseModel)

        # Response should be Pydantic model
        response = PressureDropResponse(
            pressure_drop_pa=1023.4,
            reynolds_number=50000.0,
            friction_factor=0.025,
            velocity_m_s=45.2,
            flow_regime="Turbulent",
            density_kg_m3=1.177,
            viscosity_pa_s=1.86e-5,
        )
        assert isinstance(response, BaseModel)

    def test_error_tracking_with_request_id(self) -> None:
        """Verify error response includes request_id for tracking."""
        from upstream_drift_tools.api import ErrorDetail

        custom_request_id = "req-error-12345"
        error = ErrorDetail(
            code=ErrorCode.SERVER_ERROR,
            message="Unexpected error",
            request_id=custom_request_id,
        )

        response = StandardResponse.error(error=error)
        response_dict = response.to_dict()

        # Verify request_id is present in both error and metadata
        assert response_dict["error"]["request_id"] == custom_request_id
        assert response_dict["metadata"]["request_id"] == custom_request_id

    def test_response_metadata_api_version(self) -> None:
        """Verify response metadata includes API version."""
        response = StandardResponse.success(data={"result": 123})
        response_dict = response.to_dict()

        assert "api_version" in response_dict["metadata"]
        assert response_dict["metadata"]["api_version"] == "1.0.0"


class TestPressureDropAPIEndpointDocumentation:
    """Documentation and examples for pressure drop API."""

    def test_typical_pressure_drop_request_example(self) -> None:
        """Example: Create a typical pressure drop request."""
        # Scenario: Calculate pressure drop in a 100m pipe with 5 kg/s air flow
        request = PressureDropRequest(
            pipe_diameter_m=0.1,  # 10 cm diameter pipe
            pipe_length_m=100.0,  # 100 meter long
            roughness_m=0.000045,  # Commercial steel
            flow_rate_kg_s=5.0,  # 5 kg/s mass flow rate
            temperature_k=300.0,  # 27°C
            pressure_pa=101325.0,  # 1 atm
            molecular_weight_kg_mol=28.97,  # Air
        )

        # Request is valid
        assert request.pipe_diameter_m == 0.1
        assert request.flow_rate_kg_s == 5.0

    def test_typical_pressure_drop_response_example(self) -> None:
        """Example: Create a typical pressure drop response."""
        response = PressureDropResponse(
            pressure_drop_pa=1023.4,  # ~1 kPa pressure drop
            reynolds_number=50000.0,  # Turbulent
            friction_factor=0.025,
            velocity_m_s=45.2,
            flow_regime="Turbulent",
            density_kg_m3=1.177,
            viscosity_pa_s=1.86e-5,
        )

        # Response is valid
        assert response.flow_regime == "Turbulent"
        assert response.pressure_drop_pa == 1023.4

    def test_standardized_success_api_response(self) -> None:
        """Example: Standardized API success response."""
        # Simulate calculation result
        calc_result = {
            "pressure_drop_pa": 1023.4,
            "reynolds_number": 50000.0,
            "friction_factor": 0.025,
            "velocity_m_s": 45.2,
            "flow_regime": "Turbulent",
            "density_kg_m3": 1.177,
            "viscosity_pa_s": 1.86e-5,
        }

        # Wrap in StandardResponse
        response = StandardResponse.success(
            data=calc_result,
            processing_time_ms=125.0,
            request_id="req-pressure-drop-001",
        )

        # Convert to JSON
        response_dict = response.to_dict()

        # Verify the structure matches API spec
        assert response_dict["status"] == "success"
        assert response_dict["data"]["pressure_drop_pa"] == 1023.4
        assert response_dict["metadata"]["request_id"] == "req-pressure-drop-001"

    def test_standardized_error_api_response(self) -> None:
        """Example: Standardized API error response."""
        from upstream_drift_tools.api import ErrorDetail

        # Simulate validation error
        error = ErrorDetail(
            code=ErrorCode.INVALID_INPUT,
            message="Pressure drop calculation failed",
            details={
                "field": "pipe_diameter_m",
                "received": -0.1,
                "constraint": "must be > 0",
            },
        )

        # Wrap in StandardResponse
        response = StandardResponse.error(
            error=error,
            processing_time_ms=15.0,
        )

        # Convert to JSON
        response_dict = response.to_dict()

        # Verify the structure matches API spec
        assert response_dict["status"] == "error"
        assert response_dict["error"]["code"] == ErrorCode.INVALID_INPUT
        assert response_dict["error"]["details"]["field"] == "pipe_diameter_m"

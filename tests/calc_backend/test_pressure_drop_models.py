"""Tests for pressure drop calculator Pydantic models.

These tests verify that PressureDropRequest and PressureDropResponse models
correctly validate inputs and format outputs according to specifications.

Related to issue #2411 (API Standardization).
"""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from calc_backend.models.pressure_drop import (
    PressureDropRequest,
    PressureDropResponse,
)


class TestPressureDropRequest:
    """Tests for PressureDropRequest model validation."""

    def test_valid_request_all_fields(self) -> None:
        """Verify valid request with all required fields."""
        request = PressureDropRequest(
            pipe_diameter_m=0.1,
            pipe_length_m=100.0,
            roughness_m=0.000045,
            flow_rate_kg_s=5.0,
            temperature_k=300.0,
            pressure_pa=101325.0,
            molecular_weight_kg_mol=28.97,
            viscosity_pa_s=1.86e-5,
        )
        assert request.pipe_diameter_m == 0.1
        assert request.pipe_length_m == 100.0
        assert request.flow_rate_kg_s == 5.0

    def test_valid_request_without_optional_viscosity(self) -> None:
        """Verify valid request without optional viscosity field."""
        request = PressureDropRequest(
            pipe_diameter_m=0.05,
            pipe_length_m=50.0,
            flow_rate_kg_s=2.5,
            temperature_k=300.0,
            pressure_pa=101325.0,
            molecular_weight_kg_mol=28.97,
        )
        assert request.viscosity_pa_s is None

    def test_default_roughness(self) -> None:
        """Verify roughness defaults to 0.000045 (commercial steel)."""
        request = PressureDropRequest(
            pipe_diameter_m=0.1,
            pipe_length_m=100.0,
            flow_rate_kg_s=5.0,
            temperature_k=300.0,
            pressure_pa=101325.0,
            molecular_weight_kg_mol=28.97,
        )
        assert request.roughness_m == 0.000045

    def test_invalid_negative_pipe_diameter(self) -> None:
        """Verify negative pipe_diameter_m is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=-0.1,
                pipe_length_m=100.0,
                flow_rate_kg_s=5.0,
                temperature_k=300.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=28.97,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_invalid_zero_pipe_diameter(self) -> None:
        """Verify zero pipe_diameter_m is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=0.0,
                pipe_length_m=100.0,
                flow_rate_kg_s=5.0,
                temperature_k=300.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=28.97,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_invalid_negative_pipe_length(self) -> None:
        """Verify negative pipe_length_m is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=0.1,
                pipe_length_m=-50.0,
                flow_rate_kg_s=5.0,
                temperature_k=300.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=28.97,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_invalid_negative_roughness(self) -> None:
        """Verify negative roughness_m is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=0.1,
                pipe_length_m=100.0,
                roughness_m=-0.000045,
                flow_rate_kg_s=5.0,
                temperature_k=300.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=28.97,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_valid_zero_roughness(self) -> None:
        """Verify zero roughness is valid (smooth pipe)."""
        request = PressureDropRequest(
            pipe_diameter_m=0.1,
            pipe_length_m=100.0,
            roughness_m=0.0,
            flow_rate_kg_s=5.0,
            temperature_k=300.0,
            pressure_pa=101325.0,
            molecular_weight_kg_mol=28.97,
        )
        assert request.roughness_m == 0.0

    def test_invalid_negative_flow_rate(self) -> None:
        """Verify negative flow_rate_kg_s is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=0.1,
                pipe_length_m=100.0,
                flow_rate_kg_s=-5.0,
                temperature_k=300.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=28.97,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_invalid_zero_temperature(self) -> None:
        """Verify zero temperature_k is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=0.1,
                pipe_length_m=100.0,
                flow_rate_kg_s=5.0,
                temperature_k=0.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=28.97,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_invalid_negative_pressure(self) -> None:
        """Verify negative pressure_pa is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=0.1,
                pipe_length_m=100.0,
                flow_rate_kg_s=5.0,
                temperature_k=300.0,
                pressure_pa=-101325.0,
                molecular_weight_kg_mol=28.97,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_invalid_negative_molecular_weight(self) -> None:
        """Verify negative molecular_weight_kg_mol is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=0.1,
                pipe_length_m=100.0,
                flow_rate_kg_s=5.0,
                temperature_k=300.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=-28.97,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_invalid_negative_viscosity(self) -> None:
        """Verify negative viscosity_pa_s is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PressureDropRequest(
                pipe_diameter_m=0.1,
                pipe_length_m=100.0,
                flow_rate_kg_s=5.0,
                temperature_k=300.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=28.97,
                viscosity_pa_s=-1.86e-5,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_missing_required_pipe_diameter(self) -> None:
        """Verify missing pipe_diameter_m is rejected."""
        with pytest.raises(ValidationError):
            PressureDropRequest(
                pipe_length_m=100.0,
                flow_rate_kg_s=5.0,
                temperature_k=300.0,
                pressure_pa=101325.0,
                molecular_weight_kg_mol=28.97,
            )

    def test_request_json_serializable(self) -> None:
        """Verify request can be serialized to JSON."""
        request = PressureDropRequest(
            pipe_diameter_m=0.1,
            pipe_length_m=100.0,
            flow_rate_kg_s=5.0,
            temperature_k=300.0,
            pressure_pa=101325.0,
            molecular_weight_kg_mol=28.97,
        )
        # Should not raise
        json_str = json.dumps(request.model_dump())
        assert "0.1" in json_str
        assert "100.0" in json_str

    def test_request_model_dump(self) -> None:
        """Verify request.model_dump() produces correct dict."""
        request = PressureDropRequest(
            pipe_diameter_m=0.1,
            pipe_length_m=100.0,
            roughness_m=0.000045,
            flow_rate_kg_s=5.0,
            temperature_k=300.0,
            pressure_pa=101325.0,
            molecular_weight_kg_mol=28.97,
        )
        data = request.model_dump()
        assert data["pipe_diameter_m"] == 0.1
        assert data["pipe_length_m"] == 100.0
        assert data["roughness_m"] == 0.000045


class TestPressureDropResponse:
    """Tests for PressureDropResponse model."""

    def test_valid_response_turbulent(self) -> None:
        """Verify valid response for turbulent flow."""
        response = PressureDropResponse(
            pressure_drop_pa=1023.4,
            reynolds_number=50000.0,
            friction_factor=0.025,
            velocity_m_s=45.2,
            flow_regime="Turbulent",
            density_kg_m3=1.177,
            viscosity_pa_s=1.86e-5,
        )
        assert response.pressure_drop_pa == 1023.4
        assert response.flow_regime == "Turbulent"

    def test_valid_response_laminar(self) -> None:
        """Verify valid response for laminar flow."""
        response = PressureDropResponse(
            pressure_drop_pa=50.0,
            reynolds_number=500.0,
            friction_factor=0.128,
            velocity_m_s=5.0,
            flow_regime="Laminar",
            density_kg_m3=1.2,
            viscosity_pa_s=2.0e-5,
        )
        assert response.flow_regime == "Laminar"

    def test_valid_response_transitional(self) -> None:
        """Verify valid response for transitional flow."""
        response = PressureDropResponse(
            pressure_drop_pa=200.0,
            reynolds_number=2300.0,
            friction_factor=0.08,
            velocity_m_s=10.0,
            flow_regime="Transitional",
            density_kg_m3=1.1,
            viscosity_pa_s=1.9e-5,
        )
        assert response.flow_regime == "Transitional"

    def test_response_with_very_small_values(self) -> None:
        """Verify response can handle very small values."""
        response = PressureDropResponse(
            pressure_drop_pa=1e-6,
            reynolds_number=1.5,
            friction_factor=1e-10,
            velocity_m_s=1e-8,
            flow_regime="Laminar",
            density_kg_m3=1e-3,
            viscosity_pa_s=1e-10,
        )
        assert response.pressure_drop_pa == 1e-6

    def test_response_with_very_large_values(self) -> None:
        """Verify response can handle very large values."""
        response = PressureDropResponse(
            pressure_drop_pa=1e7,
            reynolds_number=1e6,
            friction_factor=0.05,
            velocity_m_s=1000.0,
            flow_regime="Turbulent",
            density_kg_m3=1e6,
            viscosity_pa_s=1.0,
        )
        assert response.pressure_drop_pa == 1e7

    def test_response_json_serializable(self) -> None:
        """Verify response can be serialized to JSON."""
        response = PressureDropResponse(
            pressure_drop_pa=1023.4,
            reynolds_number=50000.0,
            friction_factor=0.025,
            velocity_m_s=45.2,
            flow_regime="Turbulent",
            density_kg_m3=1.177,
            viscosity_pa_s=1.86e-5,
        )
        # Should not raise
        json_str = json.dumps(response.model_dump())
        assert "1023.4" in json_str
        assert "Turbulent" in json_str

    def test_response_model_dump(self) -> None:
        """Verify response.model_dump() produces correct dict."""
        response = PressureDropResponse(
            pressure_drop_pa=1023.4,
            reynolds_number=50000.0,
            friction_factor=0.025,
            velocity_m_s=45.2,
            flow_regime="Turbulent",
            density_kg_m3=1.177,
            viscosity_pa_s=1.86e-5,
        )
        data = response.model_dump()
        assert data["pressure_drop_pa"] == 1023.4
        assert data["reynolds_number"] == 50000.0
        assert data["flow_regime"] == "Turbulent"

    def test_response_model_json_schema(self) -> None:
        """Verify response provides OpenAPI schema."""
        schema = PressureDropResponse.model_json_schema()
        assert "properties" in schema
        assert "pressure_drop_pa" in schema["properties"]
        assert "reynolds_number" in schema["properties"]
        assert "flow_regime" in schema["properties"]


class TestPressureDropIntegration:
    """Integration tests for request/response pair."""

    def test_request_response_roundtrip(self) -> None:
        """Verify request-response pair works together."""
        # Create and validate request
        request = PressureDropRequest(
            pipe_diameter_m=0.1,
            pipe_length_m=100.0,
            roughness_m=0.000045,
            flow_rate_kg_s=5.0,
            temperature_k=300.0,
            pressure_pa=101325.0,
            molecular_weight_kg_mol=28.97,
        )

        # Create corresponding response
        response = PressureDropResponse(
            pressure_drop_pa=1023.4,
            reynolds_number=50000.0,
            friction_factor=0.025,
            velocity_m_s=45.2,
            flow_regime="Turbulent",
            density_kg_m3=1.177,
            viscosity_pa_s=1.86e-5,
        )

        # Verify both can be serialized together
        request_json = request.model_dump_json()
        response_json = response.model_dump_json()
        assert "pipe_diameter_m" in request_json
        assert "pressure_drop_pa" in response_json

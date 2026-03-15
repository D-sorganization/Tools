"""Targeted coverage tests for remaining calc_backend gaps.

Covers first-party code only:
- thermal_profile._solve_thermal_profile: error path (lines 23-24), power_func
  fallback (line 43)
- wgs_reactor: equilibrium error path (lines 37-38), sizing error path (65-66),
  WGSReactorEngine None check (lines 22-26)
- contracts.rotation_converter: Pydantic model_validator branches (91-151)

NOTE: rotation_converter *router* is deliberately NOT tested here because it
wraps the deprecated `rotation_converter` package (an external dependency).
The model validators in *contracts* are pure first-party logic and safe to test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# thermal_profile router
# ---------------------------------------------------------------------------


class TestThermalProfileErrorPath:
    """Cover lines 23-24 (HTTPException from ArithmeticError) and line 43
    (power_func fallback when profile is not constant/linear_ramp/step).
    """

    def _valid_request(self, **overrides):
        from calc_backend.contracts.thermal_profile import ThermalProfileRequest

        kwargs = dict(
            initial_temp_c=20.0,
            ambient_temp_c=20.0,
            thermal_mass_j_per_k=500.0,
            heat_loss_coeff_w_per_k=5.0,
            power_w=1000.0,
            power_profile="constant",
            t_start_s=0.0,
            t_end_s=100.0,
            num_points=10,
        )
        kwargs.update(overrides)
        return ThermalProfileRequest(**kwargs)

    def test_arithmetic_error_raises_http_422(self):
        """Trigger the except-HTTPException branch (lines 23-24) in router."""
        from calc_backend.routers.thermal_profile import predict_thermal_profile

        req = self._valid_request()
        # Mock _solve_thermal_profile to raise ArithmeticError → caught by router
        with patch(
            "calc_backend.routers.thermal_profile._solve_thermal_profile",
            side_effect=ArithmeticError("division by zero"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                predict_thermal_profile(req)
        assert exc_info.value.status_code == 422
        assert "division by zero" in exc_info.value.detail

    def test_power_func_fallback_branch(self):
        """Cover line 43: power_func returns power_w for unknown profile."""
        from calc_backend.routers.thermal_profile import _solve_thermal_profile

        req = self._valid_request(power_profile="custom_unknown")
        # Should not raise; fallback just returns power_w for all t
        result = _solve_thermal_profile(req)
        assert result is not None
        assert len(result.data) == 10


# ---------------------------------------------------------------------------
# wgs_reactor router
# ---------------------------------------------------------------------------


class TestWGSReactorErrorPaths:
    """Cover equilibrium error path (lines 37-38) and sizing error path (65-66)."""

    def _minimal_payload(self):
        return {
            "inlet_composition": {"CO": 0.4, "H2O": 0.4, "CO2": 0.1, "H2": 0.1},
            "temperature_k": 700.0,
            "pressure_bar": 2.0,
            "steam_ratio": 2.0,
            "feed_rate_kmol_hr": 0.0,
        }

    def test_equilibrium_error_raises_422(self):
        """Lines 37-38: equilibrium ValueError → HTTPException 422."""
        from calc_backend.contracts.wgs_reactor import WGSReactorRequest

        req = WGSReactorRequest(**self._minimal_payload())

        mock_engine = MagicMock()
        mock_engine.calculate_equilibrium_composition.side_effect = ValueError(
            "bad equilibrium"
        )
        # WGSReactorEngine is imported inside the function; patch the source module
        with patch(
            "upstream_drift_tools.process_calculators.WGSReactorEngine",
            new=MagicMock(return_value=mock_engine),
        ):
            # Re-import triggers the local import inside calculate_wgs
            import importlib

            import calc_backend.routers.wgs_reactor as _m

            importlib.reload(_m)
            with pytest.raises(HTTPException) as exc_info:
                _m.calculate_wgs(req)
        assert exc_info.value.status_code == 422

    def test_sizing_error_raises_422(self):
        """Lines 65-66: sizing KeyError → HTTPException 422."""
        from calc_backend.contracts.wgs_reactor import WGSReactorRequest

        payload = self._minimal_payload()
        payload["feed_rate_kmol_hr"] = 100.0  # trigger sizing
        req = WGSReactorRequest(**payload)

        eq_result = {
            "conversion": 80.0,
            "composition": {"CO": 0.1, "H2O": 0.1, "CO2": 0.2, "H2": 0.6},
            "h2_co_ratio": 6.0,
            "equilibrium_constant": 4.5,
            "heat_released": 41.0,
        }
        mock_engine = MagicMock()
        mock_engine.calculate_equilibrium_composition.return_value = eq_result
        mock_engine.size_wgs_reactor.side_effect = KeyError("missing_key")

        with patch(
            "upstream_drift_tools.process_calculators.WGSReactorEngine",
            new=MagicMock(return_value=mock_engine),
        ):
            import importlib

            import calc_backend.routers.wgs_reactor as _m

            importlib.reload(_m)
            with pytest.raises(HTTPException) as exc_info:
                _m.calculate_wgs(req)
        assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# contracts.rotation_converter - Pydantic model validators (pure first-party)
# ---------------------------------------------------------------------------


class TestReferenceFrameConversionRequestValidator:
    """Cover the model_validator branches in contracts.rotation_converter (91-151)."""

    def _twist_frame_payload(self, **overrides):
        base = {
            "operation": "twist_frame_conversion",
            "transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "twist": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
        base.update(overrides)
        return base

    def _homogeneous_payload(self, **overrides):
        base = {
            "operation": "homogeneous_transform",
            "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0.0, 0.0, 1.0],
        }
        base.update(overrides)
        return base

    def test_twist_frame_conversion_missing_transform_raises(self):
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="transform and twist"):
            ReferenceFrameConversionRequest(
                operation="twist_frame_conversion",
                twist=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            )

    def test_twist_frame_conversion_extra_field_raises(self):
        """Providing rotation_matrix with twist_frame_conversion should fail."""
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="rotation_matrix"):
            ReferenceFrameConversionRequest(
                operation="twist_frame_conversion",
                transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                twist=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            )

    def test_twist_frame_conversion_valid_passes(self):
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        req = ReferenceFrameConversionRequest(**self._twist_frame_payload())
        assert req.operation == "twist_frame_conversion"

    def test_homogeneous_transform_missing_rotation_raises(self):
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="rotation_matrix and translation"):
            ReferenceFrameConversionRequest(
                operation="homogeneous_transform",
                translation=[0.0, 0.0, 1.0],
            )

    def test_homogeneous_transform_extra_field_raises(self):
        """Providing transform with homogeneous_transform should fail."""
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="transform, twist"):
            ReferenceFrameConversionRequest(
                operation="homogeneous_transform",
                rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                translation=[0.0, 0.0, 1.0],
                transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            )

    def test_homogeneous_transform_valid_passes(self):
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        req = ReferenceFrameConversionRequest(**self._homogeneous_payload())
        assert req.operation == "homogeneous_transform"

    def test_so3_maps_no_source_raises(self):
        """so3_so3_maps requires exactly one so3_vector, so3_matrix, or rotation_matrix."""
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="exactly one"):
            ReferenceFrameConversionRequest(operation="so3_so3_maps")

    def test_so3_maps_multiple_sources_raises(self):
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="exactly one"):
            ReferenceFrameConversionRequest(
                operation="so3_so3_maps",
                so3_vector=[0.0, 0.0, 1.0],
                so3_matrix=[[0, -1, 0], [1, 0, 0], [0, 0, 0]],
            )

    def test_so3_maps_extra_field_raises(self):
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="transform, twist, or translation"):
            ReferenceFrameConversionRequest(
                operation="so3_so3_maps",
                so3_vector=[0.0, 0.0, 1.0],
                transform=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            )

    def test_so3_maps_with_so3_vector_passes(self):
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        req = ReferenceFrameConversionRequest(
            operation="so3_so3_maps", so3_vector=[0.0, 0.0, 1.0]
        )
        assert req.so3_vector == [0.0, 0.0, 1.0]

    def test_so3_maps_with_rotation_matrix_passes(self):
        from calc_backend.contracts.rotation_converter import (
            ReferenceFrameConversionRequest,
        )

        req = ReferenceFrameConversionRequest(
            operation="so3_so3_maps",
            rotation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        assert req.rotation_matrix is not None

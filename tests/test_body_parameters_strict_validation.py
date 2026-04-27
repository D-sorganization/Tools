"""Tests for strict anthropometric validation contract (issue #1072).

Validates:
- validate_strict() exists and enforces documented bounds
- Negative and extreme values are rejected with PreconditionError
- All scaling factor fields are checked
- Valid parameters pass without error
"""

from __future__ import annotations

import pytest
from humanoid_character_builder.core.body_parameters import (
    ALL_FACTOR_BOUNDS,
    HEIGHT_RANGE_M,
    MASS_RANGE_KG,
    NORMALIZED_FACTOR_BOUNDS,
    PROPORTION_FACTOR_BOUNDS,
    BodyParameters,
)

from contracts import PreconditionError


class TestValidateStrictExists:
    """validate_strict() must exist and be callable."""

    def test_validate_strict_is_callable(self) -> None:
        """BodyParameters must have a callable validate_strict method."""
        params = BodyParameters()
        assert callable(getattr(params, "validate_strict", None))

    def test_validate_strict_returns_none(self) -> None:
        """validate_strict() returns None on valid input."""
        params = BodyParameters()
        result = params.validate_strict()
        assert result is None


class TestValidateStrictHeight:
    """Height validation in validate_strict()."""

    def test_negative_height_raises(self) -> None:
        """Negative height must raise PreconditionError."""
        params = BodyParameters.__new__(BodyParameters)
        object.__setattr__(params, "height_m", -1.0)
        object.__setattr__(params, "mass_kg", 75.0)
        _set_default_factors(params)
        with pytest.raises(PreconditionError, match="height_m"):
            params.validate_strict()

    def test_zero_height_raises(self) -> None:
        """Zero height must raise PreconditionError."""
        params = BodyParameters.__new__(BodyParameters)
        object.__setattr__(params, "height_m", 0.0)
        object.__setattr__(params, "mass_kg", 75.0)
        _set_default_factors(params)
        with pytest.raises(PreconditionError, match="height_m"):
            params.validate_strict()

    def test_height_below_range_raises(self) -> None:
        """Height below minimum range raises PreconditionError."""
        params = BodyParameters.__new__(BodyParameters)
        object.__setattr__(params, "height_m", HEIGHT_RANGE_M[0] - 0.01)
        object.__setattr__(params, "mass_kg", 75.0)
        _set_default_factors(params)
        with pytest.raises(PreconditionError, match="height_m"):
            params.validate_strict()

    def test_height_above_range_raises(self) -> None:
        """Height above maximum range raises PreconditionError."""
        params = BodyParameters.__new__(BodyParameters)
        object.__setattr__(params, "height_m", HEIGHT_RANGE_M[1] + 0.01)
        object.__setattr__(params, "mass_kg", 75.0)
        _set_default_factors(params)
        with pytest.raises(PreconditionError, match="height_m"):
            params.validate_strict()

    def test_height_at_lower_bound_passes(self) -> None:
        """Height at lower bound is valid."""
        params = BodyParameters(height_m=HEIGHT_RANGE_M[0], mass_kg=20.0)
        params.validate_strict()  # Should not raise

    def test_height_at_upper_bound_passes(self) -> None:
        """Height at upper bound is valid."""
        params = BodyParameters(height_m=HEIGHT_RANGE_M[1], mass_kg=75.0)
        params.validate_strict()  # Should not raise


class TestValidateStrictMass:
    """Mass validation in validate_strict()."""

    def test_negative_mass_raises(self) -> None:
        """Negative mass must raise PreconditionError."""
        params = BodyParameters.__new__(BodyParameters)
        object.__setattr__(params, "height_m", 1.75)
        object.__setattr__(params, "mass_kg", -10.0)
        _set_default_factors(params)
        with pytest.raises(PreconditionError, match="mass_kg"):
            params.validate_strict()

    def test_zero_mass_raises(self) -> None:
        """Zero mass must raise PreconditionError."""
        params = BodyParameters.__new__(BodyParameters)
        object.__setattr__(params, "height_m", 1.75)
        object.__setattr__(params, "mass_kg", 0.0)
        _set_default_factors(params)
        with pytest.raises(PreconditionError, match="mass_kg"):
            params.validate_strict()

    def test_mass_below_range_raises(self) -> None:
        """Mass below minimum range raises PreconditionError."""
        params = BodyParameters.__new__(BodyParameters)
        object.__setattr__(params, "height_m", 1.75)
        object.__setattr__(params, "mass_kg", MASS_RANGE_KG[0] - 0.01)
        _set_default_factors(params)
        with pytest.raises(PreconditionError, match="mass_kg"):
            params.validate_strict()

    def test_mass_above_range_raises(self) -> None:
        """Mass above maximum range raises PreconditionError."""
        params = BodyParameters.__new__(BodyParameters)
        object.__setattr__(params, "height_m", 1.75)
        object.__setattr__(params, "mass_kg", MASS_RANGE_KG[1] + 0.01)
        _set_default_factors(params)
        with pytest.raises(PreconditionError, match="mass_kg"):
            params.validate_strict()

    def test_mass_at_lower_bound_passes(self) -> None:
        """Mass at lower bound is valid."""
        params = BodyParameters(height_m=1.75, mass_kg=MASS_RANGE_KG[0])
        params.validate_strict()  # Should not raise

    def test_mass_at_upper_bound_passes(self) -> None:
        """Mass at upper bound is valid."""
        params = BodyParameters(height_m=1.75, mass_kg=MASS_RANGE_KG[1])
        params.validate_strict()  # Should not raise


class TestValidateStrictScalingFactors:
    """Scaling factor validation in validate_strict()."""

    @pytest.mark.parametrize(
        "field_name",
        list(ALL_FACTOR_BOUNDS.keys()),
        ids=list(ALL_FACTOR_BOUNDS.keys()),
    )
    def test_negative_factor_raises(self, field_name: str) -> None:
        """Negative scaling factor must raise PreconditionError."""
        params = BodyParameters()
        object.__setattr__(params, field_name, -0.1)
        with pytest.raises(PreconditionError, match=field_name):
            params.validate_strict()

    @pytest.mark.parametrize(
        "field_name,max_value",
        list(NORMALIZED_FACTOR_BOUNDS.items()),
        ids=list(NORMALIZED_FACTOR_BOUNDS.keys()),
    )
    def test_normalized_factor_above_max_raises(
        self, field_name: str, max_value: tuple[float, float]
    ) -> None:
        """Normalized factor above max (1.0) must raise PreconditionError."""
        params = BodyParameters()
        object.__setattr__(params, field_name, max_value[1] + 0.01)
        with pytest.raises(PreconditionError, match=field_name):
            params.validate_strict()

    @pytest.mark.parametrize(
        "field_name,max_value",
        list(PROPORTION_FACTOR_BOUNDS.items()),
        ids=list(PROPORTION_FACTOR_BOUNDS.keys()),
    )
    def test_proportion_factor_above_max_raises(
        self, field_name: str, max_value: tuple[float, float]
    ) -> None:
        """Proportion factor above hard limit (3.0) must raise PreconditionError."""
        params = BodyParameters()
        object.__setattr__(params, field_name, max_value[1] + 0.01)
        with pytest.raises(PreconditionError, match=field_name):
            params.validate_strict()

    @pytest.mark.parametrize(
        "field_name",
        list(ALL_FACTOR_BOUNDS.keys()),
        ids=list(ALL_FACTOR_BOUNDS.keys()),
    )
    def test_zero_factor_passes(self, field_name: str) -> None:
        """Zero is at the lower bound and should pass."""
        params = BodyParameters()
        object.__setattr__(params, field_name, 0.0)
        params.validate_strict()  # Should not raise

    def test_all_factors_covered(self) -> None:
        """All factor fields in BodyParameters are covered by validate_strict."""
        factor_fields = set(ALL_FACTOR_BOUNDS.keys())
        # Verify we have bounds defined for every factor
        params = BodyParameters()
        for field_name in factor_fields:
            assert hasattr(params, field_name), f"Missing factor: {field_name}"

    def test_coverage_matches_constants(self) -> None:
        """ALL_FACTOR_BOUNDS must include all normalized + proportion bounds."""
        expected = set(NORMALIZED_FACTOR_BOUNDS.keys()) | set(
            PROPORTION_FACTOR_BOUNDS.keys()
        )
        assert set(ALL_FACTOR_BOUNDS.keys()) == expected


class TestValidateStrictEdgeCases:
    """Edge cases and integration tests for validate_strict()."""

    def test_default_params_pass(self) -> None:
        """Default BodyParameters should pass strict validation."""
        params = BodyParameters()
        params.validate_strict()

    def test_extreme_but_valid_params_pass(self) -> None:
        """Parameters at extreme but valid range should pass."""
        params = BodyParameters(
            height_m=HEIGHT_RANGE_M[1],  # max height
            mass_kg=MASS_RANGE_KG[1],  # max mass
            muscularity=1.0,
            body_fat_factor=1.0,
            shoulder_width_factor=3.0,
            hip_width_factor=3.0,
            arm_length_factor=3.0,
            leg_length_factor=3.0,
            torso_length_factor=3.0,
            head_scale_factor=3.0,
            neck_length_factor=3.0,
            hand_scale_factor=3.0,
            foot_scale_factor=3.0,
        )
        params.validate_strict()  # Should not raise

    def test_factory_functions_pass_validation(self) -> None:
        """All factory functions should produce valid parameters."""
        from humanoid_character_builder.core.body_parameters import (
            create_athletic_body,
            create_average_body,
            create_heavy_body,
        )

        for factory in [create_athletic_body, create_average_body, create_heavy_body]:
            params = factory()
            params.validate_strict()  # Should not raise


# --- Helper ---


def _set_default_factors(params: BodyParameters) -> None:
    """Set default factor values on a BodyParameters instance created via __new__."""
    for field_name, (_, __) in ALL_FACTOR_BOUNDS.items():
        if not hasattr(params, field_name):
            object.__setattr__(params, field_name, 0.5)

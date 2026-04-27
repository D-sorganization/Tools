"""Tests for the mesh_generator sub-modules.

Covers the shared types (_mesh_types), PrimitiveMeshGenerator,
and the MeshGenerator factory.  MakeHuman and SMPL-X backends
are import-time optional so we only test their interface contracts.
"""

from __future__ import annotations

import pytest
import upstream_drift_tools.process_calculators.pressure_drop_calculator._pdi_unit_converters as _uc  # noqa: E501
import upstream_drift_tools.process_calculators.pressure_drop_calculator._pdi_validators as _val  # noqa: E501
import upstream_drift_tools.process_calculators.pressure_drop_calculator.pressure_drop_interface as _pdi  # noqa: E501
from humanoid_character_builder.generators._mesh_types import (
    GeneratedMeshResult,
    MeshGeneratorBackend,
    MeshGeneratorInterface,
)
from humanoid_character_builder.generators.mesh_generator import (
    MakeHumanMeshGenerator,
    MeshGenerator,
    PrimitiveMeshGenerator,
    SMPLXMeshGenerator,
)

# ---------------------------------------------------------------------------
# GeneratedMeshResult
# ---------------------------------------------------------------------------


class TestGeneratedMeshResult:
    """Unit tests for GeneratedMeshResult dataclass."""

    @pytest.mark.unit
    def test_default_fields(self):
        result = GeneratedMeshResult(success=True)
        assert result.success is True
        assert result.mesh_paths == {}
        assert result.collision_paths == {}
        assert result.texture_paths == {}
        assert result.vertex_groups == {}
        assert result.error_message is None
        assert result.metadata == {}

    @pytest.mark.unit
    def test_failed_result(self):
        result = GeneratedMeshResult(
            success=False, error_message="trimesh not available"
        )
        assert result.success is False
        assert result.error_message == "trimesh not available"


# ---------------------------------------------------------------------------
# MeshGeneratorBackend enum
# ---------------------------------------------------------------------------


class TestMeshGeneratorBackend:
    """Unit tests for MeshGeneratorBackend enum."""

    @pytest.mark.unit
    def test_all_values_are_strings(self):
        for member in MeshGeneratorBackend:
            assert isinstance(member.value, str)

    @pytest.mark.unit
    def test_primitive_value(self):
        assert MeshGeneratorBackend.PRIMITIVE.value == "primitive"

    @pytest.mark.unit
    def test_makehuman_value(self):
        assert MeshGeneratorBackend.MAKEHUMAN.value == "makehuman"

    @pytest.mark.unit
    def test_smplx_value(self):
        assert MeshGeneratorBackend.SMPLX.value == "smplx"


# ---------------------------------------------------------------------------
# MeshGeneratorInterface
# ---------------------------------------------------------------------------


class TestMeshGeneratorInterface:
    """Verify that concrete generators implement the full interface."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cls",
        [PrimitiveMeshGenerator, MakeHumanMeshGenerator, SMPLXMeshGenerator],
    )
    def test_is_subclass_of_interface(self, cls):
        assert issubclass(cls, MeshGeneratorInterface)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cls",
        [PrimitiveMeshGenerator, MakeHumanMeshGenerator, SMPLXMeshGenerator],
    )
    def test_has_backend_name(self, cls):
        gen = cls()
        assert isinstance(gen.backend_name, str)
        assert len(gen.backend_name) > 0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cls",
        [PrimitiveMeshGenerator, MakeHumanMeshGenerator, SMPLXMeshGenerator],
    )
    def test_is_available_returns_bool(self, cls):
        gen = cls()
        assert isinstance(gen.is_available, bool)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cls",
        [PrimitiveMeshGenerator, MakeHumanMeshGenerator, SMPLXMeshGenerator],
    )
    def test_get_supported_segments_returns_list(self, cls):
        gen = cls()
        segments = gen.get_supported_segments()
        assert isinstance(segments, list)
        assert len(segments) > 0


# ---------------------------------------------------------------------------
# PrimitiveMeshGenerator.is_available
# ---------------------------------------------------------------------------


class TestPrimitiveMeshGeneratorAvailability:
    """Unit tests for PrimitiveMeshGenerator.is_available."""

    @pytest.mark.unit
    def test_backend_name_is_primitive(self):
        assert PrimitiveMeshGenerator().backend_name == "primitive"

    @pytest.mark.unit
    def test_availability_depends_on_trimesh(self):
        gen = PrimitiveMeshGenerator()
        try:
            import trimesh  # noqa: F401

            assert gen.is_available is True
        except ImportError:
            assert gen.is_available is False


# ---------------------------------------------------------------------------
# MakeHumanMeshGenerator
# ---------------------------------------------------------------------------


class TestMakeHumanMeshGeneratorInterface:
    """Interface contract tests for MakeHumanMeshGenerator."""

    @pytest.mark.unit
    def test_backend_name(self):
        assert MakeHumanMeshGenerator().backend_name == "makehuman"

    @pytest.mark.unit
    def test_not_available_without_installation(self):
        """Without a real MakeHuman installation, is_available should be False."""
        gen = MakeHumanMeshGenerator(makehuman_path="/nonexistent/path")
        assert gen.is_available is False

    @pytest.mark.unit
    def test_generate_returns_failure_when_unavailable(self, tmp_path):
        """generate() should return a GeneratedMeshResult(success=False)
        when the backend is unavailable, not raise."""
        from humanoid_character_builder.core.body_parameters import BodyParameters

        gen = MakeHumanMeshGenerator(makehuman_path="/nonexistent/path")
        params = BodyParameters(height_m=1.75, mass_kg=70.0)
        result = gen.generate(params, tmp_path)
        assert isinstance(result, GeneratedMeshResult)
        assert result.success is False


# ---------------------------------------------------------------------------
# SMPLXMeshGenerator
# ---------------------------------------------------------------------------


class TestSMPLXMeshGeneratorInterface:
    """Interface contract tests for SMPLXMeshGenerator."""

    @pytest.mark.unit
    def test_backend_name(self):
        assert SMPLXMeshGenerator().backend_name == "smplx"

    @pytest.mark.unit
    def test_generate_returns_failure_when_unavailable(self, tmp_path):
        """generate() should return a GeneratedMeshResult(success=False)
        when smplx package is absent, not raise."""
        gen = SMPLXMeshGenerator()
        if gen.is_available:
            pytest.skip("smplx is installed, skipping unavailable-path test")

        from humanoid_character_builder.core.body_parameters import BodyParameters

        params = BodyParameters(height_m=1.75, mass_kg=70.0)
        result = gen.generate(params, tmp_path)
        assert isinstance(result, GeneratedMeshResult)
        assert result.success is False


# ---------------------------------------------------------------------------
# MeshGenerator factory
# ---------------------------------------------------------------------------


class TestMeshGeneratorFactory:
    """Unit tests for the MeshGenerator factory class."""

    @pytest.mark.unit
    def test_create_primitive(self):
        gen = MeshGenerator.create(MeshGeneratorBackend.PRIMITIVE)
        assert isinstance(gen, PrimitiveMeshGenerator)

    @pytest.mark.unit
    def test_create_by_string(self):
        gen = MeshGenerator.create("primitive")
        assert isinstance(gen, PrimitiveMeshGenerator)

    @pytest.mark.unit
    def test_create_makehuman(self):
        gen = MeshGenerator.create(MeshGeneratorBackend.MAKEHUMAN)
        assert isinstance(gen, MakeHumanMeshGenerator)

    @pytest.mark.unit
    def test_create_smplx(self):
        gen = MeshGenerator.create(MeshGeneratorBackend.SMPLX)
        assert isinstance(gen, SMPLXMeshGenerator)

    @pytest.mark.unit
    def test_create_unknown_string_raises(self):
        with pytest.raises(ValueError):
            MeshGenerator.create("nonexistent_backend")

    @pytest.mark.unit
    def test_get_available_backends_returns_list(self):
        backends = MeshGenerator.get_available_backends()
        assert isinstance(backends, list)

    @pytest.mark.unit
    def test_get_best_available_returns_interface(self):
        gen = MeshGenerator.get_best_available()
        assert isinstance(gen, MeshGeneratorInterface)

    @pytest.mark.unit
    def test_public_api_backward_compat(self):
        """All public symbols must be importable from the facade module."""
        import humanoid_character_builder.generators.mesh_generator as mg

        assert hasattr(mg, "MeshGenerator")
        assert hasattr(mg, "PrimitiveMeshGenerator")
        assert hasattr(mg, "MakeHumanMeshGenerator")
        assert hasattr(mg, "SMPLXMeshGenerator")
        assert hasattr(mg, "GeneratedMeshResult")
        assert hasattr(mg, "MeshGeneratorBackend")
        assert hasattr(mg, "MeshGeneratorInterface")


# ---------------------------------------------------------------------------
# _pdi_unit_converters backward-compat test
# ---------------------------------------------------------------------------


class TestPDIUnitConvertersPublicAPI:
    """Verify that unit converters are importable from the interface facade."""

    @pytest.mark.unit
    def test_convert_temperature_importable_from_interface(self):
        assert callable(_pdi._convert_temperature)

    @pytest.mark.unit
    def test_convert_pressure_importable_from_interface(self):
        assert callable(_pdi._convert_pressure)

    @pytest.mark.unit
    def test_temperature_k_to_c(self):
        result = _uc._convert_temperature(273.15, "K", "C")
        assert result == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.unit
    def test_temperature_c_to_k(self):
        result = _uc._convert_temperature(0.0, "C", "K")
        assert result == pytest.approx(273.15, abs=1e-6)

    @pytest.mark.unit
    def test_pressure_bar_to_pa(self):
        result = _uc._convert_pressure(1.0, "bar", "Pa")
        assert result == pytest.approx(1e5, rel=1e-6)

    @pytest.mark.unit
    def test_pressure_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown pressure unit"):
            _uc._convert_pressure(1.0, "bar", "foobar")


# ---------------------------------------------------------------------------
# _pdi_validators backward-compat test
# ---------------------------------------------------------------------------


class TestPDIValidatorsPublicAPI:
    """Verify validate_inputs is importable from the interface facade."""

    @pytest.mark.unit
    def test_validate_inputs_importable(self):
        assert callable(_pdi.validate_inputs)

    @pytest.mark.unit
    def test_validate_inputs_valid_case(self):
        is_valid, errors, warnings = _val.validate_inputs(
            pipe_diameter=0.1,
            flow_rate=100.0,
            flow_unit="kg/h",
            pressure=10.0,
            temperature=500.0,
        )
        assert is_valid is True
        assert errors == []

    @pytest.mark.unit
    def test_validate_inputs_missing_flow_rate(self):
        is_valid, errors, warnings = _val.validate_inputs(
            pipe_diameter=0.1,
            flow_rate=None,
            flow_unit="kg/h",
        )
        assert is_valid is False
        assert any("flow_rate" in e for e in errors)

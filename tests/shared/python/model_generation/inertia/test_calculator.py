"""Tests for the unified InertiaCalculator and InertiaResult.

Focuses on the non-mesh (pure-logic) computation paths: primitive
geometry dispatch, manual specification, mode auto-detection, mass
scaling, and result serialization.
"""

from __future__ import annotations

import pytest
from model_generation.core.contracts import PreconditionError
from model_generation.core.types import Geometry, GeometryType
from model_generation.inertia.calculator import (
    InertiaCalculator,
    InertiaMode,
    InertiaResult,
)
from model_generation.inertia.primitives import box_inertia, sphere_inertia


@pytest.fixture
def calc() -> InertiaCalculator:
    return InertiaCalculator()


class TestInertiaResult:
    def test_to_dict_round_trips_fields(self) -> None:
        result = InertiaResult(
            ixx=0.1, iyy=0.2, izz=0.3, mass=2.0, center_of_mass=(1.0, 0.0, 0.0)
        )
        d = result.to_dict()
        assert d["ixx"] == 0.1
        assert d["mass"] == 2.0
        assert d["center_of_mass"] == [1.0, 0.0, 0.0]
        assert d["mode"] == InertiaMode.PRIMITIVE.value

    def test_to_inertia_conversion(self) -> None:
        result = InertiaResult(ixx=0.5, iyy=0.5, izz=0.5, mass=1.0)
        inertia = result.to_inertia()
        assert inertia.ixx == 0.5
        assert inertia.mass == 1.0

    def test_is_valid_for_positive_definite(self) -> None:
        result = InertiaResult(ixx=0.2, iyy=0.2, izz=0.2, mass=1.0)
        assert result.is_valid() is True

    def test_scale_to_mass_scales_components(self) -> None:
        result = InertiaResult(ixx=0.1, iyy=0.2, izz=0.3, mass=2.0)
        scaled = result.scale_to_mass(4.0)
        assert scaled.mass == 4.0
        # Doubling the mass doubles every tensor component.
        assert scaled.ixx == pytest.approx(0.2)
        assert scaled.izz == pytest.approx(0.6)

    def test_scale_to_mass_rejects_nonpositive(self) -> None:
        result = InertiaResult(ixx=0.1, iyy=0.1, izz=0.1, mass=1.0)
        # The @precondition contract rejects a non-positive target mass.
        with pytest.raises(PreconditionError):
            result.scale_to_mass(0.0)

    def test_scale_from_zero_mass_raises(self) -> None:
        result = InertiaResult(ixx=0.1, iyy=0.1, izz=0.1, mass=0.0)
        with pytest.raises(ValueError):
            result.scale_to_mass(2.0)


class TestComputePrimitive:
    def test_box_geometry_matches_formula(self, calc: InertiaCalculator) -> None:
        geom = Geometry.box(0.2, 0.3, 0.4)
        result = calc.compute(geom, mass=3.0, mode=InertiaMode.PRIMITIVE)
        expected = box_inertia(3.0, 0.2, 0.3, 0.4)
        assert result.ixx == pytest.approx(expected["ixx"])
        assert result.izz == pytest.approx(expected["izz"])
        assert result.mode == InertiaMode.PRIMITIVE
        assert result.source == "primitive:box"

    def test_sphere_geometry_matches_formula(self, calc: InertiaCalculator) -> None:
        geom = Geometry.sphere(0.5)
        result = calc.compute(geom, mass=2.0, mode=InertiaMode.PRIMITIVE)
        expected = sphere_inertia(2.0, 0.5)
        assert result.ixx == pytest.approx(expected["ixx"])

    def test_compute_from_geometry_helper(self, calc: InertiaCalculator) -> None:
        result = calc.compute_from_geometry(Geometry.cylinder(0.05, 0.4), mass=1.0)
        assert result.mode == InertiaMode.PRIMITIVE
        # Cylinder perpendicular moments are equal.
        assert result.ixx == pytest.approx(result.iyy)

    def test_default_mass_is_one(self, calc: InertiaCalculator) -> None:
        result = calc.compute(Geometry.sphere(1.0), mode=InertiaMode.PRIMITIVE)
        assert result.mass == 1.0

    def test_primitive_from_dimensions_dict(self, calc: InertiaCalculator) -> None:
        result = calc.compute(
            "ignored",
            mass=1.0,
            mode=InertiaMode.PRIMITIVE,
            dimensions={"radius": 0.1, "length": 0.5},
        )
        # radius+length -> cylinder.
        assert result.source == "primitive:cylinder"

    def test_primitive_requires_usable_source(self, calc: InertiaCalculator) -> None:
        with pytest.raises(ValueError):
            calc.compute("plain.txt", mode=InertiaMode.PRIMITIVE)


class TestComputeManual:
    def test_manual_dict(self, calc: InertiaCalculator) -> None:
        result = calc.compute(
            {"ixx": 0.1, "iyy": 0.2, "izz": 0.3, "mass": 5.0},
            mode=InertiaMode.MANUAL,
        )
        assert result.mode == InertiaMode.MANUAL
        assert result.mass == 5.0
        assert result.izz == pytest.approx(0.3)

    def test_manual_helper(self, calc: InertiaCalculator) -> None:
        result = calc.compute_from_manual(0.1, 0.1, 0.05, mass=2.0)
        assert result.source == "manual"
        assert result.ixx == 0.1

    def test_manual_scales_to_target_mass(self, calc: InertiaCalculator) -> None:
        result = calc.compute(
            {"ixx": 0.1, "iyy": 0.1, "izz": 0.1, "mass": 1.0},
            mass=2.0,
            mode=InertiaMode.MANUAL,
        )
        assert result.mass == 2.0
        assert result.ixx == pytest.approx(0.2)

    def test_manual_requires_dict(self, calc: InertiaCalculator) -> None:
        with pytest.raises(ValueError):
            calc.compute("not-a-dict", mode=InertiaMode.MANUAL)

    def test_nested_inertia_dict(self, calc: InertiaCalculator) -> None:
        result = calc.compute(
            {"inertia": {"ixx": 0.4, "iyy": 0.4, "izz": 0.4}, "mass": 1.0},
            mode=InertiaMode.MANUAL,
        )
        assert result.ixx == pytest.approx(0.4)


class TestModeDetection:
    def test_dict_with_ixx_detected_manual(self, calc: InertiaCalculator) -> None:
        result = calc.compute({"ixx": 0.1, "iyy": 0.1, "izz": 0.1, "mass": 1.0})
        assert result.mode == InertiaMode.MANUAL

    def test_primitive_geometry_detected(self, calc: InertiaCalculator) -> None:
        result = calc.compute(Geometry.sphere(0.3), mass=1.0)
        assert result.mode == InertiaMode.PRIMITIVE

    def test_mesh_geometry_detects_mesh_mode(self, calc: InertiaCalculator) -> None:
        mesh_geom = Geometry(geometry_type=GeometryType.MESH, mesh_filename="nope.stl")
        # trimesh is unavailable -> falls back to a default inertia result
        # rather than raising, but the selected mode is a mesh mode.
        result = calc.compute(mesh_geom, mass=1.0)
        assert result.mode in (
            InertiaMode.MESH_UNIFORM_DENSITY,
            InertiaMode.MESH_SPECIFIED_MASS,
        )

    def test_str_mesh_path_detected_mesh(self, calc: InertiaCalculator) -> None:
        assert calc._detect_mode("arm.stl") == InertiaMode.MESH_UNIFORM_DENSITY

    def test_str_non_mesh_path_detected_primitive(
        self, calc: InertiaCalculator
    ) -> None:
        assert calc._detect_mode("config.yaml") == InertiaMode.PRIMITIVE


class TestComputeErrors:
    def test_none_source_raises(self, calc: InertiaCalculator) -> None:
        with pytest.raises(ValueError):
            calc.compute(None)

    def test_clear_cache(self, calc: InertiaCalculator) -> None:
        calc._cache["k"] = InertiaResult(ixx=0.1, iyy=0.1, izz=0.1, mass=1.0)
        calc.clear_cache()
        assert calc._cache == {}

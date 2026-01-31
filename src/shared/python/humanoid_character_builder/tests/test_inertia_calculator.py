"""
Tests for inertia calculation module.
"""

import numpy as np
import pytest
from humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    InertiaResult,
    MeshInertiaCalculator,
    validate_inertia_tensor,
)


class TestInertiaResult:
    """Tests for InertiaResult class."""

    def test_as_matrix(self):
        inertia = InertiaResult(
            ixx=1.0,
            iyy=2.0,
            izz=3.0,
            ixy=0.1,
            ixz=0.2,
            iyz=0.3,
        )
        matrix = inertia.as_matrix()

        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 1.0
        assert matrix[1, 1] == 2.0
        assert matrix[2, 2] == 3.0
        assert matrix[0, 1] == 0.1
        assert matrix[1, 0] == 0.1  # Symmetric

    def test_as_urdf_dict(self):
        inertia = InertiaResult(ixx=1.0, iyy=2.0, izz=3.0)
        urdf = inertia.as_urdf_dict()

        assert urdf["ixx"] == 1.0
        assert urdf["iyy"] == 2.0
        assert urdf["izz"] == 3.0
        assert urdf["ixy"] == 0.0

    def test_is_valid_positive(self):
        # Valid inertia (sphere)
        inertia = InertiaResult(ixx=1.0, iyy=1.0, izz=1.0)
        assert inertia.is_valid()

    def test_is_valid_negative_diagonal(self):
        # Invalid (negative moment)
        inertia = InertiaResult(ixx=-1.0, iyy=1.0, izz=1.0)
        assert not inertia.is_valid()

    def test_is_valid_triangle_inequality(self):
        # Invalid (triangle inequality: ixx + iyy >= izz)
        # 1 + 1 < 3
        inertia = InertiaResult(ixx=1.0, iyy=1.0, izz=3.0)
        assert not inertia.is_valid()

    def test_validate_positive_definite(self):
        # Valid
        inertia = InertiaResult(ixx=1.0, iyy=1.0, izz=1.0)
        assert inertia.validate_positive_definite()

        # Invalid (not positive definite matrix)
        # Off-diagonal too large: det(I) < 0 or eigenvalues < 0
        inertia_bad = InertiaResult(
            ixx=1.0,
            iyy=1.0,
            izz=1.0,
            ixy=2.0,  # Too large
        )
        assert not inertia_bad.validate_positive_definite()

    def test_create_default(self):
        inertia = InertiaResult.create_default(mass=5.0)
        assert inertia.mass == 5.0
        assert inertia.mode == InertiaMode.PRIMITIVE_APPROXIMATION
        assert inertia.ixx > 0


class TestMeshInertiaCalculator:
    """Tests for MeshInertiaCalculator."""

    @pytest.fixture
    def calculator(self):
        return MeshInertiaCalculator(default_density=1000.0)

    @pytest.fixture
    def mock_trimesh(self, monkeypatch):
        """Mock trimesh module and objects."""

        class MockTrimesh:
            def __init__(self, *args, **kwargs):
                self.is_watertight = True
                self.volume = 0.001
                self.center_mass = np.array([0.0, 0.0, 0.0])
                self.moment_inertia = np.eye(3) * 0.001  # Unit density inertia
                self.bounding_box = self

            @property
            def centroid(self):
                return self.center_mass

        monkeypatch.setattr(
            "humanoid_character_builder.mesh.inertia_calculator.trimesh.Trimesh",
            MockTrimesh,
        )
        # Also need to mock trimesh import in the method
        # This is tricky because the import is inside the method.
        # However, since we mock trimesh.Trimesh in the module scope via sys.modules
        # or similar, let's see.
        # Actually, simpler to mock the _check_trimesh method to return True
        # and mock importlib or just rely on if trimesh is installed.
        # Assuming dev env has trimesh installed.

    def test_init(self):
        calc = MeshInertiaCalculator()
        assert calc.default_density == 1050.0

    def test_create_manual_inertia(self):
        inertia = MeshInertiaCalculator.create_manual_inertia(
            ixx=1.0, iyy=2.0, izz=3.0, mass=10.0
        )
        assert inertia.mass == 10.0
        assert inertia.ixx == 1.0
        assert inertia.mode == InertiaMode.MANUAL

    def test_transform_inertia_rotation(self, calculator):
        # Identity inertia rotated 90 deg around Z
        inertia = InertiaResult(ixx=2.0, iyy=1.0, izz=3.0)  # Distinct diagonal

        # Rotate 90 deg around Z: X -> Y, Y -> -X
        # I_new should have ixx=1.0, iyy=2.0
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        new_inertia = calculator.transform_inertia(inertia, rotation=rotation)

        assert np.isclose(new_inertia.ixx, 1.0)
        assert np.isclose(new_inertia.iyy, 2.0)
        assert np.isclose(new_inertia.izz, 3.0)

    def test_transform_inertia_translation(self, calculator):
        # Parallel axis theorem
        # Point mass m at origin: I = 0
        # Moved to d: I = m * d^2 (simplified)
        mass = 2.0
        inertia = InertiaResult(ixx=0.0, iyy=0.0, izz=0.0, mass=mass)

        # Move along X by 1.0
        # Ixx should remain 0 (on axis)
        # Iyy = m * x^2 = 2 * 1 = 2
        # Izz = m * x^2 = 2 * 1 = 2
        translation = np.array([1.0, 0.0, 0.0])

        new_inertia = calculator.transform_inertia(inertia, translation=translation)

        assert np.isclose(new_inertia.ixx, 0.0)
        assert np.isclose(new_inertia.iyy, 2.0)
        assert np.isclose(new_inertia.izz, 2.0)
        # COM should be at -1.0 relative to new frame?
        # The method transforms FROM old frame TO new frame.
        # If translation is vector from old origin to new origin.
        # Old COM at 0. New COM at 0 - d = -d.
        assert np.allclose(new_inertia.center_of_mass, -translation)


class TestValidation:
    """Tests for validation functions."""

    def test_valid_tensor(self):
        # Valid diagonal tensor
        inertia_tensor = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        errors = validate_inertia_tensor(inertia_tensor)
        assert len(errors) == 0

    def test_non_symmetric(self):
        inertia_tensor = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        errors = validate_inertia_tensor(inertia_tensor)
        assert any("symmetric" in e for e in errors)

    def test_negative_diagonal(self):
        inertia_tensor = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        errors = validate_inertia_tensor(inertia_tensor)
        assert any("positive" in e.lower() for e in errors)

    def test_wrong_shape(self):
        inertia_tensor = np.array([[1.0, 0.0], [0.0, 1.0]])
        errors = validate_inertia_tensor(inertia_tensor)
        assert any("3x3" in e for e in errors)

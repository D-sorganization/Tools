"""Test suite for lighting and shading control for 3D visualization.

This module implements TDD for GitHub issue #541: Lighting & Shading Control.
Tests are organized by:
1. Unit tests on Light and MaterialProperties dataclasses
2. LightingManager preset loading and application
3. Light direction normalization and validation
4. Integration tests with UI widget updates
5. Performance tests for sub-100ms lighting adjustments

Success criteria:
- All lighting tests pass
- Presets load correctly with proper values
- Light directions are normalized to unit vectors
- Material properties respect valid ranges
- Widget updates trigger lighting changes within 100ms
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Unit Tests: Light Dataclass
# =============================================================================


class TestLightDataclass:
    """Test the Light dataclass for direction, intensity, and color."""

    @pytest.mark.unit
    def test_light_creation_valid(self) -> None:
        """Test creating a Light with valid parameters."""
        from glass_models.viz.lighting import Light

        light = Light(
            direction=np.array([1.0, 0.0, 0.0]),
            intensity=0.8,
            color=np.array([1.0, 1.0, 1.0]),
        )

        assert light is not None
        assert np.allclose(light.direction, [1.0, 0.0, 0.0])
        assert light.intensity == 0.8
        assert np.allclose(light.color, [1.0, 1.0, 1.0])

    @pytest.mark.unit
    def test_light_direction_normalization(self) -> None:
        """Test that Light normalizes non-unit direction vectors.

        Design by Contract: Light should validate that direction becomes
        a unit vector after normalization.
        """
        from glass_models.viz.lighting import Light

        # Create light with non-unit direction
        light = Light(
            direction=np.array([3.0, 4.0, 0.0]),  # magnitude = 5
            intensity=1.0,
            color=np.array([1.0, 1.0, 1.0]),
        )

        # Direction should be normalized
        magnitude = np.linalg.norm(light.direction)
        assert abs(magnitude - 1.0) < 1e-9, (
            f"Direction not normalized: magnitude = {magnitude}"
        )

    @pytest.mark.unit
    def test_light_intensity_validation(self) -> None:
        """Test intensity clamping to [0, 1] range.

        Design by Contract: intensity must be in [0, 1].
        """
        from glass_models.viz.lighting import Light

        # Valid intensity
        light_valid = Light(
            direction=np.array([1.0, 0.0, 0.0]),
            intensity=0.5,
            color=np.array([1.0, 1.0, 1.0]),
        )
        assert light_valid.intensity == 0.5

        # Out of range should raise
        with pytest.raises((ValueError, TypeError)):
            Light(
                direction=np.array([1.0, 0.0, 0.0]),
                intensity=1.5,  # > 1
                color=np.array([1.0, 1.0, 1.0]),
            )

        with pytest.raises((ValueError, TypeError)):
            Light(
                direction=np.array([1.0, 0.0, 0.0]),
                intensity=-0.1,  # < 0
                color=np.array([1.0, 1.0, 1.0]),
            )

    @pytest.mark.unit
    def test_light_color_validation(self) -> None:
        """Test color components are in [0, 1] range."""
        from glass_models.viz.lighting import Light

        # Valid color
        light_valid = Light(
            direction=np.array([1.0, 0.0, 0.0]),
            intensity=1.0,
            color=np.array([0.8, 0.5, 0.2]),
        )
        assert np.allclose(light_valid.color, [0.8, 0.5, 0.2])

        # Out of range should raise
        with pytest.raises((ValueError, TypeError)):
            Light(
                direction=np.array([1.0, 0.0, 0.0]),
                intensity=1.0,
                color=np.array([1.2, 0.5, 0.2]),  # > 1
            )


# =============================================================================
# Unit Tests: MaterialProperties Dataclass
# =============================================================================


class TestMaterialPropertiesDataclass:
    """Test the MaterialProperties dataclass."""

    @pytest.mark.unit
    def test_material_properties_creation_valid(self) -> None:
        """Test creating MaterialProperties with valid parameters."""
        from glass_models.viz.lighting import MaterialProperties

        props = MaterialProperties(
            ambient=0.2,
            diffuse=0.6,
            specular=0.8,
            shininess=32.0,
        )

        assert props.ambient == 0.2
        assert props.diffuse == 0.6
        assert props.specular == 0.8
        assert props.shininess == 32.0

    @pytest.mark.unit
    def test_material_properties_range_validation(self) -> None:
        """Test that ambient, diffuse, specular are in [0, 1]."""
        from glass_models.viz.lighting import MaterialProperties

        # Valid properties
        props_valid = MaterialProperties(
            ambient=0.1,
            diffuse=0.7,
            specular=0.9,
            shininess=64.0,
        )
        assert props_valid.ambient == 0.1

        # Invalid ambient
        with pytest.raises((ValueError, TypeError)):
            MaterialProperties(
                ambient=1.5,  # > 1
                diffuse=0.6,
                specular=0.8,
                shininess=32.0,
            )

        # Invalid diffuse
        with pytest.raises((ValueError, TypeError)):
            MaterialProperties(
                ambient=0.2,
                diffuse=-0.1,  # < 0
                specular=0.8,
                shininess=32.0,
            )

    @pytest.mark.unit
    def test_material_properties_shininess_validation(self) -> None:
        """Test shininess is a positive value (typically 1-128)."""
        from glass_models.viz.lighting import MaterialProperties

        # Valid shininess
        props_valid = MaterialProperties(
            ambient=0.2,
            diffuse=0.6,
            specular=0.8,
            shininess=32.0,
        )
        assert props_valid.shininess == 32.0

        # Negative shininess should raise
        with pytest.raises((ValueError, TypeError)):
            MaterialProperties(
                ambient=0.2,
                diffuse=0.6,
                specular=0.8,
                shininess=-1.0,
            )


# =============================================================================
# Unit Tests: LightingManager Presets
# =============================================================================


class TestLightingManagerPresets:
    """Test LightingManager preset loading and management."""

    @pytest.mark.unit
    def test_lighting_manager_creation(self) -> None:
        """Test creating a LightingManager instance."""
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()
        assert manager is not None

    @pytest.mark.unit
    def test_lighting_manager_has_presets(self) -> None:
        """Test that LightingManager has standard presets."""
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()

        # Should have at least three presets
        assert hasattr(manager, "presets") or hasattr(manager, "_presets")
        presets = manager.presets if hasattr(manager, "presets") else manager._presets

        assert "headlight" in presets
        assert "studio_3light" in presets
        assert "ambient_only" in presets

    @pytest.mark.unit
    def test_set_preset_headlight(self) -> None:
        """Test setting headlight preset.

        Headlight: single frontal light mimicking a camera-mounted light.
        """
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()
        manager.set_preset("headlight")

        # Should have one active light
        lights = manager.get_lights()
        assert len(lights) > 0

        # Headlight direction should be forward (roughly toward negative Z)
        primary_light = lights[0]
        assert primary_light.intensity > 0.5

    @pytest.mark.unit
    def test_set_preset_studio_3light(self) -> None:
        """Test setting studio 3-light preset.

        Studio 3-light: key light (main), fill light (secondary), back light.
        """
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()
        manager.set_preset("studio_3light")

        lights = manager.get_lights()

        # Should have at least 3 lights
        assert len(lights) >= 3

        # Lights should have different colors/intensities
        intensities = [light.intensity for light in lights]
        # Key light should be brightest
        assert max(intensities) >= 0.8

    @pytest.mark.unit
    def test_set_preset_ambient_only(self) -> None:
        """Test setting ambient-only preset.

        Ambient: only ambient lighting, no directional lights.
        """
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()
        manager.set_preset("ambient_only")

        lights = manager.get_lights()

        # Ambient-only should still have at least one light (the ambient)
        assert len(lights) > 0

        # All lights should have low-to-medium intensity
        for light in lights:
            assert light.intensity <= 0.5

    @pytest.mark.unit
    def test_set_invalid_preset_raises(self) -> None:
        """Test that invalid preset name raises error."""
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()

        with pytest.raises((ValueError, KeyError)):
            manager.set_preset("nonexistent_preset")


# =============================================================================
# Unit Tests: Light Direction Normalization
# =============================================================================


class TestLightDirectionNormalization:
    """Test normalization of light directions."""

    @pytest.mark.unit
    def test_update_light_direction_normalization(self) -> None:
        """Test that update_light_direction normalizes the direction."""
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()
        manager.set_preset("headlight")

        # Update with non-unit direction
        manager.update_light_direction(0, np.array([2.0, 2.0, 2.0]))

        lights = manager.get_lights()
        direction = lights[0].direction

        # Should be normalized (use float32 tolerance for f32 arrays)
        magnitude = np.linalg.norm(direction)
        assert abs(magnitude - 1.0) < 1e-6

    @pytest.mark.unit
    def test_update_light_direction_spherical_coords(self) -> None:
        """Test updating light direction using spherical coordinates.

        Spherical: azimuth (0-360°) and elevation (0-90°).
        """
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()
        manager.set_preset("headlight")

        # Update using spherical coordinates
        # azimuth=0, elevation=90 should point up (0, 0, 1)
        manager.update_light_direction_spherical(0, azimuth=0, elevation=90)

        lights = manager.get_lights()
        direction = lights[0].direction

        # Should be normalized unit vector
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-9

        # Should point roughly up
        # Z component should be close to 1
        assert direction[2] > 0.9


# =============================================================================
# Unit Tests: Material Properties Application
# =============================================================================


class TestMaterialPropertiesApplication:
    """Test application of material properties."""

    @pytest.mark.unit
    def test_apply_material_properties(self) -> None:
        """Test applying material properties."""
        from glass_models.viz.lighting import LightingManager, MaterialProperties

        manager = LightingManager()
        props = MaterialProperties(
            ambient=0.3,
            diffuse=0.7,
            specular=0.9,
            shininess=64.0,
        )

        result = manager.apply_material(props)

        # Should return the applied properties or True
        assert result is not None

    @pytest.mark.unit
    def test_material_properties_stored(self) -> None:
        """Test that material properties are stored in manager."""
        from glass_models.viz.lighting import LightingManager, MaterialProperties

        manager = LightingManager()
        props = MaterialProperties(
            ambient=0.25,
            diffuse=0.65,
            specular=0.85,
            shininess=48.0,
        )

        manager.apply_material(props)

        # Manager should store material properties
        stored_props = manager.get_material()
        assert stored_props is not None
        assert stored_props.ambient == 0.25


# =============================================================================
# Integration Tests: Lighting Application
# =============================================================================


class TestLightingApplication:
    """Integration tests for applying lighting to rendered geometry."""

    @pytest.mark.integration
    def test_apply_lighting_to_vertices(self) -> None:
        """Test applying lighting calculations to vertex normals.

        Creates a simple cube, computes lighting based on normals and
        light direction.
        """
        from glass_models.viz.lighting import LightingManager

        # Create simple vertex normals for a cube
        normals = np.array(
            [
                [1.0, 0.0, 0.0],  # +X face
                [-1.0, 0.0, 0.0],  # -X face
                [0.0, 1.0, 0.0],  # +Y face
                [0.0, -1.0, 0.0],  # -Y face
                [0.0, 0.0, 1.0],  # +Z face
                [0.0, 0.0, -1.0],  # -Z face
            ]
        )

        manager = LightingManager()
        manager.set_preset("headlight")

        result = manager.apply_lighting(normals)

        assert result is not None
        # Result should be color values (RGB or RGBA)
        assert result.shape[0] == normals.shape[0]  # One color per vertex
        assert result.shape[1] >= 3  # At least RGB

    @pytest.mark.integration
    def test_apply_lighting_returns_colors(self) -> None:
        """Test that apply_lighting returns valid color values."""
        from glass_models.viz.lighting import LightingManager

        # Simple quad normals
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )

        manager = LightingManager()
        manager.set_preset("studio_3light")

        colors = manager.apply_lighting(normals)

        # Colors should be in [0, 1] range
        assert np.all(colors >= 0.0)
        assert np.all(colors <= 1.0)


# =============================================================================
# Performance Tests
# =============================================================================


class TestLightingPerformance:
    """Performance tests for lighting adjustments."""

    @pytest.mark.performance
    def test_preset_loading_performance(self) -> None:
        """Test that loading a preset takes < 100ms."""
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()

        start = time.time()
        manager.set_preset("studio_3light")
        elapsed = (time.time() - start) * 1000  # Convert to ms

        assert elapsed < 100, f"Preset loading took {elapsed:.2f}ms (limit: 100ms)"

    @pytest.mark.performance
    def test_light_update_performance(self) -> None:
        """Test that updating light direction takes < 100ms."""
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()
        manager.set_preset("headlight")

        start = time.time()
        manager.update_light_direction_spherical(0, azimuth=45, elevation=60)
        elapsed = (time.time() - start) * 1000

        assert elapsed < 100, f"Light update took {elapsed:.2f}ms (limit: 100ms)"

    @pytest.mark.performance
    def test_lighting_application_performance(self) -> None:
        """Test that applying lighting to 10k vertices takes < 100ms."""
        from glass_models.viz.lighting import LightingManager

        # Create 10k random normals
        normals = np.random.randn(10000, 3)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        manager = LightingManager()
        manager.set_preset("studio_3light")

        start = time.time()
        colors = manager.apply_lighting(normals)
        elapsed = (time.time() - start) * 1000

        assert elapsed < 100, (
            f"Lighting application took {elapsed:.2f}ms for 10k vertices (limit: 100ms)"
        )
        assert colors.shape[0] == normals.shape[0]


# =============================================================================
# Contract Tests: API Surface
# =============================================================================


class TestLightingAPIContract:
    """Contract tests ensuring API surface for downstream consumers."""

    @pytest.mark.contract
    def test_lighting_manager_preset_dict_structure(self) -> None:
        """Test that presets dict is DRY (single source of truth)."""
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()
        presets = manager.presets if hasattr(manager, "presets") else manager._presets

        # Each preset should be consistent
        for name, preset_config in presets.items():
            assert isinstance(name, str)
            assert isinstance(preset_config, (dict, list, tuple))

    @pytest.mark.contract
    def test_lighting_manager_methods_exist(self) -> None:
        """Test that required public methods exist."""
        from glass_models.viz.lighting import LightingManager

        manager = LightingManager()

        # Required methods
        assert hasattr(manager, "set_preset")
        assert callable(manager.set_preset)

        assert hasattr(manager, "apply_lighting")
        assert callable(manager.apply_lighting)

        assert hasattr(manager, "update_light_direction")
        assert callable(manager.update_light_direction)

        assert hasattr(manager, "update_light_direction_spherical")
        assert callable(manager.update_light_direction_spherical)

        assert hasattr(manager, "get_lights")
        assert callable(manager.get_lights)

        assert hasattr(manager, "apply_material")
        assert callable(manager.apply_material)

        assert hasattr(manager, "get_material")
        assert callable(manager.get_material)

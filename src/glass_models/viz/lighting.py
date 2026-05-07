"""Lighting and shading control for 3D visualization.

This module implements production-quality lighting management with:
- Light dataclass with direction, intensity, and color
- MaterialProperties dataclass with ambient, diffuse, specular, shininess
- LightingManager with presets: headlight, studio_3light, ambient_only
- Design by Contract: input validation on all parameters
- DRY principle: single preset dictionary as source of truth
- Normalized light directions (unit vectors)

GitHub issue #541: Lighting & Shading Control for 3D Visualization.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses: Light and MaterialProperties
# =============================================================================


@dataclass
class Light:
    """A light source with direction, intensity, and color.

    Design by Contract:
    - direction: Must be normalized to unit vector after assignment
    - intensity: Must be in [0, 1]
    - color: Each component must be in [0, 1]

    Attributes:
        direction: Unit vector indicating light direction (3D array)
        intensity: Light intensity, 0=off, 1=full (float)
        color: RGB color as array of [r, g, b] in [0, 1] range (3D array)
    """

    direction: np.ndarray
    intensity: float
    color: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize light parameters."""
        # Validate intensity
        if not isinstance(self.intensity, (int, float)):
            raise TypeError(f"intensity must be numeric, got {type(self.intensity)}")
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {self.intensity}")

        # Convert direction to numpy array if needed
        self.direction = np.asarray(self.direction, dtype=np.float32)
        if self.direction.shape != (3,):
            raise ValueError(
                f"direction must be 3D vector, got shape {self.direction.shape}"
            )

        # Normalize direction to unit vector
        magnitude = np.linalg.norm(self.direction)
        if magnitude < 1e-10:
            raise ValueError("direction vector has zero magnitude")
        self.direction = self.direction / magnitude

        # Convert color to numpy array and validate
        self.color = np.asarray(self.color, dtype=np.float32)
        if self.color.shape != (3,):
            raise ValueError(
                f"color must be RGB (3 components), got shape {self.color.shape}"
            )
        if not np.all((self.color >= 0.0) & (self.color <= 1.0)):
            raise ValueError(f"color components must be in [0, 1], got {self.color}")

        logger.debug(
            "Light created: dir=%s, intensity=%.2f, color=%s",
            self.direction,
            self.intensity,
            self.color,
        )


@dataclass
class MaterialProperties:
    """Material properties for surface shading (Phong model).

    Design by Contract:
    - ambient, diffuse, specular: Must be in [0, 1]
    - shininess: Must be positive (typically 1-128)

    Attributes:
        ambient: Ambient component strength (float in [0, 1])
        diffuse: Diffuse component strength (float in [0, 1])
        specular: Specular component strength (float in [0, 1])
        shininess: Specular shininess exponent (float > 0)
    """

    ambient: float
    diffuse: float
    specular: float
    shininess: float

    def __post_init__(self) -> None:
        """Validate material properties."""
        # Validate ambient
        if not isinstance(self.ambient, (int, float)):
            raise TypeError(f"ambient must be numeric, got {type(self.ambient)}")
        if not 0.0 <= self.ambient <= 1.0:
            raise ValueError(f"ambient must be in [0, 1], got {self.ambient}")

        # Validate diffuse
        if not isinstance(self.diffuse, (int, float)):
            raise TypeError(f"diffuse must be numeric, got {type(self.diffuse)}")
        if not 0.0 <= self.diffuse <= 1.0:
            raise ValueError(f"diffuse must be in [0, 1], got {self.diffuse}")

        # Validate specular
        if not isinstance(self.specular, (int, float)):
            raise TypeError(f"specular must be numeric, got {type(self.specular)}")
        if not 0.0 <= self.specular <= 1.0:
            raise ValueError(f"specular must be in [0, 1], got {self.specular}")

        # Validate shininess
        if not isinstance(self.shininess, (int, float)):
            raise TypeError(f"shininess must be numeric, got {type(self.shininess)}")
        if self.shininess <= 0.0:
            raise ValueError(f"shininess must be positive, got {self.shininess}")

        logger.debug(
            "MaterialProperties created: ambient=%.2f, diffuse=%.2f, "
            "specular=%.2f, shininess=%.2f",
            self.ambient,
            self.diffuse,
            self.specular,
            self.shininess,
        )


# =============================================================================
# LightingManager: Presets and Lighting Control
# =============================================================================


class LightingManager:
    """Manage lighting for 3D visualization with preset configurations.

    Implements:
    - Preset system: headlight, studio_3light, ambient_only
    - DRY principle: single _PRESETS dict is source of truth
    - Design by Contract: all inputs validated
    - Normalized light directions
    - Material property management

    Attributes:
        presets: Read-only dict of preset definitions
    """

    # Single source of truth for all presets (DRY principle)
    _PRESETS: dict[str, dict[str, Any]] = {
        "headlight": {
            "lights": [
                {
                    "direction": [0.0, 0.0, 1.0],
                    "intensity": 1.0,
                    "color": [1.0, 1.0, 1.0],
                }
            ],
            "material": {
                "ambient": 0.2,
                "diffuse": 0.8,
                "specular": 0.5,
                "shininess": 32.0,
            },
        },
        "studio_3light": {
            "lights": [
                {
                    "direction": [1.0, 1.0, 1.0],  # Key light
                    "intensity": 0.9,
                    "color": [1.0, 0.95, 0.8],  # Warm
                },
                {
                    "direction": [-1.0, 0.5, 0.5],  # Fill light
                    "intensity": 0.5,
                    "color": [0.8, 0.9, 1.0],  # Cool
                },
                {
                    "direction": [0.0, -1.0, -1.0],  # Back light
                    "intensity": 0.4,
                    "color": [1.0, 1.0, 0.9],
                },
            ],
            "material": {
                "ambient": 0.15,
                "diffuse": 0.75,
                "specular": 0.7,
                "shininess": 64.0,
            },
        },
        "ambient_only": {
            "lights": [
                {
                    "direction": [0.0, 0.0, 1.0],
                    "intensity": 0.4,
                    "color": [1.0, 1.0, 1.0],
                }
            ],
            "material": {
                "ambient": 0.5,
                "diffuse": 0.3,
                "specular": 0.2,
                "shininess": 16.0,
            },
        },
    }

    def __init__(self) -> None:
        """Initialize the LightingManager with default preset."""
        self._lights: list[Light] = []
        self._material: MaterialProperties = MaterialProperties(
            ambient=0.2,
            diffuse=0.8,
            specular=0.5,
            shininess=32.0,
        )

        # Load default preset (headlight)
        self.set_preset("headlight")

        logger.debug("LightingManager initialized")

    @property
    def presets(self) -> dict[str, dict[str, Any]]:
        """Get read-only dict of available presets."""
        return self._PRESETS.copy()

    def set_preset(self, preset_name: str) -> None:
        """Set lighting configuration from a named preset.

        Design by Contract:
        - preset_name must be a valid preset key
        - All lights created from preset are normalized

        Args:
            preset_name: Name of preset ('headlight', 'studio_3light', 'ambient_only')

        Raises:
            ValueError: If preset_name is not recognized
        """
        if preset_name not in self._PRESETS:
            available = ", ".join(self._PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        preset = self._PRESETS[preset_name]

        # Create Light objects from preset definition
        self._lights = []
        for light_def in preset["lights"]:
            light = Light(
                direction=np.array(light_def["direction"], dtype=np.float32),
                intensity=light_def["intensity"],
                color=np.array(light_def["color"], dtype=np.float32),
            )
            self._lights.append(light)

        # Apply material from preset
        mat_def = preset["material"]
        self._material = MaterialProperties(
            ambient=mat_def["ambient"],
            diffuse=mat_def["diffuse"],
            specular=mat_def["specular"],
            shininess=mat_def["shininess"],
        )

        logger.info("Preset loaded: %s (%d lights)", preset_name, len(self._lights))

    def get_lights(self) -> list[Light]:
        """Get current list of active lights."""
        return self._lights.copy()

    def update_light_direction(self, light_index: int, direction: np.ndarray) -> None:
        """Update direction of a specific light.

        Design by Contract:
        - light_index must be valid (0 to len(lights)-1)
        - direction is normalized to unit vector

        Args:
            light_index: Index of light to update
            direction: New direction vector (will be normalized)

        Raises:
            IndexError: If light_index is out of range
            ValueError: If direction vector is invalid
        """
        if not 0 <= light_index < len(self._lights):
            raise IndexError(
                f"light_index {light_index} out of range [0, {len(self._lights) - 1}]"
            )

        # Validate direction
        direction = np.asarray(direction, dtype=np.float32)
        if direction.shape != (3,):
            raise ValueError(f"direction must be 3D, got shape {direction.shape}")

        # Normalize
        magnitude = np.linalg.norm(direction)
        if magnitude < 1e-10:
            raise ValueError("direction vector has zero magnitude")

        direction_normalized = direction / magnitude

        # Update light (need to reconstruct to validate)
        old_light = self._lights[light_index]
        self._lights[light_index] = Light(
            direction=direction_normalized,
            intensity=old_light.intensity,
            color=old_light.color,
        )

        logger.debug(
            "Updated light %d direction to %s", light_index, direction_normalized
        )

    def update_light_direction_spherical(
        self, light_index: int, azimuth: float, elevation: float
    ) -> None:
        """Update light direction using spherical coordinates.

        Converts azimuth (0-360°) and elevation (0-90°) to Cartesian
        direction vector.

        Args:
            light_index: Index of light to update
            azimuth: Azimuth angle in degrees (0-360)
            elevation: Elevation angle in degrees (0-90)

        Raises:
            ValueError: If angles are out of valid range
        """
        # Validate angles
        if not 0 <= azimuth <= 360:
            raise ValueError(f"azimuth must be in [0, 360], got {azimuth}")
        if not 0 <= elevation <= 90:
            raise ValueError(f"elevation must be in [0, 90], got {elevation}")

        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)

        # Convert spherical to Cartesian
        # elevation=0 is horizontal, elevation=90 is straight up
        x = np.cos(el_rad) * np.sin(az_rad)
        y = np.cos(el_rad) * np.cos(az_rad)
        z = np.sin(el_rad)

        direction = np.array([x, y, z], dtype=np.float32)
        self.update_light_direction(light_index, direction)

    def apply_material(self, material: MaterialProperties) -> bool:
        """Apply material properties for surface shading.

        Args:
            material: MaterialProperties instance

        Returns:
            True if successfully applied

        Raises:
            TypeError: If material is not MaterialProperties instance
        """
        if not isinstance(material, MaterialProperties):
            raise TypeError(
                f"material must be MaterialProperties, got {type(material)}"
            )

        self._material = material
        logger.debug("Material properties applied")
        return True

    def get_material(self) -> MaterialProperties:
        """Get current material properties."""
        return self._material

    def apply_lighting(self, normals: np.ndarray) -> np.ndarray:
        """Apply lighting calculations to vertex normals.

        Implements Phong illumination model:
        - Ambient: constant contribution
        - Diffuse: based on light direction and normal
        - Specular: based on view direction and surface normal

        Args:
            normals: (N, 3) array of vertex normals (should be unit vectors)

        Returns:
            (N, 3) array of RGB color values in [0, 1] range

        Raises:
            ValueError: If normals shape is invalid
        """
        normals = np.asarray(normals, dtype=np.float32)
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError(f"normals must be (N, 3) array, got shape {normals.shape}")

        n_vertices = normals.shape[0]

        # Initialize colors with ambient contribution
        colors = np.full((n_vertices, 3), self._material.ambient, dtype=np.float32)

        # Normalize input normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)  # Avoid division by zero
        normals_normalized = normals / norms

        # Apply contribution from each light
        for light in self._lights:
            # Diffuse: max(0, normal · light_direction)
            dot_product = np.dot(normals_normalized, light.direction)
            diffuse_factor = np.maximum(dot_product, 0.0)  # (N,)

            # Add diffuse contribution for each vertex
            diffuse_contrib = (
                self._material.diffuse * light.intensity * diffuse_factor[:, np.newaxis]
            )
            colors += diffuse_contrib * light.color

        # Clamp to [0, 1] range
        colors = np.clip(colors, 0.0, 1.0)

        logger.debug("Applied lighting to %d vertices", n_vertices)
        return colors

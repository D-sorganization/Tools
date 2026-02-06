"""Geometry generator for Glass Bath FEA.

This module generates the cylindrical vessel geometry with metal bottom
layer, glass pool, and electrode positions for finite element analysis.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import GlassBathFEAConfig

# Conversion factor
INCHES_TO_METERS = 0.0254


class GeometryGenerator:
    """Generate cylindrical vessel geometry for FEA.

    Creates the geometric definition of:
    - Cylindrical vessel boundary
    - Metal bottom layer region
    - Glass pool region
    - Electrode positions and cylinders

    Attributes:
        config: FEA configuration parameters
    """

    # Material region IDs for mesh export
    MATERIAL_ID_GLASS = 1
    MATERIAL_ID_METAL = 2
    MATERIAL_ID_ELECTRODE = 3

    def __init__(self, config: GlassBathFEAConfig) -> None:
        """Initialize geometry generator.

        Args:
            config: FEA configuration with vessel dimensions
        """
        self.config = config

        # Pre-compute dimensions in meters
        self._radius_m = (config.bath_diameter / 2) * INCHES_TO_METERS
        self._glass_depth_m = config.glass_depth * INCHES_TO_METERS
        self._metal_thickness_m = config.metal_layer_thickness * INCHES_TO_METERS
        self._electrode_diameter_m = config.electrode_diameter * INCHES_TO_METERS
        self._electrode_radius_m = self._electrode_diameter_m / 2
        self._insertion_depth_m = config.electrode_insertion_depth * INCHES_TO_METERS

        # Pre-compute electrode angles
        self._electrode_angles = config.get_electrode_angles_radians()

    def get_dimensions(self) -> dict[str, float]:
        """Get vessel dimensions in meters.

        Returns:
            Dictionary with dimension values in SI units (meters).
        """
        return {
            "radius": self._radius_m,
            "diameter": self._radius_m * 2,
            "glass_depth": self._glass_depth_m,
            "metal_thickness": self._metal_thickness_m,
            "total_height": self._glass_depth_m + self._metal_thickness_m,
            "electrode_diameter": self._electrode_diameter_m,
            "electrode_radius": self._electrode_radius_m,
            "insertion_depth": self._insertion_depth_m,
        }

    def create_vessel_geometry(self) -> dict:
        """Create complete vessel geometry definition.

        Returns:
            Dictionary containing geometry components:
            - glass_region: Glass pool geometry
            - metal_region: Metal layer geometry
            - electrodes: List of electrode geometries
            - boundary: Outer vessel boundary
        """
        dims = self.get_dimensions()

        # Define cylindrical regions by their vertical (Z) extents
        metal_region = {
            "type": "cylinder",
            "radius": dims["radius"],
            "z_min": 0.0,
            "z_max": dims["metal_thickness"],
            "material_id": self.MATERIAL_ID_METAL,
        }

        glass_region = {
            "type": "cylinder",
            "radius": dims["radius"],
            "z_min": dims["metal_thickness"],
            "z_max": dims["total_height"],
            "material_id": self.MATERIAL_ID_GLASS,
        }

        # Generate electrode geometry
        electrodes = []
        positions = self.get_electrode_positions()
        for i, pos in enumerate(positions):
            electrodes.append(
                {
                    "id": i + 1,
                    "type": "cylinder",
                    "radius": dims["electrode_radius"],
                    "base": pos["base"],
                    "tip": pos["tip"],
                    "angle": pos["angle"],
                    "material_id": self.MATERIAL_ID_ELECTRODE,
                }
            )

        return {
            "glass_region": glass_region,
            "metal_region": metal_region,
            "electrodes": electrodes,
            "boundary": {
                "type": "cylinder",
                "radius": dims["radius"],
                "height": dims["total_height"],
            },
        }

    def get_electrode_positions(self) -> list[dict]:
        """Calculate 3D electrode positions.

        Each electrode extends radially inward from the vessel wall.
        The tip is inside the glass, the base is at the wall.

        Returns:
            List of electrode position dictionaries with:
            - tip: (x, y, z) coordinates of electrode tip
            - base: (x, y, z) coordinates at vessel wall
            - angle: Angular position in radians
        """
        positions = []

        # Glass center height (electrodes positioned at mid-glass depth)
        glass_center_z = self._metal_thickness_m + self._glass_depth_m / 2

        for i, angle in enumerate(self._electrode_angles):
            # Pre-compute trig values
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)

            # Base position (at vessel wall)
            base_x = self._radius_m * cos_angle
            base_y = self._radius_m * sin_angle

            # Tip position (inside vessel, insertion depth from wall)
            tip_r = self._radius_m - self._insertion_depth_m
            tip_x = tip_r * cos_angle
            tip_y = tip_r * sin_angle

            positions.append(
                {
                    "tip": np.array([tip_x, tip_y, glass_center_z]),
                    "base": np.array([base_x, base_y, glass_center_z]),
                    "angle": angle,
                    "index": i,
                }
            )

        return positions

    def calculate_region_volumes(self) -> dict[str, float]:
        """Calculate volumes of material regions.

        Returns:
            Dictionary with volumes in cubic meters.
        """
        dims = self.get_dimensions()
        area = math.pi * dims["radius"] ** 2

        glass_volume = area * dims["glass_depth"]
        metal_volume = area * dims["metal_thickness"]

        return {
            "glass": glass_volume,
            "metal": metal_volume,
            "total": glass_volume + metal_volume,
        }

    def cylindrical_to_cartesian(
        self, r: float, theta: float, z: float
    ) -> tuple[float, float, float]:
        """Convert cylindrical to Cartesian coordinates.

        Args:
            r: Radial distance from axis
            theta: Angle in radians
            z: Height

        Returns:
            Tuple of (x, y, z) Cartesian coordinates.
        """
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        return (x, y, z)

    def get_region_bounds(self) -> dict[str, dict]:
        """Get Z-bounds for each material region.

        Returns:
            Dictionary with region bounds.
        """
        dims = self.get_dimensions()

        return {
            "metal": {
                "z_min": 0.0,
                "z_max": dims["metal_thickness"],
                "radius": dims["radius"],
            },
            "glass": {
                "z_min": dims["metal_thickness"],
                "z_max": dims["total_height"],
                "radius": dims["radius"],
            },
        }

    def get_material_ids(self) -> dict[str, int]:
        """Get material region IDs.

        Returns:
            Dictionary mapping region names to integer IDs.
        """
        return {
            "glass": self.MATERIAL_ID_GLASS,
            "metal": self.MATERIAL_ID_METAL,
            "electrode": self.MATERIAL_ID_ELECTRODE,
        }

    def export_geometry_data(self) -> dict:
        """Export all geometry data for mesh generation.

        Returns:
            Complete geometry specification dictionary.
        """
        dims = self.get_dimensions()
        positions = self.get_electrode_positions()
        bounds = self.get_region_bounds()

        # Format electrode data for export
        electrodes = []
        for pos in positions:
            electrodes.append(
                {
                    "index": pos["index"],
                    "tip": pos["tip"].tolist(),
                    "base": pos["base"].tolist(),
                    "angle": pos["angle"],
                    "radius": dims["electrode_radius"],
                }
            )

        return {
            "dimensions": dims,
            "electrodes": electrodes,
            "regions": bounds,
            "material_ids": self.get_material_ids(),
            "volumes": self.calculate_region_volumes(),
        }

"""Configuration dataclasses for Glass Bath FEA.

This module defines the configuration parameters for the glass bath
finite element analysis, including vessel geometry, electrode placement,
material properties, and mesh generation settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Conversion factor: inches to meters
INCHES_TO_METERS = 0.0254


@dataclass
class GlassComposition:
    """Glass composition specification (weight %).

    Standard soda-lime glass composition is the default.
    Composition affects electrical conductivity and viscosity.
    """

    sio2: float = 74.0  # Silicon dioxide
    na2o: float = 13.0  # Sodium oxide
    cao: float = 10.5  # Calcium oxide
    mgo: float = 0.0  # Magnesium oxide
    al2o3: float = 1.5  # Aluminum oxide
    fe2o3: float = 0.1  # Iron oxide (affects conductivity significantly)

    def validate(self) -> bool:
        """Ensure composition sums to approximately 100%.

        Returns:
            True if composition is valid (99-101%), False otherwise.
        """
        total = self.total_percent()
        return 99.0 <= total <= 101.0

    def total_percent(self) -> float:
        """Calculate total weight percent of all components.

        Returns:
            Sum of all oxide percentages.
        """
        return self.sio2 + self.na2o + self.cao + self.mgo + self.al2o3 + self.fe2o3


@dataclass
class MeshConfig:
    """Configuration for FEA mesh generation.

    Controls mesh density, quality, and export format for
    MATLAB PDE Toolbox compatibility.
    """

    # Element sizes in meters (smaller = finer mesh)
    element_size_glass: float = 0.01  # Glass region
    element_size_metal: float = 0.005  # Metal layer (finer for thin layer)
    element_size_electrodes: float = 0.003  # Near electrodes (finest)

    # Mesh quality settings
    mesh_algorithm: str = "delaunay"  # or "frontal"
    mesh_order: int = 1  # 1 = linear, 2 = quadratic elements
    optimize_mesh: bool = True

    # Export options
    export_format: str = "msh22"  # MSH v2.2 for MATLAB compatibility


@dataclass
class GlassBathFEAConfig:
    """Complete configuration for glass bath FEA.

    Contains all parameters needed to define the geometry, operating
    conditions, and material properties for the finite element analysis.
    """

    # Vessel geometry (inches)
    bath_diameter: float = 120.0  # Diameter of cylindrical vessel
    glass_depth: float = 15.0  # Depth of molten glass pool
    metal_layer_thickness: float = 2.0  # Thickness of metal bottom

    # Electrode configuration
    num_electrodes: int = 3  # Number of electrodes (typically 3 for delta)
    electrode_spacing_degrees: float = 120.0  # Angular spacing between electrodes
    electrode_diameter: float = 6.0  # Electrode diameter (inches)
    electrode_insertion_depth: float = 10.0  # How far electrode extends into glass

    # Operating conditions
    operating_temperature: float = 1350.0  # Operating temperature (°C)
    phase_voltages: tuple[float, float, float] = field(
        default_factory=lambda: (100.0, 100.0, 100.0)
    )  # Three-phase voltages (V)

    # Material properties
    metal_conductivity: float = 10000.0  # Metal conductivity (S/m)
    glass_composition: GlassComposition = field(default_factory=GlassComposition)

    # Mesh configuration
    mesh_config: MeshConfig = field(default_factory=MeshConfig)

    @property
    def total_height(self) -> float:
        """Total height of vessel (glass + metal) in inches."""
        return self.glass_depth + self.metal_layer_thickness

    @property
    def bath_radius(self) -> float:
        """Bath radius in inches."""
        return self.bath_diameter / 2.0

    def get_dimensions_meters(self) -> dict[str, float]:
        """Get all dimensions converted to meters.

        Returns:
            Dictionary with dimension names and values in meters.
        """
        return {
            "bath_diameter": self.bath_diameter * INCHES_TO_METERS,
            "bath_radius": self.bath_radius * INCHES_TO_METERS,
            "glass_depth": self.glass_depth * INCHES_TO_METERS,
            "metal_layer_thickness": self.metal_layer_thickness * INCHES_TO_METERS,
            "total_height": self.total_height * INCHES_TO_METERS,
            "electrode_diameter": self.electrode_diameter * INCHES_TO_METERS,
            "electrode_insertion_depth": (
                self.electrode_insertion_depth * INCHES_TO_METERS
            ),
        }

    def get_electrode_angles_radians(self) -> list[float]:
        """Get angular positions of electrodes in radians.

        Returns:
            List of electrode angles, starting at 0 radians.
        """
        spacing_rad = math.radians(self.electrode_spacing_degrees)
        return [i * spacing_rad for i in range(self.num_electrodes)]

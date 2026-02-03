"""
TRC Geometry Engine
===================

Core calculation engine for the Thermal Reactor Controller (TRC) vessel geometry.
Handles volume, mass, surface area, and residence time calculations.

Performance optimizations:
- Pre-computed constants (PI, PI/3, unit conversion factors)
- Reduced redundant calculations in layer loops
- Early validation for invalid inputs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Pre-computed constants for performance
_PI = math.pi
_PI_OVER_3 = math.pi / 3.0
_CUBIC_INCHES_TO_CUBIC_FEET = 1.0 / 1728.0  # 12^3
_SQUARE_INCHES_TO_SQUARE_FEET = 1.0 / 144.0  # 12^2


@dataclass
class LayerConfig:
    """Configuration for a single vessel layer"""

    name: str
    thickness: float  # inches
    density: float  # lb/ft^3
    color: str
    visible: bool = True
    transparency: float = 0.3
    top_section_name: str = ""

    def __post_init__(self) -> None:
        if not self.top_section_name:
            self.top_section_name = f"{self.name} Top"
        # Ensure numeric types
        self.thickness = float(self.thickness)
        self.density = float(self.density)
        self.transparency = float(self.transparency)


@dataclass
class VesselDimensions:
    """Dimensions of the vessel geometry"""

    cylinder_height: float  # inches
    cylinder_diameter: float  # inches
    cone_height: float  # inches
    cone_bottom_diameter: float  # inches
    cone_interior_hole: float  # inches
    top_refractory_thickness: float  # inches

    # Feature flags affecting calculation
    display_lid: bool = True
    display_cylinder: bool = True
    display_cone: bool = True


@dataclass
class LayerResult:
    """Calculation results for a single layer"""

    name: str
    volume_ft3: float
    mass_lb: float
    density: float
    outer_surface_area_ft2: float = 0.0


@dataclass
class VesselGeometryResult:
    """Total vessel geometry results"""

    layers: list[LayerResult] = field(default_factory=list)
    total_volume_ft3: float = 0.0
    total_mass_lb: float = 0.0
    outside_surface_area_ft2: float = 0.0
    interior_volume_ft3: float = 0.0  # Void volume for residence time
    void_radius_inches: float = 0.0
    void_diameter_inches: float = 0.0
    interior_height_inches: float = 0.0


class TRCGeometryEngine:
    """Engine for calculating TRC vessel geometry and physics.

    Performance optimizations applied:
    - Pre-computed math constants avoid repeated function calls
    - Consolidated unit conversions
    - Reduced redundant radius calculations
    """

    def calculate_geometry(
        self, dimensions: VesselDimensions, layers: list[LayerConfig]
    ) -> VesselGeometryResult:
        """
        Calculate vessel geometry properties.

        Args:
            dimensions: Vessel dimensions and flags
            layers: List of layer configurations (ordered from outer to inner)

        Returns:
            VesselGeometryResult containing detailed calculations
        """
        results = VesselGeometryResult()

        # Early exit for empty layers
        if not layers:
            return results

        # Pre-compute frequently used values
        half_cylinder_diameter = dimensions.cylinder_diameter * 0.5
        current_radius = half_cylinder_diameter
        interior_hole_radius = dimensions.cone_interior_hole * 0.5
        cone_bottom_radius = dimensions.cone_bottom_diameter * 0.5
        cone_height = dimensions.cone_height

        # Interior height is metal height minus the innermost roof thickness
        interior_height = dimensions.cylinder_height - (
            dimensions.top_refractory_thickness if dimensions.display_lid else 0
        )

        # Pre-compute cone height factor for truncated cone formula
        cone_height_factor = _PI_OVER_3 * cone_height

        total_mass = 0.0
        total_volume = 0.0
        outside_surface_area = 0.0

        # Track offset from original radius for cone calculations
        radius_offset = 0.0

        for layer in layers:
            if not layer.visible or layer.thickness <= 0:
                continue

            layer_thickness = layer.thickness
            inner_radius = max(current_radius - layer_thickness, interior_hole_radius)

            # Pre-compute squared values used multiple times
            current_radius_sq = current_radius * current_radius
            inner_radius_sq = inner_radius * inner_radius

            # Cylinder side-wall volume using interior height
            if dimensions.display_cylinder:
                # V = π * (R² - r²) * h = π * h * (R² - r²)
                cyl_vol = _PI * (current_radius_sq - inner_radius_sq) * interior_height
            else:
                cyl_vol = 0.0

            # Cone volume (differential annular truncated cone)
            if dimensions.display_cone:
                # Outer radius at bottom for this layer
                layer_cone_bottom_radius_outer = max(
                    cone_bottom_radius - radius_offset, interior_hole_radius
                )
                outer_bottom_sq = layer_cone_bottom_radius_outer * layer_cone_bottom_radius_outer

                # Inner radius at bottom for this layer
                layer_cone_bottom_radius_inner = max(
                    layer_cone_bottom_radius_outer - layer_thickness, interior_hole_radius
                )
                inner_bottom_sq = layer_cone_bottom_radius_inner * layer_cone_bottom_radius_inner

                # Truncated cone volume formula: V = (π/3) * h * (R² + Rr + r²)
                cone_outer_vol = cone_height_factor * (
                    current_radius_sq
                    + current_radius * layer_cone_bottom_radius_outer
                    + outer_bottom_sq
                )
                cone_inner_vol = cone_height_factor * (
                    inner_radius_sq
                    + inner_radius * layer_cone_bottom_radius_inner
                    + inner_bottom_sq
                )
                cone_vol = cone_outer_vol - cone_inner_vol
            else:
                cone_vol = 0.0

            # Top disk volume for each layer (if lid is displayed)
            top_disk_vol = (
                _PI * inner_radius_sq * layer_thickness if dimensions.display_lid else 0.0
            )

            # Total layer volume (convert to ft³)
            layer_volume_in3 = cyl_vol + cone_vol + top_disk_vol
            layer_volume_ft3 = layer_volume_in3 * _CUBIC_INCHES_TO_CUBIC_FEET

            layer_mass_lb = layer_volume_ft3 * layer.density

            total_mass += layer_mass_lb
            total_volume += layer_volume_ft3

            # Surface area (only for metal shell)
            layer_outer_surface = 0.0
            if layer.name == "Metal Shell":
                if dimensions.display_cylinder:
                    # Lateral surface: 2πrh
                    layer_outer_surface += (
                        2.0 * _PI * current_radius * dimensions.cylinder_height
                        * _SQUARE_INCHES_TO_SQUARE_FEET
                    )
                if dimensions.display_cone:
                    layer_cone_bottom_radius = max(
                        cone_bottom_radius - radius_offset, interior_hole_radius
                    )
                    # Slant height of cone frustum
                    radius_diff = current_radius - layer_cone_bottom_radius
                    slant_height = math.sqrt(
                        cone_height * cone_height + radius_diff * radius_diff
                    )
                    # Lateral surface of frustum: π(R + r) * slant
                    layer_outer_surface += (
                        _PI * (current_radius + layer_cone_bottom_radius) * slant_height
                        * _SQUARE_INCHES_TO_SQUARE_FEET
                    )
                outside_surface_area += layer_outer_surface

            results.layers.append(
                LayerResult(
                    name=layer.name,
                    volume_ft3=layer_volume_ft3,
                    mass_lb=layer_mass_lb,
                    density=layer.density,
                    outer_surface_area_ft2=(
                        layer_outer_surface if layer.name == "Metal Shell" else 0.0
                    ),
                )
            )

            # Update for next iteration
            radius_offset += current_radius - inner_radius
            current_radius = inner_radius

        results.total_volume_ft3 = total_volume
        results.total_mass_lb = total_mass
        results.outside_surface_area_ft2 = outside_surface_area

        # Calculate interior void volume
        last_inner_radius = current_radius
        last_inner_radius_sq = last_inner_radius * last_inner_radius

        if dimensions.display_cylinder:
            void_cyl_vol = _PI * last_inner_radius_sq * interior_height
        else:
            void_cyl_vol = 0.0

        if dimensions.display_cone:
            # Total shell thickness at cylinder wall
            total_thickness = half_cylinder_diameter - last_inner_radius

            # Void bottom radius
            void_cone_bottom_radius = max(
                cone_bottom_radius - total_thickness, interior_hole_radius
            )
            void_bottom_sq = void_cone_bottom_radius * void_cone_bottom_radius

            void_cone_vol = cone_height_factor * (
                last_inner_radius_sq
                + last_inner_radius * void_cone_bottom_radius
                + void_bottom_sq
            )
        else:
            void_cone_vol = 0.0

        results.interior_volume_ft3 = (void_cyl_vol + void_cone_vol) * _CUBIC_INCHES_TO_CUBIC_FEET
        results.void_radius_inches = last_inner_radius
        results.void_diameter_inches = last_inner_radius * 2.0
        results.interior_height_inches = interior_height

        return results

    def calculate_residence_time(
        self, volume_ft3: float, gas_flow_acfm: float
    ) -> float:
        """
        Calculate residence time in seconds.

        Args:
            volume_ft3: Interior void volume in cubic feet
            gas_flow_acfm: Gas flow rate in Actual Cubic Feet per Minute

        Returns:
            Residence time in seconds
        """
        if gas_flow_acfm <= 0:
            return 0.0

        # Volume (ft3) / Flow (ft3/min) = min
        # Convert to seconds
        return (volume_ft3 / gas_flow_acfm) * 60.0

"""
TRC Geometry Engine
===================

Core calculation engine for the Thermal Reactor Controller (TRC) vessel geometry.
Handles volume, mass, surface area, and residence time calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


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
    """Engine for calculating TRC vessel geometry and physics."""

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

        current_radius = dimensions.cylinder_diameter / 2
        interior_hole_radius = dimensions.cone_interior_hole / 2

        # Interior height is metal height minus the innermost roof thickness
        interior_height = dimensions.cylinder_height - (
            dimensions.top_refractory_thickness if dimensions.display_lid else 0
        )

        total_mass = 0.0
        total_volume = 0.0
        outside_surface_area = 0.0

        for layer in layers:
            if not layer.visible or layer.thickness <= 0:
                continue

            layer_thickness = layer.thickness
            inner_radius = current_radius - layer_thickness
            inner_radius = max(inner_radius, interior_hole_radius)

            # Cylinder side-wall volume using interior height
            if dimensions.display_cylinder:
                cyl_outer_vol = math.pi * current_radius**2 * interior_height
                cyl_inner_vol = math.pi * inner_radius**2 * interior_height
                cyl_vol = cyl_outer_vol - cyl_inner_vol
            else:
                cyl_vol = 0

            # Cone volume (differential annular truncated cone)
            if dimensions.display_cone:
                cone_bottom_radius = dimensions.cone_bottom_diameter / 2

                # Outer radius at bottom for this layer
                layer_cone_bottom_radius_outer = cone_bottom_radius - (
                    dimensions.cylinder_diameter / 2 - current_radius
                )
                layer_cone_bottom_radius_outer = max(
                    layer_cone_bottom_radius_outer, interior_hole_radius
                )

                # Inner radius at bottom for this layer
                layer_cone_bottom_radius_inner = (
                    layer_cone_bottom_radius_outer - layer_thickness
                )
                layer_cone_bottom_radius_inner = max(
                    layer_cone_bottom_radius_inner, interior_hole_radius
                )

                # Truncated cone volume formula: V = (π/3) * h * (R² + Rr + r²)
                cone_outer_vol = (
                    (math.pi / 3)
                    * dimensions.cone_height
                    * (
                        current_radius**2
                        + current_radius * layer_cone_bottom_radius_outer
                        + layer_cone_bottom_radius_outer**2
                    )
                )
                cone_inner_vol = (
                    (math.pi / 3)
                    * dimensions.cone_height
                    * (
                        inner_radius**2
                        + inner_radius * layer_cone_bottom_radius_inner
                        + layer_cone_bottom_radius_inner**2
                    )
                )
                cone_vol = cone_outer_vol - cone_inner_vol
            else:
                cone_vol = 0

            # Top disk volume for each layer (if lid is displayed)
            top_disk_vol = (
                math.pi * inner_radius**2 * layer_thickness
                if dimensions.display_lid
                else 0
            )

            # Total layer volume (convert to ft³)
            # Inputs are in inches, so divide by 1728 (12^3)
            layer_volume_in3 = cyl_vol + cone_vol + top_disk_vol
            layer_volume_ft3 = layer_volume_in3 / 1728.0

            layer_mass_lb = layer_volume_ft3 * layer.density

            total_mass += layer_mass_lb
            total_volume += layer_volume_ft3

            # Surface inputs
            layer_outer_surface = 0.0
            if layer.name == "Metal Shell":
                # Outside surface area (only for metal shell)
                if dimensions.display_cylinder:
                    # 2*pi*r*h / 144 (ft²)
                    layer_outer_surface += (
                        2 * math.pi * current_radius * dimensions.cylinder_height / 144
                    )
                if dimensions.display_cone:
                    cone_bottom_radius = dimensions.cone_bottom_diameter / 2
                    layer_cone_bottom_radius = cone_bottom_radius - (
                        dimensions.cylinder_diameter / 2 - current_radius
                    )
                    slant_height = math.sqrt(
                        dimensions.cone_height**2
                        + (current_radius - layer_cone_bottom_radius) ** 2
                    )
                    layer_outer_surface += (
                        math.pi
                        * (current_radius + layer_cone_bottom_radius)
                        * slant_height
                        / 144
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

            current_radius = inner_radius

        results.total_volume_ft3 = total_volume
        results.total_mass_lb = total_mass
        results.outside_surface_area_ft2 = outside_surface_area

        # Calculate interior void volume (approximate based on last inner radius)
        last_inner_radius = current_radius

        if dimensions.display_cylinder:
            void_cyl_vol = math.pi * last_inner_radius**2 * interior_height
        else:
            void_cyl_vol = 0

        if dimensions.display_cone:
            # Total shell thickness at cylinder wall
            total_thickness = (dimensions.cylinder_diameter / 2) - last_inner_radius

            # Void bottom radius
            void_cone_bottom_radius = (
                dimensions.cone_bottom_diameter / 2
            ) - total_thickness
            void_cone_bottom_radius = max(void_cone_bottom_radius, interior_hole_radius)

            void_cone_vol = (
                (math.pi / 3)
                * dimensions.cone_height
                * (
                    last_inner_radius**2
                    + last_inner_radius * void_cone_bottom_radius
                    + void_cone_bottom_radius**2
                )
            )
        else:
            void_cone_vol = 0

        results.interior_volume_ft3 = (void_cyl_vol + void_cone_vol) / 1728.0
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

"""Three-Phase Electrical Model for Electrode Systems

Enhanced electrical model that calculates system states, resistance, and current
distribution with support for multiple conductive path geometries.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from .config import ElectrodeConfig
from .glass_interface import GlassPropertiesInterface

logger = logging.getLogger(__name__)


class ThreePhaseElectricalModelEnhanced:
    """Enhanced 3-phase delta model with new conductive path geometry.

    Performance optimizations:
    - Electrode position caching to avoid redundant 3D calculations
    - Vectorized resistance calculations
    """

    def __init__(
        self,
        config: ElectrodeConfig,
        glass_interface: GlassPropertiesInterface,
    ) -> None:
        self.config = config
        self.glass_interface = glass_interface
        self.electrode_positions = np.array([0, 120, 240]) * np.pi / 180  # radians
        self.power_history: deque[float] = deque(maxlen=100)
        # Cache for electrode positions (avoids recalculation when params unchanged)
        self._position_cache_key: tuple | None = None
        self._position_cache_value: list[dict] | None = None

    def calculate_system_state(
        self,
        depths: np.ndarray,
        bath_diameter: float,
        tip_diameter: float,
        metal_depth: float,
        k_factors: dict[str, float],
        bath_temperature: float = 1200,
        voltages: np.ndarray | None = None,
        conductive_height: float = 2.0,
        metal_conductive: bool = True,
    ) -> dict:
        """Calculate complete electrical system state with new path model"""
        # Electrode tip positions
        r_bath = bath_diameter / 2.0
        tip_radius = tip_diameter / 2.0

        # Calculate 3D electrode positions
        electrode_positions = self._calculate_electrode_positions_3d(
            depths,
            r_bath,
            metal_depth,
        )

        # Calculate resistances using new path model
        resistances = {}
        current_paths = {}

        # Calculate the 6 paths (3 direct glass + 3 via-metal)
        for i in range(3):
            j = (i + 1) % 3
            phase_key = f"{i + 1}-{j + 1}"

            # 1. Direct glass conduction path (trapezoidal prism)
            direct_resistance = self._calculate_trapezoidal_path_resistance(
                electrode_positions[i],
                electrode_positions[j],
                tip_radius,
                conductive_height,
                bath_temperature,
                r_bath,
            )

            # 2. Via-metal path (3-segment composite)
            # Only calculate if metal conductivity is enabled
            if metal_conductive:
                via_metal_resistance = self._calculate_via_metal_path_resistance(
                    electrode_positions[i],
                    electrode_positions[j],
                    metal_depth,
                    tip_radius,
                    bath_temperature,
                    r_bath,
                )
                # Total resistance is parallel combination
                total_resistance = self._parallel_resistance(
                    direct_resistance,
                    via_metal_resistance,
                )

                # Calculate current fractions properly
                if (direct_resistance + via_metal_resistance) > 0:
                    direct_fraction = via_metal_resistance / (
                        direct_resistance + via_metal_resistance
                    )
                    metal_fraction = direct_resistance / (
                        direct_resistance + via_metal_resistance
                    )
                else:
                    direct_fraction = 0.5
                    metal_fraction = 0.5
            else:
                # Metal layer not conductive - only direct glass path
                via_metal_resistance = np.inf  # Infinite resistance (no conduction)
                total_resistance = direct_resistance  # Only glass path conducts
                direct_fraction = 1.0  # All current through glass
                metal_fraction = 0.0  # No current through metal

            resistances[phase_key] = total_resistance

            current_paths[phase_key] = {
                "direct_glass": direct_resistance,
                "via_metal": via_metal_resistance,
                "total": total_resistance,
                "direct_fraction": direct_fraction,
                "metal_fraction": metal_fraction,
            }

        # Current distribution analysis
        current_distribution = self._analyze_current_distribution_new(current_paths)

        # Calculate actual currents if voltages are provided
        actual_currents = (
            self._calculate_path_currents(resistances, voltages)
            if voltages is not None
            else None
        )

        return {
            "resistances": resistances,
            "current_paths": current_paths,
            "current_distribution": current_distribution,
            "electrode_positions": electrode_positions,
            "actual_currents": actual_currents,
            "path_geometry": {
                "conductive_height": conductive_height,
                "metal_depth": metal_depth,
            },
        }

    def _calculate_electrode_positions_3d(
        self,
        depths: np.ndarray,
        r_bath: float,
        metal_depth: float,
    ) -> list[dict]:
        """Calculate 3D electrode positions including tip and base locations.

        Performance: Results are cached and reused when parameters unchanged.
        """
        # Build cache key from parameters
        cache_key = (tuple(depths), r_bath, metal_depth, self.config.glass_depth)

        # Return cached value if parameters match
        if (
            self._position_cache_key == cache_key
            and self._position_cache_value is not None
        ):
            return self._position_cache_value

        # Calculate positions
        positions = []
        glass_center_z = metal_depth + self.config.glass_depth / 2

        for i in range(3):
            angle = self.electrode_positions[i]
            depth = depths[i]

            # Pre-compute trig values (used twice per electrode)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            # Electrode tip position (inside vessel)
            tip_x = (r_bath - depth) * cos_angle
            tip_y = (r_bath - depth) * sin_angle

            # Electrode base position (at vessel wall)
            base_x = r_bath * cos_angle
            base_y = r_bath * sin_angle

            positions.append(
                {
                    "tip": np.array([tip_x, tip_y, glass_center_z]),
                    "base": np.array([base_x, base_y, glass_center_z]),
                    "angle": angle,
                    "depth": depth,
                },
            )

        # Update cache
        self._position_cache_key = cache_key
        self._position_cache_value = positions

        return positions

    def _calculate_trapezoidal_path_resistance(
        self,
        electrode1_pos: dict,
        electrode2_pos: dict,
        electrode_radius: float,
        conductive_height: float,
        temperature: float,
        bath_radius: float,
    ) -> float:
        """Calculate resistance through the trapezoidal prism formed by:
        - Electrode 1 from glass wall to tip
        - Electrode 2 from glass wall to tip
        - Line connecting the tips
        - Line connecting the glass wall entry points
        This trapezoid is extruded vertically by conductive_height × vertical_spreading_factor

        Performance: Vectorized numpy operations replace 30-iteration loop.
        """
        # Get glass wall intersection points
        e1_angle = electrode1_pos["angle"]
        e1_wall_glass = np.array(
            [
                bath_radius * np.cos(e1_angle),
                bath_radius * np.sin(e1_angle),
                electrode1_pos["tip"][2],
            ],
        )

        e2_angle = electrode2_pos["angle"]
        e2_wall_glass = np.array(
            [
                bath_radius * np.cos(e2_angle),
                bath_radius * np.sin(e2_angle),
                electrode2_pos["tip"][2],
            ],
        )

        # Extract key positions
        e1_tip = electrode1_pos["tip"]
        e2_tip = electrode2_pos["tip"]

        # Apply vertical spreading factor to conductive height
        effective_height = conductive_height * self.config.vertical_spreading_factor

        # Get glass conductivity (single call - temperature is constant across segments)
        conductivity = self.glass_interface.get_conductivity(temperature)  # S/m

        # Vectorized calculation for all segments
        num_segments = 30

        # Pre-compute direction vectors
        wall_diff = e2_wall_glass - e1_wall_glass
        tip_diff = e2_tip - e1_tip

        # Generate all t values at segment centers: (0.5, 1.5, ..., 29.5) / 30
        t_values = (np.arange(num_segments) + 0.5) / num_segments

        # Vectorized interpolation: wall_positions[i] = e1_wall_glass + t[i] * wall_diff
        # Shape: (num_segments, 3)
        wall_positions = e1_wall_glass + np.outer(t_values, wall_diff)
        tip_positions = e1_tip + np.outer(t_values, tip_diff)

        # Section widths: distance from wall to tip at each segment
        # Shape: (num_segments,)
        section_widths = np.linalg.norm(tip_positions - wall_positions, axis=1)

        # Cross-sectional areas in m²
        cross_section_areas_m2 = section_widths * effective_height * 0.00064516

        # Segment distances (uniform for trapezoidal approximation)
        # All interior segments use the same distance
        base_segment_distance = np.linalg.norm(wall_diff) / num_segments
        segment_distance_m = base_segment_distance * 0.0254  # Convert to m

        # Calculate resistances for all segments at once
        # R = L / (σ * A), where L is distance, σ is conductivity, A is area
        # Avoid division by zero
        valid_mask = cross_section_areas_m2 > 0
        segment_resistances = np.zeros(num_segments)
        segment_resistances[valid_mask] = segment_distance_m / (
            conductivity * cross_section_areas_m2[valid_mask]
        )

        return float(np.sum(segment_resistances))

    def _calculate_via_metal_path_resistance(
        self,
        electrode1_pos: dict,
        electrode2_pos: dict,
        metal_depth: float,
        electrode_radius: float,
        temperature: float,
        bath_radius: float,
    ) -> float:
        """Calculate resistance of via-metal path with correct geometry:
        - Segment 1: Rectangular extrusion down from E1 glass portion
        - Segment 2: Through metal layer (wide path)
        - Segment 3: Rectangular extrusion up to E2 glass portion
        Vertical segments use horizontal_spreading_factor for width
        """
        # Get glass wall positions
        e1_angle = electrode1_pos["angle"]
        e1_wall = np.array(
            [
                bath_radius * np.cos(e1_angle),
                bath_radius * np.sin(e1_angle),
                electrode1_pos["tip"][2],
            ],
        )

        e2_angle = electrode2_pos["angle"]
        e2_wall = np.array(
            [
                bath_radius * np.cos(e2_angle),
                bath_radius * np.sin(e2_angle),
                electrode2_pos["tip"][2],
            ],
        )

        # Get electrode dimensions within glass bath
        e1_length = np.linalg.norm(electrode1_pos["tip"] - e1_wall)
        e2_length = np.linalg.norm(electrode2_pos["tip"] - e2_wall)

        # Apply horizontal spreading factor for vertical segments
        effective_width = 2 * electrode_radius * self.config.horizontal_spreading_factor

        # Segment 1: Down from E1 glass portion
        # Cross-sectional area = electrode length × effective width
        area_down = e1_length * effective_width  # inches²
        area_down_m2 = area_down * 0.00064516  # Convert to m²

        # Vertical distance from electrode center to metal top
        electrode_z = electrode1_pos["tip"][2]
        vertical_distance_1 = abs(electrode_z - metal_depth)
        distance_1_m = vertical_distance_1 * 0.0254  # Convert to m

        conductivity_glass = self.glass_interface.get_conductivity(temperature)

        if area_down_m2 > 0 and distance_1_m > 0:
            resistance_1: float = float(
                distance_1_m / (conductivity_glass * area_down_m2)
            )
        else:
            resistance_1 = 0.001

        # Segment 2: Through metal layer
        # Use center-to-center distance between electrodes within glass
        center1 = (electrode1_pos["tip"] + e1_wall) / 2
        center2 = (electrode2_pos["tip"] + e2_wall) / 2
        horizontal_distance = np.linalg.norm(center2[:2] - center1[:2])
        distance_2_m = horizontal_distance * 0.0254  # Convert to m

        # Metal path is wide - use average of electrode lengths
        avg_electrode_length = (e1_length + e2_length) / 2
        metal_path_width = avg_electrode_length + effective_width  # Wider in metal
        metal_path_height = 2.0  # inches (typical metal layer thickness)
        area_metal = metal_path_width * metal_path_height  # inches²
        area_metal_m2 = area_metal * 0.00064516  # Convert to m²

        conductivity_metal = self.glass_interface.get_conductivity(
            temperature,
            is_metal=True,
        )

        if area_metal_m2 > 0 and distance_2_m > 0:
            resistance_2: float = float(
                distance_2_m / (conductivity_metal * area_metal_m2)
            )
        else:
            resistance_2 = 0.0001

        # Segment 3: Up to E2 glass portion
        area_up = e2_length * effective_width  # inches²
        area_up_m2 = area_up * 0.00064516  # Convert to m²

        electrode2_z = electrode2_pos["tip"][2]
        vertical_distance_3 = abs(electrode2_z - metal_depth)
        distance_3_m = vertical_distance_3 * 0.0254  # Convert to m

        if area_up_m2 > 0 and distance_3_m > 0:
            resistance_3: float = float(
                distance_3_m / (conductivity_glass * area_up_m2)
            )
        else:
            resistance_3 = 0.001

        # Total resistance is sum of three segments
        return resistance_1 + resistance_2 + resistance_3

    def _analyze_current_distribution_new(self, current_paths: dict) -> dict:
        """Analyze current distribution with new path model"""
        analysis = {}

        for phase, paths in current_paths.items():
            # Get fractions already calculated
            direct_fraction = paths["direct_fraction"]
            metal_fraction = paths["metal_fraction"]

            # Calculate power dissipation ratio
            # P = I²R, and current splits inversely with resistance
            direct_power = direct_fraction**2 * paths["direct_glass"]
            metal_power = metal_fraction**2 * paths["via_metal"]
            total_power = direct_power + metal_power

            analysis[phase] = {
                "direct_glass_fraction": direct_fraction,
                "via_metal_fraction": metal_fraction,
                "direct_glass_power_fraction": (
                    direct_power / total_power if total_power > 0 else 0.5
                ),
                "via_metal_power_fraction": (
                    metal_power / total_power if total_power > 0 else 0.5
                ),
                "resistance_ratio": (
                    paths["direct_glass"] / paths["via_metal"]
                    if paths["via_metal"] > 0
                    else np.inf
                ),
            }

        return analysis

    def _parallel_resistance(self, r1: float, r2: float) -> float:
        """Calculate parallel resistance safely"""
        if np.isnan(r1) or np.isnan(r2) or r1 <= 0 or r2 <= 0:
            return max(r1, r2) if not (np.isnan(r1) or np.isnan(r2)) else np.nan
        return (r1 * r2) / (r1 + r2)

    def _calculate_path_currents(self, resistances: dict, voltages: np.ndarray) -> dict:
        """Calculate actual currents through each path using Ohm's law"""
        try:
            # Phase voltages (line-to-line)
            phase_currents = {}

            # For each phase, calculate current using I = V/R
            phase_names = ["1-2", "2-3", "3-1"]
            for i, phase in enumerate(phase_names):
                if i < len(voltages) and phase in resistances:
                    voltage = voltages[i]
                    resistance = resistances[phase]
                    current = voltage / resistance if resistance > 0 else 0.0
                    phase_currents[phase] = current
                else:
                    phase_currents[phase] = 0.0

            return phase_currents
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return {"1-2": 0.0, "2-3": 0.0, "3-1": 0.0}

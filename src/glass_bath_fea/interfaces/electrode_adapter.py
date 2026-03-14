"""Adapter for electrode adviser integration.

This module provides an adapter interface to reuse the electrical
calculations from the electrode adviser shared module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from glass_bath_fea.core.config import GlassBathFEAConfig

# Conversion factor
INCHES_TO_METERS = 0.0254


class ElectrodeAdapter:
    """Adapter to interface with electrode adviser calculations.

    Provides access to the shared electrical model while maintaining
    the FEA configuration interface.

    Attributes:
        config: FEA configuration parameters
    """

    def __init__(self, config: GlassBathFEAConfig) -> None:
        """Initialize adapter with FEA configuration.

        Args:
            config: Glass bath FEA configuration
        """
        assert config is not None, "config must be provided"
        self.config = config
        self._electrode_config: Any = None
        self._glass_interface: Any = None
        self._electrical_model: Any = None

    def get_electrode_config(self) -> Any:
        """Get or create electrode adviser compatible config.

        Returns:
            ElectrodeConfig instance from electrode adviser.
        """
        if self._electrode_config is None:
            from upstream_drift_tools.calculators.electrical.config import (
                ElectrodeConfig,
            )

            self._electrode_config = ElectrodeConfig(
                bath_diameter=self.config.bath_diameter,
                glass_depth=self.config.glass_depth,
                tip_diameter=self.config.electrode_diameter,
            )

        return self._electrode_config

    def get_glass_interface(self) -> Any:
        """Get or create glass properties interface.

        Returns:
            GlassPropertiesInterface instance.
        """
        if self._glass_interface is None:
            from upstream_drift_tools.calculators.electrical.glass_interface import (
                GlassPropertiesInterface,
            )

            # Create interface with default calculator
            self._glass_interface = GlassPropertiesInterface()

        return self._glass_interface

    def get_electrical_model(self) -> Any:
        """Get or create electrical model instance.

        Returns:
            ThreePhaseElectricalModelEnhanced instance.
        """
        if self._electrical_model is None:
            from upstream_drift_tools.calculators.electrical.electrical_model import (
                ThreePhaseElectricalModelEnhanced,
            )

            self._electrical_model = ThreePhaseElectricalModelEnhanced(
                config=self.get_electrode_config(),
                glass_interface=self.get_glass_interface(),
            )

        return self._electrical_model

    def calculate_electrode_positions(
        self, depths: np.ndarray | None = None
    ) -> list[dict[str, Any]]:
        """Calculate electrode positions using electrode adviser logic.

        Args:
            depths: Array of electrode insertion depths in inches.
                   If None, uses default from config.

        Returns:
            List of electrode position dictionaries.
        """
        # DbC preconditions
        assert (
            self.config.bath_diameter > 0
        ), f"Bath diameter must be positive, got {self.config.bath_diameter}"
        if depths is not None:
            assert len(depths) > 0, "depths array must be non-empty"
            assert all(d >= 0 for d in depths), "All depths must be non-negative"

        model = self.get_electrical_model()

        if depths is None:
            depths = np.array(
                [
                    self.config.electrode_insertion_depth,
                    self.config.electrode_insertion_depth,
                    self.config.electrode_insertion_depth,
                ]
            )

        # Use electrode adviser's position calculation
        r_bath = self.config.bath_diameter / 2
        metal_depth = self.config.metal_layer_thickness

        positions = model._calculate_electrode_positions_3d(
            depths=depths,
            r_bath=r_bath,
            metal_depth=metal_depth,
        )

        return cast(list[dict[str, Any]], positions)

    def get_glass_conductivity(self, temperature_celsius: float) -> float:
        """Get glass conductivity at specified temperature.

        Args:
            temperature_celsius: Temperature in degrees Celsius.

        Returns:
            Electrical conductivity in S/m.
        """
        assert temperature_celsius is not None, "temperature_celsius must be provided"
        glass = self.get_glass_interface()

        # Build composition dict for the interface
        composition = {
            "na2o": self.config.glass_composition.na2o,
            "fe2o3": self.config.glass_composition.fe2o3,
        }

        result = glass.get_conductivity(temperature_celsius, composition=composition)
        return float(result)

    def calculate_phase_resistances(
        self, depths: np.ndarray | None = None
    ) -> dict[str, float]:
        """Calculate phase-to-phase resistances.

        Args:
            depths: Electrode insertion depths. If None, uses defaults.

        Returns:
            Dictionary of phase-pair resistances.
        """
        results = self.calculate_system_state(depths)

        # Results from electrical model have a "resistances" key
        if "resistances" in results:
            return cast(dict[str, float], results["resistances"])

        return {}

    def calculate_system_state(
        self, depths: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Calculate full electrical system state.

        Args:
            depths: Electrode insertion depths. If None, uses defaults.

        Returns:
            Dictionary with system state results.
        """
        model = self.get_electrical_model()

        if depths is None:
            depths = np.array(
                [
                    self.config.electrode_insertion_depth,
                    self.config.electrode_insertion_depth,
                    self.config.electrode_insertion_depth,
                ]
            )

        results = model.calculate_system_state(
            depths=depths,
            bath_diameter=self.config.bath_diameter,
            tip_diameter=self.config.electrode_diameter,
            metal_depth=self.config.metal_layer_thickness,
            k_factors={"K_tt": 1.0, "K_vert": 1.0},
            bath_temperature=self.config.operating_temperature,
            voltages=np.array(self.config.phase_voltages),
            conductive_height=self.config.glass_depth,
        )

        return cast(dict[str, Any], results)

    def export_boundary_conditions(self, output_path: Path | str) -> None:
        """Export boundary condition data for MATLAB.

        Args:
            output_path: Path to output .mat file.
        """
        assert output_path is not None, "output_path must be provided"
        from scipy.io import savemat

        # Get electrode positions
        positions = self.calculate_electrode_positions()

        # Prepare data for MATLAB
        electrode_voltages = np.array(self.config.phase_voltages)
        electrode_angles = np.array([pos["angle"] for pos in positions])

        # Convert positions to arrays
        tip_positions = np.array([pos["tip"] for pos in positions])
        base_positions = np.array([pos["base"] for pos in positions])

        data = {
            "electrode_voltages": electrode_voltages,
            "phase_voltages": electrode_voltages,  # Alias
            "electrode_angles": electrode_angles,
            "tip_positions": tip_positions,
            "base_positions": base_positions,
            "num_electrodes": np.array([self.config.num_electrodes]),
            "operating_temperature": np.array([self.config.operating_temperature]),
            "electrode_diameter": np.array(
                [self.config.electrode_diameter * INCHES_TO_METERS]
            ),
        }

        savemat(str(output_path), data, format="5")

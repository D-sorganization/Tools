"""Geometry synchronization between Electrode Advisor and Glass Bath FEA.

This module translates electrode configurations from the Electrode Advisor
(shared electrical model) into FEA-compatible geometry definitions, ensuring
that electrode positions, diameters, insertion depths, and bath dimensions
are consistent across both systems.

See issue #575.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glass_bath_fea.core.config import GlassBathFEAConfig, GlassComposition

# Conversion factors
INCHES_TO_METERS = 0.0254
METERS_TO_INCHES = 1.0 / INCHES_TO_METERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation results
# ---------------------------------------------------------------------------


@dataclass
class GeometryValidationResult:
    """Result of geometry compatibility validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Synchronizer
# ---------------------------------------------------------------------------


class GeometrySynchronizer:
    """Synchronise Electrode Advisor geometry with Glass Bath FEA.

    Reads an ``ElectrodeConfig`` (from the shared electrical model) and
    produces a fully configured ``GlassBathFEAConfig`` with matching
    vessel dimensions, electrode positions, and operating conditions.

    Attributes:
        electrode_config: Source electrode configuration.
        fea_config: Target FEA configuration (generated on sync).
    """

    # Physics constraints used during validation
    MIN_INSERTION_DEPTH_INCHES = 1.0
    MAX_INSERTION_RATIO = 0.9  # electrode cannot reach > 90% across the radius
    MIN_ELECTRODE_CLEARANCE_INCHES = 2.0  # minimum tip-to-tip clearance

    def __init__(self, electrode_config: Any | None = None) -> None:
        """Initialise with an optional ``ElectrodeConfig``.

        Args:
            electrode_config: An ``ElectrodeConfig`` instance from the
                electrode advisor shared module.  If *None*, a default
                config will be created on first use.
        """
        self._electrode_config = electrode_config
        self.fea_config: GlassBathFEAConfig | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sync(self) -> GlassBathFEAConfig:
        """Translate electrode config into a GlassBathFEAConfig.

        Returns:
            A new ``GlassBathFEAConfig`` whose dimensions, electrode
            parameters, and operating conditions match the electrode
            advisor source.

        Raises:
            ValueError: If the electrode configuration is geometrically
                incompatible (e.g. insertion depth exceeds bath radius).
        """
        ec = self._get_electrode_config()

        # Validate before building
        validation = self.validate(ec)
        if not validation.is_valid:
            msg = "Geometry validation failed: " + "; ".join(validation.errors)
            raise ValueError(msg)

        for w in validation.warnings:
            logger.warning("Geometry sync warning: %s", w)

        # Build FEA config from electrode advisor parameters
        fea = GlassBathFEAConfig(
            bath_diameter=ec.bath_diameter,
            glass_depth=ec.glass_depth,
            metal_layer_thickness=getattr(ec, "metal_depth", 2.0),
            num_electrodes=(
                len(ec.electrode_depths) if hasattr(ec, "electrode_depths") else 3
            ),
            electrode_spacing_degrees=ec.electrode_spacing_degrees,
            electrode_diameter=ec.tip_diameter,
            electrode_insertion_depth=self._compute_insertion_depth(ec),
            operating_temperature=ec.bath_temperature,
            phase_voltages=tuple(float(v) for v in ec.phase_voltages[:3]),
            metal_conductivity=ec.metal_conductivity,
            glass_composition=GlassComposition(),  # default soda-lime
        )

        self.fea_config = fea
        return fea

    def validate(self, electrode_config: Any | None = None) -> GeometryValidationResult:
        """Validate geometry compatibility between the two systems.

        Args:
            electrode_config: Config to validate.  Uses the stored config
                if *None*.

        Returns:
            A ``GeometryValidationResult`` with errors and warnings.
        """
        ec = electrode_config or self._get_electrode_config()
        errors: list[str] = []
        warnings: list[str] = []

        # --- dimensional checks ---
        bath_radius = ec.bath_diameter / 2.0
        insertion = self._compute_insertion_depth(ec)

        if insertion < self.MIN_INSERTION_DEPTH_INCHES:
            errors.append(
                f"Insertion depth {insertion:.2f} in is below minimum "
                f"({self.MIN_INSERTION_DEPTH_INCHES} in)"
            )

        if insertion > self.MAX_INSERTION_RATIO * bath_radius:
            errors.append(
                f"Insertion depth {insertion:.2f} in exceeds "
                f"{self.MAX_INSERTION_RATIO * 100:.0f}% of bath radius "
                f"({bath_radius:.1f} in)"
            )

        if ec.tip_diameter >= bath_radius:
            errors.append(
                f"Electrode diameter ({ec.tip_diameter:.1f} in) must be "
                f"smaller than bath radius ({bath_radius:.1f} in)"
            )

        # Check electrode tip clearance
        clearance = self._compute_minimum_tip_clearance(ec)
        if clearance < self.MIN_ELECTRODE_CLEARANCE_INCHES:
            warnings.append(
                f"Minimum electrode tip clearance is only {clearance:.2f} in "
                f"(recommended > {self.MIN_ELECTRODE_CLEARANCE_INCHES} in)"
            )

        # Glass depth sanity
        if ec.glass_depth <= 0:
            errors.append("Glass depth must be positive")
        elif ec.glass_depth < ec.tip_diameter:
            warnings.append(
                "Glass depth is less than electrode diameter -- "
                "electrode may not be fully submerged"
            )

        return GeometryValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def get_electrode_positions_fea(self) -> list[dict[str, Any]]:
        """Return electrode positions in FEA coordinate space (metres).

        Must call ``sync()`` first.

        Returns:
            List of dicts with *tip*, *base*, *angle*, *diameter_m* keys.
        """
        if self.fea_config is None:
            self.sync()
        assert self.fea_config is not None

        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(self.fea_config)
        raw = gen.get_electrode_positions()

        result = []
        for pos in raw:
            result.append(
                {
                    "tip": pos["tip"].tolist(),
                    "base": pos["base"].tolist(),
                    "angle_rad": pos["angle"],
                    "angle_deg": math.degrees(pos["angle"]),
                    "diameter_m": self.fea_config.electrode_diameter * INCHES_TO_METERS,
                    "insertion_depth_m": (
                        self.fea_config.electrode_insertion_depth * INCHES_TO_METERS
                    ),
                }
            )
        return result

    def export_sync_report(self, path: str | Path) -> None:
        """Write a JSON report of the synchronised geometry.

        Args:
            path: Output file path.
        """
        if self.fea_config is None:
            self.sync()
        assert self.fea_config is not None

        from glass_bath_fea.core.geometry_generator import GeometryGenerator

        gen = GeometryGenerator(self.fea_config)
        geo_data = gen.export_geometry_data()

        ec = self._get_electrode_config()
        report = {
            "source": "electrode_advisor",
            "target": "glass_bath_fea",
            "electrode_config": {
                "bath_diameter_in": ec.bath_diameter,
                "glass_depth_in": ec.glass_depth,
                "tip_diameter_in": ec.tip_diameter,
                "electrode_spacing_deg": ec.electrode_spacing_degrees,
                "bath_temperature_c": ec.bath_temperature,
            },
            "fea_config": {
                "bath_diameter_in": self.fea_config.bath_diameter,
                "glass_depth_in": self.fea_config.glass_depth,
                "electrode_diameter_in": self.fea_config.electrode_diameter,
                "insertion_depth_in": self.fea_config.electrode_insertion_depth,
                "operating_temperature_c": self.fea_config.operating_temperature,
            },
            "fea_geometry": geo_data,
            "validation": self.validate().__dict__,
        }

        out = Path(path)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Sync report written to %s", out)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_electrode_config(self) -> Any:
        """Lazy-load or return the stored electrode config."""
        if self._electrode_config is None:
            from upstream_drift_tools.calculators.electrical.config import (
                ElectrodeConfig,
            )

            self._electrode_config = ElectrodeConfig()
        return self._electrode_config

    @staticmethod
    def _compute_insertion_depth(ec: Any) -> float:
        """Derive insertion depth in inches from electrode config.

        If the config carries explicit per-electrode depths we take the
        mean; otherwise we fall back to a reasonable default (40% of
        bath radius).
        """
        if hasattr(ec, "electrode_depths"):
            depths = np.asarray(ec.electrode_depths)
            nonzero = depths[depths > 0]
            if len(nonzero) > 0:
                return float(np.mean(nonzero))
        # Fallback: 40% of radius
        return float(ec.bath_diameter / 2.0 * 0.4)

    @staticmethod
    def _compute_minimum_tip_clearance(ec: Any) -> float:
        """Compute minimum clearance between electrode tips (inches)."""
        spacing_deg = ec.electrode_spacing_degrees
        bath_radius = ec.bath_diameter / 2.0
        insertion = GeometrySynchronizer._compute_insertion_depth(ec)
        tip_r = bath_radius - insertion

        # Chord distance between adjacent tips
        chord = 2.0 * tip_r * math.sin(math.radians(spacing_deg / 2.0))

        # Subtract one electrode diameter from the chord for clearance
        clearance = chord - ec.tip_diameter
        return float(max(clearance, 0.0))

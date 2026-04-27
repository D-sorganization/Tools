"""Unified Model Generation Package for URDF and Physics Simulation.

This package provides comprehensive tools for creating, editing, and
converting robot models in URDF and other formats.

Lazy-loading strategy (see issue #611):
    All heavy imports are deferred to first access via ``__getattr__``.
    This breaks the circular import chain:
        __init__ -> builders.base_builder -> core.contracts
        -> core.validation -> core.contracts
    Only lightweight constants are imported eagerly.

Refactored in issue #1696:
    The lazy-import dispatch table has been moved to ``_lazy_map.py`` and
    the convenience functions to ``_convenience.py`` to keep this file
    below 120 lines.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"
__author__ = "Golf Modeling Suite"

# --- Only lightweight constants are imported eagerly ---
from model_generation._convenience import quick_build, quick_urdf
from model_generation._lazy_map import LAZY_IMPORTS
from model_generation.core.constants import (
    DEFAULT_DENSITY_KG_M3,
    DEFAULT_HEIGHT_M,
    DEFAULT_INERTIA_KG_M2,
    DEFAULT_MASS_KG,
    GRAVITY_M_S2,
)

__all__ = [
    # Version
    "__version__",
    # Constants
    "GRAVITY_M_S2",
    "DEFAULT_DENSITY_KG_M3",
    "DEFAULT_INERTIA_KG_M2",
    "DEFAULT_HEIGHT_M",
    "DEFAULT_MASS_KG",
    # Convenience functions
    "quick_urdf",
    "quick_build",
    # All lazy-loaded names
    *LAZY_IMPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    """Lazy-load attributes on first access (see issue #611)."""
    if name in LAZY_IMPORTS:
        module_path, attr_name = LAZY_IMPORTS[name]
        # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache on the module so subsequent accesses are fast
        globals()[name] = value
        return value
    raise AttributeError(f"module 'model_generation' has no attribute {name!r}")
<<<<<<< HEAD
=======


# ---------------------------------------------------------------------------
# Convenience functions (use lazy imports internally)
# ---------------------------------------------------------------------------


def quick_urdf(
    height_m: float = DEFAULT_HEIGHT_M,
    mass_kg: float = DEFAULT_MASS_KG,
    preset: str | None = None,
    robot_name: str = "humanoid",
) -> str:
    """
    Generate a humanoid URDF quickly with minimal configuration.

    Args:
        height_m: Height in meters
        mass_kg: Mass in kg
        preset: Optional preset name (athletic, average, heavy, lean)
        robot_name: Name for the robot element

    Returns:
        URDF XML string

    Example:
        urdf = quick_urdf(height_m=1.85, preset="athletic")
    """
    from model_generation.builders.parametric_builder import ParametricBuilder

    builder = ParametricBuilder(robot_name)

    if preset:
        presets = {
            "athletic": {"gender_factor": 0.7, "shoulder_width_factor": 1.1},
            "average": {"gender_factor": 0.5},
            "heavy": {"gender_factor": 0.5, "hip_width_factor": 1.15},
            "lean": {"gender_factor": 0.5, "shoulder_width_factor": 0.95},
        }
        preset_config = presets.get(preset.lower(), {})
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg, **preset_config)
    else:
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg)

    builder.add_humanoid_segments()
    result = builder.build()

    if not result.success:
        raise ValueError(f"Failed to generate URDF: {result.error_message}")

    return str(result.urdf_xml)


def quick_build(
    height_m: float = DEFAULT_HEIGHT_M,
    mass_kg: float = DEFAULT_MASS_KG,
    preset: str | None = None,
    output_path: str | None = None,
) -> Any:
    """
    Build a humanoid model quickly with minimal configuration.

    Args:
        height_m: Height in meters
        mass_kg: Mass in kg
        preset: Optional preset name
        output_path: Optional path to save URDF

    Returns:
        BuildResult with URDF and metadata

    Example:
        result = quick_build(height_m=1.80, output_path="./humanoid.urdf")
    """
    if not (height_m is not None):
        raise ValueError("height_m must be provided")
    from pathlib import Path

    from model_generation.builders.parametric_builder import ParametricBuilder

    builder = ParametricBuilder("humanoid")

    if preset:
        presets = {
            "athletic": {"gender_factor": 0.7, "shoulder_width_factor": 1.1},
            "average": {"gender_factor": 0.5},
            "heavy": {"gender_factor": 0.5, "hip_width_factor": 1.15},
            "lean": {"gender_factor": 0.5, "shoulder_width_factor": 0.95},
        }
        preset_config = presets.get(preset.lower(), {})
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg, **preset_config)
    else:
        builder.set_parameters(height_m=height_m, mass_kg=mass_kg)

    builder.add_humanoid_segments()
    result = builder.build()

    if output_path and result.success:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.urdf_xml)
        result.output_path = path

    return result
>>>>>>> origin/main

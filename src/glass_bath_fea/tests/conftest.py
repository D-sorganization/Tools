"""Pytest configuration and fixtures for Glass Bath FEA tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

# Bootstrap for test discovery
_REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)

if TYPE_CHECKING:
    from glass_bath_fea.core.config import (
        GlassBathFEAConfig,
        GlassComposition,
        MeshConfig,
    )


@pytest.fixture
def default_fea_config() -> GlassBathFEAConfig:
    """Standard glass bath FEA configuration for testing."""
    from glass_bath_fea.core.config import GlassBathFEAConfig

    return GlassBathFEAConfig(
        bath_diameter=120.0,
        glass_depth=15.0,
        metal_layer_thickness=2.0,
        num_electrodes=3,
        electrode_spacing_degrees=120.0,
        electrode_diameter=6.0,
        electrode_insertion_depth=10.0,
        operating_temperature=1350.0,
    )


@pytest.fixture
def default_mesh_config() -> MeshConfig:
    """Standard mesh configuration for testing."""
    from glass_bath_fea.core.config import MeshConfig

    return MeshConfig(
        element_size_glass=0.02,  # Coarser for fast tests
        element_size_metal=0.01,
        element_size_electrodes=0.005,
        mesh_order=1,
    )


@pytest.fixture
def soda_lime_composition() -> GlassComposition:
    """Standard soda-lime glass composition."""
    from glass_bath_fea.core.config import GlassComposition

    return GlassComposition(
        sio2=74.0,
        na2o=13.0,
        cao=10.5,
        mgo=0.0,
        al2o3=1.5,
        fe2o3=0.1,
    )


@pytest.fixture
def high_iron_composition() -> GlassComposition:
    """High iron glass composition for testing composition effects."""
    from glass_bath_fea.core.config import GlassComposition

    return GlassComposition(
        sio2=72.0,
        na2o=13.0,
        cao=10.5,
        mgo=0.0,
        al2o3=1.5,
        fe2o3=3.0,  # Higher iron content
    )


@pytest.fixture
def mock_mesh_data() -> dict[str, npt.NDArray[np.float64]]:
    """Pre-generated mesh data for testing exporters."""
    # Simple tetrahedral mesh (4 vertices forming 1 tetrahedron)
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.289, 0.816],
        ]
    ).T  # MATLAB expects 3xN

    elements = np.array([[0, 1, 2, 3]]).T + 1  # 1-indexed for MATLAB

    material_ids = np.array([1])  # Glass region

    return {
        "nodes": nodes,
        "elements": elements,
        "material_ids": material_ids,
    }


@pytest.fixture
def temperature_range() -> npt.NDArray[np.float64]:
    """Standard temperature range for testing material properties."""
    return np.linspace(1000, 1400, 5)  # 1000°C to 1400°C

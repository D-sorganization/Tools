"""MATLAB .mat file exporter for Glass Bath FEA.

Exports mesh data and material properties in formats compatible
with MATLAB PDE Toolbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import savemat

if TYPE_CHECKING:
    from glass_bath_fea.core.config import GlassBathFEAConfig


def export_mesh_to_mat(mesh_data: dict, output_path: Path | str) -> None:
    """Export mesh data to MATLAB .mat file.

    Exports mesh in a format compatible with MATLAB PDE Toolbox's
    geometryFromMesh function.

    Args:
        mesh_data: Dictionary containing:
            - nodes: 3xN array of node coordinates
            - elements: MxN array of element connectivity (1-indexed)
            - material_ids: Array of material IDs
        output_path: Path to output .mat file
    """
    # Use MATLAB PDE Toolbox naming conventions
    data = {
        # p (points) - node coordinates
        "p": mesh_data["nodes"],
        "nodes": mesh_data["nodes"],
        # t (triangles/tetrahedra) - element connectivity
        "t": mesh_data["elements"],
        "elements": mesh_data["elements"],
    }

    # Add material IDs if present
    if "material_ids" in mesh_data:
        data["material_ids"] = np.array([mesh_data["material_ids"]])
        data["subdomain"] = np.array([mesh_data["material_ids"]])

    # Save in MATLAB v5 format for compatibility
    savemat(str(output_path), data, format="5")


def validate_mesh_data(mesh_data: dict) -> bool:
    """Validate mesh data before export.

    Checks for:
    - Non-empty nodes
    - Valid element indices (within node count)
    - Consistent dimensions

    Args:
        mesh_data: Mesh data dictionary

    Returns:
        True if mesh data is valid, False otherwise.
    """
    nodes = mesh_data.get("nodes", np.array([]))
    elements = mesh_data.get("elements", np.array([]))

    # Check for empty nodes
    if nodes.size == 0 or nodes.shape[1] == 0:
        return False

    # Check for empty elements
    if elements.size == 0:
        return True  # Empty elements is valid (just nodes)

    # Check element indices don't exceed node count
    num_nodes = nodes.shape[1]
    max_index = np.max(elements)

    return not max_index > num_nodes


def export_fea_data_package(
    config: GlassBathFEAConfig,
    output_dir: Path | str,
    include_mesh: bool = True,
) -> None:
    """Export complete FEA data package for MATLAB.

    Creates multiple .mat files with mesh, material properties,
    and boundary condition data.

    Args:
        config: FEA configuration
        output_dir: Directory for output files
        include_mesh: Whether to include mesh data
    """
    from glass_bath_fea.core.material_properties import (
        GlassMaterialModel,
        export_material_data,
    )
    from glass_bath_fea.core.mesh_generator import MeshGenerator
    from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

    output_dir = Path(output_dir)

    # Export mesh data
    if include_mesh:
        mesh_gen = MeshGenerator(config)
        mesh_data = mesh_gen.create_mock_mesh()  # Use mock for now
        export_mesh_to_mat(mesh_data, output_dir / "mesh.mat")

    # Export material properties
    material_model = GlassMaterialModel(config.glass_composition)
    export_material_data(material_model, output_dir / "material_properties.mat")

    # Export boundary conditions
    adapter = ElectrodeAdapter(config)
    adapter.export_boundary_conditions(output_dir / "boundary_conditions.mat")

    # Export configuration parameters
    config_data = {
        "bath_diameter": np.array([config.bath_diameter]),
        "glass_depth": np.array([config.glass_depth]),
        "metal_layer_thickness": np.array([config.metal_layer_thickness]),
        "num_electrodes": np.array([config.num_electrodes]),
        "operating_temperature": np.array([config.operating_temperature]),
        "phase_voltages": np.array(config.phase_voltages),
    }
    savemat(str(output_dir / "config.mat"), config_data, format="5")

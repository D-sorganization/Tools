"""MSH v2.2 format exporter for Glass Bath FEA.

Exports mesh data in Gmsh MSH v2.2 format, which can be imported
by MATLAB and other FEA tools.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_mesh_to_msh(
    mesh_data: dict,
    output_path: Path | str,
    physical_names: dict | None = None,
) -> None:
    """Export mesh data to MSH v2.2 format.

    Args:
        mesh_data: Dictionary containing:
            - nodes: 3xN array of node coordinates
            - elements: MxN array of element connectivity (1-indexed)
            - material_ids: Array of material IDs
        output_path: Path to output .msh file
        physical_names: Optional mapping of material IDs to names
    """
    nodes = mesh_data["nodes"]
    elements = mesh_data["elements"]
    material_ids = mesh_data.get("material_ids", np.ones(elements.shape[1]))

    # Default physical names
    if physical_names is None:
        physical_names = {
            1: "Glass",
            2: "Metal",
            3: "Electrode",
        }

    with open(output_path, "w") as f:
        # Write mesh format header
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")  # Version 2.2, ASCII, 8-byte floats
        f.write("$EndMeshFormat\n")

        # Write physical names if any
        unique_ids = np.unique(material_ids)
        if len(unique_ids) > 0:
            f.write("$PhysicalNames\n")
            f.write(f"{len(unique_ids)}\n")
            for mat_id in unique_ids:
                name = physical_names.get(int(mat_id), f"Region_{int(mat_id)}")
                f.write(f'3 {int(mat_id)} "{name}"\n')  # 3D regions
            f.write("$EndPhysicalNames\n")

        # Write nodes
        num_nodes = nodes.shape[1]
        f.write("$Nodes\n")
        f.write(f"{num_nodes}\n")
        for i in range(num_nodes):
            x, y, z = nodes[:, i]
            f.write(f"{i + 1} {x:.15g} {y:.15g} {z:.15g}\n")
        f.write("$EndNodes\n")

        # Write elements
        num_elements = elements.shape[1]
        f.write("$Elements\n")
        f.write(f"{num_elements}\n")

        # Determine element type based on nodes per element
        nodes_per_elem = elements.shape[0]
        if nodes_per_elem == 4:
            elem_type = 4  # 4-node tetrahedron
        elif nodes_per_elem == 8:
            elem_type = 5  # 8-node hexahedron
        elif nodes_per_elem == 3:
            elem_type = 2  # 3-node triangle
        else:
            elem_type = 4  # Default to tetrahedron

        for i in range(num_elements):
            mat_id = int(material_ids[i]) if i < len(material_ids) else 1
            node_indices = " ".join(str(int(n)) for n in elements[:, i])
            # Format: elem_id type num_tags physical_tag elementary_tag nodes...
            f.write(f"{i + 1} {elem_type} 2 {mat_id} {mat_id} {node_indices}\n")

        f.write("$EndElements\n")


def read_msh_file(input_path: Path | str) -> dict:
    """Read mesh data from MSH v2.2 format file.

    Args:
        input_path: Path to input .msh file

    Returns:
        Dictionary with mesh data.
    """
    nodes_list = []
    elements_list = []
    material_ids = []

    with open(input_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "$Nodes":
            i += 1
            num_nodes = int(lines[i].strip())
            i += 1
            for _ in range(num_nodes):
                parts = lines[i].strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                nodes_list.append([x, y, z])
                i += 1

        elif line == "$Elements":
            i += 1
            num_elements = int(lines[i].strip())
            i += 1
            for _ in range(num_elements):
                parts = lines[i].strip().split()
                _elem_type = int(parts[1])  # noqa: F841 - parsed but not used
                num_tags = int(parts[2])
                mat_id = int(parts[3])  # First tag is physical group

                # Node indices start after tags
                node_start = 3 + num_tags
                node_indices = [int(p) for p in parts[node_start:]]

                elements_list.append(node_indices)
                material_ids.append(mat_id)
                i += 1
        else:
            i += 1

    if nodes_list:
        nodes = np.array(nodes_list).T
    else:
        nodes = np.array([]).reshape(3, 0)

    if elements_list:
        elements = np.array(elements_list).T
    else:
        elements = np.array([]).reshape(4, 0)

    return {
        "nodes": nodes,
        "elements": elements,
        "material_ids": np.array(material_ids),
    }

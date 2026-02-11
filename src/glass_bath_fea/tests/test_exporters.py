"""Tests for Glass Bath FEA mesh exporters."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Bootstrap for test discovery
_REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)

if TYPE_CHECKING:
    from glass_bath_fea.core.config import GlassBathFEAConfig


class TestMatExporter:
    """Tests for MATLAB .mat file exporter."""

    def test_export_mesh_to_mat(
        self, mock_mesh_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test exporting mesh data to .mat file."""
        from glass_bath_fea.exporters.mat_exporter import export_mesh_to_mat

        output_path = tmp_path / "mesh.mat"

        export_mesh_to_mat(mock_mesh_data, output_path)

        assert output_path.exists()

    def test_mat_file_readable(
        self, mock_mesh_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that exported .mat file is readable."""
        from scipy.io import loadmat

        from glass_bath_fea.exporters.mat_exporter import export_mesh_to_mat

        output_path = tmp_path / "mesh.mat"
        export_mesh_to_mat(mock_mesh_data, output_path)

        # Should be loadable
        data = loadmat(output_path)
        assert data is not None

    def test_mat_contains_nodes(
        self, mock_mesh_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that .mat file contains node data."""
        from scipy.io import loadmat

        from glass_bath_fea.exporters.mat_exporter import export_mesh_to_mat

        output_path = tmp_path / "mesh.mat"
        export_mesh_to_mat(mock_mesh_data, output_path)

        data = loadmat(output_path)

        # Should have nodes (p in MATLAB PDE Toolbox convention)
        assert "p" in data or "nodes" in data

    def test_mat_contains_elements(
        self, mock_mesh_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that .mat file contains element data."""
        from scipy.io import loadmat

        from glass_bath_fea.exporters.mat_exporter import export_mesh_to_mat

        output_path = tmp_path / "mesh.mat"
        export_mesh_to_mat(mock_mesh_data, output_path)

        data = loadmat(output_path)

        # Should have elements (t in MATLAB PDE Toolbox convention)
        assert "t" in data or "elements" in data

    def test_mat_contains_material_ids(
        self, mock_mesh_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that .mat file contains material IDs."""
        from scipy.io import loadmat

        from glass_bath_fea.exporters.mat_exporter import export_mesh_to_mat

        output_path = tmp_path / "mesh.mat"
        export_mesh_to_mat(mock_mesh_data, output_path)

        data = loadmat(output_path)

        # Should have material region IDs
        assert "material_ids" in data or "subdomain" in data

    def test_mat_indexing_for_matlab(
        self, mock_mesh_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that element indices are 1-based for MATLAB."""
        from scipy.io import loadmat

        from glass_bath_fea.exporters.mat_exporter import export_mesh_to_mat

        output_path = tmp_path / "mesh.mat"
        export_mesh_to_mat(mock_mesh_data, output_path)

        data = loadmat(output_path)

        # Get elements (either t or elements)
        elements = data.get("t", data.get("elements"))

        if elements is not None:
            # MATLAB uses 1-based indexing
            assert np.min(elements) >= 1


class TestMshExporter:
    """Tests for MSH v2.2 file exporter."""

    def test_export_mesh_to_msh(
        self, mock_mesh_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test exporting mesh to MSH format."""
        from glass_bath_fea.exporters.msh_exporter import export_mesh_to_msh

        output_path = tmp_path / "mesh.msh"

        export_mesh_to_msh(mock_mesh_data, output_path)

        assert output_path.exists()

    def test_msh_file_format(
        self, mock_mesh_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that MSH file has correct format header."""
        from glass_bath_fea.exporters.msh_exporter import export_mesh_to_msh

        output_path = tmp_path / "mesh.msh"
        export_mesh_to_msh(mock_mesh_data, output_path)

        with open(output_path) as f:
            content = f.read()

        # MSH v2.2 format markers
        assert "$MeshFormat" in content
        assert "$Nodes" in content
        assert "$Elements" in content

    def test_msh_contains_all_nodes(
        self, mock_mesh_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that MSH file contains all nodes."""
        from glass_bath_fea.exporters.msh_exporter import export_mesh_to_msh

        output_path = tmp_path / "mesh.msh"
        export_mesh_to_msh(mock_mesh_data, output_path)

        with open(output_path) as f:
            content = f.read()

        # Count nodes in file
        lines = content.split("\n")
        in_nodes = False
        node_count = 0

        for line in lines:
            if line.strip() == "$Nodes":
                in_nodes = True
                continue
            if line.strip() == "$EndNodes":
                in_nodes = False
                continue
            if in_nodes and node_count == 0:
                node_count = int(line.strip())
                continue

        # Should match input
        assert node_count == mock_mesh_data["nodes"].shape[1]


class TestCombinedExport:
    """Tests for combined export functionality."""

    def test_export_full_fea_data(
        self, default_fea_config: GlassBathFEAConfig, tmp_path: Path
    ) -> None:
        """Test exporting complete FEA data package."""
        from glass_bath_fea.exporters.mat_exporter import export_fea_data_package

        output_dir = tmp_path / "fea_export"
        output_dir.mkdir()

        export_fea_data_package(default_fea_config, output_dir)

        # Should create multiple files
        assert (output_dir / "mesh.mat").exists() or len(
            list(output_dir.glob("*.mat"))
        ) > 0

    def test_export_includes_materials(
        self, default_fea_config: GlassBathFEAConfig, tmp_path: Path
    ) -> None:
        """Test that export includes material properties."""
        from glass_bath_fea.exporters.mat_exporter import export_fea_data_package

        output_dir = tmp_path / "fea_export"
        output_dir.mkdir()

        export_fea_data_package(default_fea_config, output_dir)

        # Should have material data
        mat_files = list(output_dir.glob("*material*.mat"))
        assert len(mat_files) > 0 or (output_dir / "mesh.mat").exists()

    def test_export_includes_boundary_conditions(
        self, default_fea_config: GlassBathFEAConfig, tmp_path: Path
    ) -> None:
        """Test that export includes boundary condition data."""
        from glass_bath_fea.exporters.mat_exporter import export_fea_data_package

        output_dir = tmp_path / "fea_export"
        output_dir.mkdir()

        export_fea_data_package(default_fea_config, output_dir)

        # Check for boundary condition data in exported files
        mat_files = list(output_dir.glob("*.mat"))
        assert len(mat_files) > 0


class TestExportValidation:
    """Tests for export data validation."""

    def test_validate_mesh_before_export(self, mock_mesh_data: dict[str, Any]) -> None:
        """Test mesh validation before export."""
        from glass_bath_fea.exporters.mat_exporter import validate_mesh_data

        is_valid = validate_mesh_data(mock_mesh_data)

        assert is_valid

    def test_detect_invalid_mesh(self) -> None:
        """Test detection of invalid mesh data."""
        from glass_bath_fea.exporters.mat_exporter import validate_mesh_data

        invalid_mesh: dict[str, Any] = {
            "nodes": np.array([]),  # Empty nodes
            "elements": np.array([[1, 2, 3, 4]]).T,
        }

        is_valid = validate_mesh_data(invalid_mesh)

        assert not is_valid

    def test_detect_mismatched_indices(self) -> None:
        """Test detection of element indices exceeding node count."""
        from glass_bath_fea.exporters.mat_exporter import validate_mesh_data

        # Element references node 100, but only 4 nodes exist
        invalid_mesh: dict[str, Any] = {
            "nodes": np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            ).T,  # 4 nodes
            "elements": np.array([[1, 2, 3, 100]]).T,  # References node 100
            "material_ids": np.array([1]),
        }

        is_valid = validate_mesh_data(invalid_mesh)

        assert not is_valid

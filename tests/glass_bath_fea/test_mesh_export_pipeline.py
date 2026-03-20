"""Tests for the Glass Bath FEA mesh export pipeline.

Verifies the end-to-end export workflow: config resolution, mesh generation,
validation, and multi-format export (.msh, .mat, .json).

See issue #579.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from glass_bath_fea.core.config import GlassBathFEAConfig
from glass_bath_fea.exporters.mat_exporter import validate_mesh_data
from glass_bath_fea.exporters.mesh_export_pipeline import (
    MeshExportPipeline,
)
from glass_bath_fea.exporters.msh_exporter import read_msh_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_electrode_config(**overrides: float) -> MagicMock:
    """Create a stub that looks like ``ElectrodeConfig``."""
    defaults = {
        "bath_diameter": 120.0,
        "glass_depth": 15.0,
        "metal_depth": 2.0,
        "tip_diameter": 6.0,
        "electrode_spacing_degrees": 120.0,
        "bath_temperature": 1350.0,
        "metal_conductivity": 10000.0,
        "electrode_depths": np.array([10.0, 10.0, 10.0]),
        "phase_voltages": np.array([100.0, 100.0, 100.0]),
    }
    defaults.update(overrides)
    cfg = MagicMock(**defaults)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Tests: direct FEA config (skips geometry sync)
# ---------------------------------------------------------------------------


class TestMeshExportPipelineDirectConfig:
    """Pipeline with a direct GlassBathFEAConfig (no electrode sync)."""

    def test_export_all_formats(self) -> None:
        """Export to all three formats and verify files exist."""
        config = GlassBathFEAConfig()
        pipeline = MeshExportPipeline(fea_config=config)

        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(output_dir=tmp, formats=["msh", "mat", "json"])

        assert result.success, f"Pipeline failed: {result.errors}"
        assert len(result.exported_files) == 3
        assert result.mesh_stats.get("num_nodes", 0) > 0
        assert result.mesh_stats.get("num_elements", 0) > 0

    def test_export_msh_only(self) -> None:
        """Export only MSH and verify the file is valid."""
        config = GlassBathFEAConfig()
        pipeline = MeshExportPipeline(fea_config=config)

        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(output_dir=tmp, formats=["msh"])
            assert result.success

            msh_path = Path(tmp) / "mesh.msh"
            assert msh_path.exists()

            # Read it back and verify structure
            mesh_back = read_msh_file(msh_path)
            assert mesh_back["nodes"].shape[0] == 3
            assert mesh_back["nodes"].shape[1] > 0

    def test_export_mat_only(self) -> None:
        """Export only MAT."""
        config = GlassBathFEAConfig()
        pipeline = MeshExportPipeline(fea_config=config)

        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(output_dir=tmp, formats=["mat"])
            assert result.success

            mat_path = Path(tmp) / "mesh.mat"
            assert mat_path.exists()
            assert mat_path.stat().st_size > 0

    def test_export_json_metadata(self) -> None:
        """Export JSON metadata and verify contents."""
        config = GlassBathFEAConfig()
        pipeline = MeshExportPipeline(fea_config=config)

        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(output_dir=tmp, formats=["json"])
            assert result.success

            json_path = Path(tmp) / "mesh_metadata.json"
            assert json_path.exists()

            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            assert "mesh_statistics" in metadata
            assert "config" in metadata
            assert "dimensions_meters" in metadata
            assert metadata["config"]["bath_diameter_in"] == config.bath_diameter

    def test_mesh_statistics_populated(self) -> None:
        """Verify mesh statistics are returned."""
        config = GlassBathFEAConfig()
        pipeline = MeshExportPipeline(fea_config=config)

        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(output_dir=tmp, formats=["json"])

        assert result.mesh_stats["num_nodes"] > 0
        assert result.mesh_stats["num_elements"] > 0

    def test_custom_physical_names(self) -> None:
        """Custom physical names propagate to MSH export."""
        config = GlassBathFEAConfig()
        pipeline = MeshExportPipeline(fea_config=config)
        names = {1: "Molten_Glass", 2: "Metal_Bottom", 3: "Electrode_Zone"}

        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(output_dir=tmp, formats=["msh"], physical_names=names)
            assert result.success

            # Read MSH and verify physical names appear
            msh_text = (Path(tmp) / "mesh.msh").read_text(encoding="utf-8")
            assert "$PhysicalNames" in msh_text


# ---------------------------------------------------------------------------
# Tests: electrode sync path
# ---------------------------------------------------------------------------


class TestMeshExportPipelineWithSync:
    """Pipeline using GeometrySynchronizer from electrode config."""

    def test_sync_and_export(self) -> None:
        """Full pipeline with electrode config sync."""
        ec = _make_stub_electrode_config()
        pipeline = MeshExportPipeline(electrode_config=ec)

        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(output_dir=tmp, formats=["msh", "json"])

        assert result.success, f"Pipeline failed: {result.errors}"
        assert len(result.exported_files) == 2

    def test_invalid_electrode_config_returns_error(self) -> None:
        """Pipeline returns error for geometrically invalid electrode config."""
        ec = _make_stub_electrode_config(
            bath_diameter=10.0,
            electrode_depths=np.array([100.0, 100.0, 100.0]),
        )
        pipeline = MeshExportPipeline(electrode_config=ec)

        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run(output_dir=tmp)

        assert not result.success
        assert len(result.errors) > 0
        assert "Configuration error" in result.errors[0]


# ---------------------------------------------------------------------------
# Tests: mesh validation edge cases
# ---------------------------------------------------------------------------


class TestMeshValidation:
    """Edge-case validation within the pipeline."""

    def test_validate_mesh_data_valid(self) -> None:
        """A proper mesh passes validation."""
        mesh_data = {
            "nodes": np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            "elements": np.array([[1], [2], [3], [4]]),
            "material_ids": np.array([1]),
        }
        assert validate_mesh_data(mesh_data) is True

    def test_validate_mesh_data_empty_nodes(self) -> None:
        """Empty nodes fails validation."""
        mesh_data = {
            "nodes": np.array([]).reshape(3, 0),
            "elements": np.array([]).reshape(4, 0),
        }
        assert validate_mesh_data(mesh_data) is False

    def test_validate_mesh_data_bad_index(self) -> None:
        """Element referencing non-existent node fails."""
        mesh_data = {
            "nodes": np.array([[0, 1], [0, 0], [0, 0]]),
            "elements": np.array([[1], [2], [3], [99]]),  # node 99 doesn't exist
        }
        assert validate_mesh_data(mesh_data) is False


# ---------------------------------------------------------------------------
# Tests: round-trip MSH
# ---------------------------------------------------------------------------


class TestMshRoundTrip:
    """Verify MSH export -> read round-trip preserves mesh data."""

    def test_round_trip(self) -> None:
        """Export a simple mesh to MSH and read it back."""
        nodes = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        elements = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        material_ids = np.array([1, 2])

        mesh_data = {
            "nodes": nodes,
            "elements": elements,
            "material_ids": material_ids,
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.msh"
            from glass_bath_fea.exporters.msh_exporter import export_mesh_to_msh

            export_mesh_to_msh(mesh_data, out)

            mesh_back = read_msh_file(out)

        assert mesh_back["nodes"].shape == nodes.shape
        np.testing.assert_allclose(mesh_back["nodes"], nodes, atol=1e-10)
        assert mesh_back["elements"].shape == elements.shape

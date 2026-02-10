"""Mesh export pipeline for Glass Bath FEA.

Orchestrates the full workflow from electrode configuration through mesh
generation to multi-format file export (.msh, .mat).  Builds on the
geometry sync from Phase 3 (issue #575).

See issue #579.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glass_bath_fea.core.config import GlassBathFEAConfig
from glass_bath_fea.core.mesh_generator import MeshGenerator
from glass_bath_fea.exporters.mat_exporter import export_mesh_to_mat, validate_mesh_data
from glass_bath_fea.exporters.msh_exporter import export_mesh_to_msh
from glass_bath_fea.interfaces.geometry_sync import GeometrySynchronizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Export result
# ---------------------------------------------------------------------------


@dataclass
class MeshExportResult:
    """Result of the mesh export pipeline."""

    success: bool
    mesh_stats: dict[str, Any]
    exported_files: list[str]
    errors: list[str]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class MeshExportPipeline:
    """End-to-end pipeline: electrode config -> mesh -> export files.

    Usage::

        pipeline = MeshExportPipeline()
        result = pipeline.run(output_dir="./export", formats=["msh", "mat"])

    Attributes:
        config: The resolved ``GlassBathFEAConfig`` (available after run).
        mesh_data: The generated mesh dict (available after run).
    """

    def __init__(
        self,
        electrode_config: Any | None = None,
        fea_config: GlassBathFEAConfig | None = None,
    ) -> None:
        """Initialise the pipeline.

        Provide *either* an ``electrode_config`` (from the Electrode Advisor)
        which will be synced via ``GeometrySynchronizer``, *or* a direct
        ``GlassBathFEAConfig``.  If both are *None*, the synchroniser will
        use its own defaults.

        Args:
            electrode_config: Optional electrode advisor config.
            fea_config: Optional direct FEA config (skips geometry sync).
        """
        self._electrode_config = electrode_config
        self._fea_config = fea_config
        self.config: GlassBathFEAConfig | None = None
        self.mesh_data: dict | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        output_dir: str | Path,
        formats: list[str] | None = None,
        coarse: bool = True,
        physical_names: dict[int, str] | None = None,
    ) -> MeshExportResult:
        """Execute the full pipeline.

        Args:
            output_dir: Directory where exported files will be written.
            formats: List of export formats.  Supported: ``"msh"``,
                ``"mat"``, ``"json"``.  Defaults to all three.
            coarse: Whether to use a coarse mesh (faster, for development).
            physical_names: Optional mapping of material IDs to region
                names for the MSH exporter.

        Returns:
            A ``MeshExportResult`` summarising what was exported.
        """
        if formats is None:
            formats = ["msh", "mat", "json"]

        errors: list[str] = []
        warnings: list[str] = []
        exported_files: list[str] = []
        output_path = Path(output_dir)

        # Step 1: Resolve configuration
        try:
            self.config = self._resolve_config(warnings)
        except Exception as exc:
            return MeshExportResult(
                success=False,
                mesh_stats={},
                exported_files=[],
                errors=[f"Configuration error: {exc}"],
                warnings=warnings,
            )

        # Step 2: Generate mesh
        try:
            self.mesh_data = self._generate_mesh(coarse)
        except ImportError:
            logger.info("pygmsh not available, using mock mesh")
            warnings.append("pygmsh not available -- using mock mesh for export")
            mesh_gen = MeshGenerator(self.config)
            self.mesh_data = mesh_gen.create_mock_mesh()
        except Exception as exc:
            return MeshExportResult(
                success=False,
                mesh_stats={},
                exported_files=[],
                errors=[f"Mesh generation error: {exc}"],
                warnings=warnings,
            )

        # Step 3: Validate mesh
        if not validate_mesh_data(self.mesh_data):
            errors.append("Mesh validation failed -- nodes or elements invalid")
            return MeshExportResult(
                success=False,
                mesh_stats=self._mesh_statistics(),
                exported_files=[],
                errors=errors,
                warnings=warnings,
            )

        # Step 4: Quality check
        mesh_gen = MeshGenerator(self.config)
        quality = mesh_gen.check_mesh_quality(self.mesh_data)
        if quality.get("mean_quality", 0) < 0.1:
            warnings.append(
                f"Low average mesh quality ({quality.get('mean_quality', 0):.3f})"
            )

        # Step 5: Export to requested formats
        output_path.mkdir(parents=True, exist_ok=True)

        if "msh" in formats:
            try:
                msh_path = output_path / "mesh.msh"
                export_mesh_to_msh(
                    self.mesh_data, msh_path, physical_names=physical_names
                )
                exported_files.append(str(msh_path))
                logger.info("Exported MSH to %s", msh_path)
            except Exception as exc:
                errors.append(f"MSH export error: {exc}")

        if "mat" in formats:
            try:
                mat_path = output_path / "mesh.mat"
                export_mesh_to_mat(self.mesh_data, mat_path)
                exported_files.append(str(mat_path))
                logger.info("Exported MAT to %s", mat_path)
            except Exception as exc:
                errors.append(f"MAT export error: {exc}")

        if "json" in formats:
            try:
                json_path = output_path / "mesh_metadata.json"
                self._export_json_metadata(json_path, quality)
                exported_files.append(str(json_path))
                logger.info("Exported JSON metadata to %s", json_path)
            except Exception as exc:
                errors.append(f"JSON metadata export error: {exc}")

        return MeshExportResult(
            success=len(errors) == 0,
            mesh_stats=self._mesh_statistics(),
            exported_files=exported_files,
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_config(self, warnings: list[str]) -> GlassBathFEAConfig:
        """Resolve FEA configuration from electrode config or direct config."""
        if self._fea_config is not None:
            return self._fea_config

        sync = GeometrySynchronizer(self._electrode_config)
        validation = sync.validate()
        for w in validation.warnings:
            warnings.append(w)
        return sync.sync()

    def _generate_mesh(self, coarse: bool) -> dict:
        """Generate mesh using MeshGenerator."""
        assert self.config is not None
        mesh_gen = MeshGenerator(self.config)
        return mesh_gen.generate_mesh(coarse=coarse)

    def _mesh_statistics(self) -> dict[str, Any]:
        """Calculate statistics for the current mesh."""
        if self.mesh_data is None or self.config is None:
            return {}
        mesh_gen = MeshGenerator(self.config)
        return mesh_gen.get_mesh_statistics(self.mesh_data)

    def _export_json_metadata(self, path: Path, quality: dict) -> None:
        """Export mesh metadata as JSON."""
        assert self.config is not None and self.mesh_data is not None

        stats = self._mesh_statistics()
        metadata = {
            "mesh_statistics": stats,
            "mesh_quality": quality,
            "config": {
                "bath_diameter_in": self.config.bath_diameter,
                "glass_depth_in": self.config.glass_depth,
                "metal_layer_thickness_in": self.config.metal_layer_thickness,
                "num_electrodes": self.config.num_electrodes,
                "electrode_diameter_in": self.config.electrode_diameter,
                "electrode_insertion_depth_in": self.config.electrode_insertion_depth,
                "operating_temperature_c": self.config.operating_temperature,
            },
            "dimensions_meters": self.config.get_dimensions_meters(),
            "export_formats": ["msh", "mat", "json"],
        }

        path.write_text(
            json.dumps(metadata, indent=2, default=_json_default),
            encoding="utf-8",
        )


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

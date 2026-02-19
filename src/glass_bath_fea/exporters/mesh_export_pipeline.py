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

    def _prepare_mesh(
        self,
        coarse: bool,
        warnings: list[str],
    ) -> tuple[dict | None, list[str]]:
        """Resolve config and generate mesh, collecting errors.

        Args:
            coarse: Whether to use a coarse mesh.
            warnings: Warning list to append to.

        Returns:
            Tuple of (mesh_data_or_None, error_list).
        """
        errors: list[str] = []
        try:
            self.config = self._resolve_config(warnings)
        except (KeyError, ValueError, TypeError) as exc:
            return None, [f"Configuration error: {exc}"]

        try:
            self.mesh_data = self._generate_mesh(coarse)
        except ImportError:
            logger.info("pygmsh not available, using mock mesh")
            warnings.append("pygmsh not available -- using mock mesh for export")
            mesh_gen = MeshGenerator(self.config)
            self.mesh_data = mesh_gen.create_mock_mesh()
        except (KeyError, ValueError, TypeError) as exc:
            return None, [f"Mesh generation error: {exc}"]

        if not validate_mesh_data(self.mesh_data):
            errors.append("Mesh validation failed -- nodes or elements invalid")
            return None, errors

        return self.mesh_data, []

    def _export_formats(
        self,
        output_path: Path,
        formats: list[str],
        physical_names: dict[int, str] | None,
        quality: dict,
    ) -> tuple[list[str], list[str]]:
        """Export mesh to each requested format.

        Args:
            output_path: Output directory.
            formats: List of format strings.
            physical_names: MSH physical names mapping.
            quality: Mesh quality dict.

        Returns:
            Tuple of (exported_file_paths, errors).
        """
        exported_files: list[str] = []
        errors: list[str] = []

        format_handlers: dict[str, tuple[str, Any]] = {}
        if "msh" in formats:
            format_handlers["msh"] = ("mesh.msh", lambda p: export_mesh_to_msh(
                self.mesh_data, p, physical_names=physical_names
            ))
        if "mat" in formats:
            format_handlers["mat"] = ("mesh.mat", lambda p: export_mesh_to_mat(
                self.mesh_data, p
            ))
        if "json" in formats:
            format_handlers["json"] = ("mesh_metadata.json", lambda p: self._export_json_metadata(
                p, quality
            ))

        for fmt, (filename, handler) in format_handlers.items():
            try:
                path = output_path / filename
                handler(path)
                exported_files.append(str(path))
                logger.info("Exported %s to %s", fmt.upper(), path)
            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
                errors.append(f"{fmt.upper()} export error: {exc}")

        return exported_files, errors

    def run(
        self,
        output_dir: str | Path,
        formats: list[str] | None = None,
        coarse: bool = True,
        physical_names: dict[int, str] | None = None,
    ) -> MeshExportResult:
        """Execute the full mesh export pipeline.

        Orchestrates config resolution, mesh generation, validation,
        quality checking, and multi-format export.

        Args:
            output_dir: Directory where exported files will be written.
            formats: List of export formats (``"msh"``, ``"mat"``, ``"json"``).
            coarse: Whether to use a coarse mesh.
            physical_names: Optional material ID to name mapping for MSH.

        Returns:
            A ``MeshExportResult`` summarising what was exported.
        """
        if formats is None:
            formats = ["msh", "mat", "json"]

        warnings: list[str] = []
        output_path = Path(output_dir)

        mesh_data, prep_errors = self._prepare_mesh(coarse, warnings)
        if prep_errors:
            return MeshExportResult(
                success=False,
                mesh_stats=self._mesh_statistics() if self.mesh_data else {},
                exported_files=[],
                errors=prep_errors,
                warnings=warnings,
            )

        mesh_gen = MeshGenerator(self.config)
        quality = mesh_gen.check_mesh_quality(self.mesh_data)
        if quality.get("mean_quality", 0) < 0.1:
            warnings.append(
                f"Low average mesh quality ({quality.get('mean_quality', 0):.3f})"
            )

        output_path.mkdir(parents=True, exist_ok=True)
        exported_files, errors = self._export_formats(
            output_path, formats, physical_names, quality,
        )

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
        return dict(mesh_gen.generate_mesh(coarse=coarse))

    def _mesh_statistics(self) -> dict[str, Any]:
        """Calculate statistics for the current mesh."""
        if self.mesh_data is None or self.config is None:
            return {}
        mesh_gen = MeshGenerator(self.config)
        return dict(mesh_gen.get_mesh_statistics(self.mesh_data))

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

"""Multi-format export for vessel drafter layouts."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vessel_drafter.analysis.vessel_drafter_metrics import build_material_metrics_report
from vessel_drafter.contracts import require
from vessel_drafter.models.vessel_drafter import (
    DEFAULT_VESSEL_DRAFTER_LAYOUT,
    VesselDrafterLayout,
)
from vessel_drafter.projects.vessel_drafter_layout import (
    build_vessel_drafter_components,
)

_SUPPORTED_FORMATS = ("step", "stl", "brep", "gltf")


def export_vessel(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
    output_dir: os.PathLike | str = ".",
    stem: str = "vessel_drafter",
    formats: Sequence[str] = ("step",),
) -> dict[str, Path]:
    """Export a vessel layout to one or more file formats.

    Returns a mapping of format name to output path.
    """
    require(layout is not None, "layout must be provided")
    from build123d import Compound  # lazy import — optional dep

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    components = build_vessel_drafter_components(layout)
    compound = Compound(
        label="vessel_drafter",
        children=[component.shape for component in components],
    )

    results: dict[str, Path] = {}
    for fmt in formats:
        fmt = fmt.lower()
        if fmt not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format {fmt!r}. Choose from {_SUPPORTED_FORMATS}"
            )
        out_file = output_path / f"{stem}.{fmt}"
        _export_compound(compound, out_file, fmt)
        results[fmt] = out_file

    _write_manifest(layout, output_path, stem)
    return results


def export_vessel_step(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
    output_dir: os.PathLike | str = ".",
    stem: str = "vessel_drafter",
) -> Path:
    return export_vessel(layout, output_dir, stem, formats=("step",))["step"]


def export_vessel_stl(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
    output_dir: os.PathLike | str = ".",
    stem: str = "vessel_drafter",
) -> Path:
    return export_vessel(layout, output_dir, stem, formats=("stl",))["stl"]


def export_vessel_brep(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
    output_dir: os.PathLike | str = ".",
    stem: str = "vessel_drafter",
) -> Path:
    return export_vessel(layout, output_dir, stem, formats=("brep",))["brep"]


def export_vessel_gltf(
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
    output_dir: os.PathLike | str = ".",
    stem: str = "vessel_drafter",
) -> Path:
    return export_vessel(layout, output_dir, stem, formats=("gltf",))["gltf"]


def _export_compound(compound: Any, out_file: Path, fmt: str) -> None:
    from build123d import export_brep, export_gltf, export_step, export_stl

    if fmt == "step":
        export_step(compound, str(out_file))
    elif fmt == "stl":
        export_stl(compound, str(out_file))
    elif fmt == "brep":
        export_brep(compound, str(out_file))
    elif fmt == "gltf":
        export_gltf(compound, str(out_file))


def _write_manifest(
    layout: VesselDrafterLayout,
    output_path: Path,
    stem: str,
) -> Path:
    require(layout is not None, "layout must be provided")
    report = build_material_metrics_report(layout)
    manifest = {
        "layout": layout.__dict__ if hasattr(layout, "__dict__") else str(layout),
        "metrics": {
            entry.label: {
                "volume_ft3": entry.volume_ft3,
                "surface_area_ft2": entry.surface_area_ft2,
                "mass_lb": entry.mass_lb,
            }
            for entry in report.component_metrics
        },
    }
    manifest_path = output_path / f"{stem}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest_path

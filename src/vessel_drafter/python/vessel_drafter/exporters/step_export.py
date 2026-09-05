"""STEP export utilities."""

from __future__ import annotations

import json
from pathlib import Path

from build123d import export_step

from vessel_drafter.analysis.vessel_drafter_metrics import (
    build_material_metrics_report,
)
from vessel_drafter.contracts import require
from vessel_drafter.models.cylindrical_bath import (
    DEFAULT_CYLINDRICAL_BATH_LAYOUT,
    CylindricalBathLayout,
)
from vessel_drafter.models.electrode_advisor import (
    DEFAULT_ELECTRODE_ADVISOR_LAYOUT,
    ElectrodeAdvisorLayout,
)
from vessel_drafter.models.vessel_drafter import (
    DEFAULT_VESSEL_DRAFTER_LAYOUT,
    VesselDrafterLayout,
)
from vessel_drafter.projects.cylindrical_bath_layout import (
    build_cylindrical_bath_layout_shape,
)
from vessel_drafter.projects.electrode_advisor_default_layout import (
    build_default_layout_shape,
)
from vessel_drafter.projects.vessel_drafter_layout import (
    build_vessel_drafter_shape,
)


def export_default_layout_step(
    output_path: Path,
    manifest_path: Path | None = None,
    layout: ElectrodeAdvisorLayout = DEFAULT_ELECTRODE_ADVISOR_LAYOUT,
) -> Path:
    require(output_path is not None, "output_path must be provided")
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = build_default_layout_shape(layout)
    export_step(shape, output_path)

    if manifest_path is not None:
        manifest_dir = manifest_path.parent
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(layout.to_manifest(), indent=2) + "\n",
            encoding="utf-8",
        )

    return output_path


def export_cylindrical_bath_layout_step(
    output_path: Path,
    manifest_path: Path | None = None,
    layout: CylindricalBathLayout = DEFAULT_CYLINDRICAL_BATH_LAYOUT,
) -> Path:
    require(output_path is not None, "output_path must be provided")
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = build_cylindrical_bath_layout_shape(layout)
    export_step(shape, output_path)

    if manifest_path is not None:
        manifest_dir = manifest_path.parent
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(layout.to_manifest(), indent=2) + "\n",
            encoding="utf-8",
        )

    return output_path


def export_vessel_drafter_step(
    output_path: Path,
    manifest_path: Path | None = None,
    layout: VesselDrafterLayout = DEFAULT_VESSEL_DRAFTER_LAYOUT,
) -> Path:
    require(output_path is not None, "output_path must be provided")
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = build_vessel_drafter_shape(layout)
    export_step(shape, output_path)

    if manifest_path is not None:
        manifest_dir = manifest_path.parent
        manifest_dir.mkdir(parents=True, exist_ok=True)
        metrics = build_material_metrics_report(layout)
        manifest = layout.to_manifest()
        manifest["computed_metrics"] = {
            "components": {
                item.label: {
                    "display_name": item.display_name,
                    "category": item.category,
                    "volume_in3": item.volume_in3,
                    "volume_ft3": item.volume_ft3,
                    "surface_area_ft2": item.surface_area_ft2,
                    "density_lb_per_ft3": item.density_lb_per_ft3,
                    "mass_lb": item.mass_lb,
                    "thermal_conductivity_w_per_mk": (
                        item.thermal_conductivity_w_per_mk
                    ),
                    "thermal_expansion_um_per_m_c": (item.thermal_expansion_um_per_m_c),
                }
                for item in metrics.component_metrics
            },
            "refractory_total_volume_in3": metrics.refractory_total_volume_in3,
            "refractory_total_volume_ft3": metrics.refractory_total_volume_ft3,
            "refractory_total_surface_area_ft2": (
                metrics.refractory_total_surface_area_ft2
            ),
            "refractory_total_mass_lb": metrics.refractory_total_mass_lb,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    return output_path


def export_vessel_drafter_default_step(
    output_path: Path,
    manifest_path: Path | None = None,
) -> Path:
    return export_vessel_drafter_step(
        output_path=output_path,
        manifest_path=manifest_path,
        layout=DEFAULT_VESSEL_DRAFTER_LAYOUT,
    )

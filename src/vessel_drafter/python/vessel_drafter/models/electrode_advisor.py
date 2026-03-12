"""Default electrode advisor drafting inputs mirrored from the current UI."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Any

MM_PER_M = 1000.0


@dataclass(frozen=True)
class BathDefaults:
    shape: str = "rectangular"
    width_m: float = 3.0
    depth_m: float = 2.0
    height_m: float = 2.5
    glass_level_m: float = 1.5


@dataclass(frozen=True)
class ElectrodeDefaults:
    electrode_type: str = "graphite_standard"
    count: int = 3
    top_offset_m: float = 0.1
    current_length_mm: float = 1500.0
    worn_length_mm: float = 150.0
    diameter_mm: float = 150.0
    operating_current_a: float = 2500.0
    plasma_temperature_c: float = 1500.0


@dataclass(frozen=True)
class DraftingEnvelope:
    bath_shell_thickness_mm: float = 25.0
    glass_clearance_mm: float = 10.0
    electrode_holder_height_mm: float = 100.0
    electrode_holder_radius_factor: float = 2.0
    tip_band_height_mm: float = 20.0


@dataclass(frozen=True)
class ElectrodePlacement:
    index: int
    angle_radians: float
    viewer_x_m: float
    viewer_y_m: float
    viewer_z_m: float
    cad_x_mm: float
    cad_y_mm: float
    cad_z_mm: float
    diameter_mm: float
    current_a: float
    current_length_mm: float
    worn_length_mm: float
    effective_length_mm: float


@dataclass(frozen=True)
class ElectrodeAdvisorLayout:
    bath: BathDefaults
    electrodes: ElectrodeDefaults
    drafting: DraftingEnvelope
    placements: tuple[ElectrodePlacement, ...]

    @property
    def bath_width_mm(self) -> float:
        return self.bath.width_m * MM_PER_M

    @property
    def bath_depth_mm(self) -> float:
        return self.bath.depth_m * MM_PER_M

    @property
    def bath_height_mm(self) -> float:
        return self.bath.height_m * MM_PER_M

    @property
    def glass_level_mm(self) -> float:
        return self.bath.glass_level_m * MM_PER_M

    def to_manifest(self) -> dict[str, Any]:
        return {
            "project": "electrode_advisor_default_layout",
            "source_viewer": {
                "component": (
                    "Tools/src/electrode_advisor/web/src/components/"
                    "GlassBath3DViewer.tsx"
                ),
                "calculator": (
                    "Tools/src/electrode_advisor/web/src/components/"
                    "ElectrodeAdvisorCalculator.tsx"
                ),
            },
            "bath": {
                "shape": self.bath.shape,
                "width_m": self.bath.width_m,
                "depth_m": self.bath.depth_m,
                "height_m": self.bath.height_m,
                "glass_level_m": self.bath.glass_level_m,
            },
            "electrodes": {
                "type": self.electrodes.electrode_type,
                "count": self.electrodes.count,
                "top_offset_m": self.electrodes.top_offset_m,
                "current_length_mm": self.electrodes.current_length_mm,
                "worn_length_mm": self.electrodes.worn_length_mm,
                "diameter_mm": self.electrodes.diameter_mm,
                "operating_current_a": self.electrodes.operating_current_a,
                "plasma_temperature_c": self.electrodes.plasma_temperature_c,
            },
            "drafting_assumptions": {
                "bath_shell_thickness_mm": self.drafting.bath_shell_thickness_mm,
                "glass_clearance_mm": self.drafting.glass_clearance_mm,
                "electrode_holder_height_mm": self.drafting.electrode_holder_height_mm,
                "electrode_holder_radius_factor": (
                    self.drafting.electrode_holder_radius_factor
                ),
                "tip_band_height_mm": self.drafting.tip_band_height_mm,
            },
            "placements": [
                {
                    "index": item.index,
                    "angle_radians": item.angle_radians,
                    "viewer_position_m": [
                        item.viewer_x_m,
                        item.viewer_y_m,
                        item.viewer_z_m,
                    ],
                    "cad_position_mm": [
                        item.cad_x_mm,
                        item.cad_y_mm,
                        item.cad_z_mm,
                    ],
                    "effective_length_mm": item.effective_length_mm,
                    "current_a": item.current_a,
                }
                for item in self.placements
            ],
        }


def _build_default_placements(
    bath: BathDefaults,
    electrodes: ElectrodeDefaults,
) -> tuple[ElectrodePlacement, ...]:
    spacing_m = bath.width_m * 0.6
    radius_m = spacing_m * 0.4
    cad_z_mm = (bath.height_m + electrodes.top_offset_m) * MM_PER_M
    current_per_electrode = electrodes.operating_current_a / electrodes.count
    effective_length_mm = electrodes.current_length_mm - electrodes.worn_length_mm

    placements: list[ElectrodePlacement] = []
    for index in range(electrodes.count):
        angle = (index / electrodes.count) * 2.0 * pi
        viewer_x_m = 0.0 if electrodes.count == 1 else cos(angle) * radius_m
        viewer_z_m = 0.0 if electrodes.count == 1 else sin(angle) * radius_m
        placements.append(
            ElectrodePlacement(
                index=index + 1,
                angle_radians=angle,
                viewer_x_m=viewer_x_m,
                viewer_y_m=electrodes.top_offset_m,
                viewer_z_m=viewer_z_m,
                cad_x_mm=viewer_x_m * MM_PER_M,
                cad_y_mm=viewer_z_m * MM_PER_M,
                cad_z_mm=cad_z_mm,
                diameter_mm=electrodes.diameter_mm,
                current_a=current_per_electrode,
                current_length_mm=electrodes.current_length_mm,
                worn_length_mm=electrodes.worn_length_mm,
                effective_length_mm=effective_length_mm,
            )
        )
    return tuple(placements)


def build_default_electrode_advisor_layout() -> ElectrodeAdvisorLayout:
    bath = BathDefaults()
    electrodes = ElectrodeDefaults()
    drafting = DraftingEnvelope()
    return ElectrodeAdvisorLayout(
        bath=bath,
        electrodes=electrodes,
        drafting=drafting,
        placements=_build_default_placements(bath, electrodes),
    )


DEFAULT_ELECTRODE_ADVISOR_LAYOUT = build_default_electrode_advisor_layout()
ElectrodeAdvisorDraftingDefaults = DraftingEnvelope

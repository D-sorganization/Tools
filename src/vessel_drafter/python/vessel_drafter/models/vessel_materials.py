"""Shared vessel drafter material defaults and validations."""

from __future__ import annotations

from dataclasses import dataclass

from vessel_drafter.contracts import require_nonnegative, require_positive


@dataclass(frozen=True)
class MaterialProperties:
    name: str
    display_name: str
    color_hex: str
    category: str
    density_lb_per_ft3: float
    thermal_conductivity_w_per_mk: float
    thermal_expansion_um_per_m_c: float
    preview_alpha: float

    def __post_init__(self) -> None:
        require_positive("density_lb_per_ft3", self.density_lb_per_ft3)
        require_positive(
            "thermal_conductivity_w_per_mk",
            self.thermal_conductivity_w_per_mk,
        )
        require_positive(
            "thermal_expansion_um_per_m_c",
            self.thermal_expansion_um_per_m_c,
        )
        require_nonnegative("preview_alpha", self.preview_alpha)
        if self.preview_alpha > 1.0:
            raise ValueError(
                f"preview_alpha must be <= 1.0, got {self.preview_alpha!r}"
            )


DEFAULT_VESSEL_MATERIALS = (
    MaterialProperties(
        name="glass_bath",
        display_name="Glass Bath",
        color_hex="#F28C28",
        category="glass",
        density_lb_per_ft3=156.0,
        thermal_conductivity_w_per_mk=1.1,
        thermal_expansion_um_per_m_c=9.0,
        preview_alpha=0.70,
    ),
    MaterialProperties(
        name="hot_face_refractory",
        display_name="Hot Face Refractory",
        color_hex="#D95F02",
        category="refractory",
        density_lb_per_ft3=165.0,
        thermal_conductivity_w_per_mk=1.6,
        thermal_expansion_um_per_m_c=5.0,
        preview_alpha=0.65,
    ),
    MaterialProperties(
        name="ifb",
        display_name="IFB",
        color_hex="#C7A46A",
        category="refractory",
        density_lb_per_ft3=52.0,
        thermal_conductivity_w_per_mk=0.25,
        thermal_expansion_um_per_m_c=4.0,
        preview_alpha=0.60,
    ),
    MaterialProperties(
        name="duraboard",
        display_name="Duraboard",
        color_hex="#F5F5F0",
        category="refractory",
        density_lb_per_ft3=18.0,
        thermal_conductivity_w_per_mk=0.08,
        thermal_expansion_um_per_m_c=2.0,
        preview_alpha=0.55,
    ),
    MaterialProperties(
        name="steel_shell",
        display_name="Steel Shell",
        color_hex="#8A8F98",
        category="shell",
        density_lb_per_ft3=490.0,
        thermal_conductivity_w_per_mk=45.0,
        thermal_expansion_um_per_m_c=12.0,
        preview_alpha=0.45,
    ),
    MaterialProperties(
        name="electrodes",
        display_name="Electrodes",
        color_hex="#2B2B2B",
        category="electrode",
        density_lb_per_ft3=112.0,
        thermal_conductivity_w_per_mk=120.0,
        thermal_expansion_um_per_m_c=4.0,
        preview_alpha=0.95,
    ),
)

DEFAULT_VESSEL_MATERIALS_BY_NAME = {
    material.name: material for material in DEFAULT_VESSEL_MATERIALS
}

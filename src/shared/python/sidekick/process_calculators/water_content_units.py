"""Shared water-content unit conversions for syngas calculators.

Concentration in g/m³, mg/Nm³, ppmv, and lb/MMscf is derived from a water
mole fraction and vapor partial pressure using one formula set so the process
engine and calc_backend fallback cannot drift (issues #6004, #3678).
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.python.contracts import require, require_positive

from ..utils.unit_constants import R_UNIVERSAL
from .constants import (
    MW_WATER_GMOL,
    NORMAL_PRESSURE_PA,
    NORMAL_TEMPERATURE_K,
    SCF_PER_LBMOL_60F_14696PSIA,
    STANDARD_BASIS_LB_MMSCF,
)

__all__ = [
    "WaterContentUnits",
    "water_content_units",
    "STANDARD_BASIS_LB_MMSCF",
]


@dataclass(frozen=True)
class WaterContentUnits:
    """Water concentrations in common engineering units."""

    g_m3: float
    mg_nm3: float
    ppmv: float
    lb_mmscf: float
    standard_basis: str = STANDARD_BASIS_LB_MMSCF


def water_content_units(
    y_water: float,
    vapor_pressure_pa: float,
    temperature_k: float,
) -> WaterContentUnits:
    """Convert water mole fraction and vapor partial pressure to unit systems.

    Precondition:
        ``y_water`` is in [0, 1], ``vapor_pressure_pa`` and ``temperature_k`` positive.
    Postcondition:
        ``g_m3`` uses actual ``vapor_pressure_pa`` and ``temperature_k``;
        ``mg_nm3`` at 0 °C and 1.01325 bar; ``lb_mmscf`` at 60 °F, 14.696 psia.
    """
    require(0.0 <= y_water <= 1.0, "y_water must be in [0, 1]")
    require_positive(vapor_pressure_pa, "vapor_pressure_pa")
    require_positive(temperature_k, "temperature_k")

    # ρ = P_w·MW / (R·T); R in J/(mol·K), MW in g/mol → g/m³ (R_UNIVERSAL).
    g_m3 = vapor_pressure_pa * MW_WATER_GMOL / (R_UNIVERSAL * temperature_k)

    mg_nm3 = (
        y_water
        * MW_WATER_GMOL
        * NORMAL_PRESSURE_PA
        / (R_UNIVERSAL * NORMAL_TEMPERATURE_K)
        * 1000.0
    )

    ppmv = y_water * 1e6

    # lb/MMscf = ppmv·(g/mol) / (scf/lb-mol); g/mol equals lb/lb-mol numerically.
    lb_mmscf = ppmv * MW_WATER_GMOL / SCF_PER_LBMOL_60F_14696PSIA

    return WaterContentUnits(
        g_m3=g_m3,
        mg_nm3=mg_nm3,
        ppmv=ppmv,
        lb_mmscf=lb_mmscf,
    )

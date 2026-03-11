"""
config/schema.py
================
Pydantic models that validate the configuration loaded from YAML files.

Why this matters:  If you give the model a negative mass flow rate or a
pressure in kPa when it expects Pa, you'll get wrong results silently.
Pydantic catches those mistakes immediately at load time with a clear
error message, before any DWSIM code runs.

Usage:
    from dwsim_model.config.schema import validate_master_config
    cfg = validate_master_config(raw_dict_from_yaml)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stream-level models
# ─────────────────────────────────────────────────────────────────────────────


class UltimateAnalysis(BaseModel):
    """Dry, ash-free ultimate analysis (mass fractions)."""

    C: float = Field(..., ge=0.0, le=1.0, description="Carbon fraction")
    H: float = Field(..., ge=0.0, le=1.0, description="Hydrogen fraction")
    O: float = Field(..., ge=0.0, le=1.0, description="Oxygen fraction")  # noqa: E741
    N: float = Field(0.0, ge=0.0, le=1.0, description="Nitrogen fraction")
    S: float = Field(0.0, ge=0.0, le=0.05, description="Sulfur fraction")
    Cl: float = Field(0.0, ge=0.0, le=0.02, description="Chlorine fraction")

    @model_validator(mode="after")
    def sum_must_be_one(self) -> UltimateAnalysis:
        total = self.C + self.H + self.O + self.N + self.S + self.Cl
        if abs(total - 1.0) > 0.02:
            raise ValueError(
                f"Ultimate analysis fractions sum to {total:.4f}, expected 1.0 ± 0.02"
            )
        return self


class ProximateAnalysis(BaseModel):
    """As-received proximate analysis (mass fractions)."""

    moisture: float = Field(..., ge=0.0, le=0.50)
    volatile_matter: float = Field(..., ge=0.0, le=1.0)
    fixed_carbon: float = Field(..., ge=0.0, le=1.0)
    ash: float = Field(..., ge=0.0, le=0.40)

    @model_validator(mode="after")
    def sum_must_be_one(self) -> ProximateAnalysis:
        total = self.moisture + self.volatile_matter + self.fixed_carbon + self.ash
        if abs(total - 1.0) > 0.02:
            raise ValueError(
                f"Proximate analysis fractions sum to {total:.4f}, expected 1.0 ± 0.02"
            )
        return self


class StreamConfig(BaseModel):
    """Configuration for a single material stream."""

    temperature_C: Optional[float] = Field(
        None,
        ge=-273.15,  # Absolute zero
        le=3000.0,  # Max reasonable process temp
        description="Stream temperature in °C",
    )
    temperature_K: Optional[float] = Field(
        None,
        ge=0.0,
        le=3273.15,
        description="Alternative: specify temperature in Kelvin",
    )
    pressure_Pa: Optional[float] = Field(
        None,
        ge=0.0,
        le=1e8,  # 1000 bar upper limit
        description="Stream pressure in Pa",
    )
    mass_flow_kg_s: Optional[float] = Field(
        None, ge=0.0, le=10000.0, description="Mass flow rate in kg/s"
    )
    components: Optional[dict[str, float]] = Field(
        None, description="Component mole fractions (must sum to ~1.0)"
    )
    ultimate_analysis: Optional[UltimateAnalysis] = None
    proximate_analysis: Optional[ProximateAnalysis] = None
    hhv_mj_kg: Optional[float] = Field(None, ge=0.0, le=60.0)

    @field_validator("pressure_Pa")
    @classmethod
    def pressure_above_vacuum(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 1000.0:
            raise ValueError(
                f"Pressure {v} Pa is very low (< 1000 Pa = 0.01 atm). "
                "Did you mean kPa? Config expects Pa."
            )
        return v

    @model_validator(mode="after")
    def components_sum_to_one(self) -> StreamConfig:
        if self.components:
            total = sum(self.components.values())
            if abs(total - 1.0) > 0.05:
                raise ValueError(
                    f"Component fractions sum to {total:.4f}, expected ~1.0. "
                    "Check that fractions are mole fractions (not percentages)."
                )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Reactor-level models
# ─────────────────────────────────────────────────────────────────────────────


class ReactionEntry(BaseModel):
    """A single reaction definition."""

    name: str
    stoichiometry: str
    base_component: Optional[str] = None
    conversion: Optional[float] = Field(None, ge=0.0, le=1.0)
    heat_of_reaction_kJ_mol: Optional[float] = None
    type: Optional[str] = None  # "equilibrium" | "kinetic" | "conversion"
    kinetics: Optional[KineticParameters] = None


class KineticParameters(BaseModel):
    """Arrhenius-style kinetic parameters for a PFR reaction."""

    pre_exponential_A: float = Field(..., gt=0.0)
    activation_energy_J_mol: float = Field(..., gt=0.0)
    reaction_order_n: float = Field(..., ge=0.0)


class ReactorConfig(BaseModel):
    """Configuration for a single reactor."""

    name: str
    type: str = Field(..., description="DWSIM reactor type, e.g. RCT_Conversion")
    temperature_C: float = Field(..., ge=0.0, le=5000.0)
    pressure_Pa: float = Field(..., ge=1000.0, le=1e7)
    mode: str = Field(
        "isothermal", description="isothermal | adiabatic | specified_duty"
    )
    volume_m3: Optional[float] = Field(None, gt=0.0)
    length_m: Optional[float] = Field(None, gt=0.0)
    diameter_m: Optional[float] = Field(None, gt=0.0)
    reactions: list[ReactionEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reactor_contract(self) -> ReactorConfig:
        for reaction in self.reactions:
            if self.type == "RCT_Conversion":
                if reaction.base_component is None:
                    raise ValueError(
                        f"Conversion reaction '{reaction.name}' requires base_component."
                    )
                if reaction.conversion is None:
                    raise ValueError(
                        f"Conversion reaction '{reaction.name}' requires conversion."
                    )

            if self.type == "RCT_PFR":
                if reaction.base_component is None:
                    raise ValueError(
                        f"PFR reaction '{reaction.name}' requires base_component."
                    )
                if reaction.kinetics is None:
                    raise ValueError(
                        f"PFR reaction '{reaction.name}' requires kinetics."
                    )

        if self.type == "RCT_PFR":
            missing_geometry = [
                field_name
                for field_name in ("volume_m3", "length_m", "diameter_m")
                if getattr(self, field_name) is None
            ]
            if missing_geometry:
                raise ValueError(
                    "PFR reactor requires geometry fields: "
                    + ", ".join(missing_geometry)
                )

        return self


# ─────────────────────────────────────────────────────────────────────────────
# Scenario model
# ─────────────────────────────────────────────────────────────────────────────


class ScenarioTargets(BaseModel):
    cold_gas_efficiency_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    carbon_conversion_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    h2_co_ratio_target: Optional[float] = Field(None, ge=0.0, le=20.0)
    syngas_temperature_outlet_C: Optional[float] = Field(None, ge=-50.0, le=1500.0)
    tar_loading_mg_Nm3_max: Optional[float] = Field(None, ge=0.0)


class ScenarioConfig(BaseModel):
    name: str = "unnamed"
    description: str = ""
    overrides: dict[str, Any] = Field(default_factory=dict)
    targets: Optional[ScenarioTargets] = None


# ─────────────────────────────────────────────────────────────────────────────
# Master config model
# ─────────────────────────────────────────────────────────────────────────────


class ModelInfo(BaseModel):
    name: str = "Gasification Model"
    version: str = "2.0"
    description: str = ""


class OutputConfig(BaseModel):
    directory: str = "results/"
    formats: list[str] = Field(default_factory=lambda: ["json", "html"])
    save_dwxml: bool = True
    dwxml_name: str = "Gasification_Model_GUI.dwxml"


class MasterConfig(BaseModel):
    model: ModelInfo = Field(default_factory=ModelInfo)
    reactor_mode: str = Field(
        "mixed", description="mixed | kinetic | equilibrium | conversion | custom"
    )
    compound_set: str = Field("standard", description="minimal | standard | extended")
    feeds: dict[str, str] = Field(
        default_factory=dict, description="Feed sub-file paths"
    )
    reactors: dict[str, str] = Field(default_factory=dict)
    energy: Optional[str] = None
    equipment: Optional[str] = None
    scenario: Optional[str] = None
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("reactor_mode")
    @classmethod
    def valid_mode(cls, v: str) -> str:
        allowed = {"mixed", "kinetic", "equilibrium", "conversion", "custom"}
        if v not in allowed:
            raise ValueError(f"reactor_mode '{v}' not in {allowed}")
        return v

    @field_validator("compound_set")
    @classmethod
    def valid_compound_set(cls, v: str) -> str:
        allowed = {"minimal", "standard", "extended"}
        if v not in allowed:
            raise ValueError(f"compound_set '{v}' not in {allowed}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Validation entry point
# ─────────────────────────────────────────────────────────────────────────────


def validate_master_config(raw: dict) -> MasterConfig:
    """
    Validate a raw dictionary against MasterConfig schema.

    Returns a validated MasterConfig object.
    Raises ``pydantic.ValidationError`` with helpful messages if invalid.

    Example::

        import yaml
        raw = yaml.safe_load(open("config/master_config.yaml"))
        cfg = validate_master_config(raw)
        print(cfg.reactor_mode)
    """
    return MasterConfig.model_validate(raw)


def validate_stream_config(raw: dict, stream_name: str = "") -> StreamConfig:
    """
    Validate a single stream configuration dict.

    Raises ``pydantic.ValidationError`` with helpful messages on failure.
    """
    try:
        return StreamConfig.model_validate(raw)
    except Exception as exc:
        prefix = f"[{stream_name}] " if stream_name else ""
        raise ValueError(f"{prefix}Invalid stream config: {exc}") from exc


def validate_reactor_config(raw: dict, reactor_name: str = "") -> ReactorConfig:
    """Validate a reactor configuration dict."""
    try:
        return ReactorConfig.model_validate(raw)
    except Exception as exc:
        prefix = f"[{reactor_name}] " if reactor_name else ""
        raise ValueError(f"{prefix}Invalid reactor config: {exc}") from exc

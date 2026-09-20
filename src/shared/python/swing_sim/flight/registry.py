"""Flight model registry — all 7 literature models with citation metadata.

Ported from UpstreamDrift ``src/shared/python/physics/flight_models.py``
(``FlightModelType``, ``FlightModelRegistry``, ``compare_models``) for
epic #4103 / flight port #4107. The five constant-coefficient presets keep
their ``ConstantCoefficientSpec`` name/description/reference metadata —
that is the citation trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import (
    BallFlightModel,
    ConstantCoefficientModel,
    ConstantCoefficientSpec,
    MacDonaldHanzelyModel,
    WaterlooPennerModel,
)
from .types import FlightResult, LaunchConditions


class FlightModelType(Enum):
    """Available ball flight physics models."""

    WATERLOO_PENNER = "waterloo_penner"
    MACDONALD_HANZELY = "macdonald_hanzely"
    NATHAN = "nathan"
    BALLANTYNE = "ballantyne"
    JCOLE = "jcole"
    ROSPIE_DL = "rospie_dl"
    CHARRY_L3 = "charry_l3"


_CONSTANT_COEFFICIENT_SPECS: dict[FlightModelType, ConstantCoefficientSpec] = {
    FlightModelType.NATHAN: ConstantCoefficientSpec(
        name="Nathan",
        description="Constant Cd/Cl model with spin decay",
        reference="Nathan et al. (2018)",
        cd=0.22,
        cl=0.24,
        spin_decay=0.03,
    ),
    FlightModelType.BALLANTYNE: ConstantCoefficientSpec(
        name="Ballantyne",
        description="Constant Cd/Cl model for steady spin",
        reference="Ballantyne et al. (2012)",
        cd=0.20,
        cl=0.18,
        spin_decay=0.02,
    ),
    FlightModelType.JCOLE: ConstantCoefficientSpec(
        name="J. Cole",
        description="Constant Cd/Cl model with moderate decay",
        reference="Cole (2016)",
        cd=0.23,
        cl=0.22,
        spin_decay=0.04,
    ),
    FlightModelType.ROSPIE_DL: ConstantCoefficientSpec(
        name="Rospie DL",
        description="Constant Cd/Cl model tuned for driver launch",
        reference="Rospie & Layton (2014)",
        cd=0.21,
        cl=0.19,
        spin_decay=0.03,
    ),
    FlightModelType.CHARRY_L3: ConstantCoefficientSpec(
        name="Charry L3",
        description="Constant Cd/Cl model with higher drag",
        reference="Charry et al. (2017)",
        cd=0.24,
        cl=0.21,
        spin_decay=0.05,
    ),
}


CANONICAL_FLIGHT_MODEL = FlightModelType.WATERLOO_PENNER
"""The designated canonical literature flight model baseline (Penner 2003)."""


@dataclass(frozen=True)
class FlightModelMetadata:
    """Documented physics parameters and citation metadata for a flight model.

    Attributes:
        model_type: Registry key enum.
        name: Display name.
        reference: Scholarly citation.
        spin_decay_s_inv: Exponential spin decay rate lambda [1/s].
        lift_family: Aerodynamic lift formulation family.
        is_canonical: Whether this model is the canonical baseline.
    """

    model_type: FlightModelType
    name: str
    reference: str
    spin_decay_s_inv: float
    lift_family: str
    is_canonical: bool = False


FLIGHT_MODEL_METADATA: dict[FlightModelType, FlightModelMetadata] = {
    FlightModelType.WATERLOO_PENNER: FlightModelMetadata(
        model_type=FlightModelType.WATERLOO_PENNER,
        name="Waterloo/Penner",
        reference="Penner (2003); McPhee et al. (Waterloo)",
        spin_decay_s_inv=0.0,
        lift_family="power_fit",
        is_canonical=True,
    ),
    FlightModelType.MACDONALD_HANZELY: FlightModelMetadata(
        model_type=FlightModelType.MACDONALD_HANZELY,
        name="MacDonald-Hanzely",
        reference="MacDonald & Hanzely (1991)",
        spin_decay_s_inv=0.05,
        lift_family="linear_ode",
        is_canonical=False,
    ),
    FlightModelType.NATHAN: FlightModelMetadata(
        model_type=FlightModelType.NATHAN,
        name="Nathan",
        reference="Nathan et al. (2018)",
        spin_decay_s_inv=0.03,
        lift_family="constant_ratio",
        is_canonical=False,
    ),
    FlightModelType.BALLANTYNE: FlightModelMetadata(
        model_type=FlightModelType.BALLANTYNE,
        name="Ballantyne",
        reference="Ballantyne et al. (2012)",
        spin_decay_s_inv=0.02,
        lift_family="constant_ratio",
        is_canonical=False,
    ),
    FlightModelType.JCOLE: FlightModelMetadata(
        model_type=FlightModelType.JCOLE,
        name="J. Cole",
        reference="Cole (2016)",
        spin_decay_s_inv=0.04,
        lift_family="constant_ratio",
        is_canonical=False,
    ),
    FlightModelType.ROSPIE_DL: FlightModelMetadata(
        model_type=FlightModelType.ROSPIE_DL,
        name="Rospie DL",
        reference="Rospie & Layton (2014)",
        spin_decay_s_inv=0.03,
        lift_family="constant_ratio",
        is_canonical=False,
    ),
    FlightModelType.CHARRY_L3: FlightModelMetadata(
        model_type=FlightModelType.CHARRY_L3,
        name="Charry L3",
        reference="Charry et al. (2017)",
        spin_decay_s_inv=0.05,
        lift_family="constant_ratio",
        is_canonical=False,
    ),
}


class FlightModelRegistry:
    """Registry for managing flight models."""

    _models: dict[FlightModelType, BallFlightModel] = {}

    @classmethod
    def get_model(cls, model_type: FlightModelType) -> BallFlightModel:
        """Return the flight model instance for the given model type."""
        if model_type is None:
            raise ValueError("model_type must be provided")
        if not cls._models:
            cls._initialize()
        return cls._models[model_type]

    @classmethod
    def get_canonical_model(cls) -> BallFlightModel:
        """Return the designated canonical baseline model instance (Waterloo/Penner)."""
        return cls.get_model(CANONICAL_FLIGHT_MODEL)

    @classmethod
    def get_metadata(cls, model_type: FlightModelType) -> FlightModelMetadata:
        """Return documented physics metadata and spin decay rate for a model."""
        if model_type not in FLIGHT_MODEL_METADATA:
            raise ValueError(f"Unknown model_type: {model_type}")
        return FLIGHT_MODEL_METADATA[model_type]

    @classmethod
    def get_all_models(cls) -> list[BallFlightModel]:
        """Return all registered flight model instances."""
        if not cls._models:
            cls._initialize()
        return list(cls._models.values())

    @classmethod
    def reset(cls) -> None:
        """Clear the registry, forcing re-initialization on next access.

        Use in test teardown to prevent cross-test pollution from the shared
        class-level ``_models`` dict (UpstreamDrift issue #1775).
        """
        cls._models.clear()

    @classmethod
    def _initialize(cls) -> None:
        cls._models[FlightModelType.WATERLOO_PENNER] = WaterlooPennerModel()
        cls._models[FlightModelType.MACDONALD_HANZELY] = MacDonaldHanzelyModel()
        for model_type, spec in _CONSTANT_COEFFICIENT_SPECS.items():
            cls._models[model_type] = ConstantCoefficientModel(spec)


def compare_models(
    launch: LaunchConditions, models: list[BallFlightModel]
) -> dict[str, FlightResult]:
    """Compare multiple models for the same launch conditions."""
    if launch is None:
        raise ValueError("launch must be provided")
    results = {}
    for model in models:
        results[model.name] = model.simulate(launch)
    return results


__all__ = [
    "CANONICAL_FLIGHT_MODEL",
    "FLIGHT_MODEL_METADATA",
    "FlightModelMetadata",
    "FlightModelRegistry",
    "FlightModelType",
    "compare_models",
]

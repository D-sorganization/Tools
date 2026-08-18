"""Flight model registry — all 7 literature models with citation metadata.

Ported from UpstreamDrift ``src/shared/python/physics/flight_models.py``
(``FlightModelType``, ``FlightModelRegistry``, ``compare_models``) for
epic #4103 / flight port #4107. The five constant-coefficient presets keep
their ``ConstantCoefficientSpec`` name/description/reference metadata —
that is the citation trail.
"""

from __future__ import annotations

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
    "FlightModelRegistry",
    "FlightModelType",
    "compare_models",
]

"""Variation-plan registration and profile sampling for turf parameters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import cast

import numpy as np

from shared.python.contracts import require
from shared.python.swing_sim.variation import (
    NoiseSpec,
    PerturbationGroup,
    VariableDef,
    VariationPlan,
    register_variable,
    sample_inputs,
    variable_registry,
)

from .turf_contact import (
    TurfCalibrationStatus,
    TurfContactProfile,
    TurfProfileProvenance,
)

CATEGORY_TURF = "golf_club.turf"
TURF_STIFFNESS_KEY = f"{CATEGORY_TURF}.normal_stiffness_n_m"
TURF_DAMPING_KEY = f"{CATEGORY_TURF}.normal_damping_n_s_m"
TURF_FRICTION_KEY = f"{CATEGORY_TURF}.friction_coefficient"
TURF_PENETRATION_LIMIT_KEY = f"{CATEGORY_TURF}.max_penetration_m"
TURF_VARIABLE_KEYS = frozenset(
    {
        TURF_STIFFNESS_KEY,
        TURF_DAMPING_KEY,
        TURF_FRICTION_KEY,
        TURF_PENETRATION_LIMIT_KEY,
    }
)


@dataclass(frozen=True)
class TurfVariationPlan:
    """Reproducible turf-only plan using the canonical seeded sampler."""

    noise: tuple[NoiseSpec, ...]
    base_variables: Mapping[str, float] = field(default_factory=dict)
    n_runs: int = 200
    seed: int = 0
    groups: tuple[PerturbationGroup, ...] = ()

    def __post_init__(self) -> None:
        require(self.n_runs >= 1, "n_runs must be >= 1", self.n_runs)
        require(self.seed >= 0, "seed must be >= 0", self.seed)
        require(self.noise, "plan must vary at least one turf variable")
        require(not self.groups, "grouped turf variation is not yet supported")
        require(
            all(isinstance(spec, NoiseSpec) for spec in self.noise),
            "noise entries must be NoiseSpec",
        )
        keys = tuple(spec.variable_key for spec in self.noise)
        require(set(keys) <= TURF_VARIABLE_KEYS, "noise contains non-turf variables")
        require(len(keys) == len(set(keys)), "noise variable keys must be unique")
        base = {str(key): float(value) for key, value in self.base_variables.items()}
        require(set(base) <= TURF_VARIABLE_KEYS, "base contains non-turf variables")
        require(
            all(math.isfinite(value) for value in base.values()),
            "base must be finite",
        )
        object.__setattr__(self, "noise", tuple(self.noise))
        object.__setattr__(self, "base_variables", MappingProxyType(base))

    def resolved_base(self) -> dict[str, float]:
        """Return registry defaults overlaid by explicit turf base values."""
        registry = variable_registry()
        resolved = {key: registry[key].default for key in TURF_VARIABLE_KEYS}
        resolved.update(self.base_variables)
        return resolved


def _register_turf_variables() -> None:
    guidance = (
        "Illustrative sensitivity scale only; use a calibrated distribution "
        "with provenance before making turf-supported comparisons."
    )
    definitions = (
        VariableDef(
            TURF_STIFFNESS_KEY,
            "Turf Normal Stiffness",
            "N/m",
            60_000.0,
            5_000.0,
            guidance,
        ),
        VariableDef(
            TURF_DAMPING_KEY,
            "Turf Normal Damping",
            "N·s/m",
            220.0,
            20.0,
            guidance,
        ),
        VariableDef(
            TURF_FRICTION_KEY,
            "Turf Friction Coefficient",
            "",
            0.35,
            0.05,
            guidance,
        ),
        VariableDef(
            TURF_PENETRATION_LIMIT_KEY,
            "Turf Penetration Limit",
            "m",
            0.025,
            0.005,
            guidance,
        ),
    )
    for definition in definitions:
        register_variable(definition)


def turf_profiles_for_variation_plan(
    plan: TurfVariationPlan,
    base_profile: TurfContactProfile,
) -> tuple[TurfContactProfile, ...]:
    """Sample a turf-only plan and conservatively downgrade derived profiles."""
    require(isinstance(plan, TurfVariationPlan), "plan must be a TurfVariationPlan")
    require(
        isinstance(base_profile, TurfContactProfile),
        "base_profile must be a TurfContactProfile",
    )
    requested = {spec.variable_key for spec in plan.noise} | set(plan.base_variables)
    require(
        requested <= TURF_VARIABLE_KEYS,
        "turf profile plan contains non-turf variables",
        sorted(requested - TURF_VARIABLE_KEYS),
    )
    # This plan mirrors VariationPlan's sampling-only shape but cannot enter
    # the generic swing/flight evaluator, where turf keys would be false inputs.
    samples = np.asarray(sample_inputs(cast(VariationPlan, plan)), dtype=float)
    profiles: list[TurfContactProfile] = []
    keys = tuple(spec.variable_key for spec in plan.noise)
    for index, row in enumerate(samples):
        values = dict(plan.base_variables)
        values.update(zip(keys, row, strict=True))
        profiles.append(
            replace(
                base_profile,
                profile_id=f"{base_profile.profile_id}-variation-{index:05d}",
                normal_stiffness_n_m=values.get(
                    TURF_STIFFNESS_KEY, base_profile.normal_stiffness_n_m
                ),
                normal_damping_n_s_m=values.get(
                    TURF_DAMPING_KEY, base_profile.normal_damping_n_s_m
                ),
                friction_coefficient=values.get(
                    TURF_FRICTION_KEY, base_profile.friction_coefficient
                ),
                max_penetration_m=values.get(
                    TURF_PENETRATION_LIMIT_KEY, base_profile.max_penetration_m
                ),
                calibration_status=TurfCalibrationStatus.ILLUSTRATIVE,
                provenance=TurfProfileProvenance(
                    source_name=base_profile.provenance.source_name,
                    parameter_basis=(
                        base_profile.provenance.parameter_basis
                        + "; sampled by a variation plan"
                    ),
                    uncertainty_note=(
                        "Derived samples are illustrative until the plan's "
                        "distribution and bounds receive calibration evidence."
                    ),
                    source_uri=base_profile.provenance.source_uri,
                ),
            )
        )
    return tuple(profiles)


_register_turf_variables()

__all__ = [
    "CATEGORY_TURF",
    "TURF_DAMPING_KEY",
    "TURF_FRICTION_KEY",
    "TURF_PENETRATION_LIMIT_KEY",
    "TURF_STIFFNESS_KEY",
    "TURF_VARIABLE_KEYS",
    "TurfVariationPlan",
    "turf_profiles_for_variation_plan",
]

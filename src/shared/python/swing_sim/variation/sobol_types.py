"""Data structures and types for Sobol sensitivity and Saltelli sampling (#4253)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from shared.python.contracts import require

from .registry import variable_registry
from .spec import NoiseSpec

SOBOL_REPORT_SCHEMA_ID = "swing_sim.variation.sobol_report"
SOBOL_REPORT_SCHEMA_VERSION = 1

_MIN_ADEQUATE_BASE_SAMPLES = 256
_MIN_MARGINAL_BASE_SAMPLES = 64
_CONSTANT_TOLERANCE = 1e-12


def _require_int(value: object, name: str, min_val: int) -> int:
    require(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer)),
        f"{name} must be an integer >= {min_val}",
        value,
    )
    val = int(cast(int | np.integer[Any], value))
    require(val >= min_val, f"{name} must be an integer >= {min_val}", val)
    return val


@dataclass(frozen=True)
class SobolFactor:
    """One bounded factor for Saltelli sampling and Sobol decomposition."""

    spec_id: str
    variable_key: str
    lower: float
    upper: float
    unit: str

    @classmethod
    def from_noise_spec(
        cls, spec: NoiseSpec, lower: float, upper: float
    ) -> SobolFactor:
        """Derive a bounded SobolFactor from an active NoiseSpec."""
        require(isinstance(spec, NoiseSpec), "spec must be a NoiseSpec", spec)
        spec_id = spec.spec_id if spec.spec_id is not None else spec.variable_key
        registry = variable_registry()
        unit = registry[spec.variable_key].unit if spec.variable_key in registry else ""
        return cls(
            spec_id=spec_id,
            variable_key=spec.variable_key,
            lower=lower,
            upper=upper,
            unit=unit,
        )

    def __post_init__(self) -> None:
        require(bool(self.spec_id.strip()), "spec_id must be non-empty")
        require(bool(self.variable_key.strip()), "variable_key must be non-empty")
        require(
            math.isfinite(self.lower)
            and math.isfinite(self.upper)
            and self.lower < self.upper,
            "bounds must be finite with lower < upper",
            (self.lower, self.upper),
        )


@dataclass(frozen=True)
class SobolOutput:
    """Target output variable for Sobol sensitivity analysis."""

    name: str
    unit: str = ""


@dataclass(frozen=True)
class SobolRunCountGuidance:
    """Run-count guidance for Sobol variance-based estimation."""

    factor_count: int
    base_samples: int
    total_runs: int
    adequacy: str
    guidance_message: str


def sobol_run_count_guidance(
    factor_count: int, base_samples: int = _MIN_ADEQUATE_BASE_SAMPLES
) -> SobolRunCountGuidance:
    """Calculate total model evaluations and adequacy guidance."""
    k = _require_int(factor_count, "factor_count", 1)
    n = _require_int(base_samples, "base_samples", 2)
    total_runs = n * (k + 2)
    adequacy = (
        "adequate"
        if n >= _MIN_ADEQUATE_BASE_SAMPLES
        else ("marginal" if n >= _MIN_MARGINAL_BASE_SAMPLES else "inadequate")
    )
    rec = _MIN_ADEQUATE_BASE_SAMPLES
    msg = (
        f"Sobol Saltelli sampling for k={k} factors with N={n} base samples requires "
        f"{total_runs} total model evaluations (N*(k+2)). Adequacy is '{adequacy}'. "
        f"Recommended N >= {rec} for stable S1 and ST convergence."
    )
    return SobolRunCountGuidance(
        factor_count=k,
        base_samples=n,
        total_runs=total_runs,
        adequacy=adequacy,
        guidance_message=msg,
    )


@dataclass(frozen=True)
class SaltelliDesign:
    """Saltelli (N*(k+2)) sampling design matrix and metadata."""

    factors: tuple[SobolFactor, ...]
    base_samples: int
    seed: int
    physical_points: np.ndarray  # Shape: (total_samples, k)

    @property
    def total_samples(self) -> int:
        return int(self.physical_points.shape[0])


@dataclass(frozen=True)
class SobolObservations:
    """Evaluated outputs collected from a Saltelli design."""

    design: SaltelliDesign
    outputs: tuple[SobolOutput, ...]
    values: np.ndarray  # Shape: (total_samples, n_outputs)
    success: np.ndarray  # Shape: (total_samples,) boolean


@dataclass(frozen=True)
class SobolEstimate:
    """First-order (S1) and total-order (ST) indices for one factor/output pair."""

    spec_id: str
    variable_key: str
    output_name: str
    s1: float
    st: float
    s1_ci: tuple[float, float]
    st_ci: tuple[float, float]
    availability: str
    total_evaluations: int
    valid_evaluations: int


@dataclass(frozen=True)
class SobolReport:
    """Comprehensive Sobol global sensitivity report."""

    estimates: tuple[SobolEstimate, ...]
    guidance: SobolRunCountGuidance
    base_samples: int
    total_design_samples: int
    seed: int
    method: str = "saltelli-sobol"

    def estimate(self, spec_id: str, output_name: str) -> SobolEstimate:
        matches = [
            e
            for e in self.estimates
            if e.spec_id == spec_id and e.output_name == output_name
        ]
        require(
            len(matches) == 1,
            "unknown Sobol factor/output pair",
            (spec_id, output_name),
        )
        return matches[0]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_id": SOBOL_REPORT_SCHEMA_ID,
            "schema_version": SOBOL_REPORT_SCHEMA_VERSION,
            "method": self.method,
            "base_samples": self.base_samples,
            "total_design_samples": self.total_design_samples,
            "seed": self.seed,
            "guidance": {
                "factor_count": self.guidance.factor_count,
                "base_samples": self.guidance.base_samples,
                "total_runs": self.guidance.total_runs,
                "adequacy": self.guidance.adequacy,
                "guidance_message": self.guidance.guidance_message,
            },
            "estimates": [
                {
                    "spec_id": e.spec_id,
                    "variable_key": e.variable_key,
                    "output_name": e.output_name,
                    "s1": e.s1,
                    "st": e.st,
                    "s1_ci": list(e.s1_ci),
                    "st_ci": list(e.st_ci),
                    "availability": e.availability,
                    "total_evaluations": e.total_evaluations,
                    "valid_evaluations": e.valid_evaluations,
                }
                for e in self.estimates
            ],
        }

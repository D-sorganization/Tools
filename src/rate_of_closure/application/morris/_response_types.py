"""Private immutable values exposed through the Morris response contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

JobStatus = Literal["queued", "running", "completed", "cancelled", "failed"]


@dataclass(frozen=True)
class MorrisCapability:
    """Validated authority discovery document."""

    available: bool
    api_prefix: str
    request_schema_id: str
    job_schema_id: str


@dataclass(frozen=True)
class MorrisSource:
    """Immutable source provenance for one elementary effect."""

    spec_id: str
    variable_key: str
    unit: str
    bounds: tuple[float, float]
    time_window_s: tuple[float, float] | None
    point_ids: tuple[str, ...]


@dataclass(frozen=True)
class MorrisTarget:
    """Immutable target provenance for one elementary effect."""

    name: str
    unit: str
    kind: str
    time_s: float | None
    point_id: str | None
    coordinate_frame: str | None


@dataclass(frozen=True)
class MorrisEffects:
    """Available or explicitly unavailable Morris effect values."""

    mu: float | None
    mu_star: float | None
    mu_star_standard_error: float | None
    sigma: float | None


@dataclass(frozen=True)
class MorrisDenominator:
    """Exact effect-pair availability diagnostics."""

    total_pairs: int
    valid_pairs: int
    typed_no_impact_pairs: int
    no_impact_unavailable_pairs: int
    failed_pairs: int
    nonfinite_pairs: int


@dataclass(frozen=True)
class MorrisResponseEstimate:
    """Validated immutable report estimate."""

    source: MorrisSource
    target: MorrisTarget
    effects: MorrisEffects
    availability: str
    sample_adequacy: str
    denominator: MorrisDenominator


@dataclass(frozen=True)
class MorrisResponseReport:
    """Validated Morris report used by UI-neutral presentation."""

    trajectories: int
    levels: int
    seed: int
    total_samples: int
    normalized_step: float
    assumptions: tuple[str, ...]
    interaction_caveat: str
    estimates: tuple[MorrisResponseEstimate, ...]


@dataclass(frozen=True)
class MorrisResponseJob:
    """Validated terminal or in-progress job response."""

    job_id: str
    request_id: str
    status: JobStatus
    completed_samples: int
    total_samples: int
    cancel_requested: bool
    report: MorrisResponseReport | None
    error_code: str | None
    error_message: str | None

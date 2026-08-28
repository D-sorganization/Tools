"""Putt dispersion outcomes, summary, and wire (epic #4800, P5).

What this module owns
---------------------
The *outcome* vocabulary of a Monte-Carlo putting study and its
versioned report wire ``swing_sim.putt_dispersion/1``. The sampling
and execution live in :mod:`.variation`, which builds on the shared
``swing_sim.variation`` engine; this module is deliberately downstream
of nothing but the shared statistics, so the same summary shape can be
computed from outcomes that came from anywhere (including the
TypeScript twin, which consumes this wire rather than re-running the
canonical seeded sampler).

Metrics
-------
Three families, the ones a putting fitting or green-reading study is
actually asked for:

* **Make percentage** — the fraction of runs the integrator captured,
  as a percentage. Capture is the integrator's decision under the
  declared capture model (P2's Holmes/Penner effective radius by
  default), never a post-hoc radius test.
* **Leave distance** — how far from the hole the ball came to rest,
  ``0.0`` for a holed putt. Reported as mean / median / p95 / max,
  because the tail is what costs strokes: the p95 leave is the length
  of the comebacker a golfer must expect one putt in twenty.
* **Start-line dispersion** — the spread of the launch azimuth off the
  target line. This is the *stroke's* dispersion, upstream of the
  green: it is the quantity a putter's MOI acts on (through P1's
  effective-mass law) and the one a fitting comparison must isolate
  from green-reading error.

Spread is the shared ``finite_sample_standard_deviation`` (sample
standard deviation, ddof = 1, ``NaN`` below two samples) and
percentiles are NumPy's linear interpolation — the same conventions
``swing_sim.variation.analysis.summary_stats`` already reports, so a
putting study and a full-swing study are read the same way.

Wire posture
------------
``swing_sim.putt_dispersion/1``: sorted keys, compact separators,
``allow_nan=False``, unknown fields refused, missing fields refused,
byte-deterministic within a runtime — the same idiom as
``swing_sim.green_surface/1``. ``allow_nan=False`` is load-bearing
here: a one-run study has an undefined sample spread, so it is
**refused** rather than serialized as ``NaN``.

The wire carries the declared distributions (variable key,
distribution, scale) alongside the summary, so a report says what was
varied and by how much — a dispersion number without its declared
input variance is not evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.python.contracts import require, require_finite
from shared.python.swing_sim.variation.analysis import (
    finite_sample_standard_deviation,
)

PUTT_DISPERSION_FORMAT = "swing_sim.putt_dispersion/1"

__all__ = [
    "PUTT_DISPERSION_FORMAT",
    "PuttDispersionReport",
    "PuttDispersionSummary",
    "PuttOutcome",
    "PuttVariableDeclaration",
    "putt_dispersion_from_json",
    "putt_dispersion_to_json",
    "summarize_putt_outcomes",
]

_OUTCOME_FIELDS = (
    "holed",
    "start_azimuth_deg",
    "leave_distance_m",
    "total_distance_m",
    "break_m",
    "capture_margin_m",
)

_SUMMARY_FIELDS = (
    "n_runs",
    "holed_count",
    "make_percent",
    "leave_mean_m",
    "leave_p50_m",
    "leave_p95_m",
    "leave_max_m",
    "start_line_mean_deg",
    "start_line_sigma_deg",
    "start_line_p05_deg",
    "start_line_p95_deg",
    "total_distance_mean_m",
    "total_distance_sigma_m",
)

_SUMMARY_FLOAT_FIELDS = _SUMMARY_FIELDS[2:]

_DECLARATION_FIELDS = frozenset({"variable_key", "distribution", "scale"})
_REPORT_FIELDS = frozenset({"format", "scenario_id", "seed", "variables", "summary"})

#: Two samples are the minimum for a sample standard deviation; a
#: one-run "study" has no dispersion and the wire refuses to pretend.
_MIN_RUNS_FOR_SPREAD = 2


@dataclass(frozen=True)
class PuttOutcome:
    """What one sampled putt did.

    Attributes:
        holed: The integrator's capture decision under the declared
            capture model.
        start_azimuth_deg: Launch direction off the target line [deg],
            ``+`` = right (P1 convention).
        leave_distance_m: Rest-to-hole distance [m]; ``0.0`` when holed.
        total_distance_m: Ground covered [m].
        break_m: Lateral offset at rest or capture [m], left positive.
        capture_margin_m: Effective hole radius at the closest approach
            minus that approach distance [m] (Holmes/Penner; positive
            iff the ball passed inside the effective hole).
    """

    holed: bool
    start_azimuth_deg: float
    leave_distance_m: float
    total_distance_m: float
    break_m: float
    capture_margin_m: float

    def __post_init__(self) -> None:
        require(isinstance(self.holed, bool), "holed must be boolean")
        for name in _OUTCOME_FIELDS[1:]:
            require_finite(getattr(self, name), name)
        require(
            self.leave_distance_m >= 0.0,
            "leave_distance_m must be non-negative",
            self.leave_distance_m,
        )
        require(
            self.total_distance_m >= 0.0,
            "total_distance_m must be non-negative",
            self.total_distance_m,
        )
        require(
            not self.holed or self.leave_distance_m == 0.0,
            "a holed putt leaves nothing",
            self.leave_distance_m,
        )


@dataclass(frozen=True)
class PuttVariableDeclaration:
    """One declared distribution, recorded beside the summary."""

    variable_key: str
    distribution: str
    scale: float

    def __post_init__(self) -> None:
        require(
            isinstance(self.variable_key, str) and bool(self.variable_key.strip()),
            "variable_key must be a name",
        )
        require(
            isinstance(self.distribution, str) and bool(self.distribution.strip()),
            "distribution must be a name",
        )
        require_finite(self.scale, "scale")
        require(self.scale > 0.0, "scale must be positive", self.scale)


@dataclass(frozen=True)
class PuttDispersionSummary:
    """Make percentage, leave distribution, and start-line dispersion."""

    n_runs: int
    holed_count: int
    make_percent: float
    leave_mean_m: float
    leave_p50_m: float
    leave_p95_m: float
    leave_max_m: float
    start_line_mean_deg: float
    start_line_sigma_deg: float
    start_line_p05_deg: float
    start_line_p95_deg: float
    total_distance_mean_m: float
    total_distance_sigma_m: float

    def __post_init__(self) -> None:
        for name in ("n_runs", "holed_count"):
            value = getattr(self, name)
            require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{name} must be a non-negative integer",
                value,
            )
        require(
            self.holed_count <= self.n_runs,
            "holed_count cannot exceed n_runs",
            (self.holed_count, self.n_runs),
        )
        for name in _SUMMARY_FLOAT_FIELDS:
            require_finite(getattr(self, name), name)
        require(
            0.0 <= self.make_percent <= 100.0,
            "make_percent must be in [0, 100]",
            self.make_percent,
        )


@dataclass(frozen=True)
class PuttDispersionReport:
    """One dispersion study: identity, declared inputs, and summary."""

    scenario_id: str
    seed: int
    variables: tuple[PuttVariableDeclaration, ...]
    summary: PuttDispersionSummary

    def __post_init__(self) -> None:
        require(
            isinstance(self.scenario_id, str) and bool(self.scenario_id.strip()),
            "scenario_id must be a name",
        )
        require(
            isinstance(self.seed, int)
            and not isinstance(self.seed, bool)
            and self.seed >= 0,
            "seed must be a non-negative integer",
            self.seed,
        )
        require(
            isinstance(self.variables, tuple)
            and all(
                isinstance(item, PuttVariableDeclaration) for item in self.variables
            ),
            "variables must be a tuple of PuttVariableDeclaration",
        )
        require(
            isinstance(self.summary, PuttDispersionSummary),
            "summary must be a PuttDispersionSummary",
        )


def _percentile(values: np.ndarray, fraction: float) -> float:
    result: np.ndarray = np.asarray(np.percentile(values, fraction), dtype=float)
    return float(result)


def summarize_putt_outcomes(
    outcomes: tuple[PuttOutcome, ...],
) -> PuttDispersionSummary:
    """Summarize a cohort of sampled putts (module docstring).

    Args:
        outcomes: At least two evaluated putts — a sample standard
            deviation is undefined below that, and this module refuses
            to report an undefined spread.

    Returns:
        The :class:`PuttDispersionSummary`.

    Raises:
        TypeError: If ``outcomes`` is not a tuple of :class:`PuttOutcome`.
        ContractViolationError: If fewer than two outcomes are given.
    """
    if not isinstance(outcomes, tuple) or not all(
        isinstance(item, PuttOutcome) for item in outcomes
    ):
        raise TypeError("outcomes must be a tuple of PuttOutcome")
    require(
        len(outcomes) >= _MIN_RUNS_FOR_SPREAD,
        "a dispersion summary needs at least two runs",
        len(outcomes),
    )
    leaves = np.asarray([item.leave_distance_m for item in outcomes], dtype=float)
    starts = np.asarray([item.start_azimuth_deg for item in outcomes], dtype=float)
    totals = np.asarray([item.total_distance_m for item in outcomes], dtype=float)
    holed = sum(1 for item in outcomes if item.holed)
    return PuttDispersionSummary(
        n_runs=len(outcomes),
        holed_count=holed,
        make_percent=100.0 * holed / len(outcomes),
        leave_mean_m=float(np.mean(leaves)),
        leave_p50_m=_percentile(leaves, 50.0),
        leave_p95_m=_percentile(leaves, 95.0),
        leave_max_m=float(np.max(leaves)),
        start_line_mean_deg=float(np.mean(starts)),
        start_line_sigma_deg=finite_sample_standard_deviation(starts),
        start_line_p05_deg=_percentile(starts, 5.0),
        start_line_p95_deg=_percentile(starts, 95.0),
        total_distance_mean_m=float(np.mean(totals)),
        total_distance_sigma_m=finite_sample_standard_deviation(totals),
    )


def putt_dispersion_to_json(report: PuttDispersionReport) -> str:
    """Serialize deterministically; identical studies are byte-identical."""
    if not isinstance(report, PuttDispersionReport):
        raise TypeError("report must be PuttDispersionReport")
    payload: dict[str, Any] = {
        "format": PUTT_DISPERSION_FORMAT,
        "scenario_id": report.scenario_id,
        "seed": report.seed,
        "variables": [
            {
                "variable_key": item.variable_key,
                "distribution": item.distribution,
                "scale": item.scale,
            }
            for item in report.variables
        ],
        "summary": {name: getattr(report.summary, name) for name in _SUMMARY_FIELDS},
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _require_exact_keys(
    data: object, expected: frozenset[str], what: str
) -> dict[str, Any]:
    """Refuse anything but an object carrying exactly ``expected``."""
    if not isinstance(data, dict):
        raise TypeError(f"{what} must be an object")
    section: dict[str, Any] = data
    require(
        set(section) == expected,
        f"{what} fields must be exactly {sorted(expected)}",
    )
    return section


def _finite_number(value: object, name: str) -> float:
    """A strict JSON number: int or float, never bool, always finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    require(math.isfinite(result), f"{name} must be finite", value)
    return result


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def putt_dispersion_from_json(text: str) -> PuttDispersionReport:
    """Parse and validate; unknown fields and wrong formats are refused."""
    require(isinstance(text, str), "text must be str")
    data = json.loads(text)
    require(
        isinstance(data, dict) and data.get("format") == PUTT_DISPERSION_FORMAT,
        f"format must be {PUTT_DISPERSION_FORMAT!r}",
    )
    body = _require_exact_keys(data, _REPORT_FIELDS, "putt dispersion")
    raw_variables = body["variables"]
    require(isinstance(raw_variables, list), "variables must be a list")
    variables = tuple(
        _declaration_from_json(item, index) for index, item in enumerate(raw_variables)
    )
    summary = _require_exact_keys(
        body["summary"], frozenset(_SUMMARY_FIELDS), "summary"
    )
    scenario_id = body["scenario_id"]
    require(isinstance(scenario_id, str), "scenario_id must be a string")
    return PuttDispersionReport(
        scenario_id=scenario_id,
        seed=_integer(body["seed"], "seed"),
        variables=variables,
        summary=PuttDispersionSummary(
            n_runs=_integer(summary["n_runs"], "n_runs"),
            holed_count=_integer(summary["holed_count"], "holed_count"),
            **{
                name: _finite_number(summary[name], name)
                for name in _SUMMARY_FLOAT_FIELDS
            },
        ),
    )


def _declaration_from_json(data: object, index: int) -> PuttVariableDeclaration:
    section = _require_exact_keys(data, _DECLARATION_FIELDS, f"variables[{index}]")
    for name in ("variable_key", "distribution"):
        require(isinstance(section[name], str), f"{name} must be a string")
    return PuttVariableDeclaration(
        variable_key=section["variable_key"],
        distribution=section["distribution"],
        scale=_finite_number(section["scale"], "scale"),
    )

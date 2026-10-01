"""Measured golfer kinematics, with sources, and a score against them.

The objective comparison can only answer "which objective looks most like a real
golfer" if there is a published, citable description of what a real golfer does.
This module holds that description as data, one entry per observable, each with
the measurement context it came from.

Heuristic Reference Intervals
-----------------------------
The six intervals below are **heuristic reference intervals** synthesized to score
planar double-pendulum simulations against plausible skilled golfer kinematics,
rather than exact primary-source reported literature bounds for a single professional-tour
cohort. Exact primary-source locations and measurement mappings for professional
tour players are not supplied for these widened heuristic bands.

For example, `Nesbit 2005 <https://www.jssm.org/jssm-04-499.xml.xml>`_ Table 3
reports grip speed 7.1–8.0 m/s and head speed 43–50 m/s for its described amateur
cohort (subjects across four handicap levels from scratch to high); this is not an
exact professional-tour 6–9 m/s / 45–55 m/s interval. Where sources diverge or describe
different cohorts, the intervals serve as heuristic working bands for model scoring
rather than certified empirical truth for elite tour players.

Sources
-------
* `Nesbit 2005, "A three dimensional kinematic and kinetic study of the golf
  swing" <https://www.jssm.org/jssm-04-499.xml.xml>`_ — full-body kinematics and
  kinetics for four skill levels; hand-path and grip-velocity profiles. Table 3
  reports amateur cohort grip speed 7.1–8.0 m/s and club head speed 43–50 m/s.
* `Nesbit & Serrano 2005, "Work and power analysis of the golf swing"
  <https://www.jssm.org/jssm-04-520.xml.xml>`_ — joint work and power budgets,
  the source for plausible hub torque magnitudes.
* `Jorgensen 1970, "On the dynamics of the swing of a golf club"
  <https://doi.org/10.1119/1.1976419>`_ — the canonical double-pendulum golf
  model and its timing.
* `Miura 2001, "Parametric acceleration — the effect of inward pull of the golf
  club at impact stage" <https://doi.org/10.1007/BF02844309>`_ — the measured
  inward hand pull and hand deceleration near impact; the real, much milder
  version of what an unconstrained optimizer exaggerates.
* `MacKenzie & Sprigings 2009, "A three-dimensional forward dynamics model of
  the golf swing" <https://doi.org/10.1007/s12283-009-0020-9>`_ — torque-driven
  forward-dynamics golfer, release timing and wrist torque magnitudes.
* `Sprigings & Neal 2000, "An insight into the importance of wrist torque in
  driving the golfball" <https://doi.org/10.1123/jab.16.4.356>`_ — wrist torque
  contribution and the passive-release argument.
* `Hill 1938, "The heat of shortening and the dynamic constants of muscle"
  <https://doi.org/10.1098/rspb.1938.0050>`_ — the torque-velocity relation used
  in :mod:`double_pendulum_golf.swing_objectives.actuation`.

Scope note: these are planar, driver-swing heuristic bands. They are used
to *score* a two-link model, not to claim the model reproduces an elite person.

Closes #4778.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ObservableBand",
    "TOUR_DRIVER_BANDS",
    "RealismScore",
    "score_against_reference",
]


@dataclass(frozen=True, slots=True)
class ObservableBand:
    """A heuristic reference interval for one swing observable.

    Attributes:
        key: Identifier used when scoring.
        label: Human-readable observable name.
        units: Units of ``low`` and ``high``.
        low: Lower end of the heuristic range.
        high: Upper end of the heuristic range.
        source: Short citation and provenance for the heuristic range.
        url: Link to the cited work.
    """

    key: str
    label: str
    units: str
    low: float
    high: float
    source: str
    url: str

    def __post_init__(self) -> None:
        """Validate that the band is a usable interval.

        Pre: none.
        Post: ``low < high`` and both are finite.
        """
        if not (np.isfinite(self.low) and np.isfinite(self.high)):
            raise ValueError(f"{self.key}: band bounds must be finite")
        if not self.low < self.high:
            raise ValueError(f"{self.key}: low must be below high")

    @property
    def midpoint(self) -> float:
        """Centre of the reported band."""
        return 0.5 * (self.low + self.high)

    @property
    def half_width(self) -> float:
        """Half the band width, used to normalise distance."""
        return 0.5 * (self.high - self.low)

    def contains(self, value: float) -> bool:
        """Whether a measurement falls inside the reported band."""
        return bool(self.low <= value <= self.high)

    def deviation(self, value: float) -> float:
        """Distance outside the band in half-widths; zero when inside.

        Using half-widths makes observables with different units comparable
        without inventing weights.
        """
        if self.contains(value):
            return 0.0
        excess = self.low - value if value < self.low else value - self.high
        return float(excess / self.half_width)


#: Heuristic reference intervals for a planar double-pendulum driver swing.
TOUR_DRIVER_BANDS: tuple[ObservableBand, ...] = (
    ObservableBand(
        key="clubhead_speed_ms",
        label="Clubhead speed at impact",
        units="m/s",
        low=45.0,
        high=55.0,
        source="Heuristic band (Nesbit 2005 Table 3 reports 43-50 m/s for amateur cohort; consistent with Jorgensen 1970)",
        url="https://www.jssm.org/jssm-04-499.xml.xml",
    ),
    ObservableBand(
        key="hand_speed_ms",
        label="Hand speed at impact",
        units="m/s",
        low=6.0,
        high=9.0,
        source="Heuristic band (Nesbit 2005 Table 3 reports 7.1-8.0 m/s grip speed for amateur cohort; Miura 2001)",
        url="https://doi.org/10.1007/BF02844309",
    ),
    ObservableBand(
        key="downswing_time_s",
        label="Downswing duration",
        units="s",
        low=0.23,
        high=0.32,
        source="Heuristic band (Jorgensen 1970; Nesbit 2005)",
        url="https://doi.org/10.1119/1.1976419",
    ),
    ObservableBand(
        key="club_arm_rate_ratio",
        label="Club / arm angular rate at impact",
        units="-",
        low=2.5,
        high=4.0,
        source="Heuristic band derived from Nesbit 2005 segment angular velocities",
        url="https://www.jssm.org/jssm-04-499.xml.xml",
    ),
    ObservableBand(
        key="wrist_cock_impact_deg",
        label="Wrist cock remaining at impact",
        units="deg",
        low=-5.0,
        high=20.0,
        source="Heuristic band (MacKenzie & Sprigings 2009 release timing)",
        url="https://doi.org/10.1007/s12283-009-0020-9",
    ),
    ObservableBand(
        key="release_fraction",
        label="Fraction of downswing before half release",
        units="-",
        low=0.55,
        high=0.80,
        source="Heuristic band (Sprigings & Neal 2000; MacKenzie & Sprigings 2009 delayed release)",
        url="https://doi.org/10.1123/jab.16.4.356",
    ),
)


@dataclass(frozen=True, slots=True)
class RealismScore:
    """How far a simulated swing sits outside measured golfer behaviour.

    Attributes:
        deviations: Per-observable distance outside its band, in half-widths.
        inside: Per-observable flag for falling inside the band.
        missing: Observables the caller did not supply.
    """

    deviations: dict[str, float]
    inside: dict[str, bool]
    missing: tuple[str, ...]

    @property
    def total_deviation(self) -> float:
        """Sum of distances outside the bands. Zero means fully golf-like."""
        return float(sum(self.deviations.values()))

    @property
    def worst(self) -> tuple[str, float]:
        """The observable furthest outside its band."""
        if not self.deviations:
            return ("", 0.0)
        key = max(self.deviations, key=lambda name: self.deviations[name])
        return key, self.deviations[key]

    @property
    def inside_count(self) -> int:
        """How many scored observables fall inside their measured band."""
        return sum(1 for value in self.inside.values() if value)


def score_against_reference(
    measurements: dict[str, float],
    bands: tuple[ObservableBand, ...] = TOUR_DRIVER_BANDS,
) -> RealismScore:
    """Score a simulated swing against measured golfer kinematics.

    Args:
        measurements: Observable key to simulated value. Keys absent from
            ``bands`` are ignored; bands absent from ``measurements`` are
            reported as missing rather than scored as zero.
        bands: Reference bands to score against.

    Returns:
        The per-observable deviations and a total.

    Pre: every supplied measurement is finite.
    Post: deviations are non-negative; zero means inside the band.
    """
    for key, value in measurements.items():
        if not np.isfinite(value):
            raise ValueError(f"measurement {key!r} must be finite, got {value}")

    deviations: dict[str, float] = {}
    inside: dict[str, bool] = {}
    missing: list[str] = []
    for band in bands:
        if band.key not in measurements:
            missing.append(band.key)
            continue
        value = measurements[band.key]
        deviations[band.key] = band.deviation(value)
        inside[band.key] = band.contains(value)
    return RealismScore(deviations=deviations, inside=inside, missing=tuple(missing))

"""Immutable measured shaft profiles with explicit SI and station semantics."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from enum import Enum

from ._validation import require_finite_float, require_identifier

_LENGTH_TOLERANCE_M = 1e-9


class ExtrapolationPolicy(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Policy applied when a requested station lies outside measured data."""

    REJECT = "reject"
    CLAMP = "clamp"


@dataclass(frozen=True)
class ShaftProfileProvenance:
    """Human-readable source and measurement limitations for one profile."""

    source_name: str
    measurement_method: str
    uncertainty_note: str
    source_uri: str | None = None
    data_license: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_name", "measurement_method", "uncertainty_note"):
            object.__setattr__(
                self,
                name,
                require_identifier(getattr(self, name), name),
            )
        for name in ("source_uri", "data_license"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_identifier(value, name))


@dataclass(frozen=True)
class ShaftStation:
    """Measured properties at one raw-shaft station from butt toward tip.

    ``ei_about_x_n_m2`` and ``ei_about_y_n_m2`` are engineering bending
    stiffnesses, not marketing flex labels. ``gj_n_m2`` is torsional
    stiffness. The spine angle is an unwrapped orientation about the shaft
    axis in the declared profile frame.
    """

    position_m: float
    outer_diameter_m: float
    inner_diameter_m: float
    linear_density_kg_m: float
    ei_about_x_n_m2: float
    ei_about_y_n_m2: float
    gj_n_m2: float
    damping_ratio: float
    spine_angle_rad: float = 0.0

    def __post_init__(self) -> None:
        position = require_finite_float(self.position_m, "position_m")
        if position < 0.0:
            raise ValueError("position_m must be >= 0")
        object.__setattr__(self, "position_m", position)
        for name in (
            "outer_diameter_m",
            "linear_density_kg_m",
            "ei_about_x_n_m2",
            "ei_about_y_n_m2",
            "gj_n_m2",
        ):
            object.__setattr__(
                self,
                name,
                require_finite_float(getattr(self, name), name, positive=True),
            )
        inner = require_finite_float(self.inner_diameter_m, "inner_diameter_m")
        if inner < 0.0 or inner >= self.outer_diameter_m:
            raise ValueError("inner_diameter_m must be >= 0 and < outer_diameter_m")
        object.__setattr__(self, "inner_diameter_m", inner)
        damping = require_finite_float(self.damping_ratio, "damping_ratio")
        if damping < 0.0 or damping >= 1.0:
            raise ValueError("damping_ratio must be in [0, 1)")
        object.__setattr__(self, "damping_ratio", damping)
        object.__setattr__(
            self,
            "spine_angle_rad",
            require_finite_float(self.spine_angle_rad, "spine_angle_rad"),
        )


@dataclass(frozen=True)
class ShaftProfile:
    """Station-based raw and trimmed shaft definition in SI units.

    Station zero is the raw butt and the final station is the raw tip.
    ``cut_length_m`` must equal raw length less declared butt and tip trims.
    Insertion depth reduces the mechanically exposed span but does not remove
    shaft mass from the cut assembly.
    """

    shaft_id: str
    frame_id: str
    raw_length_m: float
    cut_length_m: float
    tip_trim_m: float
    butt_trim_m: float
    insertion_depth_m: float
    stations: tuple[ShaftStation, ...]
    provenance: ShaftProfileProvenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shaft_id", require_identifier(self.shaft_id, "shaft_id")
        )
        object.__setattr__(
            self, "frame_id", require_identifier(self.frame_id, "frame_id")
        )
        self._normalize_lengths()
        self._validate_stations()
        if not isinstance(self.provenance, ShaftProfileProvenance):
            raise TypeError("provenance must be ShaftProfileProvenance")

    def _normalize_lengths(self) -> None:
        for name in ("raw_length_m", "cut_length_m"):
            object.__setattr__(
                self,
                name,
                require_finite_float(getattr(self, name), name, positive=True),
            )
        for name in ("tip_trim_m", "butt_trim_m", "insertion_depth_m"):
            value = require_finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)
        expected = self.raw_length_m - self.tip_trim_m - self.butt_trim_m
        if not math.isclose(
            self.cut_length_m,
            expected,
            rel_tol=0.0,
            abs_tol=_LENGTH_TOLERANCE_M,
        ):
            raise ValueError("cut_length_m must equal raw length less declared trims")
        if self.insertion_depth_m >= self.cut_length_m:
            raise ValueError("insertion_depth_m must be < cut_length_m")

    def _validate_stations(self) -> None:
        if not isinstance(self.stations, tuple):
            raise TypeError("stations must be a tuple")
        if len(self.stations) < 2 or not all(
            isinstance(station, ShaftStation) for station in self.stations
        ):
            raise ValueError("stations must contain at least two ShaftStation values")
        positions = tuple(station.position_m for station in self.stations)
        if any(
            right <= left for left, right in zip(positions, positions[1:], strict=False)
        ):
            raise ValueError("station positions must be strictly increasing")
        endpoints_match = math.isclose(
            positions[0], 0.0, abs_tol=_LENGTH_TOLERANCE_M
        ) and math.isclose(
            positions[-1], self.raw_length_m, abs_tol=_LENGTH_TOLERANCE_M
        )
        if not endpoints_match:
            raise ValueError("stations must include the raw butt and raw tip")

    @property
    def flexible_length_m(self) -> float:
        """Mechanically exposed length from fixed butt to hosel insertion."""
        return self.cut_length_m - self.insertion_depth_m

    @property
    def raw_mass_kg(self) -> float:
        """Integrated mass of the untrimmed measured shaft."""
        return self._integrated_mass_and_moment(0.0, self.raw_length_m)[0]

    @property
    def total_mass_kg(self) -> float:
        """Backward-readable alias for raw profile mass."""
        return self.raw_mass_kg

    @property
    def cut_mass_kg(self) -> float:
        """Integrated mass retained after butt and tip trimming."""
        start = self.butt_trim_m
        end = self.raw_length_m - self.tip_trim_m
        return self._integrated_mass_and_moment(start, end)[0]

    @property
    def balance_point_from_raw_butt_m(self) -> float:
        """Raw-shaft balance point measured from the raw butt datum."""
        mass, first_moment = self._integrated_mass_and_moment(0.0, self.raw_length_m)
        return first_moment / mass

    def station_at(
        self,
        position_m: float,
        policy: ExtrapolationPolicy = ExtrapolationPolicy.REJECT,
    ) -> ShaftStation:
        """Interpolate one station under an explicit extrapolation policy."""
        position = require_finite_float(position_m, "position_m")
        if not isinstance(policy, ExtrapolationPolicy):
            raise TypeError("policy must be ExtrapolationPolicy")
        if position < 0.0 or position > self.raw_length_m:
            if policy is ExtrapolationPolicy.REJECT:
                raise ValueError("position_m is outside the measured station range")
            position = min(max(position, 0.0), self.raw_length_m)
        positions = [station.position_m for station in self.stations]
        right = bisect.bisect_left(positions, position)
        if right == 0:
            return self.stations[0]
        if right == len(self.stations):
            return self.stations[-1]
        if math.isclose(positions[right], position, abs_tol=_LENGTH_TOLERANCE_M):
            return self.stations[right]
        left = right - 1
        fraction = (position - positions[left]) / (positions[right] - positions[left])
        return _interpolate_station(
            self.stations[left], self.stations[right], fraction, position
        )

    def _integrated_mass_and_moment(
        self, start_m: float, end_m: float
    ) -> tuple[float, float]:
        boundaries = [start_m, end_m]
        boundaries.extend(
            station.position_m
            for station in self.stations
            if start_m < station.position_m < end_m
        )
        points = sorted(boundaries)
        total_mass = 0.0
        first_moment = 0.0
        for left, right in zip(points, points[1:], strict=False):
            density_left = self.station_at(left).linear_density_kg_m
            density_right = self.station_at(right).linear_density_kg_m
            width = right - left
            element_mass = 0.5 * (density_left + density_right) * width
            slope = (density_right - density_left) / width
            local_moment = density_left * width**2 / 2.0 + slope * width**3 / 3.0
            total_mass += element_mass
            first_moment += left * element_mass + local_moment
        return total_mass, first_moment


def _interpolate_station(
    left: ShaftStation,
    right: ShaftStation,
    fraction: float,
    position_m: float,
) -> ShaftStation:
    def blend(name: str) -> float:
        start = float(getattr(left, name))
        return start + fraction * (float(getattr(right, name)) - start)

    return ShaftStation(
        position_m=position_m,
        outer_diameter_m=blend("outer_diameter_m"),
        inner_diameter_m=blend("inner_diameter_m"),
        linear_density_kg_m=blend("linear_density_kg_m"),
        ei_about_x_n_m2=blend("ei_about_x_n_m2"),
        ei_about_y_n_m2=blend("ei_about_y_n_m2"),
        gj_n_m2=blend("gj_n_m2"),
        damping_ratio=blend("damping_ratio"),
        spine_angle_rad=blend("spine_angle_rad"),
    )


__all__ = [
    "ExtrapolationPolicy",
    "ShaftProfile",
    "ShaftProfileProvenance",
    "ShaftStation",
]

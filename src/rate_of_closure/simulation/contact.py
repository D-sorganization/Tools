"""Typed clubhead-reference-point contact outcomes.

The fixed-ball detector is intentionally a first-order geometric
prerequisite.  It compares the sampled clubhead reference point with a
spherical ball; it does not perform clubface-mesh intersection, continuous
collision detection, or compliant-contact simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from rate_of_closure._contracts import require

__all__ = ["ContactMode", "ImpactOutcome", "ImpactStatus", "assess_fixed_contact"]

APP_FRAME = "app_frame:x_target,y_up,z_right"
FIXED_BALL_GEOMETRY = "sampled_reference_point_to_ball_sphere"
FIXED_BALL_LIMITATION = (
    "Sampled point-to-sphere proximity only; ignores the clubhead mesh, "
    "face plane and curvature, swept contact between samples, and ball compression."
)
FORCED_ALIGNMENT_LIMITATION = (
    "Delivery-inspection mode translates the entire swing so the selected "
    "clubhead reference point coincides with the ball center; this is not "
    "geometric contact detection."
)
_CONTACT_ABS_TOLERANCE_M = 1e-9


class ContactMode(StrEnum):
    """Supported policies for selecting or detecting an impact candidate."""

    DELIVERY_INSPECTION = "delivery_inspection"
    FIXED_BALL_CONTACT = "fixed_ball_contact"


class ImpactStatus(StrEnum):
    """Whether the selected contact policy produced a physical-impact candidate."""

    HIT = "hit"
    MISS = "miss"


@dataclass(frozen=True)
class ImpactOutcome:
    """Contact assessment retained by every simulation run.

    ``candidate_time_s`` is always present.  For a miss it is the sampled
    instant of closest approach, not an impact time.  A non-negative contact
    margin means the point-sphere surrogate intersects within numerical
    tolerance.
    """

    mode: ContactMode
    status: ImpactStatus
    candidate_time_s: float
    closest_approach_m: float
    contact_threshold_m: float
    contact_margin_m: float
    ball_position_m: tuple[float, float, float]
    frame: str
    geometry_model: str
    geometry_limitations: str

    def __post_init__(self) -> None:
        """Validate the immutable outcome and its signed-margin invariant."""
        require(
            math.isfinite(self.candidate_time_s) and self.candidate_time_s >= 0.0,
            "candidate_time_s must be finite and >= 0",
            self.candidate_time_s,
        )
        require(
            math.isfinite(self.closest_approach_m) and self.closest_approach_m >= 0.0,
            "closest_approach_m must be finite and >= 0",
            self.closest_approach_m,
        )
        _validate_threshold(self.contact_threshold_m)
        expected_margin = self.contact_threshold_m - self.closest_approach_m
        require(
            math.isclose(self.contact_margin_m, expected_margin, abs_tol=1e-12),
            "contact_margin_m must equal threshold minus closest approach",
            self.contact_margin_m,
        )
        expected_status = (
            ImpactStatus.HIT
            if self.contact_margin_m >= -_CONTACT_ABS_TOLERANCE_M
            else ImpactStatus.MISS
        )
        require(self.status is expected_status, "status must agree with contact margin")
        require(bool(self.frame), "frame metadata must be non-empty")
        require(bool(self.geometry_model), "geometry_model must be non-empty")
        require(
            bool(self.geometry_limitations),
            "geometry_limitations must be non-empty",
        )

    @property
    def is_hit(self) -> bool:
        """Return whether downstream impact and flight phases were run."""
        return self.status is ImpactStatus.HIT

    def to_dict(self) -> dict[str, Any]:
        """Return strict-JSON-compatible contact metadata."""
        return {
            "mode": self.mode.value,
            "status": self.status.value,
            "candidate_time_s": self.candidate_time_s,
            "closest_approach_m": self.closest_approach_m,
            "contact_threshold_m": self.contact_threshold_m,
            "contact_margin_m": self.contact_margin_m,
            "ball_position_m": list(self.ball_position_m),
            "frame": self.frame,
            "geometry_model": self.geometry_model,
            "geometry_limitations": self.geometry_limitations,
        }


def forced_alignment_outcome(
    candidate_time_s: float,
    ball_position_m: np.ndarray,
    contact_threshold_m: float,
) -> ImpactOutcome:
    """Describe the legacy translation-based delivery-inspection hit."""
    _validate_threshold(contact_threshold_m)
    return ImpactOutcome(
        mode=ContactMode.DELIVERY_INSPECTION,
        status=ImpactStatus.HIT,
        candidate_time_s=float(candidate_time_s),
        closest_approach_m=0.0,
        contact_threshold_m=float(contact_threshold_m),
        contact_margin_m=float(contact_threshold_m),
        ball_position_m=_point_tuple(ball_position_m),
        frame=APP_FRAME,
        geometry_model="forced_reference_point_alignment",
        geometry_limitations=FORCED_ALIGNMENT_LIMITATION,
    )


def assess_fixed_contact(
    times_s: np.ndarray,
    reference_positions_m: np.ndarray,
    ball_position_m: np.ndarray,
    contact_threshold_m: float,
) -> ImpactOutcome:
    """Assess sampled closest approach without translating the swing."""
    times = np.asarray(times_s, dtype=float)
    positions = np.asarray(reference_positions_m, dtype=float)
    ball = np.asarray(ball_position_m, dtype=float)
    _validate_samples(times, positions, ball)
    _validate_threshold(contact_threshold_m)
    distances = np.linalg.norm(positions - ball, axis=1)
    candidate_index = int(np.argmin(distances))
    closest = float(distances[candidate_index])
    margin = float(contact_threshold_m - closest)
    status = (
        ImpactStatus.HIT if margin >= -_CONTACT_ABS_TOLERANCE_M else ImpactStatus.MISS
    )
    return ImpactOutcome(
        mode=ContactMode.FIXED_BALL_CONTACT,
        status=status,
        candidate_time_s=float(times[candidate_index]),
        closest_approach_m=closest,
        contact_threshold_m=float(contact_threshold_m),
        contact_margin_m=margin,
        ball_position_m=_point_tuple(ball),
        frame=APP_FRAME,
        geometry_model=FIXED_BALL_GEOMETRY,
        geometry_limitations=FIXED_BALL_LIMITATION,
    )


def _validate_samples(
    times_s: np.ndarray, positions_m: np.ndarray, ball_position_m: np.ndarray
) -> None:
    """Validate the sampled proximity inputs."""
    require(times_s.ndim == 1 and len(times_s) > 0, "times_s must be non-empty 1-D")
    require(
        positions_m.shape == (len(times_s), 3),
        "reference_positions_m must have shape (N, 3)",
        positions_m.shape,
    )
    require(ball_position_m.shape == (3,), "ball_position_m must have shape (3,)")
    require(
        bool(np.all(np.isfinite(times_s)))
        and bool(np.all(np.isfinite(positions_m)))
        and bool(np.all(np.isfinite(ball_position_m))),
        "contact samples must be finite",
    )


def _validate_threshold(contact_threshold_m: float) -> None:
    """Validate the point-to-sphere contact distance."""
    require(
        math.isfinite(contact_threshold_m) and contact_threshold_m > 0.0,
        "contact_threshold_m must be finite and > 0",
        contact_threshold_m,
    )


def _point_tuple(point: np.ndarray) -> tuple[float, float, float]:
    """Convert a finite Cartesian point to an immutable tuple."""
    values = np.asarray(point, dtype=float)
    require(values.shape == (3,), "point must have shape (3,)", values.shape)
    require(bool(np.all(np.isfinite(values))), "point must be finite")
    return float(values[0]), float(values[1]), float(values[2])

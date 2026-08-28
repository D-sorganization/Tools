"""Atomic accepted-authority contracts for Putting presentation."""

import math
from dataclasses import dataclass
from numbers import Real

from rate_of_closure.putting_sample_inspector import PuttingSamplePlan
from shared.python.swing_sim.putting import PuttResult


@dataclass(frozen=True)
class AcceptedPuttingContext:
    """Complete bounded scientific authority shown beside retained evidence.

    Epic #4800 P6 widened this from the H3 pace-and-planar-slope record
    to the whole delivered configuration. The head-provenance kind and
    the green's own provenance label are part of the authority: a putt
    solved with a mesh-derived inertia tensor and one solved with the
    catalogue-default MOI are different experiments, and the displayed
    text must say which is on screen.

    ``stroke`` and ``green`` are the prebuilt bounded labels from the
    two control groups (``PuttStroke.label`` / ``PuttGreen.label``);
    this record composes them with the head identity, so the plot view
    keeps a single authority string to publish.
    """

    putter: str
    putter_source: str
    mass_kg: float
    loft_deg: float
    cor: float
    stroke: str
    green: str
    grade_percent: float
    aspect_deg: float
    hole_m: float

    def label(self) -> str:
        return (
            f"putter {self.putter} [{self.putter_source}] "
            f"({self.mass_kg:.3f} kg, {self.loft_deg:.1f} deg, "
            f"COR {self.cor:.2f}); {self.stroke}; {self.green}; "
            "kernel RK4-2ms-v1"
        )


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be finite")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def validate_putting_result_summary(
    result: PuttResult, plan: PuttingSamplePlan
) -> None:
    """Require scalar summaries to match the exact raw sample evidence."""
    for field in ("skid_distance_m", "total_distance_m", "time_s", "break_m"):
        _finite(getattr(result, field), field)
    if not isinstance(result.holed, bool):
        raise ValueError("holed must be boolean")
    for field in ("speed_at_hole_mps", "margin_mps", "miss_distance_m"):
        value = getattr(result, field)
        if value is not None and _finite(value, field) < 0:
            raise ValueError(f"{field} must be nonnegative")
    expected = (
        (result.total_distance_m, plan.cumulative_distance_m[-1]),
        (result.skid_distance_m, plan.cumulative_distance_m[plan.skid_end_index]),
        (result.time_s, plan.series.times_s[-1]),
        (result.break_m, plan.series.path_y_m[-1]),
    )
    if any(
        v < 0 for v in (result.total_distance_m, result.skid_distance_m, result.time_s)
    ):
        raise ValueError("putting distance and time summaries must be nonnegative")
    if any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in expected):
        raise ValueError("putting summary must match exact raw sample evidence")
    coherent = (
        result.speed_at_hole_mps is not None
        and result.margin_mps is not None
        and result.miss_distance_m is None
        if result.holed
        else result.margin_mps is None and result.miss_distance_m is not None
    )
    if not coherent:
        raise ValueError("putting capture summaries are internally inconsistent")


__all__ = ["AcceptedPuttingContext", "validate_putting_result_summary"]

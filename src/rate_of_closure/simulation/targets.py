"""Landing target regions for the app (epic #4125, H7b).

Thin app-side facade over the solver's
:class:`~shared.python.swing_sim.solver.TargetRegion` (green circle at a
distance / fairway corridor, exact signed distance + containment), plus
the Variation tie-in: the fraction of a Monte-Carlo dataset's landing
scatter holding the target — the headline output when a
:class:`~shared.python.swing_sim.variation.VariationDataset` exists —
and the course-layout bridge so the H7a course scene draws the green
where the target sits.
"""

from __future__ import annotations

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.ui.course import CourseLayout
from shared.python.swing_sim.solver import TargetRegion
from shared.python.swing_sim.variation import VariationDataset

__all__ = ["TargetRegion", "hold_fraction", "hold_stats", "layout_for_region"]


def hold_stats(
    carries_m: np.ndarray, laterals_m: np.ndarray, region: TargetRegion
) -> tuple[int, int]:
    """(shots holding the target, total shots) for a landing scatter.

    Points are the goals-module landing convention: carry downrange [m],
    lateral + right of the target line [m]. Non-finite (failed-run)
    points are excluded from both counts.
    """
    carries = np.asarray(carries_m, dtype=float)
    laterals = np.asarray(laterals_m, dtype=float)
    require(
        carries.shape == laterals.shape and carries.ndim == 1,
        "carry/lateral arrays must be matching 1-D",
    )
    valid = np.isfinite(carries) & np.isfinite(laterals)
    held = sum(
        region.contains(float(x), float(z))
        for x, z in zip(carries[valid], laterals[valid], strict=True)
    )
    return int(held), int(np.count_nonzero(valid))


def hold_fraction(dataset: VariationDataset, region: TargetRegion) -> float:
    """Fraction of a variation dataset's landings inside the region.

    The headline target output: over the dataset's successful runs, the
    share whose (carry, lateral) landing point the region contains.
    Returns NaN when the dataset has no successful landing outputs.
    """
    require(
        "carry_m" in dataset.output_names and "lateral_m" in dataset.output_names,
        "dataset must carry landing outputs (carry_m, lateral_m)",
    )
    held, total = hold_stats(
        dataset.output_column("carry_m"), dataset.output_column("lateral_m"), region
    )
    return held / total if total else float("nan")


def layout_for_region(
    region: TargetRegion, base: CourseLayout | None = None
) -> CourseLayout:
    """Course layout placing the H7a green where the target region sits.

    Green targets move the putting green (distance + radius); fairway
    targets widen the fairway strip to the corridor's half-width and
    keep the green at the corridor's far end for context.
    """
    base = base or CourseLayout()
    if region.kind == "green":
        return CourseLayout(
            green_distance_m=region.distance_m,
            green_radius_m=region.radius_m,
            fairway_half_width_m=base.fairway_half_width_m,
        )
    return CourseLayout(
        green_distance_m=region.distance_m + region.band_half_length_m,
        green_radius_m=base.green_radius_m,
        fairway_half_width_m=region.half_width_m,
    )

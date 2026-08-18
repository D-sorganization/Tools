"""Prevalidated text for one accepted putting result."""

from collections.abc import Callable

from shared.python.swing_sim.putting import PuttResult


def putting_result_values(
    result: PuttResult, format_m: Callable[[float], str]
) -> dict[str, str]:
    """Build every fallible scalar label before visual publication."""
    return {
        "putt_rollout_m": format_m(result.total_distance_m),
        "putt_skid_m": format_m(result.skid_distance_m),
        "putt_skid_pct": f"{100.0 * result.skid_fraction:.1f} %",
        "putt_time_s": f"{result.time_s:.2f} s",
        "putt_break_m": format_m(result.break_m),
        "putt_speed_at_hole_mps": (
            f"{result.speed_at_hole_mps:.2f} m/s"
            if result.speed_at_hole_mps is not None
            else "— (never reached)"
        ),
        "putt_margin": (
            f"HOLED (+{result.margin_mps:.2f} m/s under bound)"
            if result.holed and result.margin_mps is not None
            else (
                f"miss by {format_m(result.miss_distance_m)}"
                if result.miss_distance_m is not None
                else "—"
            )
        ),
    }

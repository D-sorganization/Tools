"""Prevalidated text for one accepted putting result.

Two independent projections, both built before anything is published
to a widget so a formatting failure can never leave half a row set:

* :func:`putting_result_values` — the #4125 H3 scalars carried by the
  integrated :class:`~shared.python.swing_sim.putting.PuttResult`.
* :func:`putting_document_values` — the fields that only exist in the
  ``swing_sim.putting_result/2`` record (#4800 P5): the start line off
  the face, the break-trajectory summary, and the geometric capture
  margin under the published effective-radius model, plus P3's
  quasi-static face twist at the strike.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from shared.python.swing_sim.putting import PuttResult

if TYPE_CHECKING:  # pragma: no cover - typing-only; both imports are lazy
    from shared.python.golf_club.putter_head import PutterTwist
    from shared.python.swing_sim.putting import PuttingResultDocument


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


def putting_document_values(
    document: PuttingResultDocument,
    twist: PutterTwist,
    format_m: Callable[[float], str],
) -> dict[str, str]:
    """Build the ``putting_result/2`` labels the P6 rows publish.

    Directions read in the plain-language sense the row explanations
    use: the record's signed degrees are right-positive off the target
    line, so the label spells out ``right``/``left`` and never leaves a
    bare sign to be misread. Lateral break keeps the record's own
    left-positive metre sign, matching the top-down view's axis.

    Capture geometry is deliberately **not** routed through the
    distance-unit chokepoint: the hole radius is 54 mm, so the session
    display unit (yards by default, two decimals) rounds the whole
    effective-rim question to ``0.06 yd`` and its margin to noise.
    These two are rim-scale quantities and read in millimetres, the
    same way speed reads in m/s and skid share reads in percent.
    """
    return {
        "putt_start_azimuth_deg": _direction(document.start_azimuth_deg),
        "putt_apex_break_m": (
            f"{format_m(document.apex_break_m)} at {format_m(document.apex_break_at_m)}"
        ),
        "putt_entry_azimuth_deg": _direction(document.entry_azimuth_deg),
        "putt_capture_margin_m": (
            f"{1000.0 * document.capture_margin_m:+.1f} mm of "
            f"{1000.0 * document.effective_hole_radius_m:.1f} mm effective radius"
        ),
        "putt_face_twist_deg": _twist(twist.face_twist_open_deg),
    }


def _twist(face_twist_open_deg: float) -> str:
    """Quasi-static face rotation, with a centred strike named as such."""
    if face_twist_open_deg == 0.0:
        return "0.000° (centred strike)"
    side = "open" if face_twist_open_deg > 0.0 else "closed"
    return f"{abs(face_twist_open_deg):.3f}° {side}"


def _direction(angle_deg: float) -> str:
    """A right-positive angle as magnitude plus the named side."""
    if angle_deg == 0.0:
        return "0.00° (on the target line)"
    side = "right" if angle_deg > 0.0 else "left"
    return f"{abs(angle_deg):.2f}° {side}"


__all__ = ["putting_document_values", "putting_result_values"]

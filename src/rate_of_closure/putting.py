"""Putting bridge: library putters + putt explanations (#4125 H3).

The physics lives in ``shared.python.swing_sim.putting`` (self-façaded
subpackage); this module adapts it to the app: it maps the H1 club-
library putters onto :class:`~shared.python.swing_sim.putting.PutterSpec`
(falling back to the module's minimal specs when the library carries no
putter), and owns the click-through explanations for the Putting tab's
result rows in both UIs.
"""

from __future__ import annotations

from rate_of_closure.club import CLUB_LIBRARY, ClubType
from shared.python.swing_sim.putting import (
    DEFAULT_PUTTER_COR,
    MINIMAL_PUTTERS,
    PutterSpec,
)

__all__ = ["PUTT_EXPLANATIONS", "putter_specs"]


def putter_specs() -> dict[str, PutterSpec]:
    """Selectable putters for the Putting tab.

    Prefers the H1 club-library putters (``rate_of_closure.club``),
    converted with the typical published putter-face COR; falls back
    to the swing_sim minimal specs when the library carries none
    (H1 reconciliation note in ``swing_sim.putting.impact``).

    Returns:
        Ordered display-name -> spec mapping, never empty.
    """
    library = {
        spec.name: PutterSpec(
            name=spec.name,
            head_mass_kg=spec.head_mass_kg,
            loft_deg=spec.loft_deg,
            cor=DEFAULT_PUTTER_COR,
        )
        for spec in CLUB_LIBRARY.values()
        if spec.club_type is ClubType.PUTTER
    }
    return library or dict(MINIMAL_PUTTERS)


#: Click-through text behind every Putting result row (both UIs).
PUTT_EXPLANATIONS: dict[str, str] = {
    "putt_rollout_m": (
        "How far the ball travels before stopping (or dropping). The "
        "skid phase sheds speed at the sliding-friction rate, then "
        "pure roll decelerates at the stimp-derived rolling rate — "
        "faster greens (higher stimp) mean a lower rolling coefficient "
        "and a longer roll-out for the same pace."
    ),
    "putt_skid_m": (
        "Ground covered while the ball is still sliding rather than "
        "rolling. A struck putt leaves the face with backspin, so "
        "friction must first spin it up to pure roll; the transition "
        "happens where ball speed equals surface spin speed (v = ωr)."
    ),
    "putt_skid_pct": (
        "The skid distance as a share of the whole putt. Good strokes "
        "keep this small — the classic no-spin result is that a "
        "sliding ball reaches pure roll at 5/7 of its launch speed, "
        "and more backspin extends the skid."
    ),
    "putt_time_s": (
        "Elapsed time from impact until the ball stops or drops. "
        "Rolling deceleration is constant on a uniform green, so time "
        "grows linearly with the speed the roll phase starts at."
    ),
    "putt_break_m": (
        "Lateral drift of the ball off the starting line (positive = "
        "left), caused by the in-plane component of gravity on the "
        "sloped green. Break grows fastest late in the putt, when the "
        "ball is slow and gravity has proportionally more say."
    ),
    "putt_speed_at_hole_mps": (
        "Ball speed when it first crosses the hole mouth. The putt "
        "drops only if this is at or below the geometric capture "
        "bound — a ball must fall half its diameter while crossing "
        "the opening, which caps the speed the lip can swallow."
    ),
    "putt_margin": (
        "Holed putts: how far under the capture-speed bound the ball "
        "crossed the hole (bigger = more comfortable drop). Missed "
        "putts: the distance from the ball's resting place back to "
        "the hole — the length of the comebacker."
    ),
}

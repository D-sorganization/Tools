"""Putting bridge: library putters + putt explanations (#4125 H3).

The physics lives in ``shared.python.swing_sim.putting`` (self-façaded
subpackage); this module adapts it to the app: it maps the H1 club-
library putters onto :class:`~shared.python.swing_sim.putting.PutterSpec`
(falling back to the module's minimal specs when the library carries no
putter), and owns the click-through explanations for the Putting tab's
result rows in both UIs.

Epic #4800 P6 adds two more binding-only adapters, both of which only
select and forward the shared authorities — no physics and no wire
parsing lives here:

* :func:`putter_head_documents` lifts the same library putters to the
  P3 ``golf_club.putter_head/1`` v2 documents (the *library* fallback
  provenance), which is what the Qt tab feeds to
  :func:`~shared.python.golf_club.putter_head.strike_with_head`; a
  mesh-derived head replaces one of these at the user's request.
* :func:`green_surface_from_document` dispatches an imported green
  between the two shared, versioned readers on the *declared* format
  field — never on shape. ``swing_sim.green_surface/1`` documents
  carry ``format``; UpstreamDrift ``_surface_io`` topographies refuse
  unknown top-level fields and therefore never do, so the dispatch is
  exact and fail-closed in both directions.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rate_of_closure.club import CLUB_LIBRARY, ClubType
from shared.python.swing_sim.putting import (
    DEFAULT_PUTTER_COR,
    GREEN_SURFACE_FORMAT,
    MINIMAL_PUTTERS,
    GreenSurface,
    PutterSpec,
    green_surface_from_json,
    green_surface_from_ud_json,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only, keeps SciPy out of import
    from shared.python.golf_club.putter_head import PutterHeadDocument

__all__ = [
    "PUTT_EXPLANATIONS",
    "green_surface_from_document",
    "putter_head_documents",
    "putter_specs",
]


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


def putter_head_documents() -> dict[str, PutterHeadDocument]:
    """Selectable putters as P3 ``PutterHeadDocument`` values (#4800 P6).

    The same putters :func:`putter_specs` offers, lifted to the v2
    document through
    :func:`~shared.python.golf_club.putter_head.putter_head_from_library`
    — the documented no-mesh fallback, which carries no inertia tensor
    and therefore reproduces P1's catalogue-default strike exactly. A
    mesh-derived head (``putter_head_from_stl``) is a drop-in
    replacement for any entry.

    ``shared.python.golf_club`` is imported lazily: it reaches SciPy
    through the fitting chain, and the Morris UI import contract
    forbids paying that at module scope (see the package handoff).

    Returns:
        Ordered display-name -> ``PutterHeadDocument`` mapping, never
        empty.
    """
    from shared.python.golf_club.putter_head import putter_head_from_library

    return {
        name: putter_head_from_library(
            name,
            head_mass_kg=spec.head_mass_kg,
            loft_deg=spec.loft_deg,
            cor=spec.cor,
        )
        for name, spec in putter_specs().items()
    }


def green_surface_from_document(text: str) -> tuple[GreenSurface, str]:
    """Read one imported green through the declared-format reader.

    Dispatch is on the ``format`` field alone (see the module
    docstring): present means the Tools ``swing_sim.green_surface/1``
    wire, absent means an UpstreamDrift ``_surface_io`` topography
    (#4800 P9's adapter). Neither reader is relaxed — both stay
    fail-closed on unknown fields — and no shape sniffing happens
    here.

    Args:
        text: The JSON document as read from disk.

    Returns:
        ``(surface, provenance_label)``; the label names the wire the
        surface actually came through, for the displayed authority.

    Raises:
        ValueError: If the text is not JSON, is not an object, or the
            selected reader refuses it.
        TypeError: If a field has the wrong type.
    """
    if not isinstance(text, str):
        raise TypeError("green document text must be str")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("green document must be a JSON object")
    if "format" in data:
        return (green_surface_from_json(text), GREEN_SURFACE_FORMAT)
    return (
        green_surface_from_ud_json(text).surface,
        "upstreamdrift.putting_green topography",
    )


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
    # ── swing_sim.putting_result/2 fields (#4800 P1/P2/P5, shown by P6)
    "putt_start_azimuth_deg": (
        "The direction the ball actually leaves the face, measured "
        "off the target line (positive = right). It is your aim plus "
        "the face angle plus a small pull toward the path: the normal "
        "impulse launches the ball along the face while the 2/7 "
        "tangential impulse drags it toward the stroke path, so face "
        "angle dominates the start line and path only trims it."
    ),
    "putt_apex_break_m": (
        "The widest the ball ever gets from the target line, and how "
        "far down the line that happens (positive = left). On a "
        "cross-slope the ball keeps turning while it slows, so the "
        "apex is the high point of the read — the aim point you pick "
        "has to cover this excursion, not just the finishing break."
    ),
    "putt_entry_azimuth_deg": (
        "The direction the ball is travelling at its closest approach "
        "to the hole, off the target line (positive = right). A putt "
        "still turning hard as it arrives presents less of the hole "
        "to fall into than one that has straightened out, which is "
        "why entry angle and speed together decide a lip-out."
    ),
    "putt_capture_margin_m": (
        "How much hole was to spare, geometrically: the effective "
        "hole radius at the arrival speed minus the closest approach. "
        "The faster the ball crosses, the less of the 54 mm radius it "
        "can use, so a positive margin means the ball passed inside "
        "the shrunken opening and a negative one says how much more "
        "hole the putt needed."
    ),
    "putt_face_twist_deg": (
        "How far the putter face rotates open during the half of "
        "contact after an off-centre strike, from the head's own "
        "moment of inertia (positive = opens, a toe strike). It is "
        "the forgiveness number: a higher-MOI head twists less, so it "
        "loses less ball speed and less start line on the same miss."
    ),
}

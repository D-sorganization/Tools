"""Glossary of every technical term used across the app (#4120 V4).

Single source of truth for the Glossary tab (PyQt6) and the Glossary
section of the web clone (``web/src/model/glossary.ts`` mirrors this
dict key-for-key; the vitest parity test pins the key list). Every
explanation panel links here, and fields that map onto one term
pre-select it via :data:`FIELD_TO_TERM`.

Definitions are 1-3 sentences each and carry their source inline —
the AffineDrift Launch Monitor Technology Review, the Cheetham 2014
closure-rate dossier, ``swing_sim`` module derivations (impact /
flight / variation), and the standard golf-physics literature
(Jorgensen, Penner, TrackMan D-plane material). Entry data lives in
the split ``glossary_entries_*`` modules (500-LOC file budget).
"""

from __future__ import annotations

from ._contracts import ensure
from .glossary_entries_a_l import ENTRIES as _ENTRIES_A_L
from .glossary_entries_l_z import ENTRIES as _ENTRIES_L_Z
from .glossary_entries_putting import ENTRIES as _ENTRIES_PUTTING
from .glossary_types import GlossaryEntry

__all__ = ["FIELD_TO_TERM", "GLOSSARY", "GlossaryEntry", "search_terms"]

#: Every term used across the app, keyed snake_case. Sorted by key.
GLOSSARY: dict[str, GlossaryEntry] = dict(
    sorted((_ENTRIES_A_L | _ENTRIES_L_Z | _ENTRIES_PUTTING).items())
)


#: Explanation field -> the glossary term it pre-selects, for every
#: field that maps cleanly onto one term (contract-tested).
FIELD_TO_TERM: dict[str, str] = {
    # RESULT_EXPLANATIONS fields
    "path_deviation_deg": "club_path",
    "aoa_deviation_deg": "attack_angle",
    "tangential_speed_mph": "twist",
    "speed_delta_mph": "twist",
    "closure_rate_dps": "ccv",
    "normalized_closure_deg_per_ft": "r_isa",
    "closure_during_contact_deg": "contact_duration",
    "loft_gain_during_contact_deg": "dynamic_loft",
    # METRIC_EXPLANATIONS fields
    "ccv_dps": "ccv",
    "closure_deg_per_ft": "r_isa",
    "closure_deg_per_inch": "closure_rate",
    "closure_deg_per_ms": "closure_rate",
    "r_isa_m": "r_isa",
    "r_isa_ft": "r_isa",
    "time_to_square_from_1deg_open_ms": "time_to_square",
    "toe_heel_speed_delta_mph": "lever_arm",
    # LAUNCH_EXPLANATIONS fields
    "ball_speed_mph": "smash_factor",
    "launch_angle_deg": "launch_angle",
    "launch_azimuth_deg": "launch_azimuth",
    "spin_rpm": "spin_rate",
    "carry_m": "carry",
    "max_height_m": "apex",
    "flight_time_s": "flight_time",
    "landing_angle_deg": "landing_angle",
    "lateral_m": "lateral_offset",
    # KINETICS_EXPLANATIONS fields (#4125 H2)
    "joint_torques": "inverse_dynamics",
    "joint_power": "power",
    "reaction_forces": "joint_reaction_force",
    "zero_torque_counterfactual": "zero_torque_counterfactual",
    # PUTT_EXPLANATIONS fields (#4125 H3)
    "putt_rollout_m": "stimp",
    "putt_skid_m": "skid",
    "putt_skid_pct": "skid",
    "putt_time_s": "pure_roll",
    "putt_break_m": "break",
    "putt_speed_at_hole_mps": "capture_speed",
    "putt_margin": "capture_speed",
}


def search_terms(query: str) -> tuple[str, ...]:
    """Glossary keys whose term or definition matches ``query``.

    Case-insensitive substring search over the display term and the
    definition body; an empty query returns every key.

    Args:
        query: Free-text filter.

    Returns:
        Matching keys in glossary (alphabetical) order.
    """
    needle = query.strip().lower()
    keys = tuple(
        key
        for key, entry in GLOSSARY.items()
        if not needle
        or needle in entry.term.lower()
        or needle in entry.definition.lower()
    )
    ensure(not needle or len(keys) <= len(GLOSSARY), "filter cannot grow the set")
    return keys


def _validate() -> None:
    keys = list(GLOSSARY)
    ensure(keys == sorted(keys), "glossary keys must be sorted")
    ensure(len(keys) >= 40, "glossary must cover the app's vocabulary")
    ensure(
        all(target in GLOSSARY for target in FIELD_TO_TERM.values()),
        "every field mapping must point at a real glossary term",
    )


_validate()

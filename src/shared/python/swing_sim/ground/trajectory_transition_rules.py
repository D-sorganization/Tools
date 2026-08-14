"""Legal phase and event transitions for the strict v1 ground ledger."""

from __future__ import annotations

from .contract_types import GroundEventType, GroundPhase

PHASE_TRANSITIONS = {
    GroundPhase.IMPACT: frozenset(GroundPhase),
    GroundPhase.BOUNCE: frozenset(
        {GroundPhase.BOUNCE, GroundPhase.SKID, GroundPhase.ROLL, GroundPhase.REST}
    ),
    GroundPhase.SKID: frozenset({GroundPhase.SKID, GroundPhase.ROLL, GroundPhase.REST}),
    GroundPhase.ROLL: frozenset({GroundPhase.ROLL, GroundPhase.REST}),
    GroundPhase.REST: frozenset({GroundPhase.REST}),
}

EVENT_TRANSITIONS = {
    GroundEventType.FIRST_CONTACT: frozenset(
        {
            GroundEventType.BOUNCE,
            GroundEventType.SKID_TO_ROLL,
            GroundEventType.SURFACE_TRANSITION,
            GroundEventType.REST,
            GroundEventType.LEFT_SURFACE,
        }
    ),
    GroundEventType.BOUNCE: frozenset(
        {
            GroundEventType.BOUNCE,
            GroundEventType.SKID_TO_ROLL,
            GroundEventType.SURFACE_TRANSITION,
            GroundEventType.REST,
            GroundEventType.LEFT_SURFACE,
        }
    ),
    GroundEventType.SKID_TO_ROLL: frozenset(
        {
            GroundEventType.SURFACE_TRANSITION,
            GroundEventType.REST,
            GroundEventType.LEFT_SURFACE,
        }
    ),
    GroundEventType.SURFACE_TRANSITION: frozenset(
        {
            GroundEventType.SURFACE_TRANSITION,
            GroundEventType.SKID_TO_ROLL,
            GroundEventType.REST,
            GroundEventType.LEFT_SURFACE,
        }
    ),
    GroundEventType.REST: frozenset(),
    GroundEventType.LEFT_SURFACE: frozenset(),
}

__all__ = ["EVENT_TRANSITIONS", "PHASE_TRANSITIONS"]

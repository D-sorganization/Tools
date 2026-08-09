"""Typed unavailable-input evidence for flight-to-ground v1."""

from __future__ import annotations

from dataclasses import dataclass

from shared.python.compatibility import StrEnum

from .contract_types import _text, _WireRecord


class GroundUnavailableFieldId(StrEnum):
    """Required handoff fields that may be unavailable upstream."""

    TERMINAL_ANGULAR_VELOCITY = "terminal_angular_velocity_rad_s"
    PHYSICAL_CONTACT_BRACKET = "physical_contact_bracket"
    SURFACE_PROFILE = "surface_profile"


class GroundUnavailableReason(StrEnum):
    """Typed causes for an unavailable physical handoff field."""

    SOURCE_DOES_NOT_PROPAGATE = "source_does_not_propagate"
    NO_PHYSICAL_CONTACT = "no_physical_contact"
    UNSUPPORTED_SURFACE = "unsupported_surface"
    SOURCE_OUT_OF_BOUNDS = "source_out_of_bounds"


@dataclass(frozen=True)
class GroundUnavailableField(_WireRecord):
    """Typed evidence explaining why one required handoff field is absent."""

    field_id: GroundUnavailableFieldId
    reason: GroundUnavailableReason
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_id", GroundUnavailableFieldId(self.field_id))
        object.__setattr__(self, "reason", GroundUnavailableReason(self.reason))
        object.__setattr__(
            self,
            "provenance",
            _text(self.provenance, "unavailable field provenance"),
        )


__all__ = [
    "GroundUnavailableField",
    "GroundUnavailableFieldId",
    "GroundUnavailableReason",
]

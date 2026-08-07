"""UI-neutral multi-view and plot-slot workspace contract (#4224/#4225)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar, cast

FORMAT = "rate_of_closure.view_workspace/1"
MAX_SLOTS = 6
EnumT = TypeVar("EnumT", bound=StrEnum)


class ViewKind(StrEnum):
    """Content rendered by one independently identifiable viewport."""

    IMPACT = "impact"
    SWING = "swing"
    KINETICS = "kinetics"
    FLIGHT = "flight"
    PLOT = "plot"


class ViewLayout(StrEnum):
    """Supported deterministic viewport arrangements."""

    SINGLE = "single"
    SPLIT_HORIZONTAL = "split_horizontal"
    SPLIT_VERTICAL = "split_vertical"
    GRID = "grid"


class LegendPlacement(StrEnum):
    """Legend visibility and placement owned by an individual slot."""

    HIDDEN = "hidden"
    OUTSIDE_RIGHT = "outside_right"
    INSIDE_UPPER_RIGHT = "inside_upper_right"
    INSIDE_LOWER_RIGHT = "inside_lower_right"
    INSIDE_LOWER_LEFT = "inside_lower_left"


@dataclass(frozen=True, slots=True)
class PlaybackState:
    """Playback state shared by every synchronized simulation viewport."""

    time_s: float = 0.0
    playing: bool = False
    loop: bool = False
    rate: float = 1.0

    def validate(self) -> None:
        """Raise when playback state cannot be applied deterministically."""
        if not isinstance(self.time_s, (int, float)) or isinstance(self.time_s, bool):
            raise ValueError("playback time_s must be numeric")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("playback time_s must be finite and non-negative")
        if not isinstance(self.rate, (int, float)) or isinstance(self.rate, bool):
            raise ValueError("playback rate must be numeric")
        if not math.isfinite(self.rate) or self.rate <= 0.0:
            raise ValueError("playback rate must be finite and positive")
        if not isinstance(self.playing, bool) or not isinstance(self.loop, bool):
            raise ValueError("playback playing and loop must be booleans")


@dataclass(frozen=True, slots=True)
class ViewSlot:
    """One viewport with stable identity and independent presentation state."""

    id: str
    kind: ViewKind
    plot_id: str | None = None
    legend: LegendPlacement = LegendPlacement.OUTSIDE_RIGHT

    def validate(self) -> None:
        """Raise when the slot violates identity or plot ownership rules."""
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("slot id must be a non-empty string")
        if self.kind is ViewKind.PLOT:
            if not isinstance(self.plot_id, str) or not self.plot_id.strip():
                raise ValueError("plot slots require a non-empty plot_id")
        elif self.plot_id is not None:
            raise ValueError("plot_id is valid only for plot slots")


@dataclass(frozen=True, slots=True)
class ViewWorkspace:
    """Serializable arrangement of synchronized, independently owned views."""

    layout: ViewLayout
    slots: tuple[ViewSlot, ...]
    active_slot_id: str
    playback: PlaybackState = field(default_factory=PlaybackState)

    @classmethod
    def default(cls) -> ViewWorkspace:
        """Return the stable first-run Swing workspace."""
        return cls(
            layout=ViewLayout.SINGLE,
            slots=(ViewSlot(id="swing", kind=ViewKind.SWING),),
            active_slot_id="swing",
        )

    def validate(self) -> None:
        """Raise when layout, identity, or playback invariants are broken."""
        slot_count = len(self.slots)
        if not 1 <= slot_count <= MAX_SLOTS:
            raise ValueError(f"workspace must contain 1 to {MAX_SLOTS} slots")
        required_count = 1 if self.layout is ViewLayout.SINGLE else 2
        if self.layout is not ViewLayout.GRID and slot_count != required_count:
            raise ValueError(
                f"{self.layout.value} layout requires {required_count} slots"
            )
        if self.layout is ViewLayout.GRID and slot_count < 2:
            raise ValueError("grid layout requires at least 2 slots")
        for slot in self.slots:
            slot.validate()
        slot_ids = tuple(slot.id for slot in self.slots)
        if len(set(slot_ids)) != slot_count:
            raise ValueError("workspace slot ids must be unique")
        if self.active_slot_id not in slot_ids:
            raise ValueError("active_slot_id must identify a workspace slot")
        self.playback.validate()


def workspace_to_document(workspace: ViewWorkspace) -> dict[str, object]:
    """Return the canonical JSON-safe version-1 workspace document."""
    workspace.validate()
    slots = [
        {
            "id": slot.id,
            "kind": slot.kind.value,
            "plot_id": slot.plot_id,
            "legend": slot.legend.value,
        }
        for slot in workspace.slots
    ]
    playback = workspace.playback
    return {
        "format": FORMAT,
        "layout": workspace.layout.value,
        "slots": slots,
        "active_slot_id": workspace.active_slot_id,
        "playback": {
            "time_s": playback.time_s,
            "playing": playback.playing,
            "loop": playback.loop,
            "rate": playback.rate,
        },
    }


def workspace_from_document(document: Mapping[str, object]) -> ViewWorkspace:
    """Parse a strict version-1 workspace document without partial mutation."""
    _require_keys(
        document,
        {"format", "layout", "slots", "active_slot_id", "playback"},
        "workspace",
    )
    if document["format"] != FORMAT:
        raise ValueError(f"unsupported workspace format: {document['format']!r}")
    slots_raw = document["slots"]
    if not isinstance(slots_raw, list):
        raise ValueError("workspace slots must be a list")
    slots = tuple(_slot_from_document(item) for item in slots_raw)
    playback_raw = _mapping(document["playback"], "playback")
    _require_keys(playback_raw, {"time_s", "playing", "loop", "rate"}, "playback")
    workspace = ViewWorkspace(
        layout=_enum_value(ViewLayout, document["layout"], "layout"),
        slots=slots,
        active_slot_id=_string(document["active_slot_id"], "active_slot_id"),
        playback=PlaybackState(
            time_s=_number(playback_raw["time_s"], "playback.time_s"),
            playing=_boolean(playback_raw["playing"], "playback.playing"),
            loop=_boolean(playback_raw["loop"], "playback.loop"),
            rate=_number(playback_raw["rate"], "playback.rate"),
        ),
    )
    workspace.validate()
    return workspace


def _slot_from_document(value: object) -> ViewSlot:
    raw = _mapping(value, "slot")
    _require_keys(raw, {"id", "kind", "plot_id", "legend"}, "slot")
    plot_id_raw = raw["plot_id"]
    plot_id = None if plot_id_raw is None else _string(plot_id_raw, "slot.plot_id")
    slot = ViewSlot(
        id=_string(raw["id"], "slot.id"),
        kind=_enum_value(ViewKind, raw["kind"], "slot.kind"),
        plot_id=plot_id,
        legend=_enum_value(LegendPlacement, raw["legend"], "slot.legend"),
    )
    slot.validate()
    return slot


def _require_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    actual = set(value)
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"{context} has unexpected fields: {sorted(unexpected)}")


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    return cast(Mapping[str, object], value)


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _number(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _enum_value(enum_type: type[EnumT], value: object, field_name: str) -> EnumT:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unsupported {field_name}: {value!r}") from exc

"""Defensive migration for the Rate multi-view workspace document."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rate_of_closure.view_workspace import (
    LegendPlacement,
    PlaybackState,
    ViewKind,
    ViewLayout,
    ViewSlot,
    ViewWorkspace,
)

SUPPORTED_VIEW_KINDS = (ViewKind.IMPACT, ViewKind.SWING, ViewKind.FLIGHT)


def recover_workspace_document(document: object) -> ViewWorkspace:
    """Migrate a saved layout, dropping unknown IDs without losing known views."""
    if not isinstance(document, Mapping):
        return ViewWorkspace.default()
    slots = _recover_slots(document)
    if not slots:
        return ViewWorkspace.default()
    identifiers = tuple(slot.id for slot in slots)
    active_raw = document.get("active_slot_id", document.get("active"))
    active = active_raw if active_raw in identifiers else identifiers[0]
    return ViewWorkspace(
        layout=_recover_layout(document.get("layout"), len(slots)),
        slots=slots,
        active_slot_id=str(active),
        playback=_recover_playback(document.get("playback")),
    )


def _recover_slots(document: Mapping[object, object]) -> tuple[ViewSlot, ...]:
    raw_slots = document.get("slots")
    if isinstance(raw_slots, Sequence) and not isinstance(raw_slots, (str, bytes)):
        candidates = (_slot(item) for item in raw_slots)
    else:
        raw_views = document.get("views")
        views = (
            raw_views
            if isinstance(raw_views, Sequence)
            and not isinstance(raw_views, (str, bytes))
            else ()
        )
        candidates = (_slot({"id": item, "kind": item}) for item in views)
    unique: dict[str, ViewSlot] = {}
    for candidate in candidates:
        if candidate is not None and candidate.id not in unique:
            unique[candidate.id] = candidate
    return tuple(unique.values())


def _slot(value: object) -> ViewSlot | None:
    if not isinstance(value, Mapping):
        return None
    identifier = value.get("id")
    kind_raw = value.get("kind")
    if not isinstance(identifier, str) or identifier != kind_raw:
        return None
    try:
        kind = ViewKind(identifier)
    except ValueError:
        return None
    if kind not in SUPPORTED_VIEW_KINDS:
        return None
    legend_raw = value.get("legend", LegendPlacement.OUTSIDE_RIGHT.value)
    try:
        legend = LegendPlacement(legend_raw)
    except (TypeError, ValueError):
        legend = LegendPlacement.OUTSIDE_RIGHT
    return ViewSlot(id=identifier, kind=kind, legend=legend)


def _recover_layout(value: object, slot_count: int) -> ViewLayout:
    if slot_count == 1:
        return ViewLayout.SINGLE
    try:
        layout = ViewLayout(value)
    except (TypeError, ValueError):
        layout = ViewLayout.SPLIT_HORIZONTAL if slot_count == 2 else ViewLayout.GRID
    if layout is ViewLayout.SINGLE:
        return ViewLayout.SPLIT_HORIZONTAL if slot_count == 2 else ViewLayout.GRID
    return layout


def _recover_playback(value: object) -> PlaybackState:
    if not isinstance(value, Mapping):
        return PlaybackState()
    try:
        playback = PlaybackState(
            time_s=float(value.get("time_s", 0.0)),
            playing=value.get("playing", False),
            loop=value.get("loop", False),
            rate=float(value.get("rate", 1.0)),
        )
        playback.validate()
    except (TypeError, ValueError, OverflowError):
        return PlaybackState()
    if not math.isfinite(playback.time_s):
        return PlaybackState()
    return playback


__all__ = ["SUPPORTED_VIEW_KINDS", "recover_workspace_document"]

"""Contracts for simultaneous simulation and plot workspace layouts (#4224/#4225)."""

from __future__ import annotations

import pytest

from rate_of_closure.view_workspace import (
    LegendPlacement,
    PlaybackState,
    ViewKind,
    ViewLayout,
    ViewSlot,
    ViewWorkspace,
    workspace_from_document,
    workspace_to_document,
)
from rate_of_closure.view_workspace_recovery import recover_workspace_document


def test_default_workspace_is_a_single_swing_view() -> None:
    workspace = ViewWorkspace.default()

    assert workspace.layout is ViewLayout.SINGLE
    assert workspace.active_slot_id == "swing"
    assert workspace.slots == (ViewSlot(id="swing", kind=ViewKind.SWING),)
    assert workspace.playback == PlaybackState()


def test_multi_view_slots_share_playback_and_keep_independent_legends() -> None:
    workspace = ViewWorkspace(
        layout=ViewLayout.GRID,
        slots=(
            ViewSlot(id="impact", kind=ViewKind.IMPACT),
            ViewSlot(id="swing", kind=ViewKind.SWING),
            ViewSlot(id="flight", kind=ViewKind.FLIGHT),
            ViewSlot(
                id="plot-a",
                kind=ViewKind.PLOT,
                plot_id="closure-sweep",
                legend=LegendPlacement.OUTSIDE_RIGHT,
            ),
            ViewSlot(
                id="plot-b",
                kind=ViewKind.PLOT,
                plot_id="closure-sweep",
                legend=LegendPlacement.HIDDEN,
            ),
        ),
        active_slot_id="swing",
        playback=PlaybackState(time_s=0.71, playing=True, loop=True, rate=0.5),
    )

    assert workspace.slots[3].plot_id == workspace.slots[4].plot_id
    assert workspace.slots[3].legend is LegendPlacement.OUTSIDE_RIGHT
    assert workspace.slots[4].legend is LegendPlacement.HIDDEN
    assert workspace.playback.time_s == pytest.approx(0.71)


def test_workspace_document_round_trip_is_strict_and_versioned() -> None:
    workspace = ViewWorkspace(
        layout=ViewLayout.SPLIT_HORIZONTAL,
        slots=(
            ViewSlot(id="impact", kind=ViewKind.IMPACT),
            ViewSlot(id="flight", kind=ViewKind.FLIGHT),
        ),
        active_slot_id="impact",
        playback=PlaybackState(time_s=0.2, rate=2.0),
    )

    document = workspace_to_document(workspace)

    assert document["format"] == "rate_of_closure.view_workspace/1"
    assert workspace_from_document(document) == workspace


@pytest.mark.parametrize(
    "workspace",
    [
        ViewWorkspace(
            layout=ViewLayout.SINGLE,
            slots=(ViewSlot(id="a", kind=ViewKind.SWING),),
            active_slot_id="missing",
        ),
        ViewWorkspace(
            layout=ViewLayout.GRID,
            slots=(
                ViewSlot(id="same", kind=ViewKind.SWING),
                ViewSlot(id="same", kind=ViewKind.FLIGHT),
            ),
            active_slot_id="same",
        ),
        ViewWorkspace(
            layout=ViewLayout.SINGLE,
            slots=(
                ViewSlot(id="a", kind=ViewKind.SWING),
                ViewSlot(id="b", kind=ViewKind.FLIGHT),
            ),
            active_slot_id="a",
        ),
    ],
)
def test_workspace_rejects_inconsistent_layouts(workspace: ViewWorkspace) -> None:
    with pytest.raises(ValueError):
        workspace.validate()


def test_only_plot_slots_accept_plot_ids() -> None:
    with pytest.raises(ValueError, match="plot_id"):
        ViewSlot(id="flight", kind=ViewKind.FLIGHT, plot_id="trajectory").validate()
    with pytest.raises(ValueError, match="plot_id"):
        ViewSlot(id="plot", kind=ViewKind.PLOT).validate()


@pytest.mark.parametrize(
    "playback",
    [
        PlaybackState(time_s=float("nan")),
        PlaybackState(time_s=float("inf")),
        PlaybackState(rate=float("nan")),
        PlaybackState(rate=float("inf")),
    ],
)
def test_playback_rejects_non_finite_values(playback: PlaybackState) -> None:
    with pytest.raises(ValueError, match="finite"):
        playback.validate()


def test_document_rejects_unknown_or_incomplete_fields() -> None:
    valid = workspace_to_document(ViewWorkspace.default())
    with pytest.raises(ValueError, match="unsupported"):
        workspace_from_document({**valid, "format": "rate_of_closure.view_workspace/9"})
    with pytest.raises(ValueError, match="unexpected"):
        workspace_from_document({**valid, "mystery": True})
    invalid_slot = {**valid, "slots": [{"id": "swing", "kind": "swing"}]}
    with pytest.raises(ValueError, match="missing"):
        workspace_from_document(invalid_slot)


def test_recovery_drops_unknown_view_ids_and_preserves_valid_playback() -> None:
    document = workspace_to_document(
        ViewWorkspace(
            layout=ViewLayout.GRID,
            slots=(
                ViewSlot(id="future", kind=ViewKind.PLOT, plot_id="future"),
                ViewSlot(id="swing", kind=ViewKind.SWING),
                ViewSlot(id="flight", kind=ViewKind.FLIGHT),
            ),
            active_slot_id="future",
            playback=PlaybackState(time_s=0.42, loop=True, rate=0.5),
        )
    )

    recovered = recover_workspace_document(document)

    assert [slot.id for slot in recovered.slots] == ["swing", "flight"]
    assert recovered.active_slot_id == "swing"
    assert recovered.layout is ViewLayout.GRID
    assert recovered.playback == PlaybackState(time_s=0.42, loop=True, rate=0.5)


def test_recovery_migrates_legacy_visible_views_with_safe_fallback() -> None:
    recovered = recover_workspace_document(
        {
            "version": 1,
            "layout": "split_horizontal",
            "views": ["impact", "future", "flight"],
            "active": "future",
        }
    )

    assert [slot.id for slot in recovered.slots] == ["impact", "flight"]
    assert recovered.active_slot_id == "impact"
    assert recovered.layout is ViewLayout.SPLIT_HORIZONTAL

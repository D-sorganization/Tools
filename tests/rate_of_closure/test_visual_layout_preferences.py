"""Presentation-only visual layout persistence authority."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rate_of_closure.club_camera import ClubCamera
from rate_of_closure.visual_layout_preferences import (
    DEFAULT_VISUAL_LAYOUT,
    MAX_SIDEBAR_FRACTION,
    MIN_SIDEBAR_FRACTION,
    VISUAL_LAYOUT_STATE_KEY,
    VisualLayoutPreferences,
    load_visual_layout,
    parse_visual_layout,
    save_visual_layout,
    visual_layout_document,
)


class MemorySettings:
    def __init__(self, value: object = None, *, fail_write: bool = False) -> None:
        self.values = {VISUAL_LAYOUT_STATE_KEY: value}
        self.fail_write = fail_write

    def value(self, key: str) -> object:
        return self.values.get(key)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        if self.fail_write:
            raise OSError("read-only settings")
        self.values[key] = value


def test_visual_layout_round_trips_exact_presentation_state() -> None:
    expected = VisualLayoutPreferences(ClubCamera(-35.0, 42.0, 2.5), True, 0.31)
    settings = MemorySettings()

    assert save_visual_layout(settings, expected)
    assert load_visual_layout(settings) == expected
    assert parse_visual_layout(visual_layout_document(expected)) == expected


@pytest.mark.parametrize(
    ("path", "forged"),
    [
        (("version",), 2),
        (("clubCamera", "azimuthDeg"), 180.0),
        (("clubCamera", "elevationDeg"), 80.1),
        (("clubCamera", "zoom"), 4.1),
        (("moduleHelpOpen",), 1),
        (("shellSidebarFraction",), MIN_SIDEBAR_FRACTION - 0.01),
        (("shellSidebarFraction",), MAX_SIDEBAR_FRACTION + 0.01),
    ],
)
def test_visual_layout_rejects_out_of_contract_values(
    path: tuple[str, ...], forged: object
) -> None:
    document = visual_layout_document(DEFAULT_VISUAL_LAYOUT)
    target = document
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = forged

    with pytest.raises(ValueError):
        parse_visual_layout(document)


def test_corrupt_or_nonfinite_storage_fails_closed_to_defaults() -> None:
    assert load_visual_layout(MemorySettings("not-json")) == DEFAULT_VISUAL_LAYOUT
    document = visual_layout_document(DEFAULT_VISUAL_LAYOUT)
    camera = document["clubCamera"]
    assert isinstance(camera, dict)
    camera["zoom"] = float("nan")
    settings = MemorySettings(json.dumps(document))
    assert load_visual_layout(settings) == DEFAULT_VISUAL_LAYOUT


def test_storage_failure_is_bounded_and_does_not_mutate_preferences() -> None:
    expected = replace(DEFAULT_VISUAL_LAYOUT, module_help_open=True)
    settings = MemorySettings(fail_write=True)

    assert not save_visual_layout(settings, expected)
    assert expected.module_help_open

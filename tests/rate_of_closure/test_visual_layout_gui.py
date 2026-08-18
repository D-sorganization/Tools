"""Real PyQt restoration of bounded presentation-only layout choices."""

from __future__ import annotations

import json

import pytest
from PyQt6.QtCore import QSettings

from rate_of_closure.club_camera import DEFAULT_CLUB_CAMERA, ClubCamera
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow
from rate_of_closure.visual_layout_preferences import (
    DEFAULT_VISUAL_LAYOUT,
    VISUAL_LAYOUT_STATE_KEY,
    visual_layout_document,
)


def _settings(path: object) -> QSettings:
    return QSettings(str(path), QSettings.Format.IniFormat)


def test_camera_and_bounded_shell_split_restore_across_windows(qtbot, tmp_path) -> None:
    path = tmp_path / "layout.ini"
    first_settings = _settings(path)
    first = RateOfClosureMainWindow(navigation_settings=first_settings)
    qtbot.addWidget(first)
    first.resize(1200, 800)
    first.show()
    qtbot.waitExposed(first)
    camera = ClubCamera(-40.0, 35.0, 2.25)
    first._club_view.set_camera(camera)
    first._shell_splitter.moveSplitter(360, 1)
    first_sizes = first._shell_splitter.sizes()
    first_fraction = first_sizes[0] / sum(first_sizes)
    first_settings.sync()
    assert first._club_view.camera() == camera
    assert first._tabs.width() >= 640
    first.close()

    second = RateOfClosureMainWindow(navigation_settings=_settings(path))
    qtbot.addWidget(second)
    second.resize(1200, 800)
    second.show()
    qtbot.waitExposed(second)
    qtbot.wait(50)

    assert second._club_view.camera() == camera
    sizes = second._shell_splitter.sizes()
    assert sizes[0] / sum(sizes) == pytest.approx(first_fraction, abs=0.01)
    assert second._tabs.width() >= 640


def test_corrupt_or_out_of_range_layout_restores_exact_defaults(
    qtbot, tmp_path
) -> None:
    settings = _settings(tmp_path / "corrupt.ini")
    document = visual_layout_document(DEFAULT_VISUAL_LAYOUT)
    document["shellSidebarFraction"] = 0.95
    settings.setValue(VISUAL_LAYOUT_STATE_KEY, json.dumps(document))
    window = RateOfClosureMainWindow(navigation_settings=settings)
    qtbot.addWidget(window)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(50)

    assert window._club_view.camera() == DEFAULT_CLUB_CAMERA
    sizes = window._shell_splitter.sizes()
    assert sizes[0] / sum(sizes) == pytest.approx(0.27, abs=0.02)
    assert window._tabs.width() >= 640


class FailingSettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, _key: str) -> object:
        return self.values.get(_key)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802
        if key == VISUAL_LAYOUT_STATE_KEY:
            raise OSError("visual layout is read-only")
        self.values[key] = value


def test_settings_write_failure_never_breaks_camera_interaction(qtbot) -> None:
    window = RateOfClosureMainWindow(navigation_settings=FailingSettings())
    qtbot.addWidget(window)
    candidate = ClubCamera(25.0, -10.0, 1.4)

    window._club_view.set_camera(candidate)

    assert window._club_view.camera() == candidate

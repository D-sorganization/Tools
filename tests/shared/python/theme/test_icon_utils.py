"""Focused coverage for shared theme icon utilities."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


class FakeQByteArray(bytes):
    pass


class FakeQPixmap:
    def __init__(self) -> None:
        self.loaded_data: bytes | None = None

    def loadFromData(self, data: bytes) -> bool:  # noqa: N802
        self.loaded_data = bytes(data)
        return True


class FakeQIcon:
    def __init__(self, pixmap: FakeQPixmap) -> None:
        self.pixmap = pixmap


@pytest.fixture
def icon_utils(monkeypatch):
    qtcore = SimpleNamespace(QByteArray=FakeQByteArray)
    qtgui = SimpleNamespace(QIcon=FakeQIcon, QPixmap=FakeQPixmap)
    monkeypatch.setitem(
        sys.modules,
        "PyQt6",
        SimpleNamespace(QtCore=qtcore, QtGui=qtgui),
    )
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", qtcore)
    monkeypatch.setitem(sys.modules, "PyQt6.QtGui", qtgui)
    sys.modules.pop("src.shared.python.theme.icon_utils", None)
    return importlib.import_module("src.shared.python.theme.icon_utils")


def test_get_icon_recolors_registered_svg(icon_utils):
    icon = icon_utils.IconColorizer.get_icon("home", "#123456")

    assert isinstance(icon, FakeQIcon)
    assert b'stroke="#123456"' in icon.pixmap.loaded_data
    assert b"{color}" not in icon.pixmap.loaded_data


def test_get_icon_rejects_unknown_icon_name(icon_utils):
    with pytest.raises(ValueError, match="not registered"):
        icon_utils.IconColorizer.get_icon("missing", "#fff")


def test_validate_icon_name(icon_utils):
    assert icon_utils.validate_icon_name("home") == "home"
    with pytest.raises(TypeError, match="name must be a string"):
        icon_utils.validate_icon_name(None)
    with pytest.raises(TypeError, match="name must be a string"):
        icon_utils.validate_icon_name(123)


def test_validate_icon_color(icon_utils):
    assert icon_utils.validate_icon_color("#ff0000") == "#ff0000"
    with pytest.raises(TypeError, match="color must be a string"):
        icon_utils.validate_icon_color(None)
    with pytest.raises(TypeError, match="color must be a string"):
        icon_utils.validate_icon_color(42)


def test_get_registered_svg(icon_utils):
    reg = {"test": "<svg>{color}</svg>"}
    assert icon_utils.get_registered_svg(reg, "test") == "<svg>{color}</svg>"
    with pytest.raises(TypeError, match="name must be a string"):
        icon_utils.get_registered_svg(reg, None)
    with pytest.raises(ValueError, match="Icon 'missing' is not registered"):
        icon_utils.get_registered_svg(reg, "missing")


def test_get_icon_validates_argument_types(icon_utils):
    with pytest.raises(TypeError, match="name must be a string"):
        icon_utils.IconColorizer.get_icon(None, "#fff")
    with pytest.raises(TypeError, match="color must be a string"):
        icon_utils.IconColorizer.get_icon("home", None)


def test_colorize_svg_file_rewrites_fill_and_stroke(icon_utils, tmp_path):
    svg_path = tmp_path / "glyph.svg"
    svg_path.write_text(
        '<svg><path fill="red" stroke="blue" d="M0 0"/></svg>',
        encoding="utf-8",
    )

    icon = icon_utils.IconColorizer.colorize_svg_file(svg_path, "currentColor")

    assert b'fill="currentColor"' in icon.pixmap.loaded_data
    assert b'stroke="currentColor"' in icon.pixmap.loaded_data


def test_colorize_svg_file_requires_existing_path(icon_utils, tmp_path):
    with pytest.raises(FileNotFoundError, match="SVG file not found"):
        icon_utils.IconColorizer.colorize_svg_file(tmp_path / "missing.svg", "#000")


def test_colorize_svg_file_validates_color_type(icon_utils, tmp_path):
    svg_path = tmp_path / "glyph.svg"
    svg_path.write_text("<svg />", encoding="utf-8")

    with pytest.raises(TypeError, match="color must be a string"):
        icon_utils.IconColorizer.colorize_svg_file(svg_path, None)

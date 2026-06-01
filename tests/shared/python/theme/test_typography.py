"""Focused coverage for shared theme typography helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.shared.python.theme import typography


class FakeQFont:
    """Small test double for PyQt6.QtGui.QFont."""

    class Weight:
        def __init__(self, value: int) -> None:
            self.value = value

        def __eq__(self, other: object) -> bool:
            if isinstance(other, FakeQFont.Weight):
                return self.value == other.value
            return self.value == other

    def __init__(self) -> None:
        self.family = None
        self.families: list[str] = []
        self.point_size = None
        self.weight = None
        self.italic = False

    def setFamily(self, family: str) -> None:
        self.family = family

    def setFamilies(self, families: list[str]) -> None:
        self.families = families

    def setPointSize(self, size: int) -> None:
        self.point_size = size

    def setWeight(self, weight: Weight) -> None:
        self.weight = weight

    def setItalic(self, italic: bool) -> None:
        self.italic = italic


@pytest.fixture
def fake_qfont(monkeypatch):
    qtgui = SimpleNamespace(QFont=FakeQFont)
    monkeypatch.setitem(sys.modules, "PyQt6", SimpleNamespace(QtGui=qtgui))
    monkeypatch.setitem(sys.modules, "PyQt6.QtGui", qtgui)
    return FakeQFont


def test_font_size_and_weight_constants_are_stable():
    assert typography.Sizes.BASE == 10
    assert typography.Sizes.XXXL == 32
    assert typography.Weights.NORMAL == 400
    assert typography.Weights.EXTRABOLD == 800


def test_css_font_stack_strings_reference_expected_families():
    assert typography.CSS_FONT_UI.startswith("font-family: ")
    assert '"Outfit"' in typography.CSS_FONT_DISPLAY
    assert '"JetBrains Mono"' in typography.CSS_FONT_MONO


def test_get_qfont_uses_ui_stack_by_default(fake_qfont):
    font = typography.get_qfont(size=12, weight=typography.Weights.SEMIBOLD)

    assert isinstance(font, fake_qfont)
    assert font.families[:3] == ["Outfit", "Inter", "SF Pro Display"]
    assert font.point_size == 12
    assert font.weight == typography.Weights.SEMIBOLD
    assert font.italic is False


def test_get_qfont_accepts_explicit_family_and_italic(fake_qfont):
    font = typography.get_qfont(
        size=9,
        weight=typography.Weights.LIGHT,
        family="Aptos",
        italic=True,
    )

    assert font.family == "Aptos"
    assert font.families == []
    assert font.point_size == 9
    assert font.weight == typography.Weights.LIGHT
    assert font.italic is True


def test_display_font_uses_display_stack(fake_qfont):
    font = typography.get_display_font(size=18, weight=typography.Weights.BOLD)

    assert font.families[:2] == ["Outfit", "SF Pro Display"]
    assert font.point_size == 18
    assert font.weight == typography.Weights.BOLD


def test_mono_font_uses_monospace_stack(fake_qfont):
    font = typography.get_mono_font(size=11, weight=typography.Weights.MEDIUM)

    assert font.families[:3] == ["JetBrains Mono", "SF Mono", "Cascadia Code"]
    assert font.point_size == 11
    assert font.weight == typography.Weights.MEDIUM


@pytest.mark.parametrize(
    "factory",
    [typography.get_qfont, typography.get_display_font, typography.get_mono_font],
)
def test_font_factories_reject_missing_size(factory, fake_qfont):
    with pytest.raises(ValueError, match="size must be provided"):
        factory(size=None)


def test_public_exports_include_font_factories_and_tokens():
    assert {"get_qfont", "get_display_font", "get_mono_font"} <= set(typography.__all__)
    assert {"FontSizes", "FontWeights", "CSS_FONT_UI"} <= set(typography.__all__)

"""H6 launcher-language stylesheet tests (#4125)."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtGui import QPalette  # noqa: E402

from rate_of_closure.ui.pyqt6.app_style import showcase_stylesheet  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def sheet(qtbot) -> str:  # type: ignore[no-untyped-def]
    # qtbot guarantees a QApplication so QPalette resolves real colors.
    return showcase_stylesheet(QPalette())


class TestShowcaseStylesheet:
    def test_no_hard_coded_hex_colors(self, sheet: str) -> None:
        """Launcher language, palette-derived only (H6 requirement)."""
        assert "#" not in sheet.replace("#resultRow", "")

    def test_buttons_have_hover_pressed_and_shadow(self, sheet: str) -> None:
        assert "QPushButton:hover" in sheet
        assert "QPushButton:pressed" in sheet
        assert "QPushButton:disabled" in sheet
        # Subtle shadow: the heavier bottom edge in the shadow tone.
        assert "border-bottom: 2px solid" in sheet

    def test_group_boxes_and_tabs_get_the_card_treatment(self, sheet: str) -> None:
        assert "QGroupBox" in sheet and "border-radius: 8px" in sheet
        assert "QTabBar::tab:hover" in sheet
        assert "QTabBar::tab:selected" in sheet

    def test_carries_the_v4_selected_row_rules(self, sheet: str) -> None:
        assert 'QFrame#resultRow[selected="true"]' in sheet
        assert "palette(highlight)" in sheet

    def test_only_palette_references_and_rgba_tints(self, sheet: str) -> None:
        """Every color literal is a palette(...) role or an rgba tint."""
        import re

        for match in re.findall(r"(?:color|background)[^;]*:\s*([^;]+);", sheet):
            value = match.strip()
            assert (
                value.startswith(("palette(", "rgba("))
                or value in ("transparent", "none")
                or "solid" in value  # border shorthands checked below
            ), value
        for border in re.findall(r"border[^;]*:\s*[^;]*solid\s+([^;]+);", sheet):
            assert border.strip().startswith(("palette(", "rgba(")), border

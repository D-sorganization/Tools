"""Regression tests for three UI/UX polish fixes:

1. ``MainWindow`` font zoom offset must be clamped to a sane range and
   must NEVER drift past it across launches via QSettings persistence.
2. ``SimulationPanel`` tab labels must use BMP-range Unicode symbols
   that ship with default Linux/Windows/Mac fonts (no high-plane
   emoji that show as missing-glyph boxes when no emoji font exists).
3. ``ToolStrip`` model dropdown must be wide enough to display its
   longest item ("Triple Pendulum") without truncation.
"""

from __future__ import annotations


import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QFont, QFontMetrics

from double_pendulum_golf.gui.simulation_panel import SimulationPanel
from double_pendulum_golf.gui.toolstrip_widget import ToolStrip

# ──────────────────────────────────────────────────────────────────────
# (1) Font zoom offset bounds + persistence
# ──────────────────────────────────────────────────────────────────────


class TestFontZoomOffset:
    """The Ctrl+wheel font zoom offset is bounded and persists correctly."""

    def setup_method(self) -> None:
        QSettings("D-sorganization", "PendulumSimulator").remove("font_zoom_pt")

    def teardown_method(self) -> None:
        QSettings("D-sorganization", "PendulumSimulator").remove("font_zoom_pt")

    def test_offset_range_constants_are_sane(self, qapp) -> None:
        """The offset is bounded to a small range; max ≤ 8 pt above base."""
        from double_pendulum_golf.gui.main_window import MainWindow

        # The offset MIN should not allow font to drop below 8pt
        # (10pt base + min offset >= 8 → min offset >= -2)
        # The offset MAX should not let the font balloon to 24pt or more
        # (10pt base + max offset <= 18 → max offset <= 8)
        assert MainWindow._FONT_OFFSET_MIN >= -4
        assert MainWindow._FONT_OFFSET_MAX <= 8
        assert MainWindow._FONT_OFFSET_MIN < MainWindow._FONT_OFFSET_MAX

    def test_loaded_offset_is_clamped(self, qapp) -> None:
        """If a corrupt/large offset was saved, it must be clamped on load."""
        QSettings("D-sorganization", "PendulumSimulator").setValue("font_zoom_pt", 24)
        from double_pendulum_golf.gui.main_window import MainWindow

        win = MainWindow()
        assert win._font_zoom_pt <= MainWindow._FONT_OFFSET_MAX
        assert win._font_zoom_pt >= MainWindow._FONT_OFFSET_MIN
        win.close()

    def test_negative_offset_is_clamped(self, qapp) -> None:
        QSettings("D-sorganization", "PendulumSimulator").setValue("font_zoom_pt", -100)
        from double_pendulum_golf.gui.main_window import MainWindow

        win = MainWindow()
        assert win._font_zoom_pt >= MainWindow._FONT_OFFSET_MIN
        win.close()

    def test_default_offset_is_zero(self, qapp) -> None:
        """First-ever launch should produce offset 0 (default font size)."""
        from double_pendulum_golf.gui.main_window import MainWindow

        win = MainWindow()
        assert win._font_zoom_pt == 0
        win.close()


# ──────────────────────────────────────────────────────────────────────
# (2) Tab labels render in default fonts
# ──────────────────────────────────────────────────────────────────────


class TestTabLabelsRenderable:
    """Every glyph in every TAB_* label must be present in a default font."""

    @pytest.fixture
    def font_metrics(self, qapp) -> QFontMetrics:
        # The tabs inherit the application font, which on Linux without
        # an emoji font is plain DejaVu Sans / Sans equivalent.
        return QFontMetrics(QFont("Sans", 11))

    def _all_tab_labels(self) -> list[str]:
        return [
            SimulationPanel.TAB_SETUP,
            SimulationPanel.TAB_MASS_MATRIX,
            SimulationPanel.TAB_PLOTS,
            SimulationPanel.TAB_OPTIMIZER,
            SimulationPanel.TAB_NOISE,
        ]

    def test_no_label_uses_high_plane_codepoint(self) -> None:
        """High-plane codepoints (≥ U+1F300) need an emoji font that
        isn't installed on bare Linux/WSL. Reject them."""
        for label in self._all_tab_labels():
            for ch in label:
                assert ord(ch) < 0x1F300, (
                    f"Label {label!r} uses codepoint U+{ord(ch):04X} which "
                    f"requires an emoji font; pick a BMP-range symbol instead."
                )

    def test_all_label_glyphs_in_default_font(self, font_metrics: QFontMetrics) -> None:
        """Every glyph must render in the default font (no missing-glyph boxes)."""
        for label in self._all_tab_labels():
            for ch in label:
                # Skip ASCII spaces and printable letters which are always
                # supported; only check the symbol prefix.
                if ch.isascii() and (ch.isalnum() or ch.isspace()):
                    continue
                assert font_metrics.inFont(ch), (
                    f"Label {label!r} contains glyph U+{ord(ch):04X} {ch!r} "
                    f"which is missing from the default font."
                )

    def test_each_label_has_a_visible_symbol_prefix(self) -> None:
        """Every tab still has a leading symbol so the user can scan them."""
        for label in self._all_tab_labels():
            stripped = label.lstrip()
            assert stripped, f"Empty label: {label!r}"
            first = stripped[0]
            # First non-space char must be non-ASCII (a symbol/icon)
            assert (
                not first.isascii()
            ), f"Label {label!r} should start with a symbol prefix, not {first!r}"


# ──────────────────────────────────────────────────────────────────────
# (3) Model dropdown is wide enough for the longest item
# ──────────────────────────────────────────────────────────────────────


class TestModelDropdownWidth:
    """ToolStrip's model combo must show its longest item without ellipsis."""

    def test_combo_minimum_width_fits_longest_item(self, qapp) -> None:
        ts = ToolStrip()
        cmb = ts.cmb_model
        # Compute the natural width of the longest item under the combo's font
        fm = QFontMetrics(cmb.font())
        longest = max(
            (cmb.itemText(i) for i in range(cmb.count())),
            key=len,
        )
        # Add a generous allowance for the combo box chrome (border, arrow,
        # padding). Qt's QComboBox needs ~30 px past the text width.
        text_w = fm.horizontalAdvance(longest)
        required = text_w + 30
        assert cmb.minimumWidth() >= required, (
            f"Model combo minimum width {cmb.minimumWidth()} px is too narrow "
            f"for longest item {longest!r} ({text_w} px text + chrome = {required} px)"
        )

    def test_combo_size_adjust_policy_grows_to_contents(self, qapp) -> None:
        """sizeAdjustPolicy=AdjustToContents lets the combo expand naturally."""
        from PyQt6.QtWidgets import QComboBox

        ts = ToolStrip()
        cmb = ts.cmb_model
        assert cmb.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToContents

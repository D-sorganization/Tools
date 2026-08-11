"""Regression tests for two UI/UX polish items:

1. **Ellipsoid scale floor too high.** The user reports that even the
   minimum 0.1× ellipsoid scale is sometimes too large. The slider must
   reach down to 0.01× (1/100th of the default) so users can shrink
   ellipsoids further.

2. **Emoji audit.** Every non-ASCII glyph in the GUI source must be
   either ASCII (≤ U+007F) or a BMP-range Unicode codepoint that the
   default Linux/Windows/Mac sans-serif font can render. High-plane
   emoji (≥ U+1F300) require a color emoji font that is not present
   on bare Linux/WSL and so render as missing-glyph boxes.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest
from PyQt6.QtGui import QFont, QFontMetrics

from double_pendulum_golf.gui.toolstrip_widget import ToolStrip

# ──────────────────────────────────────────────────────────────────────
# (1) Ellipsoid scale floor
# ──────────────────────────────────────────────────────────────────────


class TestEllipsoidScaleFloor:
    """Mobility and force ellipsoid sliders must reach a small enough scale."""

    def test_mob_slider_minimum_is_one(self, qapp) -> None:
        ts = ToolStrip()
        assert ts._sld_mob.minimum() == 1

    def test_force_ell_slider_minimum_is_one(self, qapp) -> None:
        ts = ToolStrip()
        assert ts._sld_force_ell.minimum() == 1

    def test_mob_slider_supports_ten_x(self, qapp) -> None:
        """Upper bound: 10× display scale must still be reachable."""
        ts = ToolStrip()
        assert ts._sld_mob.maximum() >= 1000

    def test_force_ell_slider_supports_ten_x(self, qapp) -> None:
        ts = ToolStrip()
        assert ts._sld_force_ell.maximum() >= 1000

    def test_mob_slider_min_emits_one_hundredth(self, qapp) -> None:
        """At slider minimum, the emitted scale must be ≤ 0.01× so the
        user can shrink ellipsoids to 1/100th of the default."""
        ts = ToolStrip()
        captured: list[float] = []
        ts.mob_scale_changed.connect(captured.append)
        ts._sld_mob.setValue(ts._sld_mob.minimum())
        assert captured, "mob_scale_changed never fired"
        # The min must be small enough — at most one-hundredth of unity
        assert captured[-1] <= 0.01 + 1e-9, (
            f"Mob slider floor is {captured[-1]}; should be ≤ 0.01 so the "
            f"user can shrink ellipsoids further than 0.1×"
        )

    def test_force_ell_slider_min_emits_one_hundredth(self, qapp) -> None:
        ts = ToolStrip()
        captured: list[float] = []
        ts.force_ell_scale_changed.connect(captured.append)
        ts._sld_force_ell.setValue(ts._sld_force_ell.minimum())
        assert captured, "force_ell_scale_changed never fired"
        assert (
            captured[-1] <= 0.01 + 1e-9
        ), f"Force ellipsoid slider floor is {captured[-1]}; should be ≤ 0.01"

    def test_default_value_still_emits_one_x(self, qapp) -> None:
        """The default slider position must still emit 1.0×, so existing
        behaviour for users who don't move the slider is unchanged."""
        ts = ToolStrip()
        # ToolStrip's default fires 1.0× — round-trip the default value
        captured: list[float] = []
        ts.mob_scale_changed.connect(captured.append)
        # Force the slider to *transition* through its default to fire
        # the signal once.
        default = ts._sld_mob.value()
        ts._sld_mob.setValue(default + 1)
        ts._sld_mob.setValue(default)
        assert captured, "mob_scale_changed did not fire on default"
        assert captured[-1] == pytest.approx(
            1.0, abs=0.05
        ), f"Default mob scale should be ~1.0×, got {captured[-1]}"


# ──────────────────────────────────────────────────────────────────────
# (2) Emoji audit
# ──────────────────────────────────────────────────────────────────────


_GUI_DIR = Path(__file__).resolve().parent.parent / "src" / "double_pendulum_golf"


def _find_high_plane_chars() -> list[tuple[Path, int, str]]:
    """Walk the package source tree and report any high-plane (≥U+1F300)
    Unicode characters in .py files (excluding tests and the pre-existing
    debt directories)."""
    hits: list[tuple[Path, int, str]] = []
    for path in _GUI_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                for ch in line:
                    if ord(ch) >= 0x1F300:
                        hits.append((path, lineno, ch))
    return hits


def _find_chars_missing_from_default_font() -> list[tuple[Path, int, str]]:
    """Find any non-ASCII glyph used in source that the default font
    cannot render. Skips the obvious math/science set (Greek letters,
    arrows, sub/superscripts, box-drawing, common punctuation)."""
    fm = QFontMetrics(QFont("Sans", 12))
    misses: list[tuple[Path, int, str]] = []
    for path in _GUI_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                for ch in line:
                    if ord(ch) < 128:
                        continue
                    # Box drawing, combining marks, variation selectors
                    cat = unicodedata.category(ch)
                    if cat in ("Mn", "Mc", "Me", "Cf"):  # combining + format
                        continue
                    # QFontMetrics.inFont requires a single BMP code unit;
                    # surrogates and high-plane chars naturally fail.
                    try:
                        if not fm.inFont(ch):
                            misses.append((path, lineno, ch))
                    except (ValueError, TypeError):
                        misses.append((path, lineno, ch))
    return misses


class TestEmojiAudit:
    """No high-plane emoji and no font-missing characters in GUI source."""

    def test_no_high_plane_emoji_in_gui_source(self, qapp) -> None:
        hits = _find_high_plane_chars()
        if hits:
            preview = "\n".join(
                f"  {p.name}:{ln}  U+{ord(ch):05X} {ch!r}" for p, ln, ch in hits[:20]
            )
            pytest.fail(
                f"Found {len(hits)} high-plane Unicode characters that won't "
                f"render without a color emoji font:\n{preview}"
            )

    def test_no_font_missing_characters_in_gui_source(self, qapp) -> None:
        misses = _find_chars_missing_from_default_font()
        if misses:
            preview = "\n".join(
                f"  {p.name}:{ln}  U+{ord(ch):05X} {ch!r} ({unicodedata.name(ch, '?')})"
                for p, ln, ch in misses[:20]
            )
            pytest.fail(
                f"Found {len(misses)} characters that the default font "
                f"cannot render:\n{preview}"
            )

    def test_simulation_panel_tab_labels_renderable(self, qapp) -> None:
        """The new tab labels (added in the previous task) must still
        all render in the default font."""
        from double_pendulum_golf.gui.simulation_panel import SimulationPanel

        fm = QFontMetrics(QFont("Sans", 11))
        for label in (
            SimulationPanel.TAB_SETUP,
            SimulationPanel.TAB_MASS_MATRIX,
            SimulationPanel.TAB_PLOTS,
            SimulationPanel.TAB_OPTIMIZER,
            SimulationPanel.TAB_NOISE,
        ):
            for ch in label:
                if ch.isascii() and (ch.isalnum() or ch.isspace()):
                    continue
                assert fm.inFont(ch), (
                    f"Tab label {label!r} contains glyph U+{ord(ch):04X} "
                    f"which is missing from the default font."
                )

"""Regression tests for two UI/UX polish items:

1. **Default theme on first launch is Dark.** The shared fleet
   ``ThemeManager`` falls back to ``"Light"`` when no preference is
   stored. The pendulum simulator should override that on first launch
   so new users see the dark theme by default — without overwriting an
   existing user preference.

2. **Header buttons must fit their text.** Several toolstrip buttons
   ("Equations of Motion", "Pop-Out Chart", "Diagnostics") were rendered
   with default Qt sizing and got truncated when the toolstrip was
   crowded. Every header button must compute its ``setMinimumWidth``
   from its label text + a fixed chrome padding so the full text is
   always visible.
"""

from __future__ import annotations


import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import QPushButton

from double_pendulum_golf.gui.toolstrip_widget import ToolStrip

# ──────────────────────────────────────────────────────────────────────
# Default theme = Dark on first launch
# ──────────────────────────────────────────────────────────────────────


_THEME_KEY = "theme"
_CONTEXT_KEY = "theme_PendulumSimulator"
_INITIAL_FLAG = "first_launch_initialized"


class TestDefaultDarkTheme:
    """First-launch theme defaults to Dark; existing prefs are preserved."""

    def setup_method(self) -> None:
        s = QSettings("D-sorganization", "PendulumSimulator")
        # Save the existing values so we can restore them after each test
        self._saved = {
            _THEME_KEY: s.value(_THEME_KEY),
            _CONTEXT_KEY: s.value(_CONTEXT_KEY),
            _INITIAL_FLAG: s.value(_INITIAL_FLAG),
        }
        s.remove(_THEME_KEY)
        s.remove(_CONTEXT_KEY)
        s.remove(_INITIAL_FLAG)

    def teardown_method(self) -> None:
        s = QSettings("D-sorganization", "PendulumSimulator")
        for k, v in self._saved.items():
            if v is None:
                s.remove(k)
            else:
                s.setValue(k, v)

    def test_first_launch_writes_dark_default(self, qapp) -> None:
        """A pristine launch (no saved prefs) seeds Dark as the default."""
        from double_pendulum_golf.gui.theme_defaults import (
            ensure_default_theme_seeded,
        )

        ensure_default_theme_seeded()

        s = QSettings("D-sorganization", "PendulumSimulator")
        assert s.value(_INITIAL_FLAG) is not None, (
            "first_launch_initialized flag should be set after seeding"
        )
        assert s.value(_THEME_KEY) == "Dark", (
            f"Default theme should be 'Dark', got {s.value(_THEME_KEY)!r}"
        )

    def test_existing_user_preference_is_not_overwritten(self, qapp) -> None:
        """If a user already chose 'Light', do not stomp it."""
        from double_pendulum_golf.gui.theme_defaults import (
            ensure_default_theme_seeded,
        )

        s = QSettings("D-sorganization", "PendulumSimulator")
        s.setValue(_THEME_KEY, "Light")
        s.setValue(_INITIAL_FLAG, "1")
        s.sync()

        ensure_default_theme_seeded()

        assert s.value(_THEME_KEY) == "Light", (
            "User-chosen 'Light' theme must not be overwritten"
        )

    def test_seeding_is_idempotent(self, qapp) -> None:
        """Calling ensure_default_theme_seeded twice does not flip
        the theme back to Dark after the user has changed it once."""
        from double_pendulum_golf.gui.theme_defaults import (
            ensure_default_theme_seeded,
        )

        # First call: seeds Dark
        ensure_default_theme_seeded()
        s = QSettings("D-sorganization", "PendulumSimulator")
        assert s.value(_THEME_KEY) == "Dark"

        # User picks Light afterwards
        s.setValue(_THEME_KEY, "Light")
        s.sync()

        # Second call: must not override the user's choice
        ensure_default_theme_seeded()
        assert s.value(_THEME_KEY) == "Light"

    def test_seeding_returns_chosen_theme_name(self, qapp) -> None:
        """The helper returns the theme name that is now active so callers
        can log it / display it without re-reading QSettings."""
        from double_pendulum_golf.gui.theme_defaults import (
            ensure_default_theme_seeded,
        )

        result = ensure_default_theme_seeded()
        assert result == "Dark"

    def test_default_constant_is_dark(self, qapp) -> None:
        """The default-name constant is part of the public contract;
        any future change must also update the test."""
        from double_pendulum_golf.gui.theme_defaults import DEFAULT_THEME_NAME

        assert DEFAULT_THEME_NAME == "Dark"


# ──────────────────────────────────────────────────────────────────────
# Header buttons fit their text
# ──────────────────────────────────────────────────────────────────────


class TestHeaderButtonWidth:
    """Every toolstrip header button is wide enough for its full label."""

    @staticmethod
    def _required_width(btn: QPushButton, padding: int = 24) -> int:
        """Return the pixels needed to render the button's text + chrome."""
        fm = QFontMetrics(btn.font())
        return fm.horizontalAdvance(btn.text()) + padding

    def test_reset_view_button_fits_text(self, qapp) -> None:
        ts = ToolStrip()
        btn = ts.btn_reset_view
        assert btn.minimumWidth() >= self._required_width(btn), (
            f"Button {btn.text()!r} minimumWidth={btn.minimumWidth()} is "
            f"too narrow for required={self._required_width(btn)}"
        )

    def test_export_csv_fits_text(self, qapp) -> None:
        ts = ToolStrip()
        btn = ts.btn_export_csv
        assert btn.minimumWidth() >= self._required_width(btn)

    def test_export_video_fits_text(self, qapp) -> None:
        ts = ToolStrip()
        btn = ts.btn_export_video
        assert btn.minimumWidth() >= self._required_width(btn)

    def test_eom_fits_text(self, qapp) -> None:
        ts = ToolStrip()
        btn = ts.btn_eom
        assert btn.minimumWidth() >= self._required_width(btn)

    def test_mass_matrix_fits_text(self, qapp) -> None:
        ts = ToolStrip()
        btn = ts.btn_mass_matrix
        assert btn.minimumWidth() >= self._required_width(btn)

    def test_popout_chart_fits_text(self, qapp) -> None:
        ts = ToolStrip()
        btn = ts.btn_popout
        assert btn.minimumWidth() >= self._required_width(btn)

    def test_diagnostics_fits_text(self, qapp) -> None:
        ts = ToolStrip()
        btn = ts.btn_diagnostics
        assert btn.minimumWidth() >= self._required_width(btn)

    def test_run_button_fits_text(self, qapp) -> None:
        ts = ToolStrip()
        btn = ts.btn_run
        assert btn.minimumWidth() >= self._required_width(btn)

    def test_reset_button_fits_text(self, qapp) -> None:
        ts = ToolStrip()
        btn = ts.btn_reset
        assert btn.minimumWidth() >= self._required_width(btn)

    def test_play_button_initial_label_fits(self, qapp) -> None:
        """The Play button starts as 'Play' but its width should be
        large enough for either label so the click target doesn't jump
        when the user toggles it."""
        ts = ToolStrip()
        btn = ts.btn_play
        # The button must accommodate both labels
        fm = QFontMetrics(btn.font())
        play_w = fm.horizontalAdvance("▶ Play") + 24
        pause_w = fm.horizontalAdvance("‖ Pause") + 24
        required = max(play_w, pause_w)
        assert btn.minimumWidth() >= required, (
            f"Play/Pause button minimumWidth={btn.minimumWidth()} should "
            f"accommodate both labels (max required={required})"
        )

    def test_button_width_helper_is_dry(self, qapp) -> None:
        """A single helper computes the minimum width — verify it exists."""
        from double_pendulum_golf.gui.button_sizing import fit_button_to_text

        # The helper accepts a button and returns the same button (chained)
        btn = QPushButton("Some Long Button Label")
        result = fit_button_to_text(btn)
        assert result is btn
        # And the minimum width must now fit the text
        fm = QFontMetrics(btn.font())
        assert btn.minimumWidth() >= fm.horizontalAdvance(btn.text()) + 16


# ──────────────────────────────────────────────────────────────────────
# DbC: button sizing helper validates inputs
# ──────────────────────────────────────────────────────────────────────


class TestButtonSizingContracts:
    def test_rejects_none_button(self, qapp) -> None:
        from double_pendulum_golf.gui.button_sizing import fit_button_to_text

        with pytest.raises(ValueError, match="button"):
            fit_button_to_text(None)  # type: ignore[arg-type]

    def test_rejects_negative_padding(self, qapp) -> None:
        from double_pendulum_golf.gui.button_sizing import fit_button_to_text

        btn = QPushButton("Hi")
        with pytest.raises(ValueError, match="padding"):
            fit_button_to_text(btn, padding=-5)

    def test_padding_default_is_reasonable(self, qapp) -> None:
        """The default padding leaves at least 16 px of breathing room."""
        from double_pendulum_golf.gui.button_sizing import (
            DEFAULT_BUTTON_PADDING_PX,
            fit_button_to_text,
        )

        assert DEFAULT_BUTTON_PADDING_PX >= 16
        btn = QPushButton("xx")
        fit_button_to_text(btn)
        fm = QFontMetrics(btn.font())
        assert btn.minimumWidth() >= fm.horizontalAdvance("xx") + 16

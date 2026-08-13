"""Help-system contract + GUI tests (#4120 V4).

Every tab must carry substantive, cold-user help (>300 chars), and the
'?' corner button must open the current tab's rich-text help panel.
"""

from __future__ import annotations

import pytest

from rate_of_closure.helptext import HELP_TEXTS

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class TestHelpContract:
    def test_every_tab_has_substantive_help(self) -> None:
        for key, entry in HELP_TEXTS.items():
            assert len(entry.html) > 300, key
            assert entry.title.strip(), key

    def test_help_covers_workflow_and_tips(self) -> None:
        for key, entry in HELP_TEXTS.items():
            assert "Workflow" in entry.html, key
            assert "Tips" in entry.html, key

    def test_keys_match_the_main_window_tabs(self) -> None:
        from rate_of_closure.ui.pyqt6.main_window import _TAB_HELP_KEYS

        assert set(_TAB_HELP_KEYS) == set(HELP_TEXTS)


class TestHelpGui:
    @pytest.fixture
    def window(self, qtbot):  # type: ignore[no-untyped-def]
        pytest.importorskip("PyQt6")
        pytest.importorskip("pytestqt")
        from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow

        win = RateOfClosureMainWindow()
        qtbot.addWidget(win)
        yield win
        if win._help_dialog is not None:
            win._help_dialog.close()
        win._club_view.stop()
        win._simulation_tab.stop()
        win._variation_tab.stop()

    def test_corner_help_button_present_with_tooltip(self, window) -> None:  # type: ignore[no-untyped-def]
        from PyQt6.QtCore import Qt

        button = window._tabs.cornerWidget(Qt.Corner.TopRightCorner)
        assert button is not None
        assert button.text() == "?"
        assert len(button.toolTip()) > 20

    def test_help_dialog_shows_current_tab_content(self, window) -> None:  # type: ignore[no-untyped-def]
        from PyQt6.QtWidgets import QTextBrowser

        window._tabs.setCurrentWidget(window._glossary_tab)
        window.show_help()
        dialog = window._help_dialog
        assert dialog is not None
        browser = dialog.findChild(QTextBrowser, "helpBrowser")
        assert browser is not None
        text = browser.toPlainText()
        assert "Glossary" in dialog.windowTitle()
        assert len(text) > 300
        assert "Workflow" in text

    def test_help_follows_the_selected_tab(self, window) -> None:  # type: ignore[no-untyped-def]
        from PyQt6.QtWidgets import QTextBrowser

        window._tabs.setCurrentIndex(3)  # Simulation
        window.show_help()
        browser = window._help_dialog.findChild(QTextBrowser, "helpBrowser")
        assert "Run Simulation" in browser.toPlainText()

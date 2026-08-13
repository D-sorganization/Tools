"""PyQt6 GUI smoke tests for the Putting tab (#4125 H3).

Headless-safe; exercises the LoD seam — inputs go in through the
public widgets, results come out through ``result()`` and the row
labels, without reaching into the physics internals.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.ui.pyqt6.putting_tab import _ROWS, PuttingTab  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = PuttingTab()
    qtbot.addWidget(widget)
    return widget


class TestPuttingTab:
    def test_constructs_with_live_results(self, tab) -> None:  # type: ignore[no-untyped-def]
        result = tab.result()
        assert result is not None
        assert result.total_distance_m > 0.0
        for field, _label in _ROWS:
            assert tab._rows[field].value_label.text() not in ("", "—")

    def test_stimp_change_recomputes(self, tab) -> None:  # type: ignore[no-untyped-def]
        tab._grade_spin.setValue(0.0)
        before = tab.result().total_distance_m
        tab._stimp_spin.setValue(13.0)
        after = tab.result().total_distance_m
        assert after > before  # faster green rolls out farther

    def test_backstroke_mode_drives_the_putt(self, tab) -> None:  # type: ignore[no-untyped-def]
        tab._pace_mode.setCurrentIndex(1)
        tab._backstroke_spin.setValue(40.0)
        result = tab.result()
        assert result is not None
        assert result.total_distance_m > 0.5

    def test_slope_produces_break(self, tab) -> None:  # type: ignore[no-untyped-def]
        tab._grade_spin.setValue(2.0)
        tab._aspect_spin.setValue(90.0)
        assert tab.result().break_m > 0.0
        tab._aspect_spin.setValue(-90.0)
        assert tab.result().break_m < 0.0

    def test_row_click_shows_explanation_and_glossary_link(self, tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        tab._show_explanation("putt_break_m")
        html = tab._explanation.toHtml()
        assert "Break" in html
        assert "glossary:" in html

    def test_glossary_link_emits_signal(self, tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PyQt6.QtCore import QUrl

        with qtbot.waitSignal(tab.glossaryRequested, timeout=2000) as blocker:
            tab._on_explanation_link(QUrl("glossary:stimp"))
        assert blocker.args == ["stimp"]

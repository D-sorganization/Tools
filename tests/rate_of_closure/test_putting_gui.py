"""PyQt6 GUI smoke tests for the Putting tab (#4125 H3).

Headless-safe; exercises the LoD seam — inputs go in through the
public widgets, results come out through ``result()`` and the row
labels, without reaching into the physics internals.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt  # noqa: E402

from rate_of_closure.ui.pyqt6.putting_tab import _ROWS, PuttingTab  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = PuttingTab()
    qtbot.addWidget(widget)
    return widget


class TestPuttingTab:
    def test_editor_domains_match_putting_contract(self, tab) -> None:  # type: ignore[no-untyped-def]
        assert (tab._stimp_spin.minimum(), tab._stimp_spin.maximum()) == (3.0, 16.0)
        assert (tab._grade_spin.minimum(), tab._grade_spin.maximum()) == (0.0, 10.0)
        assert (tab._aspect_spin.minimum(), tab._aspect_spin.maximum()) == (
            -360.0,
            360.0,
        )
        assert (tab._distance_spin.minimum(), tab._distance_spin.maximum()) == (
            0.1,
            40.0,
        )

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

    def test_keyboard_selection_is_synchronized_and_exact(self, tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        canvas = tab._plot_view.canvas()
        assert canvas.focusPolicy() == Qt.FocusPolicy.StrongFocus
        assert "putt path sample inspector" in canvas.accessibleName().lower()
        canvas.setFocus()
        qtbot.keyClick(canvas, Qt.Key.Key_Home)
        assert tab._plot_view.selected_raw_index() == 0
        assert "Source sample 0" in tab._plot_view.status_text()
        assert len(tab._plot_view.selected_artists()) == 2
        qtbot.keyClick(canvas, Qt.Key.Key_End)
        assert tab._plot_view.selected_raw_index() == len(tab.result().times_s) - 1
        qtbot.keyClick(canvas, Qt.Key.Key_Escape)
        assert tab._plot_view.selected_raw_index() is None

    def test_scientific_replacement_clears_selection_but_unit_refresh_preserves(
        self, tab, qtbot
    ) -> None:  # type: ignore[no-untyped-def]
        canvas = tab._plot_view.canvas()
        canvas.setFocus()
        qtbot.keyClick(canvas, Qt.Key.Key_Home)
        accepted = tab.result()
        assert tab._plot_view.selected_raw_index() == 0
        tab.refresh_units()
        assert tab.result() is accepted
        assert tab._plot_view.selected_raw_index() == 0
        tab._grade_spin.setValue(1.0)
        assert tab.result() is not accepted
        assert tab._plot_view.selected_raw_index() is None

    def test_failed_scientific_replacement_retains_exact_accepted_evidence(
        self, tab, qtbot, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        import rate_of_closure.ui.pyqt6.putting_tab as putting_tab_module

        accepted = tab.result()
        canvas = tab._plot_view.canvas()
        canvas.setFocus()
        qtbot.keyClick(canvas, Qt.Key.Key_Home)

        def fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("solver authority unavailable")

        monkeypatch.setattr(putting_tab_module, "simulate_putt", fail)
        tab._grade_spin.setValue(1.0)

        assert tab.result() is accepted
        assert tab._plot_view.selected_raw_index() == 0
        assert "solver authority unavailable" in tab._plot_view.error_text()
        assert "Source sample 0" in tab._plot_view.status_text()
        assert "Displayed result:" in tab._plot_view.context_text()
        retained_error = tab._plot_view.error_text()
        tab.refresh_units()
        assert tab._plot_view.error_text() == retained_error

    def test_first_failure_and_renderer_failure_are_atomic(
        self, tab, qtbot, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        import rate_of_closure.ui.pyqt6.putting_tab as putting_tab_module

        accepted = tab.result()
        original_draw = tab._plot_view._draw
        calls = 0

        def fail_once():  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            if calls == 1:
                original_draw()
                raise RuntimeError("renderer unavailable")
            original_draw()

        monkeypatch.setattr(tab._plot_view, "_draw", fail_once)
        tab._grade_spin.setValue(1.0)
        assert tab.result() is accepted
        assert "renderer unavailable" in tab._plot_view.error_text()

        def fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("solver unavailable")

        monkeypatch.setattr(putting_tab_module, "simulate_putt", fail)
        empty = putting_tab_module.PuttingTab()
        qtbot.addWidget(empty)
        assert empty.result() is None
        assert "no accepted putt is available" in empty._plot_view.error_text()
        assert empty._plot_view.context_text().startswith("No accepted")

    def test_pointer_nearest_uses_rendered_pixels_and_lower_index_tie(
        self, tab
    ) -> None:  # type: ignore[no-untyped-def]
        points = tab._plot_view.path_display_points()
        first = points[0]
        second = points[1]
        tab._plot_view.select_nearest_pixel(
            tab._plot_view.path_axes(),
            (first[1] + second[1]) / 2.0,
            (first[2] + second[2]) / 2.0,
        )
        assert tab._plot_view.selected_raw_index() == min(first[0], second[0])

    def test_same_result_object_under_new_config_is_a_new_generation(
        self, tab, qtbot, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        import rate_of_closure.ui.pyqt6.putting_tab as putting_tab_module

        accepted = tab.result()
        qtbot.keyClick(tab._plot_view.canvas(), Qt.Key.Key_Home)
        prior_context = tab._plot_view.context_text()
        monkeypatch.setattr(
            putting_tab_module, "simulate_putt", lambda *_args: accepted
        )
        tab._grade_spin.setValue(1.0)
        assert tab.result() is accepted
        assert tab._plot_view.selected_raw_index() is None
        assert tab._plot_view.context_text() != prior_context
        assert "grade 1.00%" in tab._plot_view.context_text()

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

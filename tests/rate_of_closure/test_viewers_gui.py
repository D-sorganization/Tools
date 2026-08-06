"""GUI tests for the scale-separated viewers + Flight Explorer (#4120 V2).

Headless-safe. Covers: the Strike/Swing/Flight sub-tabs inside the
Simulation tab's display area, the strike view's face-scale invariant
(extents never exceed :data:`STRIKE_MAX_EXTENT_MM`), the swing view's
'Show Ball Flight' toggle (default OFF; toggling changes the scene
limits), the dedicated flight view's panels + flight-regime extents,
sourced tooltips on every new control, and the standalone Flight
Explorer tab end-to-end in both entry modes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QToolButton  # noqa: E402

from rate_of_closure.derivation import LAUNCH_EXPLANATIONS  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.ui.pyqt6.flight_explorer_tab import (  # noqa: E402
    EXPLORER_ROWS,
    FlightExplorerTab,
)
from rate_of_closure.ui.pyqt6.flight_view import FlightView  # noqa: E402
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow  # noqa: E402
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab  # noqa: E402
from rate_of_closure.ui.pyqt6.strike_view import (  # noqa: E402
    STRIKE_MAX_EXTENT_MM,
    StrikeView,
    face_half_extents_mm,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def ran_tab(qtbot):  # type: ignore[no-untyped-def]
    tab = SimulationTab()
    qtbot.addWidget(tab)
    tab.set_scenario(
        ImpactScenario(
            clubhead_speed_mph=113.0,
            impact_offset_toe_mm=6.0,
            impact_offset_high_mm=3.0,
        )
    )
    with qtbot.waitSignal(tab.runCompleted, timeout=15000):
        tab.run_now()
    yield tab
    tab.stop()


class TestSimulationSubTabs:
    def test_display_area_hosts_strike_swing_flight_sub_tabs(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        from PyQt6.QtWidgets import QTabWidget

        assert isinstance(ran_tab.strike_view(), StrikeView)
        assert isinstance(ran_tab.flight_view(), FlightView)
        tab_widgets = ran_tab.findChildren(QTabWidget)
        texts = {
            widget.tabText(i) for widget in tab_widgets for i in range(widget.count())
        }
        assert {"Strike", "Swing", "Flight", "Inspector", "Solver"} <= texts

    def test_run_populates_all_three_viewers(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        assert ran_tab.strike_view().run() is not None
        assert len(ran_tab.flight_view().trajectory()) > 2
        assert ran_tab.view().run() is not None


class TestStrikeViewScale:
    def test_extents_never_exceed_face_scale(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        strike = ran_tab.strike_view()
        half_x, half_y = strike.extents_mm()
        assert half_x <= STRIKE_MAX_EXTENT_MM
        assert half_y <= STRIKE_MAX_EXTENT_MM
        run = strike.run()
        assert run is not None
        half_w, half_h = face_half_extents_mm(run.config.club)
        assert half_x <= max(half_w, half_h) * 1.5

    def test_strike_history_accumulates(self, ran_tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        strike = ran_tab.strike_view()
        before = len(strike.strike_history())
        with qtbot.waitSignal(ran_tab.runCompleted, timeout=15000):
            ran_tab.run_now()
        assert len(strike.strike_history()) == before + 1
        strike.clear_history()
        assert strike.strike_history() == []

    def test_display_checklist_toggles_redraw(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        strike = ran_tab.strike_view()
        for name in ("curvature", "vectors", "history", "club_info", "show_cg"):
            check = strike.display_check(name)
            check.setChecked(not check.isChecked())
            check.setChecked(not check.isChecked())

    def test_every_display_control_has_sourced_guidance(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        strike = ran_tab.strike_view()
        for name in ("curvature", "vectors", "history", "club_info", "show_cg"):
            assert "Source:" in strike.display_check(name).toolTip(), name

    def test_show_cg_defaults_off_and_marks_the_volumetric_cog(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        """H1 (#4125): the strike view's CG marker toggles on demand."""
        strike = ran_tab.strike_view()
        check = strike.display_check("show_cg")
        assert not check.isChecked()
        check.setChecked(True)
        labels = [text.get_text() for text in strike._axes.get_legend().get_texts()]
        assert any("volumetric CG" in label for label in labels)


class TestSwingViewFlightToggle:
    def test_flight_display_defaults_off(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        assert ran_tab.view().flight_shown() is False

    def test_toggle_expands_and_restores_scene_limits(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        run = view.run()
        assert run is not None
        view.set_playback_time(run.total_duration_s)  # deep in the flight
        swing_extent = view.scene_extent_m()
        assert swing_extent < 10.0, "swing scale must stay metres-sized"
        view.set_flight_shown(True)
        flight_extent = view.scene_extent_m()
        assert flight_extent > swing_extent * 5.0
        view.set_flight_shown(False)
        assert view.scene_extent_m() == pytest.approx(swing_extent)

    def test_course_elements_toggle_defaults_on_with_guidance(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        """H7a: the swing scene's Course Elements toggle + layout seam."""
        from rate_of_closure.ui.course import CourseLayout

        view = ran_tab.view()
        assert view.course_elements_shown() is True
        assert "Source:" in view._course_check.toolTip()
        view._course_check.setChecked(False)
        assert view.course_elements_shown() is False
        view._course_check.setChecked(True)
        view.set_course_layout(CourseLayout(green_distance_m=180.0))
        assert view.course_layout().green_distance_m == 180.0

    def test_toggle_carries_a_scale_warning_tooltip(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        tooltip = ran_tab.view()._flight_check.toolTip()
        assert "dwarfs" in tooltip
        assert "Source:" in tooltip


class TestFlightView:
    def test_extents_track_the_flight_regime(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        flight = ran_tab.flight_view()
        carry_ext, height_ext, lateral_ext = flight.extents_m()
        trajectory = flight.trajectory()
        assert carry_ext >= float(trajectory[:, 0].max())
        assert height_ext >= float(trajectory[:, 1].max())
        assert lateral_ext >= float(abs(trajectory[:, 2]).max())
        assert carry_ext > 50.0, "a 113 mph driver flight is flight-scale"

    def test_display_checklist_and_guidance(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        flight = ran_tab.flight_view()
        for name in ("side", "top", "three_d", "landing", "apex", "course"):
            check = flight.display_check(name)
            assert "Source:" in check.toolTip(), name
            check.setChecked(not check.isChecked())
            check.setChecked(not check.isChecked())

    def test_course_scene_toggle_and_layout_seam(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        """H7a: course elements default on; the layout seam redraws."""
        from rate_of_closure.ui.course import CourseLayout

        flight = ran_tab.flight_view()
        assert flight.display_check("course").isChecked()
        flight.display_check("course").setChecked(False)
        flight.display_check("course").setChecked(True)
        flight.set_course_layout(CourseLayout(green_distance_m=150.0))
        assert flight.course_layout().green_distance_m == 150.0

    def test_all_panels_off_shows_placeholder_without_crashing(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        flight = ran_tab.flight_view()
        for name in ("side", "top", "three_d"):
            flight.display_check(name).setChecked(False)
        for name in ("side", "top", "three_d"):
            flight.display_check(name).setChecked(True)


class TestFlightExplorerTab:
    @pytest.fixture
    def explorer(self, qtbot):  # type: ignore[no-untyped-def]
        tab = FlightExplorerTab()
        qtbot.addWidget(tab)
        return tab

    def test_main_window_hosts_the_flight_explorer_tab(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        window = RateOfClosureMainWindow()
        qtbot.addWidget(window)
        try:
            tabs = window.centralWidget().findChildren(FlightExplorerTab)
            assert tabs, "main window must host the Flight Explorer tab"
        finally:
            window._club_view.stop()
            window._simulation_tab.stop()

    def test_every_result_row_has_an_explanation(self) -> None:
        for key, _label, _unit in EXPLORER_ROWS:
            assert key in LAUNCH_EXPLANATIONS, key

    def test_launch_direction_has_an_obvious_clickable_definition(
        self, explorer  # type: ignore[no-untyped-def]
    ) -> None:
        button = explorer.findChild(QToolButton, "launch_direction_info")
        assert button is not None
        assert button.text() == "Details"
        assert button.toolTip().startswith("Suggested range:")
        assert "positive values start right" in button.toolTip()
        assert button.accessibleName() == "Explain Launch Direction"
        combo = explorer._direction_convention_combo
        assert combo.accessibleName() == "Launch Direction Convention"
        assert combo.count() == 2
        assert "0° = straight" in explorer._direction_example.text()

    def test_direct_mode_end_to_end_matches_the_pinned_case(self, explorer) -> None:  # type: ignore[no-untyped-def]
        exploration = explorer.run_now()
        assert exploration is not None
        # Defaults are the pinned tour-driver case (167 mph / 10.9 deg /
        # 2686 rpm, waterloo_penner) from test_flight_explorer.py.
        assert exploration.metrics["carry_m"] == pytest.approx(247.484, abs=0.05)
        # H6 (#4125): 247.5 m reads as 270.7 yd (yards default).
        assert "270.7 yd" in explorer._rows["carry_m"].value_label.text().replace(
            "+", ""
        )
        assert len(explorer.flight_view().trajectory()) > 2

    def test_delivery_mode_end_to_end(self, explorer) -> None:  # type: ignore[no-untyped-def]
        explorer._mode_combo.setCurrentIndex(1)
        explorer._speed_spin.setValue(112.0)  # mph clubhead speed
        exploration = explorer.run_now()
        assert exploration is not None
        assert exploration.metrics["carry_m"] > 100.0
        assert exploration.metrics["ball_speed_mph"] > 112.0  # smash > 1

    def test_speed_unit_dropdown_converts_in_place(self, explorer) -> None:  # type: ignore[no-untyped-def]
        explorer._speed_spin.setValue(167.0)
        mps_before = explorer.speed_mps()
        explorer._speed_unit_combo.setCurrentText("m/s")
        assert explorer._speed_spin.value() == pytest.approx(74.66, abs=0.05)
        assert explorer.speed_mps() == pytest.approx(mps_before, abs=0.05)

    def test_model_picker_covers_all_seven_models(self, explorer) -> None:  # type: ignore[no-untyped-def]
        assert explorer._model_combo.count() == 7

    def test_every_new_control_has_sourced_guidance(self, explorer) -> None:  # type: ignore[no-untyped-def]
        controls = [
            explorer._mode_combo,
            explorer._speed_spin,
            explorer._speed_unit_combo,
            explorer._model_combo,
            *explorer._direct_spins.values(),
            *explorer._delivery_spins.values(),
        ]
        for control in controls:
            assert "Source:" in control.toolTip(), control
        assert explorer._run_button.toolTip()

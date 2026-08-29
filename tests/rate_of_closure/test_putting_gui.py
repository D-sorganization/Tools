"""PyQt6 GUI smoke tests for the Putting tab (#4125 H3, #4800 P6/P8).

Headless-safe; exercises the LoD seam — inputs go in through the
public widgets, results come out through ``result()``/``document()``
and the row labels, without reaching into the physics internals.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt  # noqa: E402

from rate_of_closure.simulation.playback_transport import (  # noqa: E402
    PLAYBACK_SPEEDS,
    SCRUB_STEPS,
)
from rate_of_closure.ui.pyqt6.putting_tab import _ROWS, PuttingTab  # noqa: E402
from shared.python.swing_sim.putting import (  # noqa: E402
    GridGreenSurface,
    PlanarGreenSurface,
    green_surface_to_json,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: Float32-exact blade-like box (the golf_club C1 mesh test idiom), so
#: the STL round-trip is bit-exact and the import gate is deterministic.
_BLADE_EXTENTS = (0.03125, 0.03125, 0.125)


def _box_mesh() -> np.ndarray:
    """A watertight outward-wound rectangular putter-head box."""
    hx, hy, hz = (extent / 2.0 for extent in _BLADE_EXTENTS)
    corners = np.array(
        [[sx, sy, sz] for sx in (-hx, hx) for sy in (-hy, hy) for sz in (-hz, hz)]
    )
    faces = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    triangles = []
    for a, b, c, d in faces:
        triangles.append(corners[[a, b, c]])
        triangles.append(corners[[a, c, d]])
    return np.asarray(triangles, dtype=np.float64)


def _binary_stl_bytes(triangles: np.ndarray) -> bytes:
    """Serialize triangles as a binary STL (the C1 test idiom)."""
    header = b"putting tab gate".ljust(80, bytes(1))
    blob = [header, struct.pack("<I", len(triangles))]
    for a, b, c in triangles:
        normal = np.cross(b - a, c - a)
        normal = normal / np.linalg.norm(normal)
        blob.append(struct.pack("<12fH", *normal, *a, *b, *c, 0))
    return b"".join(blob)


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = PuttingTab()
    qtbot.addWidget(widget)
    return widget


class TestPuttingTab:
    def test_editor_domains_match_putting_contract(self, tab) -> None:  # type: ignore[no-untyped-def]
        green = tab.green_controls()
        assert (green.stimp_spin.minimum(), green.stimp_spin.maximum()) == (3.0, 16.0)
        assert (green.grade_spin.minimum(), green.grade_spin.maximum()) == (0.0, 10.0)
        assert (green.aspect_spin.minimum(), green.aspect_spin.maximum()) == (
            -360.0,
            360.0,
        )
        assert (green.distance_spin.minimum(), green.distance_spin.maximum()) == (
            0.1,
            40.0,
        )

    def test_stroke_editor_domains_match_the_impact_contract(self, tab) -> None:  # type: ignore[no-untyped-def]
        """Every delivery spin box is bounded by strike()'s own limit."""
        stroke = tab.stroke_controls()
        bounds = {
            stroke.aim_spin: 45.0,
            stroke.face_spin: 20.0,
            stroke.path_spin: 20.0,
            stroke.attack_spin: 10.0,
            stroke.lean_spin: 10.0,
            stroke.toe_spin: 40.0,
            stroke.high_spin: 20.0,
        }
        for spin, bound in bounds.items():
            assert (spin.minimum(), spin.maximum()) == (-bound, bound)

    def test_constructs_with_live_results(self, tab) -> None:  # type: ignore[no-untyped-def]
        result = tab.result()
        assert result is not None
        assert result.total_distance_m > 0.0
        for field, _label in _ROWS:
            assert tab._rows[field].value_label.text() not in ("", "—")

    def test_stimp_change_recomputes(self, tab) -> None:  # type: ignore[no-untyped-def]
        tab.green_controls().grade_spin.setValue(0.0)
        before = tab.result().total_distance_m
        tab.green_controls().stimp_spin.setValue(13.0)
        after = tab.result().total_distance_m
        assert after > before  # faster green rolls out farther

    def test_backstroke_mode_drives_the_putt(self, tab) -> None:  # type: ignore[no-untyped-def]
        tab.stroke_controls().pace_mode.setCurrentIndex(1)
        tab.stroke_controls().backstroke_spin.setValue(40.0)
        result = tab.result()
        assert result is not None
        assert result.total_distance_m > 0.5

    def test_slope_produces_break(self, tab) -> None:  # type: ignore[no-untyped-def]
        tab.green_controls().grade_spin.setValue(2.0)
        tab.green_controls().aspect_spin.setValue(90.0)
        assert tab.result().break_m > 0.0
        tab.green_controls().aspect_spin.setValue(-90.0)
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
        tab.green_controls().grade_spin.setValue(1.0)
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

        monkeypatch.setattr(putting_tab_module, "simulate_putt_on_surface", fail)
        tab.green_controls().grade_spin.setValue(1.0)

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
        tab.green_controls().grade_spin.setValue(1.0)
        assert tab.result() is accepted
        assert "renderer unavailable" in tab._plot_view.error_text()

        def fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("solver unavailable")

        monkeypatch.setattr(putting_tab_module, "simulate_putt_on_surface", fail)
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
            putting_tab_module,
            "simulate_putt_on_surface",
            lambda *_args, **_kwargs: accepted,
        )
        tab.green_controls().grade_spin.setValue(1.0)
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


class TestPuttingStrokeControls:
    """The #4800 P1 delivery parameters reach the impact solve."""

    def test_square_centred_stroke_starts_on_the_target_line(self, tab) -> None:  # type: ignore[no-untyped-def]
        document = tab.document()
        assert document is not None
        assert document.start_azimuth_deg == 0.0
        assert document.sidespin_rad_s == 0.0
        assert tab._rows["putt_start_azimuth_deg"].value_label.text() == (
            "0.00° (on the target line)"
        )

    def test_aim_shifts_the_start_line_one_for_one(self, tab) -> None:  # type: ignore[no-untyped-def]
        tab.stroke_controls().aim_spin.setValue(2.0)
        assert tab.document().start_azimuth_deg == pytest.approx(2.0)

    def test_face_dominates_path_on_the_start_line(self, tab) -> None:  # type: ignore[no-untyped-def]
        """Face-only beats path-only: the 2/7 tangential cap is small."""
        stroke = tab.stroke_controls()
        stroke.face_spin.setValue(1.0)
        face_only = tab.document().start_azimuth_deg
        stroke.face_spin.setValue(0.0)
        stroke.path_spin.setValue(1.0)
        path_only = tab.document().start_azimuth_deg
        assert face_only > path_only > 0.0
        assert face_only == pytest.approx(1.0, abs=0.25)

    def test_toe_strike_costs_ball_speed_and_twists_the_face(self, tab) -> None:  # type: ignore[no-untyped-def]
        centred = tab.document().ball_speed_mps
        twist_row = tab._rows["putt_face_twist_deg"].value_label
        assert twist_row.text() == "0.000° (centred strike)"
        tab.stroke_controls().toe_spin.setValue(20.0)
        assert tab.document().ball_speed_mps < centred
        assert twist_row.text().endswith("open")

    def test_heel_strike_closes_the_face_and_the_row_says_so(self, tab) -> None:  # type: ignore[no-untyped-def]
        tab.stroke_controls().toe_spin.setValue(-20.0)
        assert tab._rows["putt_face_twist_deg"].value_label.text().endswith("closed")

    def test_mesh_putter_import_replaces_the_library_head(self, tab, tmp_path) -> None:  # type: ignore[no-untyped-def]
        stl_path = tmp_path / "milled_blade.stl"
        stl_path.write_bytes(_binary_stl_bytes(_box_mesh()))
        name = tab.stroke_controls().adopt_putter_mesh(
            stl_path, loft_deg=3.0, target_mass_kg=0.35
        )
        assert name == "milled_blade"
        head = tab.stroke_controls().head_document()
        assert head.provenance.source_kind == "mesh"
        assert head.inertia_at_cg_kg_m2 is not None
        tab._recompute()
        assert tab.document().provenance.putter_source == "mesh"
        assert "[mesh]" in tab._plot_view.context_text()


class TestPuttingGreenControls:
    """The #4800 P2 surface and capture model reach the integrator."""

    def test_capture_model_reaches_the_record_provenance(self, tab) -> None:  # type: ignore[no-untyped-def]
        assert tab.document().provenance.capture_model == "effective_radius"
        tab.green_controls().capture_combo.setCurrentIndex(1)
        assert tab.document().provenance.capture_model == "speed_threshold"

    def test_imported_heightfield_replaces_the_planar_green(  # type: ignore[no-untyped-def]
        self, tab, tmp_path
    ) -> None:
        document = tmp_path / "green.json"
        document.write_text(
            green_surface_to_json(
                GridGreenSurface(
                    origin_m=(-1.0, -2.0),
                    spacing_m=0.5,
                    heights_m=tuple(
                        tuple(-0.01 * column for column in range(16))
                        for _row in range(16)
                    ),
                )
            ),
            encoding="utf-8",
        )
        green = tab.green_controls()
        label = green.adopt_green_document(document)
        assert "swing_sim.green_surface/1" in label
        assert isinstance(green.surface(), GridGreenSurface)
        assert not green.grade_spin.isEnabled()
        assert green.planar_button.isEnabled()
        assert "green.json" in tab._plot_view.context_text()
        assert tab.result() is not None

        green.use_planar_green()
        assert isinstance(green.surface(), PlanarGreenSurface)
        assert green.grade_spin.isEnabled()

    def test_upstreamdrift_topography_imports_through_the_p9_adapter(
        self, tab, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        document = tmp_path / "ud_green.json"
        document.write_text(
            json.dumps(
                {
                    "contours": [
                        {"x": x * 0.5, "y": y * 0.5, "elevation": -0.005 * x}
                        for y in range(6)
                        for x in range(6)
                    ]
                }
            ),
            encoding="utf-8",
        )
        label = tab.green_controls().adopt_green_document(document)
        assert "upstreamdrift" in label
        assert isinstance(tab.green_controls().surface(), GridGreenSurface)

    def test_refused_import_keeps_the_previous_green(self, tab, tmp_path) -> None:  # type: ignore[no-untyped-def]
        document = tmp_path / "bad.json"
        document.write_text('{"format": "swing_sim.green_surface/9"}', encoding="utf-8")
        with pytest.raises(ValueError):
            tab.green_controls().adopt_green_document(document)
        assert isinstance(tab.green_controls().surface(), PlanarGreenSurface)


class TestPuttingPlayback:
    """Playback frames come from the recorded samples, never re-simulation."""

    def test_timeline_matches_the_retained_samples(self, tab) -> None:  # type: ignore[no-untyped-def]
        view = tab.playback_view()
        result = tab.result()
        trajectory = view.trajectory()
        assert trajectory is not None
        assert len(trajectory.times_s) == len(result.times_s)
        assert view.duration_s() == pytest.approx(result.times_s[-1])
        assert view.event_times_s() == (0.0, view.duration_s())
        first = trajectory.frame_at(0.0)
        assert float(first.position_m[0]) == pytest.approx(result.path_x_m[0])
        assert float(first.position_m[1]) == pytest.approx(result.path_y_m[0])

    def test_scrubbing_moves_the_ball_and_announces_it(self, tab) -> None:  # type: ignore[no-untyped-def]
        view = tab.playback_view()
        view.set_time(0.0)
        start = view.status_text()
        view.set_time(view.duration_s())
        end = view.status_text()
        assert start != end
        assert "t 0.000 s" in start
        assert f"of {view.duration_s():.3f} s" in end

    def test_ball_rides_the_imported_surface_elevation(self, tab, tmp_path) -> None:  # type: ignore[no-untyped-def]
        tab.green_controls().grade_spin.setValue(3.0)
        tab.green_controls().aspect_spin.setValue(0.0)
        trajectory = tab.playback_view().trajectory()
        heights = trajectory.positions_m[:, 2]
        # Downhill straight ahead: elevation falls as the ball rolls out.
        assert heights[-1] < heights[0]

    def test_transport_binds_the_view_without_a_second_transport(self, tab) -> None:  # type: ignore[no-untyped-def]
        """P8's shared widget drives P6's view; nothing is re-implemented."""
        from rate_of_closure.ui.pyqt6.playback_transport_controls import (
            PlaybackTransportControls,
        )

        controls, view = tab.playback_controls(), tab.playback_view()
        assert isinstance(controls, PlaybackTransportControls)
        assert not hasattr(view, "timer")
        assert controls.duration_s() == pytest.approx(view.duration_s())
        assert controls.scrubber.maximum() == SCRUB_STEPS
        assert [button.text() for button in controls.event_buttons] == [
            "Strike",
            "Finish",
        ]
        assert [
            controls.speed_combo.itemData(index)
            for index in range(controls.speed_combo.count())
        ] == list(PLAYBACK_SPEEDS)

    def test_transport_scrub_moves_the_recorded_ball(self, tab) -> None:  # type: ignore[no-untyped-def]
        controls, view = tab.playback_controls(), tab.playback_view()
        controls.scrubber.setValue(0)
        start = view.status_text()
        controls.scrubber.setValue(SCRUB_STEPS)
        assert view.status_text() != start
        assert controls.current_time_s() == pytest.approx(view.duration_s())
        assert f"t {view.duration_s():.3f} s" in view.status_text()

    def test_transport_event_jumps_land_on_the_recorded_endpoints(self, tab) -> None:  # type: ignore[no-untyped-def]
        controls = tab.playback_controls()
        strike, finish = tab.playback_view().event_times_s()
        controls.jump_to_finish()
        assert controls.current_time_s() == pytest.approx(finish)
        controls.jump_to_strike()
        assert controls.current_time_s() == pytest.approx(strike)
        assert not controls.timer().isActive()

    def test_clearing_collapses_the_transport_with_the_scene(self, tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        """A dropped scene must not leave a playable phantom timeline."""
        from rate_of_closure.ui.pyqt6.putt_playback_controls import PuttPlaybackPanel

        panel = PuttPlaybackPanel()
        qtbot.addWidget(panel)
        result = tab.result()
        panel.set_putt(result, tab.green_controls().surface(), hole_distance_m=3.0)
        assert panel.controls.duration_s() == pytest.approx(result.times_s[-1])
        assert panel.controls.play_button.isEnabled()
        panel.clear()
        assert panel.view.trajectory() is None
        assert panel.controls.duration_s() == 0.0
        assert not panel.controls.play_button.isEnabled()

    def test_recompute_readopts_the_transport_timeline(self, tab) -> None:  # type: ignore[no-untyped-def]
        controls = tab.playback_controls()
        before = controls.duration_s()
        tab.green_controls().stimp_spin.setValue(
            tab.green_controls().stimp_spin.value() + 2.0
        )
        assert controls.duration_s() != pytest.approx(before)
        assert controls.duration_s() == pytest.approx(tab.playback_view().duration_s())
        assert controls.current_time_s() == 0.0

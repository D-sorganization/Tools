"""Glossary contract + GUI tests (#4120 V4).

Covers the DbC glossary module (key hygiene, coverage of the app's
vocabulary, explanation-field mapping), the searchable Glossary tab,
the explanation-panel Glossary links, and the persistent selected-row
state with its prominent name header.
"""

from __future__ import annotations

import pytest

from rate_of_closure.derivation import (
    LAUNCH_EXPLANATIONS,
    METRIC_EXPLANATIONS,
    RESULT_EXPLANATIONS,
)
from rate_of_closure.glossary import FIELD_TO_TERM, GLOSSARY, search_terms

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: Major terms the glossary must define (spot-pinned per the V4 scope;
#: the web mirror pins the FULL key list against this module).
MAJOR_TERMS: tuple[str, ...] = (
    "apex",
    "attack_angle",
    "bulge",
    "carry",
    "ccv",
    "cg_depth",
    "club_path",
    "cor",
    "coriolis",
    "d_plane",
    "dispersion_ellipse",
    "dynamic_loft",
    "effective_mass",
    "face_angle",
    "friction_spin_cap",
    "gear_effect",
    "htv",
    "landing_angle",
    "launch_angle",
    "launch_azimuth",
    "mass_matrix",
    "moi",
    "monte_carlo",
    "normal_distribution",
    "pitch",
    "plane_inclination",
    "r_isa",
    "roll",
    "screw_axis",
    "sensitivity_analysis",
    "smash_factor",
    "spearman",
    "spin_axis_tilt",
    "spin_loft",
    "spv",
    "triangular_distribution",
    "twist",
    "uniform_distribution",
)


class TestGlossaryContract:
    def test_keys_are_sorted_and_snake_case(self) -> None:
        keys = list(GLOSSARY)
        assert keys == sorted(keys)
        for key in keys:
            assert key == key.lower()
            assert " " not in key

    def test_every_major_term_is_defined(self) -> None:
        missing = [term for term in MAJOR_TERMS if term not in GLOSSARY]
        assert missing == []

    def test_definitions_are_substantive_and_sourced(self) -> None:
        for key, entry in GLOSSARY.items():
            assert len(entry.definition) >= 60, key
            assert entry.term.strip(), key
            # Every definition names its source in parentheses.
            assert "(" in entry.definition and ")" in entry.definition, key

    def test_every_explanation_field_maps_to_a_term(self) -> None:
        fields = (
            set(RESULT_EXPLANATIONS)
            | set(METRIC_EXPLANATIONS)
            | set(LAUNCH_EXPLANATIONS)
        )
        unmapped = fields - set(FIELD_TO_TERM)
        assert unmapped == set()
        for field, term in FIELD_TO_TERM.items():
            assert term in GLOSSARY, field

    def test_web_fixture_matches_python_glossary(self) -> None:
        """The checked-in web parity fixture mirrors this module."""
        import json
        from pathlib import Path

        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "rate_of_closure"
            / "web"
            / "src"
            / "model"
            / "glossary.fixture.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        assert fixture["keys"] == list(GLOSSARY)

        def camel(snake: str) -> str:
            head, *rest = snake.split("_")
            return head + "".join(part.title() for part in rest)

        assert fixture["field_terms"] == {
            camel(field): term for field, term in FIELD_TO_TERM.items()
        }

    def test_search_filters_and_empty_query_returns_all(self) -> None:
        assert search_terms("") == tuple(GLOSSARY)
        hits = search_terms("Cheetham")
        assert 0 < len(hits) < len(GLOSSARY)
        assert "ccv" in hits
        assert search_terms("zzz-no-such-term") == ()


@pytest.fixture
def window(qtbot):  # type: ignore[no-untyped-def]
    pytest.importorskip("PyQt6")
    pytest.importorskip("pytestqt")
    from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow

    win = RateOfClosureMainWindow()
    qtbot.addWidget(win)
    yield win
    win._club_view.stop()
    win._simulation_tab.stop()
    win._variation_tab.stop()


class TestGlossaryTab:
    def test_tab_present_and_lists_every_term(self, window) -> None:  # type: ignore[no-untyped-def]
        tab = window._glossary_tab
        labels = [window._tabs.tabText(i) for i in range(window._tabs.count())]
        assert "Glossary" in labels
        assert tab._list.count() == len(GLOSSARY)

    def test_search_filters_the_list(self, window) -> None:  # type: ignore[no-untyped-def]
        tab = window._glossary_tab
        tab._search.setText("gear")
        assert 0 < tab._list.count() < len(GLOSSARY)
        tab._search.setText("")
        assert tab._list.count() == len(GLOSSARY)

    def test_select_term_shows_definition(self, window) -> None:  # type: ignore[no-untyped-def]
        tab = window._glossary_tab
        tab.select_term("gear_effect")
        assert tab.current_term() == "gear_effect"
        assert "Gear Effect" in tab._definition.toPlainText()

    def test_open_glossary_switches_tab_and_preselects(self, window) -> None:  # type: ignore[no-untyped-def]
        window.open_glossary("d_plane")
        assert window._tabs.currentWidget() is window._glossary_tab
        assert window._glossary_tab.current_term() == "d_plane"

    def test_explanation_panel_carries_glossary_link(self, window) -> None:  # type: ignore[no-untyped-def]
        window._show_explanation("closure_rate_dps")
        assert "glossary:ccv" in window._explanation.toHtml()


class TestSelectedValueClarity:
    def test_single_selection_across_all_row_groups(self, window) -> None:  # type: ignore[no-untyped-def]
        window._show_explanation("path_deviation_deg")
        window._show_explanation("ccv_dps")  # a metric row, other group
        selected = [field for field, row in window._rows.items() if row.is_selected()]
        assert selected == ["ccv_dps"]

    def test_header_matches_selected_row_label(self, window) -> None:  # type: ignore[no-untyped-def]
        window._show_explanation("r_isa_ft")
        text = window._explanation.toPlainText()
        assert text.startswith("Distance to Screw Axis (R_ISA)")

    def test_simulation_tab_rows_select_exclusively(self, window) -> None:  # type: ignore[no-untyped-def]
        tab = window._simulation_tab
        tab._show_explanation("carry_m")
        tab._show_explanation("spin_rpm")
        selected = [field for field, row in tab._rows.items() if row.is_selected()]
        assert selected == ["spin_rpm"]
        assert tab._explanation.toPlainText().startswith("Total Spin")

    def test_flight_explorer_rows_select_exclusively(self, window) -> None:  # type: ignore[no-untyped-def]
        tab = window._flight_explorer_tab
        rows = list(tab._rows)
        assert len(rows) >= 2
        tab._show_explanation(rows[0])
        tab._show_explanation(rows[1])
        selected = [field for field, row in tab._rows.items() if row.is_selected()]
        assert selected == [rows[1]]

    def test_selection_stylesheet_uses_palette_highlight(self, window) -> None:  # type: ignore[no-untyped-def]
        sheet = window.styleSheet()
        assert 'QFrame#resultRow[selected="true"]' in sheet
        assert "palette(highlight)" in sheet
        # No hard-coded hex colors in the row styling.
        assert "#" not in sheet.replace("#resultRow", "")

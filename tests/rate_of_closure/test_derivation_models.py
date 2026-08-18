"""Sectioned full-model derivation tests (#4120 V4).

Every step's mathtext must parse, sections must toggle with the live
configuration, and the section keys are pinned (the web mirror pins
the same lists).
"""

from __future__ import annotations

import re

import pytest
from matplotlib.mathtext import MathTextParser

from rate_of_closure.derivation_models import (
    DerivationConfig,
    derivation_sections,
)
from rate_of_closure.model import ImpactScenario
from shared.python.swing_sim.flight.registry import FlightModelType

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_SCENARIO = ImpactScenario(clubhead_speed_mph=113.0)

_CONFIGS = [
    DerivationConfig(),
    DerivationConfig(gear_effect=False),
    DerivationConfig(swing_source="double_pendulum"),
    DerivationConfig(
        swing_source="triple_pendulum", plane_tilts_deg=(10.0, -50.0, 5.0)
    ),
    *[DerivationConfig(flight_model=model.value) for model in FlightModelType],
]


def _mathtext_chunks(text: str) -> list[str]:
    """The ``$...$`` chunks of a latex/values line."""
    return [f"${chunk}$" for chunk in re.findall(r"\$([^$]+)\$", text)]


class TestMathtextParses:
    @pytest.mark.parametrize("config", _CONFIGS)
    def test_every_step_parses(self, config: DerivationConfig) -> None:
        parser = MathTextParser("agg")
        for section in derivation_sections(_SCENARIO, config):
            for step in section.steps:
                for line in (step.latex, step.values):
                    chunks = _mathtext_chunks(line)
                    assert chunks, (section.key, step.title, line)
                    for chunk in chunks:
                        parser.parse(chunk)  # raises on malformed mathtext


class TestSectionToggling:
    def test_default_keys_are_pinned(self) -> None:
        keys = [s.key for s in derivation_sections(_SCENARIO)]
        assert keys == ["closure", "impact", "flight"]

    def test_pendulum_sources_add_the_swing_section(self) -> None:
        for source in ("double_pendulum", "triple_pendulum"):
            keys = [
                s.key
                for s in derivation_sections(
                    _SCENARIO, DerivationConfig(swing_source=source)
                )
            ]
            assert keys == ["closure", "impact", "flight", "swing"]

    def test_triple_pendulum_step_only_for_triple(self) -> None:
        def titles(source: str) -> list[str]:
            sections = derivation_sections(
                _SCENARIO, DerivationConfig(swing_source=source)
            )
            return [step.title for step in sections[-1].steps]

        assert "Triple-Pendulum Extension" not in titles("double_pendulum")
        assert "Triple-Pendulum Extension" in titles("triple_pendulum")

    def test_gear_effect_step_toggles(self) -> None:
        def impact_titles(gear: bool) -> list[str]:
            sections = derivation_sections(
                _SCENARIO, DerivationConfig(gear_effect=gear)
            )
            return [step.title for step in sections[1].steps]

        gear_title = "Gear Effect — Head Recoil Times CG Depth"
        assert gear_title in impact_titles(True)
        assert gear_title not in impact_titles(False)

    def test_active_flight_model_rewrites_the_coefficient_step(self) -> None:
        def law_title(model: str) -> str:
            sections = derivation_sections(
                _SCENARIO, DerivationConfig(flight_model=model)
            )
            return sections[2].steps[1].title

        assert "Waterloo/Penner" in law_title("waterloo_penner")
        assert "Nathan" in law_title("nathan")
        assert "MacDonald-Hanzely" in law_title("macdonald_hanzely")

    def test_plane_tilts_substitute_live(self) -> None:
        sections = derivation_sections(
            _SCENARIO,
            DerivationConfig(
                swing_source="double_pendulum",
                plane_tilts_deg=(12.0, -37.0, 4.0),
            ),
        )
        gravity_step = sections[-1].steps[3]
        assert "12" in gravity_step.values
        assert "-37" in gravity_step.values


class TestDerivationViewGui:
    @pytest.fixture
    def view(self, qtbot):  # type: ignore[no-untyped-def]
        pytest.importorskip("PyQt6")
        pytest.importorskip("pytestqt")
        from rate_of_closure.ui.pyqt6.derivation_view import DerivationView

        widget = DerivationView()
        qtbot.addWidget(widget)
        widget.set_scenario(_SCENARIO)
        return widget

    def test_sections_toggle_with_config(self, view) -> None:  # type: ignore[no-untyped-def]
        assert view.section_keys() == ("closure", "impact", "flight")
        view.set_config(DerivationConfig(swing_source="triple_pendulum"))
        assert view.section_keys() == ("closure", "impact", "flight", "swing")

    def test_simulation_tab_emits_config(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        pytest.importorskip("PyQt6")
        pytest.importorskip("pytestqt")
        from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab

        tab = SimulationTab()
        qtbot.addWidget(tab)
        with qtbot.waitSignal(tab.configChanged, timeout=2000) as blocker:
            tab._flight_combo.setCurrentText("nathan")
        config = blocker.args[0]
        assert config.flight_model == "nathan"
        assert config.swing_source == "manual"
        tab.stop()

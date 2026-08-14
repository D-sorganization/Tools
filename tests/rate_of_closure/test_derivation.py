"""Tests for the derivation/traceability content.

The derivation tab is a documentation surface: these tests guarantee it
stays complete (every result row has an explanation, every step has all
three text layers), numerically live (the substitutions track the
scenario), and Title Case per the house formatting rules.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from rate_of_closure.derivation import (
    RESULT_EXPLANATIONS,
    derivation_steps,
)
from rate_of_closure.model import ImpactResult, ImpactScenario

pytestmark = pytest.mark.unit

_SCENARIO = ImpactScenario(clubhead_speed_mph=120.0)

#: Minor words allowed lowercase inside Title Case headings.
_MINOR_WORDS = {"and", "or", "the", "a", "an", "of", "to", "in", "vs", "at", "on"}


def _is_title_case(text: str) -> bool:
    words = text.replace("—", " ").replace("-", " ").split()
    for index, word in enumerate(words):
        stripped = word.strip("()&,")
        if not stripped or not stripped[0].isalpha():
            continue
        if stripped.lower() in _MINOR_WORDS and index != 0:
            continue
        if not stripped[0].isupper():
            return False
    return True


class TestResultExplanations:
    def test_every_scalar_result_field_has_an_explanation(self) -> None:
        scalar_fields = [
            f.name
            for f in fields(ImpactResult)
            if f.type == "float"
            and f.name not in ("reference_speed_mph", "point_speed_mph")
        ]
        for name in scalar_fields:
            assert name in RESULT_EXPLANATIONS, name
            assert len(RESULT_EXPLANATIONS[name]) > 80, name

    def test_explanations_cite_the_key_sources(self) -> None:
        joined = " ".join(RESULT_EXPLANATIONS.values())
        assert "Cheetham" in joined
        assert "launch-monitor" in joined
        # Brand names stay out of program strings; openly available data
        # is cited neutrally.
        assert "TrackMan" not in joined
        assert "R_ISA" in joined


class TestDerivationSteps:
    def test_full_chain_is_present_and_ordered(self) -> None:
        titles = [step.title for step in derivation_steps(_SCENARIO)]
        assert titles == [
            "Frame and Sign Conventions",
            "Shaft Axis and Swing-Plane Normal",
            "Angular Velocity Assembly",
            "Lever Arm to the Impact Point",
            "Rigid-Body Point Velocity",
            "Path and Attack-Angle Deviation",
            "Closure Rate — the CCV Identity",
            "Speed-Invariant Closure and the Path Gap",
            "Face Rotation During Contact",
        ]

    def test_titles_are_title_case(self) -> None:
        for step in derivation_steps(_SCENARIO):
            assert _is_title_case(step.title), step.title

    def test_every_step_has_formula_values_and_narrative(self) -> None:
        for step in derivation_steps(_SCENARIO):
            assert step.latex.startswith("$") and step.latex.endswith("$")
            assert step.values.startswith("$") and step.values.endswith("$")
            assert len(step.narrative) > 60

    def test_substitutions_are_live(self) -> None:
        """Changing the scenario must change the numeric lines."""
        base = derivation_steps(_SCENARIO)
        changed = derivation_steps(
            ImpactScenario(clubhead_speed_mph=95.0, omega_shaft_dps=2000.0)
        )
        differing = [b.values != c.values for b, c in zip(base, changed, strict=True)]
        assert sum(differing) >= 5

    def test_traceability_pins_the_headline_numbers(self) -> None:
        """The path/CCV/deg-ft numbers in the steps match the results."""
        steps = {step.title: step for step in derivation_steps(_SCENARIO)}
        assert "-1.56" in steps["Path and Attack-Angle Deviation"].values
        assert "2099" in steps["Closure Rate — the CCV Identity"].values
        assert "11.93" in steps["Speed-Invariant Closure and the Path Gap"].values

    def test_mathtext_renders_without_error(self) -> None:
        """Every formula must be valid matplotlib mathtext."""
        from matplotlib.mathtext import MathTextParser

        parser = MathTextParser("agg")
        for step in derivation_steps(_SCENARIO):
            for text in (step.latex, step.values):
                parser.parse(text, dpi=72)

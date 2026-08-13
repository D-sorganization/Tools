"""Keeps the committed web parity fixture truthful (#4120 V3).

The fixture ``rate_of_closure/web/src/model/__fixtures__/
variation_parity.json`` pins the Python engine's dispersion for a
canonical launch-mode plan; ``variation.test.ts`` then checks the web
engine against the same numbers within a loose statistical band (exact
RNG parity is deliberately not attempted — see variation.ts). This test
regenerates the Python side and asserts it still matches the fixture
tightly, so the pin cannot silently drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.python.swing_sim.variation import (
    VariationPlan,
    run_variation,
    summary_stats,
)

pytestmark = pytest.mark.physics

_FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "variation_parity.json"
)


def test_python_engine_matches_the_committed_fixture() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    plan = VariationPlan.from_json_dict(fixture["plan"])
    dataset = run_variation(plan, n_workers=4)
    assert dataset.n_success == plan.n_runs
    stats = {s.name: s for s in summary_stats(dataset)}
    for name, expected in fixture["python_stats"].items():
        assert stats[name].mean == pytest.approx(expected["mean"], rel=1e-9), name
        assert stats[name].std == pytest.approx(expected["std"], rel=1e-9), name


def test_fixture_declares_the_web_band() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    band = fixture["web_band"]
    assert set(band["mean_abs_tolerance"]) == set(fixture["python_stats"])
    assert 0.0 < band["std_rel_tolerance"] < 1.0

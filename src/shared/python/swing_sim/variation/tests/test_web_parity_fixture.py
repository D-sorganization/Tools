"""Keeps the committed web parity and sanity fixture truthful (#4120 V3, #4456).

The fixture ``rate_of_closure/web/src/model/__fixtures__/variation_parity.json``
serves two distinct cross-runtime contracts:

1. ``explicit_design``: A deterministic cross-runtime parity gate. Identical
   explicit input vectors are evaluated in Python and TypeScript runtimes,
   matching to millimeter precision without PRNG sampling variance.
2. ``python_stats`` and ``web_band``: A statistical sanity / smoke band over
   unconstrained Monte Carlo sampling with independent PRNGs (NumPy PCG64 in
   Python vs Mulberry32 in TypeScript), verifying plausible distributional
   consistency without claiming exact mathematical parity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shared.python.swing_sim.solver.objective import EvaluationConfig
from shared.python.swing_sim.variation import (
    VariationPlan,
    evaluate_run,
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


def test_python_engine_matches_explicit_design_fixture() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    explicit = fixture["explicit_design"]
    cfg = EvaluationConfig(flight_model=explicit["flight_model"])
    keys = explicit["input_keys"]

    carries: list[float] = []
    laterals: list[float] = []
    apexes: list[float] = []

    for row in explicit["rows"]:
        variables = dict(zip(keys, row["inputs"], strict=True))
        out = evaluate_run(variables, explicit["mode"], cfg)
        for metric, expected_val in row["expected"].items():
            assert out[metric] == pytest.approx(expected_val, abs=1e-9), metric

        carries.append(out["carry_m"])
        laterals.append(out["lateral_m"])
        apexes.append(out["apex_m"])

    # Verify summary statistics over the explicit matrix
    stats_map = {
        "carry_m": carries,
        "lateral_m": laterals,
        "apex_m": apexes,
    }
    for metric, values in stats_map.items():
        expected_stat = explicit["python_stats"][metric]
        calc_mean = float(np.mean(values))
        calc_std = float(np.std(values, ddof=1))
        assert calc_mean == pytest.approx(expected_stat["mean"], abs=1e-9), metric
        assert calc_std == pytest.approx(expected_stat["std"], abs=1e-9), metric


def test_fixture_declares_explicit_design_contract() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    explicit = fixture["explicit_design"]
    assert explicit["mode"] == "launch"
    assert len(explicit["input_keys"]) >= 4
    assert len(explicit["rows"]) >= 10

    tolerances = explicit["tolerances"]
    row_tol = tolerances["row_abs_tolerance"]
    assert row_tol["carry_m"] <= 0.005
    assert row_tol["lateral_m"] <= 0.005
    assert row_tol["apex_m"] <= 0.005
    assert row_tol["landing_angle_deg"] <= 0.01
    assert row_tol["flight_time_s"] <= 0.001

    stats_mean_tol = tolerances["stats_mean_abs_tolerance"]
    assert stats_mean_tol["carry_m"] <= 0.001
    assert stats_mean_tol["lateral_m"] <= 0.001
    assert stats_mean_tol["apex_m"] <= 0.001
    assert tolerances["stats_std_rel_tolerance"] <= 0.001

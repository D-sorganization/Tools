# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for comparison module -- trial storage and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.comparison import ComparisonStore, comparison_metrics

from .conftest import make_test_result


class TestComparisonStore:
    def test_add_and_get_trials(self):
        store = ComparisonStore()
        r1 = make_test_result(seed=1)
        r2 = make_test_result(seed=2)
        store.add_trial("Trial A", r1, {"body_mass": 75.0}, 60.0)
        store.add_trial("Trial B", r2, {"body_mass": 80.0}, 80.0)

        trials = store.get_trials()
        assert len(trials) == 2
        assert trials[0]["name"] == "Trial A"
        assert trials[1]["name"] == "Trial B"
        assert trials[0]["bar_mass"] == 60.0

    def test_clear(self):
        store = ComparisonStore()
        store.add_trial("T", make_test_result(), {}, 60.0)
        assert len(store.get_trials()) == 1
        store.clear()
        assert len(store.get_trials()) == 0

    def test_empty_store(self):
        store = ComparisonStore()
        assert store.get_trials() == []

    def test_stored_trials_isolated_from_caller_mutation(self):
        store = ComparisonStore()
        r = make_test_result(seed=3)
        original_cost = r.cost
        body_params = {"body_mass": 75.0, "nested": {"k": 1}}
        store.add_trial("Trial A", r, body_params, 60.0)

        # Mutate caller's input dict and result scalar after add_trial.
        body_params["body_mass"] = 999.0
        body_params["nested"]["k"] = 999
        r.cost = -1.0

        trials = store.get_trials()
        assert trials[0]["body_params"] == {"body_mass": 75.0, "nested": {"k": 1}}
        assert trials[0]["result"].cost == original_cost

        # Mutate retrieved trial dict, nested body_params, and result scalar.
        trials[0]["name"] = "Mutated"
        trials[0]["bar_mass"] = 999.0
        trials[0]["body_params"]["body_mass"] = 123.0
        trials[0]["body_params"]["nested"]["k"] = 456
        trials[0]["result"].cost = -999.0

        fresh = store.get_trials()
        assert fresh[0]["name"] == "Trial A"
        assert fresh[0]["bar_mass"] == 60.0
        assert fresh[0]["body_params"] == {"body_mass": 75.0, "nested": {"k": 1}}
        assert fresh[0]["result"].cost == original_cost
        # Large arrays are shared rather than deep-copied.
        assert np.shares_memory(fresh[0]["result"].power, r.power)


class TestComparisonMetrics:
    def test_metrics_keys(self):
        r1 = make_test_result(seed=1)
        r2 = make_test_result(seed=2)
        trials = [
            {"name": "A", "result": r1, "body_params": {}, "bar_mass": 60.0},
            {"name": "B", "result": r2, "body_params": {}, "bar_mass": 80.0},
        ]
        metrics = comparison_metrics(trials)
        assert len(metrics) == 2
        for m in metrics:
            assert "name" in m
            assert "peak_torques" in m
            assert "total_work" in m
            assert "com_sway_cm" in m

    def test_metrics_values_are_reasonable(self):
        r = make_test_result(seed=10)
        trials = [{"name": "X", "result": r, "body_params": {}, "bar_mass": 60.0}]
        metrics = comparison_metrics(trials)
        m = metrics[0]
        assert len(m["peak_torques"]) == 3
        assert all(v >= 0 for v in m["peak_torques"])
        assert m["total_work"] >= 0
        assert m["com_sway_cm"] >= 0

    def test_empty_trials(self):
        metrics = comparison_metrics([])
        assert metrics == []

    def test_work_metrics_distinguish_net_power_vs_summed_abs_joint_work(self):
        r = make_test_result(seed=11)
        r.t = np.array([0.0, 1.0])
        r.torques = np.zeros((2, 3))
        r.power = np.array([[100.0, -100.0, 0.0], [100.0, -100.0, 0.0]])
        trials = [{"name": "Cancel", "result": r, "body_params": {}, "bar_mass": 60.0}]

        m = comparison_metrics(trials)[0]
        assert m["total_work"] == pytest.approx(0.0)
        assert m["abs_joint_work_j"] == pytest.approx(200.0)

    def test_work_metrics_time_sign_change(self):
        r = make_test_result(seed=12)
        r.t = np.array([0.0, 0.5, 1.0])
        r.torques = np.zeros((3, 3))
        # Net power reverses sign across time (+100 W -> 0 -> -100 W);
        # integral of |net power| over 1 s is 50 J (not 0 J).
        r.power = np.array(
            [
                [100.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-100.0, 0.0, 0.0],
            ]
        )
        trials = [{"name": "SignFlip", "result": r, "body_params": {}, "bar_mass": 60.0}]

        m = comparison_metrics(trials)[0]
        assert m["total_work"] == pytest.approx(50.0)
        assert m["abs_joint_work_j"] == pytest.approx(50.0)

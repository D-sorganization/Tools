"""
tests/test_sweep.py
===================
Unit tests for the ParameterSweep engine.

Because the sweep engine calls a model runner, we inject a *mock runner*
that returns predetermined KPIs without touching DWSIM at all.

This tests the sweep *mechanics*:
  - Does it produce the right number of rows?
  - Does it correctly modify the config dict before each run?
  - Does it handle errors in individual runs gracefully?
  - Does it record the requested KPIs?
"""

from types import SimpleNamespace

import pytest

from dwsim_model.analysis.sweep import (
    ParameterSweep,
    _default_model_runner,
    _get_nested,
    _set_nested,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNestedHelpers:

    def test_set_nested_simple(self):
        d = {"feeds": {"biomass": {"flow": 1.0}}}
        _set_nested(d, "feeds.biomass.flow", 2.5)
        assert d["feeds"]["biomass"]["flow"] == 2.5

    def test_set_nested_creates_new_leaf(self):
        d = {"feeds": {"biomass": {}}}
        _set_nested(d, "feeds.biomass.new_key", 99.0)
        assert d["feeds"]["biomass"]["new_key"] == 99.0

    def test_set_nested_missing_intermediate_raises(self):
        d = {"feeds": {}}
        with pytest.raises(KeyError):
            _set_nested(d, "feeds.missing.flow", 1.0)

    def test_get_nested_existing_path(self):
        d = {"a": {"b": {"c": 42}}}
        assert _get_nested(d, "a.b.c") == 42

    def test_get_nested_missing_path_returns_default(self):
        d = {"a": {}}
        assert _get_nested(d, "a.b.c", default=-1) == -1

    def test_get_nested_wrong_type_returns_default(self):
        d = {"a": 5}  # 5 is not a dict
        assert _get_nested(d, "a.b", default=None) is None


# ─────────────────────────────────────────────────────────────────────────────
# Mock runner for tests
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_runner(kpi_fn=None):
    """
    Returns a mock runner that computes fake KPIs based on the config dict.

    If kpi_fn is provided, it's called with the config and should return a dict.
    Otherwise a simple default is used.
    """
    calls = []

    def runner(config: dict) -> dict:
        calls.append(config)
        if kpi_fn:
            return kpi_fn(config)
        # Default: CGE proportional to biomass flow (arbitrary linear mock)
        flow = _get_nested(config, "feeds.biomass.flow", default=4.0)
        return {
            "cold_gas_efficiency": min(0.50 + float(flow) * 0.05, 0.95),
            "h2_co_ratio": 1.2 + float(flow) * 0.1,
            "tar_loading_mg_Nm3": max(50.0 - float(flow) * 5, 5.0),
        }

    runner.calls = calls
    return runner


# ─────────────────────────────────────────────────────────────────────────────
# ParameterSweep tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParameterSweep1D:

    def setup_method(self):
        """Fresh sweep engine with mock runner before each test."""
        self.mock_runner = _make_mock_runner()
        self.sweep = ParameterSweep(model_runner=self.mock_runner)
        self.sweep.set_base_config(
            {"feeds": {"biomass": {"flow": 4.0, "moisture": 0.15}}}
        )

    def test_correct_number_of_rows(self):
        df = self.sweep.sweep_1d("feeds.biomass.flow", [2.0, 3.0, 4.0, 5.0])
        assert len(df) == 4

    def test_runner_called_once_per_step(self):
        self.sweep.sweep_1d("feeds.biomass.flow", [1.0, 2.0, 3.0])
        assert len(self.mock_runner.calls) == 3

    def test_config_is_modified_correctly(self):
        """Each runner call should receive the patched value."""
        values = [2.0, 4.0, 6.0]
        self.sweep.sweep_1d("feeds.biomass.flow", values)
        for i, call_config in enumerate(self.mock_runner.calls):
            actual = _get_nested(call_config, "feeds.biomass.flow")
            assert (
                actual == values[i]
            ), f"Run {i}: expected flow={values[i]}, got {actual}"

    def test_base_config_is_not_mutated(self):
        """Each run should get its own copy; the base config stays unchanged."""
        original_flow = 4.0
        self.sweep.set_base_config({"feeds": {"biomass": {"flow": original_flow}}})
        self.sweep.sweep_1d("feeds.biomass.flow", [1.0, 2.0, 3.0])
        assert self.sweep._base_config["feeds"]["biomass"]["flow"] == original_flow

    def test_kpi_filter_applied(self):
        """When kpis= is specified, only those columns appear."""
        df = self.sweep.sweep_1d(
            "feeds.biomass.flow", [2.0, 4.0], kpis=["cold_gas_efficiency"]
        )
        # Check that the restricted KPI appears
        try:
            # pandas DataFrame
            cols = df.columns.tolist()
            assert "cold_gas_efficiency" in cols
            assert "h2_co_ratio" not in cols
        except AttributeError:
            # list of dicts fallback
            assert "cold_gas_efficiency" in df[0]
            assert "h2_co_ratio" not in df[0]

    def test_failed_run_does_not_crash_sweep(self):
        """If the runner raises on one step, the others should still complete."""
        call_count = [0]

        def flaky_runner(config):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated solver failure")
            return {"cold_gas_efficiency": 0.70}

        sweep = ParameterSweep(model_runner=flaky_runner)
        sweep.set_base_config({"feeds": {"biomass": {"flow": 4.0}}})
        df = sweep.sweep_1d("feeds.biomass.flow", [2.0, 3.0, 4.0])
        # Should have 2 successful rows + 1 error row (or 2 successful only)
        assert len(df) >= 2

    def test_custom_label_used_as_column_name(self):
        df = self.sweep.sweep_1d(
            "feeds.biomass.flow", [2.0, 4.0], label="Biomass_Flow_kg_s"
        )
        try:
            assert "Biomass_Flow_kg_s" in df.columns
        except AttributeError:
            assert "Biomass_Flow_kg_s" in df[0]


class TestParameterSweep2D:

    def setup_method(self):
        self.mock_runner = _make_mock_runner()
        self.sweep = ParameterSweep(model_runner=self.mock_runner)
        self.sweep.set_base_config(
            {
                "feeds": {
                    "biomass": {"flow": 4.0},
                    "steam": {"flow": 1.0},
                }
            }
        )

    def test_2d_sweep_total_rows(self):
        """3 × 4 = 12 combinations."""
        df = self.sweep.sweep_2d(
            "feeds.biomass.flow",
            [2.0, 3.0, 4.0],
            "feeds.steam.flow",
            [0.5, 1.0, 1.5, 2.0],
        )
        assert len(df) == 12

    def test_2d_runner_called_correct_count(self):
        self.sweep.sweep_2d(
            "feeds.biomass.flow",
            [2.0, 4.0],
            "feeds.steam.flow",
            [0.5, 1.5],
        )
        assert len(self.mock_runner.calls) == 4

    def test_2d_both_params_modified_independently(self):
        """Each run should have its own combination of A and B values."""
        recorded = []

        def recording_runner(config):
            recorded.append(
                (
                    _get_nested(config, "feeds.biomass.flow"),
                    _get_nested(config, "feeds.steam.flow"),
                )
            )
            return {"cold_gas_efficiency": 0.70}

        sweep = ParameterSweep(model_runner=recording_runner)
        sweep.set_base_config(
            {"feeds": {"biomass": {"flow": 4.0}, "steam": {"flow": 1.0}}}
        )
        sweep.sweep_2d(
            "feeds.biomass.flow",
            [2.0, 4.0],
            "feeds.steam.flow",
            [0.5, 1.5],
        )
        assert (2.0, 0.5) in recorded
        assert (2.0, 1.5) in recorded
        assert (4.0, 0.5) in recorded
        assert (4.0, 1.5) in recorded


class TestSensitivityOAT:

    def test_oat_produces_rows_for_each_param(self):
        pytest.importorskip("numpy")
        mock_runner = _make_mock_runner()
        sweep = ParameterSweep(model_runner=mock_runner)
        sweep.set_base_config(
            {"feeds": {"biomass": {"flow": 4.0}, "steam": {"flow": 1.0}}}
        )

        result = sweep.sensitivity_oat(
            params={
                "feeds.biomass.flow": (2.0, 6.0),
                "feeds.steam.flow": (0.5, 2.0),
            },
            n_steps=3,
        )

        # 2 parameters × 3 steps = 6 total rows
        try:
            assert len(result) == 6
        except TypeError:
            # May be a list — count items
            assert sum(1 for _ in result) == 6


def test_default_model_runner_passes_compound_set_to_extractor(monkeypatch):
    observed: dict[str, object] = {}

    class FakeFlowsheet:
        def __init__(self, runtime_config=None):
            self.compound_set = ["Hydrogen", "Carbon monoxide"]
            self.builder = object()
            self._injected_config = runtime_config

        def build_flowsheet(self):
            observed["config"] = self._injected_config

        def run(self):
            observed["ran"] = True

    class FakeExtractor:
        def __init__(self, compound_names=None, key_streams=None):
            observed["compound_names"] = compound_names

        def extract(self, builder):
            observed["builder"] = builder
            return "results"

    class FakeMetrics:
        def calculate(self, results):
            observed["results"] = results
            return SimpleNamespace(to_dict=lambda: {"cold_gas_efficiency": 0.7})

    monkeypatch.setattr(
        "dwsim_model.gasification.GasificationFlowsheet",
        FakeFlowsheet,
    )
    monkeypatch.setattr(
        "dwsim_model.results.extractor.ResultsExtractor",
        FakeExtractor,
    )
    monkeypatch.setattr(
        "dwsim_model.results.metrics.MetricsCalculator",
        FakeMetrics,
    )

    config = {"feeds": {"Gasifier_Biomass_Feed": {"mass_flow_kg_s": 5.0}}}

    metrics = _default_model_runner(config)

    assert metrics["cold_gas_efficiency"] == pytest.approx(0.7)
    assert observed["config"] == config
    assert observed["compound_names"] == ["Hydrogen", "Carbon monoxide"]
    assert observed["results"] == "results"

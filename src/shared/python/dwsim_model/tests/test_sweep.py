from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dwsim_model.analysis.sweep import (
    ParameterSweep,
    _default_model_runner,
    _get_nested,
    _set_nested,
)


def test_set_nested() -> None:
    cfg = {"a": {"b": {"c": 1}}}
    _set_nested(cfg, "a.b.c", 2)
    assert cfg["a"]["b"]["c"] == 2

    with pytest.raises(KeyError):
        _set_nested(cfg, "a.b.d.e", 3)


def test_get_nested() -> None:
    cfg = {"a": {"b": {"c": 1}}}
    assert _get_nested(cfg, "a.b.c") == 1
    assert _get_nested(cfg, "a.b.d", default="missing") == "missing"


def mock_runner(config: dict) -> dict:
    val = _get_nested(config, "param", 0)
    return {"kpi": val * 2, "converged": True, "cold_gas_efficiency": 0.5}


def test_parameter_sweep_1d() -> None:
    ps = ParameterSweep(model_runner=mock_runner)
    ps.set_base_config({"param": 10})

    df = ps.sweep_1d("param", [1, 2, 3], kpis=["kpi"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df["kpi"].tolist() == [2.0, 4.0, 6.0]
    assert df["param"].tolist() == [1.0, 2.0, 3.0]


def test_parameter_sweep_2d() -> None:
    def mock_runner_2d(config: dict) -> dict:
        a = _get_nested(config, "a", 0)
        b = _get_nested(config, "b", 0)
        return {"kpi": a + b, "cold_gas_efficiency": 0.5}

    ps = ParameterSweep(model_runner=mock_runner_2d)
    ps.set_base_config({"a": 0, "b": 0})

    df = ps.sweep_2d("a", [1, 2], "b", [3, 4])

    assert len(df) == 4
    # 1+3, 1+4, 2+3, 2+4 = 4, 5, 5, 6
    assert df["kpi"].tolist() == [4.0, 5.0, 5.0, 6.0]


def test_sensitivity_oat() -> None:
    ps = ParameterSweep(model_runner=mock_runner)
    ps.set_base_config({"param": 10, "param2": 20})

    df = ps.sensitivity_oat({"param": (1, 3)}, kpis=["kpi"], n_steps=3)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "swept_param" in df.columns
    assert df["swept_param"].iloc[0] == "param"
    assert df["param"].tolist() == [1.0, 2.0, 3.0]


@patch("dwsim_model.gasification.GasificationFlowsheet")
@patch("dwsim_model.results.extractor.ResultsExtractor")
@patch("dwsim_model.results.metrics.MetricsCalculator")
def test_default_model_runner(mock_metrics, mock_extractor, mock_flowsheet) -> None:
    mock_flowsheet_inst = MagicMock()
    mock_flowsheet_inst.compound_set = ["C"]
    mock_flowsheet.return_value = mock_flowsheet_inst

    mock_extractor_inst = MagicMock()
    mock_extractor.return_value = mock_extractor_inst

    mock_metrics_inst = MagicMock()
    mock_metrics_inst.calculate.return_value = MagicMock(to_dict=lambda: {"kpi": 1})
    mock_metrics.return_value = mock_metrics_inst

    res = _default_model_runner({})
    assert res == {"kpi": 1}
    mock_flowsheet_inst.build_flowsheet.assert_called_once()
    mock_flowsheet_inst.run.assert_called_once()
    mock_extractor_inst.extract.assert_called_once()
    mock_metrics_inst.calculate.assert_called_once()
